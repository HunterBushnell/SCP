#!/usr/bin/env python3
"""SCP environment + workspace readiness checker.

Usage examples:
  python scripts/check_setup.py --steps 0 1 2 3 4 5 --cell PV --tune seg_tuned
  python scripts/check_setup.py --steps 5 --cell SST --tune seg_tuned --compile-modfiles
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


ACT_REPO_URL = "https://github.com/V-Marco/ACT.git"
BMTOOL_REPO_URL = "https://github.com/cyneuro/bmtool.git"


def _is_repo_root(path: Path) -> bool:
    return (path / "run_pipeline.py").is_file() and (path / "modules").is_dir()


def _find_repo_root(start: Path) -> Path:
    env_root = os.environ.get("SCP_ROOT")
    if env_root:
        cand = Path(env_root).expanduser().resolve()
        if _is_repo_root(cand):
            return cand
        raise FileNotFoundError(
            f"SCP_ROOT is set but is not an SCP repo root: {cand} "
            "(expected run_pipeline.py and modules/)"
        )

    start = start.resolve()
    for cand in (start, *start.parents):
        if _is_repo_root(cand):
            return cand

    for base in (start, start.parent):
        try:
            for child in base.iterdir():
                if child.is_dir() and _is_repo_root(child):
                    return child.resolve()
        except Exception:
            pass

    raise FileNotFoundError(
        f"Could not locate SCP repo root from {start}. "
        "Run from inside the repo or set SCP_ROOT=/path/to/SCP."
    )


def _resolve_external_repo(
    repo_name: str,
    marker_rel: Path,
    env_vars: tuple[str, ...],
    repo_root: Path,
) -> tuple[Optional[Path], list[Path]]:
    candidates: list[Path] = []

    for var in env_vars:
        raw = os.environ.get(var)
        if raw:
            candidates.append(Path(raw).expanduser())

    candidates.extend(
        [
            repo_root.parent / "mods" / repo_name,
            repo_root / "mods" / repo_name,
            Path.home() / "mods" / repo_name,
        ]
    )

    seen: set[Path] = set()
    unique: list[Path] = []
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except Exception:
            resolved = cand
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
        if (resolved / marker_rel).is_file():
            return resolved, unique

    return None, unique


def _pkg_version(dist_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


class Reporter:
    def __init__(self) -> None:
        self.errors = 0
        self.warnings = 0

    def ok(self, msg: str) -> None:
        print(f"[OK]   {msg}")

    def warn(self, msg: str) -> None:
        self.warnings += 1
        print(f"[WARN] {msg}")

    def fail(self, msg: str) -> None:
        self.errors += 1
        print(f"[FAIL] {msg}")


def _check_python_packages(r: Reporter) -> None:
    recommended_versions = {
        "numpy": "1.23.5",
        "scipy": "1.10.1",
        "pandas": "1.5.3",
        "matplotlib": "3.9.2",
        "h5py": "3.12.1",
        "allensdk": "2.16.2",
        "neuron": "8.2.4",
        "ipywidgets": "8.1.5",
    }

    required = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("h5py", "h5py"),
        ("allensdk", "allensdk"),
        ("neuron", "neuron"),
    ]
    optional = [("ipywidgets", "ipywidgets")]

    for import_name, dist_name in required:
        try:
            importlib.import_module(import_name)
        except Exception as exc:
            r.fail(f"Python package missing or broken: {import_name} ({exc})")
            continue

        ver = _pkg_version(dist_name)
        if ver:
            r.ok(f"{import_name} {ver}")
            wanted = recommended_versions.get(import_name)
            if wanted and ver != wanted:
                r.warn(
                    f"{import_name} version is {ver}; "
                    f"recommended is {wanted} (see environment.yml)"
                )
        else:
            r.ok(f"{import_name} (version unknown)")

    for import_name, dist_name in optional:
        try:
            importlib.import_module(import_name)
        except Exception:
            r.warn(f"Optional package not available: {import_name} (needed for some interactive widgets)")
            continue

        ver = _pkg_version(dist_name)
        if ver:
            r.ok(f"{import_name} {ver}")
            wanted = recommended_versions.get(import_name)
            if wanted and ver != wanted:
                r.warn(
                    f"{import_name} version is {ver}; "
                    f"recommended is {wanted} (see environment.yml)"
                )
        else:
            r.ok(f"{import_name} (version unknown)")

    try:
        from neuron import h

        nrn = str(getattr(h, "nrnversion", lambda: "unknown")())
        r.ok(f"NEURON runtime: {nrn}")
    except Exception as exc:
        r.fail(f"NEURON import succeeded but runtime probe failed: {exc}")

    nrnivmodl = shutil.which("nrnivmodl")
    if nrnivmodl:
        r.ok(f"nrnivmodl found at {nrnivmodl}")
    else:
        r.fail("nrnivmodl not found on PATH (required to compile modfiles)")


def _check_external_dependencies(r: Reporter, repo_root: Path, steps: set[str]) -> None:
    if steps.intersection({"1", "2", "3"}):
        act_marker = Path("act") / "segregation.py"
        act_path, act_candidates = _resolve_external_repo(
            repo_name="ACT",
            marker_rel=act_marker,
            env_vars=("SCP_ACT_PATH", "ACT_PATH", "ACT_ROOT"),
            repo_root=repo_root,
        )
        if act_path is None:
            r.fail(
                "ACT repo not found for steps 1-3. "
                "Set SCP_ACT_PATH or clone ACT to ../mods/ACT. "
                f"Example: git clone {ACT_REPO_URL} {repo_root.parent / 'mods' / 'ACT'}"
            )
            for cand in act_candidates:
                print(f"       - checked: {cand}")
        else:
            r.ok(f"ACT repo found: {act_path}")

        # Step 2/3 need act/passive.py specifically.
        if steps.intersection({"2", "3"}) and act_path is not None:
            passive_path = act_path / "act" / "passive.py"
            if passive_path.is_file():
                r.ok(f"ACT passive module found: {passive_path}")
            else:
                r.fail(f"ACT repo found but missing {passive_path}")

    if "4" in steps:
        bmtool_path, bmtool_candidates = _resolve_external_repo(
            repo_name="bmtool",
            marker_rel=Path("bmtool") / "synapses.py",
            env_vars=("SCP_BMTOOL_PATH", "BMTOOL_PATH", "BMTOOL_ROOT"),
            repo_root=repo_root,
        )
        if bmtool_path is None:
            r.fail(
                "bmtool repo not found for step 4. "
                "Set SCP_BMTOOL_PATH or clone bmtool to ../mods/bmtool. "
                f"Example: git clone {BMTOOL_REPO_URL} {repo_root.parent / 'mods' / 'bmtool'}"
            )
            for cand in bmtool_candidates:
                print(f"       - checked: {cand}")
        else:
            r.ok(f"bmtool repo found: {bmtool_path}")


def _config_exists(tune_dir: Path, name: str) -> bool:
    return (tune_dir / "cell_configs" / name).is_file() or (tune_dir / name).is_file()


def _compile_modfiles_if_requested(tune_dir: Path, r: Reporter) -> bool:
    mod_dir = tune_dir / "modfiles"
    if not mod_dir.is_dir():
        r.fail(f"Missing modfiles directory: {mod_dir}")
        return False

    nrnivmodl = shutil.which("nrnivmodl")
    if not nrnivmodl:
        r.fail("nrnivmodl not found on PATH; cannot compile modfiles")
        return False

    r.ok(f"Compiling modfiles with {nrnivmodl} in {mod_dir}")
    proc = subprocess.run(
        [nrnivmodl],
        cwd=str(mod_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-20:])
        r.fail("nrnivmodl failed. Last output lines:\n" + tail)
        return False

    r.ok("nrnivmodl completed successfully")
    return True


def _check_tune_bundle(
    r: Reporter,
    repo_root: Path,
    cell: str,
    tune: str,
    compile_modfiles: bool,
) -> None:
    tune_dir = repo_root / "cells" / cell / "tunes" / tune
    if not tune_dir.is_dir():
        r.fail(f"Tune directory not found: {tune_dir}")
        return
    r.ok(f"Tune directory: {tune_dir}")

    for cfg in ("cell_config.json", "sim_config.json", "syn_config.json"):
        if _config_exists(tune_dir, cfg):
            r.ok(f"Config present: {cfg}")
        else:
            r.fail(f"Missing config: {cfg} (expected in {tune_dir}/cell_configs or tune root)")

    mech_candidates = [
        tune_dir / "modfiles" / "x86_64" / ".libs" / "libnrnmech.so",
        tune_dir / "modfiles" / "x86_64" / "libnrnmech.so",
    ]
    if any(p.is_file() for p in mech_candidates):
        for p in mech_candidates:
            if p.is_file():
                r.ok(f"Compiled mechanisms found: {p}")
                break
    else:
        r.warn("Compiled mechanisms not found")
        if compile_modfiles:
            compiled = _compile_modfiles_if_requested(tune_dir, r)
            if compiled and any(p.is_file() for p in mech_candidates):
                for p in mech_candidates:
                    if p.is_file():
                        r.ok(f"Compiled mechanisms created: {p}")
                        break
            elif compiled:
                r.fail("nrnivmodl finished but no libnrnmech.so found in expected locations")
        else:
            r.warn("Use --compile-modfiles to build now")


def _parse_steps(raw_steps: Iterable[str]) -> set[str]:
    valid = {"0", "1", "2", "3", "4", "5", "6"}
    steps: set[str] = set()
    for token in raw_steps:
        tok = token.strip()
        if tok in valid:
            steps.add(tok)
            continue
        if "," in tok:
            for part in tok.split(","):
                p = part.strip()
                if p in valid:
                    steps.add(p)
                else:
                    raise ValueError(f"Invalid step: {p}")
            continue
        raise ValueError(f"Invalid step: {tok}")
    return steps


def main() -> int:
    ap = argparse.ArgumentParser(description="Check SCP environment + local workspace readiness")
    ap.add_argument("--repo-root", default=None, help="Path to SCP repo root (auto-detected by default)")
    ap.add_argument("--steps", nargs="*", default=["0", "1", "2", "3", "4", "5"], help="Pipeline steps to validate, e.g. --steps 0 1 2 3 4 5")
    ap.add_argument("--cell", default="PV", help="Cell name for tune checks (default: PV)")
    ap.add_argument("--tune", default="seg_tuned", help="Tune directory name (default: seg_tuned)")
    ap.add_argument("--skip-tune-check", action="store_true", help="Skip tune/config/modfile checks")
    ap.add_argument("--compile-modfiles", action="store_true", help="Compile modfiles if compiled mechanisms are missing")
    args = ap.parse_args()

    try:
        steps = _parse_steps(args.steps)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    try:
        if args.repo_root:
            repo_root = Path(args.repo_root).expanduser().resolve()
            if not _is_repo_root(repo_root):
                print(
                    f"[FAIL] --repo-root is not an SCP repo root: {repo_root} "
                    "(expected run_pipeline.py and modules/)"
                )
                return 2
        else:
            repo_root = _find_repo_root(Path.cwd())
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 2

    print(f"Repo root: {repo_root}")
    print(f"Steps checked: {', '.join(sorted(steps))}")

    r = Reporter()

    pyv = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info.major == 3 and sys.version_info.minor == 11:
        r.ok(f"Python version {pyv} (recommended)")
    elif sys.version_info.major == 3 and sys.version_info.minor >= 9:
        r.warn(f"Python version {pyv}; recommended is 3.11 for this repo")
    else:
        r.fail(f"Python version {pyv}; expected Python >=3.9 (recommended 3.11)")

    _check_python_packages(r)
    _check_external_dependencies(r, repo_root=repo_root, steps=steps)

    if not args.skip_tune_check and steps.intersection({"0", "2", "3", "4", "5", "6"}):
        _check_tune_bundle(
            r,
            repo_root=repo_root,
            cell=args.cell,
            tune=args.tune,
            compile_modfiles=args.compile_modfiles,
        )

    print("\nSummary:")
    print(f"  errors:   {r.errors}")
    print(f"  warnings: {r.warnings}")

    if r.errors:
        print("\nSetup is NOT ready. Fix failed checks, then rerun this script.")
        return 1

    print("\nSetup is ready for the requested steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
