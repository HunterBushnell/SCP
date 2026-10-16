#!/usr/bin/env python3
"""SCP environment + workspace readiness checker.

Usage examples:
  python scripts/check_setup.py --steps 1 2 3 4 5 --cell PV --tune tuned
  python scripts/check_setup.py --steps 5 --cell SST --tune tuned --compile-modfiles
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

from scp_pipeline.runtime import code_root
REPO_ROOT_HINT = code_root()

from scp_pipeline.setup.mechanisms import (
    find_compiled_mechanism_dll,
    find_nrnivmodl,
    resolve_modfiles_dir,
)
from scp_pipeline import loaders as cell_loaders


ACT_REPO_URL = "https://github.com/V-Marco/ACT.git"


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
        ("neuron", "neuron"),
    ]
    optional = [("allensdk", "allensdk"), ("ipywidgets", "ipywidgets")]

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
            r.warn(
                f"Optional package not available: {import_name} "
                "(needed for some interactive widgets)"
            )
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

    nrnivmodl = find_nrnivmodl()
    if nrnivmodl:
        r.ok(f"nrnivmodl found at {nrnivmodl}")
    else:
        r.warn("nrnivmodl not found; only tunes with custom .mod sources need it")


def _check_external_dependencies(
    r: Reporter,
    repo_root: Path,
    steps: set[str],
    *,
    check_act: bool = False,
    check_bmtool: bool = False,
) -> None:
    if check_act and steps.intersection({"2", "3"}):
        from scp_pipeline.tuning.act_integration import import_act_active_api, import_act_passive_module
        for step, label, loader in (
            ("2", "passive", import_act_passive_module),
            ("3", "active", import_act_active_api),
        ):
            if step not in steps:
                continue
            try:
                api = loader(repo_root=repo_root, auto_clone=False)
            except Exception as exc:
                r.fail(f"Optional ACT {label} integration unavailable: {exc}")
            else:
                representative = api["ACTModule"] if isinstance(api, dict) else api
                r.ok(f"ACT {label} API available: {inspect.getfile(representative)}")

    if check_bmtool and "4" in steps:
        from scp_pipeline.tuning.bmtool_dependency import import_bmtool_synapse_api
        try:
            tuner, _ = import_bmtool_synapse_api(repo_root=repo_root, auto_clone=False)
        except Exception as exc:
            r.fail(f"Optional BMTool integration unavailable: {exc}")
        else:
            r.ok(f"BMTool synapse API available: {inspect.getfile(tuner)}")


def _config_exists(tune_dir: Path, name: str) -> bool:
    return (tune_dir / "cell_configs" / name).is_file() or (tune_dir / name).is_file()


def _config_path(tune_dir: Path, name: str) -> Optional[Path]:
    for candidate in (tune_dir / "cell_configs" / name, tune_dir / name):
        if candidate.is_file():
            return candidate
    return None


def _read_json_object(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


def _check_input_dependencies(
    r: Reporter,
    tune_dir: Path,
    syn_path: Path,
    sim_config: dict,
) -> None:
    """Check configured input paths without generating inputs or loading a cell."""
    try:
        from scp_pipeline.input_generation.group_config import _expand_group_includes
        from scp_pipeline.input_generation.mode_helpers import _resolve_source_path

        raw_groups = json.loads(syn_path.read_text(encoding="utf-8"))
        groups = _expand_group_includes(raw_groups, syn_path.parent)
    except Exception as exc:
        r.fail(
            f"Cannot inspect Step 5 input dependencies in {syn_path}: {exc}. "
            "Fix syn_config.json and its group_files includes, then rerun setup."
        )
        return

    # Resolve relative sources against the selected workspace, never a stale
    # tune_dir value inherited from a saved config or another current directory.
    source_sim = dict(sim_config, tune_dir=str(tune_dir))

    def check_source(source, mode, label: str) -> None:
        if source is None:
            return
        if not isinstance(source, dict):
            r.fail(f"Invalid Step 5 {label}: source must be a JSON object")
            return
        if mode == "precomputed" and source.get("trains") is not None:
            return  # Inline trains, including [], take precedence at runtime.
        raw_path = source.get("path")
        if raw_path in (None, ""):
            return
        if not isinstance(raw_path, str):
            r.fail(f"Invalid Step 5 {label}: source.path must be a nonempty string")
            return
        if not raw_path.strip():
            return
        try:
            resolved = _resolve_source_path(raw_path, source_sim)
            present = resolved.is_file() or (mode == "precomputed" and resolved.is_dir())
        except Exception as exc:
            r.fail(f"Cannot resolve Step 5 {label} source.path={raw_path!r}: {exc}")
            return
        if present:
            r.ok(f"Input dependency for {label}: {resolved}")
        else:
            r.fail(
                f"Missing Step 5 input for {label}: source.path={raw_path!r} "
                f"resolves to {resolved}. Supply the intended input dataset, "
                "correct its path, or explicitly disable this input group/block."
            )

    for name, group in groups.items():
        label = f"group {name!r}"
        if not isinstance(group, dict):
            r.fail(f"Invalid Step 5 {label}: config must be a JSON object")
            continue
        if not group.get("state", True):
            continue
        syns = group.get("syns", {}) or {}
        if not isinstance(syns, dict):
            r.fail(f"Invalid Step 5 {label}: syns must be a JSON object")
            continue
        if syns.get("N_syn") in (0, "0"):
            continue
        blocks = group.get("input_blocks", []) or []
        if not isinstance(blocks, list):
            r.fail(f"Invalid Step 5 {label}: input_blocks must be a JSON list")
            continue
        if blocks:
            for index, block in enumerate(blocks):
                if not isinstance(block, dict):
                    r.fail(f"Invalid Step 5 {label} input_blocks[{index}]: expected a JSON object")
                    continue
                if block.get("state", True) is False:
                    continue
                block_name = block.get("name") or f"block_{index + 1}"
                check_source(
                    block.get("source", {}),
                    block.get("mode"),
                    f"{label} block {block_name!r} (input_blocks[{index}])",
                )
        else:
            check_source(group.get("source", {}), group.get("mode"), label)


def _compile_modfiles_if_requested(
    tune_dir: Path,
    r: Reporter,
    cell_config: Optional[dict] = None,
) -> bool:
    mod_dir = resolve_modfiles_dir(tune_dir, cell_config)
    if mod_dir is None:
        r.ok("No configured modfiles directory; model uses available NEURON mechanisms")
        return True
    if not mod_dir.is_dir():
        r.ok(
            "No configured MOD source directory; model may use built-in "
            "NEURON mechanisms only"
        )
        return True
    if not any(mod_dir.glob("*.mod")):
        r.ok(f"No .mod sources in {mod_dir}; compilation is not required")
        return True

    nrnivmodl = find_nrnivmodl()
    if not nrnivmodl:
        r.fail(
            "nrnivmodl not found on PATH or next to the active Python "
            "executable; cannot compile modfiles"
        )
        return False

    from scp_pipeline.setup.mechanisms import compile_modfiles
    try:
        result = compile_modfiles(tune_dir, cell_config=cell_config, load_dll=False, allow_missing=True)
        r.ok(f"Mechanisms ready: {result.get('compiled_dir')}")
    except Exception as exc:
        r.fail(f"Mechanism build failed: {exc}")
        return False

    r.ok("nrnivmodl completed successfully")
    return True


def _check_tune_bundle(
    r: Reporter,
    repo_root: Path,
    cell: str,
    tune: str,
    compile_modfiles: bool,
    steps: set[str],
    tune_dir_override: Optional[Path] = None,
    project_name: Optional[str] = None,
) -> None:
    from scp_pipeline.workspaces import resolve_workspace
    try:
        workspace = resolve_workspace(repo_root=repo_root, project_name=project_name,
            cell_name=cell, tune_name=tune, tune_dir_override=tune_dir_override,
            relative_to=Path.cwd())
    except (ValueError, TypeError) as exc:
        r.fail(f"Invalid workspace selection: {exc}")
        return
    tune_dir = workspace.tune_dir
    if not tune_dir.is_dir():
        r.fail(f"Tune directory not found: {tune_dir}")
        return
    r.ok(f"Tune directory: {tune_dir}")

    for cfg in ("cell_config.json", "sim_config.json", "geometry.json"):
        if _config_exists(tune_dir, cfg):
            r.ok(f"Config present: {cfg}")
        else:
            r.fail(f"Missing config: {cfg} (expected in {tune_dir}/cell_configs or tune root)")

    target_path = _config_path(tune_dir, "target_config.json")
    if target_path is None:
        r.ok("No target_config.json; Steps 2-3 can run in characterization mode")
    else:
        try:
            target_cfg = _read_json_object(target_path)
            source = target_cfg.get("target_source", {})
            mode = source.get("mode", "manual") if isinstance(source, dict) else "manual"
            r.ok(f"Target config present: target_source.mode={mode}")
        except Exception as exc:
            r.fail(f"Invalid target_config.json: {exc}")

    sim_path = _config_path(tune_dir, "sim_config.json")
    sim_config: dict = {}
    if sim_path is not None:
        try:
            sim_config = _read_json_object(sim_path)
        except Exception as exc:
            r.fail(f"Invalid sim_config.json: {exc}")

    syn_path = _config_path(tune_dir, "syn_config.json")
    synaptic_step5 = "5" in steps and not bool((sim_config.get("iclamp", {}) or {}).get("enabled", False))
    if syn_path is not None:
        r.ok("Config present: syn_config.json")
        if synaptic_step5:
            _check_input_dependencies(r, tune_dir, syn_path, sim_config)
    elif synaptic_step5:
        r.fail(
            "Step 5 needs syn_config.json for synapse-driven runs, or "
            "sim_config.iclamp.enabled=true for a cell-only run"
        )
    else:
        r.ok("No syn_config.json; valid for intrinsic/current-injection workflows")

    if "4" in steps:
        step4_path = _config_path(tune_dir, "synapse_tuning_config.json")
        if step4_path is None:
            r.ok("No synapse_tuning_config.json; Step 4 will create a neutral starting config")
        else:
            r.ok(f"Step 4 tuning config present: {step4_path}")

    cell_cfg_path = _config_path(tune_dir, "cell_config.json")
    cell_config: dict = {}
    if cell_cfg_path is not None:
        try:
            cell_config = _read_json_object(cell_cfg_path)
            loader_name = cell_loaders.get_cell_loader_name(cell_config)
            r.ok(f"Cell loader: {loader_name}")
            validator = getattr(cell_loaders, "validate_cell_loader_config", None)
            if callable(validator):
                resolved = validator(cell_config, base_dir=tune_dir)
                for key, path in resolved.items():
                    r.ok(f"Loader path {key}: {path}")
            elif cell_loaders.loader_requires_manifest(loader_name):
                raw_manifest = cell_config.get("paths", {}).get("manifest", "manifest.json")
                manifest = Path(str(raw_manifest)).expanduser()
                if not manifest.is_absolute():
                    manifest = tune_dir / manifest
                if manifest.is_file():
                    r.ok(f"Loader manifest: {manifest}")
                else:
                    r.fail(f"Missing loader manifest: {manifest}")
            if loader_name == "allen_manifest":
                try:
                    importlib.import_module("allensdk")
                except Exception as exc:
                    r.fail(f"Allen loader requires a working allensdk installation ({exc})")
            else:
                try:
                    conditions = sim_config.get("conditions")
                    if not isinstance(conditions, dict):
                        raise KeyError("missing conditions object")
                    resolved_conditions = {}
                    for field in ("v_init_mV", "celsius_C"):
                        raw_value = conditions.get(field)
                        if isinstance(raw_value, bool) or raw_value in (None, ""):
                            raise ValueError(f"conditions.{field} must be explicitly numeric")
                        value = float(raw_value)
                        if not math.isfinite(value):
                            raise ValueError(f"conditions.{field} must be finite")
                        resolved_conditions[field] = value
                    r.ok(
                        "Explicit model conditions: "
                        f"v_init_mV={resolved_conditions['v_init_mV']}, "
                        f"celsius_C={resolved_conditions['celsius_C']}"
                    )
                except Exception as exc:
                    r.fail(
                        f"{loader_name} requires explicit sim_config.conditions.v_init_mV "
                        f"and celsius_C ({exc})"
                    )
                if loader_name == 'runtime_cell':
                    r.warn('runtime_cell requires an in-memory notebook override; CLI/SLURM need a saved Python/HOC builder.')
        except Exception as exc:
            r.fail(f"Invalid cell loader configuration: {exc}")

    mod_dir = resolve_modfiles_dir(tune_dir, cell_config)
    has_mod_sources = bool(
        mod_dir is not None and mod_dir.is_dir() and any(mod_dir.glob("*.mod"))
    )
    compiled_dll = find_compiled_mechanism_dll(tune_dir, cell_config=cell_config)
    if compiled_dll is not None:
        r.ok(f"Compiled mechanisms found: {compiled_dll}")
    elif has_mod_sources:
        r.warn("Compiled mechanisms not found")
        if compile_modfiles:
            compiled = _compile_modfiles_if_requested(tune_dir, r, cell_config)
            compiled_dll = find_compiled_mechanism_dll(
                tune_dir, cell_config=cell_config
            )
            if compiled and compiled_dll is not None:
                r.ok(f"Compiled mechanisms created: {compiled_dll}")
            elif compiled:
                r.fail("nrnivmodl finished but no libnrnmech.so found in expected locations")
        else:
            r.warn("Use --compile-modfiles to build now")
    else:
        r.ok("No configured .mod sources; compiled mechanisms are optional")


def _parse_steps(raw_steps: Iterable[str]) -> set[str]:
    valid = {"1", "2", "3", "4", "5", "6", "7"}
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
    ap = argparse.ArgumentParser(
        description="Check SCP environment + local workspace readiness"
    )
    ap.add_argument(
        "--repo-root",
        default=None,
        help="Path to SCP repo root (auto-detected by default)",
    )
    ap.add_argument(
        "--steps",
        nargs="*",
        default=["1", "2", "3", "4", "5"],
        help="Pipeline steps to validate, e.g. --steps 1 2 3 4 5",
    )
    ap.add_argument("--cell", default="PV", help="Cell name for tune checks (default: PV)")
    ap.add_argument("--tune", default="tuned", help="Tune directory name (default: tuned)")
    ap.add_argument("--tune-dir", default=None, help="Explicit tune directory path")
    ap.add_argument("--project", default=None, help="Optional project namespace; omitted uses cells/.")
    ap.add_argument(
        "--skip-tune-check",
        action="store_true",
        help="Skip tune/config/modfile checks",
    )
    ap.add_argument(
        "--compile-modfiles",
        action="store_true",
        help="Compile configured paths.modfiles sources when their library is missing",
    )
    ap.add_argument(
        "--check-act",
        action="store_true",
        help="Check the optional ACT optimizer for Steps 2-3.",
    )
    ap.add_argument(
        "--check-bmtool",
        action="store_true",
        help="Check the optional BMTool integration when Step 4 is selected.",
    )
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
            repo_root = REPO_ROOT_HINT
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
    _check_external_dependencies(
        r,
        repo_root=repo_root,
        steps=steps,
        check_act=bool(args.check_act),
        check_bmtool=bool(args.check_bmtool),
    )

    if not args.skip_tune_check and steps.intersection({"1", "2", "3", "4", "5", "6", "7"}):
        _check_tune_bundle(
            r,
            repo_root=repo_root,
            cell=args.cell,
            tune=args.tune,
            project_name=args.project,
            compile_modfiles=args.compile_modfiles,
            steps=steps,
            tune_dir_override=(Path(args.tune_dir) if args.tune_dir else None),
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

