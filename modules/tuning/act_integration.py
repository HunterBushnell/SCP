"""ACT discovery/import helpers for Step 2/3 notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import os
import subprocess
import sys

from .notebook_setup import resolve_repo_root


ACT_REPO_URL = "https://github.com/V-Marco/ACT.git"
ACT_ENV_VARS = ("SCP_ACT_PATH", "SCP_ACT_DIR", "ACT_PATH", "ACT_ROOT")
ACT_MARKER = Path("act") / "passive.py"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip() not in {"0", "false", "False", "no", "No", "off", "Off"}


def _is_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ


def _candidate_act_dirs(repo_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for var in ACT_ENV_VARS:
        raw = os.environ.get(var)
        if raw:
            candidates.append(Path(raw).expanduser())
    cwd = Path.cwd()
    candidates.extend(
        [
            repo_root.parent / "mods" / "ACT",
            repo_root / "mods" / "ACT",
            Path.home() / "mods" / "ACT",
            (cwd / ".." / "mods" / "ACT").resolve(),
            (cwd / "mods" / "ACT").resolve(),
        ]
    )
    return candidates


def _looks_like_act_repo(path: Path) -> bool:
    return (path / ACT_MARKER).is_file()


def resolve_act_repo(
    *,
    repo_root: Optional[Path] = None,
    extra_candidates: Sequence[str | Path] = (),
) -> Path:
    """Resolve a local ACT checkout without importing ACT."""
    root = resolve_repo_root(repo_root)
    candidates = [Path(p).expanduser() for p in extra_candidates]
    candidates.extend(_candidate_act_dirs(root))

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if _looks_like_act_repo(resolved):
            return resolved

    expected = ", ".join(ACT_ENV_VARS)
    raise FileNotFoundError(
        f"ACT repo not found. Set one of {expected} or place ACT at ../mods/ACT."
    )


def _clone_act_repo(target_dir: Path) -> Path:
    repo_url = os.environ.get("SCP_ACT_REPO_URL", ACT_REPO_URL)
    branch = os.environ.get("SCP_ACT_REPO_BRANCH", "") or None
    target_dir = Path(target_dir).expanduser().resolve()
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    clone_url = repo_url
    token = (
        os.environ.get("SCP_ACT_GIT_TOKEN")
        or os.environ.get("SCP_GIT_TOKEN")
        or os.environ.get("SCP_GITHUB_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
    )
    if token and clone_url.startswith("https://") and "@" not in clone_url:
        clone_url = clone_url.replace("https://", f"https://{token}@", 1)

    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [clone_url, str(target_dir)]
    subprocess.check_call(cmd)
    return target_dir


def ensure_act_on_syspath(
    *,
    repo_root: Optional[Path] = None,
    auto_clone: Optional[bool] = None,
    prepend: bool = False,
) -> Path:
    """Resolve or clone ACT and add it to `sys.path`."""
    root = resolve_repo_root(repo_root)
    if auto_clone is None:
        auto_clone = _env_flag("SCP_AUTO_CLONE_ACT", default=_is_colab())

    try:
        act_path = resolve_act_repo(repo_root=root)
    except FileNotFoundError:
        if not auto_clone:
            raise
        target = Path(os.environ.get("SCP_ACT_DIR", str(root.parent / "mods" / "ACT")))
        if target.exists() and any(target.iterdir()):
            raise FileNotFoundError(
                f"SCP_ACT_DIR exists but is not a valid ACT checkout: {target}"
            )
        act_path = _clone_act_repo(target)

    os.environ["SCP_ACT_PATH"] = str(act_path)
    path_str = str(act_path)
    if path_str not in sys.path:
        if prepend:
            sys.path.insert(0, path_str)
        else:
            sys.path.append(path_str)
    return act_path


def import_act_passive_module(*, repo_root: Optional[Path] = None) -> Any:
    """Return ACTPassiveModule after resolving ACT."""
    ensure_act_on_syspath(repo_root=repo_root)
    from act.passive import ACTPassiveModule

    return ACTPassiveModule


def import_act_active_api(*, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Return the core ACT classes used by active tuning notebooks."""
    ensure_act_on_syspath(repo_root=repo_root)
    from act.cell_model import ACTCellModel
    from act.module import ACTModule
    from act.simulator import ACTSimulator
    from act.types import (
        ConductanceOptions,
        ConstantCurrentInjection,
        FilterParameters,
        GaussianCurrentInjection,
        OptimizationParameters,
        RampCurrentInjection,
        SimulationParameters,
    )

    return {
        "ACTCellModel": ACTCellModel,
        "ACTModule": ACTModule,
        "ACTSimulator": ACTSimulator,
        "ConductanceOptions": ConductanceOptions,
        "ConstantCurrentInjection": ConstantCurrentInjection,
        "FilterParameters": FilterParameters,
        "GaussianCurrentInjection": GaussianCurrentInjection,
        "OptimizationParameters": OptimizationParameters,
        "RampCurrentInjection": RampCurrentInjection,
        "SimulationParameters": SimulationParameters,
    }
