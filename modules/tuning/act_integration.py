"""ACT discovery/import helpers for Step 2/3 notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .external_repos import (
    ExternalRepoSpec,
    ensure_external_repo_checkout,
    resolve_external_repo_checkout,
)
from .notebook_setup import resolve_repo_root


ACT_REPO_URL = "https://github.com/V-Marco/ACT.git"
ACT_ENV_VARS = ("SCP_ACT_PATH", "SCP_ACT_DIR", "ACT_PATH", "ACT_ROOT")
ACT_MARKER = Path("act") / "passive.py"
_ACT_SPEC = ExternalRepoSpec(
    display_name="ACT",
    directory_name="ACT",
    package_name="act",
    marker_rel=ACT_MARKER,
    repo_url=ACT_REPO_URL,
    path_env_vars=ACT_ENV_VARS,
    target_dir_env="SCP_ACT_DIR",
    repo_url_env="SCP_ACT_REPO_URL",
    repo_branch_env="SCP_ACT_REPO_BRANCH",
    auto_clone_env="SCP_AUTO_CLONE_ACT",
    canonical_path_env="SCP_ACT_PATH",
    token_env_vars=(
        "SCP_ACT_GIT_TOKEN",
        "SCP_GIT_TOKEN",
        "SCP_GITHUB_TOKEN",
        "GITHUB_TOKEN",
    ),
)


def resolve_act_repo(
    *,
    repo_root: Optional[Path] = None,
    extra_candidates: Sequence[str | Path] = (),
) -> Path:
    """Resolve a local ACT checkout without importing ACT."""
    root = resolve_repo_root(repo_root)
    return resolve_external_repo_checkout(
        _ACT_SPEC,
        repo_root=root,
        extra_candidates=extra_candidates,
    )


def ensure_act_on_syspath(
    *,
    repo_root: Optional[Path] = None,
    auto_clone: Optional[bool] = None,
    prepend: bool = True,
) -> Path:
    """Resolve or action-install ACT and add it to ``sys.path``."""
    root = resolve_repo_root(repo_root)
    return ensure_external_repo_checkout(
        _ACT_SPEC,
        repo_root=root,
        auto_clone=auto_clone,
        prepend=prepend,
    )


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
