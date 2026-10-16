"""Lazy, environment-first resolution of the optional BMTool dependency."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
from pathlib import Path

from .external_repos import (
    ExternalRepoSpec,
    _clone_external_repo,
    activate_external_repo_checkout,
    env_flag,
    resolve_external_repo_checkout,
)


BMTOOL_REQUIREMENT = "bmtool==0.8.6"
BMTOOL_REPO_URL = "https://github.com/cyneuro/bmtool.git"
BMTOOL_ENV_VARS = (
    "SCP_BMTOOL_PATH", "SCP_BMTOOL_DIR", "BMTOOL_PATH", "BMTOOL_ROOT",
)
_BMTOOL_SPEC = ExternalRepoSpec(
    display_name="BMTool",
    directory_name="bmtool",
    package_name="bmtool",
    marker_rel=Path("bmtool") / "synapses.py",
    repo_url=BMTOOL_REPO_URL,
    path_env_vars=BMTOOL_ENV_VARS,
    target_dir_env="SCP_BMTOOL_DIR",
    repo_url_env="SCP_BMTOOL_REPO_URL",
    repo_branch_env="SCP_BMTOOL_REPO_BRANCH",
    auto_clone_env="SCP_AUTO_CLONE_BMTOOL",
    canonical_path_env="SCP_BMTOOL_PATH",
    token_env_vars=("SCP_BMTOOL_GIT_TOKEN", "SCP_GIT_TOKEN", "SCP_GITHUB_TOKEN", "GITHUB_TOKEN"),
)


def _install_guidance() -> str:
    return (
        f"Install {BMTOOL_REQUIREMENT} once in the notebook/CLI Python environment "
        "using SCP's suggested requirements-bmtool.txt (or editable .[bmtool] "
        "extra with requirements.txt constraints). Restart the kernel after "
        "installing or switching BMTool."
    )


def _activate_checkout(checkout: Path, *, prepend: bool) -> Path:
    location = activate_external_repo_checkout(_BMTOOL_SPEC, checkout, prepend=prepend)
    spec = importlib.util.find_spec("bmtool")
    if spec is not None and spec.origin is not None:
        try:
            Path(spec.origin).resolve().relative_to(location / "bmtool")
        except ValueError:
            raise ImportError(
                f"Selected BMTool checkout {location} is shadowed by {spec.origin}. "
                "Use prepend=True or remove the conflicting import path, then restart the kernel."
            ) from None
    return location


def ensure_bmtool_available(
    *,
    repo_root: Path | None = None,
    auto_clone: bool | None = None,
    repo_url: str | None = None,
    target_dir: Path | None = None,
    prepend: bool = True,
) -> Path:
    """Use an explicit checkout, an importable package, or a legacy checkout.

    Installed/editable packages need no path injection. Missing dependencies do
    not trigger downloads unless automatic cloning is explicitly opted into.
    The returned path is the package's import root, not necessarily a checkout.
    """
    if repo_root is None:
        from scp_pipeline.runtime import code_root
        repo_root = code_root()
    root = Path(repo_root).expanduser().resolve()
    clone = (
        env_flag("SCP_AUTO_CLONE_BMTOOL", default=False)
        if auto_clone is None else auto_clone
    )
    explicit = target_dir or next(
        (os.environ[name] for name in BMTOOL_ENV_VARS if os.environ.get(name)), None,
    )
    if explicit is not None:
        checkout = Path(explicit).expanduser().resolve()
        if not (checkout / _BMTOOL_SPEC.marker_rel).is_file():
            if not checkout.exists() and clone:
                checkout = _clone_external_repo(
                    _BMTOOL_SPEC, repo_root=root, target_dir=checkout, repo_url=repo_url,
                )
            else:
                raise FileNotFoundError(
                    f"Explicit BMTool checkout {checkout} must contain bmtool/synapses.py. "
                    "Correct/unset the BMTool path override to use the environment package. "
                    + _install_guidance()
                )
        return _activate_checkout(checkout, prepend=prepend)

    try:
        spec = importlib.util.find_spec("bmtool")
    except (ImportError, ValueError) as exc:
        raise ImportError("Cannot resolve the current BMTool import. " + _install_guidance()) from exc
    if spec is not None:
        if spec.origin is None:
            raise ImportError(
                "The importable bmtool is a namespace package, not the expected BMTool package. "
                "Check for a project-local bmtool name collision. " + _install_guidance()
            )
        return Path(spec.origin).resolve().parent.parent

    try:
        checkout = resolve_external_repo_checkout(_BMTOOL_SPEC, repo_root=root)
    except FileNotFoundError as exc:
        if not clone:
            raise ModuleNotFoundError(
                "BMTool is unavailable. " + _install_guidance()
                + " No checkout was downloaded. Advanced users can select a checkout "
                "with SCP_BMTOOL_PATH or opt into cloning with SCP_AUTO_CLONE_BMTOOL=1.",
                name="bmtool",
            ) from exc
        checkout = _clone_external_repo(
            _BMTOOL_SPEC, repo_root=root, target_dir=target_dir, repo_url=repo_url,
        )
    return _activate_checkout(checkout, prepend=prepend)


def import_bmtool_synapse_api(*, repo_root: Path | None = None, auto_clone: bool | None = None):
    """Import and validate SCP's required API without constructing a tuner."""
    location = ensure_bmtool_available(repo_root=repo_root, auto_clone=auto_clone)
    try:
        module = importlib.import_module("bmtool.synapses")
    except Exception as exc:
        raise ImportError(
            f"BMTool synapse API could not be imported from {location}: "
            f"{type(exc).__name__}: {exc}. " + _install_guidance()
            + " An installed but broken BMTool is not replaced by another checkout."
        ) from exc

    tuner = getattr(module, "SynapseTuner", None)
    optimizer = getattr(module, "SynapseOptimizer", None)
    try:
        if not callable(tuner) or not callable(optimizer):
            raise TypeError("expected callable SynapseTuner and SynapseOptimizer")
        inspect.signature(tuner).bind(
            mechanisms_dir=None, conn_type_settings={}, general_settings={},
            connection="", current_name="i", other_vars_to_record=[],
            slider_vars=[], hoc_cell=None,
        )
        inspect.signature(optimizer).bind(object())
    except (TypeError, ValueError) as exc:
        raise ImportError(
            f"Incompatible BMTool synapse API at {location}: {exc}. "
            "SCP requires SynapseTuner(hoc_cell=..., ...) and SynapseOptimizer(tuner). "
            + _install_guidance()
        ) from exc
    return tuner, optimizer
