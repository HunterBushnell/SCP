"""Lazy, environment-first ACT discovery and stage-specific API validation."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .external_repos import (
    ExternalRepoSpec,
    _clone_external_repo,
    activate_external_repo_checkout,
    env_flag,
    resolve_external_repo_checkout,
)


ACT_REPO_URL = "https://github.com/V-Marco/ACT.git"
ACT_SOURCE_REVISION = "2b3fdc190107b8242e3496329cfdf240abd09694"
ACT_REQUIREMENT = f"act @ git+{ACT_REPO_URL}@{ACT_SOURCE_REVISION}"
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


def _install_guidance() -> str:
    return (
        "Install Automatic Cell Tuner once in this Python environment using "
        "SCP's requirements-act.txt (or editable .[act] extra with requirements.txt "
        "constraints). Use the pinned V-Marco/ACT source, not an unqualified "
        "'pip install act'. Restart the kernel after installing or switching ACT."
    )


def _root(repo_root: Optional[Path]) -> Path:
    if repo_root is None:
        from scp_pipeline.runtime import code_root
        repo_root = code_root()
    return Path(repo_root).expanduser().resolve()


def _explicit_checkout(extra_candidates: Sequence[str | Path] = ()) -> Optional[Path]:
    explicit = next(iter(extra_candidates), None) or next(
        (os.environ[name] for name in ACT_ENV_VARS if os.environ.get(name)), None,
    )
    return Path(explicit).expanduser().resolve() if explicit is not None else None


def resolve_act_repo(
    *,
    repo_root: Optional[Path] = None,
    extra_candidates: Sequence[str | Path] = (),
) -> Path:
    """Legacy name: locate ACT's import root, not necessarily a Git checkout.

    No imports of ACT, downloads, or process-state changes occur here. Explicit
    selection wins; an environment-installed package wins over legacy locations.
    """
    explicit = _explicit_checkout(extra_candidates)
    if explicit is not None:
        if not (explicit / ACT_MARKER).is_file():
            raise FileNotFoundError(
                f"Explicit ACT checkout {explicit} must contain {ACT_MARKER}. "
                "Correct/unset the ACT path override to use the environment package. "
                + _install_guidance()
            )
        return explicit
    try:
        spec = importlib.util.find_spec("act")
    except (ImportError, ValueError) as exc:
        raise ImportError("Cannot resolve the current ACT import. " + _install_guidance()) from exc
    if spec is not None:
        if spec.origin is None or spec.submodule_search_locations is None:
            raise ImportError(
                "The importable 'act' is not the expected Automatic Cell Tuner package. "
                "Check for an unrelated distribution or a project-local act name collision. "
                + _install_guidance()
            )
        location = Path(spec.origin).resolve().parent.parent
        if not (location / ACT_MARKER).is_file():
            raise ImportError(
                f"The importable 'act' at {spec.origin} is missing {ACT_MARKER}; "
                "it may be an unrelated package. " + _install_guidance()
            )
        return location
    return resolve_external_repo_checkout(
        _ACT_SPEC,
        repo_root=_root(repo_root),
    )


def ensure_act_on_syspath(
    *,
    repo_root: Optional[Path] = None,
    auto_clone: Optional[bool] = None,
    prepend: bool = True,
) -> Path:
    """Compatibility wrapper: installed ACT needs no path changes or checkout.

    Missing tools are not downloaded unless cloning is explicitly opted into.
    """
    clone = env_flag("SCP_AUTO_CLONE_ACT", default=False) if auto_clone is None else auto_clone
    explicit = _explicit_checkout()
    try:
        location = resolve_act_repo(repo_root=repo_root)
    except FileNotFoundError as exc:
        if not clone or (explicit is not None and explicit.exists()):
            if explicit is not None:
                raise
            raise ModuleNotFoundError(
                "ACT is unavailable. " + _install_guidance()
                + " No checkout was downloaded. Advanced users can set SCP_ACT_PATH "
                "or opt into cloning with SCP_AUTO_CLONE_ACT=1.", name="act",
            ) from exc
        location = _clone_external_repo(
            _ACT_SPEC, repo_root=_root(repo_root), target_dir=explicit, repo_url=None,
        )

    # An already importable environment package needs no sys.path injection.
    # The explicit branch deliberately checks loaded root/submodules first.
    if explicit is None:
        spec = importlib.util.find_spec("act")
        if spec is not None:
            return location
    location = activate_external_repo_checkout(_ACT_SPEC, location, prepend=prepend)
    spec = importlib.util.find_spec("act")
    if spec is not None and spec.origin is not None:
        try:
            Path(spec.origin).resolve().relative_to(location / "act")
        except ValueError:
            raise ImportError(
                f"Selected ACT checkout {location} is shadowed by {spec.origin}. "
                "Use prepend=True or remove the conflicting import path, then restart the kernel."
            ) from None
    return location


def _import_modules(names: Sequence[str], *, location: Path) -> Dict[str, Any]:
    try:
        return {name: importlib.import_module(f"act.{name}") for name in names}
    except Exception as exc:
        raise ImportError(
            f"ACT API could not be imported from {location}: {type(exc).__name__}: {exc}. "
            + _install_guidance()
            + " An installed but broken ACT is not replaced by another checkout."
        ) from exc


def _require_call(callable_object: Any, label: str, *args: Any, **kwargs: Any) -> None:
    if not callable(callable_object):
        raise TypeError(f"expected callable {label}")
    try:
        inspect.signature(callable_object).bind(*args, **kwargs)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label}: {exc}") from exc


def import_act_passive_module(
    *, repo_root: Optional[Path] = None, auto_clone: Optional[bool] = None,
) -> Any:
    """Validate passive helpers without importing optimization dependencies."""
    location = ensure_act_on_syspath(repo_root=repo_root, auto_clone=auto_clone)
    passive = getattr(_import_modules(["passive"], location=location)["passive"], "ACTPassiveModule", None)
    try:
        _require_call(getattr(passive, "compute_spp", None), "ACTPassiveModule.compute_spp", *([None] * 4))
        _require_call(getattr(passive, "compute_gpp", None), "ACTPassiveModule.compute_gpp", *([None] * 5))
    except TypeError as exc:
        raise ImportError(f"Incompatible ACT passive API at {location}: {exc}. " + _install_guidance()) from exc
    return passive


def import_act_active_api(
    *, repo_root: Optional[Path] = None, auto_clone: Optional[bool] = None,
) -> Dict[str, Any]:
    """Validate required active APIs without constructing models or optimizers."""
    location = ensure_act_on_syspath(repo_root=repo_root, auto_clone=auto_clone)
    modules = _import_modules(["cell_model", "module", "simulator", "types", "data_processing"], location=location)
    api = {name: getattr(modules[source], name, None) for name, source in (
        ("ACTCellModel", "cell_model"), ("ACTModule", "module"), ("ACTSimulator", "simulator"),
        ("ConductanceOptions", "types"), ("ConstantCurrentInjection", "types"),
        ("FilterParameters", "types"), ("GaussianCurrentInjection", "types"),
        ("OptimizationParameters", "types"), ("RampCurrentInjection", "types"),
        ("SimulationParameters", "types"), ("get_summary_features", "data_processing"),
    )}
    try:
        for name, value in api.items():
            if not callable(value):
                raise TypeError(f"expected callable {name}")
        constructor_fields = {
            "ACTCellModel": ["cell_name", "path_to_hoc_file", "path_to_mod_files", "passive", "active_channels"],
            "ACTModule": ["name", "cell", "simulation_parameters", "optimization_parameters", "target_file"],
            "ACTSimulator": ["output_folder_name"],
            "SimulationParameters": ["sim_name", "sim_idx", "h_v_init", "h_tstop", "h_dt", "h_celsius"],
            "ConstantCurrentInjection": ["amp", "dur", "delay"],
            "ConductanceOptions": ["variable_name", "blocked", "low", "high", "bounds_variation", "n_slices"],
            "FilterParameters": ["filtered_out_features", "window_of_inspection", "saturation_threshold"],
            "OptimizationParameters": ["n_cpus", "conductance_options", "CI_options", "random_state", "n_estimators", "max_depth", "train_features", "filter_parameters", "spike_threshold", "max_n_spikes"],
            "get_summary_features": ["V", "I", "spike_threshold", "max_n_spikes"],
        }
        for name, fields in constructor_fields.items():
            _require_call(api[name], name, **dict.fromkeys(fields))
        for owner, method, args in (
            ("ACTCellModel", "set_custom_cell_builder", [None, None]),
            ("ACTModule", "run", [None]),
            ("ACTSimulator", "submit_job", [None, None, None]),
            ("ACTSimulator", "run_jobs", [None, None]),
        ):
            _require_call(getattr(api[owner], method, None), f"{owner}.{method}", *args)
    except TypeError as exc:
        raise ImportError(f"Incompatible ACT active API at {location}: {exc}. " + _install_guidance()) from exc
    return api
