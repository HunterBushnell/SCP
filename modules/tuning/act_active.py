"""ACT active-tuning adapter for Step 3.

This module treats ACT as an external optimizer library. SCP owns the tune
directory, target data, generated cell builder, and run configuration; ACT owns
the model fitting machinery.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import csv
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
import warnings

import numpy as np

from .act_integration import ensure_act_on_syspath
from .active import DEFAULT_BIO_FI_REFERENCE
from .allen_nwb import DEFAULT_FI_STIMULUS_NAMES, write_allen_nwb_fi_target_csv
from .notebook_setup import resolve_repo_root, resolve_tune_dir
from .targets import fi_curve_from_config, load_target_config


CONFIG_NAME = "act_active_config.json"
BUILDER_NAME = "cell_builder.py"
TARGET_SF_NAME = "target_sf.csv"
PREDICTION_PREFIX = "prediction_"
METRICS_PREFIX = "metrics_"
RUN_MANIFEST_PREFIX = "run_manifest_"
EVALUATION_MANIFEST_NAME = "evaluation_manifest.json"
FIXED_PREDICTIONS_NAME = "fixed_predictions.json"

DEFAULT_PASSIVE_NAMES = ["g_pas", "e_pas", "gbar_Ih"]
DEFAULT_ACTIVE_CHANNELS = [
    "gbar_Nap",
    "gbar_Im_v2",
    "gbar_K_T",
    "gbar_NaTa",
    "gbar_Kd",
    "gbar_Ca_LVA",
    "gbar_Ca_HVA",
    "gbar_Kv2like",
    "gbar_Kv3_1",
]


def _configured_loader_name(tune_dir: Path) -> str:
    from modules.loaders import get_cell_loader_name

    for candidate in (
        tune_dir / "cell_configs" / "cell_config.json",
        tune_dir / "cell_config.json",
    ):
        if candidate.is_file():
            value = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise TypeError(f"Expected a JSON object in {candidate}")
            return get_cell_loader_name(value)
    raise FileNotFoundError(f"Missing cell_config.json under ACT tune {tune_dir}")


def _warn_for_experimental_act_loader(cfg: Mapping[str, Any]) -> None:
    loader_name = str(cfg.get("cell_loader", "allen_manifest"))
    if loader_name != "allen_manifest":
        warnings.warn(
            f"ACT execution with SCP loader {loader_name!r} is experimental; "
            "the model-neutral SCP protocols remain supported independently of ACT.",
            RuntimeWarning,
            stacklevel=2,
        )


@contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(old))


def default_act_workspace(tune_dir: str | Path) -> Path:
    """Return the default ACT workspace path inside a tune directory."""
    return Path(tune_dir).expanduser().resolve() / "act_workspace"


def default_target_fi(cell_name: str) -> tuple[list[float], list[float]]:
    """Return bundled example FI target points for a known example cell."""
    points = DEFAULT_BIO_FI_REFERENCE.get(str(cell_name).upper())
    if not points:
        points = [(50, 0), (100, 5), (150, 15), (200, 30), (250, 45), (300, 60)]
    return [float(p[0]) for p in points], [float(p[1]) for p in points]


def target_fi_for_tune(tune_dir: str | Path, cell_name: str) -> tuple[list[float], list[float]]:
    """Return tune-local FI targets when configured, else bundled defaults."""
    target_config = load_target_config(tune_dir)
    currents, rates = fi_curve_from_config(target_config)
    if currents:
        return currents, rates
    return default_target_fi(cell_name)


def default_act_module_specs(cell_name: str = "PV") -> dict[str, dict[str, Any]]:
    """Return editable ACT module presets based on the bundled ADB examples."""
    key = str(cell_name).strip().upper()
    if key == "PV":
        return {
            "lto": {
                "enabled": True,
                "name": "seg_lto",
                "description": "Low-threshold/near-threshold channels",
                "conductances": [
                    {"variable_name": "gbar_Nap", "low": 0.00008, "high": 0.08, "n_slices": 5},
                    {"variable_name": "gbar_K_T", "low": 0.0001, "high": 0.1, "n_slices": 5},
                    {"variable_name": "gbar_Im_v2", "low": 0.00002, "high": 0.02, "n_slices": 5},
                ],
            },
            "spiking": {
                "enabled": True,
                "name": "seg_spiking",
                "description": "Fast spiking channels",
                "conductances": [
                    {"variable_name": "gbar_NaTa", "low": 0.001, "high": 0.5, "n_slices": 5},
                    {"variable_name": "gbar_Kd", "low": 0.0008, "high": 0.8, "n_slices": 5},
                ],
            },
            "bursting": {
                "enabled": True,
                "name": "seg_bursting",
                "description": "Calcium and high-threshold potassium channels",
                "conductances": [
                    {"variable_name": "gbar_Ca_LVA", "low": 0.0000001, "high": 0.01, "n_slices": 5},
                    {"variable_name": "gbar_Ca_HVA", "low": 0.0000001, "high": 0.01, "n_slices": 5},
                    {"variable_name": "gbar_Kv2like", "low": 0.0000001, "high": 0.2, "n_slices": 5},
                    {"variable_name": "gbar_Kv3_1", "low": 0.0000001, "high": 0.9, "n_slices": 5},
                ],
            },
        }

    return {
        "lto": {
            "enabled": True,
            "name": "seg_lto",
            "description": "Low-threshold/near-threshold channels",
            "conductances": [
                {"variable_name": "gbar_Nap", "low": 0.0008, "high": 0.008, "n_slices": 5},
                {"variable_name": "gbar_K_T", "low": 0.001, "high": 0.01, "n_slices": 5},
                {"variable_name": "gbar_Im_v2", "low": 0.0002, "high": 0.002, "n_slices": 5},
            ],
        },
        "spiking": {
            "enabled": True,
            "name": "seg_spiking",
            "description": "Fast spiking channels",
            "conductances": [
                {"variable_name": "gbar_NaTa", "low": 0.01, "high": 0.25, "n_slices": 5},
                {"variable_name": "gbar_Kd", "low": 0.008, "high": 0.08, "n_slices": 5},
            ],
        },
        "bursting": {
            "enabled": True,
            "name": "seg_bursting",
            "description": "Calcium and high-threshold potassium channels",
            "conductances": [
                {"variable_name": "gbar_Ca_LVA", "low": 0.0008, "high": 0.008, "n_slices": 5},
                {"variable_name": "gbar_Ca_HVA", "low": 0.001, "high": 0.01, "n_slices": 5},
                {"variable_name": "gbar_Kv2like", "low": 0.001, "high": 0.02, "n_slices": 5},
                {"variable_name": "gbar_Kv3_1", "low": 0.01, "high": 0.09, "n_slices": 5},
            ],
        },
    }


def default_act_active_config(
    *,
    repo_root: str | Path,
    tune_dir: str | Path,
    cell_name: str,
    tune_name: str,
    workspace: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Build a JSON-serializable default ACT active-tuning config."""
    root = resolve_repo_root(Path(repo_root))
    tune_path = Path(tune_dir).expanduser().resolve()
    workspace_path = (
        Path(workspace).expanduser().resolve()
        if workspace
        else default_act_workspace(tune_path)
    )
    target_amps, target_freqs = target_fi_for_tune(tune_path, cell_name)

    return {
        "version": 1,
        "repo_root": str(root),
        "tune_dir": str(tune_path),
        "workspace": str(workspace_path),
        "cell_name": str(cell_name),
        "tune_name": str(tune_name),
        "cell_builder": {
            "path": BUILDER_NAME,
            "function": "build_cell",
        },
        "target": {
            "mode": "fi_arrays",
            "path": TARGET_SF_NAME,
            "fi_currents_pA": target_amps,
            "fi_frequencies_hz": target_freqs,
        },
        "act_cell": {
            "passive": list(DEFAULT_PASSIVE_NAMES),
            "active_channels": list(DEFAULT_ACTIVE_CHANNELS),
        },
        "simulation": {
            "h_v_init": -50.0,
            "h_tstop": 1000.0,
            "h_dt": 0.1,
            "h_celsius": 37.0,
            "ci_delay_ms": 100.0,
            "ci_dur_ms": 700.0,
            "ci_amps_pA": target_amps,
        },
        "optimizer": {
            "n_cpus": 4,
            "random_state": 42,
            "n_estimators": 1000,
            "max_depth": None,
            "train_features": ["spike_frequency", "mean_i"],
            "spike_threshold": -20.0,
            "max_n_spikes": 20,
        },
        "filter": {
            "filtered_out_features": None,
            "window_of_inspection": [100, 800],
            "saturation_threshold": -55.0,
        },
        "modules": default_act_module_specs(cell_name),
    }


def prepare_act_active_workspace(
    *,
    repo_root: str | Path,
    tune_dir: str | Path,
    cell_name: str,
    tune_name: str,
    workspace: Optional[str | Path] = None,
    target_mode: Optional[str] = None,
    fi_currents_pA: Optional[Sequence[float]] = None,
    fi_frequencies_hz: Optional[Sequence[float]] = None,
    fi_csv_path: Optional[str | Path] = None,
    trace_npy_path: Optional[str | Path] = None,
    nwb_path: Optional[str | Path] = None,
    nwb_stimulus_names: Optional[Sequence[str]] = None,
    nwb_include_negative_currents: bool = False,
    nwb_min_current_pA: Optional[float] = 0.0,
    nwb_max_current_pA: Optional[float] = None,
    nwb_average_repeats: bool = True,
    nwb_spike_threshold_mV: float = -20.0,
    nwb_refractory_ms: float = 1.0,
    passive_names: Optional[Sequence[str]] = None,
    active_channels: Optional[Sequence[str]] = None,
    module_specs: Optional[Mapping[str, Any]] = None,
    sim_params: Optional[Mapping[str, Any]] = None,
    optimizer: Optional[Mapping[str, Any]] = None,
    filter_params: Optional[Mapping[str, Any]] = None,
    overwrite_config: bool = True,
    preserve_existing: bool = False,
) -> dict[str, Any]:
    """Create/update an ACT workspace and write its config, builder, and target."""
    root = resolve_repo_root(Path(repo_root))
    tune_path = Path(tune_dir).expanduser().resolve()
    requested_workspace = (
        Path(workspace).expanduser().resolve()
        if workspace is not None
        else default_act_workspace(tune_path)
    )
    existing_path = requested_workspace / CONFIG_NAME
    if preserve_existing and existing_path.is_file():
        cfg = load_act_active_config(existing_path)
        cfg = json.loads(json.dumps(cfg))
        cfg.update(
            {
                "repo_root": str(root),
                "tune_dir": str(tune_path),
                "workspace": str(requested_workspace),
                "cell_name": str(cell_name),
                "tune_name": str(tune_name),
            }
        )
        cfg.setdefault("cell_builder", {"path": BUILDER_NAME, "function": "build_cell"})
        cfg.setdefault("target", {})
        cfg.setdefault("act_cell", {})
        cfg.setdefault("simulation", {})
        cfg.setdefault("optimizer", {})
        cfg.setdefault("filter", {})
        cfg.setdefault("modules", {})
    else:
        cfg = default_act_active_config(
            repo_root=root,
            tune_dir=tune_path,
            cell_name=cell_name,
            tune_name=tune_name,
            workspace=requested_workspace,
        )
    loader_name = _configured_loader_name(tune_path)
    cfg["cell_loader"] = loader_name
    cfg["act_loader_status"] = (
        "supported" if loader_name == "allen_manifest" else "experimental"
    )
    workspace_path = Path(cfg["workspace"])
    workspace_path.mkdir(parents=True, exist_ok=True)
    # A killed multiprocessing run can leave this transient adapter input behind.
    # It is never a user result and must not affect preparation/probing.
    (workspace_path / FIXED_PREDICTIONS_NAME).unlink(missing_ok=True)

    if passive_names is not None:
        cfg["act_cell"]["passive"] = list(passive_names)
    if active_channels is not None:
        cfg["act_cell"]["active_channels"] = list(active_channels)
    if module_specs is not None:
        cfg["modules"] = json.loads(json.dumps(module_specs))
    if sim_params:
        cfg["simulation"].update(dict(sim_params))
    if optimizer:
        cfg["optimizer"].update(dict(optimizer))
    if filter_params:
        cfg["filter"].update(dict(filter_params))

    cfg["target"]["mode"] = str(target_mode or cfg["target"].get("mode") or "fi_arrays")
    validate_act_module_specs(cfg.get("modules") or {})
    _write_cell_builder(
        workspace=workspace_path,
        repo_root=root,
        tune_dir=tune_path,
        cell_name=cell_name,
    )
    target_path = _prepare_target_file(
        cfg=cfg,
        workspace=workspace_path,
        fi_currents_pA=fi_currents_pA,
        fi_frequencies_hz=fi_frequencies_hz,
        fi_csv_path=fi_csv_path,
        trace_npy_path=trace_npy_path,
        nwb_path=nwb_path,
        nwb_stimulus_names=nwb_stimulus_names,
        nwb_include_negative_currents=nwb_include_negative_currents,
        nwb_min_current_pA=nwb_min_current_pA,
        nwb_max_current_pA=nwb_max_current_pA,
        nwb_average_repeats=nwb_average_repeats,
        nwb_spike_threshold_mV=nwb_spike_threshold_mV,
        nwb_refractory_ms=nwb_refractory_ms,
    )
    cfg["target"]["path"] = str(_path_relative_to_workspace(target_path, workspace_path))
    if _target_point_count(cfg) < 1:
        raise ValueError("Prepared ACT target contains no usable traces or FI points.")

    config_path = workspace_path / CONFIG_NAME
    if config_path.exists() and not overwrite_config:
        raise FileExistsError(f"Config already exists: {config_path}")
    _write_json(config_path, cfg)
    return workspace_summary(config_path)


def workspace_summary(config_path: str | Path) -> dict[str, Any]:
    """Return concise metadata for a prepared ACT active workspace."""
    cfg = load_act_active_config(config_path)
    workspace = Path(cfg["workspace"]).expanduser().resolve()
    target_path = resolve_workspace_path(workspace, cfg["target"]["path"])
    modules = cfg.get("modules", {}) or {}
    target_points = _target_point_count(cfg)
    return {
        "workspace": str(workspace),
        "config": str(workspace / CONFIG_NAME),
        "builder": str(workspace / BUILDER_NAME),
        "target": str(target_path),
        "target_mode": cfg.get("target", {}).get("mode"),
        "cell_loader": cfg.get("cell_loader", "allen_manifest"),
        "act_loader_status": cfg.get("act_loader_status", "supported"),
        "modules": [key for key, spec in modules.items() if spec.get("enabled", True)],
        "target_point_count": target_points,
        "workload": estimate_act_workload(cfg),
        "output_status": act_output_status(cfg),
        "fingerprint": act_config_fingerprint(cfg),
    }


def load_act_active_config(config_or_workspace: str | Path) -> dict[str, Any]:
    """Load `act_active_config.json` from a config path or workspace path."""
    path = Path(config_or_workspace).expanduser()
    if path.is_dir():
        path = path / CONFIG_NAME
    data = json.loads(path.read_text(encoding="utf-8"))
    workspace = Path(data.get("workspace", path.parent)).expanduser()
    if not workspace.is_absolute():
        workspace = (path.parent / workspace).resolve()
    data["workspace"] = str(workspace)
    data.setdefault("repo_root", str(resolve_repo_root(Path.cwd())))
    data.setdefault("tune_dir", str(workspace.parent))
    return data


def validate_act_module_specs(modules: Mapping[str, Any]) -> None:
    """Validate compact ACT module definitions before expensive execution."""

    if not isinstance(modules, Mapping) or not modules:
        raise ValueError("ACT configuration must define at least one module.")
    for module_key, raw_spec in modules.items():
        if not isinstance(raw_spec, Mapping):
            raise TypeError(f"ACT module {module_key!r} must be an object/dict.")
        if not raw_spec.get("enabled", True):
            continue
        conductances = raw_spec.get("conductances") or []
        if not conductances:
            raise ValueError(f"ACT module {module_key!r} has no conductances.")
        names: set[str] = set()
        for index, item in enumerate(conductances):
            if not isinstance(item, Mapping):
                raise TypeError(
                    f"ACT module {module_key!r} conductance {index} must be an object/dict."
                )
            variable = str(item.get("variable_name") or "").strip()
            if not variable:
                raise ValueError(
                    f"ACT module {module_key!r} conductance {index} needs variable_name."
                )
            if variable in names:
                raise ValueError(f"ACT conductance {variable!r} is defined more than once.")
            names.add(variable)
            if bool(item.get("blocked", False)):
                continue
            low = _float_or_none(item.get("low"))
            high = _float_or_none(item.get("high"))
            variation = _float_or_none(item.get("bounds_variation"))
            if variation is None and (low is None or high is None):
                raise ValueError(
                    f"ACT conductance {variable!r} needs low/high bounds or bounds_variation."
                )
            if low is not None and high is not None and low >= high:
                raise ValueError(
                    f"ACT conductance {variable!r} requires low < high; got {low:g}, {high:g}."
                )
            if int(item.get("n_slices", 1)) < 1:
                raise ValueError(f"ACT conductance {variable!r} n_slices must be at least 1.")


def estimate_act_workload(
    config_or_mapping: str | Path | Mapping[str, Any],
    *,
    modules: str | Sequence[str] = "all",
) -> dict[str, Any]:
    """Estimate ACT simulations without constructing a model or importing ACT."""

    cfg = _coerce_act_config(config_or_mapping)
    target_points = _target_point_count(cfg)
    requested = None
    if modules != "all":
        requested = {str(modules)} if isinstance(modules, str) else {str(v) for v in modules}
    per_module: dict[str, dict[str, int]] = {}
    for key, spec in (cfg.get("modules") or {}).items():
        if not spec.get("enabled", True) or (requested is not None and key not in requested):
            continue
        combinations = 1
        for item in spec.get("conductances", []) or []:
            if not bool(item.get("blocked", False)):
                combinations *= max(1, int(item.get("n_slices", 1)))
        training = combinations * target_points
        per_module[str(key)] = {
            "conductance_combinations": combinations,
            "current_steps": target_points,
            "training_traces": training,
            "evaluation_traces": target_points,
            "total_traces": training + target_points,
        }
    return {
        "target_points": target_points,
        "modules": per_module,
        "training_traces": sum(v["training_traces"] for v in per_module.values()),
        "evaluation_traces": sum(v["evaluation_traces"] for v in per_module.values()),
        "total_traces": sum(v["total_traces"] for v in per_module.values()),
    }


def act_config_fingerprint(config_or_mapping: str | Path | Mapping[str, Any]) -> str:
    """Hash the resolved ACT configuration and target contents."""

    cfg = _coerce_act_config(config_or_mapping)
    payload = json.loads(json.dumps(cfg))
    target_path = _target_file_for_act(cfg)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    if target_path.is_file():
        digest.update(target_path.read_bytes())
    else:
        digest.update(f"missing:{target_path}".encode("utf-8"))
    return digest.hexdigest()


def act_source_fingerprints(
    config_or_mapping: str | Path | Mapping[str, Any],
) -> dict[str, str]:
    """Return separate ACT config and prepared-target content hashes."""

    cfg = _coerce_act_config(config_or_mapping)
    workspace = Path(cfg["workspace"]).resolve()
    config_path = workspace / CONFIG_NAME
    target_path = _target_file_for_act(cfg)
    return {
        "config": _file_fingerprint(config_path),
        "target": _file_fingerprint(target_path),
    }


def act_output_status(
    config_or_mapping: str | Path | Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Describe saved module outputs as current, stale, legacy, partial, or missing."""

    cfg = _coerce_act_config(config_or_mapping)
    workspace = Path(cfg["workspace"]).resolve()
    config_hash = act_config_fingerprint(cfg)
    source_hashes = act_source_fingerprints(cfg)
    status: dict[str, dict[str, Any]] = {}
    prior: dict[str, float] = {}
    for key, spec in (cfg.get("modules") or {}).items():
        if not spec.get("enabled", True):
            continue
        prediction_path = workspace / f"{PREDICTION_PREFIX}{key}.json"
        metrics_path = workspace / f"{METRICS_PREFIX}{key}.csv"
        manifest_path = workspace / f"{RUN_MANIFEST_PREFIX}{key}.json"
        expected_prior_hash = _mapping_fingerprint(prior)
        state = "missing"
        manifest: dict[str, Any] = {}
        saved_prediction: dict[str, float] = {}
        if prediction_path.is_file():
            try:
                raw_prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
                saved_prediction = {
                    str(name): float(value) for name, value in raw_prediction.items()
                }
            except Exception:
                saved_prediction = {}
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                state = "partial"
            else:
                if (
                    manifest.get("status") != "complete"
                    or not prediction_path.is_file()
                    or not metrics_path.is_file()
                ):
                    state = "partial"
                elif any(
                    name not in manifest
                    for name in (
                        "config_fingerprint",
                        "prior_predictions_fingerprint",
                        "prediction_fingerprint",
                    )
                ):
                    state = "legacy"
                elif (
                    manifest.get("config_fingerprint") != config_hash
                    or (
                        "config_file_fingerprint" in manifest
                        and manifest.get("config_file_fingerprint")
                        != source_hashes["config"]
                    )
                    or (
                        "target_file_fingerprint" in manifest
                        and manifest.get("target_file_fingerprint")
                        != source_hashes["target"]
                    )
                    or manifest.get("prior_predictions_fingerprint") != expected_prior_hash
                    or manifest.get("prediction_fingerprint")
                    != _mapping_fingerprint(saved_prediction)
                ):
                    state = "stale"
                else:
                    state = "current"
        elif prediction_path.is_file():
            state = "legacy"
        elif metrics_path.is_file():
            state = "partial"
        status[str(key)] = {
            "status": state,
            "prediction_path": str(prediction_path),
            "metrics_path": str(metrics_path),
            "manifest_path": str(manifest_path),
            "provenance_verified": state == "current",
        }
        prior.update(saved_prediction)
    return status


def load_act_module_metrics(config_or_workspace: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load available per-module ACT metrics CSV files."""

    cfg = load_act_active_config(config_or_workspace)
    workspace = Path(cfg["workspace"]).resolve()
    result: dict[str, list[dict[str, Any]]] = {}
    for key in (cfg.get("modules") or {}):
        path = workspace / f"{METRICS_PREFIX}{key}.csv"
        if path.is_file():
            with path.open("r", encoding="utf-8", newline="") as handle:
                result[str(key)] = list(csv.DictReader(handle))
    return result


def run_act_active_module(
    config_or_workspace: str | Path,
    module_key: str,
    *,
    n_cpus: Optional[int] = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run one configured ACT active module and save JSON predictions."""
    cfg = load_act_active_config(config_or_workspace)
    _warn_for_experimental_act_loader(cfg)
    workspace = Path(cfg["workspace"]).resolve()
    module_spec = _get_module_spec(cfg, module_key)
    if not module_spec.get("enabled", True):
        raise ValueError(f"ACT module {module_key!r} is disabled.")

    module_name = module_spec.get("name") or str(module_key)
    output_dir = workspace / f"module_{module_name}"
    prediction_path = workspace / f"{PREDICTION_PREFIX}{module_key}.json"
    metrics_path = workspace / f"{METRICS_PREFIX}{module_key}.csv"
    manifest_path = workspace / f"{RUN_MANIFEST_PREFIX}{module_key}.json"
    existing_artifacts = [
        path
        for path in (output_dir, prediction_path, metrics_path, manifest_path)
        if path.exists()
    ]
    if existing_artifacts and not overwrite:
        raise FileExistsError(
            f"ACT output already exists for module {module_key!r}: "
            + ", ".join(str(path) for path in existing_artifacts)
            + ". Enable explicit overwrite to rerun."
        )
    if overwrite:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        prediction_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    validate_act_module_specs({module_key: module_spec})
    prior_predictions = _load_prior_predictions(cfg, before_module=module_key)
    config_hash = act_config_fingerprint(cfg)
    source_hashes = act_source_fingerprints(cfg)
    manifest = {
        "version": 1,
        "module": str(module_key),
        "module_name": str(module_name),
        "status": "running",
        "started_at": time.time(),
        "config_fingerprint": config_hash,
        "config_file_fingerprint": source_hashes["config"],
        "target_file_fingerprint": source_hashes["target"],
        "prior_predictions": prior_predictions,
        "prior_predictions_fingerprint": _mapping_fingerprint(prior_predictions),
        "prediction_path": str(prediction_path),
        "metrics_path": str(metrics_path),
        "output_dir": str(output_dir),
    }
    _write_json_atomic(manifest_path, manifest)
    fixed_path = workspace / FIXED_PREDICTIONS_NAME
    _write_json_atomic(fixed_path, prior_predictions)
    started = time.monotonic()
    try:
        act_api = _import_act_api(cfg)
        builder = _import_workspace_builder(cfg)
        train_cell = _build_act_cell(
            cfg, act_api=act_api, builder=builder, module_spec=module_spec
        )
        train_cell.prediction.update(prior_predictions)

        simulation_parameters = _build_simulation_parameters(cfg, act_api)
        optimization_parameters = _build_optimization_parameters(
            cfg, act_api, module_spec, n_cpus=n_cpus
        )
        target_file = str(_target_file_for_act(cfg))
        with _pushd(workspace):
            act_module = act_api["ACTModule"](
                name=module_name,
                cell=train_cell,
                simulation_parameters=simulation_parameters,
                optimization_parameters=optimization_parameters,
                target_file=target_file,
            )
            metrics = act_module.run()

        full_prediction = {
            str(k): float(v) for k, v in act_module.cell.prediction.items()
        }
        module_variables = {
            str(item["variable_name"])
            for item in module_spec.get("conductances", [])
        }
        prediction = {
            name: value
            for name, value in full_prediction.items()
            if name in module_variables
        }
        missing_predictions = sorted(module_variables - set(prediction))
        if missing_predictions:
            raise RuntimeError(
                "ACT did not return predictions for: " + ", ".join(missing_predictions)
            )
        _write_json_atomic(prediction_path, prediction)
        metrics.to_csv(metrics_path, index=False)
    except BaseException as exc:
        manifest.update(
            {
                "status": "incomplete",
                "finished_at": time.time(),
                "runtime_seconds": time.monotonic() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json_atomic(manifest_path, manifest)
        raise
    finally:
        fixed_path.unlink(missing_ok=True)

    manifest.update(
        {
            "status": "complete",
            "finished_at": time.time(),
            "runtime_seconds": time.monotonic() - started,
            "prediction_fingerprint": _mapping_fingerprint(prediction),
        }
    )
    _write_json_atomic(manifest_path, manifest)
    return {
        "module": module_key,
        "module_name": module_name,
        "prediction": prediction,
        "prediction_path": str(prediction_path),
        "metrics_path": str(metrics_path),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "runtime_seconds": manifest["runtime_seconds"],
        "config_fingerprint": config_hash,
    }


def run_act_active_modules(
    config_or_workspace: str | Path,
    *,
    modules: str | Sequence[str] = "all",
    n_cpus: Optional[int] = None,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Run one or more configured ACT active modules in config order."""
    cfg = load_act_active_config(config_or_workspace)
    keys = list((cfg.get("modules") or {}).keys())
    if modules != "all":
        requested = [modules] if isinstance(modules, str) else list(modules)
        keys = [key for key in keys if key in requested]
    return [
        run_act_active_module(cfg["workspace"], key, n_cpus=n_cpus, overwrite=overwrite)
        for key in keys
        if (cfg.get("modules", {}).get(key, {}) or {}).get("enabled", True)
    ]


def collect_act_predictions(config_or_workspace: str | Path) -> dict[str, float]:
    """Merge saved per-module prediction JSON files in config order."""
    cfg = load_act_active_config(config_or_workspace)
    return _load_prior_predictions(cfg)


def evaluate_act_predictions(
    config_or_workspace: str | Path,
    *,
    predictions: Optional[Mapping[str, float]] = None,
    n_cpus: Optional[int] = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run ACT FI evaluation using predicted conductances without editing files."""
    cfg = load_act_active_config(config_or_workspace)
    workspace = Path(cfg["workspace"]).resolve()
    (workspace / FIXED_PREDICTIONS_NAME).unlink(missing_ok=True)
    prediction = dict(predictions or collect_act_predictions(workspace))
    if not prediction:
        raise ValueError("No ACT predictions available for evaluation.")

    output_name = "act_prediction_eval"
    output_dir = workspace / "output" / output_name
    fi_path = workspace / "evaluation_fi.csv"
    manifest_path = workspace / EVALUATION_MANIFEST_NAME
    existing_artifacts = [
        path for path in (output_dir, fi_path, manifest_path) if path.exists()
    ]
    if existing_artifacts and not overwrite:
        raise FileExistsError(
            "ACT evaluation output already exists: "
            + ", ".join(str(path) for path in existing_artifacts)
            + ". Enable explicit overwrite to rerun."
        )
    if overwrite:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        fi_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    act_api = _import_act_api(cfg)
    builder = _import_workspace_builder(cfg)
    active_channels = list(prediction.keys())
    cell = _build_act_cell(cfg, act_api=act_api, builder=builder, active_channels=active_channels)
    cell.set_g_bar(active_channels, [float(prediction[name]) for name in active_channels])

    sim_cfg = cfg["simulation"]
    simulator = act_api["ACTSimulator"](output_folder_name=str(workspace / "output"))
    for sim_idx, amp_pA in enumerate(sim_cfg.get("ci_amps_pA", [])):
        sim_params = act_api["SimulationParameters"](
            sim_name=output_name,
            sim_idx=sim_idx,
            h_v_init=float(sim_cfg.get("h_v_init", -50.0)),
            h_tstop=float(sim_cfg.get("h_tstop", 1000.0)),
            h_dt=float(sim_cfg.get("h_dt", 0.1)),
            h_celsius=float(sim_cfg.get("h_celsius", 37.0)),
            CI=[
                act_api["ConstantCurrentInjection"](
                    amp=float(amp_pA) / 1000.0,
                    dur=float(sim_cfg.get("ci_dur_ms", 700.0)),
                    delay=float(sim_cfg.get("ci_delay_ms", 100.0)),
                )
            ],
        )
        simulator.submit_job(cell, sim_params)

    with _pushd(workspace):
        simulator.run_jobs(n_cpus or cfg.get("optimizer", {}).get("n_cpus"))

    data = np.load(output_dir / "combined_out.npy")
    features = _summary_features_from_dataset(cfg, data)
    fi_rows = [
        {
            "amp_pA": float(amp),
            "spike_frequency_hz": float(freq),
        }
        for amp, freq in zip(sim_cfg.get("ci_amps_pA", []), features["spike_frequency"].to_numpy())
    ]
    _write_rows_csv(fi_path, fi_rows)
    _write_json_atomic(
        manifest_path,
        {
            "version": 1,
            "status": "complete",
            "finished_at": time.time(),
            "config_fingerprint": act_config_fingerprint(cfg),
            "config_file_fingerprint": act_source_fingerprints(cfg)["config"],
            "target_file_fingerprint": act_source_fingerprints(cfg)["target"],
            "predictions_fingerprint": _mapping_fingerprint(prediction),
            "prediction": prediction,
            "fi_path": str(fi_path),
            "output_dir": str(output_dir),
        },
    )
    return {
        "prediction": prediction,
        "fi_rows": fi_rows,
        "fi_path": str(fi_path),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
    }


def resolve_workspace_path(workspace: str | Path, path: str | Path) -> Path:
    """Resolve a path relative to an ACT workspace."""
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return path_obj
    return Path(workspace).expanduser().resolve() / path_obj


def _resolve_workspace_or_tune_path(
    workspace: str | Path,
    cfg: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Resolve a path that may be workspace-relative, tune-relative, or absolute."""
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return path_obj
    workspace_path = Path(workspace).expanduser().resolve() / path_obj
    if workspace_path.exists():
        return workspace_path
    tune_path = Path(cfg.get("tune_dir", Path(workspace).parent)).expanduser().resolve() / path_obj
    if tune_path.exists():
        return tune_path
    return workspace_path


def _prepare_target_file(
    *,
    cfg: dict[str, Any],
    workspace: Path,
    fi_currents_pA: Optional[Sequence[float]],
    fi_frequencies_hz: Optional[Sequence[float]],
    fi_csv_path: Optional[str | Path],
    trace_npy_path: Optional[str | Path],
    nwb_path: Optional[str | Path],
    nwb_stimulus_names: Optional[Sequence[str]],
    nwb_include_negative_currents: bool,
    nwb_min_current_pA: Optional[float],
    nwb_max_current_pA: Optional[float],
    nwb_average_repeats: bool,
    nwb_spike_threshold_mV: float,
    nwb_refractory_ms: float,
) -> Path:
    target = cfg["target"]
    mode = str(target.get("mode", "fi_arrays")).strip().lower()
    if mode == "fi_arrays":
        currents = list(fi_currents_pA or target.get("fi_currents_pA") or [])
        freqs = list(fi_frequencies_hz or target.get("fi_frequencies_hz") or [])
        target["fi_currents_pA"] = [float(v) for v in currents]
        target["fi_frequencies_hz"] = [float(v) for v in freqs]
        cfg.setdefault("simulation", {})["ci_amps_pA"] = target["fi_currents_pA"]
        target_path = workspace / TARGET_SF_NAME
        write_fi_target_csv(target_path, currents_pA=currents, frequencies_hz=freqs)
        return target_path
    if mode == "fi_csv":
        if not fi_csv_path:
            fi_csv_path = target.get("source_csv") or target.get("path")
        source = resolve_workspace_path(workspace, fi_csv_path)
        if not source.is_file():
            raise FileNotFoundError(f"FI CSV target not found: {source}")
        target_path = workspace / TARGET_SF_NAME
        normalize_fi_target_csv(source, target_path)
        target_rows = read_act_fi_target_csv(target_path)
        target["fi_currents_pA"] = [float(row["mean_i"]) * 1000.0 for row in target_rows]
        target["fi_frequencies_hz"] = [float(row["spike_frequency"]) for row in target_rows]
        cfg.setdefault("simulation", {})["ci_amps_pA"] = target["fi_currents_pA"]
        target["source_csv"] = str(_path_relative_to_workspace(source, workspace))
        return target_path
    if mode == "allen_nwb":
        if not nwb_path:
            nwb_path = target.get("source_nwb")
        if not nwb_path:
            raise ValueError(
                "target_mode='allen_nwb' requires nwb_path/ACT_NWB_PATH. "
                "Place the downloaded Allen NWB file in the tune folder and point to it."
            )
        source = _resolve_workspace_or_tune_path(workspace, cfg, nwb_path)
        target_path = workspace / TARGET_SF_NAME
        summary_path = workspace / "allen_nwb_fi_curve.csv"
        stimulus_names = list(
            nwb_stimulus_names
            or target.get("stimulus_names")
            or DEFAULT_FI_STIMULUS_NAMES
        )
        summary = write_allen_nwb_fi_target_csv(
            source,
            target_path,
            summary_path=summary_path,
            stimulus_names=stimulus_names,
            include_negative_currents=bool(nwb_include_negative_currents),
            min_current_pA=nwb_min_current_pA,
            max_current_pA=nwb_max_current_pA,
            average_repeats=bool(nwb_average_repeats),
            spike_threshold_mV=float(nwb_spike_threshold_mV),
            refractory_ms=float(nwb_refractory_ms),
        )
        target.update(
            {
                "source_nwb": str(_path_relative_to_workspace_with_parents(source, workspace)),
                "stimulus_names": stimulus_names,
                "include_negative_currents": bool(nwb_include_negative_currents),
                "min_current_pA": nwb_min_current_pA,
                "max_current_pA": nwb_max_current_pA,
                "average_repeats": bool(nwb_average_repeats),
                "spike_threshold_mV": float(nwb_spike_threshold_mV),
                "refractory_ms": float(nwb_refractory_ms),
                "summary_csv": str(_path_relative_to_workspace(summary_path, workspace)),
                "fi_currents_pA": summary["currents_pA"],
                "fi_frequencies_hz": summary["frequencies_hz"],
            }
        )
        cfg.setdefault("simulation", {})["ci_amps_pA"] = summary["currents_pA"]
        return target_path
    if mode == "trace_npy":
        if not trace_npy_path:
            trace_npy_path = target.get("source_npy") or target.get("path")
        source = resolve_workspace_path(workspace, trace_npy_path)
        if not source.is_file():
            raise FileNotFoundError(f"Trace target .npy not found: {source}")
        target["source_npy"] = str(_path_relative_to_workspace(source, workspace))
        return source
    raise ValueError("target_mode must be 'fi_arrays', 'fi_csv', 'allen_nwb', or 'trace_npy'.")


def write_fi_target_csv(
    path: str | Path,
    *,
    currents_pA: Sequence[float],
    frequencies_hz: Sequence[float],
) -> Path:
    """Write an ACT summary-feature target CSV from FI points."""
    currents = [float(v) for v in currents_pA]
    freqs = [float(v) for v in frequencies_hz]
    if len(currents) != len(freqs):
        raise ValueError("FI current and frequency arrays must have the same length.")
    if not currents:
        raise ValueError("At least one FI target point is required.")
    rows = [
        {"mean_i": current / 1000.0, "spike_frequency": freq}
        for current, freq in zip(currents, freqs)
    ]
    return _write_rows_csv(path, rows)


def normalize_fi_target_csv(source: str | Path, target: str | Path) -> Path:
    """Normalize a user FI CSV into ACT's `mean_i`/`spike_frequency` columns."""
    with Path(source).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"FI CSV has no rows: {source}")

    current_col = _first_present(
        rows[0],
        ["mean_i", "amp_nA", "current_nA", "amp_pA", "current_pA", "I_pA"],
    )
    frequency_col = _first_present(
        rows[0],
        ["spike_frequency", "spike_frequency_hz", "frequency_hz", "freq_hz", "f_hz"],
    )
    out_rows = []
    for row in rows:
        current = float(row[current_col])
        if current_col.lower().endswith("_pa") or current_col.lower() in {"amp_pa", "i_pa"}:
            current = current / 1000.0
        out_rows.append(
            {
                "mean_i": current,
                "spike_frequency": float(row[frequency_col]),
            }
        )
    return _write_rows_csv(target, out_rows)


def read_act_fi_target_csv(source: str | Path) -> list[dict[str, float]]:
    """Read an ACT FI target CSV with `mean_i`/`spike_frequency` columns."""
    with Path(source).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"FI target CSV has no rows: {source}")
    _first_present(rows[0], ["mean_i"])
    _first_present(rows[0], ["spike_frequency"])
    return [
        {
            "mean_i": float(row["mean_i"]),
            "spike_frequency": float(row["spike_frequency"]),
        }
        for row in rows
    ]


def _write_cell_builder(
    *,
    workspace: Path,
    repo_root: Path,
    tune_dir: Path,
    cell_name: str,
) -> Path:
    builder_path = workspace / BUILDER_NAME
    source = f'''"""Generated SCP-to-ACT cell builder.

This file is generated by `modules.tuning.act_active`.
It is intentionally stored inside the tune-local ACT workspace so ACT
multiprocessing can import a top-level `build_cell()` function.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import sys

REPO_ROOT = Path({str(repo_root)!r})
TUNE_DIR = Path({str(tune_dir)!r})
CELL_NAME = {str(cell_name)!r}
FIXED_PREDICTIONS = Path(__file__).resolve().parent / {FIXED_PREDICTIONS_NAME!r}


def _set_nested_value(obj, name, value):
    parts = str(name).split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], float(value))


def _apply_fixed_predictions(cell):
    if not FIXED_PREDICTIONS.is_file():
        return cell
    values = json.loads(FIXED_PREDICTIONS.read_text(encoding="utf-8"))
    soma = getattr(cell, "soma", None)
    if soma is None:
        raise AttributeError("ACT cell builder requires a soma section.")
    sections = list(soma) if not callable(soma) else [soma]
    if not sections:
        raise AttributeError("ACT cell builder found an empty soma section list.")
    for section in sections:
        segment = section(0.5)
        for name, value in values.items():
            _set_nested_value(segment, name, value)
    return cell


def build_cell():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from modules.model.load_cell import load_cell

    cfg_path = TUNE_DIR / "cell_configs" / "cell_config.json"
    cell_config = json.loads(cfg_path.read_text(encoding="utf-8"))
    cell_config.setdefault("cell_name", CELL_NAME)
    cell_config.setdefault("paths", {{}})["tune_dir"] = str(TUNE_DIR)

    old_cwd = Path.cwd()
    os.chdir(str(TUNE_DIR))
    try:
        return _apply_fixed_predictions(load_cell(cell_config, base_dir=TUNE_DIR))
    finally:
        os.chdir(str(old_cwd))
'''
    builder_path.write_text(source, encoding="utf-8")
    return builder_path


def _import_act_api(cfg: Mapping[str, Any]) -> dict[str, Any]:
    ensure_act_on_syspath(repo_root=Path(cfg["repo_root"]))
    from act.cell_model import ACTCellModel
    from act.data_processing import get_summary_features
    from act.module import ACTModule
    from act.simulator import ACTSimulator
    from act.types import (
        ConductanceOptions,
        ConstantCurrentInjection,
        FilterParameters,
        OptimizationParameters,
        SimulationParameters,
    )

    return {
        "ACTCellModel": ACTCellModel,
        "ACTModule": ACTModule,
        "ACTSimulator": ACTSimulator,
        "ConductanceOptions": ConductanceOptions,
        "ConstantCurrentInjection": ConstantCurrentInjection,
        "FilterParameters": FilterParameters,
        "OptimizationParameters": OptimizationParameters,
        "SimulationParameters": SimulationParameters,
        "get_summary_features": get_summary_features,
    }


def _import_workspace_builder(cfg: Mapping[str, Any]):
    workspace = Path(cfg["workspace"]).resolve()
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    if "cell_builder" in sys.modules:
        loaded = getattr(sys.modules["cell_builder"], "__file__", "")
        if Path(loaded).resolve() != (workspace / BUILDER_NAME).resolve():
            del sys.modules["cell_builder"]
    module = importlib.import_module("cell_builder")
    return getattr(module, cfg.get("cell_builder", {}).get("function", "build_cell"))


def _build_act_cell(
    cfg: Mapping[str, Any],
    *,
    act_api: Mapping[str, Any],
    builder: Any,
    module_spec: Optional[Mapping[str, Any]] = None,
    active_channels: Optional[Sequence[str]] = None,
) -> Any:
    from modules.setup.mechanisms import resolve_modfiles_dir

    if active_channels is None:
        if module_spec is not None:
            active_channels = [
                str(item["variable_name"])
                for item in module_spec.get("conductances", [])
            ]
        else:
            active_channels = cfg.get("act_cell", {}).get("active_channels", [])
    modfiles_dir = resolve_modfiles_dir(Path(cfg["tune_dir"]).resolve())
    cell = act_api["ACTCellModel"](
        cell_name=None,
        path_to_hoc_file=None,
        path_to_mod_files=None if modfiles_dir is None else str(modfiles_dir),
        passive=list(cfg.get("act_cell", {}).get("passive", DEFAULT_PASSIVE_NAMES)),
        active_channels=list(active_channels),
    )
    cell.set_custom_cell_builder(builder)
    return cell


def _build_simulation_parameters(cfg: Mapping[str, Any], act_api: Mapping[str, Any]) -> Any:
    sim = cfg["simulation"]
    return act_api["SimulationParameters"](
        sim_name="cell",
        sim_idx=0,
        h_v_init=float(sim.get("h_v_init", -50.0)),
        h_tstop=float(sim.get("h_tstop", 1000.0)),
        h_dt=float(sim.get("h_dt", 0.1)),
        h_celsius=float(sim.get("h_celsius", 37.0)),
    )


def _build_optimization_parameters(
    cfg: Mapping[str, Any],
    act_api: Mapping[str, Any],
    module_spec: Mapping[str, Any],
    *,
    n_cpus: Optional[int],
) -> Any:
    opt = dict(cfg.get("optimizer", {}) or {})
    sim = cfg["simulation"]
    filter_cfg = cfg.get("filter", {}) or {}
    filter_parameters = None
    if filter_cfg:
        filter_parameters = act_api["FilterParameters"](
            filtered_out_features=filter_cfg.get("filtered_out_features"),
            window_of_inspection=_tuple_or_none(filter_cfg.get("window_of_inspection")),
            saturation_threshold=float(filter_cfg.get("saturation_threshold", -55.0)),
        )

    return act_api["OptimizationParameters"](
        n_cpus=n_cpus if n_cpus is not None else opt.get("n_cpus"),
        conductance_options=[
            _conductance_option(act_api, item)
            for item in module_spec.get("conductances", [])
        ],
        CI_options=[
            act_api["ConstantCurrentInjection"](
                amp=float(amp_pA) / 1000.0,
                dur=float(sim.get("ci_dur_ms", 700.0)),
                delay=float(sim.get("ci_delay_ms", 100.0)),
            )
            for amp_pA in sim.get("ci_amps_pA", [])
        ],
        random_state=int(opt.get("random_state", 42)),
        n_estimators=int(opt.get("n_estimators", 1000)),
        max_depth=opt.get("max_depth"),
        train_features=opt.get("train_features"),
        filter_parameters=filter_parameters,
        spike_threshold=float(opt.get("spike_threshold", -20.0)),
        max_n_spikes=int(opt.get("max_n_spikes", 20)),
    )


def _conductance_option(act_api: Mapping[str, Any], item: Mapping[str, Any]) -> Any:
    return act_api["ConductanceOptions"](
        variable_name=str(item["variable_name"]),
        blocked=bool(item.get("blocked", False)),
        low=_float_or_none(item.get("low")),
        high=_float_or_none(item.get("high")),
        bounds_variation=_float_or_none(item.get("bounds_variation")),
        n_slices=int(item.get("n_slices", 1)),
    )


def _summary_features_from_dataset(cfg: Mapping[str, Any], data: np.ndarray) -> Any:
    act_api = _import_act_api(cfg)
    opt = cfg.get("optimizer", {}) or {}
    return act_api["get_summary_features"](
        V=data[:, :, 0],
        I=data[:, :, 1],
        spike_threshold=float(opt.get("spike_threshold", -20.0)),
        max_n_spikes=int(opt.get("max_n_spikes", 20)),
    )


def _target_file_for_act(cfg: Mapping[str, Any]) -> Path:
    workspace = Path(cfg["workspace"]).resolve()
    return resolve_workspace_path(workspace, cfg["target"]["path"])


def _get_module_spec(cfg: Mapping[str, Any], module_key: str) -> Mapping[str, Any]:
    modules = cfg.get("modules", {}) or {}
    if module_key not in modules:
        raise KeyError(f"Unknown ACT module {module_key!r}; options: {', '.join(modules)}")
    return modules[module_key]


def _load_prior_predictions(
    cfg: Mapping[str, Any],
    *,
    before_module: Optional[str] = None,
) -> dict[str, float]:
    workspace = Path(cfg["workspace"]).resolve()
    merged: dict[str, float] = {}
    for key in (cfg.get("modules", {}) or {}):
        if before_module is not None and key == before_module:
            break
        path = workspace / f"{PREDICTION_PREFIX}{key}.json"
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            merged.update({str(k): float(v) for k, v in raw.items()})
    return merged


def _coerce_act_config(
    config_or_mapping: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(config_or_mapping, Mapping):
        cfg = json.loads(json.dumps(config_or_mapping))
        workspace = Path(cfg.get("workspace") or ".").expanduser().resolve()
        cfg["workspace"] = str(workspace)
        return cfg
    return load_act_active_config(config_or_mapping)


def _target_point_count(cfg: Mapping[str, Any]) -> int:
    target = cfg.get("target", {}) or {}
    currents = target.get("fi_currents_pA") or cfg.get("simulation", {}).get(
        "ci_amps_pA", []
    )
    if currents:
        return len(currents)
    try:
        path = _target_file_for_act(cfg)
        if str(target.get("mode", "")).lower() == "trace_npy" and path.is_file():
            data = np.load(path, mmap_mode="r")
            return int(data.shape[0]) if data.ndim else 0
        if path.is_file():
            return len(read_act_fi_target_csv(path))
    except Exception:
        return 0
    return 0


def _mapping_fingerprint(values: Mapping[str, Any]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_fingerprint(path: Path) -> str:
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_rows_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path_obj.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path_obj


def _write_json(path: str | Path, data: Mapping[str, Any]) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path_obj


def _write_json_atomic(path: str | Path, data: Mapping[str, Any]) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    temporary = path_obj.with_name(path_obj.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path_obj)
    return path_obj


def _first_present(row: Mapping[str, Any], names: Sequence[str]) -> str:
    lower_to_actual = {str(key).lower(): str(key) for key in row}
    for name in names:
        if name.lower() in lower_to_actual:
            return lower_to_actual[name.lower()]
    raise KeyError(f"Expected one of {names}; found {list(row)}")


def _tuple_or_none(value: Any) -> Optional[tuple[Any, ...]]:
    if value in (None, "", False):
        return None
    return tuple(value)


def _float_or_none(value: Any) -> Optional[float]:
    if value is None or value == "" or isinstance(value, bool):
        return None
    return float(value)


def _path_relative_to_workspace(path: Path, workspace: Path) -> Path:
    try:
        return path.resolve().relative_to(workspace.resolve())
    except ValueError:
        return path.resolve()


def _path_relative_to_workspace_with_parents(path: Path, workspace: Path) -> Path:
    try:
        return Path(os.path.relpath(path.resolve(), workspace.resolve()))
    except ValueError:
        return path.resolve()


def resolve_act_workspace_from_labels(
    *,
    repo_root: str | Path,
    cell_name: str,
    tune_name: str,
    tunes_parent: str = "tunes",
) -> Path:
    """Resolve the default ACT workspace from cell/tune labels."""
    tune_dir = resolve_tune_dir(
        repo_root=Path(repo_root),
        cell_name=cell_name,
        tune_name=tune_name,
        tunes_parent=tunes_parent,
    )
    return default_act_workspace(tune_dir)
