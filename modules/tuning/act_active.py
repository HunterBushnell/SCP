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
import importlib
import json
import os
import shutil
import sys

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


def default_act_module_specs(cell_name: str = "SST") -> dict[str, dict[str, Any]]:
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
    workspace_path = Path(workspace).expanduser().resolve() if workspace else default_act_workspace(tune_path)
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
    target_mode: str = "fi_arrays",
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
) -> dict[str, Any]:
    """Create/update an ACT workspace and write its config, builder, and target."""
    root = resolve_repo_root(Path(repo_root))
    tune_path = Path(tune_dir).expanduser().resolve()
    cfg = default_act_active_config(
        repo_root=root,
        tune_dir=tune_path,
        cell_name=cell_name,
        tune_name=tune_name,
        workspace=workspace,
    )
    workspace_path = Path(cfg["workspace"])
    workspace_path.mkdir(parents=True, exist_ok=True)

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

    cfg["target"]["mode"] = str(target_mode)
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
    return {
        "workspace": str(workspace),
        "config": str(workspace / CONFIG_NAME),
        "builder": str(workspace / BUILDER_NAME),
        "target": str(target_path),
        "target_mode": cfg.get("target", {}).get("mode"),
        "modules": [key for key, spec in modules.items() if spec.get("enabled", True)],
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


def run_act_active_module(
    config_or_workspace: str | Path,
    module_key: str,
    *,
    n_cpus: Optional[int] = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run one configured ACT active module and save JSON predictions."""
    cfg = load_act_active_config(config_or_workspace)
    workspace = Path(cfg["workspace"]).resolve()
    module_spec = _get_module_spec(cfg, module_key)
    if not module_spec.get("enabled", True):
        raise ValueError(f"ACT module {module_key!r} is disabled.")

    module_name = module_spec.get("name") or str(module_key)
    output_dir = workspace / f"module_{module_name}"
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"ACT output already exists for module {module_key!r}: {output_dir}. "
                "Use overwrite=True to rerun."
            )
        shutil.rmtree(output_dir)
    prediction_path = workspace / f"{PREDICTION_PREFIX}{module_key}.json"
    metrics_path = workspace / f"metrics_{module_key}.csv"
    if overwrite:
        prediction_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)

    act_api = _import_act_api(cfg)
    builder = _import_workspace_builder(cfg)
    train_cell = _build_act_cell(cfg, act_api=act_api, builder=builder, module_spec=module_spec)
    train_cell.prediction.update(_load_prior_predictions(cfg, before_module=module_key))

    simulation_parameters = _build_simulation_parameters(cfg, act_api)
    optimization_parameters = _build_optimization_parameters(cfg, act_api, module_spec, n_cpus=n_cpus)
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

    prediction = {str(k): float(v) for k, v in act_module.cell.prediction.items()}
    _write_json(prediction_path, prediction)
    metrics.to_csv(metrics_path, index=False)
    return {
        "module": module_key,
        "module_name": module_name,
        "prediction": prediction,
        "prediction_path": str(prediction_path),
        "metrics_path": str(metrics_path),
        "output_dir": str(output_dir),
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
    prediction = dict(predictions or collect_act_predictions(workspace))
    if not prediction:
        raise ValueError("No ACT predictions available for evaluation.")

    output_name = "act_prediction_eval"
    output_dir = workspace / "output" / output_name
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Evaluation output already exists: {output_dir}. Use overwrite=True to rerun."
            )
        shutil.rmtree(output_dir)

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
    fi_path = workspace / "evaluation_fi.csv"
    _write_rows_csv(fi_path, fi_rows)
    return {
        "prediction": prediction,
        "fi_rows": fi_rows,
        "fi_path": str(fi_path),
        "output_dir": str(output_dir),
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
        stimulus_names = list(nwb_stimulus_names or target.get("stimulus_names") or DEFAULT_FI_STIMULUS_NAMES)
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


def build_cell():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from modules.notebooks.helpers import build_cell_for_notebook

    cfg_path = TUNE_DIR / "cell_configs" / "cell_config.json"
    cell_config = json.loads(cfg_path.read_text(encoding="utf-8"))
    cell_config.setdefault("cell_name", CELL_NAME)
    paths = cell_config.setdefault("paths", {{}})
    manifest = Path(paths.get("manifest", "manifest.json"))
    if not manifest.is_absolute():
        manifest = TUNE_DIR / manifest
    paths["manifest"] = str(manifest)
    paths["tune_dir"] = str(TUNE_DIR)

    old_cwd = Path.cwd()
    os.chdir(str(TUNE_DIR))
    try:
        loaded = build_cell_for_notebook(cell_config)
        return getattr(loaded, "h", loaded)
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
    if active_channels is None:
        if module_spec is not None:
            active_channels = [
                str(item["variable_name"])
                for item in module_spec.get("conductances", [])
            ]
        else:
            active_channels = cfg.get("act_cell", {}).get("active_channels", [])
    cell = act_api["ACTCellModel"](
        cell_name=None,
        path_to_hoc_file=None,
        path_to_mod_files=str(Path(cfg["tune_dir"]).resolve() / "modfiles"),
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
    if value in (None, "", False):
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
