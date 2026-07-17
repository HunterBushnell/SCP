"""Target-config helpers for Step 2/3 tuning notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional


TARGET_CONFIG_FILENAME = "target_config.json"
VALID_TARGET_SOURCE_MODES = {"none", "manual", "traces", "allen_nwb"}

DEFAULT_MANUAL_BLOCK = {
    "passive": {
        "v_rest_mV": None,
        "rin_MOhm": None,
        "tau_ms": None,
    },
    "fi_curve": {
        "currents_pA": [],
        "rates_Hz": [],
        "csv": None,
    },
}

DEFAULT_TRACES_BLOCK = {
    "format": "csv",
    "passive": {
        "file": None,
        "time_column": "time_ms",
        "voltage_column": "voltage_mV",
        "current_column": "current_pA",
        "sweep_column": None,
        "stim_start_ms": None,
        "stim_stop_ms": None,
        "current_pA": None,
        "dt_ms": None,
        "end_margin_ms": 10.0,
        "reducer": "median",
        "tau_field": "tau_avg_ms",
    },
    "active": {
        "file": None,
        "format": "npy",
        "stim_start_ms": None,
        "stim_stop_ms": None,
        "dt_ms": None,
        "spike_threshold_mV": -20.0,
        "refractory_ms": 1.0,
    },
}

DEFAULT_ALLEN_NWB_BLOCK = {
    "file": None,
    "sweep_ids": [],
    "passive": {
        "stimulus_names": ["Long Square"],
        "sweep_ids": None,
        "min_current_pA": None,
        "max_current_pA": -1.0,
        "end_margin_ms": 10.0,
        "reducer": "median",
        "tau_field": "tau_avg_ms",
    },
    "active": {
        "stimulus_names": ["Long Square"],
        "min_current_pA": 0.0,
        "max_current_pA": None,
        "include_negative_currents": False,
        "average_repeats": True,
        "spike_threshold_mV": -20.0,
        "refractory_ms": 1.0,
    },
}


def target_config_path(tune_dir: str | Path) -> Path:
    """Return the standard target-config path for a tune directory."""
    return Path(tune_dir).expanduser().resolve() / "cell_configs" / TARGET_CONFIG_FILENAME


def load_target_config(tune_dir: str | Path, *, required: bool = False) -> dict[str, Any]:
    """Load optional biological/experimental tuning targets for a tune."""
    path = target_config_path(tune_dir)
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Missing target config: {path}")
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


def write_target_config(tune_dir: str | Path, config: Mapping[str, Any]) -> Path:
    """Write a target config to the standard tune-local path."""
    path = target_config_path(tune_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(config), handle, indent=2)
        handle.write("\n")
    return path


def update_passive_targets_in_config(
    tune_dir: str | Path,
    passive_targets: Mapping[str, Any],
    *,
    target_source: Optional[str] = None,
    notes: Optional[str] = None,
) -> dict[str, Any]:
    """Update only the passive target block in `target_config.json`."""
    config = ensure_target_config_shape(load_target_config(tune_dir))
    config.setdefault("schema_version", 1)
    if target_source is not None:
        source = config.setdefault("target_source", {"mode": "manual", "description": ""})
        if isinstance(source, dict):
            source["mode"] = str(target_source)
        else:
            config["target_source"] = {"mode": str(target_source), "description": ""}
    config.setdefault("manual", _copy_default(DEFAULT_MANUAL_BLOCK))
    config["manual"]["passive"] = {
        "v_rest_mV": _optional_float(passive_targets.get("v_rest_mV")),
        "rin_MOhm": _optional_float(passive_targets.get("rin_MOhm")),
        "tau_ms": _optional_float(passive_targets.get("tau_ms")),
    }
    if notes is not None:
        config["notes"] = str(notes)
    else:
        config.setdefault("notes", "")
    write_target_config(tune_dir, config)
    return config


def passive_targets_from_config(config: Mapping[str, Any]) -> dict[str, Optional[float]]:
    """Return Step 2 passive targets from `target_config.json` naming."""
    manual = manual_block(config)
    passive = manual.get("passive")
    if not isinstance(passive, Mapping):
        passive = {}
    return {
        "target_v_rest_mv": _optional_float(passive.get("v_rest_mV")),
        "target_rin_mohm": _optional_float(passive.get("rin_MOhm")),
        "target_tau_ms": _optional_float(passive.get("tau_ms")),
    }


def fi_curve_from_config(config: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    """Return FI target arrays from `target_config.json`, or empty lists."""
    curve = manual_fi_curve_block(config)
    if not isinstance(curve, Mapping):
        return [], []

    currents = [_as_float(value) for value in (curve.get("currents_pA") or [])]
    rates = [_as_float(value) for value in (curve.get("rates_Hz") or [])]
    if len(currents) != len(rates):
        raise ValueError("target_config manual.fi_curve currents_pA and rates_Hz lengths differ")
    return currents, rates


def fi_reference_points_from_config(config: Mapping[str, Any]) -> list[tuple[float, float]]:
    """Return FI target points for plotting overlays."""
    currents, rates = fi_curve_from_config(config)
    return list(zip(currents, rates))


def nwb_file_from_config(config: Mapping[str, Any], tune_dir: str | Path) -> Optional[Path]:
    """Return tune-local or absolute Allen/ADB NWB target path."""
    return resolve_tune_path(allen_nwb_block(config).get("file"), tune_dir)


def target_source_mode_from_config(config: Mapping[str, Any], *, default: str = "manual") -> str:
    """Return `target_source.mode` as one of the supported source modes."""
    source = config.get("target_source") if isinstance(config, Mapping) else {}
    if isinstance(source, Mapping):
        raw = source.get("mode", default)
    else:
        raw = default
    mode = str(raw or default).strip().lower()
    if mode not in VALID_TARGET_SOURCE_MODES:
        raise ValueError(
            "target_config target_source.mode must be one of "
            + ", ".join(sorted(VALID_TARGET_SOURCE_MODES))
        )
    return mode


def manual_block(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the manual target block."""
    block = config.get("manual") if isinstance(config, Mapping) else {}
    return dict(block) if isinstance(block, Mapping) else {}


def manual_fi_curve_block(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the manual FI-curve block."""
    curve = manual_block(config).get("fi_curve")
    return dict(curve) if isinstance(curve, Mapping) else {}


def manual_fi_csv_from_config(config: Mapping[str, Any], tune_dir: str | Path) -> Optional[Path]:
    """Return tune-local or absolute manual FI CSV path."""
    return resolve_tune_path(manual_fi_curve_block(config).get("csv"), tune_dir)


def traces_block(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the generic trace target block."""
    block = config.get("traces") if isinstance(config, Mapping) else {}
    return dict(block) if isinstance(block, Mapping) else {}


def allen_nwb_block(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the Allen/NWB target block."""
    block = config.get("allen_nwb") if isinstance(config, Mapping) else {}
    return dict(block) if isinstance(block, Mapping) else {}


def resolve_tune_path(raw: Any, tune_dir: str | Path) -> Optional[Path]:
    """Resolve a tune-local or absolute path value."""
    if raw in (None, ""):
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = Path(tune_dir).expanduser().resolve() / path
    return path.resolve()


def ensure_target_config_shape(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return config with all v1 target source blocks present."""
    data = dict(config or {})
    data.setdefault("schema_version", 1)
    source = data.get("target_source")
    if not isinstance(source, Mapping):
        data["target_source"] = {"mode": "manual", "description": str(source or "")}
    else:
        data["target_source"] = {
            "mode": str(source.get("mode") or "manual"),
            "description": str(source.get("description") or ""),
        }
    _merge_missing(data, "manual", DEFAULT_MANUAL_BLOCK)
    _merge_missing(data, "traces", DEFAULT_TRACES_BLOCK)
    _merge_missing(data, "allen_nwb", DEFAULT_ALLEN_NWB_BLOCK)
    data.setdefault("notes", "")
    return data


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return _as_float(value)


def _as_float(value: Any) -> float:
    return float(value)


def _copy_default(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _merge_missing(data: dict[str, Any], key: str, default: Mapping[str, Any]) -> None:
    block = data.get(key)
    if not isinstance(block, dict):
        data[key] = _copy_default(default)
        return
    _deep_merge_missing(block, default)


def _deep_merge_missing(data: dict[str, Any], default: Mapping[str, Any]) -> None:
    for key, value in default.items():
        if key not in data:
            data[key] = _copy_default(value)
        elif isinstance(data[key], dict) and isinstance(value, Mapping):
            _deep_merge_missing(data[key], value)
