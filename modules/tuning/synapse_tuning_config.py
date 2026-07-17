"""Tune-local BMTool synapse tuning config helpers for Step 4."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional


SYNAPSE_TUNING_CONFIG_FILENAME = "synapse_tuning_config.json"
DEFAULT_VCLAMP_MV = -65.0
DEFAULT_CELSIUS_C = 20.0


def default_synapse_tuning_config(
    *,
    vclamp_amp: float = DEFAULT_VCLAMP_MV,
    celsius: float = DEFAULT_CELSIUS_C,
) -> dict[str, Any]:
    """Return the loader-neutral Step 4 BMTool-style tuning template.

    The point-process names and parameters are editable starting points, not
    model-specific selections. Tune-local generation replaces the fallback
    voltage and temperature with values from ``sim_config.conditions`` when
    they are available.
    """
    holding_mV = float(vclamp_amp)
    temperature_C = float(celsius)
    return {
        "default_connection": "excitatory_facilitating",
        "current_name": "i",
        "slider_vars": None,
        "other_vars_to_record": None,
        "general_settings": {
            "vclamp": True,
            "rise_interval": [0.1, 0.9],
            "tstart": 500.0,
            "tdur": 100.0,
            "threshold": -15.0,
            "delay": 1.3,
            "weight": 1.0,
            "dt": 0.025,
            "celsius": temperature_C,
        },
        "connections": {
            "excitatory_facilitating": {
                "description": "Editable starting point for a facilitating excitatory synapse.",
                "spec_settings": {
                    "post_cell": "SCP_Cell",
                    "vclamp_amp": holding_mV,
                    "sec_x": 0.5,
                    "sec_id": 0,
                    "level_of_detail": "AMPA_NMDA_STP",
                },
                "spec_syn_param": {
                    "initW": 0.25,
                    "tau_r_AMPA": 0.2,
                    "tau_d_AMPA": 1.7,
                    "Use": 0.75,
                    "Dep": 0.0,
                    "Fac": 200.0,
                    "NMDA_ratio": 1.5,
                },
            },
            "excitatory_depressing": {
                "description": "Editable starting point for a depressing excitatory synapse.",
                "spec_settings": {
                    "post_cell": "SCP_Cell",
                    "vclamp_amp": holding_mV,
                    "sec_x": 0.5,
                    "sec_id": 0,
                    "level_of_detail": "AMPA_NMDA_STP",
                },
                "spec_syn_param": {
                    "initW": 1.5,
                    "tau_r_AMPA": 3.5,
                    "tau_d_AMPA": 4.0,
                    "Use": 0.80,
                    "Dep": 100.0,
                    "Fac": 0.0,
                    "NMDA_ratio": 0.0,
                },
            },
            "inhibitory_static": {
                "description": "Editable starting point for a static inhibitory synapse.",
                "spec_settings": {
                    "post_cell": "SCP_Cell",
                    "vclamp_amp": holding_mV,
                    "sec_x": 0.5,
                    "sec_id": 0,
                    "level_of_detail": "GABA_A",
                },
                "spec_syn_param": {
                    "initW": 0.1,
                    "tau_r_GABAA": 0.5,
                    "tau_d_GABAA": 5.5,
                    "e_GABAA": -75.0,
                    "gmax": 0.001,
                },
            },
            "inhibitory_stp": {
                "description": "Editable starting point for an inhibitory synapse with STP.",
                "spec_settings": {
                    "post_cell": "SCP_Cell",
                    "vclamp_amp": holding_mV,
                    "sec_x": 0.5,
                    "sec_id": 0,
                    "level_of_detail": "GABA_A_STP",
                },
                "spec_syn_param": {
                    "initW": 5.0,
                    "tau_r_GABAA": 0.5,
                    "tau_d_GABAA": 5.5,
                    "e_GABAA": -75.0,
                    "gmax": 0.001,
                    "Use": 1.0,
                    "Dep": 250.0,
                    "Fac": 0.0,
                },
            },
        },
        "optimizer": {
            "enabled": False,
            "param_bounds": {
                "Dep": [0.0, 200.0],
                "Fac": [0.0, 400.0],
                "Use": [0.1, 1.0],
                "tau_r_AMPA": [0.1, 4.0],
                "tau_d_AMPA": [1.0, 20.0],
            },
            "target_metrics": {
                "induction": -0.75,
                "ppr": 0.8,
                "recovery": 0.0,
                "max_amplitude": 25.0,
                "rise_time": 2.0,
                "decay_time": 9.0,
            },
            "cost_weights": {
                "induction": 1.0,
                "ppr": 3.0,
                "recovery": 1.0,
                "rise_time": 1.0,
                "decay_time": 1.0,
            },
            "run_single_event": True,
            "run_train_input": True,
            "train_frequency": 50,
            "train_delay": 250,
            "init_guess": "random",
            "method": "SLSQP",
        },
    }


def synapse_tuning_config_path(tune_dir: str | Path) -> Path:
    """Return the tune-local Step 4 config path."""
    return Path(tune_dir).expanduser().resolve() / "cell_configs" / SYNAPSE_TUNING_CONFIG_FILENAME


def ensure_synapse_tuning_config(
    tune_dir: str | Path,
    *,
    overwrite: bool = False,
    defaults: Optional[Mapping[str, Any]] = None,
) -> tuple[Path, dict[str, Any], str]:
    """Create or load the tune-local Step 4 config."""
    path = synapse_tuning_config_path(tune_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return path, load_synapse_tuning_config(tune_dir), "existing"

    if defaults is None:
        vclamp_amp, celsius = _runtime_defaults_from_tune(tune_dir)
        config = default_synapse_tuning_config(
            vclamp_amp=vclamp_amp,
            celsius=celsius,
        )
    else:
        config = deepcopy(dict(defaults))
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path, config, "created" if not overwrite else "overwritten"


def load_synapse_tuning_config(tune_dir: str | Path) -> dict[str, Any]:
    """Load and normalize the tune-local Step 4 config."""
    path = synapse_tuning_config_path(tune_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Missing Step 4 synapse tuning config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return normalize_synapse_tuning_config(raw)


def normalize_synapse_tuning_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with required top-level keys and BMTool-compatible names."""
    data = deepcopy(dict(config))
    defaults = default_synapse_tuning_config()
    data.setdefault("current_name", defaults["current_name"])
    data.setdefault("slider_vars", defaults["slider_vars"])
    data.setdefault("other_vars_to_record", defaults["other_vars_to_record"])
    data.setdefault("general_settings", deepcopy(defaults["general_settings"]))
    data.setdefault("connections", deepcopy(defaults["connections"]))
    data.setdefault("optimizer", deepcopy(defaults["optimizer"]))
    if not isinstance(data["general_settings"], dict):
        raise ValueError("synapse_tuning_config general_settings must be an object")
    if not isinstance(data["connections"], dict) or not data["connections"]:
        raise ValueError("synapse_tuning_config connections must be a non-empty object")
    data.setdefault("default_connection", next(iter(data["connections"])))
    return data


def _runtime_defaults_from_tune(tune_dir: str | Path) -> tuple[float, float]:
    """Read Step 4 holding voltage and temperature from tune conditions."""
    sim_config_path = (
        Path(tune_dir).expanduser().resolve() / "cell_configs" / "sim_config.json"
    )
    if not sim_config_path.is_file():
        return DEFAULT_VCLAMP_MV, DEFAULT_CELSIUS_C

    with sim_config_path.open("r", encoding="utf-8") as handle:
        sim_config = json.load(handle)
    if not isinstance(sim_config, dict):
        raise ValueError(f"Expected JSON object in {sim_config_path}")
    conditions = sim_config.get("conditions")
    if conditions is None:
        return DEFAULT_VCLAMP_MV, DEFAULT_CELSIUS_C
    if not isinstance(conditions, dict):
        raise ValueError(f"conditions must be an object in {sim_config_path}")

    raw_vclamp = conditions.get("v_init_mV", DEFAULT_VCLAMP_MV)
    raw_celsius = conditions.get("celsius_C", DEFAULT_CELSIUS_C)
    return (
        _numeric_condition(
            DEFAULT_VCLAMP_MV if raw_vclamp is None else raw_vclamp,
            name="conditions.v_init_mV",
            source=sim_config_path,
        ),
        _numeric_condition(
            DEFAULT_CELSIUS_C if raw_celsius is None else raw_celsius,
            name="conditions.celsius_C",
            source=sim_config_path,
        ),
    )


def _numeric_condition(value: Any, *, name: str, source: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric in {source}; got {value!r}")
    return float(value)


def connection_settings_for_bmtool(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return BMTool's expected `conn_type_settings` dictionary."""
    normalized = normalize_synapse_tuning_config(config)
    return {
        name: {
            "spec_settings": deepcopy(settings.get("spec_settings", {})),
            "spec_syn_param": deepcopy(settings.get("spec_syn_param", {})),
        }
        for name, settings in normalized["connections"].items()
    }


def general_settings_for_bmtool(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return BMTool's `general_settings` with JSON lists converted as needed."""
    normalized = normalize_synapse_tuning_config(config)
    settings = deepcopy(normalized["general_settings"])
    rise_interval = settings.get("rise_interval")
    if isinstance(rise_interval, list):
        settings["rise_interval"] = tuple(rise_interval)
    return settings


def selected_connection(config: Mapping[str, Any], override: Optional[str] = None) -> str:
    """Return the selected Step 4 connection key."""
    normalized = normalize_synapse_tuning_config(config)
    connection = override or normalized.get("default_connection")
    if connection not in normalized["connections"]:
        raise KeyError(
            f"connection={connection!r} not found. Available: {sorted(normalized['connections'])}"
        )
    return str(connection)


def connection_option(config: Mapping[str, Any], connection: str, key: str) -> Any:
    """Return a per-connection option with a top-level fallback."""
    normalized = normalize_synapse_tuning_config(config)
    chosen = selected_connection(normalized, connection)
    connection_block = normalized["connections"][chosen]
    if key in connection_block:
        return deepcopy(connection_block[key])
    return deepcopy(normalized.get(key))


def optimizer_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return normalized optimizer defaults from the Step 4 config."""
    return deepcopy(normalize_synapse_tuning_config(config).get("optimizer", {}))
