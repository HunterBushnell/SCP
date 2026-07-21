"""Shared settings, parsing, and component helpers for the compact UI."""

from __future__ import annotations

import math
import os
from collections.abc import MutableMapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Optional


PIPELINE_UI_DEFAULTS: dict[str, Any] = {
    "cell_name": "PV",
    "tune_name": "tuned",
    "tune_dir_override": None,
    "quiet_step1_output": True,
    "recompile_modfiles": False,
    "passive_amps_pA": [-50.0, -100.0],
    "passive_target_overrides": {},
    "passive_protocol_overrides": {},
    "active_amps_pA": [150.0, 300.0],
    "fi_amps_pA": [float(value) for value in range(0, 301, 50)],
    "active_protocol_overrides": {},
    "fi_protocol_overrides": None,
    "active_spike_threshold_mV": -20.0,
    "fi_spike_threshold_mV": -20.0,
    "active_include_currents": True,
    "active_current_display_amp_pA": None,
    "act_active_module": None,
    "act_n_cpus": None,
    "act_workspace_override": None,
    "act_overrides": {},
    "act_overwrite_outputs": False,
    "enable_synapse_tuning": False,
    "synapse_connection": None,
    "n_trials": 1,
    "seed": None,
    "run_iclamp": False,
    "output_stem": None,
    "quiet_input_preview_output": True,
    "quiet_simulation_output": True,
    "simulation_overrides": {},
    "input_preview_groups": None,
    "input_preview_plots": [
        "weight_distribution",
        "distance_distribution",
        "weight_vs_distance",
    ],
    "input_preview_trial_idx": 0,
    "input_preview_show_table": True,
    "input_preview_histogram_density": True,
    "input_preview_distance_bin_um": 25.0,
    "input_preview_weight_bin": None,
    "input_preview_plot_columns": 3,
    "input_preview_plot_size": "compact",
    "diagnostic_plots": [
        "input_rate",
        "membrane_voltage",
        "output_rate",
        "output_raster",
    ],
    "diagnostic_trial_idx": 0,
    "diagnostic_window_mode": "stimulus",
    "diagnostic_window_start_ms": None,
    "diagnostic_window_stop_ms": None,
    "diagnostic_window_padding_ms": 100.0,
    "diagnostic_rate_bin_ms": None,
    "diagnostic_smoothing_ms": None,
    "diagnostic_raster_style": "dot",
    "diagnostic_input_groups": None,
    "diagnostic_show_stimulus": True,
    "diagnostic_figure_size": "compact",
}

MODEL_SETTING_KEYS = (
    "cell_name",
    "tune_name",
    "tune_dir_override",
    "recompile_modfiles",
)


def import_widgets() -> Any:
    try:
        import ipywidgets as widgets
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Pipeline widgets require ipywidgets. Run the notebook environment "
            "cell or install the SCP environment before creating PipelineNotebookUI."
        ) from exc
    return widgets


def clear_output() -> None:
    try:
        from IPython.display import clear_output as ipython_clear_output

        ipython_clear_output(wait=True)
    except Exception:
        return


def parse_float_list(value: Any, *, name: str) -> list[float]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        raw_values: Sequence[Any] = [part.strip() for part in text.split(",")]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_values = list(value)
    else:
        raise ValueError(f"{name} must be a comma-separated list of numbers.")

    values: list[float] = []
    for raw in raw_values:
        if raw in (None, ""):
            continue
        if isinstance(raw, bool):
            raise ValueError(f"{name} must contain numbers, not booleans.")
        try:
            values.append(float(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} contains an invalid number: {raw!r}. "
                "Use comma-separated values such as -50, -100."
            ) from exc
    if not values:
        raise ValueError(f"{name} must contain at least one number.")
    return values


def format_float_list(value: Any) -> str:
    try:
        values = parse_float_list(value, name="amplitudes")
    except ValueError:
        return str(value or "")
    return ", ".join(f"{number:g}" for number in values)


def parse_optional_int(
    value: Any,
    *,
    name: str,
    positive: bool = False,
) -> Optional[int]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer or blank.")
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer or blank; got {value!r}.") from exc
    if positive and parsed <= 0:
        raise ValueError(f"{name} must be greater than zero when provided.")
    return parsed


def parse_optional_float(value: Any, *, name: str) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number or blank.")
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number or blank; got {value!r}.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite; got {value!r}.")
    return parsed


def optional_text(value: Any) -> str:
    return "" if value in (None, "") else str(value)


def merge_non_none(
    destination: MutableMapping[str, Any], source: Any
) -> MutableMapping[str, Any]:
    for key, value in dict(source or {}).items():
        if value is None:
            continue
        if isinstance(value, MutableMapping):
            nested = destination.get(key)
            if not isinstance(nested, MutableMapping):
                nested = {}
            destination[key] = merge_non_none(dict(nested), value)
        else:
            destination[key] = deepcopy(value)
    return destination


def nested_value(mapping: Any, path: Sequence[str], default: Any = None) -> Any:
    value = mapping
    for key in path:
        if not isinstance(value, MutableMapping):
            return default
        if key not in value:
            return default
        value = value[key]
    return value


def set_nested_value(
    mapping: MutableMapping[str, Any], path: Sequence[str], value: Any
) -> None:
    target = mapping
    for key in path[:-1]:
        child = target.get(key)
        if not isinstance(child, MutableMapping):
            child = {}
            target[key] = child
        target = child
    target[path[-1]] = value


@contextmanager
def quiet_neuron_startup():
    key = "NEURON_MODULE_OPTIONS"
    previous = os.environ.get(key)
    options = previous.split() if previous else []
    if "-nogui" not in options:
        os.environ[key] = " ".join([*options, "-nogui"])
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


class PipelineUIComponent:
    """Base for private step components backed by the public controller state."""

    def __init__(self, controller: Any) -> None:
        object.__setattr__(self, "_controller", controller)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_controller"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_controller":
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_controller"), name, value)

    def sync_from_settings(self, *, act_settings_changed: bool = False) -> None:
        del act_settings_changed

    def refresh_button_states(self, *, ready: bool, act_busy: bool) -> None:
        del ready, act_busy


__all__ = ["PIPELINE_UI_DEFAULTS", "PipelineUIComponent"]
