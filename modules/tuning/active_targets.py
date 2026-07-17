"""Active/FI target resolution for Step 3 notebooks and ACT preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .targets import (
    allen_nwb_block,
    fi_curve_from_config,
    load_target_config,
    manual_fi_csv_from_config,
    nwb_file_from_config,
    resolve_tune_path,
    target_source_mode_from_config,
    traces_block,
)


DEFAULT_ACTIVE_TARGET_MODE = "fi_arrays"
VALID_ACTIVE_TARGET_MODES = {"fi_arrays", "fi_csv", "allen_nwb", "trace_npy"}
DEFAULT_NWB_ACTIVE_OPTIONS = {
    "stimulus_names": ["Long Square"],
    "min_current_pA": 0.0,
    "max_current_pA": None,
    "include_negative_currents": False,
    "average_repeats": True,
    "spike_threshold_mV": -20.0,
    "refractory_ms": 1.0,
}


@dataclass(frozen=True)
class ActiveTargetResolution:
    """Resolved Step 3 target/config values."""

    target_config: dict[str, Any]
    target_mode: str
    fi_currents_pA: list[float]
    fi_frequencies_hz: list[float]
    fi_csv_path: Optional[Path]
    trace_npy_path: Optional[Path]
    nwb_path: Optional[Path]
    nwb_options: dict[str, Any]
    detected_nwb_files: list[Path]
    target_source_mode: str

    @property
    def fi_reference_points(self) -> list[tuple[float, float]]:
        """Return FI points suitable for plotting overlays."""
        return list(zip(self.fi_currents_pA, self.fi_frequencies_hz))


def resolve_active_tuning_targets(
    *,
    context: Any,
    target_mode: Optional[str] = None,
    fi_currents_pA: Optional[Sequence[float]] = None,
    fi_frequencies_hz: Optional[Sequence[float]] = None,
    fi_csv_path: Optional[str | Path] = None,
    trace_npy_path: Optional[str | Path] = None,
    nwb_path: Optional[str | Path] = None,
    require_target: bool = False,
) -> ActiveTargetResolution:
    """Resolve Step 3 target data from notebook overrides and `target_config.json`."""
    target_config = load_target_config(context.tune_dir)
    nwb_options = active_nwb_options_from_config(target_config)
    source_mode = target_source_mode_from_config(target_config, default="none")
    resolved_mode = _resolve_target_mode(target_mode, target_config, source_mode)
    detected_nwb_files = sorted(Path(context.tune_dir).glob("*_ephys.nwb"))

    config_currents, config_freqs = fi_curve_from_config(target_config)
    currents = _float_list(fi_currents_pA) if fi_currents_pA is not None else config_currents
    freqs = _float_list(fi_frequencies_hz) if fi_frequencies_hz is not None else config_freqs

    resolved_fi_csv = _resolve_optional_tune_path(
        explicit_path=fi_csv_path,
        config_value=manual_fi_csv_from_config(target_config, context.tune_dir),
        tune_dir=context.tune_dir,
    )
    resolved_trace_npy = _resolve_optional_tune_path(
        explicit_path=trace_npy_path,
        config_value=_trace_active_file(target_config),
        tune_dir=context.tune_dir,
    )
    resolved_nwb = _resolve_nwb_path(
        nwb_path=nwb_path,
        target_config=target_config,
        tune_dir=context.tune_dir,
        detected_nwb_files=detected_nwb_files,
    )

    resolution = ActiveTargetResolution(
        target_config=target_config,
        target_mode=resolved_mode,
        fi_currents_pA=currents,
        fi_frequencies_hz=freqs,
        fi_csv_path=resolved_fi_csv,
        trace_npy_path=resolved_trace_npy,
        nwb_path=resolved_nwb,
        nwb_options=nwb_options,
        detected_nwb_files=detected_nwb_files,
        target_source_mode=source_mode,
    )
    if require_target:
        validate_active_target_resolution(resolution)
    return resolution


def validate_active_target_resolution(resolution: ActiveTargetResolution) -> None:
    """Raise a clear error if the selected active target mode is incomplete."""
    mode = resolution.target_mode
    if mode == "fi_arrays":
        if not resolution.fi_currents_pA or not resolution.fi_frequencies_hz:
            raise ValueError(
                "Missing active FI targets. Fill manual.fi_curve.currents_pA and "
                "manual.fi_curve.rates_Hz in target_config.json, set ACT_FI_* "
                "notebook overrides, or choose ACT_TARGET_MODE='allen_nwb'/'fi_csv'."
            )
        if len(resolution.fi_currents_pA) != len(resolution.fi_frequencies_hz):
            raise ValueError("Active FI current and rate arrays must have the same length.")
        return
    if mode == "fi_csv":
        if resolution.fi_csv_path is None:
            raise ValueError(
                "ACT_TARGET_MODE='fi_csv' requires manual.fi_curve.csv in target_config.json "
                "or ACT_FI_CSV_PATH in the notebook."
            )
        return
    if mode == "allen_nwb":
        if resolution.nwb_path is None:
            raise ValueError(
                "ACT_TARGET_MODE='allen_nwb' requires allen_nwb.file, one "
                "*_ephys.nwb file in the tune directory, or ACT_NWB_PATH."
            )
        return
    if mode == "trace_npy":
        if resolution.trace_npy_path is None:
            raise ValueError(
                "ACT_TARGET_MODE='trace_npy' requires traces.active.file in target_config.json "
                "or ACT_TRACE_NPY_PATH in the notebook. The file must be ACT-compatible."
            )
        trace_format = _trace_active_format(resolution.target_config)
        if trace_format != "npy":
            raise ValueError(
                "ACT trace target mode currently requires traces.active.format='npy'. "
                "Use manual.fi_curve.csv for generic FI CSV targets."
            )
        return
    raise ValueError(f"Unknown active target mode: {mode!r}")


def active_nwb_options_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return Step 3 NWB FI extraction options from `allen_nwb.active`."""
    nwb_config = allen_nwb_block(config)
    raw_options = nwb_config.get("active", {})
    if not isinstance(raw_options, Mapping):
        raw_options = {}
    options = dict(DEFAULT_NWB_ACTIVE_OPTIONS)
    for key in options:
        if key in raw_options:
            options[key] = raw_options[key]
    options["stimulus_names"] = _as_list(options["stimulus_names"])
    for key in ("min_current_pA", "max_current_pA", "spike_threshold_mV", "refractory_ms"):
        options[key] = _optional_float(options[key])
    if options["spike_threshold_mV"] is None:
        options["spike_threshold_mV"] = DEFAULT_NWB_ACTIVE_OPTIONS["spike_threshold_mV"]
    if options["refractory_ms"] is None:
        options["refractory_ms"] = DEFAULT_NWB_ACTIVE_OPTIONS["refractory_ms"]
    options["include_negative_currents"] = bool(options["include_negative_currents"])
    options["average_repeats"] = bool(options["average_repeats"])
    return options


def _resolve_target_mode(
    explicit_mode: Optional[str],
    target_config: Mapping[str, Any],
    source_mode: str,
) -> str:
    if explicit_mode not in (None, ""):
        mode = str(explicit_mode).strip().lower()
    elif source_mode == "manual":
        mode = "fi_csv" if manual_fi_csv_from_config(target_config, ".") is not None else "fi_arrays"
    elif source_mode == "allen_nwb":
        mode = "allen_nwb"
    elif source_mode == "traces":
        mode = "trace_npy"
    else:
        mode = DEFAULT_ACTIVE_TARGET_MODE
    if mode not in VALID_ACTIVE_TARGET_MODES:
        raise ValueError(
            "Active target mode must be one of "
            + ", ".join(sorted(VALID_ACTIVE_TARGET_MODES))
        )
    return mode


def _resolve_nwb_path(
    *,
    nwb_path: Optional[str | Path],
    target_config: Mapping[str, Any],
    tune_dir: str | Path,
    detected_nwb_files: list[Path],
) -> Optional[Path]:
    if nwb_path not in (None, ""):
        return _resolve_tune_path(nwb_path, tune_dir)
    return nwb_file_from_config(target_config, tune_dir) or (
        detected_nwb_files[0] if len(detected_nwb_files) == 1 else None
    )


def _resolve_optional_tune_path(
    *,
    explicit_path: Optional[str | Path],
    config_value: Any,
    tune_dir: str | Path,
) -> Optional[Path]:
    raw = explicit_path if explicit_path not in (None, "") else config_value
    if raw in (None, ""):
        return None
    return _resolve_tune_path(raw, tune_dir)


def _resolve_tune_path(path: str | Path, tune_dir: str | Path) -> Path:
    path_obj = Path(str(path)).expanduser()
    if not path_obj.is_absolute():
        path_obj = Path(tune_dir).expanduser().resolve() / path_obj
    return path_obj.resolve()


def _trace_active_file(config: Mapping[str, Any]) -> Any:
    active = traces_block(config).get("active", {})
    if not isinstance(active, Mapping):
        return None
    return active.get("file")


def _trace_active_format(config: Mapping[str, Any]) -> str:
    trace_config = traces_block(config)
    active = trace_config.get("active", {})
    raw = active.get("format") if isinstance(active, Mapping) else trace_config.get("format")
    return str(raw or "npy").strip().lower()


def _float_list(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]


def _as_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)
