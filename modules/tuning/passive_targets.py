"""Passive-target resolution for Step 2 notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .act_integration import import_act_passive_module
from .allen_nwb import write_allen_nwb_passive_target_csv
from .passive import compute_settable_passive_properties, passive_area_summary
from .passive_traces import write_passive_trace_target_csv
from .targets import (
    allen_nwb_block,
    ensure_target_config_shape,
    load_target_config,
    nwb_file_from_config,
    passive_targets_from_config,
    resolve_tune_path,
    target_source_mode_from_config,
    traces_block,
    update_passive_targets_in_config,
)


DEFAULT_NWB_PASSIVE_OPTIONS = {
    "stimulus_names": ["Long Square"],
    "sweep_ids": None,
    "min_current_pA": None,
    "max_current_pA": -1.0,
    "end_margin_ms": 10.0,
    "reducer": "median",
    "tau_field": "tau_avg_ms",
}


@dataclass(frozen=True)
class PassiveTargetResolution:
    """Resolved Step 2 target/config values."""

    act_passive_module: Any
    target_config: dict[str, Any]
    passive_targets: dict[str, Any]
    passive_area: dict[str, Any]
    settable_passive_properties: Any
    detected_nwb_files: list[Path]
    nwb_passive_path: Optional[Path]
    nwb_passive_summary: Optional[dict[str, Any]]
    fit_json_candidates: list[Path]
    target_source_mode: str = "manual"
    passive_trace_path: Optional[Path] = None
    passive_trace_summary: Optional[dict[str, Any]] = None


def resolve_passive_tuning_inputs(
    *,
    context: Any,
    cell: Any,
    manual_passive_targets: Optional[Mapping[str, Any]] = None,
    use_target_config: bool = True,
    extract_from_nwb: bool = False,
    apply_nwb_targets_to_config: bool = False,
    apply_extracted_targets_to_config: bool = False,
    nwb_path: Optional[str | Path] = None,
    target_source_mode: Optional[str] = None,
    trace_path: Optional[str | Path] = None,
    passive_area_mode: str = "auto",
    passive_area_scale: float = 1.0,
    custom_passive_area_cm2: Optional[float] = None,
    nwb_export_dir: Optional[str | Path] = None,
) -> PassiveTargetResolution:
    """Resolve Step 2 passive targets, optional NWB extraction, and ACT values."""
    target_config = ensure_target_config_shape(load_target_config(context.tune_dir))
    resolved_source_mode = _resolve_source_mode(target_source_mode, target_config)
    nwb_options = passive_nwb_options_from_config(target_config)
    trace_options = passive_trace_options_from_config(target_config)
    detected_nwb_files = sorted(Path(context.tune_dir).glob("*_ephys.nwb"))
    resolved_nwb_path = _resolve_nwb_path(
        nwb_path=nwb_path,
        target_config=target_config,
        tune_dir=context.tune_dir,
        detected_nwb_files=detected_nwb_files,
    )

    act_passive_module = import_act_passive_module(repo_root=context.repo_root)
    config_passive_targets = passive_targets_from_config(target_config)
    nwb_passive_summary = None
    passive_trace_summary = None
    resolved_trace_path = _resolve_trace_path(
        trace_path=trace_path,
        target_config=target_config,
        tune_dir=context.tune_dir,
    )

    if resolved_source_mode == "traces":
        if resolved_trace_path is None:
            raise FileNotFoundError(
                "No generic passive trace file selected. Set target_config.traces.passive.file "
                "or set trace_path explicitly."
            )
        export_dir = Path(nwb_export_dir or (context.tune_dir / "notebook_exports" / "step2_passive"))
        export_dir.mkdir(parents=True, exist_ok=True)
        passive_trace_summary = write_passive_trace_target_csv(
            resolved_trace_path,
            export_dir / "generic_trace_passive_targets.csv",
            act_passive_module=act_passive_module,
            sweep_summary_path=export_dir / "generic_trace_passive_sweeps.csv",
            trace_format=trace_options["format"],
            time_column=trace_options["time_column"],
            voltage_column=trace_options["voltage_column"],
            current_column=trace_options["current_column"],
            sweep_column=trace_options["sweep_column"],
            stim_start_ms=trace_options["stim_start_ms"],
            stim_stop_ms=trace_options["stim_stop_ms"],
            current_pA=trace_options["current_pA"],
            dt_ms=trace_options["dt_ms"],
            end_margin_ms=float(trace_options["end_margin_ms"]),
            reducer=str(trace_options["reducer"]),
            tau_field=str(trace_options["tau_field"]),
        )
        config_passive_targets = passive_targets_from_config({"manual": {"passive": passive_trace_summary["targets"]}})
        if apply_extracted_targets_to_config:
            target_config = update_passive_targets_in_config(
                context.tune_dir,
                passive_trace_summary["targets"],
                target_source="manual",
                notes=f"Passive targets extracted from {Path(resolved_trace_path).name}",
            )
            config_passive_targets = passive_targets_from_config(target_config)
    elif extract_from_nwb or resolved_source_mode == "allen_nwb":
        if resolved_nwb_path is None:
            raise FileNotFoundError(
                "No NWB file selected. Set target_config.allen_nwb.file, "
                "place one *_ephys.nwb in the tune folder, or set nwb_path explicitly."
            )
        export_dir = Path(nwb_export_dir or (context.tune_dir / "notebook_exports" / "step2_passive"))
        export_dir.mkdir(parents=True, exist_ok=True)
        nwb_passive_summary = write_allen_nwb_passive_target_csv(
            resolved_nwb_path,
            export_dir / "allen_nwb_passive_targets.csv",
            act_passive_module=act_passive_module,
            sweep_summary_path=export_dir / "allen_nwb_passive_sweeps.csv",
            stimulus_names=nwb_options["stimulus_names"],
            sweep_ids=_nwb_sweep_ids(nwb_options, target_config),
            min_current_pA=nwb_options["min_current_pA"],
            max_current_pA=nwb_options["max_current_pA"],
            end_margin_ms=float(nwb_options["end_margin_ms"]),
            reducer=str(nwb_options["reducer"]),
            tau_field=str(nwb_options["tau_field"]),
        )
        config_passive_targets = passive_targets_from_config({"manual": {"passive": nwb_passive_summary["targets"]}})
        if apply_nwb_targets_to_config or apply_extracted_targets_to_config:
            target_config = update_passive_targets_in_config(
                context.tune_dir,
                nwb_passive_summary["targets"],
                target_source="manual",
                notes=f"Passive targets extracted from {Path(resolved_nwb_path).name}",
            )
            config_passive_targets = passive_targets_from_config(target_config)

    target_rin_mohm = _resolve_passive_target(
        "target_rin_mohm",
        manual_passive_targets=manual_passive_targets,
        config_passive_targets=config_passive_targets,
        use_target_config=use_target_config,
    )
    target_tau_ms = _resolve_passive_target(
        "target_tau_ms",
        manual_passive_targets=manual_passive_targets,
        config_passive_targets=config_passive_targets,
        use_target_config=use_target_config,
    )
    target_v_rest_mv = _resolve_passive_target(
        "target_v_rest_mv",
        manual_passive_targets=manual_passive_targets,
        config_passive_targets=config_passive_targets,
        use_target_config=use_target_config,
    )

    passive_area = passive_area_summary(
        cell,
        area_mode=passive_area_mode,
        area_scale=passive_area_scale,
        custom_area_cm2=custom_passive_area_cm2,
    )
    settable_passive_properties = compute_settable_passive_properties(
        act_passive_module=act_passive_module,
        cell=cell,
        rin_mohm=target_rin_mohm,
        tau_ms=target_tau_ms,
        v_rest_mv=target_v_rest_mv,
        area_mode=passive_area_mode,
        area_scale=passive_area_scale,
        custom_area_cm2=custom_passive_area_cm2,
    )
    passive_targets = {
        "target_rin_mohm": target_rin_mohm,
        "target_tau_ms": target_tau_ms,
        "target_v_rest_mv": target_v_rest_mv,
        **passive_area,
    }

    return PassiveTargetResolution(
        act_passive_module=act_passive_module,
        target_config=target_config,
        passive_targets=passive_targets,
        passive_area=passive_area,
        settable_passive_properties=settable_passive_properties,
        detected_nwb_files=detected_nwb_files,
        nwb_passive_path=resolved_nwb_path,
        nwb_passive_summary=nwb_passive_summary,
        fit_json_candidates=sorted(Path(context.tune_dir).glob("*_fit.json")),
        target_source_mode=resolved_source_mode,
        passive_trace_path=resolved_trace_path,
        passive_trace_summary=passive_trace_summary,
    )


def passive_nwb_options_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return Step 2 NWB passive extraction options from `allen_nwb.passive`."""
    nwb_config = allen_nwb_block(config)
    raw_options = nwb_config.get("passive", {})
    if not isinstance(raw_options, Mapping):
        raw_options = {}
    options = dict(DEFAULT_NWB_PASSIVE_OPTIONS)
    for key in options:
        if key in raw_options:
            options[key] = raw_options[key]
    options["stimulus_names"] = _as_list(options["stimulus_names"])
    options["sweep_ids"] = _as_optional_int_list(options["sweep_ids"])
    for key in ("min_current_pA", "max_current_pA", "end_margin_ms"):
        options[key] = _optional_float(options[key])
    if options["end_margin_ms"] is None:
        options["end_margin_ms"] = DEFAULT_NWB_PASSIVE_OPTIONS["end_margin_ms"]
    options["reducer"] = options["reducer"] or DEFAULT_NWB_PASSIVE_OPTIONS["reducer"]
    options["tau_field"] = options["tau_field"] or DEFAULT_NWB_PASSIVE_OPTIONS["tau_field"]
    return options


def passive_trace_options_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return Step 2 generic passive trace extraction options from `traces.passive`."""
    trace_config = traces_block(config)
    raw_options = trace_config.get("passive", {})
    if not isinstance(raw_options, Mapping):
        raw_options = {}
    options = {
        "format": raw_options.get("format", trace_config.get("format", "csv")),
        "time_column": raw_options.get("time_column", "time_ms"),
        "voltage_column": raw_options.get("voltage_column", "voltage_mV"),
        "current_column": raw_options.get("current_column", "current_pA"),
        "sweep_column": raw_options.get("sweep_column"),
        "stim_start_ms": _optional_float(raw_options.get("stim_start_ms")),
        "stim_stop_ms": _optional_float(raw_options.get("stim_stop_ms")),
        "current_pA": _optional_float_or_list(raw_options.get("current_pA")),
        "dt_ms": _optional_float(raw_options.get("dt_ms")),
        "end_margin_ms": _optional_float(raw_options.get("end_margin_ms")),
        "reducer": raw_options.get("reducer") or "median",
        "tau_field": raw_options.get("tau_field") or "tau_avg_ms",
    }
    if options["end_margin_ms"] is None:
        options["end_margin_ms"] = 10.0
    return options


def _resolve_passive_target(
    field: str,
    *,
    manual_passive_targets: Optional[Mapping[str, Any]],
    config_passive_targets: Mapping[str, Any],
    use_target_config: bool,
) -> float:
    if manual_passive_targets and manual_passive_targets.get(field) is not None:
        return float(manual_passive_targets[field])
    if use_target_config and config_passive_targets.get(field) is not None:
        return float(config_passive_targets[field])
    raise ValueError(
        "Missing passive targets. Provide manual.passive.v_rest_mV, "
        "manual.passive.rin_MOhm, and manual.passive.tau_ms in target_config.json; "
        "set manual_passive_targets in the notebook; or set target_source.mode to "
        "'traces'/'allen_nwb'."
    )


def _resolve_nwb_path(
    *,
    nwb_path: Optional[str | Path],
    target_config: Mapping[str, Any],
    tune_dir: str | Path,
    detected_nwb_files: list[Path],
) -> Optional[Path]:
    if nwb_path not in (None, ""):
        path = Path(str(nwb_path)).expanduser()
        if not path.is_absolute():
            path = Path(tune_dir).expanduser().resolve() / path
        return path.resolve()
    return nwb_file_from_config(target_config, tune_dir) or (
        detected_nwb_files[0] if len(detected_nwb_files) == 1 else None
    )


def _nwb_sweep_ids(options: Mapping[str, Any], target_config: Mapping[str, Any]) -> Optional[list[int]]:
    values = options.get("sweep_ids")
    if values is None:
        values = allen_nwb_block(target_config).get("sweep_ids")
    return _as_optional_int_list(values)


def _resolve_trace_path(
    *,
    trace_path: Optional[str | Path],
    target_config: Mapping[str, Any],
    tune_dir: str | Path,
) -> Optional[Path]:
    if trace_path not in (None, ""):
        return resolve_tune_path(trace_path, tune_dir)
    passive = traces_block(target_config).get("passive", {})
    if not isinstance(passive, Mapping):
        return None
    return resolve_tune_path(passive.get("file"), tune_dir)


def _resolve_source_mode(explicit_mode: Optional[str], target_config: Mapping[str, Any]) -> str:
    if explicit_mode not in (None, ""):
        return target_source_mode_from_config({"target_source": {"mode": explicit_mode}})
    return target_source_mode_from_config(target_config)


def _as_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _as_optional_int_list(value: Any) -> Optional[list[int]]:
    if value in (None, ""):
        return None
    values = [value] if isinstance(value, (str, int, float)) else list(value)
    if not values:
        return None
    return [int(item) for item in values]


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def _optional_float_or_list(value: Any) -> Optional[float | list[float]]:
    if value in (None, ""):
        return None
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return float(value)
