"""Step 1 target-config scaffolding helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .defaults import default_target_config
from .json_utils import CONFIG_MODE_VALUES, _write_json, _write_scaffold_json
from .paths import resolve_step1_paths

TARGET_SOURCE_MODES = ("none", "manual", "traces", "allen_nwb")


def prepare_target_config(
    *,
    tune_dir: Path,
    config_mode: str = "fill",
    target_source_mode: Optional[str] = None,
    target_description: Optional[str] = None,
    manual_passive: Optional[Mapping[str, Any]] = None,
    manual_fi_curve: Optional[Mapping[str, Any]] = None,
    passive_trace: Optional[Mapping[str, Any]] = None,
    active_trace: Optional[Mapping[str, Any]] = None,
    allen_nwb: Optional[Mapping[str, Any]] = None,
    include_manual_with_file_source: bool = False,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create/update `cell_configs/target_config.json`.

    New/overwritten files contain only the selected source block. When no mode
    is explicit, trace/NWB file mappings select their matching mode and no file
    mapping selects manual mode. Steps 2-3 perform file-backed extraction.
    """
    mode = str(config_mode).strip().lower()
    if mode not in CONFIG_MODE_VALUES:
        raise ValueError(f"config_mode must be one of {CONFIG_MODE_VALUES}, got {mode!r}")

    paths = resolve_step1_paths(Path(tune_dir).expanduser().resolve())
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    target_path = paths.target_config
    resolved_source_mode = infer_target_source_mode(
        target_source_mode=target_source_mode,
        passive_trace=passive_trace,
        active_trace=active_trace,
        allen_nwb=allen_nwb,
    )
    manual_data_supplied = _has_mapping_data(manual_passive) or _has_mapping_data(
        manual_fi_curve
    )
    include_manual = resolved_source_mode == "manual" or (
        resolved_source_mode in {"traces", "allen_nwb"}
        and (bool(include_manual_with_file_source) or manual_data_supplied)
    )

    status, config = _write_scaffold_json(
        target_path,
        default_target_config(
            mode=resolved_source_mode,
            include_manual_with_file_source=include_manual,
        ),
        mode,
    )
    if mode == "skip":
        return {
            "path": str(target_path),
            "status": "skipped",
            "target_source_mode": _current_target_mode(config),
        }

    before = deepcopy(config)
    _apply_target_source(config, resolved_source_mode, target_description)
    if resolved_source_mode == "manual" or include_manual:
        _apply_mapping(
            config.setdefault("manual", {}).setdefault("passive", {}),
            manual_passive,
        )
        _apply_mapping(
            config.setdefault("manual", {}).setdefault("fi_curve", {}),
            manual_fi_curve,
        )
    if resolved_source_mode == "traces":
        _apply_mapping(
            config.setdefault("traces", {}).setdefault("passive", {}),
            passive_trace,
        )
        _apply_mapping(
            config.setdefault("traces", {}).setdefault("active", {}),
            active_trace,
        )
    elif resolved_source_mode == "allen_nwb":
        _apply_allen_nwb(config, allen_nwb)
    if notes is not None:
        config["notes"] = str(notes)

    if config != before:
        _write_json(target_path, config)
        if status == "unchanged":
            status = "updated"

    return {
        "path": str(target_path),
        "status": status,
        "target_source_mode": _current_target_mode(config),
    }


def infer_target_source_mode(
    *,
    target_source_mode: Optional[str] = None,
    passive_trace: Optional[Mapping[str, Any]] = None,
    active_trace: Optional[Mapping[str, Any]] = None,
    allen_nwb: Optional[Mapping[str, Any]] = None,
) -> str:
    """Resolve a target mode, inferring file-backed modes when none is explicit."""
    trace_selected = _mapping_has_file(passive_trace) or _mapping_has_file(
        active_trace
    )
    nwb_selected = _mapping_has_file(allen_nwb)
    if trace_selected and nwb_selected:
        raise ValueError(
            "Both user-trace and Allen NWB target files were supplied. Select only "
            "one file-backed target source."
        )

    if target_source_mode not in (None, ""):
        mode = str(target_source_mode).strip().lower()
        if mode not in TARGET_SOURCE_MODES:
            raise ValueError(
                "target_source_mode must be one of " + ", ".join(TARGET_SOURCE_MODES)
            )
        if trace_selected and mode != "traces":
            raise ValueError(
                f"A user-trace target file was supplied, but target_source_mode={mode!r}. "
                "Use target_source_mode='traces' or omit the mode to infer it."
            )
        if nwb_selected and mode != "allen_nwb":
            raise ValueError(
                f"An Allen NWB target file was supplied, but target_source_mode={mode!r}. "
                "Use target_source_mode='allen_nwb' or omit the mode to infer it."
            )
        return mode
    if nwb_selected:
        return "allen_nwb"
    if trace_selected:
        return "traces"
    return "manual"


def _apply_target_source(
    config: Dict[str, Any],
    target_source_mode: Optional[str],
    target_description: Optional[str],
) -> None:
    source = config.setdefault("target_source", {})
    if not isinstance(source, dict):
        source = {}
        config["target_source"] = source

    if target_source_mode not in (None, ""):
        mode = str(target_source_mode).strip().lower()
        if mode not in TARGET_SOURCE_MODES:
            raise ValueError(
                "target_source_mode must be one of "
                + ", ".join(TARGET_SOURCE_MODES)
            )
        source["mode"] = mode
    else:
        source.setdefault("mode", "manual")

    if target_description is not None:
        source["description"] = str(target_description)
    else:
        source.setdefault("description", "")


def _apply_mapping(target: Dict[str, Any], values: Optional[Mapping[str, Any]]) -> None:
    if not values:
        return
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _apply_mapping(target[key], value)
        else:
            target[key] = _json_safe(value)


def _apply_allen_nwb(config: Dict[str, Any], values: Optional[Mapping[str, Any]]) -> None:
    if not values:
        return
    block = config.setdefault("allen_nwb", {})
    if not isinstance(block, dict):
        block = {}
        config["allen_nwb"] = block
    _apply_mapping(block, values)


def _mapping_has_file(values: Optional[Mapping[str, Any]]) -> bool:
    return bool(values and values.get("file") not in (None, ""))


def _has_mapping_data(values: Optional[Mapping[str, Any]]) -> bool:
    if not values:
        return False
    for value in values.values():
        if isinstance(value, Mapping):
            if _has_mapping_data(value):
                return True
        elif value not in (None, "", [], {}):
            return True
    return False


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _current_target_mode(config: Mapping[str, Any]) -> str:
    source = config.get("target_source") if isinstance(config, Mapping) else {}
    if isinstance(source, Mapping):
        return str(source.get("mode", "manual"))
    return "manual"
