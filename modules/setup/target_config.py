"""Step 1 target-config scaffolding helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .defaults import default_target_config
from .json_utils import CONFIG_MODE_VALUES, _write_json, _write_scaffold_json
from .paths import resolve_step1_paths

TARGET_SOURCE_MODES = ("manual", "traces", "allen_nwb")


def prepare_target_config(
    *,
    tune_dir: Path,
    config_mode: str = "fill",
    target_source_mode: Optional[str] = "manual",
    target_description: Optional[str] = None,
    manual_passive: Optional[Mapping[str, Any]] = None,
    manual_fi_curve: Optional[Mapping[str, Any]] = None,
    passive_trace: Optional[Mapping[str, Any]] = None,
    active_trace: Optional[Mapping[str, Any]] = None,
    allen_nwb: Optional[Mapping[str, Any]] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create/update `cell_configs/target_config.json`.

    Step 1 only records target metadata and paths. Steps 2-3 are responsible for
    calculating passive/FI targets from traces or Allen/ADB NWB files.
    """
    mode = str(config_mode).strip().lower()
    if mode not in CONFIG_MODE_VALUES:
        raise ValueError(f"config_mode must be one of {CONFIG_MODE_VALUES}, got {mode!r}")

    paths = resolve_step1_paths(Path(tune_dir).expanduser().resolve())
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    target_path = paths.target_config

    status, config = _write_scaffold_json(target_path, default_target_config(), mode)
    if mode == "skip":
        return {
            "path": str(target_path),
            "status": "skipped",
            "target_source_mode": _current_target_mode(config),
        }

    before = deepcopy(config)
    _apply_target_source(config, target_source_mode, target_description)
    _apply_mapping(config.setdefault("manual", {}).setdefault("passive", {}), manual_passive)
    _apply_mapping(config.setdefault("manual", {}).setdefault("fi_curve", {}), manual_fi_curve)
    _apply_mapping(config.setdefault("traces", {}).setdefault("passive", {}), passive_trace)
    _apply_mapping(config.setdefault("traces", {}).setdefault("active", {}), active_trace)
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
