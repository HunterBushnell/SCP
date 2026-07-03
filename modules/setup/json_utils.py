"""Small JSON and scaffold-writing helpers used by Step 1 setup."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import copy
import json

CONFIG_MODE_VALUES = ("fill", "overwrite", "skip")


def _deep_fill(existing: Any, defaults: Any) -> Any:
    """
    Recursively fill missing keys in `existing` from `defaults`.

    Existing values always win. Lists are not merged element-wise.
    """
    if isinstance(existing, dict) and isinstance(defaults, dict):
        merged = dict(existing)
        for key, val in defaults.items():
            if key in merged:
                merged[key] = _deep_fill(merged[key], val)
            else:
                merged[key] = copy.deepcopy(val)
        return merged
    return existing


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, found {type(data)!r}")
    return data


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _write_scaffold_json(path: Path, defaults: Dict[str, Any], mode: str) -> Tuple[str, Dict[str, Any]]:
    mode = str(mode).strip().lower()
    if mode not in CONFIG_MODE_VALUES:
        raise ValueError(f"config_mode must be one of {CONFIG_MODE_VALUES}, got {mode!r}")

    if not path.exists():
        _write_json(path, defaults)
        return "created", dict(defaults)

    if mode == "skip":
        return "unchanged", _read_json(path)

    if mode == "overwrite":
        _write_json(path, defaults)
        return "overwritten", dict(defaults)

    existing = _read_json(path)
    merged = _deep_fill(existing, defaults)
    if merged != existing:
        _write_json(path, merged)
        return "updated", merged
    return "unchanged", existing
