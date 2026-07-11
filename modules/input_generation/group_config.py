"""Synapse-group config normalization for Step 5 input generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import json


_GROUP_FILE_KEYS = ("group_files",)


def _has_group_file_manifest(groups_cfg_raw: Any) -> bool:
    return isinstance(groups_cfg_raw, dict) and (
        "group_files" in groups_cfg_raw or "__includes__" in groups_cfg_raw
    )


def _read_group_file_list(groups_cfg_raw: Dict[str, Any]) -> List[str]:
    """Read the canonical group_files manifest."""
    if "__includes__" in groups_cfg_raw:
        raise ValueError("syn_config.json no longer supports '__includes__'; use 'group_files'.")

    raw_paths = groups_cfg_raw.get("group_files", []) or []
    if not isinstance(raw_paths, list):
        raise TypeError("syn_config.json field 'group_files' must be a list of relative paths.")
    return [str(path) for path in raw_paths]


def _normalize_group_configs(
    groups_cfg_raw: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Normalize/validate per-group configs.

    Ensures each group has the expected top-level blocks with consistent keys.
    New configs define synaptic input timing/source behavior through
    ``input_blocks``. Group-level ``mode``/``source`` remain available for
    advanced direct-mode configs; non-empty group-level ``timing`` is rejected.
    """
    if not isinstance(groups_cfg_raw, dict):
        raise TypeError(
            f"'synapse_groups' must be a dict (got {type(groups_cfg_raw)!r})"
        )

    groups_cfg: Dict[str, Dict[str, Any]] = {}

    for gname, gcfg_raw in groups_cfg_raw.items():
        if not isinstance(gcfg_raw, dict):
            raise TypeError(
                f"Each synapse group config must be a dict (got {type(gcfg_raw)!r}) "
                f"for group {gname!r}"
            )

        gcfg = dict(gcfg_raw)
        gcfg.setdefault("name", str(gname))

        # state: default to True
        state = gcfg.get("state", True)
        if state is None:
            state = False
        if not isinstance(state, bool):
            raise ValueError(
                f"Group '{gname}': 'state' must be a bool (got {type(state)!r})"
            )
        gcfg["state"] = state

        input_blocks_raw = gcfg.get("input_blocks", []) or []
        if not isinstance(input_blocks_raw, list):
            raise ValueError(
                f"Group '{gname}': 'input_blocks' must be a list "
                f"(got {type(input_blocks_raw)!r})"
            )
        gcfg["input_blocks"] = [dict(block) for block in input_blocks_raw]

        # mode: required only for configs that do not use input_blocks.
        mode = gcfg.get("mode")
        if mode is None and gcfg["input_blocks"]:
            mode = "block_sequence"
        if mode is None:
            raise ValueError(f"Group '{gname}' is missing required key 'mode'")
        if not isinstance(mode, str):
            raise ValueError(
                f"Group '{gname}' has non-string 'mode' (got {type(mode)!r})"
            )
        gcfg["mode"] = mode

        # source block
        source_raw = gcfg.get("source", {})
        if not isinstance(source_raw, dict):
            raise ValueError(
                f"Group '{gname}': 'source' must be a dict (got {type(source_raw)!r})"
            )
        gcfg["source"] = dict(source_raw)

        # Old public timing fields were replaced by explicit input_blocks.
        timing_raw = gcfg.get("timing", None)
        if timing_raw not in (None, {}):
            raise ValueError(
                f"Group '{gname}': 'timing' is no longer supported; "
                "move timing/source settings into input_blocks."
            )
        if timing_raw is not None and not isinstance(timing_raw, dict):
            raise ValueError(
                f"Group '{gname}': 'timing' must be a dict (got {type(timing_raw)!r})"
            )
        gcfg.pop("timing", None)

        # syns block
        syns_raw = gcfg.get("syns", {})
        if not isinstance(syns_raw, dict):
            raise ValueError(
                f"Group '{gname}': 'syns' must be a dict (got {type(syns_raw)!r})"
            )
        syns = dict(syns_raw)
        syns.setdefault("type", None)
        syns.setdefault("N_syn", None)
        syns.setdefault("segs", None)
        syns.setdefault("dist_func", {"kind": None, "params": {}})
        syns.setdefault("params", {})
        gcfg["syns"] = syns

        groups_cfg[gname] = gcfg

    return groups_cfg


def _expand_group_includes(
    groups_cfg_raw: Any,
    root: Path,
) -> Dict[str, Dict[str, Any]]:
    """
    Allow syn_config.json to point to per-group files via an include list.

    Accepted forms:
      - Dict with key "group_files": list of relative file paths. Each file
        must be a dict mapping group_name -> config. Any other keys in the
        top-level dict are treated as inline group definitions and merged.
      - Plain dict of inline groups.
      - Top-level list: treated as the include list (no inline groups).
    """
    # Inline group dictionary.
    if isinstance(groups_cfg_raw, dict) and not _has_group_file_manifest(groups_cfg_raw):
        return groups_cfg_raw

    include_list: List[str] = []
    inline_groups: Dict[str, Any] = {}

    if isinstance(groups_cfg_raw, list):
        include_list = [str(p) for p in groups_cfg_raw]
    elif isinstance(groups_cfg_raw, dict):
        include_list = _read_group_file_list(groups_cfg_raw)
        inline_groups = {k: v for k, v in groups_cfg_raw.items() if k not in _GROUP_FILE_KEYS}
    else:
        raise TypeError(
            "syn_config must be a dict of groups, a dict with 'group_files', or a list of include paths"
        )

    merged: Dict[str, Dict[str, Any]] = {}

    def _merge_from_file(rel_path: str) -> None:
        p = (root / rel_path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Included synapse config file not found: {p}")
        with p.open("r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise TypeError(f"Included synapse config {p} must be a dict mapping group names to configs")
        for gname, gcfg in data.items():
            if gname in merged:
                raise ValueError(f"Duplicate group '{gname}' found while merging {p}")
            merged[gname] = gcfg

    for rel in include_list:
        _merge_from_file(rel)

    for gname, gcfg in inline_groups.items():
        if gname in merged:
            raise ValueError(f"Duplicate group '{gname}' from inline groups for {gname!r}")
        merged[gname] = gcfg

    return merged
