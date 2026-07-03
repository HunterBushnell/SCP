"""Synapse-group config normalization for Step 5 input generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import copy
import json


def _normalize_group_configs(
    groups_cfg_raw: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Normalize/validate per-group configs.

    Ensures each group has the expected top-level blocks (state, mode, source,
    timing, syns) with consistent keys and simple defaults.
    """
    if not isinstance(groups_cfg_raw, dict):
        raise TypeError(
            f"'synapse_groups' must be a dict (got {type(groups_cfg_raw)!r})"
        )

    groups_cfg: Dict[str, Dict[str, Any]] = {}

    timing_keys = (
        "onset_ms",
        "stim_tstart_ms",
        "stim_jitter",
        "duration_ms",
        "input_stim_tstart_ms",
        "input_duration_ms",
    )
    source_keys = (
        "freq",
        "baseline",
        "gabab",
        "freq_scale",
        "freq_shift",
        "kind",
        "path",
        "time_col",
        "rate_col",
        "bin_ms",
        "ref",
        "key",
    )

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
            # None → treated as inactive for now (legacy scratch groups, etc.)
            state = False
        if not isinstance(state, bool):
            raise ValueError(
                f"Group '{gname}': 'state' must be a bool (got {type(state)!r})"
            )
        gcfg["state"] = state

        # mode: required
        mode = gcfg.get("mode")
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
        source = dict(source_raw)
        for key in source_keys:
            source.setdefault(key, None)
        gcfg["source"] = source

        # timing block
        timing_raw = gcfg.get("timing", {})
        if not isinstance(timing_raw, dict):
            raise ValueError(
                f"Group '{gname}': 'timing' must be a dict (got {type(timing_raw)!r})"
            )
        timing = dict(timing_raw)
        for key in timing_keys:
            timing.setdefault(key, None)
        gcfg["timing"] = timing

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
      - Plain dict of groups (legacy/normal).
      - Dict with key "__includes__": list of relative file paths. Each file
        must be a dict mapping group_name -> config. Any other keys in the
        top-level dict are treated as inline group definitions and merged.
      - Top-level list: treated as the include list (no inline groups).
    """
    # Legacy: already a dict of groups
    if isinstance(groups_cfg_raw, dict) and "__includes__" not in groups_cfg_raw:
        return groups_cfg_raw

    include_list: List[str] = []
    inline_groups: Dict[str, Any] = {}

    if isinstance(groups_cfg_raw, list):
        include_list = [str(p) for p in groups_cfg_raw]
    elif isinstance(groups_cfg_raw, dict):
        include_list = [str(p) for p in groups_cfg_raw.get("__includes__", []) or []]
        inline_groups = {k: v for k, v in groups_cfg_raw.items() if k != "__includes__"}
    else:
        raise TypeError(
            "syn_config must be a dict of groups, a dict with '__includes__', or a list of include paths"
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
