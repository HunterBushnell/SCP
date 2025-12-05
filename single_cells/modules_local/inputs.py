"""
inputs.py

Step 2.3 – Generate synaptic input spike trains for the PV-SST project.

This is a first-layer scaffold:
- generate_inputs(...) wires the high-level steps (2.3.1–2.3.5)
- all heavy work is delegated to helper functions that are stubs for now
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import json
import numpy as np

from dataclasses import dataclass, field
from typing import List

from modules_local import input_modes_core  # or: from . import input_modes_core



# ---------------------------------------------------------------------
# Lightweight input checker (pre-2.3)
# ---------------------------------------------------------------------

def check_inputs(
    syn_config_path: str | Path,
    *,
    verbose: bool = True,
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Lightweight pre-2.3 sanity check.

    - Loads syn_config.json
    - Runs the same normalization as generate_inputs
    - Prints a compact summary of groups (state, mode, source.path, N_syn)
    - Returns (sim_cfg, groups_cfg) in normalized form for inspection.
    """

    path = Path(syn_config_path)
    if not path.exists():
        raise FileNotFoundError(f"syn_config.json not found at: {path}")

    # Reuse the existing internal machinery
    sim_raw, groups_raw = _load_and_split_syn_config(path)
    sim_cfg = _normalize_sim_config(sim_raw)
    groups_cfg = _normalize_group_configs(groups_raw)

    if verbose:
        print("=== check_inputs: synapse config summary ===")
        print("Sim cfg:", sim_cfg)

        print("\nSynapse groups:")
        for gname, gcfg in groups_cfg.items():
            state = gcfg.get("state", True)
            mode = gcfg.get("mode")
            source = gcfg.get("source", {}) or {}
            syns = gcfg.get("syns", {}) or {}
            src_path = source.get("path")
            n_syn = syns.get("N_syn")

            print(
                f"  - {gname:15s} state={state!r:5}  "
                f"mode={mode!r:18}  "
                f"source.path={src_path!r:25}  "
                f"N_syn={n_syn!r}"
            )

            if mode is None and state not in (False, "off", 0):
                raise ValueError(f"Active group '{gname}' is missing a 'mode'.")

    return sim_cfg, groups_cfg

# ---------------------------------------------------------------------
# Data structure for per-group inputs
# ---------------------------------------------------------------------

@dataclass
class GroupInputs:
    """
    Final per-group inputs structure produced by generate_inputs and
    consumed by the synapse-building step (2.4).

    For now we keep it minimal: name, mode, and spike trains (in ms,
    in simulation time). Later we can add timing/meta fields as needed.
    """
    name: str
    mode: str
    spike_trains: List[np.ndarray] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------


def generate_inputs(
    syn_config_path: Path | str,
    geometry: Optional[Any] = None,
    rng: Optional[np.random.Generator] = None,
    mode_registry: Optional[Mapping[str, Any]] = None,
# ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Any]]:
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, GroupInputs]]:
    """
    Top-level entry point for Step 2.3.

    Parameters
    ----------
    syn_config_path
        Path to the synapse configuration JSON (e.g. TUNE_DIR / "syn_config.json").
    geometry
        Geometry/segment-group information from Step 2.2 (optional for now).
    rng
        Optional NumPy random number generator for reproducible inputs.
        If None, an internal generator will be created.
    mode_registry
        Optional mapping from mode names (str) to handler callables.
        For now we will support three default modes (precomputed, homogeneous, inhomogeneous),
        but the design is intended to allow user-defined modes later.

    Returns
    -------
    sim_cfg, groups_cfg, inputs
        sim_cfg:   normalized simulation config dict
        groups_cfg:normalized per-group config dict
        inputs:    dict of per-group input objects (exact structure to be
                   defined as we implement later steps).
    """

    # 2.3.1 – Load and split config
    sim_cfg_raw, groups_cfg_raw = _load_and_split_syn_config(syn_config_path)

    # 2.3.2 – Normalize configs
    sim_cfg = _normalize_sim_config(sim_cfg_raw)
    groups_cfg = _normalize_group_configs(groups_cfg_raw)

    # 2.3.3 – Set up shared objects/resources
    rng = _init_rng(rng)
    default_registry = _build_default_mode_registry()
    if mode_registry is None:
        mode_registry = default_registry

    # 2.3.4 – Loop over groups and generate per-group inputs
    inputs = _process_all_groups(
        sim_cfg=sim_cfg,
        groups_cfg=groups_cfg,
        geometry=geometry,
        mode_registry=mode_registry,
        rng=rng,
    )

    # 2.3.5 – Finalize and sanity-check
    _finalize_inputs(sim_cfg, groups_cfg, inputs)

    return sim_cfg, groups_cfg, inputs
        



#######################################################################
#######################################################################
# Helper stubs for 2.3.1–2.3.5
#######################################################################
#######################################################################

# ---------------------------------------------------------------------
# 2.3.1
# ---------------------------------------------------------------------

def _load_and_split_syn_config(
    syn_config_path: Path | str,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    2.3.1 – Load syn_config.json and split into:
        sim_cfg_raw, groups_cfg_raw

    Inputs
    ------
    syn_config_path: Path | str
        Path to JSON with top-level keys "sim" and "synapse_groups".

    Outputs
    -------
    sim_cfg_raw: dict
    groups_cfg_raw: dict[str, dict]

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    ValueError
        If required top-level keys are missing or of the wrong type.
    """
    syn_config_path = Path(syn_config_path)

    # Load the JSON
    with syn_config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Basic structure checks
    if "sim" not in cfg or "synapse_groups" not in cfg:
        raise ValueError(
            f"syn_config.json must contain top-level keys 'sim' and "
            f"'synapse_groups' (got keys: {list(cfg.keys())})"
        )

    sim_cfg_raw = cfg["sim"]
    groups_cfg_raw = cfg["synapse_groups"]

    if not isinstance(sim_cfg_raw, dict):
        raise ValueError(
            f"'sim' block must be a dict (got {type(sim_cfg_raw)!r})"
        )

    if not isinstance(groups_cfg_raw, dict):
        raise ValueError(
            "'synapse_groups' block must be a dict mapping group_name -> group_cfg "
            f"(got {type(groups_cfg_raw)!r})"
        )

    # Optional: enforce that each group config is a dict
    for gname, gcfg in groups_cfg_raw.items():
        if not isinstance(gcfg, dict):
            raise ValueError(
                f"Group '{gname}' in 'synapse_groups' must be a dict "
                f"(got {type(gcfg)!r})"
            )

    return sim_cfg_raw, groups_cfg_raw


# ---------------------------------------------------------------------
# 2.3.2
# ---------------------------------------------------------------------

def _normalize_sim_config(sim_cfg_raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    2.3.2 – Normalize/validate the simulation config.

    Inputs
    ------
    sim_cfg_raw: dict
        Raw sim config from JSON.

    Outputs
    -------
    sim_cfg: dict
        Normalized config (required keys, types, defaults).
    """
    if not isinstance(sim_cfg_raw, dict):
        raise ValueError(f"'sim' block must be a dict (got {type(sim_cfg_raw)!r})")

    sim_cfg: Dict[str, Any] = dict(sim_cfg_raw)  # shallow copy

    # Required numeric fields
    for key in ("tstart", "tstop", "dt"):
        if key not in sim_cfg:
            raise ValueError(f"sim config missing required key '{key}'")
        try:
            sim_cfg[key] = float(sim_cfg[key])
        except (TypeError, ValueError):
            raise ValueError(f"sim['{key}'] must be numeric (got {sim_cfg[key]!r})")

    if sim_cfg["tstop"] <= sim_cfg["tstart"]:
        raise ValueError(
            f"sim['tstop']={sim_cfg['tstop']} must be > sim['tstart']={sim_cfg['tstart']}"
        )

    # Optional jitter
    if "jitter" not in sim_cfg or sim_cfg["jitter"] is None:
        sim_cfg["jitter"] = None
    else:
        try:
            sim_cfg["jitter"] = float(sim_cfg["jitter"])
        except (TypeError, ValueError):
            raise ValueError(f"sim['jitter'] must be numeric or null (got {sim_cfg['jitter']!r})")

    # Leave other keys (e.g. 'cell', 'tune') as-is
    return sim_cfg
    # raise NotImplementedError


def _normalize_group_configs(
    groups_cfg_raw: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    2.3.2 – Normalize/validate per-group configs.

    Ensures each group has the expected top-level blocks (state, mode, source,
    timing, syns) with consistent keys and simple defaults.

    Inputs
    ------
    groups_cfg_raw: dict[str, dict]
        Raw group configs from JSON ("synapse_groups").

    Outputs
    -------
    groups_cfg: dict[str, dict]
        Normalized group configs, ready for per-group processing.
    """
    if not isinstance(groups_cfg_raw, dict):
        raise ValueError(
            f"'synapse_groups' must be a dict mapping group_name -> group_cfg "
            f"(got {type(groups_cfg_raw)!r})"
        )

    groups_cfg: Dict[str, Dict[str, Any]] = {}

    timing_keys = (
        "onset_ms",
        "stim_tstart_ms",
        "duration_ms",
        "input_stim_tstart_ms",
        "input_duration_ms",
    )
    source_keys = (
        "freq",
        "baseline",
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
            raise ValueError(
                f"Group '{gname}' in 'synapse_groups' must be a dict "
                f"(got {type(gcfg_raw)!r})"
            )

        gcfg: Dict[str, Any] = dict(gcfg_raw)  # shallow copy

        # state: default to True if missing
        state = gcfg.get("state", True)
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

        # syns block (keep this light; we mainly ensure dict shape)
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


# ---------------------------------------------------------------------
# 2.3.3 – Mode registry and built-in mode stubs
# ---------------------------------------------------------------------

def _init_rng(
    rng: Optional[np.random.Generator],
) -> np.random.Generator:
    """
    2.3.3 – Ensure we have a NumPy Generator.

    Inputs
    ------
    rng: np.random.Generator | None

    Outputs
    -------
    rng: np.random.Generator
    """
    if rng is not None:
        return rng
    return np.random.default_rng()

def _build_default_mode_registry() -> Dict[str, Any]:
    """
    Build the default mode registry by delegating to input_modes_core.
    """
    return input_modes_core.get_default_mode_registry()



# ---------------------------------------------------------------------
# 2.3.4 – Process all groups in synapse_groups
# ---------------------------------------------------------------------

def _process_all_groups(
    sim_cfg: Dict[str, Any],
    groups_cfg: Dict[str, Dict[str, Any]],
    geometry: Optional[Any],
    mode_registry: Mapping[str, Any],
    rng: np.random.Generator,
) -> Dict[str, GroupInputs]:
    """
    2.3.4 – Loop over all groups and generate per-group inputs.

    High-level steps (per group):
      2.3.4.1 – decide whether to skip the group (inactive / invalid)
      2.3.4.2 – resolve N_syn (including geometry/density when needed)
      2.3.4.3 – compute time window for this group
      2.3.4.4 – resolve the mode handler from mode_registry
      2.3.4.5 – run the handler to obtain spike trains
      2.3.4.6 – package into a GroupInputs object

    Returns
    -------
    inputs: dict[str, GroupInputs]
        Mapping from group name to the final per-group inputs structure.
    """
    inputs: Dict[str, GroupInputs] = {}

    for gname, gcfg in groups_cfg.items():
        # 2.3.4.1 – Skip inactive or invalid groups
        if _should_skip_group(gname, gcfg):
            continue

        # 2.3.4.2 – Resolve N_syn using geometry/density rules
        # This helper should:
        #   - compute N_syn_resolved according to the contract
        #   - write it into gcfg["syns"]["N_syn_resolved"]
        #   - return the resolved integer
        n_syn_resolved = _resolve_n_syn(
            sim_cfg=sim_cfg,
            group_cfg=gcfg,
            geometry=geometry,
        )

        # Basic sanity check (defensive)
        if n_syn_resolved < 0:
            raise ValueError(
                f"Group '{gname}': resolved N_syn_resolved < 0 ({n_syn_resolved})."
            )

        # 2.3.4.3 – Compute this group’s effective time window
        t_start_ms, t_end_ms = _get_group_time_window(sim_cfg, gcfg)

        # 2.3.4.4 – Resolve mode handler
        handler = _resolve_mode_handler(gname, gcfg, mode_registry)

        # 2.3.4.5 – Run handler to get spike trains
        # _run_mode_handler should:
        #   - call the handler(sim_cfg, gcfg, geometry, rng)
        #   - ensure the result is a list[np.ndarray]
        spike_trains = _run_mode_handler(
            handler=handler,
            sim_cfg=sim_cfg,
            group_name=gname,
            group_cfg=gcfg,
            geometry=geometry,
            rng=rng,
        )

        # Optional consistency check: number of trains vs N_syn_resolved
        if len(spike_trains) != n_syn_resolved:
            # For now, treat this as an error; if you later allow tiling/reuse,
            # you can relax this or handle it explicitly here.
            raise ValueError(
                f"Group '{gname}': handler returned {len(spike_trains)} trains, "
                f"but N_syn_resolved={n_syn_resolved}."
            )

        # 2.3.4.6 – Package into GroupInputs
        # You may need to update _build_group_inputs to accept t_window if it
        # doesn’t already.
        group_inputs = _build_group_inputs(
            group_name=gname,
            group_cfg=gcfg,
            spike_trains=spike_trains,
            t_window=(t_start_ms, t_end_ms),
        )

        inputs[gname] = group_inputs

    return inputs

def _should_skip_group(
    group_name: str,
    group_cfg: Dict[str, Any],
) -> bool:
    """
    2.3.4.1 – Decide whether to skip this group.

    Rules:
      - If state is explicitly in {False, 0, "off", None} → skip.
      - Otherwise, treat as active and require a valid mode.
    """
    state = group_cfg.get("state", True)
    if state in (False, 0, "off", None):
        return True

    # Active group must have a mode
    if group_cfg.get("mode") is None:
        raise ValueError(
            f"Group '{group_name}' is active (state={state!r}) "
            f"but has no 'mode' specified."
        )
    return False

def _resolve_mode_handler(
    group_name: str,
    group_cfg: Dict[str, Any],
    mode_registry: Mapping[str, Any],
) -> Any:
    """
    2.3.4.4 – Look up the mode handler for this group.

    - Reads group_cfg['mode'] (must be a string).
    - Looks up mode_registry[mode_name].
    - Raises a clear error if the mode is unknown.
    """
    mode_name = group_cfg.get("mode")
    if not isinstance(mode_name, str):
        raise ValueError(
            f"Group '{group_name}' has invalid mode {mode_name!r}; "
            f"'mode' must be a non-empty string."
        )

    handler = mode_registry.get(mode_name)
    if handler is None:
        raise ValueError(
            f"Group '{group_name}' specifies unknown mode {mode_name!r}; "
            f"make sure it is registered in mode_registry."
        )
    return handler

def _run_mode_handler(
    handler: Any,
    sim_cfg: Dict[str, Any],
    group_name: str,
    group_cfg: Dict[str, Any],
    geometry: Optional[Any],
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    2.3.4.5 – Call the mode handler to obtain spike trains.

    Contract:
      - handler(sim_cfg, group_cfg, geometry, rng) -> list[np.ndarray]
      - each array is 1D spike times in ms, in simulation time.
    """
    spike_trains = handler(sim_cfg, group_cfg, geometry, rng)

    if not isinstance(spike_trains, list):
        raise TypeError(
            f"Mode handler for group '{group_name}' must return a list of "
            f"np.ndarray, got {type(spike_trains)!r}"
        )

    # Optional: light type check on elements
    for i, arr in enumerate(spike_trains):
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"Mode handler for group '{group_name}' returned element {i} "
                f"of type {type(arr)!r}, expected np.ndarray."
            )

    return spike_trains

def _build_group_inputs(
    group_name: str,
    group_cfg: Dict[str, Any],
    spike_trains: List[np.ndarray],
    t_window: Tuple[float, float],
) -> GroupInputs:
    """
    2.3.4.6 – Package per-group data into a GroupInputs object.

    Stores:
      - name
      - mode
      - spike_trains
      - meta: cfg, t_window, N_syn_resolved, etc.
    """
    mode = group_cfg.get("mode", "unknown")
    syn_cfg = group_cfg.get("syns", {})
    n_syn_resolved = syn_cfg.get("N_syn_resolved", len(spike_trains))

    meta: Dict[str, Any] = {
        "cfg": group_cfg,
        "t_window": t_window,
        "N_syn": n_syn_resolved,
    }

    return GroupInputs(
        name=group_name,
        mode=mode,
        spike_trains=spike_trains,
        meta=meta,
    )



# ---------------------------------------------------------------------
# 2.3.5 – Process all groups in synapse_groups
# ---------------------------------------------------------------------

def _finalize_inputs(
    sim_cfg: Dict[str, Any],
    groups_cfg: Dict[str, Dict[str, Any]],
    inputs: Dict[str, GroupInputs],
) -> None:
    """
    2.3.5 – Final sanity checks / adjustments on assembled inputs.

    Big picture: make sure every active group has a GroupInputs entry and
    that spike times (if any) live inside the global simulation window.

    Inputs
    ------
    sim_cfg: dict
    groups_cfg: dict[str, dict]
    inputs: dict[str, Any]

    Outputs
    -------
    None
        May raise if inconsistencies are detected; otherwise does nothing.
    """
    
    tstart = float(sim_cfg["tstart"])
    tstop = float(sim_cfg["tstop"])

    for gname, gcfg in groups_cfg.items():
        state = gcfg.get("state", True)
        if not state:
            # Inactive groups are allowed to be absent from 'inputs'
            continue

        if gname not in inputs:
            raise ValueError(
                f"Active group '{gname}' has no entry in inputs; "
                f"check _process_all_groups implementation."
            )

        gin = inputs[gname]
        # Optional: basic consistency check on mode name
        if gin.mode != gcfg.get("mode"):
            raise ValueError(
                f"Group '{gname}': mode mismatch between config ({gcfg.get('mode')!r}) "
                f"and inputs ({gin.mode!r})."
            )

        # Optional: check spike times lie within [tstart, tstop]
        for idx, train in enumerate(gin.spike_trains):
            if train.size == 0:
                continue
            if train.min() < tstart - 1e-9 or train.max() > tstop + 1e-9:
                raise ValueError(
                    f"Group '{gname}', train {idx}: spike times must be within "
                    f"[{tstart}, {tstop}] ms (got min={float(train.min()):.3f}, "
                    f"max={float(train.max()):.3f})."
                )

    # If all checks pass, we don't need to modify anything
    return
