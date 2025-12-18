from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, List, Union


import json
import math
import numpy as np
import hashlib

from dataclasses import dataclass, field

from modules_local import input_modes_core  # or: from . import input_modes_core
from modules_local import randomness


# ===================================================================
# Core data structure
# ===================================================================

@dataclass
class GroupInputs:
    """
    Final per-group inputs structure produced by generate_inputs and
    consumed by the synapse-building step (2.4).

    For now we keep it minimal: name, mode, and spike trains (in ms,
    in simulation time). meta can hold timing and other snapshot info.
    """
    name: str
    mode: str
    spike_trains: List[np.ndarray] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


# ====================================================================
# 2.3 Top-level API
# ====================================================================

from pathlib import Path
from typing import Any, Dict, Optional, Union, Mapping, Tuple

import json
import numpy as np

# assuming GroupInputs, _normalize_sim_config, _normalize_group_configs,
# _init_rng, _build_default_mode_registry, _process_all_groups,
# and _finalize_inputs are already defined above.


def generate_inputs(
    path: Optional[Union[Path, str]] = None,
    geometry: Optional[Any] = None,
    mode_registry: Optional[Mapping[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
    seed_override: Optional[int] = None,
    trial_rng: Optional[randomness.TrialRandomness] = None,
    # cache: Optional[Dict[str, "GroupInputs"]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, "GroupInputs"]]:
    """
    Step 2.3 main entry point: load, normalize, and materialize synaptic inputs.

    This function assumes the new split-config layout:
      - sim_config.json: simulation-level settings (dt, tstart, tstop, seed, etc.)
      - syn_config.json: per-group input/synapse configs (one top-level key per group)

    Path resolution:
      - If `path` is None:
          Use the current working directory as the tune directory.
          Expect `sim_config.json` and `syn_config.json` there.
      - If `path` is a directory:
          Treat it as the tune directory. Expect `sim_config.json` and `syn_config.json` inside.
      - If `path` is a file:
          Treat it as `syn_config.json` and use its parent directory as the tune directory.
          Expect `sim_config.json` next to it.

    Parameters
    ----------
    path
        Optional directory or file path controlling where configs are loaded from.
        See rules above. Typical notebook usage is `path=None` with the tune
        directory as the working directory.
    geometry
        Geometry / segment-group information from Step 2.2 (may be None if not used).
    rng
        Optional NumPy random number generator. If None, a new generator is created
        based on the seed in `sim_config.json`.
    mode_registry
        Optional mapping from mode names (str) to handler callables. If None,
        a default registry for core modes is constructed.
    cache
        Optional dictionary for caching `GroupInputs` objects keyed by a stable
        signature of (sim_cfg, group_cfg). If provided, per-group inputs may be
        reused instead of regenerated.

    Returns
    -------
    sim_cfg : dict
        Normalized simulation-level config.
    groups_cfg : dict[str, dict]
        Normalized per-group configs (one entry per group name).
    inputs_by_group : dict[str, GroupInputs]
        Materialized per-group input objects, one entry per active synapse group.
    """
    # ------------------------------------------------------------------
    # Resolve the tune directory and config file paths
    # ------------------------------------------------------------------
    if path is None:
        root = Path.cwd()
    else:
        p = Path(path)
        if p.is_dir():
            root = p
        else:
            root = p.parent

    syn_path = root / "syn_config.json"
    sim_path = root / "sim_config.json"

    if not syn_path.is_file():
        raise FileNotFoundError(f"Missing syn_config.json in {root}")
    if not sim_path.is_file():
        raise FileNotFoundError(f"Missing sim_config.json in {root}")

    # ------------------------------------------------------------------
    # Load raw JSON configs
    # ------------------------------------------------------------------
    with sim_path.open("r") as f:
        sim_cfg_raw = json.load(f)
    with syn_path.open("r") as f:
        groups_cfg_raw = json.load(f)

    # ------------------------------------------------------------------
    # 2.3.2 – Normalize configs
    # ------------------------------------------------------------------
    sim_cfg = _normalize_sim_config(sim_cfg_raw)
    groups_cfg = _normalize_group_configs(groups_cfg_raw)

    # ------------------------------------------------------------------
    # 2.3.3 – Shared resources: RNG and mode registry
    # ------------------------------------------------------------------
    if seed_override is not None:
        sim_cfg["seed"] = int(seed_override)

    base_rng = None if trial_rng is not None else _init_rng(rng, sim_cfg)
    if mode_registry is None:
        mode_registry = _build_default_mode_registry()

    # ------------------------------------------------------------------
    # 2.3.4 – Per-group processing
    # ------------------------------------------------------------------
    inputs_by_group = _process_all_groups(
        sim_cfg=sim_cfg,
        groups_cfg=groups_cfg,
        geometry=geometry,
        mode_registry=mode_registry,
        rng=base_rng,
        trial_rng=trial_rng,
        # cache=cache,
    )

    # ------------------------------------------------------------------
    # 2.3.5 – Final sanity checks
    # ------------------------------------------------------------------
    _finalize_inputs(sim_cfg, groups_cfg, inputs_by_group)

    return sim_cfg, groups_cfg, inputs_by_group

# ====================================================================
# Pre-2.3 preview helper (optional, not called by generate_inputs)
# ====================================================================


def check_inputs(
    path: Union[str, Path, None] = None,
    *,
    verbose: bool = True,
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Lightweight pre-2.3 sanity check.

    - Loads config using the same rules as generate_inputs
      (split or combined layout)
    - Runs the same normalization as generate_inputs
    - Prints a concise summary of sim + per-group configs
    If path is None: assume current working directory and look there.
    If path is a file: treat that as syn_config.json and look for sim_config.json next to it.
    If path is a directory: look for syn_config.json and sim_config.json inside it.
    """

    if path is None:
        root = Path.cwd()
    else:
        p = Path(path)
        if p.is_dir():
            root = p
        else:
            # file: use its parent as root
            root = p.parent

    syn_path = root / "syn_config.json"
    sim_path = root / "sim_config.json"

    if not syn_path.is_file():
        raise FileNotFoundError(f"Missing syn_config.json in {root}")
    if not sim_path.is_file():
        raise FileNotFoundError(f"Missing sim_config.json in {root}")

    with sim_path.open("r") as f:
        sim_raw = json.load(f)
    with syn_path.open("r") as f:
        groups_raw = json.load(f)

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
                f"  - {gname:<12}  state={state!r:5}  mode={mode!r:<18}  "
                f"source.path={repr(src_path) if src_path is not None else 'None':<30}  "
                f"N_syn={n_syn}"
            )

    return sim_cfg, groups_cfg


# ====================================================================
# 2.3.2 – Configure all groups
# ====================================================================

def _normalize_sim_config(sim_cfg_raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize/validate simulation-level config.

    Ensures keys:
      - cell (string, optional)
      - tune (string, optional)
      - dt, tstart, tstop (floats)
      - jitter (float or None)
      - seed (int or None)
      - n_trials (int >= 1)
      - trial_randomness (one of 'inputs', 'synapses', 'both', 'none')
      - n_traces_to_save (int >= 0)
      - output (string or None)
      - output_format ('npz' or 'pkl')
      - load (string path or None)          # <-- add this line
      - param_study (dict with standard keys)
    """
    sim_cfg = dict(sim_cfg_raw)

    # Required numeric fields
    for key in ("dt", "tstart", "tstop"):
        if key not in sim_cfg:
            raise ValueError(f"sim config missing required key {key!r}")
        try:
            sim_cfg[key] = float(sim_cfg[key])
        except Exception as exc:
            raise ValueError(
                f"sim[{key!r}] must be convertible to float (got {sim_cfg[key]!r})"
            ) from exc

    # jitter is optional
    jitter = sim_cfg.get("jitter", None)
    if jitter is None:
        sim_cfg["jitter"] = None
    else:
        sim_cfg["jitter"] = float(jitter)

    # Cell and tune labels are just passed through if present
    for key in ("cell", "tune"):
        if key in sim_cfg and sim_cfg[key] is not None:
            sim_cfg[key] = str(sim_cfg[key])

    # seed is optional
    seed_raw = sim_cfg.get("seed", None)
    if seed_raw is None:
        sim_cfg["seed"] = None
    else:
        try:
            sim_cfg["seed"] = int(seed_raw)
        except Exception as exc:
            raise ValueError(
                f"sim['seed'] must be integer-like or null (got {seed_raw!r})"
            ) from exc

    # n_trials: optional, default 1
    n_trials_raw = sim_cfg.get("n_trials", 1)
    try:
        n_trials = int(n_trials_raw)
    except Exception as exc:
        raise ValueError(
            f"sim['n_trials'] must be integer-like (got {n_trials_raw!r})"
        ) from exc
    if n_trials < 1:
        raise ValueError("sim['n_trials'] must be >= 1")
    sim_cfg["n_trials"] = n_trials

    # trial_randomness: optional, default 'synapses'
    tr = sim_cfg.get("trial_randomness", "synapses")
    if tr is None:
        tr = "synapses"
    tr = str(tr)
    allowed_tr = {"inputs", "synapses", "both", "none"}
    if tr not in allowed_tr:
        raise ValueError(
            f"sim['trial_randomness'] must be one of {sorted(allowed_tr)} (got {tr!r})"
        )
    sim_cfg["trial_randomness"] = tr

    # n_traces_to_save: optional, default 1
    n_traces_raw = sim_cfg.get("n_traces_to_save", 1)
    try:
        n_traces = int(n_traces_raw)
    except Exception as exc:
        raise ValueError(
            f"sim['n_traces_to_save'] must be integer-like (got {n_traces_raw!r})"
        ) from exc
    if n_traces < 0:
        raise ValueError("sim['n_traces_to_save'] must be >= 0")
    sim_cfg["n_traces_to_save"] = n_traces

    # output + output_format
    if "output" in sim_cfg and sim_cfg["output"] is not None:
        sim_cfg["output"] = str(sim_cfg["output"])
    else:
        sim_cfg.setdefault("output", None)

    ofmt = sim_cfg.get("output_format", "pkl")
    if ofmt is None:
        ofmt = "pkl"
    ofmt = str(ofmt)
    if ofmt not in {"npz", "pkl"}:
        raise ValueError(
            f"sim['output_format'] must be 'npz' or 'pkl' (got {ofmt!r})"
        )
    sim_cfg["output_format"] = ofmt

    # param_study: optional
    param_raw = sim_cfg.get("param_study", None)
    if param_raw is None:
        sim_cfg["param_study"] = {
            "input_type": None,
            "param_type": None,
            "param_vals": [],
            "n_trials": None,
        }
    else:
        if not isinstance(param_raw, dict):
            raise TypeError("sim['param_study'] must be a dict or null")
        param = dict(param_raw)
        param.setdefault("input_type", None)
        param.setdefault("param_type", None)
        vals = param.get("param_vals", [])
        if vals is None:
            vals = []
        if not isinstance(vals, list):
            vals = list(vals)
        param["param_vals"] = vals
        if "n_trials" in param and param["n_trials"] is not None:
            try:
                param["n_trials"] = int(param["n_trials"])
            except Exception as exc:
                raise ValueError(
                    "param_study['n_trials'] must be integer-like or null"
                ) from exc
        sim_cfg["param_study"] = param

    # --- NEW: normalize 'load' ---
    load_raw = sim_cfg.get("load", None)
    if load_raw in (None, "", False):
        sim_cfg["load"] = None        # meaning: run a new simulation
    else:
        sim_cfg["load"] = str(load_raw)  # meaning: treat as file/path to load

    return sim_cfg


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
            raise TypeError(
                f"Each synapse group config must be a dict (got {type(gcfg_raw)!r}) "
                f"for group {gname!r}"
            )

        gcfg = dict(gcfg_raw)

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


def _init_rng(
    rng: Optional[np.random.Generator],
    sim_cfg: Dict[str, Any],
) -> np.random.Generator:
    """
    Initialize RNG.

    Priority:
      1) If an explicit Generator is passed in, use it as-is.
      2) Else, if sim_cfg['seed'] is not None, use it to seed a new Generator.
      3) Else, create an unseeded Generator (non-deterministic).
    """
    if rng is not None:
        return rng

    seed = sim_cfg.get("seed", None)
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


# NOTE: The cache key currently ignores cell geometry. This is fine as long
# as each run uses a single morphology, but if we ever reuse the same
# synapse config across different geometries, the cache must be extended
# to include a geometry identifier to avoid reusing mismatched inputs.
def _make_group_signature(
    sim_cfg: Dict[str, Any],
    group_name: str,
    group_cfg: Dict[str, Any],
) -> str:
    """
    Build a stable signature string for a (sim_cfg, group_cfg) pair.

    Used as a key into an inputs cache so that if the sim/group
    parameters are unchanged, we can reuse previously generated
    spike trains instead of regenerating.

    Notes:
      - We include only JSON-serializable pieces; anything non-serializable
        is converted via str(...).
      - Geometry is deliberately NOT included here; if you want geometry-
        specific caching later, you can add a geometry identifier.
    """
    # Keep only the sim fields that actually affect inputs
    sim_subset = {
        "tstart": sim_cfg.get("tstart"),
        "tstop": sim_cfg.get("tstop"),
        "dt": sim_cfg.get("dt"),
        "seed": sim_cfg.get("seed"),
        "jitter": sim_cfg.get("jitter"),
    }

    payload = {
        "sim": sim_subset,
        "group_name": group_name,
        "group_cfg": group_cfg,
    }

    # JSON with sorted keys for stability; default=str to survive odd types
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _build_default_mode_registry() -> Dict[str, Any]:
    """
    Build the default mode registry from the core module.

    This can be extended by user-defined modes at the notebook level.
    """
    return input_modes_core.get_default_mode_registry()


# ====================================================================
# 2.3.3 – Per-group processing
# ====================================================================

def _process_all_groups(
    sim_cfg: Dict[str, Any],
    groups_cfg: Dict[str, Dict[str, Any]],
    geometry: Optional[Any],
    mode_registry: Mapping[str, Any],
    rng: Optional[np.random.Generator],
    trial_rng: Optional[randomness.TrialRandomness] = None,
    # cache: Optional[Dict[str, GroupInputs]] = None,
) -> Dict[str, GroupInputs]:
    """
    Loop over all groups and generate per-group inputs.

    High-level steps (per group):
      1) decide whether to skip the group (inactive / invalid)
      2) resolve N_syn (including geometry/density when needed)
      3) compute timing configuration for this group
      4) resolve the mode handler from mode_registry
      5) run the handler to obtain spike trains (or reuse from cache)
      6) package into GroupInputs
    """
    inputs: Dict[str, GroupInputs] = {}

    for gname, gcfg in groups_cfg.items():
        # 1) Skip inactive or invalid groups
        if _should_skip_group(gname, gcfg):
            continue

        # OPTIONAL: cache lookup before doing any heavy work
        # sig: Optional[str] = None
        # if cache is not None:
        #     sig = _make_group_signature(sim_cfg, gname, gcfg)
        #     cached = cache.get(sig)
        #     if cached is not None:
        #         inputs[gname] = cached
        #         continue

        # 2) Resolve number of synapses for this group
        #    This also stashes syns["N_syn_resolved"] into the group config.
        n_syn_resolved = _resolve_n_syn(
            sim_cfg=sim_cfg,
            group_cfg=gcfg,
            geometry=geometry,
        )
        if n_syn_resolved < 0:
            raise ValueError(
                f"Group '{gname}': resolved N_syn_resolved < 0 ({n_syn_resolved})."
            )

        # 3) Generate timing configuration for this group
        time_cfg = _calculate_timing(sim_cfg, gcfg)
        gcfg["time_cfg"] = time_cfg  # Attach for modes and downstream consumers

        # 4) Resolve mode handler
        handler = _resolve_mode_handler(gname, gcfg, mode_registry)

        # 5) Choose RNG for this group and run handler to get spike trains
        mode_name = gcfg.get("mode")
        setting_path = mode_name if mode_name else "inputs"
        group_rng = rng
        rand_meta = None
        if trial_rng is not None:
            group_rng = trial_rng.rng(
                setting_path,
                group=gname,
                stream=mode_name or "inputs",
            )
            rand_meta = {
                "trial_idx": getattr(trial_rng, "trial_idx", None),
                "setting_path": setting_path,
                "group": gname,
                "stream": mode_name or "inputs",
            }
        elif group_rng is None:
            group_rng = _init_rng(None, sim_cfg)

        spike_trains = _run_mode_handler(
            handler=handler,
            sim_cfg=sim_cfg,
            group_name=gname,
            group_cfg=gcfg,
            geometry=geometry,
            rng=group_rng,
        )

        # Optional consistency check: number of trains vs N_syn_resolved
        if len(spike_trains) != n_syn_resolved:
            raise ValueError(
                f"Group '{gname}': handler returned {len(spike_trains)} trains, "
                f"but N_syn_resolved={n_syn_resolved}."
            )

        # 6) Package into GroupInputs
        group_inputs = _build_group_inputs(
            group_name=gname,
            group_cfg=gcfg,
            spike_trains=spike_trains,
            randomness_meta=rand_meta,
        )

        inputs[gname] = group_inputs

        # Store into cache if enabled
        # if cache is not None and sig is not None:
        #     cache[sig] = group_inputs

    return inputs


# ---------------------------------------------------------------------
# 2.3.3 Helper functions
# ---------------------------------------------------------------------

def _should_skip_group(
    group_name: str,
    group_cfg: Dict[str, Any],
) -> bool:
    """
    Decide whether to skip this group.

    Rules:
      - If group_cfg['state'] is explicitly False, skip it;
      - If 'mode' is missing or empty, skip it;
      - Otherwise, keep it.
    """
    state = group_cfg.get("state", True)
    if state is False:
        return True

    mode = group_cfg.get("mode")
    if mode is None or (isinstance(mode, str) and mode.strip() == ""):
        return True

    return False


def _compile_density_from_spec(dist_spec: Any):
    """Convert a 'dist_func' spec from JSON into a callable density function.

    The returned function dens(dist_um) should yield synapses-per-µm at that distance.

    Supported forms:
      - None          → dens(d) = 1.0
      - number        → dens(d) = const
      - callable      → dens(d) used as-is
      - dict          → {"kind": "uniform", "params": {"c": float, "multi": optional}}
    """
    # None → uniform density 1.0
    if dist_spec is None:
        return lambda d: 1.0

    # Already a callable or a simple numeric constant
    if callable(dist_spec):
        return dist_spec
    if isinstance(dist_spec, (int, float)):
        const = float(dist_spec)
        return lambda d, c=const: c

    # JSON-style spec
    if isinstance(dist_spec, dict):
        kind = dist_spec.get("kind") or "uniform"
        params = dist_spec.get("params", {}) or {}

        if kind == "uniform":
            c = float(params.get("c", 1.0))
            multi = float(params.get("multi", 1.0))
            const = c * multi
            return lambda d, c=const: c

        # Placeholder for future shapes (linear, gaussian, etc.)
        raise ValueError(f"dist_func spec with kind={kind!r} is not yet supported for N_syn resolution")

    raise TypeError(
        f"dist_func must be None, number, callable, or dict-spec; got {type(dist_spec)!r}"
    )


def _resolve_n_syn(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
    geometry: Optional[Any],
) -> int:
    """Resolve the effective N_syn for a group, including geometry/density.

    Logic:
      1. If syns["N_syn"] is an explicit integer ≥ 0, use it.
      2. If N_syn is None:
         - Require geometry and a valid dist_func spec.
         - Use the selected geometry group (from syns["segs"]) and the
           density function to compute a deterministic synapse count via:

               n_seg = floor( density(dist_um) * seg_length_um )

           summed over all segments.

    Geometry contract (from Step 2.2):
      - geometry["groups"][<name>] must be a list of segment references
        each with:
          * .sec     → NEURON Section
          * .dist_um → distance from soma in µm
        and the Section supplies L and nseg.
    """
    syns = group_cfg.get("syns", {}) or {}
    n_syn_raw = syns.get("N_syn", None)

    # Case 1: explicit N_syn
    if n_syn_raw is not None:
        try:
            n_syn = int(n_syn_raw)
        except Exception as exc:  # pragma: no cover
            raise ValueError(
                f"Group {group_cfg.get('name', '<unnamed>')!r}: syns['N_syn'] must be int or None, "
                f"got {n_syn_raw!r} of type {type(n_syn_raw)!r}."
            ) from exc
        if n_syn < 0:
            raise ValueError(
                f"Group {group_cfg.get('name', '<unnamed>')!r}: syns['N_syn'] must be ≥ 0, got {n_syn}."
            )
        # Stash resolved value for downstream consumers (e.g. 2.4)
        syns["N_syn_resolved"] = n_syn
        group_cfg["syns"] = syns
        return n_syn

    # Case 2: N_syn is None → geometry/density-based count
    if geometry is None:
        raise ValueError(
            "_resolve_n_syn: geometry is required when syns['N_syn'] is None; "
            f"group={group_cfg.get('name', '<unnamed>')!r}"
        )

    # Select segment group from geometry based on syns['segs']
    segs_key = syns.get("segs") or "all"
    group_map = {
        "all": "all_dend",
        "proximal": "proximal",
        "distal": "distal",
        "soma": "soma",
    }
    geom_group_name = group_map.get(segs_key)
    if geom_group_name is None:
        raise ValueError(
            f"_resolve_n_syn: unknown segs selector {segs_key!r} for group {group_cfg.get('name', '<unnamed>')!r}."
        )

    geom_groups = geometry.get("groups", {})
    seg_refs = geom_groups.get(geom_group_name, [])
    if not seg_refs:
        # No segments available in this geometry group
        syns["N_syn_resolved"] = 0
        group_cfg["syns"] = syns
        return 0

    # Build density function
    dens_eq = _compile_density_from_spec(syns.get("dist_func"))

    # Deterministic density-based count, mirroring _gen_distr_synlocs
    total_n_syn = 0
    for ref in seg_refs:
        sec = ref.sec
        seg_len = float(sec.L) / float(sec.nseg or 1)
        dens = float(dens_eq(ref.dist_um))
        if dens <= 0.0:
            continue
        n_seg = math.floor(dens * seg_len)
        if n_seg <= 0:
            continue
        total_n_syn += n_seg

    if total_n_syn < 0:
        raise ValueError(
            f"_resolve_n_syn: computed negative synapse count ({total_n_syn}) for group "
            f"{group_cfg.get('name', '<unnamed>')!r}."
        )

    syns["N_syn_resolved"] = int(total_n_syn)
    group_cfg["syns"] = syns
    return int(total_n_syn)


# ====================================================================
# 2.3.3.3 – Timing configuration
# ====================================================================

def _calculate_timing(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute resolved timing information for a single synapse group.

    High-level:
      1) Derive concrete time anchors (ms) for this group from sim_cfg
         and group_cfg (tstart/tstop, onset, source start/stop, baseline).
      2) Build an ordered, non-overlapping list of time blocks of three kinds:
           - "quiescent" : no spikes should be generated.
           - "baseline"  : constant-rate (homogeneous) baseline spikes.
           - "source"    : main source-driven spikes (mode-specific).

    This function is mode-agnostic: it does not generate any spikes and
    does not depend on which mode (homogeneous, precomputed, etc.) is used.
    """
    anchors = _compute_time_anchors(sim_cfg, group_cfg)
    blocks = _build_time_blocks_from_anchors(anchors)

    time_cfg = {
        "anchors": anchors,
        "blocks": blocks,
    }

    return time_cfg


def _compute_time_anchors(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    """
    Derive concrete time anchors for this group.

    Output keys (all in ms, floats or None where applicable):
        "sim_tstart"      : simulation start time (sim_cfg["tstart"])
        "sim_tstop"       : simulation stop time  (sim_cfg["tstop"])
        "onset"           : first time this group is allowed to produce spikes
                           (defaults to sim_tstart).
        "source_tstart"   : start of main source-driven input segment
                           (aligned to stim_tstart_ms and input_stim_tstart_ms).
        "source_tstop"    : end of main source-driven input segment.
        "baseline_rate_hz": resolved baseline rate in Hz for this group,
                           or None if there is no baseline.
    """
    tstart = float(sim_cfg["tstart"])
    tstop = float(sim_cfg["tstop"])

    timing = group_cfg.get("timing", {}) or {}
    source = group_cfg.get("source", {}) or {}

    def _as_float_or_none(val: Any, label: str) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be numeric or null (got {val!r})") from exc

    onset_raw = _as_float_or_none(timing.get("onset_ms"), "timing['onset_ms']")
    stim_tstart_raw = _as_float_or_none(timing.get("stim_tstart_ms"), "timing['stim_tstart_ms']")
    duration_raw = _as_float_or_none(timing.get("duration_ms"), "timing['duration_ms']")
    input_stim_raw = _as_float_or_none(
        timing.get("input_stim_tstart_ms"), "timing['input_stim_tstart_ms']"
    )


    def _parse_baseline_spec(
        val: Any,
        label: str = "source['baseline']",
    ) -> Tuple[Dict[str, Any], Optional[float]]:
        """
        Parse baseline specification into:
        - baseline_spec: rich descriptor dict
        - baseline_hz: numeric baseline (float) or None

        baseline_hz is only non-None for simple fixed numeric baselines.
        All tokenized / curve-dependent cases return baseline_hz = None.
        """
        # Null → no baseline
        if val is None:
            return {"kind": "none"}, None

        # Plain numeric → fixed Hz
        if isinstance(val, (int, float)):
            hz = float(val)
            return {"kind": "fixed", "hz": hz}, hz

        # String tokens
        if isinstance(val, str):
            v = val.strip()

            if v == "start":
                return {"kind": "from_curve", "where": "start"}, None

            if v == "end":
                return {"kind": "from_curve", "where": "end"}, None

            if v == "peak":
                return {"kind": "from_curve", "where": "peak"}, None

            if v.startswith("time:"):
                # e.g. "time: 234" → 234 ms
                _, _, num_str = v.partition(":")
                try:
                    t_ms = float(num_str.strip())
                except ValueError as exc:
                    raise ValueError(f"{label} malformed time spec {val!r}") from exc
                return {"kind": "from_curve_at", "t_ms": t_ms}, None

        # Anything else is invalid
        raise ValueError(
            f"{label} must be numeric, null, or a recognized string spec (got {val!r})"
        )

    baseline_spec, baseline_hz = _parse_baseline_spec(
        source.get("baseline"), "source['baseline']"
    )   
     
    # Onset: default to sim start, clamped into [tstart, tstop]
    if onset_raw is None:
        onset = tstart
    else:
        onset = max(min(onset_raw, tstop), tstart)

    # stim_tstart: if not specified, and we have a source stim delay,
    # default it to onset; otherwise leave None.
    if stim_tstart_raw is None:
        stim_tstart = onset if input_stim_raw is not None else None
    else:
        stim_tstart = max(min(stim_tstart_raw, tstop), tstart)

    # Source start: align source data so that its own stim event occurs
    # at stim_tstart in the simulation.
    source_tstart: Optional[float] = None
    if input_stim_raw is not None:
        # If stim_tstart is still None here, fall back to onset as a last resort
        if stim_tstart is None:
            stim_tstart = onset
        source_tstart = stim_tstart - input_stim_raw

    # Clamp source_tstart into the simulation window (but do not force it
    # to be >= onset; that interaction is handled by the block builder).
    if source_tstart is not None:
        if source_tstart < tstart:
            source_tstart = tstart
        if source_tstart > tstop:
            source_tstart = tstop

    # Source stop: choose a duration for the main source segment.
    source_tstop: Optional[float] = None
    if source_tstart is not None:
        remaining = max(0.0, tstop - source_tstart)
        if remaining <= 0.0:
            source_tstop = source_tstart
        else:
            if duration_raw is not None:
                dur = max(0.0, duration_raw)
                dur = min(dur, remaining)
            else:
                # No explicit duration → run until the end of the simulation window
                dur = remaining

            if dur <= 0.0:
                source_tstop = source_tstart
            else:
                source_tstop = source_tstart + dur

    anchors: Dict[str, Optional[float]] = {
        "sim_tstart": tstart,
        "sim_tstop": tstop,
        "onset": onset,
        "source_tstart": source_tstart,
        "source_tstop": source_tstop,
        "baseline_rate_hz": baseline_hz, #baseline_raw,
        # Extra fields for debugging / inspection (not used by block builder):
        "stim_tstart_ms": stim_tstart,
        "input_stim_tstart_ms": input_stim_raw,
        "duration_ms": duration_raw,
        "input_duration_ms": _as_float_or_none(timing.get("input_duration_ms"), "timing['input_duration_ms']"),
    }

    return anchors


def _build_time_blocks_from_anchors(
    anchors: Dict[str, Optional[float]],
) -> List[Dict[str, Any]]:
    """
    Build an ordered, non-overlapping list of time blocks from anchors.

    Inputs (anchors):
        sim_tstart, sim_tstop, onset,
        source_tstart, source_tstop, baseline_rate_hz
    """
    sim_tstart = float(anchors.get("sim_tstart", 0.0))
    sim_tstop = float(anchors.get("sim_tstop", sim_tstart))
    onset = anchors.get("onset")
    source_tstart = anchors.get("source_tstart")
    source_tstop = anchors.get("source_tstop")
    baseline_rate = anchors.get("baseline_rate_hz")

    # Clamp helpers
    def _clamp(t: Optional[float]) -> Optional[float]:
        if t is None:
            return None
        t_f = float(t)
        if t_f < sim_tstart:
            return sim_tstart
        if t_f > sim_tstop:
            return sim_tstop
        return t_f

    onset = _clamp(onset) or sim_tstart
    source_tstart = _clamp(source_tstart)
    source_tstop = _clamp(source_tstop)

    blocks: List[Dict[str, Any]] = []

    def _add_block(kind: str, t0: float, t1: float) -> None:
        if t1 <= t0:
            return
        blocks.append({"kind": kind, "t_start": float(t0), "t_end": float(t1)})

    # Case: no valid source window → only quiescent + baseline/quiescent
    if source_tstart is None or source_tstop is None or source_tstop <= source_tstart:
        _add_block("quiescent", sim_tstart, onset)
        kind = "baseline" if (baseline_rate is not None and baseline_rate > 0.0) else "quiescent"
        _add_block(kind, onset, sim_tstop)
        return blocks

    # Ensure ordering for source anchors
    src_start = max(min(source_tstart, sim_tstop), sim_tstart)
    src_stop = max(min(source_tstop, sim_tstop), sim_tstart)
    if src_stop <= src_start:
        _add_block("quiescent", sim_tstart, onset)
        kind = "baseline" if (baseline_rate is not None and baseline_rate > 0.0) else "quiescent"
        _add_block(kind, onset, sim_tstop)
        return blocks

    # 1) steady-state / quiescent before onset
    _add_block("quiescent", sim_tstart, min(onset, src_start))

    # 2) pre-source baseline or quiescent
    pre_start = max(onset, sim_tstart)
    pre_end = min(src_start, sim_tstop)
    pre_kind = "baseline" if (baseline_rate is not None and baseline_rate > 0.0) else "quiescent"
    _add_block(pre_kind, pre_start, pre_end)

    # 3) main source window
    _add_block("source", src_start, src_stop)

    # 4) post-source baseline or quiescent
    post_start = max(src_stop, sim_tstart)
    post_end = sim_tstop
    post_kind = "baseline" if (baseline_rate is not None and baseline_rate > 0.0) else "quiescent"
    _add_block(post_kind, post_start, post_end)

    return blocks


# ====================================================================
# 2.3.3.4 – Mode resolution and execution
# ====================================================================

def _resolve_mode_handler(
    group_name: str,
    group_cfg: Dict[str, Any],
    mode_registry: Mapping[str, Any],
) -> Any:
    """
    Look up the mode handler for this group.

    - Reads group_cfg['mode'] (must be a string).
    - Looks up mode_registry[mode_name].
    - Raises a clear error if the mode is unknown.
    """
    mode_name = group_cfg.get("mode")
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
    Call the mode handler to obtain spike trains.

    Contract:
      - handler(sim_cfg, group_cfg, geometry, rng) -> list[np.ndarray]
      - each array is 1D spike times in ms, in simulation time.

    Note:
      - group_cfg may contain a 'time_cfg' key if timing has been
        resolved upstream; modes are encouraged to use it instead of
        reinterpreting the raw timing fields.
    """
    spike_trains = handler(sim_cfg, group_cfg, geometry, rng)

    if not isinstance(spike_trains, list):
        raise TypeError(
            f"Mode handler for group '{group_name}' must return a list of np.ndarray, "
            f"got {type(spike_trains)!r}"
        )

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
    randomness_meta: Optional[Dict[str, Any]] = None,
) -> GroupInputs:
    """
    Package per-group data into a GroupInputs object.

    Stores:
      - name
      - mode
      - spike_trains
      - meta: timing anchors, resolved N_syn, and any other useful snapshot info.
    """
    mode = group_cfg.get("mode", "unknown")
    meta: Dict[str, Any] = {}

    # Stash resolved N_syn and timing info if available
    syns = group_cfg.get("syns", {}) or {}
    if "N_syn_resolved" in syns:
        meta["N_syn_resolved"] = int(syns["N_syn_resolved"])

    time_cfg = group_cfg.get("time_cfg")
    if isinstance(time_cfg, dict):
        anchors = time_cfg.get("anchors", {}) or {}
        meta["time_anchors_ms"] = {
            "sim_tstart": float(anchors["sim_tstart"]) if "sim_tstart" in anchors and anchors["sim_tstart"] is not None else None,
            "sim_tstop": float(anchors["sim_tstop"]) if "sim_tstop" in anchors and anchors["sim_tstop"] is not None else None,
            "onset": anchors.get("onset"),
            "source_tstart": anchors.get("source_tstart"),
            "source_tstop": anchors.get("source_tstop"),
            "baseline_rate_hz": anchors.get("baseline_rate_hz"),
        }
        meta["time_blocks"] = time_cfg.get("blocks", [])

    if randomness_meta:
        meta["randomness"] = randomness_meta

    return GroupInputs(
        name=group_name,
        mode=mode,
        spike_trains=spike_trains,
        meta=meta,
    )


# ====================================================================
# 2.3.4 – Final sanity checks
# ====================================================================

def _finalize_inputs(
    sim_cfg: Dict[str, Any],
    groups_cfg: Dict[str, Dict[str, Any]],
    inputs: Dict[str, GroupInputs],
) -> None:
    """
    Final sanity checks / adjustments on assembled inputs.

    Big picture: make sure every active group has a GroupInputs entry and
    that spike times (if any) live inside the global simulation window.
    """
    tstart = float(sim_cfg.get("tstart", 0.0))
    tstop = float(sim_cfg.get("tstop", 0.0))

    for gname, gcfg in groups_cfg.items():
        if _should_skip_group(gname, gcfg):
            continue

        if gname not in inputs:
            raise ValueError(
                f"Active synapse group '{gname}' has no generated inputs. "
                "Check its 'mode', 'source', and mode handler implementation."
            )

        gin = inputs[gname]

        # Check spike times lie within [tstart, tstop]
        for idx, train in enumerate(gin.spike_trains):
            if train.size == 0:
                continue
            if train.min() < tstart - 1e-9 or train.max() > tstop + 1e-9:
                raise ValueError(
                    f"Group '{gname}', train {idx}: spike times must be within "
                    f"[{tstart}, {tstop}] ms (got min={float(train.min()):.3f}, "
                    f"max={float(train.max()):.3f})."
                )

    return
