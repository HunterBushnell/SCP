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


def _resolve_config_root(path: Optional[Union[Path, str]] = None) -> Path:
    """
    Resolve where sim_config.json and syn_config.json live.

    Priority:
      - If an explicit file is provided, use its parent directory (unless missing
        and a cell_configs/ folder with configs exists).
      - If a directory is provided (or path is None), prefer cell_configs/
        when it contains sim_config.json or syn_config.json.
      - Fall back to the provided directory.
    """
    explicit_file = False
    p = None
    if path is None:
        base = Path.cwd()
    else:
        p = Path(path)
        if p.is_dir():
            base = p
        elif p.is_file():
            base = p.parent
            explicit_file = True
        elif p.suffix == ".json":
            base = p.parent
            explicit_file = True
        else:
            base = p

    if base.name == "cell_configs":
        return base

    candidate = base / "cell_configs"
    if (candidate / "sim_config.json").is_file() or (candidate / "syn_config.json").is_file():
        if not explicit_file:
            return candidate
        if p is not None and not p.is_file():
            return candidate
    if explicit_file:
        return base
    return base


def _inject_path_metadata(sim_cfg: Dict[str, Any], config_root: Path) -> None:
    """
    Populate sim_cfg with tune/cell labels inferred from the config path
    when those fields are missing.
    """
    tune_dir = config_root.parent if config_root.name == "cell_configs" else config_root
    sim_cfg.setdefault("tune_dir", str(tune_dir))
    cell_cfg_path = (
        config_root / "cell_config.json"
        if config_root.name == "cell_configs"
        else tune_dir / "cell_configs" / "cell_config.json"
    )
    cell_cfg = {}
    if cell_cfg_path.is_file():
        try:
            cell_cfg = json.loads(cell_cfg_path.read_text())
        except Exception:
            cell_cfg = {}

    if not sim_cfg.get("cell") and cell_cfg.get("cell_name"):
        sim_cfg["cell"] = str(cell_cfg.get("cell_name"))
    if not sim_cfg.get("tune") and cell_cfg.get("tune"):
        sim_cfg["tune"] = str(cell_cfg.get("tune"))
    if not sim_cfg.get("color") and cell_cfg.get("color") is not None:
        sim_cfg["color"] = cell_cfg.get("color")

    if not sim_cfg.get("tune"):
        sim_cfg["tune"] = tune_dir.name
    if not sim_cfg.get("cell") and tune_dir.parent.name == "tunes":
        sim_cfg["cell"] = tune_dir.parent.parent.name


def generate_inputs(
    path: Optional[Union[Path, str]] = None,
    geometry: Optional[Any] = None,
    mode_registry: Optional[Mapping[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
    seed_override: Optional[int] = None,
    trial_rng: Optional[randomness.TrialRandomness] = None,
    sim_cfg_override: Optional[Dict[str, Any]] = None,
    # cache: Optional[Dict[str, "GroupInputs"]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, "GroupInputs"]]:
    """
    Step 2.3 main entry point: load, normalize, and materialize synaptic inputs.

    This function assumes the split-config layout:
      - sim_config.json: simulation-level settings (dt, tstart, tstop, seed, etc.)
      - syn_config.json: per-group input/synapse configs (one top-level key per group)
    Configs may live in the tune directory or in tune_dir/cell_configs/.

    Path resolution:
      - If `path` is None:
          Use the current working directory as the tune directory.
          Prefer `cell_configs/` when it contains config files.
      - If `path` is a directory:
          Treat it as the tune directory. Prefer `cell_configs/` when present.
      - If `path` is a file:
          Treat its parent directory as the config root (no auto-redirect).
          Expect both `sim_config.json` and `syn_config.json` there.

    Parameters
    ----------
    path
        Optional directory or file path controlling where configs are loaded from.
        See rules above. Typical notebook usage is `path=None` with the tune
        directory as the working directory.
    sim_cfg_override
        Optional simulation config dict. When provided, it replaces the contents
        of sim_config.json while syn_config.json is still loaded from disk.
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
    config_root = _resolve_config_root(path)
    syn_path = config_root / "syn_config.json"
    sim_path = config_root / "sim_config.json"

    if not syn_path.is_file():
        raise FileNotFoundError(f"Missing syn_config.json in {config_root}")
    if not sim_path.is_file():
        raise FileNotFoundError(f"Missing sim_config.json in {config_root}")

    # ------------------------------------------------------------------
    # Load raw JSON configs (or use override)
    # ------------------------------------------------------------------
    if sim_cfg_override is None:
        with sim_path.open("r") as f:
            sim_cfg_raw = json.load(f)
    else:
        sim_cfg_raw = dict(sim_cfg_override)
    with syn_path.open("r") as f:
        groups_cfg_raw = json.load(f)

    # ------------------------------------------------------------------
    # 2.3.2 – Normalize configs
    # ------------------------------------------------------------------
    sim_cfg = _normalize_sim_config(sim_cfg_raw)
    _inject_path_metadata(sim_cfg, config_root)
    groups_cfg_expanded = _expand_group_includes(groups_cfg_raw, config_root)
    groups_cfg = _normalize_group_configs(groups_cfg_expanded)

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
      (tune root or cell_configs/)
    - Runs the same normalization as generate_inputs
    - Prints a concise summary of sim + per-group configs
    If path is None: assume current working directory and look there.
    If path is a file: treat its parent as the config root (no auto-redirect).
    If path is a directory: prefer cell_configs/ when present.
    """

    config_root = _resolve_config_root(path)
    syn_path = config_root / "syn_config.json"
    sim_path = config_root / "sim_config.json"

    if not syn_path.is_file():
        raise FileNotFoundError(f"Missing syn_config.json in {config_root}")
    if not sim_path.is_file():
        raise FileNotFoundError(f"Missing sim_config.json in {config_root}")

    with sim_path.open("r") as f:
        sim_raw = json.load(f)
    with syn_path.open("r") as f:
        groups_raw = json.load(f)

    sim_cfg = _normalize_sim_config(sim_raw)
    _inject_path_metadata(sim_cfg, config_root)
    groups_cfg_expanded = _expand_group_includes(groups_raw, config_root)
    groups_cfg = _normalize_group_configs(groups_cfg_expanded)


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
      - save_profile (optional: 'lean'|'standard'|'full')
        - fills n_traces_to_save / n_inputs_to_save if those keys are absent
      - n_traces_to_save (int >= 0)
      - n_inputs_to_save (int >= 0 or 'all')
      - trial_randomness (one of 'inputs', 'synapses', 'both', 'none')
      - load (string path or [enabled, path])
      - save/output (string stem or [enabled, stem, format, full_results])
      - append/append_to (string path or [enabled, path])
      - output_format ('npz' or 'pkl')
      - plots_profile ('off'|'basic'|'inputs'|'full')
      - param_study (dict with standard keys)
      - snapshot (optional dict): enables full debug capture
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

    # optional stimulation markers (used for plotting or bookkeeping)
    for key in ("stim_start_ms", "stim_stop_ms", "stim_duration_ms"):
        if key in sim_cfg and sim_cfg[key] is not None:
            try:
                sim_cfg[key] = float(sim_cfg[key])
            except Exception as exc:
                raise ValueError(f"sim['{key}'] must be numeric or null (got {sim_cfg[key]!r})") from exc
        else:
            sim_cfg[key] = None

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
    random_seed_raw = sim_cfg.get("random_seed", None)
    if sim_cfg["seed"] is None and random_seed_raw not in (None, "", False):
        try:
            sim_cfg["seed"] = int(random_seed_raw)
        except Exception as exc:
            raise ValueError(
                f"sim['random_seed'] must be integer-like or null (got {random_seed_raw!r})"
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

    # save_profile: optional, defaults to None
    save_profile = sim_cfg.get("save_profile", None)
    raw_has_traces = "n_traces_to_save" in sim_cfg_raw
    raw_has_inputs = "n_inputs_to_save" in sim_cfg_raw
    if save_profile not in (None, "", False):
        prof = str(save_profile).strip().lower()
        defaults = {
            "lean": {"n_traces_to_save": 1, "n_inputs_to_save": 1},
            "standard": {"n_traces_to_save": 1, "n_inputs_to_save": 10},
            "full": {"n_traces_to_save": 1, "n_inputs_to_save": "all"},
        }
        if prof not in defaults:
            raise ValueError(
                f"sim['save_profile'] must be one of {sorted(defaults)} (got {save_profile!r})"
            )
        if not raw_has_traces:
            sim_cfg["n_traces_to_save"] = defaults[prof]["n_traces_to_save"]
        if not raw_has_inputs:
            sim_cfg["n_inputs_to_save"] = defaults[prof]["n_inputs_to_save"]
        sim_cfg["save_profile"] = prof

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

    # n_inputs_to_save: optional, default n_traces_to_save
    n_inputs_raw = sim_cfg.get("n_inputs_to_save", n_traces)
    if isinstance(n_inputs_raw, str):
        if n_inputs_raw.strip().lower() in ("all",):
            sim_cfg["n_inputs_to_save"] = "all"
        else:
            try:
                sim_cfg["n_inputs_to_save"] = int(n_inputs_raw)
            except Exception as exc:
                raise ValueError(
                    f"sim['n_inputs_to_save'] must be integer-like or 'all' (got {n_inputs_raw!r})"
                ) from exc
    else:
        try:
            sim_cfg["n_inputs_to_save"] = int(n_inputs_raw)
        except Exception as exc:
            raise ValueError(
                f"sim['n_inputs_to_save'] must be integer-like or 'all' (got {n_inputs_raw!r})"
            ) from exc

    # load: allow [enabled, path] or {"enabled":..., "path":...}
    load_raw = sim_cfg.get("load", None)
    load_enabled = None
    load_path = None
    if isinstance(load_raw, (list, tuple)):
        if len(load_raw) >= 1:
            load_enabled = bool(load_raw[0])
        if len(load_raw) >= 2:
            load_path = load_raw[1]
    elif isinstance(load_raw, dict):
        load_enabled = bool(load_raw.get("enabled", False))
        load_path = load_raw.get("path")
    else:
        if load_raw in (None, "", False):
            load_enabled = False
            load_path = None
        else:
            load_enabled = True
            load_path = load_raw
    sim_cfg["load_enabled"] = bool(load_enabled)
    sim_cfg["load"] = None if load_path in (None, "", False) else str(load_path)

    # save/output + output_format (allow [enabled, stem, format, full_results] or dict form)
    if "save" in sim_cfg_raw:
        output_raw = sim_cfg_raw.get("save")
    else:
        output_raw = sim_cfg.get("output")
    output_enabled = None
    output_stem = None
    output_fmt = None
    output_full = None
    if isinstance(output_raw, (list, tuple)):
        if len(output_raw) >= 1:
            output_enabled = bool(output_raw[0])
        if len(output_raw) >= 2:
            output_stem = output_raw[1]
        if len(output_raw) >= 3:
            output_fmt = output_raw[2]
        if len(output_raw) >= 4:
            output_full = output_raw[3]
    elif isinstance(output_raw, dict):
        output_enabled = output_raw.get("enabled")
        output_stem = output_raw.get("path") or output_raw.get("stem") or output_raw.get("name")
        output_fmt = output_raw.get("format")
        output_full = output_raw.get("full_results")
        if output_full is None:
            output_full = output_raw.get("save_full_results")
    else:
        if output_raw not in (None, "", False):
            output_stem = output_raw
            output_enabled = True
        else:
            output_stem = None
            output_enabled = False

    # save_output: optional, default True (unless output tuple/dict provided)
    if not isinstance(output_raw, (list, tuple, dict)):
        if "save_output" in sim_cfg:
            output_enabled = bool(sim_cfg.get("save_output"))
    if output_enabled is None:
        output_enabled = True
    sim_cfg["save_output"] = bool(output_enabled)
    sim_cfg["output"] = None if output_stem in (None, "", False) else str(output_stem)
    sim_cfg["save"] = sim_cfg["output"]

    if output_fmt is not None:
        sim_cfg["output_format"] = str(output_fmt)

    ofmt = sim_cfg.get("output_format", "pkl")
    if ofmt is None:
        ofmt = "pkl"
    ofmt = str(ofmt)
    if ofmt not in {"npz", "pkl"}:
        raise ValueError(
            f"sim['output_format'] must be 'npz' or 'pkl' (got {ofmt!r})"
        )
    sim_cfg["output_format"] = ofmt

    if output_full is not None:
        sim_cfg["save_full_results"] = bool(output_full)
        if bool(output_full) and "save_sidecars" not in sim_cfg_raw:
            sim_cfg["save_sidecars"] = False

    # plots_profile: optional presets for save_plots flags
    plots_profile = sim_cfg.get("plots_profile", None)
    if plots_profile not in (None, "", False):
        prof = str(plots_profile).strip().lower()
        defaults = {
            "off": {"save_plots": False, "save_plots_inputs": False, "save_plots_synapses": False},
            "basic": {"save_plots": True, "save_plots_inputs": False, "save_plots_synapses": False},
            "inputs": {"save_plots": True, "save_plots_inputs": True, "save_plots_synapses": False},
            "full": {"save_plots": True, "save_plots_inputs": True, "save_plots_synapses": True},
        }
        if prof not in defaults:
            raise ValueError(
                f"sim['plots_profile'] must be one of {sorted(defaults)} (got {plots_profile!r})"
            )
        if "save_plots" not in sim_cfg_raw:
            sim_cfg["save_plots"] = defaults[prof]["save_plots"]
        if "save_plots_inputs" not in sim_cfg_raw:
            sim_cfg["save_plots_inputs"] = defaults[prof]["save_plots_inputs"]
        if "save_plots_synapses" not in sim_cfg_raw:
            sim_cfg["save_plots_synapses"] = defaults[prof]["save_plots_synapses"]
        sim_cfg["plots_profile"] = prof

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

    # append/append_to: allow [enabled, path] or {"enabled":..., "path":...}
    if "append" in sim_cfg_raw:
        append_raw = sim_cfg_raw.get("append")
    else:
        append_raw = sim_cfg.get("append_to", None)
    append_enabled = None
    append_path = None
    if isinstance(append_raw, (list, tuple)):
        if len(append_raw) >= 1:
            append_enabled = bool(append_raw[0])
        if len(append_raw) >= 2:
            append_path = append_raw[1]
    elif isinstance(append_raw, dict):
        append_enabled = bool(append_raw.get("enabled", False))
        append_path = append_raw.get("path")
    else:
        if append_raw in (None, "", False):
            append_enabled = False
            append_path = None
        else:
            append_enabled = True
            append_path = append_raw
    sim_cfg["append_enabled"] = bool(append_enabled)
    sim_cfg["append_to"] = None if append_path in (None, "", False) else str(append_path)
    sim_cfg["append"] = sim_cfg["append_to"]

    # Snapshot mode: force full capture for debugging comparisons
    snapshot_raw = sim_cfg_raw.get("snapshot", sim_cfg.get("snapshot"))
    snapshot_cfg = None
    snapshot_enabled = False
    if isinstance(snapshot_raw, dict):
        snapshot_cfg = dict(snapshot_raw)
        snapshot_enabled = bool(snapshot_cfg.get("enabled", False))
    elif isinstance(snapshot_raw, str):
        snapshot_enabled = snapshot_raw.strip().lower() in ("true", "1", "yes", "on")
        snapshot_cfg = {}
    elif snapshot_raw is True:
        snapshot_enabled = True
        snapshot_cfg = {}

    if snapshot_enabled:
        snapshot_cfg = snapshot_cfg or {}
        snapshot_cfg["enabled"] = True

        snap_trials = snapshot_cfg.get("n_trials", None)
        if snap_trials is None:
            snap_trials = 1
        try:
            snap_trials = int(snap_trials)
        except Exception:
            snap_trials = 1
        if snap_trials < 1:
            snap_trials = 1
        sim_cfg["n_trials"] = snap_trials

        if snapshot_cfg.get("save_all_inputs", True):
            sim_cfg["n_inputs_to_save"] = "all"
        if snapshot_cfg.get("save_all_traces", True):
            sim_cfg["n_traces_to_save"] = snap_trials

        sim_cfg["save_full_results"] = True
        sim_cfg["save_sidecars"] = True
        sim_cfg["save_syn_records_sidecar"] = True
        sim_cfg["save_syn_records_by_trial"] = True
        sim_cfg["save_input_stats"] = True
        sim_cfg["log_input_summary"] = True
        sim_cfg["save_output"] = True

        snap_output = snapshot_cfg.get("output") or snapshot_cfg.get("output_stem")
        if snap_output not in (None, "", False):
            sim_cfg["output"] = str(snap_output)

        sim_cfg["snapshot"] = snapshot_cfg

    # Optional simplified randomness mode
    sim_cfg = randomness.apply_randomness_mode(sim_cfg)

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

def _lognormal_mu_sigma(mean: float, std: float) -> Tuple[float, float]:
    """
    Return (mu, sigma) for np.random.lognormal given arithmetic mean & std.
    """
    if std <= 0 or mean <= 0:
        return 0.0, 0.0
    mu = math.log(mean**2 / math.sqrt(std**2 + mean**2))
    sig = math.sqrt(math.log(1 + (std**2 / mean**2)))
    return mu, sig

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

    # Optional: per-trial jitter of input start (global), used to delay baseline onset
    jitter_tstart: Optional[float] = None
    jitter_std = sim_cfg.get("jitter", None)
    if jitter_std is not None:
        try:
            jitter_std = float(jitter_std)
        except Exception as exc:
            raise ValueError(f"sim['jitter'] must be numeric or null (got {jitter_std!r})") from exc
        if jitter_std > 0:
            if trial_rng is not None:
                jitter_rng = trial_rng.rng("inputs", stream="jitter")
            else:
                seed = sim_cfg.get("seed", None)
                if seed is None:
                    jitter_rng = np.random.default_rng()
                else:
                    jitter_rng = np.random.default_rng(int(seed) ^ 0xA5A5A5A5)

            mean = float(sim_cfg.get("tstart", 0.0))
            mu, sig = _lognormal_mu_sigma(mean, jitter_std)
            jitter_tstart = float(jitter_rng.lognormal(mu, sig)) if sig > 0 else mean
            # Clamp to simulation window start
            jitter_tstart = max(mean, jitter_tstart)

    # Stash for timing calculation (used by modes)
    sim_cfg["_jitter_tstart_ms"] = jitter_tstart

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
        setting_path = "inputs"
        group_rng = rng
        rand_meta = None
        if trial_rng is not None:
            if mode_name:
                sentinel = object()
                mode_setting = trial_rng.setting(f"modes.{mode_name}", sentinel)
                if mode_setting is not sentinel:
                    setting_path = f"modes.{mode_name}"
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
      - dict          → {"kind": "linear", "params": {"m": float, "b": float, "multi": optional}}
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

        if kind == "linear":
            m = params.get("m", params.get("slope", 0.0))
            b = params.get("b", params.get("intercept", 0.0))
            multi = float(params.get("multi", 1.0))
            m = float(m)
            b = float(b)
            return lambda d, m=m, b=b, multi=multi: (m * d + b) * multi

        # Placeholder for future shapes (gaussian, etc.)
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
    source_tstart_raw: Optional[float] = None


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

    # stim_tstart: explicit marker in sim time for the stimulus; do not infer.
    if stim_tstart_raw is None:
        stim_tstart = None
    else:
        stim_tstart = max(min(stim_tstart_raw, tstop), tstart)

    # Source start: align source data so that its own stim event occurs
    # at stim_tstart in the simulation.
    source_tstart: Optional[float] = None
    if (stim_tstart is not None) and (input_stim_raw is not None):
        source_tstart_raw = stim_tstart - input_stim_raw
        # Clamp to onset/sim window, but remember how much we trimmed so
        # modes can discard the leading portion to keep stim alignment.
        source_tstart = max(onset, tstart, source_tstart_raw)

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

        # Clamp to sim window
        source_tstop = min(source_tstop, tstop)

    # If we ended up with an invalid source window, drop it (baseline/quiescent only)
    if source_tstop is not None and source_tstart is not None and source_tstop <= source_tstart:
        source_tstart = None
        source_tstop = None

    # How much of the source should be trimmed (when raw start < resolved start)
    source_trim_ms: Optional[float] = None
    if source_tstart is not None and source_tstart_raw is not None:
        delta = source_tstart - source_tstart_raw
        source_trim_ms = max(0.0, float(delta))

    anchors: Dict[str, Optional[float]] = {
        "sim_tstart": tstart,
        "sim_tstop": tstop,
        "onset": onset,
        "source_tstart": source_tstart,
        "source_tstop": source_tstop,
        "baseline_rate_hz": baseline_hz,  # numeric baseline only
        "baseline_spec": baseline_spec,   # tokenized baseline (if any)
        "source_trim_ms": source_trim_ms,
        "jitter_tstart_ms": _as_float_or_none(sim_cfg.get("_jitter_tstart_ms"), "sim['_jitter_tstart_ms']"),
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
    baseline_spec = anchors.get("baseline_spec") or {}
    baseline_active = (baseline_rate is not None and baseline_rate > 0.0) or (
        baseline_spec.get("kind") not in (None, "none")
    )

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
        kind = "baseline" if baseline_active else "quiescent"
        _add_block(kind, onset, sim_tstop)
        return blocks

    # Ensure ordering for source anchors
    src_start = max(min(source_tstart, sim_tstop), sim_tstart)
    src_stop = max(min(source_tstop, sim_tstop), sim_tstart)
    if src_stop <= src_start:
        _add_block("quiescent", sim_tstart, onset)
        kind = "baseline" if baseline_active else "quiescent"
        _add_block(kind, onset, sim_tstop)
        return blocks

    # 1) steady-state / quiescent before onset
    _add_block("quiescent", sim_tstart, min(onset, src_start))

    # 2) pre-source baseline or quiescent
    pre_start = max(onset, sim_tstart)
    pre_end = min(src_start, sim_tstop)
    pre_kind = "baseline" if baseline_active else "quiescent"
    _add_block(pre_kind, pre_start, pre_end)

    # 3) main source window
    _add_block("source", src_start, src_stop)

    # 4) post-source baseline or quiescent
    post_start = max(src_stop, sim_tstart)
    post_end = sim_tstop
    post_kind = "baseline" if baseline_active else "quiescent"
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
            "source_trim_ms": anchors.get("source_trim_ms"),
            "jitter_tstart_ms": anchors.get("jitter_tstart_ms"),
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
