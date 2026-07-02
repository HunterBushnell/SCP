"""
Core Step 5 simulation routines.

This module keeps the historical `modules.run_sim` public API while delegating
current-injection helpers and result I/O to focused `modules.simulation` modules.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import copy
import hashlib
import os
import random
import sys
import time

import numpy as np
from neuron import h

from . import inputs as inputs_mod
from . import randomness, synapses
from .simulation.cell_runtime import (
    _build_cell_recorders_for_site,
    _get_cell_recording_cfg,
    _get_soma_segment,
    _normalize_runtime_recording_site,
    _parse_bool_like,
    _resolve_recording_site,
    run_cell,
)
from .simulation.current_injection import (
    _get_hoc,
    get_frequency,
    get_rec_vars_for_i_in_sec,
    looped_current_injection,
    plot_looped_currents,
    run_FI,
    run_current_injection,
    run_iclamp_test,
)
from .simulation.result_helpers import (
    _aggregate_input_stats,
    _resolve_inputs_to_save,
    _resolve_trace_trials_to_save,
    _smooth_rate_curve,
)
from .simulation.results import (
    _append_results_to_path,
    _build_output_path,
    _copy_fit_json_sidecar,
    _ensure_multi_results,
    _find_fit_json_path,
    _json_default,
    _load_from_manifest,
    _resolve_tune_path,
    _save_sidecars,
    _sha256_file,
    _write_json,
    _write_results_file,
    _write_results_to_run_dir,
    append_multi_results,
    load_old_multi_results,
    load_results,
    save_results,
    save_results_with_name,
)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _infer_mode(sim_cfg: Dict[str, Any]) -> str:
    """
    Infer mode from sim_cfg:
      - 'param' if param_study has non-empty param_vals
      - 'multi' if n_trials > 1
      - 'single' otherwise
    """
    param = sim_cfg.get("param_study") or {}
    param_vals = param.get("param_vals") or []
    if len(param_vals) > 0:
        return "param"

    n_trials = int(sim_cfg.get("n_trials", 1))
    if n_trials > 1:
        return "multi"

    return "single"


def _clear_cell_state(cell: Any) -> None:
    """
    Best-effort clearing of NEURON-related state on the cell between trials.
    Assumes the cell exposes lists/containers with these attribute names
    (missing ones are ignored).
    """
    for attr in ("syn_locs", "vecs", "stims", "synapses", "netcons"):
        if hasattr(cell, attr):
            lst = getattr(cell, attr)
            try:
                lst.clear()
            except AttributeError:
                # older code may use h.List or similar; fall back to manual deletion
                try:
                    while len(lst) > 0:
                        lst.remove(lst[0])
                except Exception:
                    pass


def _warn_preexisting_synapses(cell: Any, *, context: str = "") -> None:
    counts = []
    for attr in ("synapses", "netcons", "stims", "vecs"):
        if hasattr(cell, attr):
            try:
                n = len(getattr(cell, attr))
            except Exception:
                n = None
            if n:
                counts.append(f"{attr}={n}")
    if counts:
        label = f" ({context})" if context else ""
        msg = "WARNING: pre-attached synapse objects detected"
        print(f"{msg}{label}: " + ", ".join(counts))
        print("         This can change results; attach synapses inside run_sim only.")


def _detect_spikes(T: np.ndarray, V: np.ndarray, v_thresh: float = 0.0) -> np.ndarray:
    """
    Simple spike detector: returns times where V crosses v_thresh from below.
    This is intentionally minimal and can be replaced later with a better detector.
    """
    above = V > v_thresh
    crossings = np.where(above[1:] & ~above[:-1])[0] + 1
    return T[crossings]


def _as_bool(val: Any, default: bool = True) -> bool:
    if val is None:
        return default
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("false", "0", "no", "off", ""):
            return False
        if v in ("true", "1", "yes", "on"):
            return True
    return bool(val)


def _snapshot_cfg(sim_cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    snap = sim_cfg.get("snapshot", None)
    if isinstance(snap, dict):
        return bool(snap.get("enabled", False)), snap
    if snap is True:
        return True, {"enabled": True}
    if isinstance(snap, str) and snap.strip().lower() in ("true", "1", "yes", "on"):
        return True, {"enabled": True}
    return False, {}


def _apply_snapshot_deterministic(sim_cfg: Dict[str, Any], snapshot_cfg: Dict[str, Any]) -> None:
    """
    Best-effort deterministic settings for snapshot comparisons.
    Only applies if snapshot_cfg.force_deterministic is True (default).
    """
    if not snapshot_cfg.get("force_deterministic", True):
        snapshot_cfg["deterministic_applied"] = False
        return

    seed = snapshot_cfg.get("deterministic_seed")
    if seed is None:
        seed = sim_cfg.get("random_seed", sim_cfg.get("seed", 0))
    try:
        seed = int(seed)
    except Exception:
        seed = 0

    try:
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass

    try:
        if hasattr(h, "cvode"):
            h.cvode.active(0)
    except Exception:
        pass
    try:
        if hasattr(h, "nthread"):
            h.nthread(1)
    except Exception:
        pass
    try:
        if hasattr(h, "Random123_globalindex"):
            h.Random123_globalindex(seed)
    except Exception:
        pass

    snapshot_cfg["deterministic_applied"] = True
    snapshot_cfg["deterministic_seed"] = seed


def _collect_versions() -> Dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "python_exe": sys.executable,
        "numpy": np.__version__,
    }
    try:
        versions["neuron"] = str(getattr(h, "nrnversion", lambda: "unknown")())
    except Exception:
        versions["neuron"] = "unknown"
    return versions


def _collect_env_snapshot() -> Dict[str, Any]:
    keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NEURONHOME",
        "NRNHOME",
        "NRN_NMODL_PATH",
        "NRNMECH_DLL",
    )
    snap: Dict[str, Any] = {}
    for key in keys:
        val = os.environ.get(key)
        if val not in (None, ""):
            snap[key] = val
    return snap


def _collect_neuron_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    for key in ("dt", "tstop", "t", "celsius", "secondorder", "v_init", "steps_per_ms"):
        try:
            state[key] = float(getattr(h, key))
        except Exception:
            pass
    try:
        if hasattr(h, "cvode"):
            cvode = h.cvode
            state["cvode_active"] = int(cvode.active())
            for name in ("atol", "rtol", "minstep", "maxstep"):
                try:
                    state[f"cvode_{name}"] = float(getattr(cvode, name)())
                except Exception:
                    pass
    except Exception:
        pass
    try:
        if hasattr(h, "secondorder"):
            state["secondorder"] = int(h.secondorder)
    except Exception:
        pass
    try:
        if hasattr(h, "nthread"):
            state["nthread"] = int(h.nthread())
    except Exception:
        pass
    return state


def _collect_mechanism_info(sim_cfg: Dict[str, Any]) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    tune_path = _resolve_tune_path(sim_cfg)
    if tune_path is None:
        return info

    info["tune_dir"] = str(tune_path)
    mod_dir = tune_path / "modfiles"
    mod_files = sorted(p for p in mod_dir.glob("*.mod")) if mod_dir.is_dir() else []
    if mod_files:
        info["modfiles_count"] = len(mod_files)
        info["modfiles"] = [p.name for p in mod_files]
        hsh = hashlib.sha256()
        for p in mod_files:
            try:
                hsh.update(p.name.encode("ascii", errors="ignore"))
                hsh.update(_sha256_file(p).encode("ascii"))
            except Exception:
                continue
        info["modfiles_sha256"] = hsh.hexdigest()

    dll_candidates = [
        mod_dir / "x86_64" / ".libs" / "libnrnmech.so",
        mod_dir / "x86_64" / "libnrnmech.so",
    ]
    for dll in dll_candidates:
        if dll.is_file():
            info["dll_path"] = str(dll)
            try:
                stat = dll.stat()
                info["dll_size"] = int(stat.st_size)
                info["dll_mtime"] = float(stat.st_mtime)
            except Exception:
                pass
            try:
                info["dll_sha256"] = _sha256_file(dll)
            except Exception:
                pass
            break

    return info


def _snapshot_netcon_state(syn_state: Dict[str, Any]) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    netcons = syn_state.get("netcons", {}) or {}
    for group, ncs in netcons.items():
        weights: List[Optional[float]] = []
        delays: List[Optional[float]] = []
        for nc in ncs or []:
            try:
                weights.append(float(nc.weight[0]))
            except Exception:
                weights.append(None)
            try:
                delays.append(float(nc.delay))
            except Exception:
                delays.append(None)
        snapshot[group] = {"n": len(ncs), "weights": weights, "delays": delays}
    return snapshot


def _snapshot_synapse_params(
    syn_state: Dict[str, Any],
    groups_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    syn_by_group = syn_state.get("synapses", {}) or {}
    for group, syn_list in syn_by_group.items():
        if not syn_list:
            continue
        syn = syn_list[0]
        gcfg = groups_cfg.get(group, {}) or {}
        syn_cfg = gcfg.get("syns", {}) or {}
        params_cfg = syn_cfg.get("params", {}) or {}
        present = {}
        missing = []
        for key in params_cfg:
            if hasattr(syn, key):
                try:
                    present[key] = float(getattr(syn, key))
                except Exception:
                    present[key] = getattr(syn, key)
            else:
                missing.append(key)
        snapshot[group] = {
            "type": syn_cfg.get("type"),
            "params_present": present,
            "params_missing": missing,
        }
    return snapshot


def _set_trace_trials_to_save(sim_cfg: Dict[str, Any], n_traces: int) -> None:
    n = max(0, int(n_traces))
    sim_cfg["n_traces_to_save"] = n
    cell_rec = sim_cfg.get("cell_recording")
    if isinstance(cell_rec, dict):
        cell_rec = dict(cell_rec)
        cell_rec["n_trials"] = n
        sim_cfg["cell_recording"] = cell_rec


def _coerce_bin_width(val: Any, default: float) -> float:
    try:
        bw = float(val)
    except Exception:
        bw = float(default)
    if bw <= 0:
        bw = float(default)
    return bw


def _prepare_input_stats_bins(
    tstart: float,
    tstop: float,
    bin_width: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    bw = _coerce_bin_width(bin_width, 25.0)
    t0 = float(tstart)
    t1 = float(tstop)
    if t1 < t0:
        t1 = t0
    bins = np.arange(t0, t1 + bw, bw, dtype=float)
    if bins.size < 2:
        bins = np.array([t0, t0 + bw], dtype=float)
    centers = bins[:-1] + 0.5 * bw
    return bw, bins, centers


def _compute_input_stats_for_trial(
    inputs_by_group: Dict[str, Any],
    bins: np.ndarray,
    bin_width: float,
    tstart: float,
    tstop: float,
) -> Dict[str, Any]:
    bw_s = bin_width / 1000.0
    dur_s = max(1e-9, (float(tstop) - float(tstart)) / 1000.0)
    groups: Dict[str, Any] = {}

    for g, gi in inputs_by_group.items():
        trains = [np.asarray(tr, dtype=float) for tr in (gi.spike_trains or [])]
        n_syn = len(trains)
        if n_syn:
            all_spikes = np.concatenate(trains)
        else:
            all_spikes = np.array([], dtype=float)

        counts, _ = np.histogram(all_spikes, bins=bins)
        total_spikes = int(all_spikes.size)
        rate_hz_total = total_spikes / dur_s
        rate_hz_per_syn = rate_hz_total / n_syn if n_syn > 0 else 0.0

        rate_hz_by_bin_total = counts / bw_s
        if n_syn > 0:
            rate_hz_by_bin_per_syn = rate_hz_by_bin_total / n_syn
        else:
            rate_hz_by_bin_per_syn = np.zeros_like(rate_hz_by_bin_total, dtype=float)

        groups[g] = {
            "n_syn": int(n_syn),
            "total_spikes": total_spikes,
            "rate_hz_total": float(rate_hz_total),
            "rate_hz_per_syn": float(rate_hz_per_syn),
            "counts_by_bin": counts.tolist(),
            "rate_hz_by_bin_total": rate_hz_by_bin_total.tolist(),
            "rate_hz_by_bin_per_syn": rate_hz_by_bin_per_syn.tolist(),
        }

    return groups


# ---------------------------------------------------------------------
# core run functions
# ---------------------------------------------------------------------

def run_single(
    cell: Any,
    geom: Any,
    sim_cfg: Dict[str, Any],
    groups_cfg: Dict[str, Any],
    inputs_by_group: Dict[str, Any],
    *,
    rm: Optional[randomness.RandomnessManager] = None,
) -> Dict[str, Any]:
    """
    Run a single simulation (one trial) with the given config and inputs.

    Returns a standardized results dict:
      {
        "mode": "single",
        "sim_cfg": { ... },
        "spikes": 1D np.ndarray of spike times,
        "traces": {
           "T": 1D np.ndarray (time),
           "V": 1D np.ndarray (soma voltage)
        } or {},
        "meta": { ... }
      }
    """
    sim_cfg_local = copy.deepcopy(sim_cfg)
    snapshot_enabled, snapshot_cfg = _snapshot_cfg(sim_cfg_local)
    if snapshot_enabled:
        _apply_snapshot_deterministic(sim_cfg_local, snapshot_cfg)
    trial_rng = rm.trial(0) if rm is not None else None
    n_traces_to_save = _resolve_trace_trials_to_save(sim_cfg_local, fallback=1)
    if snapshot_enabled and snapshot_cfg.get("save_all_traces", True):
        n_traces_to_save = max(n_traces_to_save, 1)
        _set_trace_trials_to_save(sim_cfg_local, n_traces_to_save)
    else:
        _set_trace_trials_to_save(sim_cfg_local, n_traces_to_save)
    n_inputs_to_save = _resolve_inputs_to_save(sim_cfg_local, 1, n_traces_to_save)
    tstart = float(sim_cfg_local.get("tstart", 0.0))
    tstop = float(sim_cfg_local.get("tstop", tstart))

    input_stats = None
    if _as_bool(sim_cfg_local.get("save_input_stats", True), default=True):
        bin_width = sim_cfg_local.get("input_stats_bin_ms", sim_cfg_local.get("bins", 25.0))
        bin_width, bins, centers = _prepare_input_stats_bins(tstart, tstop, bin_width)
        trial_groups = _compute_input_stats_for_trial(
            inputs_by_group, bins, bin_width, tstart, tstop
        )
        trial_stats = [{"trial_idx": 0, "groups": trial_groups}]
        input_stats = {
            "bin_ms": bin_width,
            "t_ms": centers.tolist(),
            "tstart_ms": tstart,
            "tstop_ms": tstop,
            "trials": trial_stats,
            "group_means": _aggregate_input_stats(trial_stats),
        }

    # reset cell state and attach synapses
    _warn_preexisting_synapses(cell, context="run_single")
    _clear_cell_state(cell)
    syn_state = synapses.add_synapses(
        cell, geom, sim_cfg_local, groups_cfg, inputs_by_group, trial_rng=trial_rng
    )
    syn_records = syn_state.get("records", {})
    syn_records_by_trial: Optional[List[Dict[str, Any]]] = None
    if _as_bool(sim_cfg_local.get("save_syn_records_by_trial", False), default=False):
        syn_records_by_trial = [{"trial_idx": 0, "records": syn_records}]
    syn_param_snapshot: Optional[Dict[str, Any]] = None
    netcon_snapshot: Optional[Dict[str, Any]] = None
    if snapshot_enabled:
        syn_param_snapshot = _snapshot_synapse_params(syn_state, groups_cfg)
        netcon_snapshot = _snapshot_netcon_state(syn_state)

    # run the actual simulation (existing 3.1 primitive)
    sim_traces = run_cell(cell, sim_cfg_local)  # assumes this is defined below / in this module

    T = np.asarray(sim_traces.get("T", []), dtype=float)
    V = np.asarray(sim_traces.get("V", []), dtype=float)
    spikes = _detect_spikes(T, V) if T.size and V.size else np.array([], dtype=float)
    cell_recordings = sim_traces.get("cell_recordings")

    traces_out: Dict[str, Any] = {}
    if n_traces_to_save > 0 and T.size and V.size:
        traces_out = {"T": T, "V": V}

    inputs_out: Optional[Dict[str, Any]] = None
    if n_inputs_to_save > 0:
        inputs_out = {}
        for g, gi in inputs_by_group.items():
            inputs_out[g] = {
                "mode": gi.mode,
                "spike_trains": [np.asarray(tr).copy() for tr in gi.spike_trains],
                "meta": gi.meta,
            }

    result = {
        "mode": "single",
        "sim_cfg": sim_cfg_local,
        "spikes": spikes,
        "traces": traces_out,
        "cell_recordings": cell_recordings,
        "syn_records": syn_records,
        "syn_records_by_trial": syn_records_by_trial,
        "inputs": inputs_out,
        "meta": {
            "cell": sim_cfg_local.get("cell"),
            "tune": sim_cfg_local.get("tune"),
            "n_trials": 1,
            "syn_config": copy.deepcopy(groups_cfg),
        },
    }
    if rm is not None:
        result["meta"]["randomness"] = rm.meta().as_dict()
    if snapshot_enabled:
        result["meta"]["snapshot"] = copy.deepcopy(snapshot_cfg)
        result["meta"]["versions"] = _collect_versions()
        result["meta"]["neuron_state"] = _collect_neuron_state()
        result["meta"]["env"] = _collect_env_snapshot()
        result["meta"]["mechanisms"] = _collect_mechanism_info(sim_cfg_local)
        if netcon_snapshot is not None:
            result["meta"]["netcon_snapshot"] = netcon_snapshot
        if syn_param_snapshot is not None:
            result["meta"]["synapse_param_snapshot"] = syn_param_snapshot
    if input_stats is not None:
        result["meta"]["input_stats"] = input_stats
    return result


def run_multi(
    cell: Any,
    geom: Any,
    sim_cfg: Dict[str, Any],
    groups_cfg: Dict[str, Any],
    inputs_by_group: Dict[str, Any],
    *,
    rm: Optional[randomness.RandomnessManager] = None,
    mode_registry: Optional[Dict[str, Any]] = None,
    trial_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run multiple trials with the same config. Currently:
      - reuses inputs_by_group for all trials,
      - re-attaches synapses fresh for each trial.

    Returns:
      {
        "mode": "multi",
        "sim_cfg": { ... },
        "spikes": [np.ndarray, ...],      # one per trial
        "traces": {
           "T": 1D np.ndarray,
           "V": [np.ndarray, ...]         # up to cell_recording.n_trials (legacy: n_traces_to_save)
        } or {},
        "meta": {
           "n_trials": int,
           "trial_ids": [0, 1, ...]
        }
      }
    """
    sim_cfg_local = copy.deepcopy(sim_cfg)
    snapshot_enabled, snapshot_cfg = _snapshot_cfg(sim_cfg_local)
    if snapshot_enabled:
        _apply_snapshot_deterministic(sim_cfg_local, snapshot_cfg)
    n_trials = int(sim_cfg_local.get("n_trials", 1))
    n_traces_to_save = _resolve_trace_trials_to_save(sim_cfg_local, fallback=1)
    if snapshot_enabled and snapshot_cfg.get("save_all_traces", True):
        n_traces_to_save = max(n_traces_to_save, n_trials)
        _set_trace_trials_to_save(sim_cfg_local, n_traces_to_save)
    else:
        _set_trace_trials_to_save(sim_cfg_local, n_traces_to_save)
    trial_offset = int(sim_cfg_local.get("trial_offset", 0) or 0)

    regen_inputs = _as_bool(sim_cfg_local.get("regen_inputs_each_trial", True), default=True)

    # Prebuild mode registry once for per-trial input regeneration
    if mode_registry is None:
        mode_registry = inputs_mod._build_default_mode_registry()
        try:
            from modules import input_modes_user

            user_reg = input_modes_user.get_user_mode_registry()
            # user registry wins on name collisions
            mode_registry = {**mode_registry, **user_reg}
        except Exception:
            pass

    spikes_by_trial: List[np.ndarray] = []
    trace_V_store: List[np.ndarray] = []
    cell_recordings_store: List[Dict[str, Any]] = []
    T_ref: Optional[np.ndarray] = None
    input_summaries: List[Dict[str, Any]] = []
    inputs_store: List[Dict[str, Any]] = []
    tstart = float(sim_cfg_local.get("tstart", 0.0))
    tstop = float(sim_cfg_local.get("tstop", 0.0))
    sim_dur_s = max(1e-9, (tstop - tstart) / 1000.0)
    inputs_to_save = _resolve_inputs_to_save(sim_cfg_local, n_trials, n_traces_to_save)
    input_stats_enabled = _as_bool(sim_cfg_local.get("save_input_stats", True), default=True)
    save_syn_records_by_trial = _as_bool(
        sim_cfg_local.get("save_syn_records_by_trial", False), default=False
    )
    input_stats_trials: List[Dict[str, Any]] = []
    syn_records_by_trial: List[Dict[str, Any]] = []
    syn_param_snapshot: Optional[Dict[str, Any]] = None
    netcon_snapshot: Optional[Dict[str, Any]] = None
    input_bin_width = sim_cfg_local.get("input_stats_bin_ms", sim_cfg_local.get("bins", 25.0))
    input_bin_width, input_bins, input_centers = _prepare_input_stats_bins(
        tstart, tstop, input_bin_width
    )

    _warn_preexisting_synapses(cell, context="run_multi")
    for trial_idx in range(n_trials):
        trial_start = time.perf_counter()
        trial_rng_idx = trial_idx + trial_offset
        trial_rng = rm.trial(trial_rng_idx) if rm is not None else None

        # Optionally regenerate inputs per trial (fresh randomness)
        if regen_inputs:
            gcfg_trial = copy.deepcopy(groups_cfg)
            inputs_trial = inputs_mod._process_all_groups(
                sim_cfg=sim_cfg_local,
                groups_cfg=gcfg_trial,
                geometry=geom,
                mode_registry=mode_registry,
                rng=None,
                trial_rng=trial_rng,
            )
            groups_cfg_for_trial = gcfg_trial
        else:
            inputs_trial = inputs_by_group
            groups_cfg_for_trial = groups_cfg

        if inputs_to_save > 0 and len(inputs_store) < inputs_to_save:
            trial_inputs: Dict[str, Any] = {}
            for g, gi in inputs_trial.items():
                trial_inputs[g] = {
                    "mode": gi.mode,
                    "spike_trains": [np.asarray(tr).copy() for tr in gi.spike_trains],
                    "meta": gi.meta,
            }
            inputs_store.append({"trial_idx": trial_idx, "inputs": trial_inputs})

        if input_stats_enabled:
            groups_stats = _compute_input_stats_for_trial(
                inputs_trial, input_bins, input_bin_width, tstart, tstop
            )
            input_stats_trials.append({"trial_idx": trial_idx, "groups": groups_stats})

        # Optional per-trial input summary (helps detect identical inputs)
        log_input_summary = bool(sim_cfg_local.get("log_input_summary", True))
        summary: Dict[str, Any] = {}
        for g, gi in inputs_trial.items():
            trains = gi.spike_trains or []
            total_spikes = int(sum(len(tr) for tr in trains))
            sum_spike_times = float(sum(float(np.sum(tr)) for tr in trains)) if trains else 0.0
            summary[g] = {
                "n_syn": int(len(trains)),
                "total_spikes": total_spikes,
                "sum_spike_times": sum_spike_times,
            }
        input_summaries.append({"trial_idx": trial_idx, "groups": summary})
        if log_input_summary:
            parts = [f"{g}={summary[g]['total_spikes']}" for g in summary]
            print(f"[trial {trial_idx+1}/{n_trials}] input_spikes: " + " ".join(parts))

        _clear_cell_state(cell)
        syn_state = synapses.add_synapses(
            cell, geom, sim_cfg_local, groups_cfg_for_trial, inputs_trial, trial_rng=trial_rng
        )
        if save_syn_records_by_trial:
            syn_records_by_trial.append(
                {"trial_idx": trial_idx, "records": syn_state.get("records", {})}
            )
        if snapshot_enabled and trial_idx == 0:
            syn_param_snapshot = _snapshot_synapse_params(syn_state, groups_cfg_for_trial)
            netcon_snapshot = _snapshot_netcon_state(syn_state)

        sim_traces = run_cell(cell, sim_cfg_local)

        T = np.asarray(sim_traces.get("T", []), dtype=float)
        V = np.asarray(sim_traces.get("V", []), dtype=float)
        spikes = _detect_spikes(T, V) if T.size and V.size else np.array([], dtype=float)

        spikes_by_trial.append(spikes)

        # save traces for a subset of trials
        if n_traces_to_save > 0 and len(trace_V_store) < n_traces_to_save and T.size and V.size:
            if T_ref is None:
                T_ref = T
            trace_V_store.append(V)
            if sim_traces.get("cell_recordings") is not None:
                cell_recordings_store.append(
                    {
                        "trial_idx": trial_idx,
                        "recordings": sim_traces.get("cell_recordings"),
                    }
                )

        if trial_callback is not None:
            try:
                trial_callback(
                    {
                        "trial_idx": trial_idx,
                        "spikes": spikes,
                        "traces": sim_traces,
                        "sim_cfg": sim_cfg_local,
                        "syn_records": syn_state.get("records", {}),
                    }
                )
            except Exception:
                # Do not fail the simulation because a callback failed
                pass

        # Progress log to stdout (captured by SLURM)
        spike_count = len(spikes)
        rate_hz = spike_count / sim_dur_s if sim_dur_s > 0 else 0.0
        elapsed = time.perf_counter() - trial_start
        print(f"[trial {trial_idx+1}/{n_trials}] spikes={spike_count}  rate={rate_hz:.2f} Hz  time={elapsed:.2f}s")

    input_stats = None
    if input_stats_enabled:
        input_stats = {
            "bin_ms": input_bin_width,
            "t_ms": input_centers.tolist(),
            "tstart_ms": tstart,
            "tstop_ms": tstop,
            "trials": input_stats_trials,
            "group_means": _aggregate_input_stats(input_stats_trials),
        }

    traces_out: Dict[str, Any] = {}
    if T_ref is not None and trace_V_store:
        traces_out = {
            "T": T_ref,
            "V": trace_V_store,
        }

    # Compute and store average firing-rate curve (raw, unsmoothed)
    bin_width = float(sim_cfg_local.get("bins", 25.0))
    bins = np.arange(0, tstop + bin_width, bin_width)
    centers = bins[:-1] + 0.5 * bin_width
    bw_s = bin_width / 1000.0
    if spikes_by_trial:
        per_trial_rates = []
        for tr in spikes_by_trial:
            tr = np.asarray(tr)
            counts, _ = np.histogram(tr, bins=bins)
            per_trial_rates.append(counts / bw_s)
        mean_rate = np.mean(per_trial_rates, axis=0)
    else:
        mean_rate = np.array([], dtype=float)

    smooth_ms = sim_cfg_local.get("avg_rate_curve_smooth_ms", 25.0)
    smooth_mode = sim_cfg_local.get("avg_rate_curve_smooth_mode", "center") or "center"
    centers, mean_rate = _smooth_rate_curve(
        centers,
        mean_rate,
        bin_width,
        smooth_ms,
        mode=str(smooth_mode),
    )
    try:
        smooth_ms_val = float(smooth_ms) if smooth_ms is not None else 0.0
    except Exception:
        smooth_ms_val = 0.0

    avg_rate_curve = {
        "bin_ms": bin_width,
        "smooth_ms": smooth_ms_val,
        "smooth_mode": str(smooth_mode),
        "t_ms": centers.tolist(),
        "rate_hz": mean_rate.tolist(),
    }

    result = {
        "mode": "multi",
        "sim_cfg": sim_cfg_local,
        "spikes": spikes_by_trial,
        "traces": traces_out,
        "cell_recordings_by_trial": cell_recordings_store if cell_recordings_store else None,
        "inputs_by_trial": inputs_store if inputs_store else None,
        "syn_records_by_trial": syn_records_by_trial if syn_records_by_trial else None,
        "meta": {
            "cell": sim_cfg_local.get("cell"),
            "tune": sim_cfg_local.get("tune"),
            "n_trials": n_trials,
            "trial_ids": list(range(n_trials)),
            "avg_rate_curve": avg_rate_curve,
            "input_summaries": input_summaries,
            "syn_config": copy.deepcopy(groups_cfg),
        },
    }
    if rm is not None:
        result["meta"]["randomness"] = rm.meta().as_dict()
    if snapshot_enabled:
        result["meta"]["snapshot"] = copy.deepcopy(snapshot_cfg)
        result["meta"]["versions"] = _collect_versions()
        result["meta"]["neuron_state"] = _collect_neuron_state()
        result["meta"]["env"] = _collect_env_snapshot()
        result["meta"]["mechanisms"] = _collect_mechanism_info(sim_cfg_local)
        if netcon_snapshot is not None:
            result["meta"]["netcon_snapshot"] = netcon_snapshot
        if syn_param_snapshot is not None:
            result["meta"]["synapse_param_snapshot"] = syn_param_snapshot
    if input_stats is not None:
        result["meta"]["input_stats"] = input_stats
    return result


def run_param(
    cell: Any,
    geom: Any,
    sim_cfg: Dict[str, Any],
    groups_cfg: Dict[str, Any],
    inputs_by_group: Dict[str, Any],
    *,
    rm: Optional[randomness.RandomnessManager] = None,
) -> Dict[str, Any]:
    """
    Placeholder for parametric study mode.

    Intended final shape:
      {
        "mode": "param",
        "sim_cfg": { ... },
        "param_study": { ... },
        "spikes": { param_val: [np.ndarray, ...], ... },
        "traces": { ... },
        "meta": { ... }
      }

    Not implemented yet.
    """
    raise NotImplementedError("Parametric mode is not implemented yet.")


# ---------------------------------------------------------------------
# unified entrypoint
# ---------------------------------------------------------------------

def run_sim(
    cell: Any,
    geom: Any,
    sim_cfg: Dict[str, Any],
    groups_cfg: Dict[str, Any],
    inputs_by_group: Dict[str, Any],
    mode_registry: Optional[Dict[str, Any]] = None,
    trial_callback: Optional[Any] = None,
    meta_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Unified entrypoint: infers mode from sim_cfg and dispatches to the
    appropriate run_* function.

    Mode inference:
      - 'param' if param_study.param_vals is non-empty
      - 'multi' if n_trials > 1
      - 'single' otherwise
    """
    
    sim_cfg_local = copy.deepcopy(sim_cfg)

    # If sim_cfg["load"] is a filename/path, load instead of running NEURON
    load_target = sim_cfg_local.get("load")
    load_enabled = sim_cfg_local.get("load_enabled", True)
    if load_enabled and load_target:
        p = Path(load_target)
        if not p.is_absolute():
            p = Path("output_data") / p  # interpret as relative to output_data/
        result = load_results(p)
        meta = result.get("meta", {})
        meta["loaded_from"] = str(p)
        result["meta"] = meta
        return result
    

    rm = randomness.RandomnessManager(sim_cfg_local)
    mode = _infer_mode(sim_cfg_local)

    if mode == "single":
        result = run_single(cell, geom, sim_cfg_local, groups_cfg, inputs_by_group, rm=rm)
    elif mode == "multi":
        result = run_multi(
            cell,
            geom,
            sim_cfg_local,
            groups_cfg,
            inputs_by_group,
            rm=rm,
            mode_registry=mode_registry,
            trial_callback=trial_callback,
        )
    elif mode == "param":
        result = run_param(cell, geom, sim_cfg_local, groups_cfg, inputs_by_group, rm=rm)
    else:
        raise ValueError(f"run_sim: unrecognized mode '{mode}'")

    # Record randomness metadata (e.g., auto-generated base seeds)
    meta = result.setdefault("meta", {})
    meta["randomness"] = rm.meta().as_dict()
    if meta_overrides:
        for key, value in meta_overrides.items():
            meta[key] = copy.deepcopy(value)

    # auto-save if sim_cfg['output'] is set
    save_results(result)  # no-op if output is None/empty
    return result

def summarize_results(results):
    mode = results["mode"]
    print(f"mode={mode}, n_traces_to_save={results['sim_cfg'].get('n_traces_to_save')}")

    if mode == "single":
        T = results["traces"].get("T", [])
        V = results["traces"].get("V", [])
        print(f"  single: len(T)={len(T)}, len(V)={len(V)}, n_spikes={len(results['spikes'])}")
    elif mode == "multi":
        spikes = results["spikes"]
        print(f"  multi: n_trials={len(spikes)}, spike_counts={[len(s) for s in spikes]}")
        if results["traces"]:
            print(f"  multi: traces V stored for {len(results['traces']['V'])} trial(s)")
