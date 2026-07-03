from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..core import randomness
from ..input_generation.processing import _build_default_mode_registry, _process_all_groups
from ..model import synapses
from .cell_runtime import run_cell
from .result_helpers import (
    _aggregate_input_stats,
    _resolve_inputs_to_save,
    _resolve_trace_trials_to_save,
    _smooth_rate_curve,
)
from .snapshots import (
    _apply_snapshot_deterministic,
    _collect_env_snapshot,
    _collect_mechanism_info,
    _collect_neuron_state,
    _collect_versions,
    _snapshot_cfg,
    _snapshot_netcon_state,
    _snapshot_synapse_params,
)
from .trial_helpers import (
    _as_bool,
    _clear_cell_state,
    _compute_input_stats_for_trial,
    _detect_spikes,
    _prepare_input_stats_bins,
    _set_trace_trials_to_save,
    _warn_preexisting_synapses,
)


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
        mode_registry = _build_default_mode_registry()
        try:
            from modules.input_generation import modes_user as input_modes_user

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
            inputs_trial = _process_all_groups(
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
