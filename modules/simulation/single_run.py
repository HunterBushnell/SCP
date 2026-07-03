from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np

from ..core import randomness
from ..model import synapses
from .cell_runtime import run_cell
from .result_helpers import (
    _aggregate_input_stats,
    _resolve_inputs_to_save,
    _resolve_trace_trials_to_save,
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
