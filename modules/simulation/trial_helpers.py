from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


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
