from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _resolve_trace_trials_to_save(sim_cfg: Dict[str, Any], fallback: int) -> int:
    raw = None
    cell_rec = sim_cfg.get("cell_recording")
    if isinstance(cell_rec, dict):
        raw = cell_rec.get("n_trials", None)
    if raw is None:
        raw = sim_cfg.get("n_traces_to_save", fallback)
    try:
        n = int(raw)
    except Exception:
        return max(0, int(fallback))
    if n < 0:
        return max(0, int(fallback))
    return max(0, n)


def _resolve_inputs_to_save(sim_cfg: Dict[str, Any], n_trials: int, fallback: int) -> int:
    raw = sim_cfg.get("n_inputs_to_save", None)
    if raw is None:
        raw = _resolve_trace_trials_to_save(sim_cfg, fallback=fallback)
    if isinstance(raw, str):
        if raw.strip().lower() in ("all",):
            return max(0, int(n_trials))
        try:
            raw = int(raw)
        except Exception:
            return max(0, int(fallback))
    try:
        raw = int(raw)
    except Exception:
        return max(0, int(fallback))
    if raw < 0:
        return max(0, int(n_trials))
    return max(0, raw)


def _smooth_rate_curve(
    centers: np.ndarray,
    rates: np.ndarray,
    bin_ms: float,
    smooth_ms: Optional[float],
    *,
    mode: str = "center",
) -> Tuple[np.ndarray, np.ndarray]:
    if smooth_ms is None:
        return centers, rates
    try:
        smooth_ms = float(smooth_ms)
    except Exception:
        return centers, rates
    if smooth_ms <= 0 or bin_ms <= 0:
        return centers, rates

    k = int(round(smooth_ms / bin_ms))
    if k <= 1 or rates.size < k:
        return centers, rates
    if mode == "center" and k % 2 == 0:
        k += 1

    kernel = np.ones(k, dtype=float) / float(k)
    if mode == "center":
        y = np.convolve(rates, kernel, mode="valid")
        drop = (len(centers) - len(y)) // 2
        if drop < 0:
            return centers, rates
        return centers[drop : drop + len(y)], y
    if mode == "causal":
        pad = (k - 1, 0)
        y = np.convolve(np.pad(rates, pad), kernel, mode="valid")
        return centers[: len(y)], y
    return centers, rates


def _aggregate_input_stats(trial_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trial_stats:
        return {}

    group_names = set(trial_stats[0].get("groups", {}).keys())
    group_means: Dict[str, Any] = {}

    for g in group_names:
        n_syns: List[int] = []
        totals: List[int] = []
        rate_totals: List[float] = []
        rate_per_syns: List[float] = []
        counts_stack: List[List[float]] = []
        rate_bin_total_stack: List[List[float]] = []
        rate_bin_per_syn_stack: List[List[float]] = []

        for trial in trial_stats:
            gstats = trial.get("groups", {}).get(g)
            if not gstats:
                continue
            n_syns.append(int(gstats.get("n_syn", 0)))
            totals.append(int(gstats.get("total_spikes", 0)))
            rate_totals.append(float(gstats.get("rate_hz_total", 0.0)))
            rate_per_syns.append(float(gstats.get("rate_hz_per_syn", 0.0)))
            counts_stack.append(list(gstats.get("counts_by_bin", [])))
            rate_bin_total_stack.append(list(gstats.get("rate_hz_by_bin_total", [])))
            rate_bin_per_syn_stack.append(list(gstats.get("rate_hz_by_bin_per_syn", [])))

        if not totals:
            continue

        mean_counts = np.mean(np.asarray(counts_stack, dtype=float), axis=0).tolist() if counts_stack else []
        mean_rate_bin_total = (
            np.mean(np.asarray(rate_bin_total_stack, dtype=float), axis=0).tolist()
            if rate_bin_total_stack
            else []
        )
        mean_rate_bin_per_syn = (
            np.mean(np.asarray(rate_bin_per_syn_stack, dtype=float), axis=0).tolist()
            if rate_bin_per_syn_stack
            else []
        )

        group_means[g] = {
            "n_syn": int(n_syns[0]) if n_syns else 0,
            "mean_total_spikes": float(np.mean(totals)),
            "mean_rate_hz_total": float(np.mean(rate_totals)),
            "mean_rate_hz_per_syn": float(np.mean(rate_per_syns)),
            "mean_counts_by_bin": mean_counts,
            "mean_rate_hz_by_bin_total": mean_rate_bin_total,
            "mean_rate_hz_by_bin_per_syn": mean_rate_bin_per_syn,
        }

    return group_means


