"""
Utilities for sampling generated inputs (Step 5.2.3) to build summary
firing-rate curves for a synapse group.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import pandas as pd

from . import inputs as inputs_mod
from . import input_modes_core


def _build_mode_registry():
    reg = input_modes_core.get_default_mode_registry()
    try:
        from . import input_modes_user  # type: ignore

        user_reg = input_modes_user.get_user_mode_registry()
        reg = {**reg, **user_reg}
    except Exception:
        pass
    return reg


def _bin_trains(trains, tstart, tstop, bin_ms):
    edges = np.arange(tstart, tstop + bin_ms, bin_ms, dtype=float)
    if edges.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    counts = np.zeros(edges.size - 1, dtype=float)
    for tr in trains:
        if len(tr) == 0:
            continue
        c, _ = np.histogram(tr, bins=edges)
        counts += c
    n_syn = max(len(trains), 1)
    rate = counts / (n_syn * (bin_ms / 1000.0))  # Hz per-synapse average
    centers = edges[:-1] + bin_ms * 0.5
    return centers, rate


def _load_reference_curve(groups_cfg, sim_cfg, group_name, bin_ms):
    """Reconstruct the rate curve actually used for the source block."""
    gcfg = groups_cfg.get(group_name, {}) or {}
    source = gcfg.get("source", {}) or {}
    path = source.get("path")
    if not path:
        return None

    time_col = source.get("time_col") or "Time"
    rate_col = source.get("rate_col") or "AvgFiringRate"

    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if time_col not in df or rate_col not in df:
        return None

    times_ms = np.asarray(df[time_col], dtype=float) * 1000.0
    rates_hz = np.asarray(df[rate_col], dtype=float)
    keep = times_ms >= 0.0
    times_ms = times_ms[keep]
    rates_hz = rates_hz[keep]
    if times_ms.size == 0:
        return None
    times_ms = times_ms - times_ms[0]

    # Timing
    try:
        time_cfg = inputs_mod._calculate_timing(sim_cfg, gcfg)
    except Exception:
        return None
    blocks = time_cfg.get("blocks", []) or []
    anchors = time_cfg.get("anchors", {}) or {}
    baseline_rate = anchors.get("baseline_rate_hz", None)

    src_blocks = [b for b in blocks if b.get("kind") == "source"]
    if not src_blocks:
        return None
    b = src_blocks[0]
    t0 = float(b["t_start"])
    t1 = float(b["t_end"])
    duration = max(0.0, t1 - t0)
    if duration <= 0.0:
        return None

    n_bins_needed = int(np.ceil(duration / bin_ms))
    if n_bins_needed <= 0:
        return None

    avail = min(rates_hz.size, n_bins_needed)
    rates_block = rates_hz[:avail]
    if avail < n_bins_needed:
        pad_rate = float(baseline_rate) if (baseline_rate is not None) else 0.0
        pad = np.full(n_bins_needed - avail, pad_rate, dtype=float)
        rates_block = np.concatenate([rates_block, pad])

    centers = t0 + (np.arange(n_bins_needed) + 0.5) * bin_ms
    return centers, rates_block


def sample_group_rates(
    tune_dir: Path,
    group: str,
    runs: int = 10,
    bin_ms: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any], Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    Generate inputs multiple times for a single group (all others disabled),
    bin spike trains into rates, and return mean/std.

    Returns:
        centers_ms, mean_rate, std_rate, sim_cfg, meta, ref_curve
        where ref_curve is (t_ms, rate_hz) or None.
    """
    tune_dir = Path(tune_dir).expanduser().resolve()
    config_root = inputs_mod._resolve_config_root(tune_dir)
    sim_path = config_root / "sim_config.json"
    syn_path = config_root / "syn_config.json"
    if not sim_path.is_file() or not syn_path.is_file():
        raise FileNotFoundError(f"Missing sim_config.json or syn_config.json in {config_root}.")

    with sim_path.open("r") as f:
        sim_raw = json.load(f)
    with syn_path.open("r") as f:
        groups_raw = json.load(f)

    groups_raw = inputs_mod._expand_group_includes(groups_raw, config_root)
    if group not in groups_raw:
        raise KeyError(f"Group {group!r} not found in syn_config.json")

    # Deactivate others
    for gname, gcfg in groups_raw.items():
        gcfg["state"] = (gname == group)

    sim_cfg = inputs_mod._normalize_sim_config(sim_raw)
    inputs_mod._inject_path_metadata(sim_cfg, config_root)
    groups_cfg = inputs_mod._normalize_group_configs(groups_raw)

    mode_registry = _build_mode_registry()

    # Determine bin size
    if bin_ms is None:
        bin_ms = groups_cfg.get(group, {}).get("source", {}).get("bin_ms", None)
    bin_ms = float(bin_ms) if bin_ms else 5.0

    def _gen_inputs_with_rng(rng):
        return inputs_mod._process_all_groups(
            sim_cfg=sim_cfg,
            groups_cfg=groups_cfg.copy(),
            geometry=None,
            mode_registry=mode_registry,
            rng=rng,
            trial_rng=None,
        )

    rng0 = np.random.default_rng(seed)
    inputs_by_group = _gen_inputs_with_rng(rng0)
    gi0 = inputs_by_group.get(group)
    if gi0 is None:
        raise KeyError(f"Group {group!r} not present/active after normalization.")

    n_syn = len(gi0.spike_trains)
    tstart = float(sim_cfg["tstart"])
    tstop = float(sim_cfg["tstop"])

    centers, rate0 = _bin_trains(gi0.spike_trains, tstart, tstop, bin_ms)
    rates = [rate0]

    for i in range(1, max(runs, 1)):
        rng = np.random.default_rng() if seed is None else np.random.default_rng(
            np.random.SeedSequence([seed, i])
        )
        inputs_by_group = _gen_inputs_with_rng(rng)
        gi = inputs_by_group.get(group)
        if gi is None:
            raise KeyError(f"Group {group!r} missing in run {i}.")
        _, rate = _bin_trains(gi.spike_trains, tstart, tstop, bin_ms)
        rates.append(rate)

    rates_arr = np.vstack(rates)
    mean_rate = rates_arr.mean(axis=0)
    std_rate = rates_arr.std(axis=0)

    ref_curve = _load_reference_curve(groups_cfg, sim_cfg, group, bin_ms)

    meta = {
        "n_runs": len(rates),
        "n_syn": n_syn,
        "bin_ms": bin_ms,
        "group": group,
        "tune_dir": str(tune_dir),
        "seed": seed,
    }
    return centers, mean_rate, std_rate, sim_cfg, meta, ref_curve
