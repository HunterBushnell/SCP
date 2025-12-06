from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np


# =====================================================================
# Shared helpers for modes
# =====================================================================

def _get_n_syn(group_cfg: Dict[str, Any]) -> int:
    """
    Read the final synapse count for this group.

    Contract with inputs.py:
      - inputs._resolve_n_syn(...) runs before modes and writes
        syns["N_syn_resolved"] for active groups.
      - Modes should use that value when present.

    Fallback:
      - if N_syn_resolved is absent, fall back to syns["N_syn"] as integer.
    """
    syns = group_cfg.get("syns", {}) or {}

    # Preferred path: use N_syn_resolved if set by inputs._resolve_n_syn
    if "N_syn_resolved" in syns and syns["N_syn_resolved"] is not None:
        try:
            n = int(syns["N_syn_resolved"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"N_syn_resolved must be integer-like "
                f"(got {syns['N_syn_resolved']!r})"
            ) from exc
        if n < 0:
            raise ValueError(f"N_syn_resolved must be >= 0 (got {n})")
        return n

    # Fallback: raw N_syn (for safety / testing)
    n_syn = syns.get("N_syn")
    if n_syn is None:
        # If nothing is specified, treat as 1 synapse-equivalent; this is mainly
        # for testing and should not happen in the normal pipeline where
        # _resolve_n_syn runs first.
        return 1

    try:
        n_syn_int = int(n_syn)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"syns['N_syn'] must be integer-like (got {n_syn!r})"
        ) from exc

    if n_syn_int < 0:
        raise ValueError(f"syns['N_syn'] must be >= 0 (got {n_syn_int})")

    return n_syn_int

def _get_active_window_from_time_cfg(time_cfg: Dict[str, Any]) -> Tuple[float, float]:
    """
    Derive an overall active window [t_start, t_end] in ms from time_cfg.

    Rules:
      - Use time_cfg["blocks"] if present:
          * collect all blocks with kind != "quiescent"
          * t_start = min(t_start) over those blocks
          * t_end   = max(t_end)   over those blocks
      - If there are no non-quiescent blocks, fall back to anchors:
          [sim_tstart, sim_tstop].
    """
    anchors = (time_cfg or {}).get("anchors", {}) or {}
    blocks = (time_cfg or {}).get("blocks", []) or []

    non_quiescent = [b for b in blocks if b.get("kind") != "quiescent"]

    if non_quiescent:
        t_start = min(float(b["t_start"]) for b in non_quiescent)
        t_end = max(float(b["t_end"]) for b in non_quiescent)
        # Degenerate protection
        if t_end <= t_start:
            return t_start, t_start
        return t_start, t_end

    # Fallback: anchors only
    sim_tstart = float(anchors.get("sim_tstart", 0.0))
    sim_tstop = float(anchors.get("sim_tstop", sim_tstart))
    if sim_tstop <= sim_tstart:
        return sim_tstart, sim_tstart
    return sim_tstart, sim_tstop

# ---------------------------------------------------------------------
# Shared helper: homogeneous Poisson spike train generator
# ---------------------------------------------------------------------
def _generate_homogeneous_poisson_trains(
    rate_hz: float,
    t_start_ms: float,
    t_end_ms: float,
    n_syn: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Generate homogeneous Poisson spike trains for n_syn independent sources.
    """
    trains: List[np.ndarray] = []

    if n_syn <= 0:
        return trains

    if rate_hz <= 0.0 or t_end_ms <= t_start_ms:
        # Valid config but no spikes: return n_syn empty trains
        return [np.array([], dtype=float) for _ in range(n_syn)]

    # Mean inter-spike interval (ms)
    mean_isi_ms = 1000.0 / float(rate_hz)

    for _ in range(n_syn):
        t = float(t_start_ms)
        spikes: List[float] = []

        # Standard thinning-free homogeneous Poisson in continuous time
        while True:
            isi = rng.exponential(mean_isi_ms)
            t += isi
            if t > t_end_ms:
                break
            spikes.append(t)

        trains.append(np.asarray(spikes, dtype=float))

    return trains

# ---------------------------------------------------------------------
# Shared helper: inhomogeneous Poisson spike train generator
# ---------------------------------------------------------------------
def _generate_inhomogeneous_from_curve(
    rates_hz: np.ndarray,
    t0_ms: float,
    bin_ms: float,
    n_syn: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Generate inhomogeneous Poisson spike trains from a piecewise-constant
    rate curve.
    """
    rates_hz = np.asarray(rates_hz, dtype=float).ravel()
    trains: List[np.ndarray] = []

    if n_syn <= 0 or rates_hz.size == 0:
        return [np.array([], dtype=float) for _ in range(max(n_syn, 0))]

    if bin_ms <= 0.0:
        raise ValueError(f"bin_ms must be > 0, got {bin_ms!r}")

    bin_ms = float(bin_ms)
    n_bins = rates_hz.size

    # λ_k = rate_k * bin_ms / 1000 (ms → s)
    lam_per_bin = rates_hz * (bin_ms / 1000.0)

    for _ in range(n_syn):
        spikes: List[float] = []

        for k in range(n_bins):
            lam_k = lam_per_bin[k]
            if lam_k <= 0.0:
                continue

            count = rng.poisson(lam_k)
            if count <= 0:
                continue

            bin_start = t0_ms + k * bin_ms
            offsets = rng.uniform(0.0, bin_ms, size=count)
            times = bin_start + offsets
            spikes.extend(times.tolist())

        if spikes:
            spikes_arr = np.sort(np.asarray(spikes, dtype=float))
        else:
            spikes_arr = np.array([], dtype=float)

        trains.append(spikes_arr)

    return trains


# =====================================================================
# Default mode functions
# =====================================================================

# ---------------------------------------------------------------------
# Mode: precomputed
# ---------------------------------------------------------------------
def _mode_precomputed(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
    geometry: Optional[Any],
    time_cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Built-in handler for mode == "precomputed".

    SOURCE CONTRACT
    ---------------
    Expected `group_cfg["source"]`:
        {
            "trains": [[...], [...], ...],         # optional inline trains (ms, sim-time)
            "path": "pn_precomputed_trains.json"   # OR JSON file with trains
        }

    The active window used for clipping is derived from time_cfg
    via _get_active_window_from_time_cfg (non-quiescent blocks).
    """
    source = group_cfg.get("source", {}) or {}

    trains_raw = source.get("trains")
    path = source.get("path")

    if trains_raw is None and path is None:
        raise ValueError(
            "Mode 'precomputed' requires either source['trains'] (inline) "
            "or source['path'] (JSON file with spike trains)."
        )

    # If trains are not provided inline, load from JSON file
    if trains_raw is None and path is not None:
        path_obj = Path(path)
        with path_obj.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "trains" in data:
            trains_raw = data["trains"]
        elif isinstance(data, list):
            trains_raw = data
        else:
            raise ValueError(
                f"Precomputed file {path_obj} must be either "
                f'{{"trains": [...]}} or a raw [[...], [...], ...] list.'
            )

    if not isinstance(trains_raw, list):
        raise TypeError(
            "source['trains'] / loaded trains must be a list of lists of spike times."
        )

    # Resolve number of synapses/trains
    n_syn = _get_n_syn(group_cfg)
    if n_syn <= 0:
        return []

    # Ensure we have a list-of-lists
    trains_list: List[List[float]] = []
    for idx, tr in enumerate(trains_raw):
        if tr is None:
            trains_list.append([])
        elif isinstance(tr, (list, tuple)):
            trains_list.append(list(tr))
        else:
            raise TypeError(
                f"Train {idx} in precomputed source is not list-like; got {type(tr)!r}"
            )

    # If more trains are provided than needed, truncate
    if len(trains_list) >= n_syn:
        trains_list = trains_list[:n_syn]
    # If fewer trains than N_syn are provided, we return what we have;
    # 2.3 will raise on the length mismatch.

    t_start, t_end = _get_active_window_from_time_cfg(time_cfg)

    trains: List[np.ndarray] = []
    for tr in trains_list:
        arr = np.asarray(tr, dtype=float).ravel()
        if arr.size == 0:
            trains.append(arr)
            continue
        mask = (arr >= t_start) & (arr <= t_end)
        arr = np.sort(arr[mask])
        trains.append(arr)

    return trains


# ---------------------------------------------------------------------
# Mode: homogeneous_poisson
# ---------------------------------------------------------------------
def _mode_homogeneous_poisson(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
    geometry: Optional[Any],
    time_cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Built-in handler for mode == "homogeneous_poisson".

    SOURCE CONTRACT
    ---------------
    Expected `group_cfg["source"]` keys:
        {
            "freq": float,   # firing rate in Hz (used if no timing baseline)
            ...
        }

    TIMING
    ------
    - Uses time_cfg["anchors"]["baseline_rate_hz"] if present.
    - Falls back to source['freq'] if baseline_rate_hz is None.
    - Active window is derived from time_cfg via _get_active_window_from_time_cfg.
    """
    anchors = (time_cfg or {}).get("anchors", {}) or {}
    baseline_rate = anchors.get("baseline_rate_hz", None)

    source = group_cfg.get("source", {}) or {}
    freq = source.get("freq", None)

    # Resolve rate:
    #   1) prefer timing-derived baseline_rate_hz if present,
    #   2) else fall back to source['freq'].
    if baseline_rate is not None:
        try:
            rate_hz = float(baseline_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Mode 'homogeneous_poisson' got non-numeric baseline_rate_hz={baseline_rate!r}."
            ) from exc
    else:
        if freq is None:
            raise ValueError(
                "Mode 'homogeneous_poisson' requires either anchors['baseline_rate_hz'] "
                "or source['freq'] (Hz)."
            )
        try:
            rate_hz = float(freq)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Mode 'homogeneous_poisson' expects numeric source['freq'], got {freq!r}."
            ) from exc

    # Resolve number of synapses / trains
    n_syn = _get_n_syn(group_cfg)
    if n_syn <= 0:
        return []

    # Time window for this group from time_cfg
    t_start_ms, t_end_ms = _get_active_window_from_time_cfg(time_cfg)

    # Generate Poisson trains
    trains = _generate_homogeneous_poisson_trains(
        rate_hz=rate_hz,
        t_start_ms=t_start_ms,
        t_end_ms=t_end_ms,
        n_syn=n_syn,
        rng=rng,
    )

    return trains


# ---------------------------------------------------------------------
# Mode: inhomogeneous_poisson (stub)
# ---------------------------------------------------------------------
def _mode_inhomogeneous_poisson(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
    geometry: Optional[Any],
    time_cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Built-in handler for mode == "inhomogeneous_poisson".

    Currently only loads and validates the rate curve; alignment + generation
    will later be based on time_cfg (anchors + blocks).
    """
    source = group_cfg.get("source", {}) or {}
    path = source.get("path")
    time_col = source.get("time_col")
    rate_col = source.get("rate_col")
    bin_ms = source.get("bin_ms")
    baseline = source.get("baseline")

    if path is None:
        raise ValueError("Mode 'inhomogeneous_poisson' requires source['path'].")

    if not time_col or not rate_col:
        raise ValueError(
            "Mode 'inhomogeneous_poisson' requires source['time_col'] and source['rate_col'] "
            "to identify columns/keys in the rate-curve file."
        )

    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Rate curve file not found: {path_obj}")

    # --- Load rate curve ---

    if path_obj.suffix.lower() in {".csv", ".txt"}:
        data = np.genfromtxt(path_obj, delimiter=",", names=True)
        try:
            times_ms = np.asarray(data[time_col], dtype=float)
            rates_hz = np.asarray(data[rate_col], dtype=float)
        except ValueError as exc:
            raise KeyError(
                f"Columns {time_col!r} and/or {rate_col!r} not found in CSV header of {path_obj}."
            ) from exc
    else:
        with path_obj.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"Inhomogeneous rate file {path_obj} must be a dict-like JSON "
                f"containing {time_col!r} and {rate_col!r} arrays."
            )
        try:
            times_ms = np.asarray(data[time_col], dtype=float)
            rates_hz = np.asarray(data[rate_col], dtype=float)
        except KeyError as exc:
            raise KeyError(
                f"JSON in {path_obj} must contain keys {time_col!r} and {rate_col!r}."
            ) from exc

    if times_ms.ndim != 1 or rates_hz.ndim != 1 or times_ms.shape != rates_hz.shape:
        raise ValueError(
            "Rate curve arrays 'time_col' and 'rate_col' must be 1D and of equal length."
        )

    # Sort by time
    order = np.argsort(times_ms)
    times_ms = times_ms[order]
    rates_hz = rates_hz[order]

    # Infer bin width if not provided
    if bin_ms is None:
        if times_ms.size < 2:
            raise ValueError(
                "Cannot infer bin_ms from a single-point rate curve; "
                "please specify source['bin_ms']."
            )
        dt = np.median(np.diff(times_ms))
        bin_ms = float(dt)
    else:
        bin_ms = float(bin_ms)

    # Default baseline: first rate value if not explicitly provided
    if baseline is None:
        baseline = float(rates_hz[0])
    else:
        baseline = float(baseline)

    # Later we will combine:
    #   - this curve (times_ms, rates_hz, bin_ms, baseline)
    #   - time_cfg["anchors"] / ["blocks"]
    #   - N_syn from group_cfg
    # to generate inhomogeneous trains.
    raise NotImplementedError(
        "Mode 'inhomogeneous_poisson' has loaded the rate curve but the "
        "time-alignment and spike-train generation logic are not yet implemented."
    )


# =====================================================================
# Registry
# =====================================================================

def get_default_mode_registry() -> Dict[str, Any]:
    """
    Return the default mode registry for Step 2.3.
    """
    return {
        "homogeneous_poisson": _mode_homogeneous_poisson,
        "precomputed": _mode_precomputed,
        "inhomogeneous_poisson": _mode_inhomogeneous_poisson,
    }
