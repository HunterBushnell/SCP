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
      - Modes should use that value when present, via this helper.

    Fallback:
      - If N_syn_resolved is absent, fall back to syns["N_syn"] as integer.
        This is mainly for testing / non-geometry cases.
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

    Usage:
      - Modes obtain time_cfg from group_cfg["time_cfg"] and pass it here.

    Rules:
      - Use time_cfg["blocks"] if present:
          * collect all blocks with kind != "quiescent"
          * t_start = min(block["t_start"]) over those blocks
          * t_end   = max(block["t_end"])   over those blocks
      - If there are no non-quiescent blocks, fall back to anchors:
          [sim_tstart, sim_tstop].

    Notes:
      - This is mainly a convenience for modes (e.g. precomputed) that only
        need a single “union” active window and don’t care about individual
        blocks; more complex modes should work block-by-block instead.
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

    This helper is mode-agnostic; modes are responsible for:
      - choosing (rates_hz, t0_ms, bin_ms) consistent with their time_cfg,
      - splitting/combining per-block segments as needed,
      - stitching segments per synapse in chronological order.
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
# Default mode functions (stubs for 2.3 contract alignment)
# =====================================================================

# ---------------------------------------------------------------------
# Mode: precomputed (STUB)
# ---------------------------------------------------------------------
def _mode_precomputed(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
    geometry: Optional[Any],
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Precomputed spike trains sampled from stored data.

    Source options:
      - source["trains"]: inline list of spike-time arrays (ms, starting at 0)
      - source["path"]: path to file containing trains. Supported:
          * .pkl/.p: pickled list, or pickled dict with a single key whose value is the list
          * .json: either {"trains": [...]} or raw list

    Timing:
      - Use time_cfg blocks; "source" blocks place sampled trains shifted to block start.
      - Baseline blocks use baseline_rate_hz (anchors) to generate homogeneous Poisson if set.
      - Quiescent blocks add nothing.
    """
    time_cfg = (group_cfg or {}).get("time_cfg") or {}
    anchors = time_cfg.get("anchors", {}) or {}
    blocks = time_cfg.get("blocks", []) or []

    try:
        sim_tstart = float(sim_cfg["tstart"])
        sim_tstop = float(sim_cfg["tstop"])
    except Exception as exc:
        raise ValueError("sim_cfg must contain tstart and tstop for precomputed mode") from exc

    n_syn = _get_n_syn(group_cfg)
    if n_syn <= 0:
        return []

    source = group_cfg.get("source", {}) or {}

    def _load_trains_from_path(p: Path) -> List[np.ndarray]:
        if not p.is_file():
            raise FileNotFoundError(f"precomputed: file not found {p}")
        suffix = p.suffix.lower()
        if suffix in (".pkl", ".p"):
            import pickle

            with p.open("rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and len(obj) == 1:
                obj = next(iter(obj.values()))
            if isinstance(obj, list):
                return [np.asarray(x, dtype=float) for x in obj]
            raise ValueError(f"precomputed: unsupported pickle structure in {p}")
        if suffix == ".json":
            with p.open("r") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and "trains" in obj:
                obj = obj["trains"]
            if isinstance(obj, list):
                return [np.asarray(x, dtype=float) for x in obj]
            raise ValueError(f"precomputed: unsupported JSON structure in {p}")
        raise ValueError(f"precomputed: unsupported file type {p.suffix} for {p}")

    if source.get("trains") is not None:
        pool = [np.asarray(x, dtype=float) for x in source["trains"]]
    elif source.get("path"):
        pool = _load_trains_from_path(Path(source["path"]))
    else:
        raise ValueError("precomputed mode requires source['trains'] or source['path']")

    if not pool:
        return [np.array([], dtype=float) for _ in range(n_syn)]

    # helper: sample trains for n_syn
    pool_size = len(pool)
    def _sample_trains():
        if n_syn == pool_size:
            idx = np.arange(pool_size)
        elif n_syn < pool_size:
            idx = rng.choice(pool_size, size=n_syn, replace=False)
        else:
            idx = rng.choice(pool_size, size=n_syn, replace=True)
        return [np.asarray(pool[i], dtype=float).copy() for i in idx]

    baseline_rate = anchors.get("baseline_rate_hz", None)
    trains_accum: List[List[float]] = [[] for _ in range(n_syn)]

    for block in blocks:
        kind = block.get("kind")
        t0 = float(block.get("t_start", sim_tstart))
        t1 = float(block.get("t_end", t0))
        if t1 <= t0:
            continue
        t0 = max(t0, sim_tstart)
        t1 = min(t1, sim_tstop)
        if t1 <= t0:
            continue

        if kind == "quiescent":
            continue

        if kind == "baseline":
            rate = baseline_rate
            if rate is None or rate <= 0.0:
                continue
            seg_trains = _generate_homogeneous_poisson_trains(
                rate_hz=float(rate),
                t_start_ms=t0,
                t_end_ms=t1,
                n_syn=n_syn,
                rng=rng,
            )
        elif kind == "source":
            sampled = _sample_trains()
            seg_trains = []
            for tr in sampled:
                shifted = tr + t0
                clipped = shifted[(shifted >= t0) & (shifted <= t1)]
                seg_trains.append(clipped)
        else:
            continue

        for i in range(n_syn):
            trains_accum[i].extend(seg_trains[i].tolist())

    out: List[np.ndarray] = []
    for spikes in trains_accum:
        if not spikes:
            out.append(np.array([], dtype=float))
            continue
        arr = np.asarray(spikes, dtype=float)
        arr = arr[(arr >= sim_tstart) & (arr <= sim_tstop)]
        arr.sort()
        out.append(arr)

    return out


# ---------------------------------------------------------------------
# Mode: homogeneous_poisson
# ---------------------------------------------------------------------
def _mode_homogeneous_poisson(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
    geometry: Optional[Any],
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Built-in handler for mode == "homogeneous_poisson".

    Semantics:
      - Pure constant-rate Poisson drive.
      - Uses a single rate taken from source["freq"] (Hz).
      - Respects the onset anchor: drives from max(onset, tstart) to tstop.
      - Does not use per-block structure (baseline/source blocks are ignored).

    Requirements:
      - source["freq"] must be float-like; if not, raise ValueError.
      - If freq <= 0 or effective window has no duration, return n_syn empty trains.
      - n_syn is obtained via _get_n_syn(group_cfg).
      - All spikes lie in [tstart, tstop] (ms).
    """
    # Resolve sim window
    try:
        sim_tstart = float(sim_cfg["tstart"])
        sim_tstop  = float(sim_cfg["tstop"])
    except KeyError as exc:
        raise KeyError(
            f"sim_cfg is missing required key {exc!r} for homogeneous_poisson mode"
        ) from exc

    # Resolve anchors from group_cfg["time_cfg"], if present
    time_cfg = group_cfg.get("time_cfg") or {}
    anchors  = time_cfg.get("anchors", {}) or {}
    onset    = float(anchors.get("onset", sim_tstart))

    # Effective window: from onset (clipped to sim start) to sim end
    t_start_ms = max(onset, sim_tstart)
    t_end_ms   = sim_tstop

    # Resolve synapse count
    n_syn = _get_n_syn(group_cfg)

    # Resolve constant rate from source["freq"]
    source = (group_cfg or {}).get("source", {}) or {}
    if "freq" not in source:
        raise KeyError(
            "homogeneous_poisson mode requires source['freq'] (Hz); "
            "no 'freq' key found in source config"
        )

    try:
        rate_hz = float(source["freq"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"homogeneous_poisson mode requires source['freq'] to be float-like; "
            f"got {source['freq']!r}"
        ) from exc

    # Degenerate cases: no time or no rate => n_syn empty trains
    if n_syn <= 0 or t_end_ms <= t_start_ms or rate_hz <= 0.0:
        return [np.array([], dtype=float) for _ in range(max(n_syn, 0))]

    # Generate homogeneous Poisson trains over [t_start_ms, t_end_ms]
    trains = _generate_homogeneous_poisson_trains(
        rate_hz=rate_hz,
        t_start_ms=t_start_ms,
        t_end_ms=t_end_ms,
        n_syn=n_syn,
        rng=rng,
    )

    return trains

# ---------------------------------------------------------------------
# Mode: inhomogeneous_poisson (STUB)
# ---------------------------------------------------------------------
def _mode_inhomogeneous_poisson(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
    geometry: Optional[Any],
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Inhomogeneous Poisson driven by a rate curve (CSV).

    Assumptions for this implementation (current SST2 use case):
      - source["path"] points to a CSV with columns time_col (seconds) and rate_col (Hz).
      - time_col defaults to "Time", rate_col defaults to "AvgFiringRate".
      - Times < 0 are dropped; remaining times are shifted so the first sample is at 0 ms.
      - bin_ms is taken from source["bin_ms"] if provided; otherwise inferred from median Δt.
      - Baseline blocks use anchors["baseline_rate_hz"] (numeric) if present; otherwise quiescent.
      - Source blocks use the rate curve, truncated/padded to the block duration
        (padding uses baseline_rate_hz or 0.0 if absent).
    """
    time_cfg = (group_cfg or {}).get("time_cfg") or {}
    anchors = time_cfg.get("anchors", {}) or {}
    blocks = time_cfg.get("blocks", []) or []

    try:
        sim_tstart = float(sim_cfg["tstart"])
        sim_tstop = float(sim_cfg["tstop"])
    except Exception as exc:
        raise ValueError("sim_cfg must contain tstart and tstop for inhomogeneous_poisson") from exc

    n_syn = _get_n_syn(group_cfg)
    if n_syn <= 0:
        return []

    source = group_cfg.get("source", {}) or {}
    path = source.get("path")
    if not path:
        raise ValueError("inhomogeneous_poisson requires source['path'] to a rate curve CSV")

    time_col = source.get("time_col") or "Time"
    rate_col = source.get("rate_col") or "AvgFiringRate"
    bin_ms_cfg = source.get("bin_ms", None)

    # Load curve
    import pandas as pd  # local import to avoid forcing pandas on import

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Rate curve file not found: {p}")

    df = pd.read_csv(p)
    if time_col not in df or rate_col not in df:
        raise ValueError(f"Rate curve file {p} missing required columns {time_col!r}/{rate_col!r}")

    times_ms = np.asarray(df[time_col], dtype=float) * 1000.0  # seconds → ms
    rates_hz = np.asarray(df[rate_col], dtype=float)

    # Drop times < 0 and shift so first sample is at 0
    keep = times_ms >= 0.0
    times_ms = times_ms[keep]
    rates_hz = rates_hz[keep]
    if times_ms.size == 0:
        raise ValueError(f"Rate curve {p} has no samples with time >= 0 ms after clipping.")
    times_ms = times_ms - times_ms[0]

    # Determine bin_ms
    if bin_ms_cfg is not None:
        try:
            bin_ms = float(bin_ms_cfg)
        except Exception as exc:
            raise ValueError(f"source['bin_ms'] must be numeric (got {bin_ms_cfg!r})") from exc
    else:
        if times_ms.size < 2:
            raise ValueError("Cannot infer bin_ms from a single time sample; specify source['bin_ms'].")
        diffs = np.diff(times_ms)
        bin_ms = float(np.median(diffs))
    if bin_ms <= 0.0:
        raise ValueError(f"bin_ms must be > 0 (got {bin_ms!r})")

    # Build per-synapse accumulators
    trains: List[List[float]] = [[] for _ in range(n_syn)]
    baseline_rate = anchors.get("baseline_rate_hz", None)

    # Precompute truncated/padded rate slices for any source block
    rates_len = rates_hz.size

    for block in blocks:
        kind = block.get("kind")
        t0 = float(block.get("t_start", sim_tstart))
        t1 = float(block.get("t_end", t0))
        if t1 <= t0:
            continue
        # Clamp to simulation window just in case
        t0 = max(t0, sim_tstart)
        t1 = min(t1, sim_tstop)
        if t1 <= t0:
            continue

        if kind == "quiescent":
            continue

        if kind == "baseline":
            rate = baseline_rate
            if rate is None or rate <= 0.0:
                continue
            seg_trains = _generate_homogeneous_poisson_trains(
                rate_hz=float(rate),
                t_start_ms=t0,
                t_end_ms=t1,
                n_syn=n_syn,
                rng=rng,
            )
        elif kind == "source":
            duration = t1 - t0
            n_bins_needed = int(np.ceil(duration / bin_ms))
            if n_bins_needed <= 0:
                continue

            # Use available curve up to n_bins_needed bins; pad remainder with baseline or zeros
            avail_bins = min(rates_len, n_bins_needed)
            rates_block = rates_hz[:avail_bins]
            if avail_bins < n_bins_needed:
                pad_rate = float(baseline_rate) if (baseline_rate is not None) else 0.0
                pad = np.full(n_bins_needed - avail_bins, pad_rate, dtype=float)
                rates_block = np.concatenate([rates_block, pad])

            seg_trains = _generate_inhomogeneous_from_curve(
                rates_hz=np.asarray(rates_block, dtype=float),
                t0_ms=t0,
                bin_ms=bin_ms,
                n_syn=n_syn,
                rng=rng,
            )
        else:
            continue

        # Accumulate
        for i in range(n_syn):
            trains[i].extend(seg_trains[i].tolist())

    # Finalize: clip to sim window and sort
    out: List[np.ndarray] = []
    for spikes in trains:
        if not spikes:
            out.append(np.array([], dtype=float))
            continue
        arr = np.asarray(spikes, dtype=float)
        arr = arr[(arr >= sim_tstart) & (arr <= sim_tstop)]
        if arr.size == 0:
            out.append(np.array([], dtype=float))
            continue
        arr.sort()
        out.append(arr)

    return out


# =====================================================================
# Registry
# =====================================================================

def get_default_mode_registry() -> Dict[str, Any]:
    """
    Return the default mode registry for Step 2.3.

    All registered handlers obey the 4-argument mode contract:
        handler(sim_cfg, group_cfg, geometry, rng)
    and must return List[np.ndarray] of length N_syn as resolved by
    _get_n_syn(group_cfg).
    """
    return {
        "homogeneous_poisson": _mode_homogeneous_poisson,
        "precomputed": _mode_precomputed,
        "inhomogeneous_poisson": _mode_inhomogeneous_poisson,
    }
