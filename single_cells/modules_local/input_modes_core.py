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
    Built-in handler for mode == "precomputed" (STUB for 2.3).

    Canonical 2.3 handler contract:
      - Signature: handler(sim_cfg, group_cfg, geometry, rng)
      - Modes must:
          * read time_cfg from group_cfg["time_cfg"],
          * read N_syn via _get_n_syn(group_cfg),
          * return List[np.ndarray] of length N_syn,
          * keep all spikes within [sim_cfg["tstart"], sim_cfg["tstop"]].

    Intended SOURCE CONTRACT (to be implemented later):
      - group_cfg["source"] may provide either:
          * "trains": [[...], [...], ...]  (inline spike trains, ms, sim-time)
          * "path": "file.json"            (JSON with {"trains": [...]} or raw list)
      - time_cfg["blocks"] may be used to:
          * clip / shift trains into the active windows,
          * optionally insert or respect baseline segments.

    For now this is a stub; the actual implementation will be added in a
    later step.
    """
    # Access time_cfg for completeness; real logic will use it.
    _ = (group_cfg or {}).get("time_cfg", {}) or {}
    raise NotImplementedError(
        "Mode 'precomputed' is not yet implemented in core; "
        "stub provided for 2.3 handler contract alignment."
    )


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
    Built-in handler for mode == "inhomogeneous_poisson" (STUB for 2.3).

    Canonical 2.3 handler contract:
      - Signature: handler(sim_cfg, group_cfg, geometry, rng)
      - Modes must:
          * read time_cfg from group_cfg["time_cfg"],
          * read N_syn via _get_n_syn(group_cfg),
          * return List[np.ndarray] of length N_syn,
          * keep all spikes within [sim_cfg["tstart"], sim_cfg["tstop"]].

    Intended SOURCE CONTRACT (for later implementation):
      - group_cfg["source"] keys:
          * "path": str       (CSV or JSON with rate curve),
          * "time_col": str   (column/key name for times, ms),
          * "rate_col": str   (column/key name for rates, Hz),
          * "bin_ms": float   (optional; infer from time_col if missing),
          * "baseline": float (optional; default from first rate).

    Intended TIMING semantics:
      - Combine:
          * (times_ms, rates_hz, bin_ms, baseline) from the rate curve,
          * group_cfg["time_cfg"]["anchors"] / ["blocks"],
          * N_syn from group_cfg,
        to generate inhomogeneous Poisson spike trains, typically by:
          * restricting the rate curve to "source" blocks,
          * using _generate_inhomogeneous_from_curve per block,
          * stitching segments per synapse.

    For now this is a stub; the actual loading and generation logic will
    be implemented in a later step.
    """
    _ = (group_cfg or {}).get("time_cfg", {}) or {}
    raise NotImplementedError(
        "Mode 'inhomogeneous_poisson' is not yet implemented in core; "
        "stub provided for 2.3 handler contract alignment."
    )


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
