"""Shared utilities used by built-in input-generation modes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _get_n_syn(group_cfg: Dict[str, Any]) -> int:
    """
    Read the final synapse count for this group.

    Contract with inputs.py:
      - density._resolve_n_syn(...) runs before modes and writes
        syns["N_syn_resolved"] for active groups.
      - Modes should use that value when present, via this helper.

    Fallback:
      - If N_syn_resolved is absent, fall back to syns["N_syn"] as integer.
        This is mainly for testing / non-geometry cases.
    """
    syns = group_cfg.get("syns", {}) or {}

    # Preferred path: use N_syn_resolved if set by density._resolve_n_syn
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


def _find_scp_root(start: Path) -> Optional[Path]:
    for p in [start] + list(start.parents):
        if (p / "cells").is_dir() and (p / "run_pipeline.py").is_file():
            return p
    return None


def _resolve_source_path(raw_path: str, sim_cfg: Dict[str, Any]) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p

    tune_dir_raw = sim_cfg.get("tune_dir")
    tune_dir = Path(tune_dir_raw) if tune_dir_raw else Path.cwd()
    tune_dir = tune_dir.resolve()
    repo_root = _find_scp_root(tune_dir)

    if repo_root and p.parts and p.parts[0] in ("external_data", "cells"):
        return (repo_root / p).resolve()

    return (tune_dir / p).resolve()


def _parse_gabab_cfg(source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = (source or {}).get("gabab", None)
    if raw in (None, False):
        return None
    if raw is True:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"source['gabab'] must be dict/bool (got {type(raw)!r})")
    if raw.get("enabled") is False:
        return None

    cfg = dict(raw)
    mode = str(cfg.get("mode", "delayed")).strip().lower()
    if mode not in ("delayed", "simple"):
        raise ValueError(f"source['gabab'].mode must be 'delayed' or 'simple' (got {mode!r})")

    history = str(cfg.get("history", "full")).strip().lower()
    if history not in ("full", "trimmed"):
        raise ValueError(f"source['gabab'].history must be 'full' or 'trimmed' (got {history!r})")

    tau_s_raw = cfg.get("tau_s", None)
    tau_ms_raw = cfg.get("tau_ms", None)
    if tau_s_raw is None and tau_ms_raw is None:
        tau_s = 0.01
    elif tau_s_raw is not None:
        tau_s = float(tau_s_raw)
    else:
        tau_s = float(tau_ms_raw) / 1000.0
    if tau_s <= 0.0:
        raise ValueError(f"source['gabab'].tau_s must be > 0 (got {tau_s!r})")

    delay_ms_raw = cfg.get("delay_ms", cfg.get("delay", 50.0))
    delay_ms = 0.0 if delay_ms_raw is None else float(delay_ms_raw)
    alpha = float(cfg.get("alpha", 1.0))
    init = str(cfg.get("init", "match"))
    robust_norm = bool(cfg.get("robust_norm", False))
    pctl = float(cfg.get("pctl", 99.0))

    return {
        "mode": mode,
        "history": history,
        "tau_s": tau_s,
        "delay_ms": delay_ms,
        "alpha": alpha,
        "init": init,
        "robust_norm": robust_norm,
        "pctl": pctl,
    }


def _apply_gabab_to_curve(
    times_ms: np.ndarray,
    rates_hz: np.ndarray,
    cfg: Dict[str, Any],
) -> np.ndarray:
    if rates_hz.size == 0 or times_ms.size < 2:
        return rates_hz

    dt_ms = float(np.median(np.diff(times_ms[: min(times_ms.size, 500)])))
    if dt_ms <= 0.0:
        raise ValueError(f"GABAB: invalid dt_ms {dt_ms!r}")
    dt_s = dt_ms / 1000.0

    r = np.asarray(rates_hz, dtype=float)
    if cfg["robust_norm"]:
        r_ref = np.percentile(r, cfg["pctl"])
    else:
        r_ref = r.max()
    r_ref = max(float(r_ref), 1e-12)
    r_norm = r / r_ref

    if cfg["mode"] == "simple":
        r_drive = r_norm
    else:
        k = int(round(cfg["delay_ms"] / max(dt_ms, 1e-12)))
        if k <= 0:
            r_drive = r_norm
        elif k >= r.size:
            base = r_norm[0] if cfg["init"] == "match" else 0.0
            r_drive = np.full_like(r_norm, base)
        else:
            base = r_norm[0] if cfg["init"] == "match" else 0.0
            r_drive = np.empty_like(r_norm)
            r_drive[:k] = base
            r_drive[k:] = r_norm[:-k]

    S = np.zeros_like(r_norm)
    S[0] = r_norm[0] if cfg["init"] == "match" else 0.0
    coef = dt_s / cfg["tau_s"]
    for i in range(1, r.size):
        S[i] = S[i - 1] + coef * (r_drive[i - 1] - S[i - 1])
    S = np.clip(S, 0.0, 1.0)

    I = r * (1.0 - cfg["alpha"] * S)
    I[I < 0.0] = 0.0
    return I


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
