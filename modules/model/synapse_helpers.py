"""Reusable synapse placement and mechanism-construction helpers."""

from __future__ import annotations

from typing import Any, Dict, List

import math
import numpy as np


def _lognormal_mu_sigma(mean: float, std: float) -> (float, float):
    """Return μ and σ for np.random.lognormal given arithmetic mean & std."""
    if std <= 0:
        return 0.0, 0.0
    mu = math.log(mean**2 / math.sqrt(std**2 + mean**2))
    sig = math.sqrt(math.log(1 + (std**2 / mean**2)))
    return mu, sig


def _draw_syn_weight(rng: np.random.Generator, syn_params: Dict[str, Any]) -> float:
    """
    Generate a synaptic weight from a lognormal distribution, matching the
    original AllenCell.draw_syn_wt behavior.
    """
    wt_mean = syn_params.get("wt_mean", syn_params.get("initW", 0.001))
    # syn_params_wt_mean = syn_params.get("wt_mean", None)
    # print(f"Weight mean: {wt_mean} | Syn params: {syn_params_wt_mean}")
    wt_std_factor = syn_params.get("wt_std", 0.0)
    wt_std = wt_std_factor * wt_mean  # scaled off mean, wt_std=0 → fixed

    if wt_std > 0:
        mu, sig = _lognormal_mu_sigma(wt_mean, wt_std)
        syn_wt = float(rng.lognormal(mu, sig))
        # Cap high-tail draws to < mean + 3*std for this synapse group.
        wt_cap = float(wt_mean + 3.0 * wt_std)
        if syn_wt >= wt_cap:
            syn_wt = float(np.nextafter(wt_cap, -np.inf))
        return syn_wt
    else:
        return float(wt_mean)


def _gen_n_synlocs(h, rng: np.random.Generator, n_syns: int, dens_eq, seg_list: List[Any]) -> List[Any]:
    """
    Sample exactly n_syns synapse locations from seg_list using a desired
    density per µm vs. distance (dens_eq). Returns a list[Segment].

    Functional version of AllenCell.gen_n_synlocs:
      - uses h.distance(seg) for distance,
      - seg.sec.L / seg.sec.nseg for segment length.
    """
    from math import isfinite

    distances = np.array([h.distance(seg) for seg in seg_list], dtype=float)
    seg_lengths = np.array([seg.sec.L / seg.sec.nseg for seg in seg_list], dtype=float)

    n = len(seg_list)
    if n == 0:
        return []

    # Evaluate dens_eq safely to get density per µm
    if dens_eq is None:
        dens = np.ones(n, dtype=float)
    elif callable(dens_eq):
        try:
            out = dens_eq(distances)
            if np.isscalar(out):
                dens = np.full(n, float(out), dtype=float)
            else:
                out = np.asarray(out, dtype=float)
                if out.shape != distances.shape:
                    raise ValueError
                dens = out
        except Exception:
            dens = np.array([float(dens_eq(float(d))) for d in distances], dtype=float)
    elif isinstance(dens_eq, (int, float, np.floating)):
        dens = np.full(n, float(dens_eq), dtype=float)
    else:
        dens = np.asarray(dens_eq, dtype=float)
        if dens.shape != distances.shape:
            raise ValueError(
                "dens_eq provided as array-like must have same length as seg_list"
            )

    dens = np.clip(dens, 0.0, None)

    raw = dens * seg_lengths
    s = raw.sum()
    if not isfinite(s) or s <= 0:
        # fallback to length-only (uniform density)
        raw = np.clip(seg_lengths, 0.0, None)
        s = raw.sum()
        if not isfinite(s) or s <= 0:
            raise ValueError("All segment lengths are zero; cannot sample.")

    p = raw / s
    idx = rng.choice(len(seg_list), size=int(n_syns), replace=True, p=p)
    return [seg_list[i] for i in idx]


def _gen_distr_synlocs(h, dens_eq, seg_list: List[Any]) -> List[Any]:
    """
    Original density-based placement from AllenCell.gen_distr_synlocs.
    """
    all_syn_locs: List[Any] = []
    for seg in seg_list:
        seg_dist = h.distance(seg)
        seg_len = seg.sec.L / seg.sec.nseg

        syn_dens = dens_eq(seg_dist)
        if syn_dens <= 0:
            continue

        n_syns = math.floor(syn_dens * seg_len)
        if n_syns <= 0:
            continue

        for _ in range(n_syns):
            all_syn_locs.append(seg)

    return all_syn_locs


def _gen_syn_locs(h, rng: np.random.Generator, n_syns, dens_eq, seg_list: List[Any]) -> List[Any]:
    """
    Functional variant of AllenCell.gen_syn_locs.

    If n_syns is not None, sample exactly n_syns using _gen_n_synlocs.
    Otherwise, place synapses based on density using _gen_distr_synlocs.
    """
    if not seg_list:
        return []

    if n_syns is not None:
        return _gen_n_synlocs(h, rng, n_syns, dens_eq, seg_list)
    else:
        return _gen_distr_synlocs(h, dens_eq, seg_list)


def _gen_syn_mech(h, rng: np.random.Generator, syn_loc, syn_params: Dict[str, Any]):
    """
    Generate a synaptic mechanism at `syn_loc` with parameters `syn_params`.

    Mirrors AllenCell.gen_syn_mechs:
    - instantiates h.<type>(syn_loc)
    - sets attributes present on the mechanism
    - draws a lognormal weight and stores it in syn.initW
    """
    syn_type = syn_params["type"]
    syn = getattr(h, syn_type)(syn_loc)

    for param, val in syn_params.items():
        if hasattr(syn, param):
            setattr(syn, param, val)
    # print(f"Created synapse of type {syn_type} at {syn_loc.sec.name()}({syn_loc.x:.3f})")
    # print(f"  with params: {syn_params}")

    syn_wt = _draw_syn_weight(rng, syn_params)
    # print(f"Generated synaptic weight: {syn_wt:.4f} for synapse type {syn_type}")
    syn.initW = syn_wt
    return syn


def _fallback_rng_pair(seed):
    """
    Build placement/weight RNGs when TrialRandomness is not supplied.
    Split a SeedSequence so streams are distinct but reproducible.
    """
    ss = np.random.SeedSequence() if seed is None else np.random.SeedSequence(int(seed))
    ss_place, ss_weight = ss.spawn(2)
    return np.random.default_rng(ss_place), np.random.default_rng(ss_weight)
