from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import math
import numpy as np


@dataclass
class SynapseRecord:
    syn_id: int
    group: str
    type: str
    weight: float
    distance: float
    section: str
    x: float
    spike_times: List[float]


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
    wt_std_factor = syn_params.get("wt_std", 0.0)
    wt_std = wt_std_factor * wt_mean  # scaled off mean, wt_std=0 → fixed

    if wt_std > 0:
        mu, sig = _lognormal_mu_sigma(wt_mean, wt_std)
        return float(rng.lognormal(mu, sig))
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

    syn_wt = _draw_syn_weight(rng, syn_params)
    syn.initW = syn_wt
    return syn


def add_synapses(
    cell: Any,
    geom: Dict[str, Any],
    syn_params: Dict[str, Dict[str, Any]],
    sim_params: Dict[str, Any],
    inputs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Step 2.4 — Add Synapses

    Given:
    - cell: LoadedCell (or similar) from load_cell(...)
    - geom: geometry dict from define_geometry(...)
    - syn_params: per-group synapse configuration (as in original notebook)
    - sim_params: simulation configuration (currently only used for RNG seed)
    - inputs: dict of spike trains per synapse group, shape:

        inputs = {
            "syn_group": {
                "trains": [np.array([...]), np.array([...]), ...],
                "meta": {...},   # optional
            },
            ...
        }

      If len(trains) == 1, the single train is reused for all synapses in that group.
      If len(trains) == N_syn, each synapse uses its own train.

    Returns
    -------
    syn_state : dict
        {
            "synapses": {group: [syn, ...]},
            "netcons":  {group: [nc, ...]},
            "stims":    {group: [stim, ...]},
            "vecs":     {group: [vec, ...]},
            "records":  {group: [SynapseRecord, ...]},
        }

    Side effects
    ------------
    - Attaches lists `synapses`, `netcons`, `stims`, `vecs` to the `cell` object
      (creating them if missing) so that objects are kept alive.
    """
    h = cell.h

    # RNG (reproducible if user sets sim_params["seed"])
    seed = sim_params.get("seed") if isinstance(sim_params, dict) else None
    rng = np.random.default_rng(seed)

    # Persistent lists on cell
    if not hasattr(cell, "synapses"):
        cell.synapses = []
    if not hasattr(cell, "netcons"):
        cell.netcons = []
    if not hasattr(cell, "stims"):
        cell.stims = []
    if not hasattr(cell, "vecs"):
        cell.vecs = []

    syn_state: Dict[str, Any] = {
        "synapses": {},
        "netcons": {},
        "stims": {},
        "vecs": {},
        "records": {},
    }

    syn_id = 0

    # Map old 'segs' keys to geometry groups
    group_map = {
        "all": "all_dend",
        "proximal": "proximal",
        "distal": "distal",
        "soma": "soma",
    }

    for syn_group, syn_cfg in syn_params.items():
        # Skip if N_syn < 1
        n_syn = syn_cfg.get("N_syn", None)
        if n_syn is not None and n_syn < 1:
            continue

        # Target segments from geometry
        segs_key = syn_cfg.get("segs")
        geom_group_name = group_map.get(segs_key)
        if geom_group_name is None:
            raise ValueError(
                f"add_synapses: unknown segs selector {segs_key!r} "
                f"for syn_group {syn_group!r}."
            )

        seg_refs = geom["groups"].get(geom_group_name, [])
        seg_list = [ref.sec(ref.x) for ref in seg_refs]

        dens_eq = syn_cfg.get("dist_func", None)
        all_syn_locs = _gen_syn_locs(h, rng, n_syn, dens_eq, seg_list)

        # Inputs for this group
        group_input = inputs.get(syn_group)
        if group_input is None:
            raise ValueError(
                f"add_synapses: no inputs provided for synapse group {syn_group!r}."
            )

        trains = group_input.get("trains", [])
        if not trains:
            raise ValueError(
                f"add_synapses: empty 'trains' list for synapse group {syn_group!r}."
            )

        # Map trains → synapses
        if len(trains) == 1:
            def _get_train(idx: int):
                return trains[0]
        elif len(trains) == len(all_syn_locs):
            def _get_train(idx: int):
                return trains[idx]
        else:
            raise ValueError(
                f"add_synapses: mismatch between number of trains ({len(trains)}) "
                f"and synapse locations ({len(all_syn_locs)}) for group {syn_group!r}. "
                "Provide either 1 train or N_syn trains."
            )

        group_syns: List[Any] = []
        group_ncs: List[Any] = []
        group_stims: List[Any] = []
        group_vecs: List[Any] = []
        group_records: List[SynapseRecord] = []

        for i, syn_loc in enumerate(all_syn_locs):
            syn = _gen_syn_mech(h, rng, syn_loc, syn_cfg)

            train_arr = np.asarray(_get_train(i), dtype=float)
            spike_times = train_arr.tolist()

            vec = h.Vector(spike_times)
            stim = h.VecStim()
            stim.play(vec)
            nc = h.NetCon(stim, syn)
            nc.weight[0] = 1.0  # actual synaptic weight is syn.initW

            cell.synapses.append(syn)
            cell.vecs.append(vec)
            cell.stims.append(stim)
            cell.netcons.append(nc)

            group_syns.append(syn)
            group_vecs.append(vec)
            group_stims.append(stim)
            group_ncs.append(nc)

            dist = float(h.distance(syn_loc))
            rec = SynapseRecord(
                syn_id=syn_id,
                group=syn_group,
                type=syn_cfg["type"],
                weight=float(getattr(syn, "initW", 0.0)),
                distance=dist,
                section=syn_loc.sec.name(),
                x=float(syn_loc.x),
                spike_times=spike_times,
            )
            group_records.append(rec)
            syn_id += 1

        syn_state["synapses"][syn_group] = group_syns
        syn_state["netcons"][syn_group] = group_ncs
        syn_state["stims"][syn_group] = group_stims
        syn_state["vecs"][syn_group] = group_vecs
        syn_state["records"][syn_group] = group_records

        print(f"{syn_group}: generated {len(all_syn_locs)} synapses.")

    return syn_state
