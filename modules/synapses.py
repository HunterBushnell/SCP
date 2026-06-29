from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import math
import numpy as np

from modules.inputs import _compile_density_from_spec
from modules import randomness


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


def add_synapses(
    cell: Any,
    geom: Dict[str, Any],
    sim_cfg: Dict[str, Any],
    groups_cfg: Dict[str, Dict[str, Any]],
    inputs_by_group: Dict[str, Any],
    *,
    trial_rng: randomness.TrialRandomness = None,
    preview_only: bool = False,
) -> Dict[str, Any]:
    """
    Step 5.2.4 — Add Synapses

    New 5.2.3→5.2.4 contract:

    Parameters
    ----------
    cell : Any
        Loaded cell object (e.g. from load_cell.load_cell); must expose `cell.h`.
    geom : dict
        Geometry dict from Step 5.2.2, with `geom["groups"]` containing segment refs
        for keys like "soma", "proximal", "distal", "all_dend".
    sim_cfg : dict
        Simulation configuration from Step 5.2.3 (normalized "sim" block).
        Used here primarily for RNG seeding via sim_cfg["seed"].
    groups_cfg : dict
        Per-group configs from Step 5.2.3 (normalized "synapse_groups" block),
        keyed by group name. For each synapse group `g` that appears in
        `inputs_by_group`, we expect:

            groups_cfg[g]["syns"] = {
                "type": str,
                "N_syn_resolved": int,  # preferred
                "N_syn": Optional[int], # optional fallback
                "segs": str,
                "dist_func": Any,       # dist_func spec (None/number/callable/dict)
                "params": dict,         # mech + weight/delay params
                ...
            }

    inputs_by_group : dict
        Per-group inputs from Step 5.2.3, keyed by group name. Values are
        GroupInputs-like objects with at least:

            gi.mode: str
            gi.spike_trains: List[np.ndarray]
            gi.meta: dict

        For each group g:
          - len(gi.spike_trains) must be either:
              * 1 (broadcast to all synapses), or
              * N_syn_resolved (1:1 mapping to synapses).
    preview_only : bool
        If True, do not create NEURON synapses/VecStims/NetCons; only return
        synapse records for inspection.

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
      (creating them if missing) so that NEURON objects are kept alive.
      Skipped when preview_only is True.

    Notes
    -----
    - For now, this function trusts `syns["N_syn_resolved"]` as the canonical
      synapse count for each group. A future extension could allow alternative
      N_syn schemes (e.g. driven directly by spike-train count) in a controlled
      way; that should be implemented explicitly rather than inferred.
    """
    h = cell.h
    preview_only = bool(preview_only)

    # Persistent lists on cell
    if not preview_only:
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
    if preview_only:
        syn_state["preview_only"] = True

    syn_id = 0

    # Map 'segs' keys to geometry groups
    group_map = {
        "all": "all_dend",
        "proximal": "proximal",
        "distal": "distal",
        "soma": "soma",
    }


    # Loop over groups that actually have inputs
    for syn_group, group_inputs in inputs_by_group.items():
        # Look up corresponding group_cfg
        group_cfg = groups_cfg.get(syn_group)
        if group_cfg is None:
            raise ValueError(
                f"add_synapses: group {syn_group!r} present in inputs_by_group "
                "but missing from groups_cfg."
            )

        # Respect group "state" flag if present
        if not group_cfg.get("state", True):
            continue

        # syn_cfg = group_cfg.get("syns")
        syn_cfg = group_cfg.get("syns", {})
        if not syn_cfg:
            raise ValueError(
                f"add_synapses: group {syn_group!r} has no 'syns' config in groups_cfg."
            )
        
        # Build mechanism-only params from the nested "params" block
        mech_params = dict(syn_cfg.get("params", {}))
        mech_params["type"] = syn_cfg["type"]

        # Resolve N_syn (prefer N_syn_resolved; fallback N_syn with explicit comment)
        n_syn = syn_cfg.get("N_syn_resolved", None)
        if n_syn is None:
            # Fallback path – primarily for testing / non-geometry uses.
            # NOTE: For the standard 2.3 pipeline, N_syn_resolved should be set;
            # if you rely on this fallback in production, consider updating the
            # pipeline to handle that case explicitly.
            n_syn = syn_cfg.get("N_syn", None)

        if n_syn is None:
            raise ValueError(
                f"add_synapses: neither 'N_syn_resolved' nor 'N_syn' set for "
                f"group {syn_group!r}; ensure Step 5.2.3 resolved syn counts."
            )

        n_syn = int(n_syn)
        if n_syn < 1:
            # Nothing to attach for this group
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

        # Compile dist_func spec into a density function, as in Step 5.2.3
        dist_spec = syn_cfg.get("dist_func", None)
        dens_eq = _compile_density_from_spec(dist_spec)

        # Density-aware placement, mirroring Step 5.2.2/5.2.3 semantics
        # RNGs: placement and weights get distinct streams
        if trial_rng is not None:
            placement_rng = trial_rng.rng(
                "synapses.placement", group=syn_group, stream="placement"
            )
            weight_rng = trial_rng.rng(
                "synapses.weights", group=syn_group, stream="weights"
            )
        else:
            placement_rng, weight_rng = _fallback_rng_pair(
                sim_cfg.get("seed") if isinstance(sim_cfg, dict) else None
            )

        all_syn_locs = _gen_syn_locs(h, placement_rng, n_syn, dens_eq, seg_list)

        # Inputs for this group
        trains = list(group_inputs.spike_trains)
        if not trains:
            raise ValueError(
                f"add_synapses: empty spike_trains for synapse group {syn_group!r}."
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
                "Provide either 1 train or N_syn_resolved trains."
            )

        group_syns: List[Any] = []
        group_ncs: List[Any] = []
        group_stims: List[Any] = []
        group_vecs: List[Any] = []
        group_records: List[SynapseRecord] = []

        for i, syn_loc in enumerate(all_syn_locs):
            if preview_only:
                syn = None
                syn_wt = _draw_syn_weight(weight_rng, mech_params)
            else:
                syn = _gen_syn_mech(h, weight_rng, syn_loc, mech_params)
                # syn = _gen_syn_mech(h, weight_rng, syn_loc, syn_cfg)
                syn_wt = float(getattr(syn, "initW", 0.0))

            train_arr = np.asarray(_get_train(i), dtype=float)
            spike_times = train_arr.tolist()

            if not preview_only:
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
                weight=float(syn_wt),
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

        # print(f"{syn_group}: generated {len(all_syn_locs)} synapses.")

    return syn_state


def preview_synapses(
    cell: Any,
    geom: Dict[str, Any],
    sim_cfg: Dict[str, Any],
    groups_cfg: Dict[str, Dict[str, Any]],
    inputs_by_group: Dict[str, Any],
    *,
    trial_rng: randomness.TrialRandomness = None,
) -> Dict[str, Any]:
    """
    Preview synapse placement/weights without creating NEURON objects.

    This returns a syn_state dict with records only (preview_only=True), which is
    useful for notebook plotting/debugging without affecting simulations.
    """
    return add_synapses(
        cell=cell,
        geom=geom,
        sim_cfg=sim_cfg,
        groups_cfg=groups_cfg,
        inputs_by_group=inputs_by_group,
        trial_rng=trial_rng,
        preview_only=True,
    )
