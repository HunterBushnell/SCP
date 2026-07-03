"""Synapse-count and distance-density helpers for input generation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import math


def _lognormal_mu_sigma(mean: float, std: float) -> Tuple[float, float]:
    """
    Return (mu, sigma) for np.random.lognormal given arithmetic mean & std.
    """
    if std <= 0 or mean <= 0:
        return 0.0, 0.0
    mu = math.log(mean**2 / math.sqrt(std**2 + mean**2))
    sig = math.sqrt(math.log(1 + (std**2 / mean**2)))
    return mu, sig


def _compile_density_from_spec(dist_spec: Any):
    """Convert a 'dist_func' spec from JSON into a callable density function.

    The returned function dens(dist_um) should yield synapses-per-µm at that distance.

    Supported forms:
      - None          → dens(d) = 1.0
      - number        → dens(d) = const
      - callable      → dens(d) used as-is
      - dict          → {"kind": "uniform", "params": {"c": float, "multi": optional}}
      - dict          → {"kind": "linear", "params": {"m": float, "b": float, "multi": optional}}
    """
    # None → uniform density 1.0
    if dist_spec is None:
        return lambda d: 1.0

    # Already a callable or a simple numeric constant
    if callable(dist_spec):
        return dist_spec
    if isinstance(dist_spec, (int, float)):
        const = float(dist_spec)
        return lambda d, c=const: c

    # JSON-style spec
    if isinstance(dist_spec, dict):
        kind = dist_spec.get("kind") or "uniform"
        params = dist_spec.get("params", {}) or {}

        if kind == "uniform":
            c = float(params.get("c", 1.0))
            multi = float(params.get("multi", 1.0))
            const = c * multi
            return lambda d, c=const: c

        if kind == "linear":
            m = params.get("m", params.get("slope", 0.0))
            b = params.get("b", params.get("intercept", 0.0))
            multi = float(params.get("multi", 1.0))
            m = float(m)
            b = float(b)
            return lambda d, m=m, b=b, multi=multi: (m * d + b) * multi

        # Placeholder for future shapes (gaussian, etc.)
        raise ValueError(f"dist_func spec with kind={kind!r} is not yet supported for N_syn resolution")

    raise TypeError(
        f"dist_func must be None, number, callable, or dict-spec; got {type(dist_spec)!r}"
    )


def _resolve_n_syn(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
    geometry: Optional[Any],
) -> int:
    """Resolve the effective N_syn for a group, including geometry/density.

    Logic:
      1. If syns["N_syn"] is an explicit integer ≥ 0, use it.
      2. If N_syn is None:
         - Require geometry and a valid dist_func spec.
         - Use the selected geometry group (from syns["segs"]) and the
           density function to compute a deterministic synapse count via:

               n_seg = floor( density(dist_um) * seg_length_um )

           summed over all segments.

    Geometry contract (from Step 5.2.2):
      - geometry["groups"][<name>] must be a list of segment references
        each with:
          * .sec     → NEURON Section
          * .dist_um → distance from soma in µm
        and the Section supplies L and nseg.
    """
    syns = group_cfg.get("syns", {}) or {}
    n_syn_raw = syns.get("N_syn", None)

    # Case 1: explicit N_syn
    if n_syn_raw is not None:
        try:
            n_syn = int(n_syn_raw)
        except Exception as exc:  # pragma: no cover
            raise ValueError(
                f"Group {group_cfg.get('name', '<unnamed>')!r}: syns['N_syn'] must be int or None, "
                f"got {n_syn_raw!r} of type {type(n_syn_raw)!r}."
            ) from exc
        if n_syn < 0:
            raise ValueError(
                f"Group {group_cfg.get('name', '<unnamed>')!r}: syns['N_syn'] must be ≥ 0, got {n_syn}."
            )
        # Stash resolved value for downstream consumers (e.g. 2.4)
        syns["N_syn_resolved"] = n_syn
        group_cfg["syns"] = syns
        return n_syn

    # Case 2: N_syn is None → geometry/density-based count
    if geometry is None:
        raise ValueError(
            "_resolve_n_syn: geometry is required when syns['N_syn'] is None; "
            f"group={group_cfg.get('name', '<unnamed>')!r}"
        )

    # Select segment group from geometry based on syns['segs']
    segs_key = syns.get("segs") or "all"
    group_map = {
        "all": "all_dend",
        "proximal": "proximal",
        "distal": "distal",
        "soma": "soma",
    }
    geom_group_name = group_map.get(segs_key)
    if geom_group_name is None:
        raise ValueError(
            f"_resolve_n_syn: unknown segs selector {segs_key!r} for group {group_cfg.get('name', '<unnamed>')!r}."
        )

    geom_groups = geometry.get("groups", {})
    seg_refs = geom_groups.get(geom_group_name, [])
    if not seg_refs:
        # No segments available in this geometry group
        syns["N_syn_resolved"] = 0
        group_cfg["syns"] = syns
        return 0

    # Build density function
    dens_eq = _compile_density_from_spec(syns.get("dist_func"))

    # Deterministic density-based count, mirroring _gen_distr_synlocs
    total_n_syn = 0
    for ref in seg_refs:
        sec = ref.sec
        seg_len = float(sec.L) / float(sec.nseg or 1)
        dens = float(dens_eq(ref.dist_um))
        if dens <= 0.0:
            continue
        n_seg = math.floor(dens * seg_len)
        if n_seg <= 0:
            continue
        total_n_syn += n_seg

    if total_n_syn < 0:
        raise ValueError(
            f"_resolve_n_syn: computed negative synapse count ({total_n_syn}) for group "
            f"{group_cfg.get('name', '<unnamed>')!r}."
        )

    syns["N_syn_resolved"] = int(total_n_syn)
    group_cfg["syns"] = syns
    return int(total_n_syn)
