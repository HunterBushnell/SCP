from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SegmentRef:
    """Lightweight reference to a segment with distance information."""
    sec: Any           # NEURON Section
    x: float           # normalized position (0..1)
    dist_um: float     # path distance from origin
    sec_name: str      # section name


def define_geometry(cell: Any, geom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build a standardized geometry view for the given cell.

    Parameters
    ----------
    cell : LoadedCell-like object
        Must have attribute `.h` which is a NEURON hoc interpreter with sections.
    geom_config : dict, optional
        Optional configuration. All fields are optional; if omitted we fall back
        to your original AllenCell behavior:

        {
            "label": "optional_name",
            "distance_origin": {
                "kind": "soma",   # currently only "soma" is supported
                "x": 0.5,        # location along soma section
            },
            "thresholds_um": {
                "proximal": {"low": 20.0, "high": 100.0},
                "distal":   {"low": 100.0},          # high=None → no upper bound
            },
        }

    Returns
    -------
    geom : dict
        {
            "label": str,
            "groups": {
                "soma":     [SegmentRef, ...],
                "proximal": [SegmentRef, ...],
                "distal":   [SegmentRef, ...],
                "all_dend": [SegmentRef, ...],
            },
            "meta": {
                "origin": {...},
                "thresholds_um": {...},
                "counts": {...},
            },
        }
    """
    geom_config = geom_config or {}
    h = cell.h

    # ---- basic section lists ----
    soma_secs = list(h.soma) if hasattr(h, "soma") else []
    dend_secs = list(h.dend) if hasattr(h, "dend") else []
    apic_secs = list(h.apic) if hasattr(h, "apic") else []
    axon_secs = list(h.axon) if hasattr(h, "axon") else []
    all_secs = [sec for sec in h.allsec()]

    if not soma_secs:
        raise RuntimeError("define_geometry: cell has no soma sections.")

    # ---- distance origin (configurable, defaults to soma(0.5)) ----
    origin_cfg = geom_config.get("distance_origin", {})
    origin_kind = origin_cfg.get("kind", "soma")
    origin_x = float(origin_cfg.get("x", 0.5))

    if origin_kind != "soma":
        # for now we only support soma; can be extended later
        raise ValueError(
            f"define_geometry: unsupported distance_origin.kind={origin_kind!r}; "
            "currently only 'soma' is supported."
        )

    origin_sec = soma_secs[0]
    h.distance(0, origin_sec(origin_x))

    # ---- thresholds (configurable, defaults to your original 20/100 µm) ----
    thr_cfg = geom_config.get("thresholds_um", {})

    prox_cfg = thr_cfg.get("proximal", {})
    dist_cfg = thr_cfg.get("distal", {})

    prox_low = float(prox_cfg.get("low", 20.0))
    prox_high = prox_cfg.get("high", 100.0)
    prox_high = float(prox_high) if prox_high is not None else None

    dist_low = float(dist_cfg.get("low", 100.0))
    dist_high = dist_cfg.get("high", None)
    dist_high = float(dist_high) if dist_high is not None else None  # not used yet

    # ---- classify dendritic segments using these thresholds ----
    proximal_segs: List[SegmentRef] = []
    distal_segs: List[SegmentRef] = []
    all_dend_segs: List[SegmentRef] = []

    for sec in dend_secs:
        for seg in sec:
            dist = float(h.distance(seg))
            ref = SegmentRef(sec=sec, x=seg.x, dist_um=dist, sec_name=sec.name())
            all_dend_segs.append(ref)

            if prox_low < dist and (prox_high is None or dist < prox_high):
                proximal_segs.append(ref)
            elif dist_low is not None and dist >= dist_low:
                # matches original "distal if distance >= 100"
                distal_segs.append(ref)

    # ---- soma segments ----
    soma_segs: List[SegmentRef] = []
    for sec in soma_secs:
        for seg in sec:
            dist = float(h.distance(seg))
            soma_segs.append(SegmentRef(sec=sec, x=seg.x, dist_um=dist, sec_name=sec.name()))

    # ---- label ----
    label = geom_config.get("label")
    if label is None:
        label = cell.config.get("cell_name", "<cell>") + "_geometry"

    geom: Dict[str, Any] = {
        "label": label,
        "groups": {
            "soma": soma_segs,
            "proximal": proximal_segs,
            "distal": distal_segs,
            "all_dend": all_dend_segs,
        },
        "meta": {
            "origin": {
                "sec_name": origin_sec.name(),
                "x": origin_x,
                "kind": origin_kind,
            },
            "thresholds_um": {
                "proximal": {"low": prox_low, "high": prox_high},
                "distal":   {"low": dist_low, "high": dist_high},
            },
            "counts": {
                "soma_secs": len(soma_secs),
                "dend_secs": len(dend_secs),
                "apic_secs": len(apic_secs),
                "axon_secs": len(axon_secs),
                "all_secs": len(all_secs),
                "soma_segs": len(soma_segs),
                "proximal_segs": len(proximal_segs),
                "distal_segs": len(distal_segs),
                "all_dend_segs": len(all_dend_segs),
            },
        },
    }

    print(
        f"Geometry defined for {cell.config.get('cell_name', '<cell>')!r}: "
        f"{len(soma_segs)} soma segs, "
        f"{len(proximal_segs)} proximal dend segs, "
        f"{len(distal_segs)} distal dend segs, "
        f"{len(all_dend_segs)} total dend segs."
    )

    return geom
