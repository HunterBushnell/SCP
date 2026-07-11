from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class SegmentRef:
    """Lightweight reference to a segment with distance information."""
    sec: Any           # NEURON Section
    x: float           # normalized position (0..1)
    dist_um: float     # path distance from origin
    sec_name: str      # section name


_GEOMETRY_KEYS = {"label", "distance_origin", "thresholds_um"}
_ORIGIN_KEYS = {"kind", "x"}
_THRESHOLD_GROUP_KEYS = {"proximal", "distal"}
_THRESHOLD_BAND_KEYS = {"low", "high"}


def _reject_unknown_keys(config: Dict[str, Any], allowed: set[str], label: str) -> None:
    """Reject fields that are not part of the current public geometry schema."""
    unknown = sorted(set(config) - allowed)
    if unknown:
        unknown_text = ", ".join(repr(key) for key in unknown)
        raise ValueError(f"define_geometry: unsupported {label} field(s): {unknown_text}.")


def _optional_float(value: Any, field: str) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"define_geometry: {field} must be numeric or null, got {value!r}.") from exc


def _read_threshold_band(
    thresholds_cfg: Dict[str, Any],
    band_name: str,
    *,
    default_low: Optional[float],
    default_high: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    band_cfg = thresholds_cfg.get(band_name, {}) or {}
    if not isinstance(band_cfg, dict):
        raise ValueError(f"define_geometry: thresholds_um.{band_name} must be an object.")
    _reject_unknown_keys(band_cfg, _THRESHOLD_BAND_KEYS, f"thresholds_um.{band_name}")

    low = _optional_float(band_cfg.get("low", default_low), f"thresholds_um.{band_name}.low")
    high = _optional_float(band_cfg.get("high", default_high), f"thresholds_um.{band_name}.high")
    if low is not None and high is not None and low >= high:
        raise ValueError(
            f"define_geometry: thresholds_um.{band_name}.low must be less than high "
            f"when both are set; got low={low}, high={high}."
        )
    return low, high


def _in_distance_band(
    distance_um: float,
    *,
    low: Optional[float],
    high: Optional[float],
    include_low: bool,
) -> bool:
    if low is not None:
        if include_low:
            if distance_um < low:
                return False
        elif distance_um <= low:
            return False
    if high is not None and distance_um >= high:
        return False
    return True


def _unique_sections(*section_groups: Iterable[Any]) -> List[Any]:
    """Return sections in first-seen order, avoiding duplicate section names."""
    seen_names = set()
    unique_sections = []
    for section_group in section_groups:
        for section in section_group:
            name = section.name()
            if name in seen_names:
                continue
            seen_names.add(name)
            unique_sections.append(section)
    return unique_sections


def define_geometry(cell: Any, geom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build a standardized geometry view for the given cell.

    Parameters
    ----------
    cell : LoadedCell-like object
        Must have attribute `.h` which is a NEURON hoc interpreter with sections.
    geom_config : dict, optional
        Optional configuration. Unknown fields are rejected so stale geometry
        options do not appear to be active when they are not implemented:

        {
            "label": "optional_name",
            "distance_origin": {
                "kind": "soma",   # currently only "soma" is supported
                "x": 0.5,        # location along soma section
            },
            "thresholds_um": {
                "proximal": {"low": 20.0, "high": 100.0},
                "distal":   {"low": 100.0, "high": None},
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
    if not isinstance(geom_config, dict):
        raise ValueError("define_geometry: geom_config must be an object/dict.")
    _reject_unknown_keys(geom_config, _GEOMETRY_KEYS, "geometry config")

    h = cell.h

    # ---- basic section lists ----
    soma_secs = list(h.soma) if hasattr(h, "soma") else []
    dend_secs = list(h.dend) if hasattr(h, "dend") else []
    apic_secs = list(h.apic) if hasattr(h, "apic") else []
    axon_secs = list(h.axon) if hasattr(h, "axon") else []
    all_secs = [sec for sec in h.allsec()]
    dendritic_secs = _unique_sections(dend_secs, apic_secs)

    if not soma_secs:
        raise RuntimeError("define_geometry: cell has no soma sections.")

    # ---- distance origin (configurable, defaults to soma(0.5)) ----
    origin_cfg = geom_config.get("distance_origin", {}) or {}
    if not isinstance(origin_cfg, dict):
        raise ValueError("define_geometry: distance_origin must be an object.")
    _reject_unknown_keys(origin_cfg, _ORIGIN_KEYS, "distance_origin")

    origin_kind = origin_cfg.get("kind", "soma")
    origin_x = _optional_float(origin_cfg.get("x", 0.5), "distance_origin.x")
    if origin_x is None:
        raise ValueError("define_geometry: distance_origin.x must be numeric, not null.")
    if not 0.0 <= origin_x <= 1.0:
        raise ValueError(f"define_geometry: distance_origin.x must be in [0, 1], got {origin_x}.")

    if origin_kind != "soma":
        # for now we only support soma; can be extended later
        raise ValueError(
            f"define_geometry: unsupported distance_origin.kind={origin_kind!r}; "
            "currently only 'soma' is supported."
        )

    origin_sec = soma_secs[0]
    h.distance(0, origin_sec(origin_x))

    # ---- thresholds (configurable, defaults to your original 20/100 µm) ----
    thr_cfg = geom_config.get("thresholds_um", {}) or {}
    if not isinstance(thr_cfg, dict):
        raise ValueError("define_geometry: thresholds_um must be an object.")
    _reject_unknown_keys(thr_cfg, _THRESHOLD_GROUP_KEYS, "thresholds_um")

    prox_low, prox_high = _read_threshold_band(
        thr_cfg,
        "proximal",
        default_low=20.0,
        default_high=100.0,
    )
    dist_low, dist_high = _read_threshold_band(
        thr_cfg,
        "distal",
        default_low=100.0,
        default_high=None,
    )

    # ---- classify dendritic segments using these thresholds ----
    proximal_segs: List[SegmentRef] = []
    distal_segs: List[SegmentRef] = []
    all_dend_segs: List[SegmentRef] = []

    for sec in dendritic_secs:
        for seg in sec:
            dist = float(h.distance(seg))
            ref = SegmentRef(sec=sec, x=seg.x, dist_um=dist, sec_name=sec.name())
            all_dend_segs.append(ref)

            if _in_distance_band(dist, low=prox_low, high=prox_high, include_low=False):
                proximal_segs.append(ref)
            elif _in_distance_band(dist, low=dist_low, high=dist_high, include_low=True):
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
                "target_dend_secs": len(dendritic_secs),
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
