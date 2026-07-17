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


def _section_list(value: Any) -> List[Any]:
    """Normalize a NEURON/Python section collection to a plain list."""
    if value is None:
        return []
    if callable(getattr(value, "name", None)) and callable(value) and hasattr(value, "nseg"):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def cell_sections(cell: Any, group: str) -> List[Any]:
    """Return a section collection scoped to one loaded cell.

    New loaders expose canonical section groups directly on ``LoadedCell``.
    ``cell.model`` is retained as a compatibility source for object-owned
    models, while the process-global hoc namespace is only consulted for
    legacy cells that do not expose a model owner.
    """
    name = str(group)

    if hasattr(cell, name):
        return _section_list(getattr(cell, name))

    model = getattr(cell, "model", None)
    if model is not None and hasattr(model, name):
        return _section_list(getattr(model, name))

    if name == "all":
        derived = _unique_sections(
            cell_sections(cell, "soma"),
            cell_sections(cell, "dend"),
            cell_sections(cell, "apic"),
            cell_sections(cell, "axon"),
        )
        if derived or model is not None:
            return derived

    # Legacy Allen/direct-hoc compatibility. Object-owned loaders must expose
    # canonical groups so unrelated global sections are never captured here.
    h_obj = getattr(cell, "h", cell)
    if model is None and hasattr(h_obj, name):
        return _section_list(getattr(h_obj, name))
    if model is None and name == "all" and hasattr(h_obj, "allsec"):
        return list(h_obj.allsec())
    return []


def cell_soma_segment(cell: Any, x: float = 0.5, *, index: int = 0) -> Any:
    """Resolve a canonical somatic segment for a loaded cell."""
    soma = cell_sections(cell, "soma")
    if not soma:
        raise AttributeError("Could not find canonical soma sections on the loaded cell.")
    idx = int(index)
    if idx < 0 or idx >= len(soma):
        raise IndexError(f"Soma section index out of range (idx={idx}, n={len(soma)}).")
    loc = float(x)
    if not 0.0 <= loc <= 1.0:
        raise ValueError(f"Section location x must be in [0, 1], got {loc}.")
    return soma[idx](loc)


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
    soma_secs = cell_sections(cell, "soma")
    dend_secs = cell_sections(cell, "dend")
    apic_secs = cell_sections(cell, "apic")
    axon_secs = cell_sections(cell, "axon")
    all_secs = cell_sections(cell, "all")
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
        cell_config = getattr(cell, "config", {}) or {}
        label = cell_config.get("cell_name", "<cell>") + "_geometry"

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
        f"Geometry defined for {(getattr(cell, 'config', {}) or {}).get('cell_name', '<cell>')!r}: "
        f"{len(soma_segs)} soma segs, "
        f"{len(proximal_segs)} proximal dend segs, "
        f"{len(distal_segs)} distal dend segs, "
        f"{len(all_dend_segs)} total dend segs."
    )

    return geom
