"""Passive tuning helpers for Step 2 notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


def _cell_h(cell: Any) -> Any:
    return getattr(cell, "h", cell)


def _iter_section_collection(sections: Any) -> list[Any]:
    if sections is None:
        return []
    try:
        return list(sections)
    except TypeError:
        return [sections]


def _soma_section(cell: Any, *, soma_index: int = 0) -> Any:
    h_obj = _cell_h(cell)
    soma_sections = getattr(cell, "soma", None)
    if soma_sections is None:
        soma_sections = getattr(h_obj, "soma", None)
    sections = _iter_section_collection(soma_sections)
    if not sections:
        raise AttributeError("Could not find soma sections on cell or cell.h.")
    return sections[int(soma_index)]


def iter_cell_sections(cell: Any) -> list[Any]:
    """Return sections scoped to the loaded cell when available."""
    h_obj = _cell_h(cell)
    for attr in ("all", "all_sections", "sections"):
        sections = _iter_section_collection(getattr(cell, attr, None))
        if sections:
            return sections
    if hasattr(h_obj, "allsec"):
        return list(h_obj.allsec())
    return [_soma_section(cell)]


def section_area_cm2(cell: Any, section: Any) -> float:
    """Return section area in cm² by summing all NEURON segments."""
    h_obj = _cell_h(cell)
    return sum(float(h_obj.area(segment.x, sec=section)) for segment in section) * 1e-8


def soma_area_cm2(cell: Any, *, soma_index: int = 0) -> float:
    """Return soma area in cm² for ACT passive calculations."""
    return section_area_cm2(cell, _soma_section(cell, soma_index=soma_index))


def total_area_cm2(cell: Any) -> float:
    """Return total cell area in cm² by summing all available sections."""
    return sum(section_area_cm2(cell, section) for section in iter_cell_sections(cell))


def passive_area_summary(
    cell: Any,
    *,
    area_mode: str = "auto",
    area_scale: float = 1.0,
    custom_area_cm2: Optional[float] = None,
    soma_index: int = 0,
) -> Dict[str, Any]:
    """Resolve the area ACT should use for passive-property estimates.

    `auto` follows ACT notebook guidance: use soma area for simple one-section
    cells and total area for detailed cells.
    """
    sections = iter_cell_sections(cell)
    segment_count = sum(1 for section in sections for _ in section)
    soma_area = soma_area_cm2(cell, soma_index=soma_index)
    total_area = total_area_cm2(cell)

    requested_mode = str(area_mode or "auto").strip().lower()
    if requested_mode == "auto":
        selected_mode = "total" if len(sections) > 1 else "soma"
    elif requested_mode in {"soma", "total", "custom"}:
        selected_mode = requested_mode
    else:
        raise ValueError("area_mode must be one of: auto, soma, total, custom")

    if selected_mode == "soma":
        unscaled_area = soma_area
    elif selected_mode == "total":
        unscaled_area = total_area
    else:
        if custom_area_cm2 is None:
            raise ValueError("custom_area_cm2 is required when area_mode='custom'.")
        unscaled_area = float(custom_area_cm2)

    scale = float(area_scale)
    return {
        "requested_area_mode": requested_mode,
        "selected_area_mode": selected_mode,
        "area_scale": scale,
        "selected_area_cm2": float(unscaled_area) * scale,
        "unscaled_selected_area_cm2": float(unscaled_area),
        "soma_area_cm2": float(soma_area),
        "total_area_cm2": float(total_area),
        "custom_area_cm2": None if custom_area_cm2 is None else float(custom_area_cm2),
        "section_count": len(sections),
        "segment_count": int(segment_count),
    }


def compute_settable_passive_properties(
    *,
    act_passive_module: Any,
    cell: Any,
    rin_mohm: float,
    tau_ms: float,
    v_rest_mv: float,
    area_mode: str = "auto",
    area_scale: float = 1.0,
    custom_area_cm2: Optional[float] = None,
) -> Any:
    """Compute ACT settable passive properties from user-facing units."""
    rin_ohm = float(rin_mohm) * 1e6
    tau_s = float(tau_ms) * 1e-3
    area_cm2 = passive_area_summary(
        cell,
        area_mode=area_mode,
        area_scale=area_scale,
        custom_area_cm2=custom_area_cm2,
    )["selected_area_cm2"]
    return act_passive_module.compute_spp(
        rin_ohm,
        area_cm2,
        tau_s,
        float(v_rest_mv),
    )


def run_passive_protocol(
    *,
    cell: Any,
    sim_params: Mapping[str, Any],
    sim_amps: Sequence[float],
) -> Dict[str, Any]:
    """Run current-injection traces with the existing SCP run_sim helper."""
    from modules import run_sim

    return run_sim.looped_current_injection(cell, dict(sim_params), list(sim_amps))


def passive_metric_rows(
    *,
    act_passive_module: Any,
    looped_records: Mapping[str, Any],
    sim_params: Mapping[str, Any],
    sim_amps: Iterable[float],
    dt_ms: Optional[float] = None,
) -> list[Dict[str, Any]]:
    """Return table-friendly passive metrics for negative current injections."""
    rows: list[Dict[str, Any]] = []
    dt = float(dt_ms if dt_ms is not None else sim_params.get("h_dt", sim_params.get("dt", 0.025)))
    stim_start = float(sim_params["stim_delay"])
    stim_end = float(sim_params["stim_delay"]) + float(sim_params["stim_dur"]) - 10.0

    for amp in sim_amps:
        amp_f = float(amp)
        row: Dict[str, Any] = {
            "amp_pA": amp_f,
            "spike_frequency_hz": float(looped_records["F"][amp]),
        }
        if amp_f < 0:
            gpp = act_passive_module.compute_gpp(
                looped_records["V"][amp],
                dt,
                stim_start,
                stim_end,
                amp_f / 1000.0,
            )
            for key in (
                "R_in_rest_to_final",
                "tau_rest_to_trough",
                "tau_avg",
                "sag_ratio",
                "V_rest",
            ):
                if hasattr(gpp, key):
                    row[key] = float(getattr(gpp, key))
        rows.append(row)
    return rows


def passive_proposal_changes(
    *,
    settable_passive_properties: Any,
    target_file: str | Path = "manual_review",
) -> list[Dict[str, Any]]:
    """Convert ACT settable passive properties into proposal change rows."""
    rows: list[Dict[str, Any]] = []
    for field in ("e_rev_leak", "g_bar_leak", "Cm"):
        if hasattr(settable_passive_properties, field):
            rows.append(
                {
                    "file": str(target_file),
                    "field": field,
                    "old": None,
                    "new": float(getattr(settable_passive_properties, field)),
                    "note": "Computed by ACTPassiveModule.compute_spp; review before applying to model files.",
                }
            )
    return rows
