"""Passive tuning helpers for Step 2 notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from modules.model.geometry import cell_sections


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
    sections = cell_sections(cell, "soma")
    if not sections:
        raise AttributeError("Could not find canonical soma sections on the loaded cell.")
    return sections[int(soma_index)]


def iter_cell_sections(cell: Any) -> list[Any]:
    """Return sections scoped to the loaded cell when available."""
    sections = cell_sections(cell, "all")
    if sections:
        return sections
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


def passive_amplitude_colors(
    sim_amps: Sequence[float],
    *,
    single_trace_color: Optional[str] = None,
) -> dict[float, str]:
    """Return deterministic colors shared by passive plots and result tables."""

    from matplotlib import rcParams
    from matplotlib.colors import to_hex

    amplitudes = list(dict.fromkeys(float(value) for value in sim_amps))
    if not amplitudes:
        return {}
    if len(amplitudes) == 1 and single_trace_color:
        return {amplitudes[0]: to_hex(single_trace_color)}
    cycle = list(rcParams["axes.prop_cycle"].by_key().get("color", ("#1f77b4",)))
    return {
        amp: to_hex(cycle[index % len(cycle)])
        for index, amp in enumerate(amplitudes)
    }


def plot_passive_trace_check(
    *,
    looped_records: Mapping[str, Any],
    sim_params: Mapping[str, Any],
    sim_amps: Sequence[float],
    cell_name: str,
    tune_name: str,
    xlim: Optional[tuple[Optional[float], Optional[float]]] = None,
    ylim: Optional[tuple[Optional[float], Optional[float]]] = None,
    single_trace_color: Optional[str] = None,
    amplitude_colors: Optional[Mapping[float, str]] = None,
) -> Any:
    """Plot passive voltage traces using the colors shared with result tables."""

    import matplotlib.pyplot as plt

    amplitudes = [float(value) for value in sim_amps]
    colors = dict(
        amplitude_colors
        or passive_amplitude_colors(
            amplitudes,
            single_trace_color=single_trace_color,
        )
    )
    figure, axis = plt.subplots(figsize=(7, 4))
    for amp in amplitudes:
        axis.plot(
            looped_records["T"][amp],
            looped_records["V"][amp],
            label=f"{amp:g} pA",
            color=colors.get(amp),
        )
    stim_start = float(sim_params["stim_delay"])
    stim_stop = stim_start + float(sim_params["stim_dur"])
    axis.axvspan(stim_start, stim_stop, alpha=0.12, color="gray", label="stimulus")
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("Membrane voltage (mV)")
    axis.set_title(f"{cell_name} {tune_name} passive check")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    if xlim is not None:
        axis.set_xlim(*xlim)
    if ylim is not None:
        axis.set_ylim(*ylim)
    figure.tight_layout()
    return figure


def _local_gettable_passive_metrics(
    voltage_mv: Sequence[float],
    *,
    dt_ms: float,
    stim_start_ms: float,
    stim_end_ms: float,
    amp_nA: float,
) -> dict[str, float]:
    """Return ACT-compatible passive trace metrics without importing ACT."""

    voltage = np.asarray(voltage_mv, dtype=float)
    dt = float(dt_ms)
    if voltage.ndim != 1 or voltage.size < 3:
        raise ValueError("Passive voltage traces must be one-dimensional.")
    if dt <= 0:
        raise ValueError("Passive protocol dt must be positive.")

    rest_index = int(float(stim_start_ms) / dt) - 1
    final_index = int(float(stim_end_ms) / dt) - 1
    if rest_index < 0 or final_index >= voltage.size or final_index <= rest_index:
        raise ValueError(
            "Passive metric window falls outside the recorded voltage trace."
        )
    trough_index = rest_index + int(np.argmin(voltage[rest_index:]))

    v_rest = voltage[rest_index]
    v_trough = voltage[trough_index]
    v_final = voltage[final_index]
    tau1_threshold = v_rest - (v_rest - v_trough) * 0.632
    tau1_index = int(np.argmax(voltage[rest_index:] < tau1_threshold))
    tau1_ms = tau1_index * dt

    tau2_threshold = v_trough - (v_trough - v_final) * 0.632
    tau2_index = int(np.argmax(voltage[trough_index:] > tau2_threshold))
    tau2_ms = tau2_index * dt
    trough_delta = v_rest - v_trough
    sag_delta = v_final - v_trough

    with np.errstate(divide="ignore", invalid="ignore"):
        rin_mohm = np.divide(v_rest - v_final, -float(amp_nA))
        tau_avg_ms = np.divide(
            trough_delta * tau1_ms + sag_delta * tau2_ms,
            trough_delta + sag_delta,
        )
        sag_ratio = np.divide(sag_delta, trough_delta)
    return {
        "R_in_rest_to_final": float(rin_mohm),
        "tau_rest_to_trough": float(tau1_ms),
        "tau_avg": float(tau_avg_ms),
        "sag_ratio": float(sag_ratio),
        "V_rest": float(v_rest),
    }


def passive_metric_rows(
    *,
    act_passive_module: Any = None,
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
            if act_passive_module is None:
                measured = _local_gettable_passive_metrics(
                    looped_records["V"][amp],
                    dt_ms=dt,
                    stim_start_ms=stim_start,
                    stim_end_ms=stim_end,
                    amp_nA=amp_f / 1000.0,
                )
            else:
                gpp = act_passive_module.compute_gpp(
                    looped_records["V"][amp],
                    dt,
                    stim_start,
                    stim_end,
                    amp_f / 1000.0,
                )
                measured = {
                    key: float(getattr(gpp, key))
                    for key in (
                        "R_in_rest_to_final",
                        "tau_rest_to_trough",
                        "tau_avg",
                        "sag_ratio",
                        "V_rest",
                    )
                    if hasattr(gpp, key)
                }
            row.update(measured)
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
                    "note": (
                        "Computed by ACTPassiveModule.compute_spp; review before "
                        "applying to model files."
                    ),
                }
            )
    return rows
