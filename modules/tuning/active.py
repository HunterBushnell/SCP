"""Active tuning helpers for Step 3 notebooks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional

import numpy as np


DEFAULT_BIO_FI_REFERENCE: dict[str, list[tuple[float, float]]] = {
    "SST": [
        (0, 0),
        (25, 1),
        (50, 5.1),
        (75, 9.5),
        (100, 15.7),
        (125, 23.9),
        (150, 33),
        (175, 41.7),
        (200, 48.5),
        (225, 54.2),
        (250, 59.1),
        (275, 64.6),
        (300, 67.2),
    ],
    "PV": [
        (0, 0),
        (25, 0),
        (50, 0),
        (75, 0.1),
        (100, 3.3),
        (125, 6.2),
        (150, 12.9),
        (175, 19.7),
        (200, 27.9),
        (225, 35.2),
        (250, 44.8),
        (275, 55.5),
        (300, 57.4),
    ],
}


def active_amplitude_colors(
    sim_amps: Sequence[float],
    *,
    single_trace_color: Optional[str] = None,
) -> dict[float, str]:
    """Return stable current-step colors shared by Step 3 plots and tables."""
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_hex

    amplitudes = [float(value) for value in sim_amps]
    if len(amplitudes) == 1 and single_trace_color:
        return {amplitudes[0]: to_hex(single_trace_color)}
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    return {
        amp: to_hex(cycle[index % len(cycle)])
        for index, amp in enumerate(amplitudes)
    }


def fi_series_colors(
    *,
    model_color: Optional[str] = None,
    reference_color: Optional[str] = None,
) -> dict[str, str]:
    """Return the model/reference colors shared by the FI plot and table."""
    from matplotlib.colors import to_hex

    resolved_model = model_color or "C0"
    resolved_reference = reference_color or (
        "0.35" if _is_black_color(resolved_model) else "black"
    )
    return {
        "model": to_hex(resolved_model),
        "reference": to_hex(resolved_reference),
    }


def run_active_protocol(
    *,
    cell: Any,
    sim_params: Mapping[str, Any],
    sim_amps: Sequence[float],
) -> dict[str, Any]:
    """Run positive current-injection traces with the existing SCP helper."""
    from modules import run_sim

    return run_sim.looped_current_injection(cell, dict(sim_params), list(sim_amps))


def spike_times_from_trace(
    time_ms: Sequence[float],
    voltage_mv: Sequence[float],
    *,
    threshold_mv: float = -20.0,
    start_ms: Optional[float] = None,
    stop_ms: Optional[float] = None,
) -> np.ndarray:
    """Return spike peak times using a local-maximum threshold rule."""
    time_array = np.asarray(time_ms, dtype=float)
    voltage_array = np.asarray(voltage_mv, dtype=float)
    if time_array.size < 3 or voltage_array.size < 3:
        return np.asarray([], dtype=float)

    mask = np.ones(time_array.shape, dtype=bool)
    if start_ms is not None:
        mask &= time_array >= float(start_ms)
    if stop_ms is not None:
        mask &= time_array <= float(stop_ms)

    selected_indices = np.flatnonzero(mask)
    if selected_indices.size < 3:
        return np.asarray([], dtype=float)

    selected_voltage = voltage_array[selected_indices]
    rising = np.diff(selected_voltage[:-1]) > 0
    falling = np.diff(selected_voltage[1:]) < 0
    above_threshold = selected_voltage[1:-1] > float(threshold_mv)
    peak_offsets = np.flatnonzero(rising & falling & above_threshold) + 1
    return time_array[selected_indices[peak_offsets]]


def active_trace_metrics(
    *,
    time_ms: Sequence[float],
    voltage_mv: Sequence[float],
    sim_params: Mapping[str, Any],
    amp_pA: float,
    frequency_hz: Optional[float] = None,
    threshold_mv: float = -20.0,
) -> dict[str, Any]:
    """Return table-friendly active-spiking metrics for one current step."""
    time_array = np.asarray(time_ms, dtype=float)
    voltage_array = np.asarray(voltage_mv, dtype=float)
    stim_start = float(sim_params["stim_delay"])
    stim_stop = stim_start + float(sim_params["stim_dur"])

    baseline_mask = time_array < stim_start
    stim_mask = (time_array >= stim_start) & (time_array <= stim_stop)
    if not np.any(stim_mask):
        stim_mask = np.ones(time_array.shape, dtype=bool)

    spike_times = spike_times_from_trace(
        time_array,
        voltage_array,
        threshold_mv=threshold_mv,
        start_ms=stim_start,
        stop_ms=stim_stop,
    )
    interspike_intervals = np.diff(spike_times)
    spike_count = int(spike_times.size)

    if frequency_hz is None:
        duration_s = max(float(sim_params["stim_dur"]) / 1000.0, np.finfo(float).eps)
        frequency_hz = spike_count / duration_s

    return {
        "amp_pA": float(amp_pA),
        "spike_count": spike_count,
        "spike_frequency_hz": float(frequency_hz),
        "rest_voltage_mv": _safe_mean(voltage_array[baseline_mask]),
        "peak_voltage_mv": _safe_max(voltage_array[stim_mask]),
        "min_voltage_mv": _safe_min(voltage_array[stim_mask]),
        "first_spike_latency_ms": (
            float(spike_times[0] - stim_start) if spike_count else None
        ),
        "mean_isi_ms": _safe_mean(interspike_intervals),
        "min_isi_ms": _safe_min(interspike_intervals),
        "isi_cv": _safe_cv(interspike_intervals),
        "adaptation_ratio": _adaptation_ratio(interspike_intervals),
    }


def active_metric_rows(
    *,
    looped_records: Mapping[str, Any],
    sim_params: Mapping[str, Any],
    sim_amps: Iterable[float],
    threshold_mv: float = -20.0,
) -> list[dict[str, Any]]:
    """Return active metrics for each current step in `looped_records`."""
    rows: list[dict[str, Any]] = []
    for amp_value in sim_amps:
        rows.append(
            active_trace_metrics(
                time_ms=looped_records["T"][amp_value],
                voltage_mv=looped_records["V"][amp_value],
                frequency_hz=looped_records.get("F", {}).get(amp_value),
                sim_params=sim_params,
                amp_pA=float(amp_value),
                threshold_mv=threshold_mv,
            )
        )
    return rows


def get_bio_fi_reference(cell_name: str) -> list[tuple[float, float]]:
    """Return bundled biological FI reference points for a known example cell."""
    return list(DEFAULT_BIO_FI_REFERENCE.get(str(cell_name), []))


def fi_rows_from_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    """Extract `(amp_pA, frequency_hz)` FI rows from active metric rows."""
    return [
        {
            "amp_pA": float(row["amp_pA"]),
            "spike_frequency_hz": float(row["spike_frequency_hz"]),
        }
        for row in rows
    ]


def plot_active_trace_check(
    *,
    looped_records: Mapping[str, Any],
    sim_params: Mapping[str, Any],
    sim_amps: Sequence[float],
    cell_name: str,
    tune_name: str,
    xlim: Optional[tuple[Optional[float], Optional[float]]] = None,
    ylim: Optional[tuple[Optional[float], Optional[float]]] = None,
    include_currents: bool = True,
    current_amp: Optional[float] = None,
    current_names: Optional[Sequence[str]] = None,
    current_xlim: Optional[tuple[Optional[float], Optional[float]]] = None,
    current_ylim: Optional[tuple[Optional[float], Optional[float]]] = None,
    max_auto_currents: int = 8,
    trace_color: Optional[str] = None,
    amplitude_colors: Optional[Mapping[float, str]] = None,
) -> Any:
    """Plot active voltage traces plus optional current traces."""
    import matplotlib.pyplot as plt

    current_trace_names = _resolve_current_trace_names(
        looped_records=looped_records,
        current_amp=current_amp if current_amp is not None else sim_amps[-1],
        current_names=current_names,
        max_auto_currents=max_auto_currents,
    )
    show_currents = bool(include_currents and current_trace_names)
    figure_rows = 2 if show_currents else 1
    fig, axes = plt.subplots(
        figure_rows,
        1,
        figsize=(8, 4 if figure_rows == 1 else 7),
        sharex=False,
    )
    if figure_rows == 1:
        voltage_axis = axes
        current_axis = None
    else:
        voltage_axis, current_axis = axes

    colors = (
        {float(amp): str(color) for amp, color in amplitude_colors.items()}
        if amplitude_colors is not None
        else active_amplitude_colors(
            sim_amps,
            single_trace_color=trace_color,
        )
    )
    for amp_value in sim_amps:
        plot_kwargs = {
            "label": f"{float(amp_value):g} pA",
            "color": colors.get(float(amp_value)),
        }
        voltage_axis.plot(
            looped_records["T"][amp_value],
            looped_records["V"][amp_value],
            **plot_kwargs,
        )
    _shade_stimulus(voltage_axis, sim_params)
    voltage_axis.set_xlabel("Time (ms)")
    voltage_axis.set_ylabel("Vm (mV)")
    voltage_axis.set_title(f"{cell_name} {tune_name} active sweep")
    voltage_axis.grid(True, alpha=0.3)
    voltage_axis.legend(loc="best")
    _apply_limits(voltage_axis, xlim=xlim, ylim=ylim)

    if current_axis is not None:
        selected_amp = current_amp if current_amp is not None else sim_amps[-1]
        current_records = looped_records.get("I", {}).get(selected_amp, {}) or {}
        for current_name in current_trace_names:
            current_axis.plot(
                looped_records["T"][selected_amp],
                current_records[current_name],
                label=current_name,
            )
        _shade_stimulus(current_axis, sim_params)
        current_axis.set_xlabel("Time (ms)")
        current_axis.set_ylabel("Current (recorded units)")
        current_axis.set_title(f"Recorded currents @ {float(selected_amp):g} pA")
        current_axis.grid(True, alpha=0.3)
        current_axis.legend(loc="best", fontsize="small")
        _apply_limits(current_axis, xlim=current_xlim or xlim, ylim=current_ylim)

    fig.tight_layout()
    return fig


def plot_fi_curve(
    *,
    fi_rows: Sequence[Mapping[str, Any]],
    cell_name: str,
    tune_name: str,
    bio_reference: Optional[Sequence[tuple[float, float]]] = None,
    show_bio_reference: bool = True,
    model_color: Optional[str] = None,
    reference_color: Optional[str] = None,
) -> Any:
    """Plot modeled FI points with optional biological reference points."""
    import matplotlib.pyplot as plt

    amp_values = [float(row["amp_pA"]) for row in fi_rows]
    frequency_values = [float(row["spike_frequency_hz"]) for row in fi_rows]
    colors = fi_series_colors(
        model_color=model_color,
        reference_color=reference_color,
    )

    fig, axis = plt.subplots(figsize=(6, 3.5))
    axis.plot(
        amp_values,
        frequency_values,
        marker="o",
        color=colors["model"],
        label=f"{cell_name} model",
    )
    if show_bio_reference and bio_reference:
        bio_amps = [float(point[0]) for point in bio_reference]
        bio_freqs = [float(point[1]) for point in bio_reference]
        axis.plot(
            bio_amps,
            bio_freqs,
            marker="o",
            linestyle="--",
            color=colors["reference"],
            label=f"{cell_name} reference",
        )

    axis.set_title(f"{cell_name} {tune_name} FI curve")
    axis.set_xlabel("Stimulus amplitude (pA)")
    axis.set_ylabel("Frequency (Hz)")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    return fig


def _is_black_color(color: Optional[str]) -> bool:
    if color is None:
        return False
    text = str(color).strip().lower()
    return text in {"k", "black", "#000", "#000000", "0", "0.0"}


def _safe_mean(values: np.ndarray) -> Optional[float]:
    return float(np.mean(values)) if values.size else None


def _safe_min(values: np.ndarray) -> Optional[float]:
    return float(np.min(values)) if values.size else None


def _safe_max(values: np.ndarray) -> Optional[float]:
    return float(np.max(values)) if values.size else None


def _safe_cv(values: np.ndarray) -> Optional[float]:
    if values.size < 2:
        return None
    mean_value = float(np.mean(values))
    if mean_value == 0:
        return None
    return float(np.std(values) / mean_value)


def _adaptation_ratio(interspike_intervals: np.ndarray) -> Optional[float]:
    if interspike_intervals.size < 2:
        return None
    first_interval = float(interspike_intervals[0])
    if first_interval == 0:
        return None
    return float(interspike_intervals[-1] / first_interval)


def _shade_stimulus(axis: Any, sim_params: Mapping[str, Any]) -> None:
    stim_start = float(sim_params["stim_delay"])
    stim_stop = stim_start + float(sim_params["stim_dur"])
    axis.axvspan(stim_start, stim_stop, alpha=0.08, color="gray", label="stimulus")


def _apply_limits(
    axis: Any,
    *,
    xlim: Optional[tuple[Optional[float], Optional[float]]] = None,
    ylim: Optional[tuple[Optional[float], Optional[float]]] = None,
) -> None:
    if xlim is not None:
        axis.set_xlim(*xlim)
    if ylim is not None:
        axis.set_ylim(*ylim)


def _resolve_current_trace_names(
    *,
    looped_records: Mapping[str, Any],
    current_amp: float,
    current_names: Optional[Sequence[str]],
    max_auto_currents: int,
) -> list[str]:
    current_records = looped_records.get("I", {}).get(current_amp, {}) or {}
    available = sorted(str(name) for name in current_records)
    if current_names:
        requested = [str(name) for name in current_names]
        return [name for name in requested if name in current_records]
    return available[: max(0, int(max_auto_currents))]
