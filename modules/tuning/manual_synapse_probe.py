"""Independent one-synapse checks for the Step 4 tuning notebook.

The probe intentionally uses SCP's cell/runtime helpers instead of BMTool's
simulation or widget machinery.  Its short-term-plasticity formulas mirror the
current BMTool definitions so the notebook can compare like-for-like metrics
while retaining an independent execution path.
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


def _normalize_spike_times(spike_times_ms: Sequence[float]) -> np.ndarray:
    times = np.asarray(list(spike_times_ms), dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("spike_times_ms must contain at least one spike time.")
    if not np.all(np.isfinite(times)):
        raise ValueError("spike_times_ms must contain only finite values.")
    if np.any(times < 0.0):
        raise ValueError("spike_times_ms cannot contain negative times.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("spike_times_ms must be strictly increasing.")
    return times


def _trace_arrays(traces: Mapping[str, Any], *, label: str) -> tuple[np.ndarray, ...]:
    time = np.asarray(traces.get("T", []), dtype=float)
    voltage = np.asarray(traces.get("V", []), dtype=float)
    current = np.asarray(traces.get("I", []), dtype=float)
    if not time.size or time.size != voltage.size:
        raise ValueError(f"{label} traces must contain equal-length T and V arrays.")
    if current.size != time.size:
        raise ValueError(
            f"{label} traces do not contain the tested synapse current as I. "
            "The point process must expose an i variable and the cell must not "
            "contain another pre-attached synapse."
        )
    return time, voltage, current


def _sample_index(time: np.ndarray, event_time_ms: float) -> int:
    index = int(np.searchsorted(time, float(event_time_ms), side="left"))
    return min(max(index, 0), time.size - 1)


def _response_polarity(signal: np.ndarray, start_index: int) -> float:
    response = signal[int(start_index) :]
    if response.size == 0:
        return 1.0
    extremum = float(response[int(np.argmax(np.abs(response)))])
    sign = float(np.sign(extremum))
    return sign if sign else 1.0


def _first_crossing(signal: np.ndarray, threshold: float) -> Optional[int]:
    indices = np.flatnonzero(signal >= float(threshold))
    return int(indices[0]) if indices.size else None


def _single_event_current_metrics(
    time: np.ndarray,
    current_nA: np.ndarray,
    *,
    source_time_ms: float,
    rise_interval: tuple[float, float],
) -> dict[str, float]:
    source_index = _sample_index(time, source_time_ms)
    baseline_nA = float(current_nA[source_index])
    delta_nA = current_nA - baseline_nA
    polarity = _response_polarity(delta_nA, source_index)
    magnitude_nA = delta_nA * polarity
    response = magnitude_nA[source_index:]
    peak_offset = int(np.argmax(response))
    peak_index = source_index + peak_offset
    peak_nA = float(max(magnitude_nA[peak_index], 0.0))

    rising = magnitude_nA[source_index : peak_index + 1]
    rising_diff = np.diff(rising)
    onset_candidates = np.flatnonzero(rising_diff > 0.0)
    onset_offset = int(onset_candidates[0] + 1) if onset_candidates.size else 0
    onset_index = source_index + onset_offset

    low_fraction, high_fraction = (float(value) for value in rise_interval)
    rise_signal = magnitude_nA[onset_index : peak_index + 1]
    low_crossing = _first_crossing(rise_signal, low_fraction * peak_nA)
    high_crossing = _first_crossing(rise_signal, high_fraction * peak_nA)
    if low_crossing is None or high_crossing is None:
        rise_time_ms = math.nan
    else:
        rise_time_ms = float(
            time[onset_index + high_crossing] - time[onset_index + low_crossing]
        )

    half_rise = _first_crossing(rise_signal, 0.5 * peak_nA)
    decay_signal = magnitude_nA[peak_index:]
    below_half = np.flatnonzero(decay_signal <= 0.5 * peak_nA)
    if half_rise is None or not below_half.size:
        half_width_ms = math.nan
    else:
        half_width_ms = float(
            time[peak_index + int(below_half[0])] - time[onset_index + half_rise]
        )

    decay_time_ms = math.nan
    if peak_nA > 0.0 and decay_signal.size >= 3:
        upward_turn = np.flatnonzero(np.diff(decay_signal) > 0.0)
        decay_end = int(upward_turn[0] + 1) if upward_turn.size else decay_signal.size
        decay_y = np.asarray(decay_signal[:decay_end], dtype=float)
        decay_t = np.asarray(time[peak_index : peak_index + decay_end] - time[peak_index])
        if decay_y.size >= 3 and float(decay_t[-1]) > 0.0:
            try:
                from scipy.optimize import curve_fit

                fit, _ = curve_fit(
                    lambda values, amplitude, tau: amplitude * np.exp(-values / tau),
                    decay_t,
                    decay_y,
                    p0=(peak_nA, max(float(decay_t[-1]) / 2.0, 1e-9)),
                    bounds=((0.0, 1e-12), (np.inf, np.inf)),
                    maxfev=10000,
                )
                decay_time_ms = float(fit[1])
            except (RuntimeError, TypeError, ValueError, FloatingPointError):
                decay_time_ms = math.nan

    integral_fn = getattr(np, "trapezoid", np.trapz)
    current_integral_pA_ms = float(
        integral_fn(delta_nA[source_index:] * 1000.0, time[source_index:])
    )
    return {
        "single_current_baseline_nA": baseline_nA,
        "single_current_polarity": polarity,
        "single_event_peak_amplitude_pA": peak_nA * 1000.0,
        "latency_ms": float(time[onset_index] - source_time_ms),
        "rise_time_ms": rise_time_ms,
        "decay_time_ms": decay_time_ms,
        "half_width_ms": half_width_ms,
        "current_integral_pA_ms": current_integral_pA_ms,
    }


def _train_current_metrics(
    time: np.ndarray,
    current_nA: np.ndarray,
    *,
    spike_times_ms: np.ndarray,
    netcon_delay_ms: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    source_index = _sample_index(time, float(spike_times_ms[0]))
    baseline_nA = float(current_nA[source_index])
    delta_nA = current_nA - baseline_nA
    polarity = _response_polarity(delta_nA, source_index)
    magnitude_nA = delta_nA * polarity

    delivered_times_ms = spike_times_ms + float(netcon_delay_ms)
    event_indices = np.searchsorted(time, delivered_times_ms, side="left").astype(int)
    window_ends = np.r_[event_indices[1:], time.size]
    pulse_amplitudes_pA: list[float] = []
    pulse_peak_indices: list[int] = []
    for start, stop in zip(event_indices, window_ends):
        start = min(max(int(start), 0), time.size - 1)
        stop = min(max(int(stop), start + 1), time.size)
        window = magnitude_nA[start:stop]
        offset = int(np.argmax(window))
        pulse_peak_indices.append(start + offset)
        pulse_amplitudes_pA.append(float(max(window[offset], 0.0) * 1000.0))

    amplitudes = np.asarray(pulse_amplitudes_pA, dtype=float)
    metrics: dict[str, Any] = {
        "train_current_baseline_nA": baseline_nA,
        "train_current_polarity": polarity,
        "first_pulse_amplitude_pA": float(amplitudes[0]),
        "max_amplitude_pA": float(np.max(amplitudes)),
        "stp_protocol_complete": bool(amplitudes.size == 12),
        "simple_ppr": math.nan,
        "ppr": math.nan,
        "induction": math.nan,
        "recovery": math.nan,
    }

    # Each BMTool-compatible metric becomes meaningful at a different point
    # in the protocol: PPR uses pulses 1-2, induction uses pulses 1-8, and
    # recovery uses the complete 8 + recovery + 4 layout.
    percentile_90 = float(np.percentile(amplitudes, 90))
    first = float(amplitudes[0])
    if amplitudes.size >= 2:
        second = float(amplitudes[1])
        if first != 0.0:
            metrics["simple_ppr"] = second / first
        if percentile_90 != 0.0:
            metrics["ppr"] = (second - first) / percentile_90

    if amplitudes.size >= 8 and percentile_90 != 0.0:
        metrics["induction"] = (
            float(np.mean(amplitudes[5:8])) - first
        ) / percentile_90

    if amplitudes.size >= 12 and percentile_90 != 0.0:
        metrics["recovery"] = (
            float(np.mean(amplitudes[8:12]))
            - float(np.mean(amplitudes[0:4]))
        ) / percentile_90

    return metrics, amplitudes, np.asarray(pulse_peak_indices, dtype=int)


def calculate_manual_synapse_metrics(
    *,
    single_event_traces: Mapping[str, Any],
    train_traces: Mapping[str, Any],
    spike_times_ms: Sequence[float],
    netcon_delay_ms: float = 1.0,
    baseline_window_ms: float = 100.0,
    rise_interval: Sequence[float] = (0.1, 0.9),
) -> dict[str, Any]:
    """Calculate voltage, single-event, and BMTool-compatible train metrics."""

    spikes = _normalize_spike_times(spike_times_ms)
    if len(rise_interval) != 2:
        raise ValueError("rise_interval must contain exactly two fractions.")
    rise_pair = (float(rise_interval[0]), float(rise_interval[1]))
    if not 0.0 <= rise_pair[0] < rise_pair[1] <= 1.0:
        raise ValueError("rise_interval must satisfy 0 <= low < high <= 1.")
    if float(baseline_window_ms) <= 0.0:
        raise ValueError("baseline_window_ms must be greater than zero.")

    single_t, _, single_i = _trace_arrays(single_event_traces, label="Single-event")
    train_t, train_v, train_i = _trace_arrays(train_traces, label="Train")
    first_spike_ms = float(spikes[0])
    baseline_start = first_spike_ms - float(baseline_window_ms)
    baseline_mask = (train_t >= baseline_start) & (train_t < first_spike_ms)
    if not np.any(baseline_mask):
        raise ValueError(
            "The train trace does not contain the requested pre-stimulus baseline window."
        )
    response_mask = train_t >= first_spike_ms
    if not np.any(response_mask):
        raise ValueError("The train trace ends before the first spike time.")

    rest_voltage_mV = float(np.mean(train_v[baseline_mask]))
    response_voltage = train_v[response_mask]
    peak_voltage_mV = float(np.max(response_voltage))
    trough_voltage_mV = float(np.min(response_voltage))
    deviations = response_voltage - rest_voltage_mV
    largest_index = int(np.argmax(np.abs(deviations)))

    metrics: dict[str, Any] = {
        "rest_voltage_mV": rest_voltage_mV,
        "peak_voltage_mV": peak_voltage_mV,
        "trough_voltage_mV": trough_voltage_mV,
        "psp_amplitude_mV": float(deviations[largest_index]),
    }
    metrics.update(
        _single_event_current_metrics(
            single_t,
            single_i,
            source_time_ms=first_spike_ms,
            rise_interval=rise_pair,
        )
    )
    train_metrics, amplitudes, peak_indices = _train_current_metrics(
        train_t,
        train_i,
        spike_times_ms=spikes,
        netcon_delay_ms=float(netcon_delay_ms),
    )
    metrics.update(train_metrics)
    metrics["pulse_amplitudes_pA"] = amplitudes
    metrics["pulse_peak_times_ms"] = train_t[peak_indices]
    return metrics


def _nonempty_synapse_state(cell: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in ("synapses", "netcons", "stims", "vecs"):
        value = getattr(cell, name, None)
        try:
            count = len(value) if value is not None else 0
        except TypeError:
            count = 0
        if count:
            counts[name] = int(count)
    return counts


def _remove_state_object(cell: Any, container_name: str, value: Any) -> None:
    container = getattr(cell, container_name, None)
    if container is None:
        return
    try:
        container.remove(value)
    except (AttributeError, ValueError):
        pass


def run_manual_synapse_probe(
    cell: Any,
    *,
    connection_settings: Mapping[str, Any],
    general_settings: Mapping[str, Any],
    section_group: str = "soma",
    section_index: int = 0,
    section_x: float = 0.5,
    spike_times_ms: Sequence[float],
    param_overrides: Optional[Mapping[str, float]] = None,
    tstop_ms: Optional[float] = None,
    tail_ms: float = 150.0,
    baseline_window_ms: float = 100.0,
    v_init_mV: Optional[float] = None,
    celsius_C: Optional[float] = None,
) -> dict[str, Any]:
    """Run independent single-event and train checks on one SCP synapse.

    The caller must use a clean, already-built Step 4 cell and invoke this
    before constructing BMTool's tuner.  All objects attached by this function
    are deactivated and removed from the cell's notebook state containers.
    """

    from neuron import h

    from modules.model.geometry import cell_sections
    from modules.notebooks.helpers import add_single_synapse_for_notebook
    from modules.simulation.cell_runtime import run_cell

    spikes = _normalize_spike_times(spike_times_ms)
    existing = _nonempty_synapse_state(cell)
    if existing:
        detail = ", ".join(f"{name}={count}" for name, count in existing.items())
        raise RuntimeError(
            "The manual synapse probe requires a clean Step 4 cell; found "
            f"{detail}. Run it before other synapses are attached or restart "
            "the kernel and rerun through the Step 4 configuration cell."
        )

    sections = cell_sections(cell, str(section_group))
    index = int(section_index)
    location = float(section_x)
    if not sections:
        raise ValueError(f"No canonical section group {section_group!r} exists on the cell.")
    if index < 0 or index >= len(sections):
        raise IndexError(
            f"Section index out of range for {section_group!r}: "
            f"index={index}, n={len(sections)}."
        )
    if not 0.0 <= location <= 1.0:
        raise ValueError(f"section_x must be in [0, 1], got {location}.")
    if float(tail_ms) <= 0.0:
        raise ValueError("tail_ms must be greater than zero.")

    settings = deepcopy(dict(connection_settings))
    spec_settings = settings.get("spec_settings")
    spec_params = settings.get("spec_syn_param")
    if not isinstance(spec_settings, dict) or not isinstance(spec_params, dict):
        raise ValueError(
            "connection_settings must contain spec_settings and spec_syn_param objects."
        )
    if param_overrides:
        spec_params.update(dict(param_overrides))

    netcon_weight = float(general_settings.get("weight", 1.0))
    netcon_delay_ms = float(general_settings.get("delay", 1.0))
    dt_ms = float(general_settings.get("dt", 0.025))
    single_duration_ms = float(general_settings.get("tdur", 100.0))
    resolved_tstop_ms = float(
        tstop_ms if tstop_ms is not None else float(spikes[-1]) + float(tail_ms)
    )
    if resolved_tstop_ms <= float(spikes[-1]) + netcon_delay_ms:
        raise ValueError("tstop_ms must extend beyond the final delivered synaptic event.")

    conditions: dict[str, float] = {}
    if v_init_mV is not None:
        conditions["v_init_mV"] = float(v_init_mV)
    if celsius_C is not None:
        conditions["celsius_C"] = float(celsius_C)
    elif general_settings.get("celsius") is not None:
        conditions["celsius_C"] = float(general_settings["celsius"])

    section = sections[index]
    synapse_state: Optional[dict[str, Any]] = None
    train_vector = None
    try:
        synapse_state = add_single_synapse_for_notebook(
            cell,
            section(location),
            settings,
            [float(spikes[0])],
            netcon_weight=netcon_weight,
        )
        synapse_state["netcon"].delay = netcon_delay_ms
        synapse = synapse_state["syn"]
        missing_params = sorted(name for name in spec_params if not hasattr(synapse, name))
        if missing_params:
            raise AttributeError(
                "Configured synapse parameter(s) are not exposed by "
                f"{spec_settings.get('level_of_detail')!r}: {missing_params}."
            )

        actual_params = {
            name: float(getattr(synapse, name))
            for name in spec_params
            if isinstance(getattr(synapse, name), (int, float))
        }
        single_event_traces = run_cell(
            cell,
            {
                "dt": dt_ms,
                "tstart": 0.0,
                "tstop": float(spikes[0]) + single_duration_ms,
                "conditions": conditions,
            },
        )

        train_vector = h.Vector(spikes.tolist())
        synapse_state["stim"].play(train_vector)
        train_traces = run_cell(
            cell,
            {
                "dt": dt_ms,
                "tstart": 0.0,
                "tstop": resolved_tstop_ms,
                "conditions": conditions,
            },
        )
        metrics = calculate_manual_synapse_metrics(
            single_event_traces=single_event_traces,
            train_traces=train_traces,
            spike_times_ms=spikes,
            netcon_delay_ms=netcon_delay_ms,
            baseline_window_ms=float(baseline_window_ms),
            rise_interval=general_settings.get("rise_interval", (0.1, 0.9)),
        )
    finally:
        if synapse_state is not None:
            try:
                synapse_state["netcon"].active(0)
            except Exception:
                pass
            _remove_state_object(cell, "synapses", synapse_state.get("syn"))
            _remove_state_object(cell, "netcons", synapse_state.get("netcon"))
            _remove_state_object(cell, "stims", synapse_state.get("stim"))
            _remove_state_object(cell, "vecs", synapse_state.get("vec"))

    return {
        "mechanism": str(spec_settings.get("level_of_detail", "")),
        "section_group": str(section_group),
        "section_index": index,
        "section_x": location,
        "section_name": str(section.name()),
        "spike_times_ms": spikes,
        "delivered_times_ms": spikes + netcon_delay_ms,
        "netcon_delay_ms": netcon_delay_ms,
        "netcon_weight": netcon_weight,
        "parameters": actual_params,
        "single_event_traces": single_event_traces,
        "train_traces": train_traces,
        "metrics": metrics,
    }


def plot_manual_synapse_probe(
    result: Mapping[str, Any],
    *,
    figsize: tuple[float, float] = (8.0, 5.0),
    xlim: Optional[tuple[float, float]] = None,
    voltage_ylim: Optional[tuple[float, float]] = None,
    current_ylim: Optional[tuple[float, float]] = None,
    voltage_color: str = "C0",
    current_color: str = "C1",
    spike_color: str = "0.25",
    linewidth: float = 1.5,
    save_path: Optional[str | Path] = None,
    dpi: int = 150,
    show: bool = True,
) -> tuple[Any, Any, Optional[Path]]:
    """Plot the free membrane voltage and synaptic current for the train run."""

    import matplotlib.pyplot as plt

    time, voltage, current_nA = _trace_arrays(result["train_traces"], label="Train")
    metrics = result["metrics"]
    current_pA = (
        current_nA - float(metrics["train_current_baseline_nA"])
    ) * 1000.0

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axes[0].plot(time, voltage, color=voltage_color, linewidth=float(linewidth))
    axes[0].axhline(
        float(metrics["rest_voltage_mV"]),
        color="0.55",
        linestyle=":",
        linewidth=1.0,
        label="rest",
    )
    axes[0].set_ylabel("Soma Vm (mV)")
    axes[0].legend(loc="best", fontsize="small")

    axes[1].plot(time, current_pA, color=current_color, linewidth=float(linewidth))
    axes[1].axhline(0.0, color="0.55", linestyle=":", linewidth=1.0)
    axes[1].set_ylabel("Synaptic current (pA)")
    axes[1].set_xlabel("Time (ms)")

    for axis in axes:
        for spike_time in np.asarray(result["spike_times_ms"], dtype=float):
            axis.axvline(
                spike_time,
                color=spike_color,
                linestyle="--",
                linewidth=0.8,
                alpha=0.55,
            )
        axis.grid(True, alpha=0.25)
        if xlim is not None:
            axis.set_xlim(*xlim)
    if voltage_ylim is not None:
        axes[0].set_ylim(*voltage_ylim)
    if current_ylim is not None:
        axes[1].set_ylim(*current_ylim)

    fig.suptitle(
        "Manual one-synapse probe: "
        f"{result['mechanism']} at {result['section_name']}({float(result['section_x']):g})"
    )
    fig.tight_layout()

    saved_path = None
    if save_path is not None:
        saved_path = Path(save_path).expanduser()
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved_path, dpi=int(dpi), bbox_inches="tight")
        print(f"Saved manual synapse plot: {saved_path.resolve()}")
    if show:
        plt.show()
    return fig, axes, saved_path


__all__ = [
    "calculate_manual_synapse_metrics",
    "plot_manual_synapse_probe",
    "run_manual_synapse_probe",
]
