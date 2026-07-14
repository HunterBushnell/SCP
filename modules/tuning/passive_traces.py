"""Generic passive-trace target extraction for Step 2."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .allen_nwb import aggregate_passive_sweeps


def write_passive_trace_target_csv(
    trace_path: str | Path,
    target_path: str | Path,
    *,
    act_passive_module: Any,
    trace_format: str = "csv",
    sweep_summary_path: Optional[str | Path] = None,
    time_column: str = "time_ms",
    voltage_column: str = "voltage_mV",
    current_column: Optional[str] = "current_pA",
    sweep_column: Optional[str] = None,
    stim_start_ms: Optional[float] = None,
    stim_stop_ms: Optional[float] = None,
    current_pA: Optional[float | Sequence[float]] = None,
    dt_ms: Optional[float] = None,
    end_margin_ms: float = 10.0,
    reducer: str = "median",
    tau_field: str = "tau_avg_ms",
) -> dict[str, Any]:
    """Write passive target and sweep-summary CSVs from generic traces."""
    sweep_rows = extract_passive_trace_sweeps(
        trace_path,
        act_passive_module=act_passive_module,
        trace_format=trace_format,
        time_column=time_column,
        voltage_column=voltage_column,
        current_column=current_column,
        sweep_column=sweep_column,
        stim_start_ms=stim_start_ms,
        stim_stop_ms=stim_stop_ms,
        current_pA=current_pA,
        dt_ms=dt_ms,
        end_margin_ms=end_margin_ms,
    )
    targets = aggregate_passive_sweeps(
        sweep_rows,
        reducer=reducer,
        tau_field=tau_field,
    )
    target_path = Path(target_path)
    _write_rows_csv(target_path, [targets])
    if sweep_summary_path is not None:
        _write_rows_csv(sweep_summary_path, sweep_rows)
    return {
        "source": str(Path(trace_path).expanduser()),
        "targets": targets,
        "sweeps": sweep_rows,
        "target_csv": str(target_path),
        "sweep_summary_csv": str(sweep_summary_path) if sweep_summary_path else None,
    }


def extract_passive_trace_sweeps(
    trace_path: str | Path,
    *,
    act_passive_module: Any,
    trace_format: str = "csv",
    time_column: str = "time_ms",
    voltage_column: str = "voltage_mV",
    current_column: Optional[str] = "current_pA",
    sweep_column: Optional[str] = None,
    stim_start_ms: Optional[float] = None,
    stim_stop_ms: Optional[float] = None,
    current_pA: Optional[float | Sequence[float]] = None,
    dt_ms: Optional[float] = None,
    end_margin_ms: float = 10.0,
) -> list[dict[str, Any]]:
    """Compute ACT passive metrics from generic CSV or NPY voltage traces."""
    fmt = str(trace_format or "csv").strip().lower()
    if fmt == "csv":
        traces = _load_csv_traces(
            trace_path,
            time_column=time_column,
            voltage_column=voltage_column,
            current_column=current_column,
            sweep_column=sweep_column,
        )
    elif fmt == "npy":
        traces = _load_npy_traces(
            trace_path,
            dt_ms=dt_ms,
            current_pA=current_pA,
        )
    else:
        raise ValueError("Generic passive trace format must be 'csv' or 'npy'.")

    rows = []
    for index, trace in enumerate(traces):
        time_ms = trace["time_ms"]
        voltage_mV = trace["voltage_mV"]
        current_trace_pA = trace.get("current_pA")
        resolved_dt_ms = _resolve_dt_ms(time_ms, dt_ms)
        resolved_start, resolved_stop, resolved_current = _resolve_passive_window(
            time_ms=time_ms,
            current_trace_pA=current_trace_pA,
            stim_start_ms=stim_start_ms,
            stim_stop_ms=stim_stop_ms,
            current_pA=_select_current(current_pA, index),
        )
        metric_stop_ms = max(
            float(resolved_start) + float(resolved_dt_ms),
            float(resolved_stop) - float(end_margin_ms),
        )
        gpp = act_passive_module.compute_gpp(
            voltage_mV,
            float(resolved_dt_ms),
            float(resolved_start),
            float(metric_stop_ms),
            float(resolved_current) / 1000.0,
        )
        row = {
            "sweep": index,
            "source_sweep": trace.get("sweep", index),
            "amp_pA": float(resolved_current),
            "stim_start_ms": float(resolved_start),
            "stim_stop_ms": float(resolved_stop),
            "metric_stop_ms": float(metric_stop_ms),
            "dt_ms": float(resolved_dt_ms),
            "sample_rate_hz": float(1000.0 / resolved_dt_ms),
        }
        for attr, out_name in (
            ("R_in_rest_to_final", "rin_MOhm"),
            ("tau_rest_to_trough", "tau_rest_to_trough_ms"),
            ("tau_avg", "tau_avg_ms"),
            ("sag_ratio", "sag_ratio"),
            ("V_rest", "v_rest_mV"),
        ):
            if hasattr(gpp, attr):
                row[out_name] = float(getattr(gpp, attr))
        rows.append(row)

    if not rows:
        raise ValueError(f"No passive traces found in {trace_path!s}.")
    return rows


def _load_csv_traces(
    trace_path: str | Path,
    *,
    time_column: str,
    voltage_column: str,
    current_column: Optional[str],
    sweep_column: Optional[str],
) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, str]]] = defaultdict(list)
    with Path(trace_path).expanduser().open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Trace CSV has no header: {trace_path}")
        required = {time_column, voltage_column}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Trace CSV missing required columns: {missing}")
        for row_index, row in enumerate(reader):
            sweep_key = row.get(sweep_column) if sweep_column else "trace"
            grouped[sweep_key or row_index].append(row)

    traces = []
    for sweep, rows in grouped.items():
        rows = sorted(rows, key=lambda row: float(row[time_column]))
        time_ms = np.asarray([float(row[time_column]) for row in rows], dtype=float)
        voltage_mV = np.asarray([float(row[voltage_column]) for row in rows], dtype=float)
        current_pA = None
        if current_column and current_column in rows[0] and rows[0].get(current_column) not in (None, ""):
            current_pA = np.asarray([float(row[current_column]) for row in rows], dtype=float)
        traces.append(
            {
                "sweep": sweep,
                "time_ms": time_ms,
                "voltage_mV": voltage_mV,
                "current_pA": current_pA,
            }
        )
    return traces


def _load_npy_traces(
    trace_path: str | Path,
    *,
    dt_ms: Optional[float],
    current_pA: Optional[float | Sequence[float]],
) -> list[dict[str, Any]]:
    if dt_ms is None:
        raise ValueError("Generic passive NPY traces require traces.passive.dt_ms.")
    if current_pA in (None, ""):
        raise ValueError("Generic passive NPY traces require traces.passive.current_pA.")
    data = np.load(Path(trace_path).expanduser())
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        voltage_rows = data.reshape(1, -1)
    elif data.ndim == 2:
        voltage_rows = data
    else:
        raise ValueError("Generic passive NPY traces must be 1D or 2D voltage arrays.")
    time_ms = np.arange(voltage_rows.shape[1], dtype=float) * float(dt_ms)
    return [
        {
            "sweep": index,
            "time_ms": time_ms,
            "voltage_mV": voltage,
            "current_pA": None,
        }
        for index, voltage in enumerate(voltage_rows)
    ]


def _resolve_dt_ms(time_ms: np.ndarray, dt_ms: Optional[float]) -> float:
    if dt_ms is not None:
        return float(dt_ms)
    if len(time_ms) < 2:
        raise ValueError("At least two time points are required to infer dt_ms.")
    diffs = np.diff(time_ms)
    dt = float(np.median(diffs))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Could not infer a positive dt_ms from trace time values.")
    return dt


def _resolve_passive_window(
    *,
    time_ms: np.ndarray,
    current_trace_pA: Optional[np.ndarray],
    stim_start_ms: Optional[float],
    stim_stop_ms: Optional[float],
    current_pA: Optional[float],
) -> tuple[float, float, float]:
    if stim_start_ms is not None and stim_stop_ms is not None and current_pA not in (None, ""):
        return float(stim_start_ms), float(stim_stop_ms), float(current_pA)
    if current_trace_pA is None:
        raise ValueError(
            "Generic passive traces require stim_start_ms, stim_stop_ms, and current_pA "
            "unless the CSV includes a current_pA column for automatic window detection."
        )
    baseline = float(np.median(current_trace_pA[: max(1, len(current_trace_pA) // 20)]))
    delta = np.abs(current_trace_pA - baseline)
    threshold = max(1e-9, float(np.nanmax(delta)) * 0.5)
    active_indices = np.flatnonzero(delta > threshold)
    if len(active_indices) == 0:
        raise ValueError("Could not detect a current-injection window from current_pA column.")
    start_idx = int(active_indices[0])
    stop_idx = int(active_indices[-1]) + 1
    start = float(stim_start_ms if stim_start_ms is not None else time_ms[start_idx])
    stop = float(stim_stop_ms if stim_stop_ms is not None else time_ms[min(stop_idx, len(time_ms) - 1)])
    active_current = float(
        current_pA
        if current_pA not in (None, "")
        else np.median(current_trace_pA[start_idx:stop_idx])
    )
    return start, stop, active_current


def _select_current(value: Optional[float | Sequence[float]], index: int) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, (str, int, float)):
        return float(value)
    values = list(value)
    if len(values) == 1:
        return float(values[0])
    if index >= len(values):
        raise ValueError("Not enough current_pA values for all passive NPY traces.")
    return float(values[index])


def _write_rows_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
