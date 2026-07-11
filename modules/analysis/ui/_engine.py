from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import csv
from datetime import datetime
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from pathlib import Path
import textwrap

from modules import run_sim
from modules.input_generation import inputs
from modules.input_generation import sampling as input_sampling
from .. import analysis, plotting

HELP_SELECTION = textwrap.dedent(
    """
    Selection UI
    - Cell/Tune/Model: choose which output_data tree to browse.
    - Compare list: multi-select run folders to plot/compare.
    - Compare paths: comma-separated curve files. Syntax: path@shift:scale;key=val
      Keys: color, label, linestyle, shift, scale (shift in ms, scale is multiplicative).
    - compare_list_dir_paths (defaults JSON): add folders of CSV curves to the Compare list.
    - Use compare paths toggle: enables/disables Compare paths without deleting them.
    - compare_list_paths entries can be objects with {path, enabled, shift_ms, scale, color, label, linestyle}.
    - Tokens: latest/previous/prev/latest-1 resolve to runs in output_data.
    - Save plots/analysis toggles affect saved figures/JSON.
    Defaults live in modules/analysis/analysis_defaults.json.
    """
).strip()

HELP_OUTPUTS = textwrap.dedent(
    """
    Outputs UI

    Run
    - Full output: generates the full saved-output plot for the selected run/compare selection.
    - Rate/ISI curve: generates the compact average firing-rate, ISI, or stacked curve plot.
    - Spike stats: optionally summarizes trial spike counts/timing for single-run outputs.
    - Output raster: adds spike raster content where the selected plot supports it.
    - Raster: dot or line rendering for raster spikes.
    - Smooth ms: smoothing window for the full output plot.

    Window
    - x start/x stop: manually crop the time axis in ms.
    - y min/y max: manually crop output-rate axes; blank uses automatic limits.
    - Auto x-window: uses the stimulus window plus Window +/-ms padding when stim bounds are available.
    - Stim start/stop: optional manual stimulus-window override in ms.
    - x origin 0: displays the x-axis relative to the crop start; data and metrics are unchanged.

    Curve
    - Units: raw keeps Hz; normalized rescales to the selected baseline/stim normalization.
    - Curve type: choose rate, ISI, or stacked rate+ISI.
    - Normalize by: avg or peak value in the normalization window.
    - Fixed norm: optional fixed firing-rate divisor for multi-run output plots.
    - Bin ms: bin size for output-rate/ISI curves.
    - Smooth mode: causal or centered smoothing.

    Compare
    - Layout: side-by-side, stacked, or overlay when multiple runs/curves are selected.
    - Band: optional sem/std shading where the selected plot supports repeated trials.
    - Use compare preset: loads entries/options from compare_preset_path when configured.

    Export
    - Save path: optional output path; blank uses a default plots/analysis location.
    - Save type: CSV plotted data, PNG image, or SVG image.
    - CSV format: Trace rows stores one row per plotted trace; Long rows stores one row per point.
    - Auto-save: saves after each Run output plots click.
    """
).strip()

HELP_INPUTS = textwrap.dedent(
    """
    Inputs UI

    Run
    - Mean rates: plots per-group input-rate summaries.
    - Input raster: plots saved input spike trains; available for single-run selections.
    - Show band: adds uncertainty shading to mean-rate plots.
    - Input source: stats uses saved summaries, saved uses saved spike trains, auto chooses available data.
    - Band mode: std or sem for shaded uncertainty bands.
    - Legend: matplotlib legend location; none hides legends.

    Groups
    - Groups: comma-separated synapse groups; blank means all available groups.
    - Bin ms: bin size for input-rate summaries.
    - Smooth ms: smoothing window for input-rate summaries.

    Raster
    - Trial: saved-input trial index to display.
    - Max trains: maximum spike trains shown per group to avoid heavy plots.
    - Raster bin: bin/window size used by raster summary overlays.
    - Raster: dot or line rendering for raster spikes.

    Window
    - x start/x stop: manually crop the time axis in ms.
    - Auto x-window: uses the simulation stimulus window plus Window +/-ms padding.
    - Window +/-ms: padding around the stimulus window when auto window is enabled.

    Compare
    - Layout: side-by-side, stacked, or overlay when multiple runs are selected.
    - Compare band: adds uncertainty shading in compare plots where supported.

    Export
    - CSV path: optional output path; blank uses a default analysis location.
    - Format: Trace rows stores one row per plotted trace; Long rows stores one row per point.
    - Auto-save CSV: saves after each Run input plots click.
    """
).strip()

HELP_EXTRA = textwrap.dedent(
    """
    Extra UI
    - Output metrics: summary table for selected runs/curves.
    - Output metrics spread: choose std or sem across saved trials.
    - Metric dist plot: box/whisker or bar by run for selected metrics.
    - Bar mode uses mean bars; optional error bars (std/sem) apply to run entries only.
    - Trial points and point jitter can be toggled for cleaner panels.
    - Matrix shape is controlled by columns + panel size; legend can be toggled/relocated.
    - Table metrics can reuse the same metric selection as the distribution plot.
    - Compare configs: restore-style config compare across selected runs (sim/cell/geom/syn/syn_groups/fit).
    - Input sampling: synthesize input curves from synapse configs.
    - Synapse plots: summarize and plot saved synapse weights/distances from Step 5 outputs.
    - Snapshot compare: compare notebook vs slurm outputs.
    - Recording tables: summarize cell-recorded traces and total synaptic I/G.
    """
).strip()


def _print_help(out, text: str) -> None:
    if out is None:
        print(text)
        return
    with out:
        out.clear_output()
        print(text)


def compare_enabled(selection: Dict[str, Any]) -> bool:
    list_entries = _compare_list_entries(selection)
    if len(list_entries) >= 2:
        return True
    run_a_path = selection.get("run_a_path")
    run_b_path = selection.get("run_b_path")
    return bool(run_a_path and run_b_path)


def _coerce_run_path(path_val: Any, base_dir: Path) -> Optional[Path]:
    if path_val in (None, "", "none", "None"):
        return None
    token = str(path_val)
    if token in ("latest", "previous", "prev", "latest-1"):
        try:
            return analysis.resolve_run(base_dir, token)
        except Exception:
            pass
    p = Path(str(path_val)).expanduser()
    if not p.is_absolute():
        repo_root = analysis.find_scp_root(base_dir)
        if p.parts and p.parts[0] in ("external_data", "cells"):
            p = (repo_root / p).resolve()
        else:
            p = (base_dir / p).resolve()
    return p


def _safe_load_results(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return run_sim.load_results(path)
    except FileNotFoundError:
        return None


def _is_curve_path(path: Path) -> bool:
    return path.suffix.lower() in (".csv", ".tsv", ".txt")


def resolve_single(selection: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    run_dir = analysis.resolve_run(selection["base"], selection["run_single"])
    res = run_sim.load_results(run_dir)
    return run_dir, res


def resolve_compare(
    selection: Dict[str, Any],
) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not compare_enabled(selection):
        return None, None, None, None
    base_dir = selection["base"]
    run_a_path = _coerce_run_path(selection.get("run_a_path"), base_dir)
    run_b_path = _coerce_run_path(selection.get("run_b_path"), base_dir)
    run_a = run_a_path if run_a_path is not None else analysis.resolve_run(base_dir, selection["run_a"])
    run_b = run_b_path if run_b_path is not None else analysis.resolve_run(base_dir, selection["run_b"])
    res_a = None if (run_a is not None and _is_curve_path(Path(run_a))) else run_sim.load_results(run_a)
    res_b = None if (run_b is not None and _is_curve_path(Path(run_b))) else run_sim.load_results(run_b)
    return run_a, run_b, res_a, res_b


def _save_fig(fig, out_path, *, enabled: bool, dpi: int, overwrite: bool = False) -> None:
    analysis.save_figure(fig, out_path, enabled=enabled, dpi=dpi, overwrite=overwrite)


def _save_json(data: dict, out_path, *, enabled: bool) -> None:
    analysis.save_json(data, out_path, enabled=enabled)


_PLOT_DATA_FIELDS = [
    "figure_type",
    "mode",
    "plot_name",
    "axis_index",
    "trace_label",
    "series_kind",
    "run_label",
    "units",
    "time_ms",
    "value",
    "value_low",
    "value_high",
]

_PLOT_DATA_TRACE_FIELDS = [
    "trace_name",
    "trace_label",
    "series_kind",
    "figure_type",
    "mode",
    "plot_name",
    "axis_index",
    "run_label",
    "units",
    "n_points",
    "time_ms",
    "value",
    "value_low",
    "value_high",
]


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _normalize_label(value: Any, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text or text.startswith("_"):
        return fallback
    return text


def _slug_token(value: str) -> str:
    chars = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_"):
            chars.append(ch)
        else:
            chars.append("_")
    token = "".join(chars).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token or "plot"


def _entry_label_for_name(entry: Any) -> str:
    spec = _parse_compare_list_item(entry)
    label = spec.get("label")
    if label:
        return _slug_token(str(label))
    path_raw = spec.get("path")
    if path_raw:
        p = Path(str(path_raw))
        if p.suffix:
            return _slug_token(p.stem)
        return _slug_token(p.name)
    return "entry"


def _selection_name_tokens(selection: Dict[str, Any]) -> list[str]:
    entries = _compare_list_entries(selection)
    if entries:
        labels = [_entry_label_for_name(v) for v in entries[:4]]
        return [v for v in labels if v]
    if compare_enabled(selection):
        run_a = selection.get("run_a_path") or selection.get("run_a") or "run_a"
        run_b = selection.get("run_b_path") or selection.get("run_b") or "run_b"
        a_lbl = Path(str(run_a)).stem if str(run_a).endswith((".csv", ".tsv", ".txt")) else Path(str(run_a)).name
        b_lbl = Path(str(run_b)).stem if str(run_b).endswith((".csv", ".tsv", ".txt")) else Path(str(run_b)).name
        return [_slug_token(a_lbl), _slug_token(b_lbl)]
    run_single = selection.get("run_single") or "single"
    return [_slug_token(Path(str(run_single)).name)]


def _default_plot_export_filename(
    selection: Dict[str, Any],
    *,
    figure_type: str,
    mode: str,
    suffix: str,
) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tokens = _selection_name_tokens(selection)[:4]
    token_text = "_".join(tokens) if tokens else "plot"
    suffix_clean = str(suffix or "").strip().lstrip(".") or "csv"
    return f"{_slug_token(figure_type)}_{_slug_token(mode)}_{token_text}_{stamp}.{suffix_clean}"


def _default_plot_data_filename(
    selection: Dict[str, Any],
    *,
    figure_type: str,
    mode: str,
) -> str:
    return _default_plot_export_filename(selection, figure_type=figure_type, mode=mode, suffix="csv")


def _resolve_plot_data_target_path(
    selection: Dict[str, Any],
    requested_path: Any,
    *,
    figure_type: str,
    mode: str,
) -> Path:
    base_dir = selection.get("base")
    if base_dir is None:
        base_dir = Path.cwd()
    base_dir = Path(base_dir)
    default_dir = base_dir / "plot_data"
    requested = str(requested_path or "").strip()
    if not requested:
        return default_dir / _default_plot_data_filename(selection, figure_type=figure_type, mode=mode)

    p = Path(requested).expanduser()
    if not p.suffix:
        p = p.with_suffix(".csv")
    if p.is_absolute():
        return p
    if p.parent == Path("."):
        return default_dir / p.name
    return p.resolve()


def _resolve_plot_figure_target_path(
    selection: Dict[str, Any],
    requested_path: Any,
    *,
    figure_type: str,
    mode: str,
    image_format: str,
) -> Path:
    base_dir = selection.get("base")
    if base_dir is None:
        base_dir = Path.cwd()
    base_dir = Path(base_dir)
    default_dir = base_dir / "plot_data"
    fmt = str(image_format or "png").strip().lower()
    if fmt not in {"png", "svg"}:
        fmt = "png"
    requested = str(requested_path or "").strip()
    if not requested:
        return default_dir / _default_plot_export_filename(
            selection,
            figure_type=figure_type,
            mode=mode,
            suffix=fmt,
        )

    p = Path(requested).expanduser()
    suffix = f".{fmt}"
    if p.suffix.lower() != suffix:
        p = p.with_suffix(suffix)
    if p.is_absolute():
        return p
    if p.parent == Path("."):
        return default_dir / p.name
    return p.resolve()


def _new_plot_data_row(
    *,
    figure_type: str,
    mode: str,
    plot_name: str,
    axis_index: int,
    trace_label: str,
    series_kind: str,
    run_label: str,
    units: str,
    time_ms: Any,
    value: Any = None,
    value_low: Any = None,
    value_high: Any = None,
) -> Dict[str, Any]:
    return {
        "figure_type": str(figure_type),
        "mode": str(mode),
        "plot_name": str(plot_name),
        "axis_index": int(axis_index),
        "trace_label": str(trace_label),
        "series_kind": str(series_kind),
        "run_label": str(run_label),
        "units": str(units or ""),
        "time_ms": _safe_float(time_ms),
        "value": _safe_float(value),
        "value_low": _safe_float(value_low),
        "value_high": _safe_float(value_high),
    }


def _trace_cell_value(value: Any) -> str:
    num = _safe_float(value)
    if num is None:
        return ""
    return f"{num:.10g}"


def _encode_trace_series(values: list[Any]) -> str:
    return "|".join(_trace_cell_value(v) for v in values)


def _normalize_plot_data_format(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token in {"long", "long_rows", "tidy", "point_rows", "points"}:
        return "long_rows"
    return "trace_rows"


def _normalize_output_plot_export_type(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token in {"png", "svg"}:
        return token
    return "csv"


def _rows_to_trace_rows(rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, int, str, str, str, str], list[Dict[str, Any]]] = {}
    order: list[Tuple[str, str, str, int, str, str, str, str]] = []

    for row in rows:
        figure_type = str(row.get("figure_type") or "")
        mode = str(row.get("mode") or "")
        plot_name = str(row.get("plot_name") or "")
        axis_index = int(row.get("axis_index") or 0)
        trace_label = str(row.get("trace_label") or "")
        series_kind = str(row.get("series_kind") or "")
        run_label = str(row.get("run_label") or "")
        units = str(row.get("units") or "")
        key = (
            figure_type,
            mode,
            plot_name,
            axis_index,
            trace_label,
            series_kind,
            run_label,
            units,
        )
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)

    trace_rows: list[Dict[str, Any]] = []
    name_counts: Dict[str, int] = {}
    for key in order:
        figure_type, mode, plot_name, axis_index, trace_label, series_kind, run_label, units = key
        points = grouped.get(key, [])
        if not points:
            continue

        def _sort_key(item: Dict[str, Any]) -> tuple[float, int]:
            t = _safe_float(item.get("time_ms"))
            if t is None:
                return (float("inf"), 1)
            return (t, 0)

        points_sorted = sorted(points, key=_sort_key)
        time_vals = [pt.get("time_ms") for pt in points_sorted]
        value_vals = [pt.get("value") for pt in points_sorted]
        low_vals = [pt.get("value_low") for pt in points_sorted]
        high_vals = [pt.get("value_high") for pt in points_sorted]

        base_name = trace_label.strip() or f"{series_kind}_trace"
        count = name_counts.get(base_name, 0) + 1
        name_counts[base_name] = count
        trace_name = base_name if count == 1 else f"{base_name} ({count})"

        trace_rows.append(
            {
                "trace_name": trace_name,
                "trace_label": trace_label,
                "series_kind": series_kind,
                "figure_type": figure_type,
                "mode": mode,
                "plot_name": plot_name,
                "axis_index": axis_index,
                "run_label": run_label,
                "units": units,
                "n_points": len(points_sorted),
                "time_ms": _encode_trace_series(time_vals),
                "value": _encode_trace_series(value_vals),
                "value_low": _encode_trace_series(low_vals),
                "value_high": _encode_trace_series(high_vals),
            }
        )
    return trace_rows


def _rows_from_line(
    line,
    *,
    figure_type: str,
    mode: str,
    plot_name: str,
    axis_index: int,
    run_label: str,
    units: str,
    fallback_label: str,
) -> list[Dict[str, Any]]:
    x_raw = np.asarray(line.get_xdata(), dtype=float)
    y_raw = np.asarray(line.get_ydata(), dtype=float)
    if x_raw.size == 0 or y_raw.size == 0:
        return []
    n = min(x_raw.size, y_raw.size)
    x = x_raw[:n]
    y = y_raw[:n]
    if x.size == 0:
        return []
    label = _normalize_label(line.get_label(), fallback_label)
    if x.size == 2 and np.isfinite(x[0]) and np.isfinite(x[1]) and np.isclose(x[0], x[1]):
        return [
            _new_plot_data_row(
                figure_type=figure_type,
                mode=mode,
                plot_name=plot_name,
                axis_index=axis_index,
                trace_label=label,
                series_kind="vline",
                run_label=run_label,
                units=units,
                time_ms=x[0],
                value_low=min(y[0], y[1]),
                value_high=max(y[0], y[1]),
            )
        ]
    rows: list[Dict[str, Any]] = []
    for xi, yi in zip(x, y):
        rows.append(
            _new_plot_data_row(
                figure_type=figure_type,
                mode=mode,
                plot_name=plot_name,
                axis_index=axis_index,
                trace_label=label,
                series_kind="line",
                run_label=run_label,
                units=units,
                time_ms=xi,
                value=yi,
            )
        )
    return rows


def _rows_from_path_collection(
    collection,
    *,
    figure_type: str,
    mode: str,
    plot_name: str,
    axis_index: int,
    run_label: str,
    units: str,
    fallback_label: str,
) -> list[Dict[str, Any]]:
    try:
        offsets = np.asarray(collection.get_offsets(), dtype=float)
    except Exception:
        return []
    if offsets.size == 0:
        return []
    offsets = offsets.reshape((-1, 2))
    label = _normalize_label(collection.get_label(), fallback_label)
    rows: list[Dict[str, Any]] = []
    for xi, yi in offsets:
        rows.append(
            _new_plot_data_row(
                figure_type=figure_type,
                mode=mode,
                plot_name=plot_name,
                axis_index=axis_index,
                trace_label=label,
                series_kind="scatter",
                run_label=run_label,
                units=units,
                time_ms=xi,
                value=yi,
            )
        )
    return rows


def _rows_from_line_collection(
    collection,
    *,
    figure_type: str,
    mode: str,
    plot_name: str,
    axis_index: int,
    run_label: str,
    units: str,
    fallback_label: str,
) -> list[Dict[str, Any]]:
    try:
        segments = collection.get_segments()
    except Exception:
        return []
    if not segments:
        return []
    label = _normalize_label(collection.get_label(), fallback_label)
    rows: list[Dict[str, Any]] = []
    for seg in segments:
        pts = np.asarray(seg, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        x0, y0 = pts[0]
        x1, y1 = pts[-1]
        if np.isclose(x0, x1):
            rows.append(
                _new_plot_data_row(
                    figure_type=figure_type,
                    mode=mode,
                    plot_name=plot_name,
                    axis_index=axis_index,
                    trace_label=label,
                    series_kind="vline_segment",
                    run_label=run_label,
                    units=units,
                    time_ms=x0,
                    value_low=min(y0, y1),
                    value_high=max(y0, y1),
                )
            )
            continue
        rows.append(
            _new_plot_data_row(
                figure_type=figure_type,
                mode=mode,
                plot_name=plot_name,
                axis_index=axis_index,
                trace_label=label,
                series_kind="segment_start",
                run_label=run_label,
                units=units,
                time_ms=x0,
                value=y0,
            )
        )
        rows.append(
            _new_plot_data_row(
                figure_type=figure_type,
                mode=mode,
                plot_name=plot_name,
                axis_index=axis_index,
                trace_label=label,
                series_kind="segment_end",
                run_label=run_label,
                units=units,
                time_ms=x1,
                value=y1,
            )
        )
    return rows


def _rows_from_poly_collection(
    collection,
    *,
    figure_type: str,
    mode: str,
    plot_name: str,
    axis_index: int,
    run_label: str,
    units: str,
    fallback_label: str,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    label = _normalize_label(collection.get_label(), fallback_label)
    try:
        paths = collection.get_paths()
    except Exception:
        return rows
    for path in paths:
        verts = np.asarray(path.vertices, dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 3:
            continue
        if np.allclose(verts[0], verts[-1]):
            verts = verts[:-1]
        if verts.shape[0] < 2:
            continue
        x_vals = np.round(verts[:, 0], 9)
        uniq_x = np.unique(x_vals)
        for x_val in uniq_x:
            ys = verts[x_vals == x_val, 1]
            if ys.size == 0:
                continue
            rows.append(
                _new_plot_data_row(
                    figure_type=figure_type,
                    mode=mode,
                    plot_name=plot_name,
                    axis_index=axis_index,
                    trace_label=label,
                    series_kind="shade_band",
                    run_label=run_label,
                    units=units,
                    time_ms=float(x_val),
                    value_low=float(np.min(ys)),
                    value_high=float(np.max(ys)),
                )
            )
    return rows


def _rows_from_figures(
    figures: list[tuple[Any, str]],
    *,
    figure_type: str,
    mode: str,
    run_label: str,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for fig_obj, plot_name in figures:
        fig = analysis.resolve_figure(fig_obj)
        axes = list(fig.axes or [])
        for ax_idx, ax in enumerate(axes):
            units = str(ax.get_ylabel() or "")
            for line_idx, line in enumerate(ax.get_lines()):
                rows.extend(
                    _rows_from_line(
                        line,
                        figure_type=figure_type,
                        mode=mode,
                        plot_name=plot_name,
                        axis_index=ax_idx,
                        run_label=run_label,
                        units=units,
                        fallback_label=f"line_{ax_idx}_{line_idx}",
                    )
                )
            for coll_idx, coll in enumerate(ax.collections):
                fallback = f"collection_{ax_idx}_{coll_idx}"
                if isinstance(coll, PathCollection):
                    rows.extend(
                        _rows_from_path_collection(
                            coll,
                            figure_type=figure_type,
                            mode=mode,
                            plot_name=plot_name,
                            axis_index=ax_idx,
                            run_label=run_label,
                            units=units,
                            fallback_label=fallback,
                        )
                    )
                elif isinstance(coll, LineCollection):
                    rows.extend(
                        _rows_from_line_collection(
                            coll,
                            figure_type=figure_type,
                            mode=mode,
                            plot_name=plot_name,
                            axis_index=ax_idx,
                            run_label=run_label,
                            units=units,
                            fallback_label=fallback,
                        )
                    )
                elif isinstance(coll, PolyCollection):
                    rows.extend(
                        _rows_from_poly_collection(
                            coll,
                            figure_type=figure_type,
                            mode=mode,
                            plot_name=plot_name,
                            axis_index=ax_idx,
                            run_label=run_label,
                            units=units,
                            fallback_label=fallback,
                        )
                    )
    return rows


def _save_plot_data_rows(
    rows: list[Dict[str, Any]],
    *,
    selection: Dict[str, Any],
    requested_path: Any,
    figure_type: str,
    mode: str,
    export_format: Any = "trace_rows",
) -> Optional[Path]:
    if not rows:
        print("Plot data CSV not saved: no plotted data captured in the current figure(s).")
        return None
    fmt = _normalize_plot_data_format(export_format)
    out_path = _resolve_plot_data_target_path(
        selection,
        requested_path,
        figure_type=figure_type,
        mode=mode,
    )
    if out_path.exists():
        print(f"Plot data CSV not saved: file already exists: {out_path}")
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "long_rows":
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_PLOT_DATA_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in _PLOT_DATA_FIELDS})
        print(f"Saved plot data CSV: {out_path} ({len(rows)} rows, format=long_rows)")
        return out_path

    trace_rows = _rows_to_trace_rows(rows)
    if not trace_rows:
        print("Plot data CSV not saved: no trace rows were generated from plotted data.")
        return None
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_PLOT_DATA_TRACE_FIELDS)
        writer.writeheader()
        for row in trace_rows:
            writer.writerow({key: row.get(key) for key in _PLOT_DATA_TRACE_FIELDS})
    print(f"Saved plot data CSV: {out_path} ({len(trace_rows)} traces, {len(rows)} points, format=trace_rows)")
    return out_path


def _stim_window(sim_cfg: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    stim_start = sim_cfg.get("stim_start_ms")
    stim_stop = sim_cfg.get("stim_stop_ms")
    stim_dur = sim_cfg.get("stim_duration_ms")
    if stim_start is None:
        delay = sim_cfg.get("delay")
        if delay is not None:
            stim_start = float(delay) + 100.0
    if stim_stop is None and stim_start is not None and stim_dur is not None:
        stim_stop = float(stim_start) + float(stim_dur)
    return (
        float(stim_start) if stim_start is not None else None,
        float(stim_stop) if stim_stop is not None else None,
    )


def _stim_window_for_opts(sim_cfg: Dict[str, Any], opts: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    stim_start, stim_stop = _stim_window(sim_cfg)
    override_start = opts.get("output_stim_start_ms")
    override_stop = opts.get("output_stim_stop_ms")
    if override_start is not None:
        try:
            stim_start = float(override_start)
        except Exception:
            pass
    if override_stop is not None:
        try:
            stim_stop = float(override_stop)
        except Exception:
            pass
    return stim_start, stim_stop


def _coerce_plot_window(window_val: Any) -> Tuple[Optional[float], Optional[float]]:
    if isinstance(window_val, (list, tuple)) and len(window_val) >= 2:
        return _safe_float(window_val[0]), _safe_float(window_val[1])
    return None, None


def _resolve_plot_window_for_opts(
    sim_cfg: Dict[str, Any],
    opts: Dict[str, Any],
    *,
    window_key: str,
    context: str,
) -> Tuple[Tuple[Optional[float], Optional[float]], Optional[str]]:
    manual_window = _coerce_plot_window(opts.get(window_key))
    if not bool(opts.get("auto_plot_window_from_stim", False)):
        return manual_window, None

    adjust_ms = _safe_float(opts.get("plot_window_adjustment_ms"))
    if adjust_ms is None:
        adjust_ms = 100.0
    if adjust_ms < 0:
        adjust_ms = abs(adjust_ms)

    stim_start, stim_stop = _stim_window_for_opts(sim_cfg or {}, opts)
    if stim_start is None or stim_stop is None:
        return (
            manual_window,
            f"{context}: auto plot window enabled but stim start/stop unavailable; using manual {window_key}.",
        )
    if stim_stop < stim_start:
        stim_start, stim_stop = stim_stop, stim_start
    return (float(stim_start) - float(adjust_ms), float(stim_stop) + float(adjust_ms)), None


def _apply_output_window_origin_zero(
    fig_obj: Any,
    *,
    enabled: bool,
    plot_window: Optional[Tuple[Optional[float], Optional[float]]],
) -> None:
    if not enabled or fig_obj is None:
        return
    fig = analysis.resolve_figure(fig_obj)
    if fig is None:
        return

    offset: Optional[float] = None
    if isinstance(plot_window, (list, tuple)) and len(plot_window) >= 1:
        offset = _safe_float(plot_window[0])
    if offset is None:
        for ax in fig.axes:
            try:
                xmin, _ = ax.get_xlim()
            except Exception:
                continue
            if np.isfinite(xmin):
                offset = float(xmin)
                break
    if offset is None:
        return

    def _fmt(x: float, _pos: int) -> str:
        val = float(x) - float(offset)
        if abs(val) < 1e-9:
            val = 0.0
        return f"{val:g}"

    formatter = FuncFormatter(_fmt)
    for ax in fig.axes:
        try:
            ax.xaxis.set_major_formatter(formatter)
            if str(ax.get_xlabel()).strip() == "Time (ms)":
                ax.set_xlabel("Time (ms, window-relative)")
        except Exception:
            continue


def _apply_output_y_window(
    fig_obj: Any,
    *,
    y_window: Optional[Tuple[Optional[float], Optional[float]]],
) -> None:
    if fig_obj is None:
        return
    if not isinstance(y_window, (list, tuple)) or len(y_window) < 2:
        return
    y0 = _safe_float(y_window[0])
    y1 = _safe_float(y_window[1])
    if y0 is None and y1 is None:
        return
    fig = analysis.resolve_figure(fig_obj)
    if fig is None:
        return
    for ax in fig.axes:
        # Only clamp firing-rate axes (skip raster/ISI/other panels).
        y_label = str(ax.get_ylabel() or "").lower()
        if "rate" not in y_label:
            continue
        try:
            ax.set_ylim(y0, y1)
        except Exception:
            continue


def _sim_cfg_with_output_stim_overrides(sim_cfg: Dict[str, Any], opts: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(sim_cfg or {})
    stim_start, stim_stop = _stim_window_for_opts(cfg, opts)
    if stim_start is not None:
        cfg["stim_start_ms"] = float(stim_start)
    if stim_stop is not None:
        cfg["stim_stop_ms"] = float(stim_stop)
    return cfg


def _sim_cfg_with_shifted_stim(sim_cfg: Dict[str, Any], shift_ms: Optional[float]) -> Dict[str, Any]:
    cfg = dict(sim_cfg or {})
    if shift_ms is None:
        return cfg
    try:
        shift_val = float(shift_ms)
    except Exception:
        return cfg
    if shift_val == 0.0:
        return cfg
    stim_start, stim_stop = _stim_window(cfg)
    if stim_start is not None:
        cfg["stim_start_ms"] = float(stim_start) + shift_val
    if stim_stop is not None:
        cfg["stim_stop_ms"] = float(stim_stop) + shift_val
    return cfg


def _plot_metric_points(
    ax,
    metrics: Dict[str, Any],
    *,
    color: Optional[str] = None,
    label_prefix: Optional[str] = None,
    show_labels: bool = False,
    size: float = 36.0,
) -> None:
    if ax is None or not metrics:
        return
    entries = [
        ("peak_time_ms", "peak_value", "o", "Peak"),
        ("tpeak10_time_ms", "tpeak10_value", "s", "Tpeak10"),
        ("drop_time_ms", "drop_value", "v", "+100ms"),
        ("t50_time_ms", "t50_value", "D", "T50"),
        ("rebound_time_ms", "rebound_value", "^", "+300ms"),
    ]
    any_label = False
    for t_key, y_key, marker, label in entries:
        t_val = metrics.get(t_key)
        y_val = metrics.get(y_key)
        if t_val is None or y_val is None:
            continue
        lab = None
        if show_labels:
            lab = f"{label_prefix} {label}" if label_prefix else label
            lab = _legend_safe_label(lab)
            any_label = True
        ax.scatter(
            [t_val],
            [y_val],
            s=size,
            marker=marker,
            color=color,
            edgecolor="k",
            linewidth=0.6,
            zorder=5,
            label=lab,
        )
    if show_labels and any_label:
        ax.legend()


def _apply_output_norm(curve: Optional[Dict[str, Any]], output_norm: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not curve or not output_norm:
        return curve
    rate = np.asarray(curve.get("rate_hz", []) or [], dtype=float)
    baseline_mean = output_norm.get("baseline_mean")
    norm_scale = output_norm.get("norm_scale")
    rate_bs = rate
    if baseline_mean is not None:
        rate_bs = rate - float(baseline_mean)
        rate = rate_bs
    if norm_scale not in (None, 0):
        rate = rate / float(norm_scale)
    updated = dict(curve)
    updated["rate_hz"] = rate.tolist()
    updated["units"] = "normalized"
    updated["normalized"] = True
    if baseline_mean is not None:
        updated["baseline_mean"] = float(baseline_mean)
        updated["baseline_ms"] = output_norm.get("baseline_ms")
        updated["baseline_mode"] = output_norm.get("baseline_mode")
        updated["baseline_time_ms"] = output_norm.get("baseline_time_ms")
        updated["baseline_subtracted"] = True
        updated["rate_hz_baseline_sub"] = rate_bs.tolist()
    updated["norm_scale"] = norm_scale
    if output_norm.get("norm_mode") is not None:
        updated["norm_mode"] = output_norm.get("norm_mode")
    if output_norm.get("norm_window") is not None:
        updated["norm_window"] = output_norm.get("norm_window")
    return updated


def _parse_plot_scale(scale: Any) -> float:
    if scale is None:
        return 1.0
    try:
        return float(scale)
    except Exception:
        return 1.0


def _legend_safe_label(label: Any) -> Optional[str]:
    if label is None:
        return None
    txt = str(label)
    if not txt:
        return None
    # Matplotlib ignores labels that start with "_"
    if txt.startswith("_"):
        return f" {txt}"
    return txt


def _curve_smooth_ms(opts: Dict[str, Any]) -> Optional[float]:
    smooth = opts.get("win_size")
    if smooth is None:
        smooth = opts.get("output_smooth_ms")
    return smooth


def _output_curve_plot_mode(opts: Dict[str, Any]) -> str:
    mode = str(opts.get("output_curve_plot_mode", "rate") or "rate").strip().lower()
    if mode in ("isi", "isi_only"):
        return "isi"
    if mode in ("rate_isi", "rate+isi", "both", "stacked"):
        return "rate_isi"
    return "rate"


def _unique_compare_colors(n: int) -> list:
    if n <= 0:
        return []
    cmap = plt.cm.tab20 if n <= 20 else plt.cm.hsv
    vals = np.linspace(0.0, 1.0, n, endpoint=False)
    colors = [cmap(v) for v in vals]
    rng = np.random.default_rng()
    rng.shuffle(colors)
    return colors


def _scale_curve_for_plot(curve: Optional[Dict[str, Any]], scale: Any) -> Optional[Dict[str, Any]]:
    if not curve:
        return curve
    scale_val = _parse_plot_scale(scale)
    if scale_val == 1.0:
        return curve
    updated = dict(curve)
    rate = np.asarray(curve.get("rate_hz", []) or [], dtype=float) * scale_val
    updated["rate_hz"] = rate.tolist()
    if curve.get("rate_hz_baseline_sub") is not None:
        rate_bs = np.asarray(curve.get("rate_hz_baseline_sub") or [], dtype=float) * scale_val
        updated["rate_hz_baseline_sub"] = rate_bs.tolist()
    return updated


def _scale_metrics_for_plot(metrics: Optional[Dict[str, Any]], scale: Any) -> Optional[Dict[str, Any]]:
    if not metrics:
        return metrics
    scale_val = _parse_plot_scale(scale)
    if scale_val == 1.0:
        return metrics
    scaled = dict(metrics)
    for key in (
        "peak_value",
        "peak_rate_hz",
        "tpeak10_value",
        "drop_value",
        "t50_value",
        "rebound_value",
        "peak_value_raw",
        "peak_rate_hz_raw",
    ):
        val = scaled.get(key)
        if val is not None:
            try:
                scaled[key] = float(val) * scale_val
            except Exception:
                pass
    return scaled


def _plot_metric_window_markers(
    ax,
    metrics: Optional[Dict[str, Any]],
    *,
    color: Optional[str],
    alpha: float = 0.15,
    linewidth: float = 1.0,
) -> None:
    if not metrics or ax is None:
        return
    def _vline(x, ls="--"):
        if x is None:
            return
        ax.axvline(float(x), color=color or "0.4", linestyle=ls, linewidth=linewidth)

    _vline(metrics.get("baseline_time_ms"))
    _vline(metrics.get("drop_center_ms"))
    _vline(metrics.get("rebound_center_ms"))

    for key_start, key_stop in (
        ("baseline_window_start_ms", "baseline_window_stop_ms"),
        ("drop_window_start_ms", "drop_window_stop_ms"),
        ("rebound_window_start_ms", "rebound_window_stop_ms"),
    ):
        start = metrics.get(key_start)
        stop = metrics.get(key_stop)
        if start is None or stop is None:
            continue
        ax.axvspan(float(start), float(stop), color=color or "0.4", alpha=alpha, linewidth=0)


def _metric_window_kwargs(opts: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "alpha": float(opts.get("output_metric_window_alpha", 0.15)),
        "linewidth": float(opts.get("output_metric_linewidth", 1.0)),
    }


def _compute_output_metrics(
    curve: Optional[Dict[str, Any]],
    sim_cfg: Dict[str, Any],
    opts: Dict[str, Any],
    **overrides: Any,
) -> Dict[str, Any]:
    def _pick(name: str, opt_key: str, default: Any) -> Any:
        if name in overrides:
            return overrides[name]
        return opts.get(opt_key, default)

    return analysis.compute_output_metrics(
        curve or {},
        sim_cfg or {},
        peak_window_ms=_pick("peak_window_ms", "output_peak_window_ms", 100.0),
        drop_window_ms=_pick("drop_window_ms", "output_drop_window_ms", 100.0),
        rebound_window_ms=_pick("rebound_window_ms", "output_rebound_window_ms", 300.0),
        auc_window=_pick("auc_window", "output_auc_window", "stim"),
        t50_mode=_pick("t50_mode", "output_t50_mode", "absolute"),
        pdp_mode=_pick("pdp_mode", "output_metric_mode", "point"),
        pdp_window_ms=_pick("pdp_window_ms", "output_metric_window_ms", 0.0),
        baseline_ms=_pick("baseline_ms", "output_metric_window_ms", 100.0),
        baseline_mode=_pick("baseline_mode", "output_metric_mode", "point"),
        baseline_center_ms=_pick("baseline_center_ms", "output_baseline_center_ms", None),
        stim_start_ms=_pick("stim_start_ms", "output_stim_start_ms", None),
        stim_stop_ms=_pick("stim_stop_ms", "output_stim_stop_ms", None),
    )


def _output_metrics_std_mode(opts: Dict[str, Any]) -> str:
    mode = str(opts.get("output_metrics_std_mode", "std") or "std").strip().lower()
    return mode if mode in ("std", "sem") else "std"


def _output_metric_overrides(opts: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "peak_window_ms": opts.get("output_peak_window_ms", 100.0),
        "drop_window_ms": opts.get("output_drop_window_ms", 100.0),
        "rebound_window_ms": opts.get("output_rebound_window_ms", 300.0),
        "auc_window": opts.get("output_auc_window", "stim"),
        "t50_mode": opts.get("output_t50_mode", "absolute"),
    }


def _extract_spike_trials(results: Optional[Dict[str, Any]]) -> list[Any]:
    if not isinstance(results, dict):
        return []
    spikes = results.get("spikes")
    if spikes is None:
        return []
    if results.get("mode") != "multi":
        return [spikes]
    if isinstance(spikes, np.ndarray):
        if spikes.dtype == object:
            return list(spikes.tolist())
        if spikes.ndim > 1:
            return list(spikes)
        return [spikes]
    if isinstance(spikes, (list, tuple)):
        if not spikes:
            return []
        first = spikes[0]
        if isinstance(first, (int, float, np.integer, np.floating)):
            return [spikes]
        return list(spikes)
    return [spikes]


def _run_output_curve_from_results(
    results: Dict[str, Any],
    opts: Dict[str, Any],
    *,
    shift_ms: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    curve = analysis.compute_output_curve_from_results(
        results,
        bin_ms=opts.get("output_bin_ms"),
        smooth_ms=_curve_smooth_ms(opts),
        smooth_mode=opts.get("output_smooth_mode", "causal"),
    )
    if not curve:
        return None
    sim_cfg_norm = _sim_cfg_with_output_stim_overrides(results.get("sim_cfg", {}) or {}, opts)
    curve = analysis.normalize_output_curve(
        curve,
        sim_cfg_norm,
        mode=opts.get("output_curve_mode", "raw"),
        norm_mode=opts.get("output_norm_mode", "avg"),
        baseline_ms=opts.get("output_metric_window_ms", 100.0),
        baseline_mode=opts.get("output_metric_mode", "point"),
        baseline_center_ms=opts.get("output_baseline_center_ms"),
        norm_window=opts.get("output_norm_window", "stim"),
    )
    if shift_ms is not None:
        t_ms = np.asarray(curve.get("t_ms", []) or [], dtype=float)
        if t_ms.size:
            curve = dict(curve)
            curve["t_ms"] = (t_ms + float(shift_ms)).tolist()
    return curve


def _run_output_isi_curve_from_results(
    results: Dict[str, Any],
    opts: Dict[str, Any],
    *,
    shift_ms: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    curve = analysis.compute_output_isi_curve_from_results(
        results,
        bin_ms=opts.get("output_bin_ms"),
        smooth_ms=_curve_smooth_ms(opts),
        smooth_mode=opts.get("output_smooth_mode", "causal"),
    )
    if not curve:
        return None
    if shift_ms is not None:
        t_ms = np.asarray(curve.get("t_ms", []) or [], dtype=float)
        if t_ms.size:
            curve = dict(curve)
            curve["t_ms"] = (t_ms + float(shift_ms)).tolist()
    return curve


def _compute_output_trial_metrics(
    results: Optional[Dict[str, Any]],
    sim_cfg: Dict[str, Any],
    opts: Dict[str, Any],
    *,
    shift_ms: Optional[float] = None,
    **metric_overrides: Any,
) -> list[Dict[str, Any]]:
    if not isinstance(results, dict):
        return []
    trial_spikes = _extract_spike_trials(results)
    if not trial_spikes:
        return []

    trial_metrics: list[Dict[str, Any]] = []
    for spikes_trial in trial_spikes:
        sim_cfg_norm = _sim_cfg_with_output_stim_overrides(sim_cfg or {}, opts)
        sim_cfg_metrics = _sim_cfg_with_shifted_stim(sim_cfg_norm, shift_ms)
        trial_results: Dict[str, Any] = {
            "mode": "single",
            "spikes": spikes_trial,
            "sim_cfg": sim_cfg_norm,
        }
        traces = results.get("traces")
        if traces is not None:
            trial_results["traces"] = traces

        trial_curve = analysis.compute_output_curve_from_results(
            trial_results,
            bin_ms=opts.get("output_bin_ms"),
            smooth_ms=_curve_smooth_ms(opts),
            smooth_mode=opts.get("output_smooth_mode", "causal"),
        )
        if not trial_curve:
            continue
        trial_curve = analysis.normalize_output_curve(
            trial_curve,
            sim_cfg_norm,
            mode=opts.get("output_curve_mode", "raw"),
            norm_mode=opts.get("output_norm_mode", "avg"),
            baseline_ms=opts.get("output_metric_window_ms", 100.0),
            baseline_mode=opts.get("output_metric_mode", "point"),
            baseline_center_ms=opts.get("output_baseline_center_ms"),
            norm_window=opts.get("output_norm_window", "stim"),
        )
        if shift_ms is not None:
            t_ms = np.asarray(trial_curve.get("t_ms", []) or [], dtype=float)
            if t_ms.size:
                trial_curve = dict(trial_curve)
                trial_curve["t_ms"] = (t_ms + float(shift_ms)).tolist()

        trial_metrics.append(_compute_output_metrics(trial_curve, sim_cfg_metrics, opts, **metric_overrides))
    return trial_metrics


def _coerce_metric_numeric(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _extract_output_color_from_results(results: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(results, dict):
        return None
    sim_cfg = results.get("sim_cfg") or {}
    if isinstance(sim_cfg, dict):
        for key in ("color", "cell_color", "plot_color"):
            val = sim_cfg.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def _coerce_plot_color(value: Any, fallback: Any) -> Any:
    if isinstance(value, str) and value.strip():
        try:
            mcolors.to_rgba(value.strip())
            return value.strip()
        except Exception:
            pass
    return fallback


def _attach_output_metric_spread(
    metrics: Dict[str, Any],
    *,
    results: Optional[Dict[str, Any]],
    sim_cfg: Dict[str, Any],
    opts: Dict[str, Any],
    shift_ms: Optional[float] = None,
    **metric_overrides: Any,
) -> Dict[str, Any]:
    out = dict(metrics)
    spread_mode = _output_metrics_std_mode(opts)
    out["output_metrics_std_mode"] = spread_mode
    if not isinstance(results, dict):
        out["output_metrics_n_trials"] = None
        return out

    trial_metrics = _compute_output_trial_metrics(
        results,
        sim_cfg,
        opts,
        shift_ms=shift_ms,
        **metric_overrides,
    )
    out["output_metrics_n_trials"] = len(trial_metrics)
    if not trial_metrics:
        return out

    for key in _OUTPUT_METRIC_VALUE_ORDER:
        if key == "output_metrics_n_trials":
            continue
        vals: list[float] = []
        for trial in trial_metrics:
            val = _coerce_metric_numeric(trial.get(key))
            if val is not None:
                vals.append(val)
        if not vals:
            continue
        spread = float(np.std(np.asarray(vals, dtype=float)))
        if spread_mode == "sem":
            spread = spread / float(np.sqrt(len(vals)))
        out[f"{key}_spread"] = spread

    return out


def _compute_output_metrics_with_spread(
    curve: Optional[Dict[str, Any]],
    sim_cfg: Dict[str, Any],
    opts: Dict[str, Any],
    *,
    results: Optional[Dict[str, Any]] = None,
    shift_ms: Optional[float] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    sim_cfg_metrics = _sim_cfg_with_shifted_stim(sim_cfg or {}, shift_ms)
    metrics = _compute_output_metrics(curve, sim_cfg_metrics, opts, **overrides)
    return _attach_output_metric_spread(
        metrics,
        results=results,
        sim_cfg=sim_cfg,
        opts=opts,
        shift_ms=shift_ms,
        **overrides,
    )


def _smooth_curve_if_requested(
    curve: Optional[Dict[str, Any]],
    *,
    bin_ms: Optional[float],
    smooth_ms: Optional[float],
    smooth_mode: str,
) -> Optional[Dict[str, Any]]:
    if not curve:
        return curve
    t_ms = np.asarray(curve.get("t_ms", []) or [], dtype=float)
    rate = np.asarray(curve.get("rate_hz", []) or [], dtype=float)
    if t_ms.size < 2 or rate.size < 2:
        return curve
    use_bin_ms = bin_ms
    if use_bin_ms is None:
        diffs = np.diff(t_ms)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            return curve
        use_bin_ms = float(np.median(diffs))
    centers, y = analysis._smooth_rate_curve(
        t_ms,
        rate,
        float(use_bin_ms),
        smooth_ms,
        mode=str(smooth_mode or "center").lower(),
    )
    updated = dict(curve)
    updated["t_ms"] = centers.tolist()
    updated["rate_hz"] = y.tolist()
    return updated


def _default_sim_cfg_for_curve(
    curve: Dict[str, Any],
    *,
    fallback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sim_cfg: Dict[str, Any] = dict(fallback or {})
    t_ms = np.asarray(curve.get("t_ms", []) or [], dtype=float)
    if t_ms.size:
        sim_cfg.setdefault("tstart", float(np.min(t_ms)))
        sim_cfg.setdefault("tstop", float(np.max(t_ms)))
    sim_cfg.setdefault("stim_start_ms", 0.0)
    return sim_cfg


def _load_curve_from_path(
    path: Path,
    opts: Dict[str, Any],
    *,
    fallback_sim_cfg: Optional[Dict[str, Any]] = None,
    shift_ms: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    curve = analysis.load_scatter_curve_optional(
        enabled=True,
        path=str(path),
        time_unit=opts.get("output_scatter_time_unit", "s"),
        shift_ms=shift_ms if shift_ms is not None else opts.get("output_scatter_shift_ms"),
        quiet=True,
    )
    if not curve:
        return None
    curve = _smooth_curve_if_requested(
        curve,
        bin_ms=opts.get("output_bin_ms"),
        smooth_ms=_curve_smooth_ms(opts),
        smooth_mode=opts.get("output_smooth_mode", "causal"),
    )
    if opts.get("output_curve_mode", "raw") == "normalized":
        sim_cfg = _default_sim_cfg_for_curve(curve, fallback=fallback_sim_cfg)
        if opts.get("output_stim_start_ms") is not None:
            sim_cfg["stim_start_ms"] = float(opts.get("output_stim_start_ms"))
        if opts.get("output_stim_stop_ms") is not None:
            sim_cfg["stim_stop_ms"] = float(opts.get("output_stim_stop_ms"))
        curve = analysis.normalize_output_curve(
            curve,
            sim_cfg,
            mode="normalized",
            norm_mode=opts.get("output_norm_mode", "avg"),
            baseline_ms=opts.get("output_metric_window_ms", 100.0),
            baseline_mode=opts.get("output_metric_mode", "point"),
            baseline_center_ms=opts.get("output_baseline_center_ms"),
            norm_window=opts.get("output_norm_window", "stim"),
        )
    return curve


def _curve_from_xy(t_ms: np.ndarray, rate: np.ndarray, *, units: str = "Hz") -> Dict[str, Any]:
    return {
        "t_ms": np.asarray(t_ms, dtype=float).tolist(),
        "rate_hz": np.asarray(rate, dtype=float).tolist(),
        "units": units,
    }


def _curve_has_series(curve: Optional[Dict[str, Any]], y_key: str) -> bool:
    if not curve:
        return False
    t_ms = np.asarray(curve.get("t_ms", []) or [], dtype=float)
    y = np.asarray(curve.get(y_key, []) or [], dtype=float)
    if t_ms.size == 0 or y.size == 0:
        return False
    return bool(np.isfinite(y).any())


def _plot_output_curve_mode_figure(
    *,
    mode: str,
    rate_curves: list[Optional[Dict[str, Any]]],
    isi_curves: list[Optional[Dict[str, Any]]],
    labels: list[str],
    colors: Optional[list[Optional[str]]] = None,
    linestyles: Optional[list[Optional[str]]] = None,
    plot_window: Optional[Tuple[Optional[float], Optional[float]]] = None,
    stim_start: Optional[float] = None,
    stim_stop: Optional[float] = None,
    title: str = "Output curve",
    line_width: float = 2.0,
    stim_linewidth: float = 1.0,
    figsize: Optional[tuple[float, float]] = None,
) -> tuple[Any, Dict[str, Any]]:
    mode_norm = mode if mode in ("rate", "isi", "rate_isi") else "rate"
    show_rate = mode_norm in ("rate", "rate_isi")
    show_isi = mode_norm in ("isi", "rate_isi")
    n_series = max(len(labels), len(rate_curves), len(isi_curves))

    if show_rate and show_isi:
        if figsize is None:
            figsize = (6.0, 6.0)
        fig, axes_arr = plt.subplots(2, 1, figsize=figsize, sharex=True)
        axes = np.atleast_1d(axes_arr)
        ax_rate = axes[0]
        ax_isi = axes[1]
    else:
        if figsize is None:
            figsize = (6.0, 4.0)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax_rate = ax if show_rate else None
        ax_isi = ax if show_isi else None
        axes = np.asarray([ax])

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plotted_rate = False
    plotted_isi = False

    for idx in range(n_series):
        label = labels[idx] if idx < len(labels) else None
        plot_label = _legend_safe_label(label)
        col = None
        if colors is not None and idx < len(colors):
            col = colors[idx]
        if col is None:
            col = color_cycle[idx % len(color_cycle)] if color_cycle else None
        ls = "-"
        if linestyles is not None and idx < len(linestyles) and linestyles[idx]:
            ls = str(linestyles[idx])

        rate_curve = rate_curves[idx] if idx < len(rate_curves) else None
        if ax_rate is not None and _curve_has_series(rate_curve, "rate_hz"):
            t_rate = np.asarray(rate_curve.get("t_ms", []) or [], dtype=float)
            y_rate = np.asarray(rate_curve.get("rate_hz", []) or [], dtype=float)
            ax_rate.plot(t_rate, y_rate, lw=float(line_width), color=col, linestyle=ls, label=plot_label)
            plotted_rate = True

        isi_curve = isi_curves[idx] if idx < len(isi_curves) else None
        if ax_isi is not None and _curve_has_series(isi_curve, "isi_ms"):
            t_isi = np.asarray(isi_curve.get("t_ms", []) or [], dtype=float)
            y_isi = np.asarray(isi_curve.get("isi_ms", []) or [], dtype=float)
            ax_isi.plot(t_isi, y_isi, lw=float(line_width), color=col, linestyle=ls, label=plot_label)
            plotted_isi = True

    if ax_rate is not None:
        units = None
        for curve in rate_curves:
            if curve:
                units = curve.get("units")
                if units is not None:
                    break
        y_label = "Rate (Hz)" if str(units or "Hz") == "Hz" else "Rate (normalized)"
        ax_rate.set_ylabel(y_label)
        ax_rate.set_title(title if ax_isi is None else f"{title} - rate")
        if plotted_rate:
            handles, labels_ax = ax_rate.get_legend_handles_labels()
            if any(str(lbl).strip() and not str(lbl).startswith("_") for lbl in labels_ax):
                ax_rate.legend()
        else:
            ax_rate.text(0.5, 0.5, "No rate data", transform=ax_rate.transAxes, ha="center", va="center")

    if ax_isi is not None:
        ax_isi.set_ylabel("ISI (ms)")
        ax_isi.set_title(title if ax_rate is None else f"{title} - ISI")
        if plotted_isi:
            handles, labels_ax = ax_isi.get_legend_handles_labels()
            if any(str(lbl).strip() and not str(lbl).startswith("_") for lbl in labels_ax):
                ax_isi.legend()
        else:
            ax_isi.text(0.5, 0.5, "No ISI data", transform=ax_isi.transAxes, ha="center", va="center")

    for ax in axes:
        if plot_window is not None:
            ax.set_xlim(plot_window[0], plot_window[1])
        if stim_start is not None:
            ax.axvline(float(stim_start), color="k", linestyle="-", linewidth=float(stim_linewidth))
        if stim_stop is not None:
            ax.axvline(float(stim_stop), color="k", linestyle="-", linewidth=float(stim_linewidth))
        ax.grid(True)

    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()
    return fig, {"rate": ax_rate, "isi": ax_isi}


def _parse_figsize(value: Any) -> Optional[tuple[float, float]]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except Exception:
            return None
    return None


def _split_path_and_shift(item: str) -> Tuple[str, Optional[float], Optional[float]]:
    if "@" not in item:
        return item.strip(), None, None
    path_part, spec = item.split("@", 1)
    spec = spec.strip()
    if ";" in spec:
        spec = spec.split(";", 1)[0].strip()
    if not spec:
        return path_part.strip(), None, None
    shift_part = spec
    scale_part = None
    if "," in spec:
        shift_part, scale_part = spec.split(",", 1)
    elif ":" in spec:
        shift_part, scale_part = spec.split(":", 1)
    shift_part = shift_part.strip()
    if shift_part.lower().endswith("ms"):
        shift_part = shift_part[:-2].strip()
    try:
        shift_val = float(shift_part) if shift_part else None
    except Exception:
        shift_val = None
    scale_val = None
    if scale_part is not None:
        scale_part = scale_part.strip()
        if scale_part.lower().endswith("x"):
            scale_part = scale_part[:-1].strip()
        try:
            scale_val = float(scale_part) if scale_part else None
        except Exception:
            scale_val = None
    return path_part.strip(), shift_val, scale_val


def _parse_shift_value(text: str) -> Optional[float]:
    val = text.strip()
    if val.lower().endswith("ms"):
        val = val[:-2].strip()
    try:
        return float(val) if val else None
    except Exception:
        return None


def _parse_scale_value(text: str) -> Optional[float]:
    val = text.strip()
    if val.lower().endswith("x"):
        val = val[:-1].strip()
    try:
        return float(val) if val else None
    except Exception:
        return None


def _parse_compare_list_item_str(raw: str) -> Dict[str, Any]:
    parts = [p.strip() for p in str(raw).split(";") if p.strip()]
    base = parts[0] if parts else str(raw).strip()
    path_part, shift_val, scale_val = _split_path_and_shift(base)
    color = None
    label = None
    linestyle = None
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip().lower()
        val = val.strip()
        if not val:
            continue
        if key in ("color", "colour", "c"):
            color = val
        elif key in ("scale", "scale_x", "gain", "s"):
            scale_val = _parse_scale_value(val)
        elif key in ("shift", "shift_ms", "offset", "tshift"):
            shift_val = _parse_shift_value(val)
        elif key in ("label", "name"):
            label = val
        elif key in ("linestyle", "ls", "style"):
            linestyle = val
    return {
        "path": path_part.strip(),
        "shift_ms": shift_val,
        "scale": scale_val,
        "color": color,
        "label": label,
        "linestyle": linestyle,
    }


def _parse_compare_list_item(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        base = raw.get("path") or raw.get("spec") or raw.get("entry") or ""
        spec = _parse_compare_list_item_str(str(base)) if base else _parse_compare_list_item_str("")
        if raw.get("shift_ms") is not None or raw.get("shift") is not None:
            spec["shift_ms"] = raw.get("shift_ms", raw.get("shift"))
        if raw.get("scale") is not None or raw.get("gain") is not None:
            spec["scale"] = raw.get("scale", raw.get("gain"))
        if raw.get("color") is not None:
            spec["color"] = raw.get("color")
        label_val = raw.get("label") if raw.get("label") is not None else raw.get("name")
        if label_val is not None:
            spec["label"] = label_val
        if raw.get("linestyle") is not None:
            spec["linestyle"] = raw.get("linestyle")
        spec["path"] = str(spec.get("path") or base).strip()
        return spec
    return _parse_compare_list_item_str(str(raw))


def _compare_entry_enabled(entry: Any) -> bool:
    if isinstance(entry, dict):
        return bool(entry.get("enabled", True))
    return True


def _compare_list_paths_text(entries: list[Any]) -> str:
    parts: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            if not _compare_entry_enabled(entry):
                continue
            spec = _parse_compare_list_item(entry)
            path = spec.get("path")
            if not path:
                continue
            token = str(path)
            shift_val = spec.get("shift_ms")
            scale_val = spec.get("scale")
            if shift_val is not None or scale_val is not None:
                shift_str = "" if shift_val is None else str(shift_val)
                scale_str = "" if scale_val is None else str(scale_val)
                if scale_str:
                    token = f"{token}@{shift_str}:{scale_str}"
                else:
                    token = f"{token}@{shift_str}"
            if spec.get("color"):
                token += f";color={spec['color']}"
            if spec.get("label"):
                token += f";label={spec['label']}"
            if spec.get("linestyle"):
                token += f";linestyle={spec['linestyle']}"
            parts.append(token)
        else:
            parts.append(str(entry))
    return ",".join(parts)


def _compare_list_dir_options(g: Dict[str, Any], base_dir: Optional[Path]) -> list[tuple[str, str]]:
    raw_dirs = g.get("compare_list_dir_paths") or []
    if isinstance(raw_dirs, (str, Path)):
        raw_dirs = [raw_dirs]
    root = analysis.find_scp_root(base_dir or Path.cwd())
    options: list[tuple[str, str]] = []
    for raw in raw_dirs:
        if raw in (None, "", "none", "None"):
            continue
        dir_path = Path(str(raw)).expanduser()
        if not dir_path.is_absolute():
            dir_path = (root / dir_path).resolve()
        if not dir_path.is_dir():
            continue
        try:
            files = sorted([p for p in dir_path.iterdir() if p.is_file()])
        except Exception:
            continue
        for file_path in files:
            if file_path.suffix.lower() not in (".csv", ".tsv", ".txt"):
                continue
            label = f"{dir_path.name}/{file_path.name}"
            options.append((label, str(file_path)))
    return options


def _read_compare_preset(path_val: Any, base_dir: Optional[Path]) -> tuple[Optional[Path], Optional[Any]]:
    if path_val in (None, "", "none", "None"):
        return None, None
    preset_path = Path(str(path_val)).expanduser()
    if not preset_path.is_absolute():
        repo_root = analysis.find_scp_root(base_dir or Path.cwd())
        preset_path = (repo_root / preset_path).resolve()
    if not preset_path.exists():
        print(f"Compare preset not found: {preset_path}")
        return None, None
    try:
        payload = json.loads(preset_path.read_text())
    except Exception:
        print(f"Compare preset unreadable: {preset_path}")
        return preset_path, None
    return preset_path, payload


def _load_compare_preset(path_val: Any, base_dir: Optional[Path]) -> list[Dict[str, Any]]:
    preset_path, payload = _read_compare_preset(path_val, base_dir)
    if payload is None:
        return []
    entries = payload.get("entries") if isinstance(payload, dict) else payload if isinstance(payload, list) else []
    if not isinstance(entries, list):
        return []
    out: list[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("enabled", True) is False:
            continue
        path_raw = entry.get("path") or entry.get("run") or entry.get("file")
        if not path_raw:
            continue
        path = _coerce_run_path(path_raw, base_dir or (preset_path.parent if preset_path else Path.cwd()))
        if path is None:
            continue
        if not path.exists():
            print(f"Skipping missing preset path: {path}")
            continue
        out.append({
            "path": path,
            "shift_ms": entry.get("shift_ms") if entry.get("shift_ms") is not None else entry.get("shift"),
            "scale": entry.get("scale"),
            "color": entry.get("color"),
            "linestyle": entry.get("linestyle") or entry.get("ls"),
            "label": entry.get("label"),
        })
    return out


def _load_compare_preset_defaults(path_val: Any, base_dir: Optional[Path]) -> Dict[str, Any]:
    _, payload = _read_compare_preset(path_val, base_dir)
    if not isinstance(payload, dict):
        return {}
    defaults = payload.get("defaults")
    return defaults if isinstance(defaults, dict) else {}


def _is_missing_default(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, tuple)):
        if not value:
            return True
        return all(v is None or (isinstance(v, str) and not v.strip()) for v in value)
    return False


def _merge_preset_defaults(opts: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    if not defaults:
        return opts
    merged = dict(opts)
    for key, val in defaults.items():
        if val is None:
            continue
        if _is_missing_default(merged.get(key)):
            merged[key] = val
    return merged


def _parse_compare_list_paths(text: str) -> list[str]:
    if not text:
        return []

    entries: list[str] = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if not parts:
            continue
        for part in parts:
            entries.append(part)
    return entries


def _compare_list_entries(selection: Dict[str, Any]) -> list[Any]:
    entries: list[Any] = []
    entries.extend(selection.get("compare_list") or [])
    entries.extend(selection.get("compare_list_paths") or [])
    entries = [entry for entry in entries if _compare_entry_enabled(entry)]
    # de-dupe while preserving order (by path + shift + scale + extras)
    seen = set()
    out: list[Any] = []
    for item in entries:
        spec = _parse_compare_list_item(item)
        key = (
            spec.get("path"),
            spec.get("shift_ms"),
            spec.get("scale"),
            spec.get("color"),
            spec.get("label"),
            spec.get("linestyle"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _compare_list_run_paths(selection: Dict[str, Any]) -> list[Path]:
    base_dir = selection.get("base")
    if base_dir is None:
        return []
    entries = _compare_list_entries(selection)
    if not entries:
        return []
    seen: set[Path] = set()
    out: list[Path] = []
    for item in entries:
        spec = _parse_compare_list_item(item)
        path_raw = spec.get("path")
        path = _coerce_run_path(path_raw, base_dir)
        if path is None or not path.exists():
            continue
        if _is_curve_path(path):
            continue
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _selected_config_compare_runs(
    selection: Dict[str, Any],
) -> tuple[list[Path], list[Optional[str]], list[str]]:
    base_dir = selection.get("base")
    if base_dir is None:
        return [], [], ["Selection base directory is unavailable."]

    warnings: list[str] = []
    runs: list[Path] = []
    labels: list[Optional[str]] = []
    seen: set[Path] = set()

    def _add_candidate(path: Optional[Path], label: Optional[str] = None) -> None:
        if path is None:
            return
        if not path.exists():
            warnings.append(f"Skipping missing path: {path}")
            return
        if _is_curve_path(path):
            warnings.append(f"Skipping curve path in compare-configs mode: {path}")
            return
        resolved = analysis.resolve_run_dir(path)
        if resolved in seen:
            return
        seen.add(resolved)
        runs.append(resolved)
        labels.append(label)

    preset_entries = _load_compare_preset(selection.get("compare_preset_path"), base_dir)
    list_entries = _compare_list_entries(selection) if not preset_entries else []
    if preset_entries:
        for entry in preset_entries:
            path = entry.get("path")
            if not isinstance(path, Path):
                continue
            _add_candidate(path, entry.get("label"))
        return runs, labels, warnings

    if list_entries:
        for item in list_entries:
            spec = _parse_compare_list_item(item)
            path_raw = spec.get("path")
            path = _coerce_run_path(path_raw, base_dir)
            _add_candidate(path, spec.get("label"))
        return runs, labels, warnings

    run_a_path = _coerce_run_path(selection.get("run_a_path"), base_dir)
    run_b_path = _coerce_run_path(selection.get("run_b_path"), base_dir)
    run_a = run_a_path if run_a_path is not None else _coerce_run_path(selection.get("run_a"), base_dir)
    run_b = run_b_path if run_b_path is not None else _coerce_run_path(selection.get("run_b"), base_dir)
    run_single = _coerce_run_path(selection.get("run_single"), base_dir)

    _add_candidate(run_a)
    _add_candidate(run_b)
    if not runs:
        _add_candidate(run_single)
    return runs, labels, warnings


def run_output_plots(
    selection: Dict[str, Any],
    opts: Dict[str, Any],
    *,
    save_plots: bool = False,
    save_analysis: bool = False,
    plots_dpi: int = 150,
) -> Dict[str, Any]:
    export_figures: list[tuple[Any, str]] = []
    export_mode = "single"
    export_run_label = ""

    def _remember_figure(fig_obj: Any, plot_name: str) -> None:
        if fig_obj is None:
            return
        if str(plot_name) != "spike_stats":
            _apply_output_y_window(
                fig_obj,
                y_window=opts.get("y_window"),
            )
        if str(plot_name) != "spike_stats":
            _apply_output_window_origin_zero(
                fig_obj,
                enabled=bool(opts.get("output_plot_window_zero_origin", False)),
                plot_window=opts.get("plot_window"),
            )
        export_figures.append((analysis.resolve_figure(fig_obj), str(plot_name)))

    def _payload() -> Dict[str, Any]:
        rows = _rows_from_figures(
            export_figures,
            figure_type="output",
            mode=export_mode,
            run_label=export_run_label,
        )
        figures = [
            {
                "figure": analysis.resolve_figure(fig_obj),
                "plot_name": str(plot_name),
            }
            for fig_obj, plot_name in export_figures
        ]
        return {
            "rows": rows,
            "mode": export_mode,
            "run_label": export_run_label,
            "figures": figures,
        }

    output_scale = 1.0
    external_scale = 1.0
    base_dir = selection.get("base")
    preset_defaults = _load_compare_preset_defaults(selection.get("compare_preset_path"), base_dir)
    if preset_defaults:
        opts = _merge_preset_defaults(opts, preset_defaults)
    save_overwrite = bool(opts.get("save_overwrite", False))
    preset_entries = _load_compare_preset(selection.get("compare_preset_path"), base_dir)
    list_entries = _compare_list_entries(selection) if not preset_entries else []
    if preset_entries or list_entries:
        export_mode = "compare_list"
        export_run_label = "compare_list"
        paths: list[Path] = []
        shifts: list[Optional[float]] = []
        scales: list[Optional[float]] = []
        labels_override: list[Optional[str]] = []
        colors_override: list[Optional[str]] = []
        linestyles_override: list[Optional[str]] = []
        paths_from_compare_list: list[bool] = []
        has_curve = False
        if preset_entries:
            for entry in preset_entries:
                path = entry["path"]
                shift_ms = entry.get("shift_ms")
                scale_val = entry.get("scale")
                if _is_curve_path(path):
                    has_curve = True
                paths.append(path)
                shifts.append(shift_ms)
                scales.append(scale_val)
                labels_override.append(entry.get("label"))
                colors_override.append(entry.get("color"))
                linestyles_override.append(entry.get("linestyle"))
                paths_from_compare_list.append(False)
        else:
            compare_list_set = {str(v) for v in (selection.get("compare_list") or [])}
            for item in list_entries:
                spec = _parse_compare_list_item(item)
                path_raw = spec.get("path")
                shift_ms = spec.get("shift_ms")
                scale_val = spec.get("scale")
                path = _coerce_run_path(path_raw, base_dir)
                if path is None:
                    continue
                if not path.exists():
                    print(f"Skipping missing path: {path}")
                    continue
                paths.append(path)
                shifts.append(shift_ms)
                scales.append(scale_val)
                labels_override.append(spec.get("label"))
                colors_override.append(spec.get("color"))
                linestyles_override.append(spec.get("linestyle"))
                from_compare_list = False if isinstance(item, dict) else str(item) in compare_list_set
                paths_from_compare_list.append(from_compare_list)
                if _is_curve_path(path):
                    has_curve = True

        fallback_sim_cfg = None
        for path in paths:
            if not _is_curve_path(path):
                res = _safe_load_results(path)
                fallback_sim_cfg = (res.get("sim_cfg") or {}) if res else None
                break

        if len(paths) == 1 and not has_curve and not preset_entries:
            selection["run_single"] = paths[0]
            selection["compare_list"] = []
            selection["compare_list_paths"] = []
        else:
            curves = []
            isi_curves = []
            labels = []
            colors = []
            linestyles = []
            sim_cfgs = []
            curve_paths = []
            curve_scales = []
            curve_from_compare_list: list[bool] = []
            for idx, (path, shift_ms, scale_val) in enumerate(zip(paths, shifts, scales)):
                if _is_curve_path(path):
                    curve = _load_curve_from_path(path, opts, shift_ms=shift_ms, fallback_sim_cfg=fallback_sim_cfg)
                    if not curve:
                        continue
                    label_override = labels_override[idx] if idx < len(labels_override) else None
                    color_override = colors_override[idx] if idx < len(colors_override) else None
                    linestyle_override = linestyles_override[idx] if idx < len(linestyles_override) else None
                    curves.append(curve)
                    isi_curves.append(None)
                    labels.append(label_override or Path(path).stem)
                    colors.append(color_override)
                    linestyles.append(linestyle_override)
                    sim_cfgs.append(
                        _sim_cfg_with_shifted_stim(
                            _default_sim_cfg_for_curve(curve, fallback=fallback_sim_cfg),
                            shift_ms,
                        )
                    )
                    curve_paths.append(path)
                    curve_scales.append(scale_val)
                    curve_from_compare_list.append(paths_from_compare_list[idx])
                else:
                    res = _safe_load_results(path)
                    if res is None:
                        print(f"Skipping missing run: {path}")
                        continue
                    curve = _run_output_curve_from_results(res, opts, shift_ms=shift_ms)
                    if not curve:
                        continue
                    isi_curve = _run_output_isi_curve_from_results(res, opts, shift_ms=shift_ms)
                    label_override = labels_override[idx] if idx < len(labels_override) else None
                    color_override = colors_override[idx] if idx < len(colors_override) else None
                    linestyle_override = linestyles_override[idx] if idx < len(linestyles_override) else None
                    curves.append(curve)
                    isi_curves.append(isi_curve)
                    labels.append(label_override or analysis.run_label(path))
                    colors.append(color_override or (res.get("sim_cfg", {}) or {}).get("color", None))
                    linestyles.append(linestyle_override)
                    sim_cfgs.append(_sim_cfg_with_shifted_stim(res.get("sim_cfg", {}) or {}, shift_ms))
                    curve_paths.append(path)
                    curve_scales.append(scale_val)
                    curve_from_compare_list.append(paths_from_compare_list[idx])

            if not curves:
                print("No valid curves found in compare list.")
                return _payload()

            run_indices = [i for i, p in enumerate(curve_paths) if not _is_curve_path(p)]
            selected_run_indices = [
                i for i in run_indices
                if i < len(curve_from_compare_list) and curve_from_compare_list[i]
            ]
            if len(selected_run_indices) >= 2:
                seen_colors: Dict[Any, int] = {}
                override_indices: list[int] = []
                for idx in selected_run_indices:
                    col = colors[idx]
                    if col is None:
                        override_indices.append(idx)
                        continue
                    if col in seen_colors:
                        override_indices.append(idx)
                        override_indices.append(seen_colors[col])
                    else:
                        seen_colors[col] = idx
                if override_indices:
                    uniq_indices = sorted(set(override_indices))
                    rand_colors = _unique_compare_colors(len(uniq_indices))
                    for idx, col in zip(uniq_indices, rand_colors):
                        colors[idx] = col

            layout = (opts.get("compare_output_layout") or "overlay").lower()
            overlay_layouts = {"overlay", "same", "same-plot", "overlap"}
            stacked_layouts = {"stacked", "top-bottom", "vertical"}
            side_layouts = {"side-by-side", "side_by_side", "horizontal"}
            compare_title = "Output curves"
            compare_figsize = _parse_figsize(opts.get("output_compare_figsize"))
            compare_panel = _parse_figsize(opts.get("output_compare_panel_size"))

            curve_plot_mode = _output_curve_plot_mode(opts)
            stim_start, stim_stop = _stim_window_for_opts(sim_cfgs[0] or {}, opts)
            plot_window_cmp, plot_window_warn = _resolve_plot_window_for_opts(
                sim_cfgs[0] or {},
                opts,
                window_key="plot_window",
                context="Output compare list",
            )
            if plot_window_warn:
                print(plot_window_warn)
            if curve_plot_mode != "rate":
                if curve_plot_mode == "isi" and not any(_curve_has_series(c, "isi_ms") for c in isi_curves):
                    print("Output ISI compare skipped: no ISI data available in selected runs.")
                    return _payload()

                rate_curves_plot: list[Optional[Dict[str, Any]]] = []
                for idx, curve in enumerate(curves):
                    scale_val = curve_scales[idx] if curve_scales[idx] is not None else (
                        external_scale if _is_curve_path(curve_paths[idx]) else output_scale
                    )
                    rate_curves_plot.append(_scale_curve_for_plot(curve, scale_val))

                curve_figsize = compare_figsize
                if curve_plot_mode == "rate_isi":
                    if compare_figsize:
                        curve_figsize = (compare_figsize[0], max(compare_figsize[1] * 1.6, compare_figsize[1] + 1.6))
                    else:
                        curve_figsize = (6.0, 6.0)

                fig_curve, axes_map = _plot_output_curve_mode_figure(
                    mode=curve_plot_mode,
                    rate_curves=rate_curves_plot,
                    isi_curves=isi_curves,
                    labels=labels,
                    colors=colors,
                    linestyles=linestyles,
                    plot_window=plot_window_cmp,
                    stim_start=stim_start,
                    stim_stop=stim_stop,
                    title=compare_title,
                    line_width=float(opts.get("output_linewidth", 2.0)),
                    stim_linewidth=float(opts.get("output_stim_linewidth", 1.0)),
                    figsize=curve_figsize,
                )
                if curve_plot_mode == "rate_isi" and opts.get("output_show_metric_points", True):
                    ax_rate = axes_map.get("rate")
                    if ax_rate is not None:
                        for idx, (curve, label) in enumerate(zip(curves, labels)):
                            scale_val = curve_scales[idx] if curve_scales[idx] is not None else (
                                external_scale if _is_curve_path(curve_paths[idx]) else output_scale
                            )
                            metrics = _compute_output_metrics(
                                curve,
                                sim_cfgs[idx] or {},
                                opts,
                                peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                                drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                                rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                                auc_window=opts.get("output_auc_window", "stim"),
                                stim_start_ms=stim_start,
                                stim_stop_ms=stim_stop,
                            )
                            metrics_plot = _scale_metrics_for_plot(metrics, scale_val)
                            col = colors[idx] if idx < len(colors) and colors[idx] is not None else None
                            _plot_metric_points(
                                ax_rate,
                                metrics_plot,
                                color=col,
                                label_prefix=label,
                                show_labels=bool(opts.get("output_metric_label_points", False)),
                                size=float(opts.get("output_metric_marker_size", 36.0)),
                            )
                            if opts.get("output_metric_window_markers", False):
                                _plot_metric_window_markers(ax_rate, metrics_plot, color=col, **_metric_window_kwargs(opts))
                _save_fig(
                    fig_curve,
                    analysis.plot_dir_for_compare(selection["base"], Path("compare_list"), Path("curves")) / "compare_output_curve_list.png",
                    enabled=save_plots,
                    dpi=plots_dpi,
                    overwrite=save_overwrite,
                )
                _remember_figure(fig_curve, "compare_output_curve_list")
                return _payload()

            fig_curve = None
            ax = None
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            y_label = "Rate (Hz)" if (curves[0].get("units", "Hz") == "Hz") else "Rate (normalized)"
            if layout in stacked_layouts or layout in side_layouts:
                n = len(curves)
                if layout in stacked_layouts:
                    if compare_panel:
                        panel_w, panel_h = compare_panel
                        fig_curve, axes = plt.subplots(
                            n, 1, figsize=(panel_w, max(panel_h, panel_h * n)), sharex=True
                        )
                    else:
                        fig_curve, axes = plt.subplots(n, 1, figsize=(6, max(3.2, 3.2 * n)), sharex=True)
                else:
                    if compare_panel:
                        panel_w, panel_h = compare_panel
                        fig_curve, axes = plt.subplots(
                            1, n, figsize=(max(panel_w, panel_w * n), panel_h), sharey=True
                        )
                    else:
                        fig_curve, axes = plt.subplots(1, n, figsize=(max(4.0, 4.2 * n), 4), sharey=True)
                axes = np.atleast_1d(axes)
                for idx, (curve, label) in enumerate(zip(curves, labels)):
                    ax_i = axes[idx] if idx < axes.size else axes[-1]
                    scale_val = curve_scales[idx] if curve_scales[idx] is not None else (external_scale if _is_curve_path(curve_paths[idx]) else output_scale)
                    curve_plot = _scale_curve_for_plot(curve, scale_val)
                    t_ms = np.asarray(curve_plot.get("t_ms", []) or [], dtype=float)
                    y = np.asarray(curve_plot.get("rate_hz", []) or [], dtype=float)
                    col = colors[idx] if colors[idx] is not None else color_cycle[idx % len(color_cycle)]
                    ls = linestyles[idx] if idx < len(linestyles) and linestyles[idx] else "-"
                    plot_label = _legend_safe_label(label)
                    ax_i.plot(t_ms, y, lw=float(opts.get("output_linewidth", 2.0)), color=col, linestyle=ls, label=plot_label)
                    stim_start_i, stim_stop_i = _stim_window_for_opts(sim_cfgs[idx] or {}, opts)
                    if opts.get("output_show_metric_points", True):
                        metrics = _compute_output_metrics(
                            curve,
                            sim_cfgs[idx] or {},
                            opts,
                            peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                            drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                            rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                            auc_window=opts.get("output_auc_window", "stim"),
                            stim_start_ms=stim_start_i,
                            stim_stop_ms=stim_stop_i,
                        )
                        metrics_plot = _scale_metrics_for_plot(metrics, scale_val)
                        _plot_metric_points(
                            ax_i,
                            metrics_plot,
                            color=col,
                            label_prefix=label,
                            show_labels=bool(opts.get("output_metric_label_points", False)),
                            size=float(opts.get("output_metric_marker_size", 36.0)),
                        )
                        if opts.get("output_metric_window_markers", False):
                            _plot_metric_window_markers(ax_i, metrics_plot, color=col, **_metric_window_kwargs(opts))
                    if stim_start_i is not None:
                        ax_i.axvline(float(stim_start_i), color="k", linestyle="-", linewidth=float(opts.get("output_stim_linewidth", 1.0)))
                    if stim_stop_i is not None:
                        ax_i.axvline(float(stim_stop_i), color="k", linestyle="-", linewidth=float(opts.get("output_stim_linewidth", 1.0)))
                    if plot_window_cmp is not None:
                        ax_i.set_xlim(plot_window_cmp[0], plot_window_cmp[1])
                    ax_i.set_title(label)
                    if plot_label:
                        ax_i.legend()
                    ax_i.grid(True)
                    if layout in side_layouts:
                        ax_i.set_xlabel("Time (ms)")
                        if idx == 0:
                            ax_i.set_ylabel(y_label)
                    else:
                        ax_i.set_ylabel(y_label)
                        if idx == axes.size - 1:
                            ax_i.set_xlabel("Time (ms)")
                if fig_curve is not None:
                    fig_curve.suptitle(compare_title)
                    plt.tight_layout(rect=[0, 0, 1, 0.96])
            else:
                fig_curve, ax = plt.subplots(figsize=compare_figsize or (6, 4))
                for idx, (curve, label) in enumerate(zip(curves, labels)):
                    scale_val = curve_scales[idx] if curve_scales[idx] is not None else (external_scale if _is_curve_path(curve_paths[idx]) else output_scale)
                    curve_plot = _scale_curve_for_plot(curve, scale_val)
                    t_ms = np.asarray(curve_plot.get("t_ms", []) or [], dtype=float)
                    y = np.asarray(curve_plot.get("rate_hz", []) or [], dtype=float)
                    col = colors[idx] if colors[idx] is not None else color_cycle[idx % len(color_cycle)]
                    ls = linestyles[idx] if idx < len(linestyles) and linestyles[idx] else "-"
                    plot_label = _legend_safe_label(label)
                    ax.plot(t_ms, y, lw=float(opts.get("output_linewidth", 2.0)), color=col, linestyle=ls, label=plot_label)
                    if opts.get("output_show_metric_points", True):
                        metrics = _compute_output_metrics(
                            curve,
                            sim_cfgs[idx] or {},
                            opts,
                            peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                            drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                            rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                            auc_window=opts.get("output_auc_window", "stim"),
                            stim_start_ms=stim_start,
                            stim_stop_ms=stim_stop,
                        )
                        metrics_plot = _scale_metrics_for_plot(metrics, scale_val)
                        _plot_metric_points(
                            ax,
                            metrics_plot,
                            color=col,
                            label_prefix=label,
                            show_labels=bool(opts.get("output_metric_label_points", False)),
                            size=float(opts.get("output_metric_marker_size", 36.0)),
                        )
                        if opts.get("output_metric_window_markers", False):
                            _plot_metric_window_markers(ax, metrics_plot, color=col, **_metric_window_kwargs(opts))

                if stim_start is not None:
                    ax.axvline(float(stim_start), color="k", linestyle="-", linewidth=float(opts.get("output_stim_linewidth", 1.0)))
                if stim_stop is not None:
                    ax.axvline(float(stim_stop), color="k", linestyle="-", linewidth=float(opts.get("output_stim_linewidth", 1.0)))
                if plot_window_cmp is not None:
                    ax.set_xlim(plot_window_cmp[0], plot_window_cmp[1])
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel(y_label)
                ax.set_title(compare_title)
                ax.grid(True)
                handles, labels_ax = ax.get_legend_handles_labels()
                if any(str(lbl).strip() and not str(lbl).startswith("_") for lbl in labels_ax):
                    ax.legend()
                plt.tight_layout()
            _save_fig(
                fig_curve,
                analysis.plot_dir_for_compare(selection["base"], Path("compare_list"), Path("curves")) / "compare_output_curve_list.png",
                enabled=save_plots,
                dpi=plots_dpi,
                overwrite=save_overwrite,
            )
            _remember_figure(fig_curve, "compare_output_curve_list")
            return _payload()

    if compare_enabled(selection):
        export_mode = "compare"
        run_a, run_b, res_a, res_b = resolve_compare(selection)
        if run_b is None or res_a is None or res_b is None:
            if run_b is None:
                print("Comparison disabled (set Compare B to a run name).")
                return _payload()
        curve_only_a = res_a is None
        curve_only_b = res_b is None
        if curve_only_a and curve_only_b and (run_a is None or run_b is None):
            print("Comparison disabled (set Compare A/B to a run name or curve path).")
            return _payload()
        label_a = analysis.run_label(run_a) if not curve_only_a else Path(run_a).stem
        label_b = analysis.run_label(run_b) if not curve_only_b else Path(run_b).stem
        export_run_label = f"{label_a}_vs_{label_b}"
        if res_a is not None:
            smooth_mode = (res_a.get("sim_cfg", {}) or {}).get("avg_rate_curve_smooth_mode", "center")
        elif res_b is not None:
            smooth_mode = (res_b.get("sim_cfg", {}) or {}).get("avg_rate_curve_smooth_mode", "center")
        else:
            smooth_mode = "center"
        smooth_mode = opts.get("output_smooth_mode") or smooth_mode
        output_norms = None
        if opts.get("output_curve_mode", "raw") == "normalized":
            if not curve_only_a and not curve_only_b and res_a is not None and res_b is not None:
                curve_a = analysis.compute_output_curve_from_results(
                    res_a,
                    bin_ms=None,
                    smooth_ms=opts.get("win_size", 25) or None,
                    smooth_mode=smooth_mode,
                )
                curve_b = analysis.compute_output_curve_from_results(
                    res_b,
                    bin_ms=None,
                    smooth_ms=opts.get("win_size", 25) or None,
                    smooth_mode=smooth_mode,
                )
                if curve_a and curve_b:
                    sim_cfg_a_norm = _sim_cfg_with_output_stim_overrides(res_a.get("sim_cfg", {}) or {}, opts)
                    sim_cfg_b_norm = _sim_cfg_with_output_stim_overrides(res_b.get("sim_cfg", {}) or {}, opts)
                    norm_a = analysis.normalize_output_curve(
                        curve_a,
                        sim_cfg_a_norm,
                        mode="normalized",
                        norm_mode=opts.get("output_norm_mode", "avg"),
                        baseline_ms=opts.get("output_metric_window_ms", 100.0),
                        baseline_mode=opts.get("output_metric_mode", "point"),
                        baseline_center_ms=opts.get("output_baseline_center_ms"),
                        norm_window=opts.get("output_norm_window", "stim"),
                    )
                    norm_b = analysis.normalize_output_curve(
                        curve_b,
                        sim_cfg_b_norm,
                        mode="normalized",
                        norm_mode=opts.get("output_norm_mode", "avg"),
                        baseline_ms=opts.get("output_metric_window_ms", 100.0),
                        baseline_mode=opts.get("output_metric_mode", "point"),
                        baseline_center_ms=opts.get("output_baseline_center_ms"),
                        norm_window=opts.get("output_norm_window", "stim"),
                    )
                    output_norms = (
                        {
                            "baseline_mean": norm_a.get("baseline_mean"),
                            "norm_scale": norm_a.get("norm_scale"),
                            "norm_mode": norm_a.get("norm_mode"),
                            "norm_window": norm_a.get("norm_window"),
                            "baseline_ms": norm_a.get("baseline_ms"),
                            "baseline_mode": norm_a.get("baseline_mode"),
                            "baseline_time_ms": norm_a.get("baseline_time_ms"),
                        },
                        {
                            "baseline_mean": norm_b.get("baseline_mean"),
                            "norm_scale": norm_b.get("norm_scale"),
                            "norm_mode": norm_b.get("norm_mode"),
                            "norm_window": norm_b.get("norm_window"),
                            "baseline_ms": norm_b.get("baseline_ms"),
                            "baseline_mode": norm_b.get("baseline_mode"),
                            "baseline_time_ms": norm_b.get("baseline_time_ms"),
                        },
                    )

        if opts.get("plot_outputs", True):
            compare_window_ref_cfg = (res_a.get("sim_cfg", {}) if res_a else (res_b.get("sim_cfg", {}) if res_b else {})) or {}
            plot_window_compare, plot_window_warn = _resolve_plot_window_for_opts(
                compare_window_ref_cfg,
                opts,
                window_key="plot_window",
                context="Output compare",
            )
            if plot_window_warn:
                print(plot_window_warn)
            if curve_only_a or curve_only_b:
                print("Output plots skipped: curve-only compare uses output curve plot only.")
                fig_cmp = None
            else:
                fig_cmp, _ = plotting.plot_compare_side_by_side(
                    res_a,
                    res_b,
                    labels=(label_a, label_b),
                    win_size=opts.get("win_size", 25),
                    bin_ms=opts.get("output_bin_ms"),
                    plot_window=plot_window_compare,
                    smooth_mode=smooth_mode,
                    output_norms=output_norms,
                    layout=opts.get("compare_output_layout", "side-by-side"),
                    output_scale=output_scale,
                    stim_start_ms=opts.get("output_stim_start_ms"),
                    stim_stop_ms=opts.get("output_stim_stop_ms"),
                )
            if fig_cmp is not None and opts.get("output_show_metric_points", True):
                curve_a_out = analysis.compute_output_curve_from_results(
                    res_a,
                    bin_ms=opts.get("output_bin_ms"),
                    smooth_ms=opts.get("win_size", 25) or None,
                    smooth_mode=smooth_mode,
                )
                curve_b_out = analysis.compute_output_curve_from_results(
                    res_b,
                    bin_ms=opts.get("output_bin_ms"),
                    smooth_ms=opts.get("win_size", 25) or None,
                    smooth_mode=smooth_mode,
                )
                norm_a = output_norms[0] if output_norms else None
                norm_b = output_norms[1] if output_norms else None
                curve_a_out = _apply_output_norm(curve_a_out, norm_a)
                curve_b_out = _apply_output_norm(curve_b_out, norm_b)
                metrics_a = _compute_output_metrics(
                    curve_a_out or {},
                    res_a.get("sim_cfg", {}) or {},
                    opts,
                    peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                    drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                    rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                    auc_window=opts.get("output_auc_window", "stim"),
                ) if curve_a_out else None
                metrics_b = _compute_output_metrics(
                    curve_b_out or {},
                    res_b.get("sim_cfg", {}) or {},
                    opts,
                    peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                    drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                    rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                    auc_window=opts.get("output_auc_window", "stim"),
                ) if curve_b_out else None
                metrics_a_plot = _scale_metrics_for_plot(metrics_a, output_scale)
                metrics_b_plot = _scale_metrics_for_plot(metrics_b, output_scale)

                fig_cmp_res = analysis.resolve_figure(fig_cmp)
                axes = np.atleast_1d(fig_cmp_res.axes)
                if axes.size == 1:
                    ax = axes[0]
                    line_colors = [line.get_color() for line in ax.lines if len(line.get_xdata()) > 2]
                    color_a = line_colors[0] if len(line_colors) > 0 else None
                    color_b = line_colors[1] if len(line_colors) > 1 else None
                    if metrics_a:
                        _plot_metric_points(
                            ax,
                            metrics_a_plot,
                            color=color_a,
                            label_prefix=label_a,
                            show_labels=bool(opts.get("output_metric_label_points", False)),
                            size=float(opts.get("output_metric_marker_size", 36.0)),
                        )
                        if opts.get("output_metric_window_markers", False):
                            _plot_metric_window_markers(ax, metrics_a_plot, color=color_a, **_metric_window_kwargs(opts))
                    if metrics_b:
                        _plot_metric_points(
                            ax,
                            metrics_b_plot,
                            color=color_b,
                            label_prefix=label_b,
                            show_labels=bool(opts.get("output_metric_label_points", False)),
                            size=float(opts.get("output_metric_marker_size", 36.0)),
                        )
                        if opts.get("output_metric_window_markers", False):
                            _plot_metric_window_markers(ax, metrics_b_plot, color=color_b, **_metric_window_kwargs(opts))
                else:
                    if axes.size > 0 and metrics_a:
                        line_colors = [line.get_color() for line in axes[0].lines if len(line.get_xdata()) > 2]
                        color_a = line_colors[0] if line_colors else None
                        _plot_metric_points(
                            axes[0],
                            metrics_a_plot,
                            color=color_a,
                            label_prefix=label_a,
                            show_labels=bool(opts.get("output_metric_label_points", False)),
                            size=float(opts.get("output_metric_marker_size", 36.0)),
                        )
                        if opts.get("output_metric_window_markers", False):
                            _plot_metric_window_markers(axes[0], metrics_a_plot, color=color_a, **_metric_window_kwargs(opts))
                    if axes.size > 1 and metrics_b:
                        line_colors = [line.get_color() for line in axes[1].lines if len(line.get_xdata()) > 2]
                        color_b = line_colors[0] if line_colors else None
                        _plot_metric_points(
                            axes[1],
                            metrics_b_plot,
                            color=color_b,
                            label_prefix=label_b,
                            show_labels=bool(opts.get("output_metric_label_points", False)),
                            size=float(opts.get("output_metric_marker_size", 36.0)),
                        )
                        if opts.get("output_metric_window_markers", False):
                            _plot_metric_window_markers(axes[1], metrics_b_plot, color=color_b, **_metric_window_kwargs(opts))
            if fig_cmp is not None:
                _save_fig(
                    fig_cmp,
                    analysis.plot_dir_for_compare(selection["base"], run_a, run_b) / "compare_outputs.png",
                    enabled=save_plots,
                    dpi=plots_dpi,
                    overwrite=save_overwrite,
                )
                _remember_figure(fig_cmp, "compare_outputs")

        if opts.get("plot_output_curve", True):
            curve_a = _load_curve_from_path(
                Path(run_a),
                opts,
                fallback_sim_cfg=(res_b.get("sim_cfg") if res_b else None),
            ) if curve_only_a else _run_output_curve_from_results(res_a, opts)
            curve_b = _load_curve_from_path(
                Path(run_b),
                opts,
                fallback_sim_cfg=(res_a.get("sim_cfg") if res_a else None),
            ) if curve_only_b else _run_output_curve_from_results(res_b, opts)
            isi_a = None if curve_only_a else _run_output_isi_curve_from_results(res_a, opts)
            isi_b = None if curve_only_b else _run_output_isi_curve_from_results(res_b, opts)
            scatter_curve = analysis.load_scatter_curve_optional(
                enabled=opts.get("output_scatter_enabled", False),
                path=opts.get("output_scatter_path", ""),
                time_unit=opts.get("output_scatter_time_unit", "s"),
                bin_ms=opts.get("output_bin_ms"),
                smooth_ms=_curve_smooth_ms(opts),
                smooth_mode=opts.get("output_smooth_mode", "causal"),
                shift_ms=opts.get("output_scatter_shift_ms"),
                quiet=True,
            )
            if curve_a and curve_b:
                curve_mode = _output_curve_plot_mode(opts)
                sim_cfg_a = (res_a.get("sim_cfg") if res_a else _default_sim_cfg_for_curve(curve_a, fallback=(res_b.get("sim_cfg") if res_b else None))) or {}
                sim_cfg_b = (res_b.get("sim_cfg") if res_b else _default_sim_cfg_for_curve(curve_b, fallback=(res_a.get("sim_cfg") if res_a else None))) or {}
                stim_start, stim_stop = _stim_window_for_opts(sim_cfg_a, opts)
                plot_window_curve_compare, plot_window_warn = _resolve_plot_window_for_opts(
                    sim_cfg_a,
                    opts,
                    window_key="plot_window",
                    context="Output curve compare",
                )
                if plot_window_warn:
                    print(plot_window_warn)
                scale_a = external_scale if curve_only_a else output_scale
                scale_b = external_scale if curve_only_b else output_scale
                curve_a_plot = _scale_curve_for_plot(curve_a, scale_a)
                curve_b_plot = _scale_curve_for_plot(curve_b, scale_b)
                fig_curve = None
                ax_rate = None
                if curve_mode == "rate":
                    fig_curve, _ = plotting.plot_compare_output_curves(
                        curve_a_plot,
                        curve_b_plot,
                        labels=(label_a, label_b),
                        plot_window=plot_window_curve_compare,
                        stim_start=stim_start,
                        stim_stop=stim_stop,
                        title="Output curves",
                        line_width=float(opts.get("output_linewidth", 2.0)),
                        stim_linewidth=float(opts.get("output_stim_linewidth", 1.0)),
                    )
                    ax_rate = fig_curve.axes[0] if fig_curve.axes else None
                else:
                    if curve_mode == "isi" and not any(_curve_has_series(c, "isi_ms") for c in (isi_a, isi_b)):
                        print("Output ISI compare skipped: no ISI data available in selected runs.")
                    else:
                        fig_curve, axes_map = _plot_output_curve_mode_figure(
                            mode=curve_mode,
                            rate_curves=[curve_a_plot, curve_b_plot],
                            isi_curves=[isi_a, isi_b],
                            labels=[label_a, label_b],
                            plot_window=plot_window_curve_compare,
                            stim_start=stim_start,
                            stim_stop=stim_stop,
                            title="Output curves",
                            line_width=float(opts.get("output_linewidth", 2.0)),
                            stim_linewidth=float(opts.get("output_stim_linewidth", 1.0)),
                        )
                        ax_rate = axes_map.get("rate")

                if fig_curve is not None and ax_rate is not None and opts.get("output_show_metric_points", True):
                    metrics_a = _compute_output_metrics(
                        curve_a,
                        sim_cfg_a,
                        opts,
                        peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                        drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                        rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                        auc_window=opts.get("output_auc_window", "stim"),
                        stim_start_ms=stim_start,
                        stim_stop_ms=stim_stop,
                    )
                    metrics_b = _compute_output_metrics(
                        curve_b,
                        sim_cfg_b,
                        opts,
                        peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                        drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                        rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                        auc_window=opts.get("output_auc_window", "stim"),
                        stim_start_ms=stim_start,
                        stim_stop_ms=stim_stop,
                    )
                    metrics_a_plot = _scale_metrics_for_plot(metrics_a, scale_a)
                    metrics_b_plot = _scale_metrics_for_plot(metrics_b, scale_b)
                    line_colors = [line.get_color() for line in ax_rate.lines]
                    color_a = line_colors[0] if len(line_colors) > 0 else None
                    color_b = line_colors[1] if len(line_colors) > 1 else None
                    _plot_metric_points(
                        ax_rate,
                        metrics_a_plot,
                        color=color_a,
                        label_prefix=label_a,
                        show_labels=bool(opts.get("output_metric_label_points", False)),
                        size=float(opts.get("output_metric_marker_size", 36.0)),
                    )
                    if opts.get("output_metric_window_markers", False):
                        _plot_metric_window_markers(ax_rate, metrics_a_plot, color=color_a, **_metric_window_kwargs(opts))
                    _plot_metric_points(
                        ax_rate,
                        metrics_b_plot,
                        color=color_b,
                        label_prefix=label_b,
                        show_labels=bool(opts.get("output_metric_label_points", False)),
                        size=float(opts.get("output_metric_marker_size", 36.0)),
                    )
                    if opts.get("output_metric_window_markers", False):
                        _plot_metric_window_markers(ax_rate, metrics_b_plot, color=color_b, **_metric_window_kwargs(opts))

                if fig_curve is not None and scatter_curve and ax_rate is not None:
                    scatter_plot = _scale_curve_for_plot(scatter_curve, external_scale)
                    ax_rate.plot(
                        np.asarray(scatter_plot.get("t_ms", []), dtype=float),
                        np.asarray(scatter_plot.get("rate_hz", []), dtype=float),
                        color=opts.get("output_scatter_color", "0.4"),
                        lw=float(opts.get("output_linewidth", 2.0)),
                        ls="--",
                        label=opts.get("output_scatter_label", "External curve"),
                    )
                    if opts.get("output_show_metric_points", True):
                        scatter_metrics = _compute_output_metrics(
                            scatter_curve,
                            sim_cfg_a,
                            opts,
                            peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                            drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                            rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                            auc_window=opts.get("output_auc_window", "stim"),
                            stim_start_ms=stim_start,
                            stim_stop_ms=stim_stop,
                        )
                        scatter_metrics_plot = _scale_metrics_for_plot(scatter_metrics, external_scale)
                        _plot_metric_points(
                            ax_rate,
                            scatter_metrics_plot,
                            color=opts.get("output_scatter_color", "0.4"),
                            label_prefix=opts.get("output_scatter_label", "External curve"),
                            show_labels=bool(opts.get("output_metric_label_points", False)),
                            size=float(opts.get("output_metric_marker_size", 36.0)),
                        )
                        if opts.get("output_metric_window_markers", False):
                            _plot_metric_window_markers(
                                ax_rate,
                                scatter_metrics_plot,
                                color=opts.get("output_scatter_color", "0.4"),
                                **_metric_window_kwargs(opts),
                            )
                    ax_rate.legend()

                if fig_curve is not None:
                    _save_fig(
                        fig_curve,
                        analysis.plot_dir_for_compare(selection["base"], run_a, run_b) / "compare_output_curve.png",
                        enabled=save_plots,
                        dpi=plots_dpi,
                        overwrite=save_overwrite,
                    )
                    _remember_figure(fig_curve, "compare_output_curve")
            else:
                print("Output curve compare skipped: missing spikes in one run.")

        if opts.get("plot_spike_stats", False):
            print("Spike stats are only shown for single runs.")
        return _payload()

    run_dir, res = resolve_single(selection)
    export_mode = "single"
    export_run_label = analysis.run_label(run_dir)
    smooth_mode = (res.get("sim_cfg", {}) or {}).get("avg_rate_curve_smooth_mode", "center")
    smooth_mode = opts.get("output_smooth_mode") or smooth_mode
    plot_window_single, plot_window_warn = _resolve_plot_window_for_opts(
        (res.get("sim_cfg", {}) or {}),
        opts,
        window_key="plot_window",
        context="Output single",
    )
    if plot_window_warn:
        print(plot_window_warn)

    if opts.get("plot_outputs", True):
        in_vivo_curve = analysis.load_bio_curve_optional(
            enabled=opts.get("multi_use_bio_curve", False),
            path=opts.get("bio_curve_path", ""),
            time_col=opts.get("bio_curve_time_col", "Time"),
            rate_col=opts.get("bio_curve_rate_col", "AvgFiringRate"),
            t_min=opts.get("bio_curve_t_min", 0.0),
            delay_ms=opts.get("bio_curve_delay_ms", 0.0),
            time_unit=opts.get("bio_curve_time_unit", "s"),
            shift_ms=opts.get("bio_curve_shift_ms"),
            quiet=True,
        )

        if (res.get("mode") or "single") == "multi":
            all_param_data, sim_params, pw = analysis.build_multi_plot_inputs(
                res,
                plot_window=plot_window_single,
            )
            if opts.get("output_bin_ms") is not None:
                sim_params["bins"] = float(opts.get("output_bin_ms"))
            stim_start, stim_stop = _stim_window_for_opts(res.get("sim_cfg", {}) or {}, opts)
            if stim_start is not None:
                sim_params["stim_start_ms"] = stim_start
            if stim_stop is not None:
                sim_params["stim_stop_ms"] = stim_stop
            plot_bio = None
            if in_vivo_curve is not None:
                plot_bio = (True, in_vivo_curve[0], in_vivo_curve[1])
            output_norm = None
            if opts.get("output_curve_mode", "raw") == "normalized" and opts.get("multi_norm_fr") is None:
                curve_norm = analysis.compute_output_curve_from_results(
                    res,
                    bin_ms=sim_params.get("bins"),
                    smooth_ms=opts.get("win_size", 25) or None,
                    smooth_mode=smooth_mode,
                )
                if curve_norm:
                    sim_cfg_norm = _sim_cfg_with_output_stim_overrides(res.get("sim_cfg", {}) or {}, opts)
                    norm_curve = analysis.normalize_output_curve(
                        curve_norm,
                        sim_cfg_norm,
                        mode="normalized",
                        norm_mode=opts.get("output_norm_mode", "avg"),
                        baseline_ms=opts.get("output_metric_window_ms", 100.0),
                        baseline_mode=opts.get("output_metric_mode", "point"),
                        baseline_center_ms=opts.get("output_baseline_center_ms"),
                        norm_window=opts.get("output_norm_window", "stim"),
                    )
                    output_norm = {
                        "baseline_mean": norm_curve.get("baseline_mean"),
                        "norm_scale": norm_curve.get("norm_scale"),
                        "norm_mode": norm_curve.get("norm_mode"),
                        "norm_window": norm_curve.get("norm_window"),
                        "baseline_ms": norm_curve.get("baseline_ms"),
                        "baseline_mode": norm_curve.get("baseline_mode"),
                        "baseline_time_ms": norm_curve.get("baseline_time_ms"),
                    }

            plotting.plot_multi(
                all_param_data,
                sim_params=sim_params,
                win_size=opts.get("win_size", 25),
                plot_type=opts.get("multi_plot_type", "line"),
                plot_bio=plot_bio,
                plot_raster=opts.get("plot_raster", True),
                raster_style=opts.get("raster_style", "dot"),
                plot_window=pw,
                norm_fr=opts.get("multi_norm_fr"),
                shade_mode=opts.get("multi_shade_mode"),
                set_color=(res.get("sim_cfg", {}) or {}).get("color", None),
                smooth_mode=smooth_mode,
                output_norm=output_norm,
                output_scale=output_scale,
                bio_scale=external_scale,
            )
            if opts.get("output_show_metric_points", True):
                curve_out = analysis.compute_output_curve_from_results(
                    res,
                    bin_ms=sim_params.get("bins"),
                    smooth_ms=opts.get("win_size", 25) or None,
                    smooth_mode=smooth_mode,
                )
                curve_out = _apply_output_norm(curve_out, output_norm)
                metrics_out = _compute_output_metrics(
                    curve_out or {},
                    res.get("sim_cfg", {}) or {},
                    opts,
                    peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                    drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                    rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                    auc_window=opts.get("output_auc_window", "stim"),
                    stim_start_ms=stim_start,
                    stim_stop_ms=stim_stop,
                ) if curve_out else None
                fig_out = analysis.resolve_figure(None)
                ax_rate = fig_out.axes[0] if fig_out.axes else None
                if ax_rate is not None and metrics_out:
                    line_colors = [line.get_color() for line in ax_rate.lines if len(line.get_xdata()) > 2]
                    color_out = line_colors[0] if line_colors else None
                    metrics_plot = _scale_metrics_for_plot(metrics_out, output_scale)
                    _plot_metric_points(
                        ax_rate,
                        metrics_plot,
                        color=color_out,
                        label_prefix=analysis.run_label(run_dir),
                        show_labels=bool(opts.get("output_metric_label_points", False)),
                        size=float(opts.get("output_metric_marker_size", 36.0)),
                    )
                    if opts.get("output_metric_window_markers", False):
                        _plot_metric_window_markers(ax_rate, metrics_plot, color=color_out, **_metric_window_kwargs(opts))
                if ax_rate is not None and in_vivo_curve is not None:
                    bio_curve = _curve_from_xy(in_vivo_curve[0] * 1000.0, in_vivo_curve[1])
                    bio_metrics = _compute_output_metrics(
                        bio_curve,
                        res.get("sim_cfg", {}) or {},
                        opts,
                        peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                        drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                        rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                        auc_window=opts.get("output_auc_window", "stim"),
                        stim_start_ms=stim_start,
                        stim_stop_ms=stim_stop,
                    )
                    bio_metrics_plot = _scale_metrics_for_plot(bio_metrics, external_scale)
                    _plot_metric_points(
                        ax_rate,
                        bio_metrics_plot,
                        color="k",
                        label_prefix="Bio",
                        show_labels=bool(opts.get("output_metric_label_points", False)),
                        size=float(opts.get("output_metric_marker_size", 36.0)),
                    )
                    if opts.get("output_metric_window_markers", False):
                        _plot_metric_window_markers(ax_rate, bio_metrics_plot, color="k", **_metric_window_kwargs(opts))
            _save_fig(
                plt.gcf(),
                analysis.plot_dir_for_run(run_dir) / "output_plot.png",
                enabled=save_plots,
                dpi=plots_dpi,
                overwrite=save_overwrite,
            )
            _remember_figure(plt.gcf(), "output_plot")
        else:
            in_vivo_curve_plot = in_vivo_curve
            if in_vivo_curve_plot is not None and external_scale not in (None, 1.0):
                try:
                    t_s, rate = in_vivo_curve_plot
                    in_vivo_curve_plot = (t_s, np.asarray(rate, dtype=float) * float(external_scale))
                except Exception:
                    pass
            fig_out = plotting.plot_results(
                res,
                syn_records=res.get("syn_records"),
                in_vivo_curve=in_vivo_curve_plot,
                win_size=opts.get("win_size", 25),
                raster_style=opts.get("raster_style", "dot"),
                plot_raster=opts.get("plot_raster", True),
                plot_window=plot_window_single,
                smooth_mode=smooth_mode,
                bin_ms=opts.get("output_bin_ms"),
                line_width=opts.get("output_linewidth", 2.0),
                shade_alpha=opts.get("output_shade_alpha", 0.25),
            )
            _save_fig(
                fig_out,
                analysis.plot_dir_for_run(run_dir) / "output_plot.png",
                enabled=save_plots,
                dpi=plots_dpi,
                overwrite=save_overwrite,
            )
            _remember_figure(fig_out, "output_plot")

    if opts.get("plot_output_curve", True):
        curve_single = _run_output_curve_from_results(res, opts)
        isi_single = _run_output_isi_curve_from_results(res, opts)
        scatter_curve = analysis.load_scatter_curve_optional(
            enabled=opts.get("output_scatter_enabled", False),
            path=opts.get("output_scatter_path", ""),
            time_unit=opts.get("output_scatter_time_unit", "s"),
            bin_ms=opts.get("output_bin_ms"),
            smooth_ms=_curve_smooth_ms(opts),
            smooth_mode=opts.get("output_smooth_mode", "causal"),
            shift_ms=opts.get("output_scatter_shift_ms"),
            quiet=True,
        )
        if curve_single:
            curve_mode = _output_curve_plot_mode(opts)
            stim_start, stim_stop = _stim_window_for_opts(res.get("sim_cfg", {}) or {}, opts)
            curve_single_plot = _scale_curve_for_plot(curve_single, output_scale)
            run_color = (res.get("sim_cfg", {}) or {}).get("color", None)
            fig_curve = None
            ax_rate = None
            if curve_mode == "rate":
                fig_curve, _ = plotting.plot_output_curve(
                    curve_single_plot,
                    label=analysis.run_label(run_dir),
                    color=run_color,
                    plot_window=plot_window_single,
                    stim_start=stim_start,
                    stim_stop=stim_stop,
                    title="Output curve (avg)",
                    line_width=float(opts.get("output_linewidth", 2.0)),
                    stim_linewidth=float(opts.get("output_stim_linewidth", 1.0)),
                )
                ax_rate = fig_curve.axes[0] if fig_curve.axes else None
            else:
                if curve_mode == "isi" and not _curve_has_series(isi_single, "isi_ms"):
                    print("Output ISI plot skipped: no ISI data available in this run.")
                else:
                    fig_curve, axes_map = _plot_output_curve_mode_figure(
                        mode=curve_mode,
                        rate_curves=[curve_single_plot],
                        isi_curves=[isi_single],
                        labels=[analysis.run_label(run_dir)],
                        colors=[run_color],
                        plot_window=plot_window_single,
                        stim_start=stim_start,
                        stim_stop=stim_stop,
                        title="Output curve (avg)",
                        line_width=float(opts.get("output_linewidth", 2.0)),
                        stim_linewidth=float(opts.get("output_stim_linewidth", 1.0)),
                    )
                    ax_rate = axes_map.get("rate")

            if fig_curve is not None and ax_rate is not None and opts.get("output_show_metric_points", True):
                metrics_single = _compute_output_metrics(
                    curve_single,
                    res.get("sim_cfg", {}) or {},
                    opts,
                    peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                    drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                    rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                    auc_window=opts.get("output_auc_window", "stim"),
                    stim_start_ms=stim_start,
                    stim_stop_ms=stim_stop,
                )
                metrics_single_plot = _scale_metrics_for_plot(metrics_single, output_scale)
                line_color = ax_rate.lines[0].get_color() if ax_rate.lines else run_color
                _plot_metric_points(
                    ax_rate,
                    metrics_single_plot,
                    color=line_color,
                    label_prefix=analysis.run_label(run_dir),
                    show_labels=bool(opts.get("output_metric_label_points", False)),
                    size=float(opts.get("output_metric_marker_size", 36.0)),
                )
                if opts.get("output_metric_window_markers", False):
                    _plot_metric_window_markers(ax_rate, metrics_single_plot, color=line_color, **_metric_window_kwargs(opts))
            if fig_curve is not None and scatter_curve and ax_rate is not None:
                scatter_plot = _scale_curve_for_plot(scatter_curve, external_scale)
                ax_rate.plot(
                    np.asarray(scatter_plot.get("t_ms", []), dtype=float),
                    np.asarray(scatter_plot.get("rate_hz", []), dtype=float),
                    color=opts.get("output_scatter_color", "0.4"),
                    lw=float(opts.get("output_linewidth", 2.0)),
                    ls="--",
                    label=opts.get("output_scatter_label", "External curve"),
                )
                if opts.get("output_show_metric_points", True):
                    scatter_metrics = _compute_output_metrics(
                        scatter_curve,
                        res.get("sim_cfg", {}) or {},
                        opts,
                        peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                        drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                        rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                        auc_window=opts.get("output_auc_window", "stim"),
                        stim_start_ms=stim_start,
                        stim_stop_ms=stim_stop,
                    )
                    scatter_metrics_plot = _scale_metrics_for_plot(scatter_metrics, external_scale)
                    _plot_metric_points(
                        ax_rate,
                        scatter_metrics_plot,
                        color=opts.get("output_scatter_color", "0.4"),
                        label_prefix=opts.get("output_scatter_label", "External curve"),
                        show_labels=bool(opts.get("output_metric_label_points", False)),
                        size=float(opts.get("output_metric_marker_size", 36.0)),
                    )
                    if opts.get("output_metric_window_markers", False):
                        _plot_metric_window_markers(
                            ax_rate,
                            scatter_metrics_plot,
                            color=opts.get("output_scatter_color", "0.4"),
                            **_metric_window_kwargs(opts),
                        )
                ax_rate.legend()

            if fig_curve is not None:
                _save_fig(
                    fig_curve,
                    analysis.plot_dir_for_run(run_dir) / "output_curve.png",
                    enabled=save_plots,
                    dpi=plots_dpi,
                    overwrite=save_overwrite,
                )
                _remember_figure(fig_curve, "output_curve")
            _save_json(
                curve_single,
                analysis.analysis_dir_for_run(run_dir) / "output_curve.json",
                enabled=save_analysis,
            )
        else:
            print("Output curve plot skipped: missing spikes in this run.")

    if opts.get("plot_spike_stats", False):
        stats_single = analysis.summarize_spike_trials(res, plot=True, print_summary=False)
        _save_fig(
            plt.gcf(),
            analysis.plot_dir_for_run(run_dir) / "spike_stats.png",
            enabled=save_plots,
            dpi=plots_dpi,
            overwrite=save_overwrite,
        )
        _remember_figure(plt.gcf(), "spike_stats")
        _save_json(
            stats_single,
            analysis.analysis_dir_for_run(run_dir) / "spike_stats.json",
            enabled=save_analysis,
        )

    return _payload()


def run_input_plots(
    selection: Dict[str, Any],
    opts: Dict[str, Any],
    *,
    save_plots: bool = False,
    save_analysis: bool = False,
    plots_dpi: int = 150,
) -> Dict[str, Any]:
    export_figures: list[tuple[Any, str]] = []
    export_mode = "single"
    export_run_label = ""
    save_overwrite = bool(opts.get("save_overwrite", False))

    def _remember_figure(fig_obj: Any, plot_name: str) -> None:
        if fig_obj is None:
            return
        export_figures.append((analysis.resolve_figure(fig_obj), str(plot_name)))

    def _payload() -> Dict[str, Any]:
        rows = _rows_from_figures(
            export_figures,
            figure_type="input",
            mode=export_mode,
            run_label=export_run_label,
        )
        return {"rows": rows, "mode": export_mode, "run_label": export_run_label}

    compare_runs = _compare_list_run_paths(selection)
    if len(compare_runs) >= 2:
        export_mode = "compare"
        run_a, run_b = compare_runs[0], compare_runs[1]
        res_a = _safe_load_results(run_a)
        res_b = _safe_load_results(run_b)
        if res_a is None or res_b is None:
            print("Comparison disabled (missing input data in one or more runs).")
            return _payload()

        label_a = analysis.run_label(run_a)
        label_b = analysis.run_label(run_b)
        export_run_label = f"{label_a}_vs_{label_b}"
        group_colors = analysis.merge_group_colors(res_a, res_b)
        plot_window_input_compare, plot_window_warn = _resolve_plot_window_for_opts(
            (res_a.get("sim_cfg", {}) or {}),
            opts,
            window_key="input_plot_window",
            context="Input compare",
        )
        if plot_window_warn:
            print(plot_window_warn)

        if opts.get("plot_inputs_mean", True):
            summary_a = analysis.summarize_inputs_from_results(
                res_a,
                groups=opts.get("input_groups"),
                bin_ms=opts.get("input_bin_ms"),
                smooth_ms=opts.get("input_smooth_ms"),
                input_source=opts.get("input_source", "saved"),
                std_mode=opts.get("input_std_mode", "std"),
            )
            summary_b = analysis.summarize_inputs_from_results(
                res_b,
                groups=opts.get("input_groups"),
                bin_ms=opts.get("input_bin_ms"),
                smooth_ms=opts.get("input_smooth_ms"),
                input_source=opts.get("input_source", "saved"),
                std_mode=opts.get("input_std_mode", "std"),
            )
            fig_cmp_in, _ = plotting.plot_compare_input_means(
                summary_a,
                summary_b,
                labels=(label_a, label_b),
                groups=opts.get("input_groups"),
                layout=opts.get("compare_input_layout", "side-by-side"),
                show_std=opts.get("compare_show_input_std", False),
                output_curves=(
                    (res_a.get("meta") or {}).get("avg_rate_curve"),
                    (res_b.get("meta") or {}).get("avg_rate_curve"),
                ),
                legend_loc=opts.get("input_legend_loc"),
                group_colors=group_colors,
                line_width=opts.get("input_linewidth", 2.0),
                shade_alpha=opts.get("input_shade_alpha", 0.2),
                output_linewidth=opts.get("input_output_linewidth", 1.5),
                plot_window=plot_window_input_compare,
            )
            _save_fig(
                fig_cmp_in,
                analysis.plot_dir_for_compare(selection["base"], run_a, run_b) / "compare_inputs.png",
                enabled=save_plots,
                dpi=plots_dpi,
                overwrite=save_overwrite,
            )
            _remember_figure(fig_cmp_in, "compare_inputs")
        if opts.get("plot_input_raster", False):
            print("Input raster is only available for single runs.")
        return _payload()
    if len(compare_runs) == 1:
        selection = dict(selection)
        selection["run_single"] = compare_runs[0]
        selection["compare_list"] = []
        selection["compare_list_paths"] = []

    run_dir, res = resolve_single(selection)
    export_mode = "single"
    export_run_label = analysis.run_label(run_dir)
    group_colors = analysis.group_colors_from_results(res)
    plot_window_input_single, plot_window_warn = _resolve_plot_window_for_opts(
        (res.get("sim_cfg", {}) or {}),
        opts,
        window_key="input_plot_window",
        context="Input single",
    )
    if plot_window_warn:
        print(plot_window_warn)

    if opts.get("plot_inputs_mean", True):
        summary_single = analysis.summarize_inputs_from_results(
            res,
            groups=opts.get("input_groups"),
            bin_ms=opts.get("input_bin_ms"),
            smooth_ms=opts.get("input_smooth_ms"),
            input_source=opts.get("input_source", "saved"),
            std_mode=opts.get("input_std_mode", "std"),
        )
        curve_single = (res.get("meta") or {}).get("avg_rate_curve")
        fig_in, _ = plotting.plot_input_means(
            summary_single,
            label=analysis.run_label(run_dir),
            groups=opts.get("input_groups"),
            show_std=opts.get("show_input_std", False),
            output_curve=curve_single,
            plot_window=plot_window_input_single,
            legend_loc=opts.get("input_legend_loc"),
            group_colors=group_colors,
            line_width=opts.get("input_linewidth", 2.0),
            shade_alpha=opts.get("input_shade_alpha", 0.2),
            output_linewidth=opts.get("input_output_linewidth", 1.5),
        )
        _save_fig(
            fig_in,
            analysis.plot_dir_for_run(run_dir) / "inputs_mean.png",
            enabled=save_plots,
            dpi=plots_dpi,
            overwrite=save_overwrite,
        )
        _remember_figure(fig_in, "inputs_mean")
        _save_json(
            summary_single,
            analysis.analysis_dir_for_run(run_dir) / "inputs_summary.json",
            enabled=save_analysis,
        )

    if opts.get("plot_input_raster", False):
        payload = analysis.select_inputs_payload(res, trial_idx=opts.get("input_raster_trial_idx", 0))
        if payload:
            plotting.plot_inputs_by_group(
                payload,
                res.get("sim_cfg", {}) or {},
                groups=opts.get("input_groups"),
                bin_ms=opts.get("input_bin_ms"),
                win_size=opts.get("input_raster_win_size", 25.0),
                group_colors=group_colors,
                raster_style=opts.get("input_raster_style", "dot"),
                max_trains_per_group=opts.get("input_raster_max_trains", 200),
                plot_window=plot_window_input_single,
                legend_loc=opts.get("input_legend_loc"),
                plot_raster=True,
                line_width=opts.get("input_linewidth", 2.0),
                raster_linewidth=opts.get("input_raster_linewidth", 0.8),
                stim_linewidth=opts.get("input_stim_linewidth", 1.0),
            )
            _save_fig(
                plt.gcf(),
                analysis.plot_dir_for_run(run_dir) / "inputs_raster.png",
                enabled=save_plots,
                dpi=plots_dpi,
                overwrite=save_overwrite,
            )
            _remember_figure(plt.gcf(), "inputs_raster")
        else:
            print("No saved inputs available for raster plot.")

    return _payload()


def run_output_metrics(
    selection: Dict[str, Any],
    opts: Dict[str, Any],
    *,
    save_analysis: bool = False,
) -> Optional[Dict[str, Any]]:
    base_dir = selection.get("base")
    preset_defaults = _load_compare_preset_defaults(selection.get("compare_preset_path"), base_dir)
    if preset_defaults:
        opts = _merge_preset_defaults(opts, preset_defaults)
    metric_overrides = _output_metric_overrides(opts)
    preset_entries = _load_compare_preset(selection.get("compare_preset_path"), base_dir)
    list_entries = _compare_list_entries(selection) if not preset_entries else []
    if preset_entries or list_entries:
        paths: list[Path] = []
        shifts: list[Optional[float]] = []
        scales: list[Optional[float]] = []
        labels_override: list[Optional[str]] = []
        has_curve = False
        if preset_entries:
            for entry in preset_entries:
                path = entry["path"]
                shift_ms = entry.get("shift_ms")
                scale_val = entry.get("scale")
                if _is_curve_path(path):
                    has_curve = True
                paths.append(path)
                shifts.append(shift_ms)
                scales.append(scale_val)
                labels_override.append(entry.get("label"))
        else:
            for item in list_entries:
                spec = _parse_compare_list_item(item)
                path_raw = spec.get("path")
                shift_ms = spec.get("shift_ms")
                scale_val = spec.get("scale")
                path = _coerce_run_path(path_raw, base_dir)
                if path is None:
                    continue
                if not path.exists():
                    print(f"Skipping missing path: {path}")
                    continue
                paths.append(path)
                shifts.append(shift_ms)
                scales.append(scale_val)
                labels_override.append(spec.get("label"))
                if _is_curve_path(path):
                    has_curve = True

        if len(paths) == 1 and not has_curve:
            selection["run_single"] = paths[0]
        else:
            fallback_sim_cfg = None
            for path in paths:
                if not _is_curve_path(path):
                    res = _safe_load_results(path)
                    fallback_sim_cfg = (res.get("sim_cfg") or {}) if res else None
                    break

            metrics_map: Dict[str, Any] = {}
            for idx, (path, shift_ms, _scale_val) in enumerate(zip(paths, shifts, scales)):
                if _is_curve_path(path):
                    curve = _load_curve_from_path(path, opts, shift_ms=shift_ms, fallback_sim_cfg=fallback_sim_cfg)
                    if not curve:
                        continue
                    label_override = labels_override[idx] if idx < len(labels_override) else None
                    label = label_override or Path(path).stem
                    sim_cfg = _default_sim_cfg_for_curve(curve, fallback=fallback_sim_cfg)
                else:
                    res = _safe_load_results(path)
                    if res is None:
                        print(f"Skipping missing run: {path}")
                        continue
                    curve = _run_output_curve_from_results(res, opts, shift_ms=shift_ms)
                    if not curve:
                        continue
                    label_override = labels_override[idx] if idx < len(labels_override) else None
                    label = label_override or analysis.run_label(path)
                    sim_cfg = res.get("sim_cfg", {}) or {}

                metrics_map[label] = _compute_output_metrics_with_spread(
                    curve,
                    sim_cfg,
                    opts,
                    results=None if _is_curve_path(path) else res,
                    shift_ms=shift_ms,
                    **metric_overrides,
                )

            if not metrics_map:
                print("Output metrics skipped: no valid curves in compare list.")
                return None
            _save_json(
                metrics_map,
                analysis.analysis_dir_for_compare(selection["base"], Path("compare_list"), Path("curves")) / "output_metrics_list.json",
                enabled=save_analysis,
            )
            return metrics_map

    if compare_enabled(selection):
        run_a, run_b, res_a, res_b = resolve_compare(selection)
        if run_b is None:
            print("Output metrics skipped: compare B not set.")
            return None
        curve_only_a = res_a is None
        curve_only_b = res_b is None
        curve_a = _load_curve_from_path(
            Path(run_a),
            opts,
            fallback_sim_cfg=(res_b.get("sim_cfg") if res_b else None),
        ) if curve_only_a else _run_output_curve_from_results(res_a, opts)
        curve_b = _load_curve_from_path(
            Path(run_b),
            opts,
            fallback_sim_cfg=(res_a.get("sim_cfg") if res_a else None),
        ) if curve_only_b else _run_output_curve_from_results(res_b, opts)
        if not curve_a or not curve_b:
            print("Output metrics skipped: missing curves in compare selection.")
            return None
        sim_cfg_a = res_a.get("sim_cfg", {}) if res_a else _default_sim_cfg_for_curve(curve_a)
        sim_cfg_b = res_b.get("sim_cfg", {}) if res_b else _default_sim_cfg_for_curve(curve_b)
        metrics_map = {
            analysis.run_label(run_a) if not curve_only_a else Path(run_a).stem: _compute_output_metrics_with_spread(
                curve_a,
                sim_cfg_a or {},
                opts,
                results=None if curve_only_a else res_a,
                **metric_overrides,
            ),
            analysis.run_label(run_b) if not curve_only_b else Path(run_b).stem: _compute_output_metrics_with_spread(
                curve_b,
                sim_cfg_b or {},
                opts,
                results=None if curve_only_b else res_b,
                **metric_overrides,
            ),
        }
        _save_json(
            metrics_map,
            analysis.analysis_dir_for_compare(selection["base"], Path(run_a), Path(run_b)) / "output_metrics_compare.json",
            enabled=save_analysis,
        )
        return metrics_map
    run_dir, res = resolve_single(selection)
    curve = _run_output_curve_from_results(res, opts)
    if not curve:
        print("Output metrics skipped: missing spikes in this run.")
        return None
    metrics = _compute_output_metrics_with_spread(
        curve,
        res.get("sim_cfg", {}) or {},
        opts,
        results=res,
        **metric_overrides,
    )
    _save_json(
        metrics,
        analysis.analysis_dir_for_run(run_dir) / "output_metrics.json",
        enabled=save_analysis,
    )
    return metrics


_OUTPUT_METRIC_DIST_FIELDS = [
    "metric",
    "run_label",
    "source",
    "point_kind",
    "trial_index",
    "value",
]


def _collect_output_metric_distributions(
    selection: Dict[str, Any],
    opts: Dict[str, Any],
) -> Dict[str, Any]:
    base_dir = selection.get("base")
    preset_defaults = _load_compare_preset_defaults(selection.get("compare_preset_path"), base_dir)
    if preset_defaults:
        opts = _merge_preset_defaults(opts, preset_defaults)
    metric_overrides = _output_metric_overrides(opts)
    payload: Dict[str, Any] = {
        "mode": "single",
        "by_label": {},
        "run_dir": None,
        "run_a": None,
        "run_b": None,
    }

    preset_entries = _load_compare_preset(selection.get("compare_preset_path"), base_dir)
    list_entries = _compare_list_entries(selection) if not preset_entries else []
    if preset_entries or list_entries:
        paths: list[Path] = []
        shifts: list[Optional[float]] = []
        labels_override: list[Optional[str]] = []
        colors_override: list[Optional[str]] = []
        has_curve = False
        if preset_entries:
            for entry in preset_entries:
                path = entry["path"]
                if _is_curve_path(path):
                    has_curve = True
                paths.append(path)
                shifts.append(entry.get("shift_ms"))
                labels_override.append(entry.get("label"))
                colors_override.append(entry.get("color"))
        else:
            for item in list_entries:
                spec = _parse_compare_list_item(item)
                path = _coerce_run_path(spec.get("path"), base_dir)
                if path is None:
                    continue
                if not path.exists():
                    print(f"Skipping missing path: {path}")
                    continue
                paths.append(path)
                shifts.append(spec.get("shift_ms"))
                labels_override.append(spec.get("label"))
                colors_override.append(spec.get("color"))
                if _is_curve_path(path):
                    has_curve = True

        if len(paths) == 1 and not has_curve:
            selection = dict(selection)
            selection["run_single"] = paths[0]
            selection["compare_list"] = []
            selection["compare_list_paths"] = []
        else:
            payload["mode"] = "list"
            fallback_sim_cfg = None
            for path in paths:
                if _is_curve_path(path):
                    continue
                res_fb = _safe_load_results(path)
                if res_fb is not None:
                    fallback_sim_cfg = (res_fb.get("sim_cfg") or {}) if isinstance(res_fb, dict) else None
                    break
            by_label: Dict[str, Any] = {}
            for idx, (path, shift_ms) in enumerate(zip(paths, shifts)):
                color_override = colors_override[idx] if idx < len(colors_override) else None
                res: Optional[Dict[str, Any]] = None
                if _is_curve_path(path):
                    curve = _load_curve_from_path(path, opts, shift_ms=shift_ms, fallback_sim_cfg=fallback_sim_cfg)
                    if not curve:
                        continue
                    sim_cfg = _default_sim_cfg_for_curve(curve, fallback=fallback_sim_cfg)
                    source = "curve"
                    label = labels_override[idx] if idx < len(labels_override) else None
                    label = label or Path(path).stem
                    color_val = color_override
                else:
                    res = _safe_load_results(path)
                    if res is None:
                        print(f"Skipping missing run: {path}")
                        continue
                    curve = _run_output_curve_from_results(res, opts, shift_ms=shift_ms)
                    if not curve:
                        continue
                    sim_cfg = res.get("sim_cfg", {}) or {}
                    source = "run"
                    label = labels_override[idx] if idx < len(labels_override) else None
                    label = label or analysis.run_label(path)
                    color_val = _extract_output_color_from_results(res) or color_override
                sim_cfg_metrics = _sim_cfg_with_shifted_stim(sim_cfg, shift_ms)
                summary = _compute_output_metrics(curve, sim_cfg_metrics, opts, **metric_overrides)
                by_label[str(label)] = {
                    "summary": summary,
                    "trials": _compute_output_trial_metrics(
                        res,
                        sim_cfg,
                        opts,
                        shift_ms=shift_ms,
                        **metric_overrides,
                    ) if res is not None else [],
                    "source": source,
                    "color": color_val,
                }
            payload["by_label"] = by_label
            return payload

    if compare_enabled(selection):
        run_a, run_b, res_a, res_b = resolve_compare(selection)
        payload["mode"] = "compare"
        payload["run_a"] = run_a
        payload["run_b"] = run_b
        if run_a is None or run_b is None:
            return payload
        curve_only_a = res_a is None
        curve_only_b = res_b is None
        curve_a = (
            _load_curve_from_path(
                Path(run_a),
                opts,
                fallback_sim_cfg=(res_b.get("sim_cfg") if res_b else None),
            )
            if curve_only_a
            else _run_output_curve_from_results(res_a, opts)
        )
        curve_b = (
            _load_curve_from_path(
                Path(run_b),
                opts,
                fallback_sim_cfg=(res_a.get("sim_cfg") if res_a else None),
            )
            if curve_only_b
            else _run_output_curve_from_results(res_b, opts)
        )
        if not curve_a or not curve_b:
            return payload
        sim_cfg_a = res_a.get("sim_cfg", {}) if res_a else _default_sim_cfg_for_curve(curve_a)
        sim_cfg_b = res_b.get("sim_cfg", {}) if res_b else _default_sim_cfg_for_curve(curve_b)
        label_a = analysis.run_label(run_a) if not curve_only_a else Path(run_a).stem
        label_b = analysis.run_label(run_b) if not curve_only_b else Path(run_b).stem
        payload["by_label"] = {
            str(label_a): {
                "summary": _compute_output_metrics(curve_a, sim_cfg_a or {}, opts, **metric_overrides),
                "trials": _compute_output_trial_metrics(res_a, sim_cfg_a or {}, opts, **metric_overrides),
                "source": "curve" if curve_only_a else "run",
                "color": None if curve_only_a else _extract_output_color_from_results(res_a),
            },
            str(label_b): {
                "summary": _compute_output_metrics(curve_b, sim_cfg_b or {}, opts, **metric_overrides),
                "trials": _compute_output_trial_metrics(res_b, sim_cfg_b or {}, opts, **metric_overrides),
                "source": "curve" if curve_only_b else "run",
                "color": None if curve_only_b else _extract_output_color_from_results(res_b),
            },
        }
        return payload

    run_dir, res = resolve_single(selection)
    payload["mode"] = "single"
    payload["run_dir"] = run_dir
    curve = _run_output_curve_from_results(res, opts)
    if not curve:
        return payload
    sim_cfg = res.get("sim_cfg", {}) or {}
    label = analysis.run_label(run_dir)
    payload["by_label"] = {
        str(label): {
            "summary": _compute_output_metrics(curve, sim_cfg, opts, **metric_overrides),
            "trials": _compute_output_trial_metrics(res, sim_cfg, opts, **metric_overrides),
            "source": "run",
            "color": _extract_output_color_from_results(res),
        }
    }
    return payload


def _coerce_output_metric_plot_keys(metric_keys: Any) -> list[str]:
    options = [key for key in _OUTPUT_METRIC_VALUE_ORDER if key != "output_metrics_n_trials"]
    if metric_keys is None:
        requested = list(_OUTPUT_METRIC_PLOT_DEFAULT_KEYS)
    elif isinstance(metric_keys, str):
        requested = [metric_keys]
    else:
        requested = [str(k) for k in list(metric_keys)]
    out = [key for key in requested if key in options]
    if out:
        return out
    return [key for key in _OUTPUT_METRIC_PLOT_DEFAULT_KEYS if key in options] or options


def _coerce_output_metric_plot_style(style: Any) -> str:
    value = str(style or "box").strip().lower()
    return value if value in ("box", "bar") else "box"


def _metric_values_from_trials(trials: list[Dict[str, Any]], metric_key: str) -> list[float]:
    vals: list[float] = []
    for trial in trials:
        val = _coerce_metric_numeric(trial.get(metric_key))
        if val is not None:
            vals.append(val)
    return vals


def _output_metric_spread_from_trials(
    trials: list[float],
    *,
    std_mode: str,
) -> Optional[float]:
    if len(trials) < 2:
        return None
    spread = float(np.std(np.asarray(trials, dtype=float)))
    if std_mode == "sem":
        spread = spread / float(np.sqrt(len(trials)))
    return spread


def _output_metric_distribution_rows(
    payload: Dict[str, Any],
    metric_keys: list[str],
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    by_label = payload.get("by_label") or {}
    for metric_key in metric_keys:
        for label, entry in by_label.items():
            source = str(entry.get("source") or "run")
            trials = _metric_values_from_trials(entry.get("trials") or [], metric_key)
            for idx, val in enumerate(trials):
                rows.append({
                    "metric": metric_key,
                    "run_label": str(label),
                    "source": source,
                    "point_kind": "trial",
                    "trial_index": idx,
                    "value": val,
                })
            mean_val = _coerce_metric_numeric((entry.get("summary") or {}).get(metric_key))
            if mean_val is not None:
                rows.append({
                    "metric": metric_key,
                    "run_label": str(label),
                    "source": source,
                    "point_kind": "mean",
                    "trial_index": "",
                    "value": mean_val,
                })
    return rows


def _save_output_metric_distribution_rows(
    rows: list[Dict[str, Any]],
    *,
    selection: Dict[str, Any],
    requested_path: Any,
) -> Optional[Path]:
    if not rows:
        print("Output metric CSV not saved: no metric values were available.")
        return None
    out_path = _resolve_plot_data_target_path(
        selection,
        requested_path,
        figure_type="output_metrics_dist",
        mode="distribution",
    )
    if out_path.exists():
        print(f"Output metric CSV not saved: file already exists: {out_path}")
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_OUTPUT_METRIC_DIST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in _OUTPUT_METRIC_DIST_FIELDS})
    print(f"Saved output metric CSV: {out_path} ({len(rows)} rows)")
    return out_path


def _output_metric_distribution_plot_path(
    selection: Dict[str, Any],
    payload: Dict[str, Any],
) -> Path:
    mode = str(payload.get("mode") or "single")
    if mode == "single" and payload.get("run_dir") is not None:
        return analysis.plot_dir_for_run(payload["run_dir"]) / "output_metric_distributions.png"
    if mode == "compare" and payload.get("run_a") is not None and payload.get("run_b") is not None:
        return analysis.plot_dir_for_compare(
            selection["base"],
            Path(payload["run_a"]),
            Path(payload["run_b"]),
        ) / "output_metric_distributions.png"
    return analysis.plot_dir_for_compare(selection["base"], Path("compare_list"), Path("curves")) / "output_metric_distributions.png"


def _plot_output_metric_distributions(
    payload: Dict[str, Any],
    *,
    metric_keys: list[str],
    plot_style: str = "box",
    show_points: bool = True,
    jitter_points: bool = True,
    show_error: bool = False,
    std_mode: str = "std",
    ncols: Optional[int] = None,
    panel_size: tuple[float, float] = (4.8, 3.6),
    bar_alpha: float = 0.25,
    show_legend: bool = True,
    legend_loc: str = "best",
) -> Optional[Any]:
    by_label = payload.get("by_label") or {}
    labels = list(by_label.keys())
    if not labels:
        print("Output metric plot skipped: no valid runs/curves for current selection.")
        return None
    if not metric_keys:
        print("Output metric plot skipped: no metrics selected.")
        return None
    plot_style_use = _coerce_output_metric_plot_style(plot_style)
    std_mode_use = "sem" if str(std_mode or "std").strip().lower() == "sem" else "std"

    n_metrics = len(metric_keys)
    try:
        ncols_use = int(ncols) if ncols is not None else 0
    except Exception:
        ncols_use = 0
    if ncols_use <= 0:
        ncols_use = 1 if n_metrics == 1 else min(3, n_metrics)
    ncols_use = max(1, min(ncols_use, n_metrics))
    nrows = int(np.ceil(float(n_metrics) / float(ncols_use)))
    panel_w, panel_h = 4.8, 3.6
    if isinstance(panel_size, (list, tuple)) and len(panel_size) >= 2:
        panel_w_in = _safe_float(panel_size[0])
        panel_h_in = _safe_float(panel_size[1])
        if panel_w_in is not None and panel_w_in > 0:
            panel_w = float(panel_w_in)
        if panel_h_in is not None and panel_h_in > 0:
            panel_h = float(panel_h_in)
    bar_alpha_use = _safe_float(bar_alpha)
    if bar_alpha_use is None:
        bar_alpha_use = 0.25
    bar_alpha_use = min(1.0, max(0.0, float(bar_alpha_use)))
    fallback_colors = _unique_compare_colors(len(labels))
    color_by_label: Dict[str, Any] = {}
    for idx, label in enumerate(labels):
        fallback = fallback_colors[idx] if idx < len(fallback_colors) else plt.cm.tab10(idx % 10)
        entry = by_label.get(label) or {}
        color_by_label[label] = _coerce_plot_color(entry.get("color"), fallback)

    def _with_alpha(color: Any, alpha: float) -> Any:
        r, g, b, _ = mcolors.to_rgba(color)
        return (r, g, b, alpha)

    fig, axes = plt.subplots(
        nrows,
        ncols_use,
        figsize=(panel_w * ncols_use, panel_h * nrows),
        squeeze=False,
    )
    rng = np.random.default_rng(0)
    first_ax = None
    for idx, metric_key in enumerate(metric_keys):
        r = idx // ncols_use
        c = idx % ncols_use
        ax = axes[r][c]
        if first_ax is None:
            first_ax = ax
        has_data = False
        for pos, label in enumerate(labels, start=1):
            entry = by_label.get(label) or {}
            source = str(entry.get("source") or "run")
            series_color = color_by_label.get(label, "#4C72B0")
            trials = _metric_values_from_trials(entry.get("trials") or [], metric_key)
            mean_val = _coerce_metric_numeric((entry.get("summary") or {}).get(metric_key))
            if mean_val is None and trials:
                mean_val = float(np.mean(np.asarray(trials, dtype=float)))

            if plot_style_use == "box":
                if len(trials) >= 2:
                    has_data = True
                    ax.boxplot(
                        [trials],
                        positions=[pos],
                        widths=0.55,
                        showfliers=False,
                        patch_artist=True,
                        boxprops={"facecolor": _with_alpha(series_color, 0.25), "edgecolor": series_color, "linewidth": 1.1},
                        medianprops={"color": series_color, "linewidth": 1.4},
                        whiskerprops={"color": series_color, "linewidth": 1.0},
                        capprops={"color": series_color, "linewidth": 1.0},
                    )
                if mean_val is not None:
                    has_data = True
                    if source == "curve":
                        ax.scatter([pos], [mean_val], marker="D", s=40, color=series_color, zorder=4)
                    else:
                        ax.scatter(
                            [pos],
                            [mean_val],
                            marker="o",
                            s=40,
                            facecolors="none",
                            edgecolors=series_color,
                            linewidths=1.2,
                            zorder=4,
                        )
            else:
                if mean_val is not None:
                    has_data = True
                    spread = None
                    if show_error and source != "curve":
                        spread = _output_metric_spread_from_trials(trials, std_mode=std_mode_use)
                    bar_kwargs: Dict[str, Any] = {}
                    if spread is not None:
                        bar_kwargs.update({
                            "yerr": [spread],
                            "ecolor": series_color,
                            "capsize": 3,
                            "error_kw": {"linewidth": 1.0},
                        })
                    ax.bar(
                        [pos],
                        [mean_val],
                        width=0.62,
                        color=_with_alpha(series_color, bar_alpha_use),
                        edgecolor=series_color,
                        linewidth=1.1,
                        zorder=2,
                        **bar_kwargs,
                    )

            if show_points and trials:
                has_data = True
                if jitter_points:
                    x = np.full(len(trials), float(pos)) + rng.uniform(-0.1, 0.1, size=len(trials))
                else:
                    x = np.full(len(trials), float(pos))
                ax.scatter(
                    x,
                    trials,
                    s=16,
                    color=_with_alpha(series_color, 0.75),
                    zorder=3,
                )
        ax.set_xticks(list(range(1, len(labels) + 1)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(metric_key)
        ax.grid(axis="y", alpha=0.25)
        if not has_data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="0.45")
    for idx in range(n_metrics, nrows * ncols_use):
        r = idx // ncols_use
        c = idx % ncols_use
        axes[r][c].set_visible(False)

    if first_ax is not None and show_legend:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        legend_handles = []
        if show_points:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor="#2A9D8F",
                    markeredgecolor="#2A9D8F",
                    markersize=6,
                    label="Trials",
                )
            )
        if plot_style_use == "box":
            legend_handles.extend([
                Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#111111", markersize=6, label="Run mean"),
                Line2D([0], [0], marker="D", color="none", markerfacecolor="#E07A5F", markeredgecolor="#E07A5F", markersize=6, label="Curve mean"),
            ])
        else:
            legend_handles.append(
                Patch(facecolor=(0.25, 0.25, 0.25, bar_alpha_use), edgecolor="#333333", linewidth=1.0, label="Mean bar")
            )
            if show_error:
                legend_handles.append(
                    Line2D([0], [0], color="#333333", linewidth=1.2, label=f"Run {std_mode_use.upper()}")
                )
        if legend_handles:
            first_ax.legend(handles=legend_handles, loc=(str(legend_loc or "best")), fontsize=8)
    fig.suptitle(f"Output metric distributions ({plot_style_use})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def run_output_metric_distributions(
    selection: Dict[str, Any],
    opts: Dict[str, Any],
    *,
    metric_keys: Optional[list[str]] = None,
    plot_style: str = "box",
    show_points: bool = True,
    jitter_points: bool = True,
    show_error: bool = False,
    ncols: Optional[int] = None,
    panel_size: tuple[float, float] = (4.8, 3.6),
    bar_alpha: float = 0.25,
    show_legend: bool = True,
    legend_loc: str = "best",
    save_plots: bool = False,
    save_data: bool = False,
    plots_dpi: int = 150,
    data_path: Any = "",
    overwrite: bool = False,
) -> Optional[Dict[str, Any]]:
    payload = _collect_output_metric_distributions(selection, opts)
    metric_keys_use = _coerce_output_metric_plot_keys(metric_keys)
    plot_style_use = _coerce_output_metric_plot_style(plot_style)
    std_mode = _output_metrics_std_mode(opts)
    fig = _plot_output_metric_distributions(
        payload,
        metric_keys=metric_keys_use,
        plot_style=plot_style_use,
        show_points=bool(show_points),
        jitter_points=bool(jitter_points),
        show_error=bool(show_error),
        std_mode=std_mode,
        ncols=ncols,
        panel_size=panel_size,
        bar_alpha=bar_alpha,
        show_legend=bool(show_legend),
        legend_loc=legend_loc,
    )
    if fig is None:
        return None
    rows = _output_metric_distribution_rows(payload, metric_keys_use)
    payload["metric_keys"] = metric_keys_use
    payload["plot_style"] = plot_style_use
    payload["show_points"] = bool(show_points)
    payload["show_error"] = bool(show_error)
    payload["std_mode"] = std_mode
    payload["ncols"] = ncols
    payload["panel_size"] = panel_size
    payload["bar_alpha"] = bar_alpha
    payload["show_legend"] = bool(show_legend)
    payload["legend_loc"] = legend_loc
    payload["rows"] = rows
    payload["figure"] = analysis.resolve_figure(fig)
    if save_plots:
        _save_fig(
            fig,
            _output_metric_distribution_plot_path(selection, payload),
            enabled=True,
            dpi=plots_dpi,
            overwrite=overwrite,
        )
    if save_data:
        _save_output_metric_distribution_rows(
            rows,
            selection=selection,
            requested_path=data_path,
        )
    return payload


def _maybe_import_widgets():
    try:
        import ipywidgets as widgets  # type: ignore
    except Exception:
        return None
    return widgets


def _maybe_import_display():
    try:
        from IPython.display import display, Markdown  # type: ignore
    except Exception:
        return None, None
    return display, Markdown


def show_md(text: str) -> None:
    display, Markdown = _maybe_import_display()
    if display is not None and Markdown is not None:
        display(Markdown(text))
    else:
        print(text)

_HIGHLIGHT_METRICS = {
    "peak_latency_ms",
    "tpeak10_ms",
    "drop_pct",
    "t50_ms",
    "rebound_pct",
    "auc",
    "baseline_mean",
    "peak_rate_hz_raw",
    "peak_value_raw",
}


def _format_metric_key(key: str) -> str:
    if key == "t50_ms":
        label = "T50"
    elif key == "tpeak10_ms":
        label = "Tpeak10"
    else:
        label = key
    if key in _HIGHLIGHT_METRICS:
        return f"**{label}**"
    return label


def _format_metric_value(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float, np.integer, np.floating)):
        try:
            val_f = float(val)
        except Exception:
            return str(val)
        if np.isnan(val_f):
            return ""
        return f"{val_f:.3g}"
    return str(val)


def _filter_metrics_for_display(data: Dict[str, Any]) -> Dict[str, Any]:
    if not data:
        return {}
    filtered = dict(data)
    peak_raw = filtered.get("peak_rate_hz_raw")
    if peak_raw is None:
        peak_raw = filtered.get("peak_value_raw")
    if peak_raw is not None:
        for key in ("peak_value", "peak_rate_hz"):
            val = filtered.get(key)
            try:
                if val is not None and abs(float(val) - 1.0) < 1e-6:
                    filtered.pop(key, None)
            except Exception:
                pass
    if "peak_value_raw" in filtered and "peak_rate_hz_raw" in filtered:
        try:
            if abs(float(filtered["peak_value_raw"]) - float(filtered["peak_rate_hz_raw"])) < 1e-6:
                filtered.pop("peak_value_raw", None)
        except Exception:
            pass
    return filtered


_OUTPUT_METRIC_VALUE_ORDER = [
    "output_metrics_n_trials",
    "baseline_mean",
    "peak_rate_hz_raw",
    "peak_value_raw",
    "peak_rate_hz",
    "peak_value",
    "peak_latency_ms",
    "tpeak10_ms",
    "drop_value",
    "drop_pct",
    "t50_ms",
    "rebound_value",
    "rebound_pct",
    "auc",
]

_OUTPUT_METRIC_PLOT_DEFAULT_KEYS = [
    "baseline_mean",
    "peak_rate_hz_raw",
    "peak_latency_ms",
    "tpeak10_ms",
    "drop_pct",
    "t50_ms",
    "rebound_pct",
    "auc",
]


def _format_metric_cell(data: Dict[str, Any], key: str) -> str:
    val = data.get(key)
    cell = _format_metric_value(val)
    spread_key = f"{key}_spread"
    spread_val = data.get(spread_key)
    spread_txt = _format_metric_value(spread_val)
    if spread_txt == "":
        return cell
    mode = str(data.get("output_metrics_std_mode", "std") or "std").strip().lower()
    mode_label = "SEM" if mode == "sem" else "STD"
    if cell == "":
        return ""
    return f"{cell} +/- {spread_txt} ({mode_label})"


def _ordered_metric_keys(keys: list[str], order: Optional[list[str]] = None) -> list[str]:
    if not order:
        return keys
    ordered = [key for key in order if key in keys]
    for key in keys:
        if key not in ordered:
            ordered.append(key)
    return ordered


def format_kv_table(
    data: dict,
    *,
    title: str = "Output metrics",
    order: Optional[list[str]] = None,
    metric_keys: Optional[list[str]] = None,
) -> str:
    lines = [f"### {title}", "| Metric | Value |", "| --- | --- |"]
    data = _filter_metrics_for_display(data)
    keys = [
        key
        for key in _ordered_metric_keys(list(data.keys()), order)
        if not key.endswith("_spread") and key != "output_metrics_std_mode"
    ]
    if metric_keys:
        selected = [key for key in metric_keys if key in keys]
        if selected:
            keys = selected
    for key in keys:
        lines.append(f"| {_format_metric_key(key)} | {_format_metric_cell(data, key)} |")
    return "\n".join(lines)


def _format_delta(value: Any, ref: Any) -> str:
    try:
        val_f = float(value)
        ref_f = float(ref)
    except Exception:
        return ""
    delta = val_f - ref_f
    delta_str = f"{delta:+.3g}"
    if ref_f == 0:
        return f"Δ={delta_str}"
    pct = (delta / ref_f) * 100.0
    pct_str = f"{pct:+.3g}%"
    return f"Δ={delta_str} ({pct_str})"


def _best_labels_by_metric(
    data_by_label: Dict[str, Dict[str, Any]],
    *,
    reference_label: Optional[str],
) -> Dict[str, str]:
    if not reference_label or reference_label not in data_by_label:
        return {}
    best: Dict[str, str] = {}
    ref_vals = data_by_label.get(reference_label, {})
    for key, ref_val in ref_vals.items():
        try:
            ref_f = float(ref_val)
        except Exception:
            continue
        best_label = None
        best_err = None
        for label, metrics in data_by_label.items():
            if label == reference_label:
                continue
            try:
                val_f = float(metrics.get(key))
            except Exception:
                continue
            err = abs(val_f - ref_f)
            if best_err is None or err < best_err:
                best_err = err
                best_label = label
        if best_label is not None:
            best[key] = best_label
    return best


def format_kv_table_columns(
    data_by_label: Dict[str, Dict[str, Any]],
    *,
    title: str = "Output metrics",
    reference_label: Optional[str] = None,
    show_deltas: bool = False,
    highlight_best: bool = False,
    order: Optional[list[str]] = None,
    metric_keys: Optional[list[str]] = None,
) -> str:
    labels = list(data_by_label.keys())
    if not labels:
        return format_kv_table({}, title=title)
    filtered = {label: _filter_metrics_for_display(data_by_label[label]) for label in labels}
    metric_keys_all = list(filtered[labels[0]].keys())
    for label in labels[1:]:
        for key in filtered[label].keys():
            if key not in metric_keys_all:
                metric_keys_all.append(key)
    metric_keys_use = [
        key
        for key in metric_keys_all
        if not key.endswith("_spread") and key != "output_metrics_std_mode"
    ]
    metric_keys_use = _ordered_metric_keys(metric_keys_use, order)
    if metric_keys:
        selected = [key for key in metric_keys if key in metric_keys_use]
        if selected:
            metric_keys_use = selected

    best_by_metric = _best_labels_by_metric(filtered, reference_label=reference_label) if highlight_best else {}
    header_labels = [
        f"{label} (ref)" if reference_label and label == reference_label else label for label in labels
    ]
    header = "| Metric | " + " | ".join(header_labels) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(labels)) + " |"
    lines = [f"### {title}", header, sep]
    for key in metric_keys_use:
        row = []
        for label in labels:
            val = filtered[label].get(key, "")
            cell = _format_metric_cell(filtered[label], key)
            if show_deltas and reference_label and reference_label in filtered and label != reference_label:
                ref_val = filtered[reference_label].get(key, None)
                delta_str = _format_delta(val, ref_val)
                if delta_str:
                    cell = f"{cell} ({delta_str})"
            if highlight_best and best_by_metric.get(key) == label:
                cell = f"**{cell}**"
            row.append(cell)
        lines.append("| " + _format_metric_key(key) + " | " + " | ".join(row) + " |")
    return "\n".join(lines)


_OUTPUT_PARAM_KEYS = {
    "peak_window_ms",
    "drop_window_ms",
    "rebound_window_ms",
    "auc_window",
    "auc_window_start_ms",
    "auc_window_stop_ms",
    "auc_units",
    "pdp_mode",
    "pdp_window_ms",
    "t50_mode",
    "stim_start_ms",
    "stim_stop_ms",
    "baseline_ms",
    "baseline_mode",
    "baseline_center_ms",
    "baseline_time_ms",
    "baseline_window_start_ms",
    "baseline_window_stop_ms",
    "peak_time_ms",
    "tpeak10_time_ms",
    "drop_time_ms",
    "t50_time_ms",
    "rebound_time_ms",
    "tpeak10_value",
    "t50_value",
    "drop_center_ms",
    "drop_window_start_ms",
    "drop_window_stop_ms",
    "rebound_center_ms",
    "rebound_window_start_ms",
    "rebound_window_stop_ms",
    "norm_mode",
    "norm_window",
    "norm_scale",
    "avg_norm_scale",
    "output_metrics_std_mode",
}


def split_output_metrics(metrics: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    params: Dict[str, Any] = {}
    values: Dict[str, Any] = {}
    for key, val in metrics.items():
        if key in _OUTPUT_PARAM_KEYS:
            params[key] = val
            if key == "output_metrics_std_mode":
                values[key] = val
        else:
            values[key] = val
    return params, values


def split_output_metrics_columns(
    data_by_label: Dict[str, Dict[str, Any]],
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    params_by_label: Dict[str, Dict[str, Any]] = {}
    values_by_label: Dict[str, Dict[str, Any]] = {}
    for label, metrics in data_by_label.items():
        params, values = split_output_metrics(metrics)
        params_by_label[label] = params
        values_by_label[label] = values
    return params_by_label, values_by_label


def format_output_metrics_tables(
    metrics: Dict[str, Any],
    *,
    title: str = "Output metrics",
    show_params: bool = True,
    metric_keys: Optional[list[str]] = None,
) -> str:
    params, values = split_output_metrics(metrics)
    parts = []
    if values:
        parts.append(format_kv_table(values, title=title, order=_OUTPUT_METRIC_VALUE_ORDER, metric_keys=metric_keys))
    if show_params and params:
        parts.append(format_kv_table(params, title=f"{title} (params)"))
    return "\n\n".join(parts)


def format_output_metrics_tables_columns(
    data_by_label: Dict[str, Dict[str, Any]],
    *,
    title: str = "Output metrics",
    show_params: bool = True,
    reference_label: Optional[str] = None,
    show_deltas: bool = False,
    highlight_best: bool = False,
    metric_keys: Optional[list[str]] = None,
) -> str:
    params_by_label, values_by_label = split_output_metrics_columns(data_by_label)
    parts = []
    if any(values_by_label.values()):
        parts.append(
            format_kv_table_columns(
                values_by_label,
                title=title,
                reference_label=reference_label,
                show_deltas=show_deltas,
                highlight_best=highlight_best,
                order=_OUTPUT_METRIC_VALUE_ORDER,
                metric_keys=metric_keys,
            )
        )
    if show_params and any(params_by_label.values()):
        parts.append(format_kv_table_columns(params_by_label, title=f"{title} (params)"))
    return "\n\n".join(parts)
