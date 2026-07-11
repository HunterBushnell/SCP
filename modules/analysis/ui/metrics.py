"""Output metric tables and metric-distribution helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ._engine import (
    format_kv_table,
    format_kv_table_columns,
    format_output_metrics_tables,
    format_output_metrics_tables_columns,
    run_output_metric_distributions,
    run_output_metrics,
    split_output_metrics,
    split_output_metrics_columns,
    _safe_float,
)
from .state import get_selection_from_globals, output_opts_from_globals

def run_output_metrics_from_globals(g: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sel = get_selection_from_globals(g)
    opts = output_opts_from_globals(g)
    opts.update({
        "output_peak_window_ms": g.get("output_peak_window_ms"),
        "output_drop_window_ms": g.get("output_drop_window_ms"),
        "output_rebound_window_ms": g.get("output_rebound_window_ms"),
        "output_auc_window": g.get("output_auc_window"),
        "output_metric_mode": g.get("output_metric_mode"),
        "output_metric_window_ms": g.get("output_metric_window_ms"),
    })
    return run_output_metrics(
        sel,
        opts,
        save_analysis=bool(g.get("save_analysis", False)),
    )


def run_output_metric_distributions_from_globals(g: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sel = get_selection_from_globals(g)
    opts = output_opts_from_globals(g)
    opts.update({
        "output_peak_window_ms": g.get("output_peak_window_ms"),
        "output_drop_window_ms": g.get("output_drop_window_ms"),
        "output_rebound_window_ms": g.get("output_rebound_window_ms"),
        "output_auc_window": g.get("output_auc_window"),
        "output_metric_mode": g.get("output_metric_mode"),
        "output_metric_window_ms": g.get("output_metric_window_ms"),
    })
    ncols_raw = _safe_float(g.get("output_metrics_plot_ncols", 0))
    ncols_val = int(ncols_raw) if ncols_raw is not None else 0
    panel_size_raw = g.get("output_metrics_plot_panel_size", (4.8, 3.6))
    if isinstance(panel_size_raw, (list, tuple)) and len(panel_size_raw) >= 2:
        panel_size_val = (panel_size_raw[0], panel_size_raw[1])
    else:
        panel_size_val = (4.8, 3.6)
    bar_alpha_raw = _safe_float(g.get("output_metrics_plot_bar_alpha", 0.25))
    bar_alpha_val = float(bar_alpha_raw) if bar_alpha_raw is not None else 0.25
    payload = run_output_metric_distributions(
        sel,
        opts,
        metric_keys=list(g.get("output_metrics_plot_keys") or []),
        plot_style=str(g.get("output_metrics_plot_style", "box") or "box"),
        show_points=bool(g.get("output_metrics_plot_show_points", g.get("output_metrics_plot_jitter", True))),
        jitter_points=bool(g.get("output_metrics_plot_jitter", True)),
        show_error=bool(g.get("output_metrics_plot_show_error", False)),
        ncols=ncols_val,
        panel_size=panel_size_val,
        bar_alpha=bar_alpha_val,
        show_legend=bool(g.get("output_metrics_plot_show_legend", True)),
        legend_loc=str(g.get("output_metrics_plot_legend_loc", "best") or "best"),
        save_plots=bool(g.get("output_metrics_plot_save_plot", False)),
        save_data=bool(g.get("output_metrics_plot_save_data", False)),
        plots_dpi=int(g.get("plots_dpi", 150)),
        data_path=g.get("output_metrics_plot_data_path", ""),
        overwrite=bool(g.get("save_overwrite", False)),
    )
    g["_last_output_metric_dist_export"] = payload or {"rows": [], "mode": "single", "metric_keys": []}
    g["_last_output_metric_dist_selection"] = sel
    return payload

__all__ = [
    "format_kv_table",
    "format_kv_table_columns",
    "format_output_metrics_tables",
    "format_output_metrics_tables_columns",
    "run_output_metric_distributions",
    "run_output_metric_distributions_from_globals",
    "run_output_metrics",
    "run_output_metrics_from_globals",
    "split_output_metrics",
    "split_output_metrics_columns",
]
