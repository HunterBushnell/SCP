"""Notebook global-state helpers for Step 6 analysis UIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from .. import analysis
from ._engine import _parse_compare_list_paths, resolve_compare, resolve_single

def sync_common_from_globals(g: Dict[str, Any]) -> None:
    if g.get("save_plots_cb") is not None:
        g["save_plots"] = bool(g["save_plots_cb"].value)
    if g.get("save_analysis_cb") is not None:
        g["save_analysis"] = bool(g["save_analysis_cb"].value)


def get_selection_from_globals(g: Dict[str, Any]) -> Dict[str, Any]:
    use_widgets = bool(g.get("use_widgets", False))
    have_widgets = bool(g.get("_HAVE_WIDGETS", False))
    if use_widgets and have_widgets and g.get("cell_dd") is not None:
        cell = g["cell_dd"].value
        tunes = g["tunes_dd"].value
        model = g["model_dd"].value
        run_single = g.get("run_single_stem", "latest")
        run_a = g.get("run_compare_a", "latest")
        run_b = g.get("run_compare_b", "none")
        comp_a = g.get("compare_a_path", "")
        comp_b = g.get("compare_b_path", "")
        compare_list = list(g["compare_list_sel"].value) if g.get("compare_list_sel") is not None else []
        compare_paths_enabled = bool(
            g.get("compare_paths_cb").value if g.get("compare_paths_cb") is not None else g.get("compare_list_paths_enabled", True)
        )
        compare_list_paths = (
            _parse_compare_list_paths(
                g.get("compare_list_paths_txt").value if g.get("compare_list_paths_txt") is not None else ""
            )
            if compare_paths_enabled
            else []
        )
    else:
        cell = g.get("cell_name")
        tunes = g.get("tunes_dir")
        model = g.get("model_dir")
        run_single = g.get("run_single_stem")
        run_a = g.get("run_compare_a")
        run_b = g.get("run_compare_b")
        comp_a = g.get("compare_a_path", "")
        comp_b = g.get("compare_b_path", "")
        compare_list = g.get("compare_list", []) or []
        compare_paths_enabled = bool(g.get("compare_list_paths_enabled", True))
        compare_list_paths = g.get("compare_list_paths", []) or []
        if not compare_paths_enabled:
            compare_list_paths = []

    if compare_list_paths and not compare_list:
        compare_list = ["latest"]

    cells_root = g.get("CELLS_DIR")
    if cells_root is None:
        base_root = g.get("BASE_DIR")
        if base_root is None:
            base_root = analysis.find_scp_root(Path.cwd())
            g["BASE_DIR"] = base_root
        cells_root = Path(base_root) / "cells"
        g["CELLS_DIR"] = cells_root

    base_dir = Path(cells_root) / cell / tunes / model / "output_data"
    return {
        "cell": cell,
        "tunes": tunes,
        "model": model,
        "base": base_dir,
        "run_single": run_single,
        "run_a": run_a,
        "run_b": run_b,
        "run_a_path": comp_a,
        "run_b_path": comp_b,
        "compare_list": compare_list,
        "compare_list_paths": compare_list_paths,
        "compare_preset_path": g.get("compare_preset_path"),
    }


def output_opts_from_globals(g: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plot_outputs": bool(g.get("plot_outputs", False)),
        "plot_output_curve": bool(g.get("plot_output_curve", True)),
        "plot_spike_stats": g.get("plot_spike_stats"),
        "plot_raster": g.get("plot_raster"),
        "raster_style": g.get("raster_style"),
        "plot_window": g.get("plot_window"),
        "y_window": g.get("y_window"),
        "compare_output_layout": g.get("compare_output_layout"),
        "output_compare_figsize": g.get("output_compare_figsize"),
        "output_compare_panel_size": g.get("output_compare_panel_size"),
        "win_size": g.get("win_size"),
        "multi_plot_type": g.get("multi_plot_type"),
        "multi_shade_mode": g.get("multi_shade_mode"),
        "multi_norm_fr": g.get("multi_norm_fr"),
        "output_curve_mode": g.get("output_curve_mode"),
        "output_curve_plot_mode": g.get("output_curve_plot_mode", "rate"),
        "output_norm_mode": g.get("output_norm_mode"),
        "output_metric_window_ms": g.get("output_metric_window_ms"),
        "output_metric_mode": g.get("output_metric_mode"),
        "output_baseline_center_ms": g.get("output_baseline_center_ms"),
        "output_metric_window_markers": g.get("output_metric_window_markers", True),
        "output_plot_window_zero_origin": bool(g.get("output_plot_window_zero_origin", False)),
        "output_norm_window": g.get("output_norm_window"),
        "output_stim_start_ms": g.get("output_stim_start_ms"),
        "output_stim_stop_ms": g.get("output_stim_stop_ms"),
        "output_bin_ms": g.get("output_bin_ms"),
        "output_smooth_mode": g.get("output_smooth_mode"),
        "output_peak_window_ms": g.get("output_peak_window_ms"),
        "output_drop_window_ms": g.get("output_drop_window_ms"),
        "output_rebound_window_ms": g.get("output_rebound_window_ms"),
        "output_auc_window": g.get("output_auc_window"),
        "output_t50_mode": g.get("output_t50_mode", "absolute"),
        "output_show_metric_points": g.get("output_show_metric_points"),
        "output_metric_label_points": g.get("output_metric_label_points"),
        "output_metric_marker_size": g.get("output_metric_marker_size"),
        "output_linewidth": g.get("output_linewidth"),
        "output_stim_linewidth": g.get("output_stim_linewidth"),
        "output_metric_linewidth": g.get("output_metric_linewidth"),
        "output_metric_window_alpha": g.get("output_metric_window_alpha"),
        "output_shade_alpha": g.get("output_shade_alpha"),
        "output_metrics_std_mode": g.get("output_metrics_std_mode", "std"),
        "auto_plot_window_from_stim": bool(g.get("auto_plot_window_from_stim", False)),
        "plot_window_adjustment_ms": g.get("plot_window_adjustment_ms"),
        "save_overwrite": bool(g.get("save_overwrite", False)),
    }


def input_opts_from_globals(g: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plot_inputs_mean": g.get("plot_inputs_mean"),
        "plot_input_raster": g.get("plot_input_raster"),
        "show_input_std": g.get("show_input_std"),
        "input_source": g.get("input_source"),
        "input_std_mode": g.get("input_std_mode"),
        "input_groups": g.get("input_groups"),
        "input_bin_ms": g.get("input_bin_ms"),
        "input_smooth_ms": g.get("input_smooth_ms"),
        "input_raster_trial_idx": g.get("input_raster_trial_idx"),
        "input_raster_max_trains": g.get("input_raster_max_trains"),
        "input_raster_win_size": g.get("input_raster_win_size"),
        "input_raster_style": g.get("input_raster_style"),
        "input_plot_window": g.get("input_plot_window"),
        "input_legend_loc": g.get("input_legend_loc"),
        "compare_input_layout": g.get("compare_input_layout"),
        "compare_show_input_std": g.get("compare_show_input_std"),
        "input_linewidth": g.get("input_linewidth"),
        "input_shade_alpha": g.get("input_shade_alpha"),
        "input_output_linewidth": g.get("input_output_linewidth"),
        "input_raster_linewidth": g.get("input_raster_linewidth"),
        "input_stim_linewidth": g.get("input_stim_linewidth"),
        "auto_plot_window_from_stim": bool(g.get("auto_plot_window_from_stim", False)),
        "plot_window_adjustment_ms": g.get("plot_window_adjustment_ms"),
        "save_overwrite": bool(g.get("save_overwrite", False)),
    }


def resolve_single_from_globals(g: Dict[str, Any]) -> Tuple[Dict[str, Any], Path, Dict[str, Any]]:
    sel = get_selection_from_globals(g)
    run_dir, res = resolve_single(sel)
    return sel, run_dir, res


def resolve_compare_from_globals(
    g: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[Any], Optional[Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    sel = get_selection_from_globals(g)
    run_a, run_b, res_a, res_b = resolve_compare(sel)
    return sel, run_a, run_b, res_a, res_b

__all__ = [
    "sync_common_from_globals",
    "get_selection_from_globals",
    "output_opts_from_globals",
    "input_opts_from_globals",
    "resolve_single_from_globals",
    "resolve_compare_from_globals",
]
