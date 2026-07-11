"""Output-plot UI and execution helpers for Step 6 analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt

from .. import analysis
from ._engine import (
    HELP_OUTPUTS,
    _maybe_import_display,
    _maybe_import_widgets,
    _normalize_output_plot_export_type,
    _normalize_plot_data_format,
    _output_curve_plot_mode,
    _print_help,
    _resolve_plot_figure_target_path,
    _safe_float,
    _save_fig,
    _save_json,
    _save_plot_data_rows,
    _slug_token,
    resolve_single,
    run_output_plots,
)
from .state import get_selection_from_globals, output_opts_from_globals, sync_common_from_globals

def run_output_plots_from_globals(g: Dict[str, Any]) -> Dict[str, Any]:
    sel = get_selection_from_globals(g)
    payload = run_output_plots(
        sel,
        output_opts_from_globals(g),
        save_plots=bool(g.get("save_plots", False)),
        save_analysis=bool(g.get("save_analysis", False)),
        plots_dpi=int(g.get("plots_dpi", 150)),
    )
    g["_last_output_plot_export"] = payload or {"rows": [], "mode": "single", "run_label": ""}
    g["_last_output_plot_selection"] = sel
    return g["_last_output_plot_export"]


def save_output_plot_data_from_globals(g: Dict[str, Any]) -> Optional[Path]:
    payload = g.get("_last_output_plot_export") or {"rows": [], "mode": "single"}
    selection = g.get("_last_output_plot_selection") or get_selection_from_globals(g)
    return _save_plot_data_rows(
        payload.get("rows") or [],
        selection=selection,
        requested_path=g.get("output_plot_data_path", ""),
        figure_type="output",
        mode=str(payload.get("mode") or "single"),
        export_format=g.get("output_plot_data_format", "trace_rows"),
    )


def _collect_output_plot_figures(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    figures_raw = payload.get("figures") or []
    figures: list[Dict[str, Any]] = []
    for item in figures_raw:
        fig_obj = None
        plot_name = "output_plot"
        if isinstance(item, dict):
            fig_obj = item.get("figure")
            plot_name = str(item.get("plot_name") or plot_name)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            fig_obj = item[0]
            plot_name = str(item[1] or plot_name)
        elif item is not None:
            fig_obj = item
        if fig_obj is None:
            continue
        figures.append({"figure": analysis.resolve_figure(fig_obj), "plot_name": plot_name})
    return figures


def save_output_plot_figure_from_globals(
    g: Dict[str, Any],
    *,
    image_format: Any = "png",
) -> list[Path]:
    fmt = _normalize_output_plot_export_type(image_format)
    if fmt not in {"png", "svg"}:
        fmt = "png"
    payload = g.get("_last_output_plot_export") or {"rows": [], "mode": "single", "figures": []}
    figures = _collect_output_plot_figures(payload)
    if not figures:
        print("Output plot image not saved: no output plot figures are available. Run output plots first.")
        return []

    selection = g.get("_last_output_plot_selection") or get_selection_from_globals(g)
    out_path = _resolve_plot_figure_target_path(
        selection,
        g.get("output_plot_data_path", ""),
        figure_type="output",
        mode=str(payload.get("mode") or "single"),
        image_format=fmt,
    )
    overwrite = bool(g.get("save_overwrite", False))
    dpi = int(g.get("plots_dpi", 150))

    saved_paths: list[Path] = []
    if len(figures) == 1:
        only_fig = figures[0]
        saved_path = analysis.save_figure(
            only_fig["figure"],
            out_path,
            enabled=True,
            dpi=dpi,
            overwrite=overwrite,
        )
        if saved_path is not None:
            saved_paths.append(saved_path)
    else:
        stem = out_path.stem
        suffix = out_path.suffix
        parent = out_path.parent
        name_counts: Dict[str, int] = {}
        for idx, entry in enumerate(figures, start=1):
            token = _slug_token(str(entry.get("plot_name") or "plot"))
            count = name_counts.get(token, 0) + 1
            name_counts[token] = count
            token_suffix = token if count == 1 else f"{token}_{count}"
            item_path = parent / f"{stem}_{idx:02d}_{token_suffix}{suffix}"
            saved_path = analysis.save_figure(
                entry["figure"],
                item_path,
                enabled=True,
                dpi=dpi,
                overwrite=overwrite,
            )
            if saved_path is not None:
                saved_paths.append(saved_path)

    if not saved_paths:
        print(f"Output plot {fmt.upper()} export skipped.")
    elif len(saved_paths) == 1:
        print(f"Saved output plot {fmt.upper()}: {saved_paths[0]}")
    else:
        print(f"Saved {len(saved_paths)} output plot {fmt.upper()} files under {saved_paths[0].parent}")
    return saved_paths


def save_output_plot_export_from_globals(g: Dict[str, Any]) -> Any:
    export_type = _normalize_output_plot_export_type(g.get("output_plot_export_type", "csv"))
    if export_type == "csv":
        return save_output_plot_data_from_globals(g)
    return save_output_plot_figure_from_globals(g, image_format=export_type)


def run_spike_stats_from_globals(g: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sel = get_selection_from_globals(g)
    run_dir, res = resolve_single(sel)
    stats_single = analysis.summarize_spike_trials(res, plot=True, print_summary=False)
    _save_fig(
        plt.gcf(),
        analysis.plot_dir_for_run(run_dir) / "spike_stats.png",
        enabled=bool(g.get("save_plots", False)),
        dpi=int(g.get("plots_dpi", 150)),
        overwrite=bool(g.get("save_overwrite", False)),
    )
    _save_json(
        stats_single,
        analysis.analysis_dir_for_run(run_dir) / "spike_stats.json",
        enabled=bool(g.get("save_analysis", False)),
    )
    return stats_single


def build_outputs_ui(g: Dict[str, Any]) -> None:
    if not g.get("use_widgets", True):
        print("Widgets disabled (use_widgets=False). Run run_output_plots() manually.")
        return
    widgets = _maybe_import_widgets()
    display, _ = _maybe_import_display()
    if widgets is None or display is None:
        print("Widgets not enabled or ipywidgets unavailable. Run run_output_plots() manually.")
        return

    out_outputs = widgets.Output()

    outputs_full_cb = widgets.Checkbox(
        value=bool(g.get("plot_outputs", False)),
        description="Full output",
        indent=False,
    )
    output_curve_cb = widgets.Checkbox(
        value=bool(g.get("plot_output_curve", True)),
        description="Rate/ISI curve",
        indent=False,
    )
    output_spike_stats_cb = widgets.Checkbox(
        value=bool(g.get("plot_spike_stats", False)),
        description="Spike stats",
        indent=False,
    )
    outputs_raster_cb = widgets.Checkbox(
        value=bool(g.get("plot_raster", False)),
        description="Output raster",
        indent=False,
    )
    output_window_zero_cb = widgets.Checkbox(
        value=bool(g.get("output_plot_window_zero_origin", False)),
        description="x origin 0",
        indent=False,
    )
    outputs_style_dd = widgets.Dropdown(options=["dot", "line"], value=g.get("raster_style"), description="Raster")
    outputs_win_txt = widgets.FloatText(value=g.get("win_size"), description="Smooth ms")
    plot_window = g.get("plot_window")
    if not isinstance(plot_window, (list, tuple)) or len(plot_window) < 2:
        plot_window = (None, None)
    y_window = g.get("y_window")
    if not isinstance(y_window, (list, tuple)) or len(y_window) < 2:
        y_window = (None, None)
    window_start_txt = widgets.Text(value="" if plot_window[0] is None else str(plot_window[0]), description="x start")
    window_end_txt = widgets.Text(value="" if plot_window[1] is None else str(plot_window[1]), description="x stop")
    window_y_start_txt = widgets.Text(value="" if y_window[0] is None else str(y_window[0]), description="y min")
    window_y_end_txt = widgets.Text(value="" if y_window[1] is None else str(y_window[1]), description="y max")
    window_adjust_default = _safe_float(g.get("plot_window_adjustment_ms"))
    if window_adjust_default is None:
        window_adjust_default = 100.0
    auto_window_cb = widgets.Checkbox(
        value=bool(g.get("auto_plot_window_from_stim", False)),
        description="Auto x-window",
        indent=False,
    )
    window_adjust_txt = widgets.FloatText(
        value=float(window_adjust_default),
        description="Window ±ms",
    )
    output_stim_start_txt = widgets.Text(
        value="" if g.get("output_stim_start_ms") is None else str(g.get("output_stim_start_ms")),
        description="Stim start",
    )
    output_stim_stop_txt = widgets.Text(
        value="" if g.get("output_stim_stop_ms") is None else str(g.get("output_stim_stop_ms")),
        description="Stim stop",
    )

    output_curve_mode_dd = widgets.Dropdown(options=["raw", "normalized"], value=g.get("output_curve_mode"), description="Units")
    output_curve_plot_mode_dd = widgets.Dropdown(
        options=[
            ("Rate", "rate"),
            ("ISI", "isi"),
            ("Rate + ISI (stacked)", "rate_isi"),
        ],
        value=_output_curve_plot_mode(g),
        description="Curve type",
    )
    output_norm_mode_dd = widgets.Dropdown(options=["avg", "peak"], value=g.get("output_norm_mode"), description="Normalize by")
    outputs_norm_txt = widgets.Text(value="" if g.get("multi_norm_fr") is None else str(g.get("multi_norm_fr")), description="Fixed norm")
    output_bin_txt = widgets.Text(value="" if g.get("output_bin_ms") is None else str(g.get("output_bin_ms")), description="Bin ms")
    output_smooth_mode_dd = widgets.Dropdown(options=["causal", "center"], value=g.get("output_smooth_mode"), description="Smooth mode")

    shade_val = g.get("multi_shade_mode")
    outputs_shade_dd = widgets.Dropdown(options=["none", "sem", "std"], value="none" if shade_val is None else shade_val, description="Band")
    outputs_compare_layout_dd = widgets.Dropdown(options=["side-by-side", "stacked", "overlay"], value=g.get("compare_output_layout"), description="Layout")

    preset_path_default = g.get("compare_preset_path") or ""
    compare_preset_cb = widgets.Checkbox(
        value=bool(g.get("compare_preset_path")),
        description="Use compare preset",
        disabled=not bool(str(preset_path_default).strip()),
        indent=False,
    )

    outputs_btn = widgets.Button(description="Run output plots")
    outputs_btn.layout = widgets.Layout(width="160px", flex="0 0 auto")
    outputs_help_btn = widgets.Button(description="Help")
    outputs_help_btn.layout = widgets.Layout(width="80px", flex="0 0 auto")
    output_csv_path_txt = widgets.Text(
        value=str(g.get("output_plot_data_path", "") or ""),
        description="Save path",
        layout=widgets.Layout(width="60%"),
    )
    output_save_type_dd = widgets.Dropdown(
        options=[
            ("CSV data", "csv"),
            ("PNG image", "png"),
            ("SVG image", "svg"),
        ],
        value=_normalize_output_plot_export_type(g.get("output_plot_export_type", "csv")),
        description="Save type",
    )
    output_csv_format_dd = widgets.Dropdown(
        options=[
            ("Trace rows", "trace_rows"),
            ("Long rows", "long_rows"),
        ],
        value=_normalize_plot_data_format(g.get("output_plot_data_format", "trace_rows")),
        description="CSV format",
    )
    output_csv_auto_cb = widgets.Checkbox(
        value=bool(g.get("output_plot_data_auto_save", False)),
        description="Auto-save",
    )
    output_csv_btn = widgets.Button(description="Save plotted")
    output_csv_btn.layout = widgets.Layout(width="150px", flex="0 0 auto")

    def _sync_output_export_ui(*_):
        export_type = _normalize_output_plot_export_type(output_save_type_dd.value)
        is_csv = export_type == "csv"
        output_csv_format_dd.disabled = not is_csv
        output_csv_format_dd.layout.display = "" if is_csv else "none"
        if is_csv:
            output_csv_path_txt.description = "CSV path"
            output_csv_auto_cb.description = "Auto-save CSV"
            output_csv_btn.description = "Save plotted CSV"
        else:
            tag = export_type.upper()
            output_csv_path_txt.description = f"{tag} path"
            output_csv_auto_cb.description = f"Auto-save {tag}"
            output_csv_btn.description = f"Save plotted {tag}"

    output_save_type_dd.observe(_sync_output_export_ui, names="value")
    _sync_output_export_ui()

    def _on_outputs(_):
        sync_common_from_globals(g)
        g["compare_preset_path"] = preset_path_default if compare_preset_cb.value else None
        g["plot_outputs"] = bool(outputs_full_cb.value)
        g["plot_output_curve"] = bool(output_curve_cb.value)
        g["plot_spike_stats"] = bool(output_spike_stats_cb.value)
        g["plot_raster"] = outputs_raster_cb.value
        g["raster_style"] = outputs_style_dd.value
        g["win_size"] = float(outputs_win_txt.value)
        g["plot_window"] = (
            analysis.parse_optional_float(window_start_txt.value),
            analysis.parse_optional_float(window_end_txt.value),
        )
        g["y_window"] = (
            analysis.parse_optional_float(window_y_start_txt.value),
            analysis.parse_optional_float(window_y_end_txt.value),
        )
        g["output_plot_window_zero_origin"] = bool(output_window_zero_cb.value)
        g["auto_plot_window_from_stim"] = bool(auto_window_cb.value)
        g["plot_window_adjustment_ms"] = float(window_adjust_txt.value)
        g["output_stim_start_ms"] = analysis.parse_optional_float(output_stim_start_txt.value)
        g["output_stim_stop_ms"] = analysis.parse_optional_float(output_stim_stop_txt.value)
        g["output_curve_mode"] = output_curve_mode_dd.value
        g["output_curve_plot_mode"] = output_curve_plot_mode_dd.value
        g["output_norm_mode"] = output_norm_mode_dd.value
        g["output_bin_ms"] = analysis.parse_optional_float(output_bin_txt.value)
        g["output_smooth_mode"] = output_smooth_mode_dd.value
        shade_val_local = outputs_shade_dd.value
        g["multi_shade_mode"] = None if shade_val_local in ("none", "", None) else shade_val_local
        g["multi_norm_fr"] = analysis.parse_optional_float(outputs_norm_txt.value)
        g["compare_output_layout"] = outputs_compare_layout_dd.value
        g["output_plot_data_path"] = str(output_csv_path_txt.value or "").strip()
        g["output_plot_data_format"] = output_csv_format_dd.value
        g["output_plot_export_type"] = output_save_type_dd.value
        g["output_plot_data_auto_save"] = bool(output_csv_auto_cb.value)

        with out_outputs:
            out_outputs.clear_output()
            run_output_plots_from_globals(g)
            if g.get("output_plot_data_auto_save", False):
                save_output_plot_export_from_globals(g)

    def _on_save_output_csv(_):
        g["output_plot_data_path"] = str(output_csv_path_txt.value or "").strip()
        g["output_plot_data_format"] = output_csv_format_dd.value
        g["output_plot_export_type"] = output_save_type_dd.value
        with out_outputs:
            save_output_plot_export_from_globals(g)

    outputs_btn.on_click(_on_outputs)
    output_csv_btn.on_click(_on_save_output_csv)
    outputs_help_btn.on_click(lambda *_: _print_help(out_outputs, HELP_OUTPUTS))

    g["out_outputs"] = out_outputs
    g["_on_outputs"] = _on_outputs

    run_section = widgets.VBox([
        widgets.HTML("<b>Generate</b>"),
        widgets.HBox([outputs_full_cb, output_curve_cb, output_spike_stats_cb]),
        widgets.HBox([outputs_raster_cb, outputs_style_dd, outputs_win_txt]),
    ])
    window_section = widgets.VBox([
        widgets.HTML("<b>Display window</b>"),
        widgets.HBox([window_start_txt, window_end_txt, window_y_start_txt, window_y_end_txt]),
        widgets.HBox([auto_window_cb, window_adjust_txt, output_stim_start_txt, output_stim_stop_txt, output_window_zero_cb]),
    ])
    curve_section = widgets.VBox([
        widgets.HTML("<b>Curve options</b>"),
        widgets.HBox([output_curve_mode_dd, output_curve_plot_mode_dd, output_norm_mode_dd, outputs_norm_txt]),
        widgets.HBox([output_bin_txt, output_smooth_mode_dd]),
    ])
    compare_section = widgets.VBox([
        widgets.HTML("<b>Comparison</b>"),
        widgets.HBox([outputs_compare_layout_dd, outputs_shade_dd, compare_preset_cb]),
    ])
    export_section = widgets.VBox([
        widgets.HTML("<b>Export current plotted data or figure</b>"),
        widgets.HBox([output_csv_path_txt, output_save_type_dd, output_csv_format_dd]),
        widgets.HBox([output_csv_auto_cb, output_csv_btn]),
    ])

    sections = widgets.Accordion(children=[
        run_section,
        window_section,
        curve_section,
        compare_section,
        export_section,
    ])
    for idx, title in enumerate(["Run", "Window", "Curve", "Compare", "Export"]):
        sections.set_title(idx, title)
    sections.selected_index = 0

    display(
        widgets.VBox([
            widgets.HBox([outputs_btn, outputs_help_btn]),
            sections,
            out_outputs,
        ])
    )

    if g.get("auto_run_outputs"):
        _on_outputs(None)

__all__ = [
    "HELP_OUTPUTS",
    "build_outputs_ui",
    "output_opts_from_globals",
    "run_output_plots",
    "run_output_plots_from_globals",
    "run_spike_stats_from_globals",
    "save_output_plot_data_from_globals",
    "save_output_plot_export_from_globals",
    "save_output_plot_figure_from_globals",
]
