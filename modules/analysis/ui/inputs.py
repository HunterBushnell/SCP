"""Input-plot UI and execution helpers for Step 6 analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .. import analysis
from ._engine import (
    HELP_INPUTS,
    _maybe_import_display,
    _maybe_import_widgets,
    _normalize_plot_data_format,
    _print_help,
    _safe_float,
    _save_plot_data_rows,
    run_input_plots,
)
from .state import get_selection_from_globals, input_opts_from_globals, sync_common_from_globals

def run_input_plots_from_globals(g: Dict[str, Any]) -> Dict[str, Any]:
    sel = get_selection_from_globals(g)
    payload = run_input_plots(
        sel,
        input_opts_from_globals(g),
        save_plots=bool(g.get("save_plots", False)),
        save_analysis=bool(g.get("save_analysis", False)),
        plots_dpi=int(g.get("plots_dpi", 150)),
    )
    g["_last_input_plot_export"] = payload or {"rows": [], "mode": "single", "run_label": ""}
    g["_last_input_plot_selection"] = sel
    return g["_last_input_plot_export"]


def save_input_plot_data_from_globals(g: Dict[str, Any]) -> Optional[Path]:
    payload = g.get("_last_input_plot_export") or {"rows": [], "mode": "single"}
    selection = g.get("_last_input_plot_selection") or get_selection_from_globals(g)
    return _save_plot_data_rows(
        payload.get("rows") or [],
        selection=selection,
        requested_path=g.get("input_plot_data_path", ""),
        figure_type="input",
        mode=str(payload.get("mode") or "single"),
        export_format=g.get("input_plot_data_format", "trace_rows"),
    )


def build_inputs_ui(g: Dict[str, Any]) -> None:
    if not g.get("use_widgets", True):
        print("Widgets disabled (use_widgets=False). Run run_input_plots() manually.")
        return
    widgets = _maybe_import_widgets()
    display, _ = _maybe_import_display()
    if widgets is None or display is None:
        print("Widgets not enabled or ipywidgets unavailable. Run run_input_plots() manually.")
        return

    out_inputs = widgets.Output()

    inputs_mean_cb = widgets.Checkbox(
        value=bool(g.get("plot_inputs_mean", True)),
        description="Mean rates",
        indent=False,
    )
    inputs_raster_cb = widgets.Checkbox(
        value=bool(g.get("plot_input_raster", False)),
        description="Input raster",
        indent=False,
    )
    inputs_std_cb = widgets.Checkbox(
        value=bool(g.get("show_input_std", False)),
        description="Show band",
        indent=False,
    )

    input_groups = g.get("input_groups")
    if input_groups is None:
        input_groups_text = ""
    elif isinstance(input_groups, str):
        input_groups_text = input_groups
    else:
        input_groups_text = ",".join(str(group) for group in input_groups)
    inputs_groups_txt = widgets.Text(value=input_groups_text, description="Groups")
    inputs_bin_txt = widgets.Text(value="" if g.get("input_bin_ms") is None else str(g.get("input_bin_ms")), description="Bin ms")
    inputs_smooth_txt = widgets.Text(value="" if g.get("input_smooth_ms") is None else str(g.get("input_smooth_ms")), description="Smooth ms")
    inputs_source_dd = widgets.Dropdown(
        options=["auto", "saved", "stats"],
        value=g.get("input_source", "auto"),
        description="Input source",
    )
    inputs_std_mode_dd = widgets.Dropdown(
        options=["std", "sem"],
        value=g.get("input_std_mode", "std"),
        description="Band mode",
    )
    legend_options = [
        "best",
        "upper right",
        "upper left",
        "lower right",
        "lower left",
        "right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
        "none",
    ]
    legend_default = g.get("input_legend_loc") or "best"
    if legend_default not in legend_options:
        legend_options = [legend_default] + legend_options
    inputs_legend_dd = widgets.Dropdown(
        options=legend_options,
        value=legend_default,
        description="Legend",
    )

    raster_trial_default = _safe_float(g.get("input_raster_trial_idx"))
    raster_max_default = _safe_float(g.get("input_raster_max_trains"))
    raster_win_default = _safe_float(g.get("input_raster_win_size"))
    raster_trial_txt = widgets.IntText(
        value=max(0, int(raster_trial_default)) if raster_trial_default is not None else 0,
        description="Trial",
    )
    raster_max_txt = widgets.IntText(
        value=max(1, int(raster_max_default)) if raster_max_default is not None else 200,
        description="Max trains",
    )
    raster_win_txt = widgets.FloatText(
        value=float(raster_win_default) if raster_win_default is not None else 25.0,
        description="Raster bin",
    )
    raster_style_dd = widgets.Dropdown(options=["dot", "line"], value=g.get("input_raster_style"), description="Raster")

    input_window = g.get("input_plot_window")
    if not isinstance(input_window, (list, tuple)) or len(input_window) < 2:
        input_window = (None, None)
    input_window_start_txt = widgets.Text(value="" if input_window[0] is None else str(input_window[0]), description="x start")
    input_window_end_txt = widgets.Text(value="" if input_window[1] is None else str(input_window[1]), description="x stop")
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

    compare_layout_dd = widgets.Dropdown(options=["side-by-side", "stacked", "overlay"], value=g.get("compare_input_layout"), description="Layout")
    compare_std_cb = widgets.Checkbox(
        value=bool(g.get("compare_show_input_std", False)),
        description="Compare band",
        indent=False,
    )

    inputs_btn = widgets.Button(description="Run input plots")
    inputs_btn.layout = widgets.Layout(width="160px", flex="0 0 auto")
    inputs_help_btn = widgets.Button(description="Help")
    inputs_help_btn.layout = widgets.Layout(width="80px", flex="0 0 auto")
    input_csv_path_txt = widgets.Text(
        value=str(g.get("input_plot_data_path", "") or ""),
        description="CSV path",
        layout=widgets.Layout(width="60%"),
    )
    input_csv_format_dd = widgets.Dropdown(
        options=[
            ("Trace rows", "trace_rows"),
            ("Long rows", "long_rows"),
        ],
        value=_normalize_plot_data_format(g.get("input_plot_data_format", "trace_rows")),
        description="Format",
    )
    input_csv_auto_cb = widgets.Checkbox(
        value=bool(g.get("input_plot_data_auto_save", False)),
        description="Auto-save CSV",
    )
    input_csv_btn = widgets.Button(description="Save plotted CSV")
    input_csv_btn.layout = widgets.Layout(width="150px", flex="0 0 auto")

    def _on_inputs(_):
        sync_common_from_globals(g)
        g["plot_inputs_mean"] = inputs_mean_cb.value
        g["plot_input_raster"] = inputs_raster_cb.value
        g["show_input_std"] = inputs_std_cb.value
        g["input_source"] = inputs_source_dd.value
        g["input_std_mode"] = inputs_std_mode_dd.value
        g["input_groups"] = analysis.parse_groups(inputs_groups_txt.value)
        g["input_bin_ms"] = analysis.parse_optional_float(inputs_bin_txt.value)
        g["input_smooth_ms"] = analysis.parse_optional_float(inputs_smooth_txt.value)
        g["input_raster_trial_idx"] = int(raster_trial_txt.value)
        g["input_raster_max_trains"] = int(raster_max_txt.value)
        g["input_raster_win_size"] = float(raster_win_txt.value)
        g["input_raster_style"] = raster_style_dd.value
        g["input_plot_window"] = (
            analysis.parse_optional_float(input_window_start_txt.value),
            analysis.parse_optional_float(input_window_end_txt.value),
        )
        g["auto_plot_window_from_stim"] = bool(auto_window_cb.value)
        g["plot_window_adjustment_ms"] = float(window_adjust_txt.value)
        g["input_legend_loc"] = inputs_legend_dd.value
        g["compare_input_layout"] = compare_layout_dd.value
        g["compare_show_input_std"] = compare_std_cb.value
        g["input_plot_data_path"] = str(input_csv_path_txt.value or "").strip()
        g["input_plot_data_format"] = input_csv_format_dd.value
        g["input_plot_data_auto_save"] = bool(input_csv_auto_cb.value)

        with out_inputs:
            out_inputs.clear_output()
            run_input_plots_from_globals(g)
            if g.get("input_plot_data_auto_save", False):
                save_input_plot_data_from_globals(g)

    def _on_save_input_csv(_):
        g["input_plot_data_path"] = str(input_csv_path_txt.value or "").strip()
        g["input_plot_data_format"] = input_csv_format_dd.value
        with out_inputs:
            save_input_plot_data_from_globals(g)

    inputs_btn.on_click(_on_inputs)
    input_csv_btn.on_click(_on_save_input_csv)
    inputs_help_btn.on_click(lambda *_: _print_help(out_inputs, HELP_INPUTS))

    g["out_inputs"] = out_inputs
    g["_on_inputs"] = _on_inputs

    def _hint(text: str) -> Any:
        return widgets.HTML(
            f"<div style='font-size: 90%; color: #555; margin: 0 0 4px 0;'>{text}</div>"
        )

    run_section = widgets.VBox([
        _hint("Choose which input figures to generate. Mean rates work for single or compare views; rasters require saved input spikes and are single-run only."),
        widgets.HBox([inputs_mean_cb, inputs_raster_cb, inputs_std_cb]),
        widgets.HBox([inputs_source_dd, inputs_std_mode_dd, inputs_legend_dd]),
    ])
    groups_section = widgets.VBox([
        _hint("Leave Groups blank for all synapse groups, or enter names like pn_exc,vip_inh. Bin/smooth control input-rate summaries."),
        widgets.HBox([inputs_groups_txt, inputs_bin_txt, inputs_smooth_txt]),
    ])
    raster_section = widgets.VBox([
        _hint("Raster options only affect saved input spike-train rasters. Max trains limits heavy plots."),
        widgets.HBox([raster_trial_txt, raster_max_txt, raster_win_txt, raster_style_dd]),
    ])
    window_section = widgets.VBox([
        _hint("Use explicit x start/stop, or enable auto x-window to use the simulation stimulus window plus padding."),
        widgets.HBox([input_window_start_txt, input_window_end_txt, auto_window_cb, window_adjust_txt]),
    ])
    compare_section = widgets.VBox([
        _hint("Compare settings apply when multiple runs are selected in the Selection UI."),
        widgets.HBox([compare_layout_dd, compare_std_cb]),
    ])
    export_section = widgets.VBox([
        _hint("Save the currently plotted input data after running this UI. This exports plotted traces, not raw simulation files."),
        widgets.HBox([input_csv_path_txt, input_csv_format_dd]),
        widgets.HBox([input_csv_auto_cb, input_csv_btn]),
    ])

    sections = widgets.Accordion(children=[
        run_section,
        groups_section,
        raster_section,
        window_section,
        compare_section,
        export_section,
    ])
    for idx, title in enumerate(["Run", "Groups", "Raster", "Window", "Compare", "Export"]):
        sections.set_title(idx, title)
    sections.selected_index = 0

    display(
        widgets.VBox([
            widgets.HBox([inputs_btn, inputs_help_btn]),
            sections,
            out_inputs,
        ])
    )

    if g.get("auto_run_inputs"):
        _on_inputs(None)

__all__ = [
    "HELP_INPUTS",
    "build_inputs_ui",
    "input_opts_from_globals",
    "run_input_plots",
    "run_input_plots_from_globals",
    "save_input_plot_data_from_globals",
]
