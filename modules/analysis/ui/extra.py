"""Extra Analysis UI and advanced diagnostic helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import json
import matplotlib.pyplot as plt

from modules import run_sim
from modules.input_generation import inputs
from modules.input_generation import sampling as input_sampling
from .. import analysis, plotting
from ._engine import (
    HELP_EXTRA,
    _OUTPUT_METRIC_PLOT_DEFAULT_KEYS,
    _OUTPUT_METRIC_VALUE_ORDER,
    _coerce_output_metric_plot_style,
    _coerce_run_path,
    _compare_list_entries,
    _is_curve_path,
    _maybe_import_display,
    _maybe_import_widgets,
    _parse_compare_list_item,
    _print_help,
    _safe_float,
    _save_fig,
    _save_json,
    _selected_config_compare_runs,
    format_output_metrics_tables,
    format_output_metrics_tables_columns,
    show_md,
)
from .metrics import run_output_metric_distributions_from_globals, run_output_metrics_from_globals
from .state import get_selection_from_globals, resolve_compare_from_globals, resolve_single_from_globals

def build_extra_ui(g: Dict[str, Any]) -> None:
    if not g.get("use_widgets", True):
        print("Widgets disabled (use_widgets=False).")
        return
    widgets = _maybe_import_widgets()
    display, _ = _maybe_import_display()
    if widgets is None or display is None:
        print("Widgets not enabled or ipywidgets unavailable.")
        return

    extra_mode_options = [
        ("Output metrics (table)", "output_metrics"),
        ("Compare configs (restore-style)", "compare_configs"),
        ("Input sampling (preview)", "input_sampling"),
        ("Synapse plots", "synapse_plots"),
        ("Snapshot compare", "snapshot_compare"),
        ("Single-run tables", "single_tables"),
        ("IClamp analysis", "iclamp"),
    ]
    extra_mode_values = {value for _, value in extra_mode_options}
    extra_mode_value = g.get("extra_mode", "output_metrics")
    if extra_mode_value not in extra_mode_values:
        extra_mode_value = "output_metrics"
    mode_dd = widgets.Dropdown(
        options=extra_mode_options,
        value=extra_mode_value,
        description="Extra mode",
    )

    apply_choices = list(analysis.RESTORE_CONFIG_APPLY_CHOICES)
    apply_defaults = [str(v) for v in (g.get("extra_compare_apply") or apply_choices)]
    apply_defaults = [v for v in apply_defaults if v in apply_choices]
    if not apply_defaults:
        apply_defaults = apply_choices

    cfg_sim_cb = widgets.Checkbox(
        value=bool("sim_config" in apply_defaults),
        description="sim_config",
    )
    cfg_cell_cb = widgets.Checkbox(
        value=bool("cell_config" in apply_defaults),
        description="cell_config",
    )
    cfg_geom_cb = widgets.Checkbox(
        value=bool("geometry" in apply_defaults),
        description="geometry",
    )
    cfg_syn_cb = widgets.Checkbox(
        value=bool("syn_config" in apply_defaults),
        description="syn_config",
    )
    cfg_syn_groups_cb = widgets.Checkbox(
        value=bool("syn_groups" in apply_defaults),
        description="syn_groups",
    )
    cfg_fit_cb = widgets.Checkbox(
        value=bool("fit_json" in apply_defaults),
        description="fit_json",
    )
    cfg_diff_cb = widgets.Checkbox(
        value=bool(g.get("extra_compare_diff_only", True)),
        description="Diff only",
    )
    cfg_syn_groups_txt = widgets.Text(
        value=str(g.get("extra_compare_syn_groups_selector", "all") or "all"),
        description="Syn groups",
        layout=widgets.Layout(width="280px"),
    )

    snap_diff_cb = widgets.Checkbox(
        value=bool(g.get("extra_snapshot_diff_only", True)),
        description="Diff only",
    )
    snap_save_cb = widgets.Checkbox(
        value=bool(g.get("save_snapshot_compare_table", False)),
        description="Save table",
    )
    snap_scope_dd = widgets.Dropdown(
        options=["full", "meta", "snapshot"],
        value=g.get("snapshot_compare_scope", "full"),
        description="Scope",
    )
    snap_fmt_dd = widgets.Dropdown(
        options=["csv", "xlsx", "both"],
        value=g.get("snapshot_compare_format", "csv"),
        description="Format",
    )
    snap_depth_txt = widgets.Text(
        value=str(g.get("snapshot_compare_max_depth", 60)),
        description="Max depth",
        layout=widgets.Layout(width="200px"),
    )
    snap_list_txt = widgets.Text(
        value=str(g.get("snapshot_compare_max_list_items", 200)),
        description="Max list items",
        layout=widgets.Layout(width="220px"),
    )

    run_btn = widgets.Button(description="Run extra analysis")
    run_btn.layout = widgets.Layout(width="160px", flex="0 0 auto")
    extra_help_btn = widgets.Button(description="Help")
    extra_help_btn.layout = widgets.Layout(width="80px", flex="0 0 auto")
    out_extra = widgets.Output()

    cfg_box = widgets.VBox([
        widgets.HBox([cfg_sim_cb, cfg_cell_cb, cfg_geom_cb, cfg_syn_cb]),
        widgets.HBox([cfg_syn_groups_cb, cfg_fit_cb, cfg_diff_cb, cfg_syn_groups_txt]),
    ])
    table_cell_cb = widgets.Checkbox(
        value=bool(g.get("extra_cell_tables", True)),
        description="Cell sections",
    )
    table_geom_cb = widgets.Checkbox(
        value=bool(g.get("extra_geometry_tables", True)),
        description="Geometry",
    )
    table_syn_cb = widgets.Checkbox(
        value=bool(g.get("extra_synapse_tables", True)),
        description="Synapses",
    )
    table_rec_cb = widgets.Checkbox(
        value=bool(g.get("extra_recording_tables", True)),
        description="Recordings",
    )
    tables_box = widgets.VBox([
        widgets.HTML(
            "<div style='font-size: 90%; color: #555;'>Choose which single-run summary tables to generate. "
            "Cell and geometry tables may load the cell model.</div>"
        ),
        widgets.HBox([table_cell_cb, table_geom_cb, table_syn_cb, table_rec_cb]),
    ])
    def _initial_metrics_ref_options() -> list[str]:
        sel = get_selection_from_globals(g)
        entries = _compare_list_entries(sel)
        labels: list[str] = []
        for item in entries:
            spec = _parse_compare_list_item(item)
            path_raw = spec.get("path")
            path = _coerce_run_path(path_raw, sel.get("base"))
            if path is None or not path.exists():
                continue
            label = Path(path).stem if _is_curve_path(Path(path)) else analysis.run_label(path)
            if label not in labels:
                labels.append(label)
        return ["(none)"] + labels

    metrics_ref_options = _initial_metrics_ref_options()
    metrics_ref_value = g.get("output_metrics_ref_label") or "(none)"
    if metrics_ref_value not in metrics_ref_options:
        metrics_ref_value = "(none)"
    metrics_ref_dd = widgets.Dropdown(
        options=metrics_ref_options,
        value=metrics_ref_value,
        description="Reference",
        layout=widgets.Layout(width="40%"),
    )
    metrics_show_params_cb = widgets.Checkbox(
        value=bool(g.get("output_metrics_show_params", True)),
        description="Show params",
    )
    metrics_std_mode_val = str(g.get("output_metrics_std_mode", "std") or "std").lower()
    if metrics_std_mode_val not in ("std", "sem"):
        metrics_std_mode_val = "std"
    metrics_std_mode_dd = widgets.Dropdown(
        options=["std", "sem"],
        value=metrics_std_mode_val,
        description="Spread",
        layout=widgets.Layout(width="220px"),
    )
    metrics_show_delta_cb = widgets.Checkbox(
        value=bool(g.get("output_metrics_show_delta", True)),
        description="Show deltas",
    )
    metrics_highlight_cb = widgets.Checkbox(
        value=bool(g.get("output_metrics_highlight_best", True)),
        description="Highlight best",
    )
    metric_plot_options = [k for k in _OUTPUT_METRIC_VALUE_ORDER if k != "output_metrics_n_trials"]
    metrics_plot_default = list(g.get("output_metrics_plot_keys") or _OUTPUT_METRIC_PLOT_DEFAULT_KEYS)
    metrics_plot_default = [k for k in metrics_plot_default if k in metric_plot_options]
    if not metrics_plot_default:
        metrics_plot_default = [k for k in _OUTPUT_METRIC_PLOT_DEFAULT_KEYS if k in metric_plot_options]
    metrics_plot_sel = widgets.SelectMultiple(
        options=metric_plot_options,
        value=tuple(metrics_plot_default),
        description="Plot metrics",
        rows=min(8, max(6, len(metric_plot_options))),
        layout=widgets.Layout(width="96%"),
    )
    metrics_plot_btn = widgets.Button(description="Plot metric dist")
    metrics_plot_btn.layout = widgets.Layout(width="150px", flex="0 0 auto")
    metrics_plot_style_val = _coerce_output_metric_plot_style(g.get("output_metrics_plot_style", "box"))
    metrics_plot_style_dd = widgets.Dropdown(
        options=[("Box/whisker", "box"), ("Bar (mean)", "bar")],
        value=metrics_plot_style_val,
        description="Style",
        layout=widgets.Layout(width="220px"),
    )
    metrics_plot_show_points_cb = widgets.Checkbox(
        value=bool(g.get("output_metrics_plot_show_points", g.get("output_metrics_plot_jitter", True))),
        description="Trial points",
    )
    metrics_plot_jitter_cb = widgets.Checkbox(
        value=bool(g.get("output_metrics_plot_jitter", True)),
        description="Jitter points",
    )
    metrics_plot_show_error_cb = widgets.Checkbox(
        value=bool(g.get("output_metrics_plot_show_error", False)),
        description="Error bars",
    )
    ncols_default = _safe_float(g.get("output_metrics_plot_ncols", 0))
    metrics_plot_ncols_txt = widgets.IntText(
        value=int(ncols_default) if ncols_default is not None else 0,
        description="Cols (0=auto)",
        layout=widgets.Layout(width="170px"),
    )
    panel_size_raw = g.get("output_metrics_plot_panel_size", (4.8, 3.6))
    if isinstance(panel_size_raw, (list, tuple)) and len(panel_size_raw) >= 2:
        panel_w_default = _safe_float(panel_size_raw[0])
        panel_h_default = _safe_float(panel_size_raw[1])
    else:
        panel_w_default = None
        panel_h_default = None
    if panel_w_default is None:
        panel_w_default = 4.8
    if panel_h_default is None:
        panel_h_default = 3.6
    metrics_plot_panel_w_txt = widgets.FloatText(
        value=float(panel_w_default),
        description="Panel w",
        layout=widgets.Layout(width="150px"),
    )
    metrics_plot_panel_h_txt = widgets.FloatText(
        value=float(panel_h_default),
        description="Panel h",
        layout=widgets.Layout(width="150px"),
    )
    bar_alpha_default = _safe_float(g.get("output_metrics_plot_bar_alpha", 0.25))
    if bar_alpha_default is None:
        bar_alpha_default = 0.25
    metrics_plot_bar_alpha_txt = widgets.FloatText(
        value=float(bar_alpha_default),
        description="Bar alpha",
        layout=widgets.Layout(width="160px"),
    )
    metrics_plot_show_legend_cb = widgets.Checkbox(
        value=bool(g.get("output_metrics_plot_show_legend", True)),
        description="Legend",
    )
    metrics_plot_legend_loc_txt = widgets.Text(
        value=str(g.get("output_metrics_plot_legend_loc", "best") or "best"),
        description="Legend loc",
        layout=widgets.Layout(width="220px"),
    )
    metrics_plot_save_plot_cb = widgets.Checkbox(
        value=bool(g.get("output_metrics_plot_save_plot", False)),
        description="Save plot",
    )
    metrics_plot_save_data_cb = widgets.Checkbox(
        value=bool(g.get("output_metrics_plot_save_data", False)),
        description="Save CSV",
    )
    metrics_plot_data_path_txt = widgets.Text(
        value=str(g.get("output_metrics_plot_data_path", "") or ""),
        description="CSV path",
        layout=widgets.Layout(width="70%"),
    )
    metrics_use_plot_keys_cb = widgets.Checkbox(
        value=bool(g.get("output_metrics_use_plot_keys_for_tables", True)),
        description="Table uses Plot metrics",
    )

    def _toggle_metric_plot_controls(*_):
        metrics_plot_jitter_cb.disabled = not metrics_plot_show_points_cb.value
        metrics_plot_bar_alpha_txt.disabled = metrics_plot_style_dd.value != "bar"
        metrics_plot_legend_loc_txt.disabled = not metrics_plot_show_legend_cb.value

    _toggle_metric_plot_controls()
    metrics_plot_show_points_cb.observe(_toggle_metric_plot_controls, names="value")
    metrics_plot_style_dd.observe(_toggle_metric_plot_controls, names="value")
    metrics_plot_show_legend_cb.observe(_toggle_metric_plot_controls, names="value")

    metrics_box = widgets.VBox([
        widgets.HBox([metrics_ref_dd, metrics_std_mode_dd, metrics_show_params_cb]),
        widgets.HBox([metrics_show_delta_cb, metrics_highlight_cb, metrics_use_plot_keys_cb]),
        widgets.HBox([
            metrics_plot_btn,
            metrics_plot_style_dd,
            metrics_plot_show_points_cb,
            metrics_plot_jitter_cb,
            metrics_plot_show_error_cb,
            metrics_plot_save_plot_cb,
            metrics_plot_save_data_cb,
        ]),
        widgets.HBox([
            metrics_plot_ncols_txt,
            metrics_plot_panel_w_txt,
            metrics_plot_panel_h_txt,
            metrics_plot_bar_alpha_txt,
            metrics_plot_show_legend_cb,
            metrics_plot_legend_loc_txt,
        ]),
        metrics_plot_sel,
        metrics_plot_data_path_txt,
    ])

    def _initial_sample_runs() -> list[str]:
        sel = get_selection_from_globals(g)
        base = sel.get("base")
        try:
            names = [analysis.run_label(p) for p in analysis.collect_run_candidates(base)]
        except Exception:
            names = []
        return names or ["latest"]

    sample_source_value = g.get("input_sample_source", "selection")
    sample_run_options = _initial_sample_runs()
    sample_run_value = g.get("input_sample_run", "latest")
    if sample_run_value not in sample_run_options:
        sample_run_value = sample_run_options[0]
    sample_path_value = str(g.get("input_sample_path") or "")

    sample_source_dd = widgets.Dropdown(
        options=[
            ("Selection (cell/tune/model)", "selection"),
            ("Output run", "run"),
            ("Path", "path"),
        ],
        value=sample_source_value,
        description="Source",
        layout=widgets.Layout(width="40%"),
    )
    sample_run_dd = widgets.Dropdown(
        options=sample_run_options,
        value=sample_run_value,
        description="Run",
        layout=widgets.Layout(width="40%"),
    )
    sample_path_txt = widgets.Text(
        value=sample_path_value,
        description="Path",
        layout=widgets.Layout(width="70%"),
    )
    def _initial_sampling_path() -> Optional[Path]:
        sel = get_selection_from_globals(g)
        if sample_source_value == "run":
            try:
                return Path(analysis.resolve_run(sel["base"], sample_run_value))
            except Exception:
                return None
        if sample_source_value == "path":
            raw = sample_path_value.strip()
            return Path(raw).expanduser() if raw else None
        return (g.get("CELLS_DIR") / sel["cell"] / sel["tunes"] / sel["model"]).resolve()

    sample_groups_options = _list_groups_for_sampling(_initial_sampling_path())
    prev_groups = list(g.get("input_sample_groups") or [])
    if prev_groups:
        selected_groups = [gname for gname in prev_groups if gname in sample_groups_options]
    else:
        selected_groups = sample_groups_options
    sample_groups_sel = widgets.SelectMultiple(
        options=sample_groups_options,
        value=tuple(selected_groups),
        description="Groups",
        rows=6,
    )
    sample_runs_txt = widgets.IntText(
        value=int(g.get("input_sample_runs", 200)),
        description="Runs",
    )
    sample_bin_txt = widgets.Text(
        value="" if g.get("input_sample_bin_ms") is None else str(g.get("input_sample_bin_ms")),
        description="Bin ms",
        layout=widgets.Layout(width="160px"),
    )
    sample_seed_txt = widgets.Text(
        value="" if g.get("input_sample_seed") is None else str(g.get("input_sample_seed")),
        description="Seed",
        layout=widgets.Layout(width="160px"),
    )
    sample_std_cb = widgets.Checkbox(
        value=bool(g.get("input_sample_show_std", True)),
        description="Show std",
    )
    sample_ref_cb = widgets.Checkbox(
        value=bool(g.get("input_sample_show_ref", True)),
        description="Show ref",
    )
    sample_box = widgets.VBox([
        widgets.HBox([sample_source_dd, sample_run_dd]),
        widgets.HBox([sample_path_txt]),
        widgets.HBox([sample_groups_sel, widgets.VBox([sample_runs_txt, sample_bin_txt, sample_seed_txt, sample_std_cb, sample_ref_cb])]),
    ])

    syn_groups_value = g.get("extra_synapse_groups")
    if syn_groups_value is None:
        syn_groups_text = ""
    elif isinstance(syn_groups_value, str):
        syn_groups_text = syn_groups_value
    else:
        syn_groups_text = ",".join(str(group) for group in syn_groups_value)
    syn_trial_default = _safe_float(g.get("extra_synapse_trial_idx", 0))
    syn_weight_bin_default = _safe_float(g.get("extra_synapse_weight_bin", 0.1))
    syn_dist_bin_default = _safe_float(g.get("extra_synapse_distance_bin", 25.0))
    syn_plot_type = str(g.get("extra_synapse_plot_type", "hist") or "hist")
    if syn_plot_type not in ("hist", "line", "both"):
        syn_plot_type = "hist"
    syn_table_cb = widgets.Checkbox(
        value=bool(g.get("extra_synapse_show_table", True)),
        description="Summary table",
    )
    syn_weight_cb = widgets.Checkbox(
        value=bool(g.get("extra_synapse_weight_plot", True)),
        description="Weight hist",
    )
    syn_distance_cb = widgets.Checkbox(
        value=bool(g.get("extra_synapse_distance_plot", True)),
        description="Distance hist",
    )
    syn_scatter_cb = widgets.Checkbox(
        value=bool(g.get("extra_synapse_scatter_plot", True)),
        description="Weight x distance",
    )
    syn_density_cb = widgets.Checkbox(
        value=bool(g.get("extra_synapse_density", False)),
        description="Distance density",
    )
    syn_groups_txt = widgets.Text(
        value=syn_groups_text,
        description="Groups",
        layout=widgets.Layout(width="60%"),
    )
    syn_trial_txt = widgets.IntText(
        value=max(0, int(syn_trial_default)) if syn_trial_default is not None else 0,
        description="Trial",
    )
    syn_weight_bin_txt = widgets.FloatText(
        value=float(syn_weight_bin_default) if syn_weight_bin_default is not None else 0.1,
        description="Weight bin",
    )
    syn_dist_bin_txt = widgets.FloatText(
        value=float(syn_dist_bin_default) if syn_dist_bin_default is not None else 25.0,
        description="Distance bin",
    )
    syn_plot_type_dd = widgets.Dropdown(
        options=["hist", "line", "both"],
        value=syn_plot_type,
        description="Plot type",
    )
    syn_box = widgets.VBox([
        widgets.HTML(
            "<div style='font-size: 90%; color: #555;'>Uses saved Step 5 synapse records. "
            "If records are missing, rerun Step 5 with synapse-record saving enabled.</div>"
        ),
        widgets.HBox([syn_table_cb, syn_weight_cb, syn_distance_cb, syn_scatter_cb, syn_density_cb]),
        widgets.HBox([syn_groups_txt, syn_trial_txt]),
        widgets.HBox([syn_weight_bin_txt, syn_dist_bin_txt, syn_plot_type_dd]),
    ])

    snap_box = widgets.VBox([
        widgets.HBox([snap_diff_cb, snap_save_cb]),
        widgets.HBox([snap_scope_dd, snap_fmt_dd]),
        widgets.HBox([snap_depth_txt, snap_list_txt]),
    ])

    def _toggle_sections(*_):
        mode = mode_dd.value
        cfg_box.layout.display = "flex" if mode == "compare_configs" else "none"
        metrics_box.layout.display = "flex" if mode == "output_metrics" else "none"
        sample_box.layout.display = "flex" if mode == "input_sampling" else "none"
        syn_box.layout.display = "flex" if mode == "synapse_plots" else "none"
        snap_box.layout.display = "flex" if mode == "snapshot_compare" else "none"
        tables_box.layout.display = "flex" if mode == "single_tables" else "none"

    def _resolve_sampling_path() -> Optional[Path]:
        sel = get_selection_from_globals(g)
        source = sample_source_dd.value
        if source == "run":
            run_label = sample_run_dd.value or "latest"
            try:
                return Path(analysis.resolve_run(sel["base"], run_label))
            except Exception:
                return None
        if source == "path":
            raw = sample_path_txt.value.strip()
            return Path(raw).expanduser() if raw else None
        return (g.get("CELLS_DIR") / sel["cell"] / sel["tunes"] / sel["model"]).resolve()

    def _refresh_sample_runs():
        sel = get_selection_from_globals(g)
        base = sel.get("base")
        try:
            names = [analysis.run_label(p) for p in analysis.collect_run_candidates(base)]
        except Exception:
            names = []
        if not names:
            names = ["latest"]
        sample_run_dd.options = names
        if sample_run_dd.value not in names:
            sample_run_dd.value = names[0]

    def _refresh_sample_groups():
        path = _resolve_sampling_path()
        groups = _list_groups_for_sampling(path) if path is not None else []
        sample_groups_sel.options = groups
        prev = list(sample_groups_sel.value)
        if prev:
            selected = [gname for gname in prev if gname in groups]
        else:
            selected = groups
        sample_groups_sel.value = tuple(selected)

    def _refresh_metrics_refs():
        sel = get_selection_from_globals(g)
        entries = _compare_list_entries(sel)
        labels: list[str] = []
        for item in entries:
            spec = _parse_compare_list_item(item)
            path_raw = spec.get("path")
            path = _coerce_run_path(path_raw, sel.get("base"))
            if path is None:
                continue
            if not path.exists():
                continue
            label = Path(path).stem if _is_curve_path(Path(path)) else analysis.run_label(path)
            if label not in labels:
                labels.append(label)
        options = ["(none)"] + labels
        metrics_ref_dd.options = options
        if metrics_ref_dd.value not in options:
            metrics_ref_dd.value = "(none)"

    _toggle_sections()
    mode_dd.observe(_toggle_sections, names="value")
    sample_source_dd.observe(_toggle_sections, names="value")
    sample_source_dd.observe(lambda *_: _refresh_sample_groups(), names="value")
    sample_run_dd.observe(lambda *_: _refresh_sample_groups(), names="value")
    sample_path_txt.observe(lambda *_: _refresh_sample_groups(), names="value")
    if g.get("cell_dd") is not None:
        g["cell_dd"].observe(lambda *_: _refresh_sample_runs(), names="value")
        g["cell_dd"].observe(lambda *_: _refresh_sample_groups(), names="value")
    if g.get("tunes_dd") is not None:
        g["tunes_dd"].observe(lambda *_: _refresh_sample_runs(), names="value")
        g["tunes_dd"].observe(lambda *_: _refresh_sample_groups(), names="value")
    if g.get("model_dd") is not None:
        g["model_dd"].observe(lambda *_: _refresh_sample_runs(), names="value")
        g["model_dd"].observe(lambda *_: _refresh_sample_groups(), names="value")

    def _apply_extra_opts():
        g["extra_mode"] = mode_dd.value
        compare_apply: list[str] = []
        if cfg_sim_cb.value:
            compare_apply.append("sim_config")
        if cfg_cell_cb.value:
            compare_apply.append("cell_config")
        if cfg_geom_cb.value:
            compare_apply.append("geometry")
        if cfg_syn_cb.value:
            compare_apply.append("syn_config")
        if cfg_syn_groups_cb.value:
            compare_apply.append("syn_groups")
        if cfg_fit_cb.value:
            compare_apply.append("fit_json")
        if not compare_apply:
            compare_apply = list(analysis.RESTORE_CONFIG_APPLY_CHOICES)
        g["extra_compare_apply"] = compare_apply
        g["extra_compare_syn_groups_selector"] = str(cfg_syn_groups_txt.value or "all").strip() or "all"
        g["extra_compare_diff_only"] = cfg_diff_cb.value
        g["extra_cell_tables"] = bool(table_cell_cb.value)
        g["extra_geometry_tables"] = bool(table_geom_cb.value)
        g["extra_synapse_tables"] = bool(table_syn_cb.value)
        g["extra_recording_tables"] = bool(table_rec_cb.value)

        g["extra_snapshot_diff_only"] = snap_diff_cb.value
        g["save_snapshot_compare_table"] = snap_save_cb.value
        g["snapshot_compare_scope"] = snap_scope_dd.value
        g["snapshot_compare_format"] = snap_fmt_dd.value
        depth_val = analysis.parse_optional_float(snap_depth_txt.value)
        if depth_val is not None:
            g["snapshot_compare_max_depth"] = int(depth_val)
        list_val = analysis.parse_optional_float(snap_list_txt.value)
        if list_val is not None:
            g["snapshot_compare_max_list_items"] = int(list_val)
        g["output_metrics_show_params"] = metrics_show_params_cb.value
        g["output_metrics_std_mode"] = metrics_std_mode_dd.value
        g["output_metrics_show_delta"] = metrics_show_delta_cb.value
        g["output_metrics_highlight_best"] = metrics_highlight_cb.value
        g["output_metrics_use_plot_keys_for_tables"] = bool(metrics_use_plot_keys_cb.value)
        g["output_metrics_plot_keys"] = list(metrics_plot_sel.value)
        g["output_metrics_plot_style"] = metrics_plot_style_dd.value
        g["output_metrics_plot_show_points"] = bool(metrics_plot_show_points_cb.value)
        g["output_metrics_plot_jitter"] = bool(metrics_plot_jitter_cb.value)
        g["output_metrics_plot_show_error"] = bool(metrics_plot_show_error_cb.value)
        g["output_metrics_plot_ncols"] = int(metrics_plot_ncols_txt.value)
        g["output_metrics_plot_panel_size"] = [
            float(metrics_plot_panel_w_txt.value),
            float(metrics_plot_panel_h_txt.value),
        ]
        g["output_metrics_plot_bar_alpha"] = float(metrics_plot_bar_alpha_txt.value)
        g["output_metrics_plot_show_legend"] = bool(metrics_plot_show_legend_cb.value)
        g["output_metrics_plot_legend_loc"] = str(metrics_plot_legend_loc_txt.value or "best").strip() or "best"
        g["output_metrics_plot_save_plot"] = bool(metrics_plot_save_plot_cb.value)
        g["output_metrics_plot_save_data"] = bool(metrics_plot_save_data_cb.value)
        g["output_metrics_plot_data_path"] = str(metrics_plot_data_path_txt.value or "").strip()
        ref_val = metrics_ref_dd.value
        g["output_metrics_ref_label"] = None if ref_val in (None, "", "(none)") else ref_val
        g["input_sample_path"] = sample_path_txt.value.strip() or None
        g["input_sample_source"] = sample_source_dd.value
        g["input_sample_run"] = sample_run_dd.value
        g["input_sample_groups"] = list(sample_groups_sel.value)
        g["input_sample_runs"] = int(sample_runs_txt.value)
        g["input_sample_show_std"] = sample_std_cb.value
        g["input_sample_show_ref"] = sample_ref_cb.value
        g["extra_synapse_show_table"] = bool(syn_table_cb.value)
        g["extra_synapse_weight_plot"] = bool(syn_weight_cb.value)
        g["extra_synapse_distance_plot"] = bool(syn_distance_cb.value)
        g["extra_synapse_scatter_plot"] = bool(syn_scatter_cb.value)
        g["extra_synapse_density"] = bool(syn_density_cb.value)
        g["extra_synapse_groups"] = analysis.parse_groups(syn_groups_txt.value)
        g["extra_synapse_trial_idx"] = int(syn_trial_txt.value)
        g["extra_synapse_weight_bin"] = float(syn_weight_bin_txt.value)
        g["extra_synapse_distance_bin"] = float(syn_dist_bin_txt.value)
        g["extra_synapse_plot_type"] = syn_plot_type_dd.value
        sample_bin = analysis.parse_optional_float(sample_bin_txt.value)
        if sample_bin is not None:
            g["input_sample_bin_ms"] = sample_bin
        else:
            g["input_sample_bin_ms"] = None
        sample_seed = analysis.parse_optional_float(sample_seed_txt.value)
        if sample_seed is not None:
            g["input_sample_seed"] = int(sample_seed)
        else:
            g["input_sample_seed"] = None

    def _on_run(_):
        _refresh_metrics_refs()
        _refresh_sample_runs()
        _refresh_sample_groups()
        _apply_extra_opts()
        with out_extra:
            out_extra.clear_output()
            mode = mode_dd.value
            if mode == "compare_configs":
                run_extra_compare_from_globals(g)
            elif mode == "input_sampling":
                run_input_sampling_from_globals(g)
            elif mode == "synapse_plots":
                run_synapse_plots_from_globals(g)
            elif mode == "snapshot_compare":
                g["extra_snapshot_tables"] = True
                run_snapshot_compare_from_globals(g)
            elif mode == "single_tables":
                run_extra_tables_from_globals(g)
            elif mode == "iclamp":
                run_iclamp_analysis_from_globals(g)
            else:
                metrics = run_output_metrics_from_globals(g)
                if metrics:
                    show_params = bool(g.get("output_metrics_show_params", True))
                    ref_label = g.get("output_metrics_ref_label")
                    show_delta = bool(g.get("output_metrics_show_delta", False))
                    highlight_best = bool(g.get("output_metrics_highlight_best", False))
                    table_metric_keys = (
                        list(g.get("output_metrics_plot_keys") or [])
                        if bool(g.get("output_metrics_use_plot_keys_for_tables", True))
                        else None
                    )
                    if isinstance(metrics, dict) and all(isinstance(v, dict) for v in metrics.values()):
                        show_md(
                            format_output_metrics_tables_columns(
                                metrics,
                                title="Output metrics",
                                show_params=show_params,
                                reference_label=ref_label,
                                show_deltas=show_delta,
                                highlight_best=highlight_best,
                                metric_keys=table_metric_keys,
                            )
                        )
                    else:
                        sel = get_selection_from_globals(g)
                        label = analysis.run_label(analysis.resolve_run(sel["base"], sel["run_single"]))
                        show_md(
                            format_output_metrics_tables(
                                metrics,
                                title=f"Output metrics ({label})",
                                show_params=show_params,
                                metric_keys=table_metric_keys,
                            )
                        )

    def _on_plot_metrics(_):
        _refresh_metrics_refs()
        _refresh_sample_runs()
        _refresh_sample_groups()
        _apply_extra_opts()
        with out_extra:
            out_extra.clear_output()
            payload = run_output_metric_distributions_from_globals(g)
            if not payload:
                return
            metric_keys = payload.get("metric_keys") or []
            if len(metric_keys) > 1:
                print(f"Plotted {len(metric_keys)} metrics as a panel grid.")
            curve_labels = [
                label
                for label, entry in (payload.get("by_label") or {}).items()
                if str((entry or {}).get("source") or "") == "curve"
            ]
            plot_style = str(payload.get("plot_style") or "box")
            if plot_style == "bar":
                print("Bar mode shows mean bars per run/curve.")
                if payload.get("show_error", False):
                    std_mode = str(payload.get("std_mode") or "std").upper()
                    print(f"Error bars are shown for run entries ({std_mode}); curve entries have no error bars.")
            elif curve_labels:
                print("Curve-only entries are shown as mean markers (no trial boxes).")

    run_btn.on_click(_on_run)
    metrics_plot_btn.on_click(_on_plot_metrics)
    extra_help_btn.on_click(lambda *_: _print_help(out_extra, HELP_EXTRA))

    g["extra_mode_dd"] = mode_dd
    g["out_extra"] = out_extra

    display(
        widgets.VBox([
            widgets.HBox([mode_dd, run_btn, extra_help_btn]),
            cfg_box,
            metrics_box,
            sample_box,
            syn_box,
            snap_box,
            tables_box,
            out_extra,
        ])
    )


def run_input_sampling_from_globals(g: Dict[str, Any]) -> None:
    sel = get_selection_from_globals(g)
    source = g.get("input_sample_source", "selection")
    path_raw = g.get("input_sample_path")
    if source == "run":
        run_label = g.get("input_sample_run") or "latest"
        try:
            path_raw = analysis.resolve_run(sel["base"], run_label)
        except Exception:
            print(f"Input sampling: could not resolve run {run_label!r}")
            return
    elif source == "selection" or not path_raw:
        path_raw = (g.get("CELLS_DIR") / sel["cell"] / sel["tunes"] / sel["model"]).resolve()
    groups = g.get("input_sample_groups") or []
    group = g.get("input_sample_group")
    runs = int(g.get("input_sample_runs", 200))
    bin_ms = g.get("input_sample_bin_ms")
    seed = g.get("input_sample_seed")
    show_std = bool(g.get("input_sample_show_std", True))
    show_ref = bool(g.get("input_sample_show_ref", True))

    try:
        path = Path(path_raw)
        if not groups:
            groups = [group] if group else []
        if not groups:
            groups = _list_groups_for_sampling(path)
        if not groups:
            raise ValueError("No groups selected/found for input sampling.")
    except Exception as exc:
        print("Input sampling failed:", exc)
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, group_name in enumerate(groups):
        try:
            centers, mean_rate, std_rate, _, meta, ref_curve = input_sampling.sample_group_rates_from_path(
                path,
                group=group_name,
                runs=runs,
                bin_ms=bin_ms,
                seed=seed,
            )
        except Exception as exc:
            print(f"Input sampling failed for {group_name}: {exc}")
            continue
        plotted = True
        col = color_cycle[idx % len(color_cycle)]
        ax.plot(centers, mean_rate, label=f"{group_name} mean", color=col)
        if show_std:
            ax.fill_between(
                centers,
                mean_rate - std_rate,
                mean_rate + std_rate,
                alpha=0.15,
                color=col,
                label=f"{group_name} ±std",
            )
        if show_ref and ref_curve:
            ref_t, ref_r = ref_curve
            ax.plot(ref_t, ref_r, "--", color=col, linewidth=1.5, label=f"{group_name} source")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Rate (Hz per synapse)")
    ax.set_title(f"Input sampling: {runs} runs")
    if plotted:
        ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def _list_groups_for_sampling(path: Optional[Path]) -> list[str]:
    if path is None:
        return []
    p = Path(path).expanduser().resolve()
    groups = {}
    if p.is_file() and p.suffix == ".json":
        try:
            with p.open("r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if {"tstart", "tstop", "dt"}.issubset(data.keys()):
                    data = {}
                groups = data
        except Exception:
            groups = {}
    if not groups:
        config_root = inputs._resolve_config_root(p)
        results_root = p / "results" if p.is_dir() else None
        if results_root and (results_root / "syn_config.json").is_file():
            config_root = results_root
        syn_path = config_root / "syn_config.json"
        if syn_path.is_file():
            try:
                with syn_path.open("r") as f:
                    groups = json.load(f)
                groups = inputs._expand_group_includes(groups, config_root)
            except Exception:
                groups = {}
    return sorted(list(groups.keys())) if isinstance(groups, dict) else []


def _synapse_records_from_results(
    results: Dict[str, Any],
    *,
    trial_idx: int = 0,
) -> tuple[Dict[str, Any], str]:
    syn_records = results.get("syn_records")
    if isinstance(syn_records, dict) and any(syn_records.values()):
        return syn_records, "syn_records"

    by_trial = results.get("syn_records_by_trial") or []
    if by_trial:
        idx = min(max(int(trial_idx), 0), len(by_trial) - 1)
        entry = by_trial[idx] or {}
        if isinstance(entry, dict):
            records = entry.get("records") or entry.get("syn_records") or {}
        else:
            records = {}
        if isinstance(records, dict) and any(records.values()):
            return records, f"syn_records_by_trial[{idx}]"

    return {}, ""


def _resolve_synapse_plot_groups(
    syn_records: Dict[str, Any],
    groups: Any,
) -> tuple[list[str], list[str]]:
    available = [group for group, recs in (syn_records or {}).items() if recs]
    if not available:
        return [], []
    if groups is None:
        return available, []
    if isinstance(groups, str):
        parsed = analysis.parse_groups(groups)
    else:
        parsed = list(groups)
    if not parsed:
        return available, []
    if parsed == ["all"]:
        return ["all"], []
    missing = [group for group in parsed if group not in syn_records]
    selected = [group for group in parsed if group in syn_records and syn_records.get(group)]
    return selected, missing


def run_synapse_plots_from_globals(g: Dict[str, Any]) -> None:
    sel, run_dir, res = resolve_single_from_globals(g)
    trial_idx = int(g.get("extra_synapse_trial_idx", 0) or 0)
    syn_records, source = _synapse_records_from_results(res, trial_idx=trial_idx)
    if not syn_records:
        print(
            "Synapse plots skipped: this run has no saved synapse records. "
            "Rerun Step 5 with save_syn_records_sidecar and/or save_syn_records_by_trial enabled."
        )
        return

    plotted_groups, missing_groups = _resolve_synapse_plot_groups(
        syn_records,
        g.get("extra_synapse_groups"),
    )
    if missing_groups:
        print("Synapse groups not found:", ", ".join(missing_groups))
    if not plotted_groups:
        print("Synapse plots skipped: no matching groups with saved records.")
        return

    shown_groups = list(syn_records.keys()) if plotted_groups == ["all"] else plotted_groups
    print(f"Run: {analysis.run_label(run_dir)}")
    print(f"Synapse records source: {source}")
    for group in shown_groups:
        print(f"  {group}: {len(syn_records.get(group, []) or [])} synapses")

    duration_ms = analysis._get_duration_ms(res)
    if bool(g.get("extra_synapse_show_table", True)):
        summary = analysis.summarize_synapse_records(
            syn_records,
            duration_ms=duration_ms,
        )
        show_md(
            analysis.format_synapse_summary_table(
                summary,
                title="Synapse placement + weights",
                groups=None if plotted_groups == ["all"] else plotted_groups,
            )
        )
        _save_json(
            summary,
            analysis.analysis_dir_for_run(run_dir) / "synapse_summary.json",
            enabled=bool(g.get("save_analysis", False)),
        )

    plot_type = str(g.get("extra_synapse_plot_type", "hist") or "hist")
    if plot_type not in ("hist", "line", "both"):
        plot_type = "hist"
    weight_bin = float(g.get("extra_synapse_weight_bin", 0.1) or 0.1)
    distance_bin = float(g.get("extra_synapse_distance_bin", 25.0) or 25.0)
    figsize = (6.0, 4.0)
    save_enabled = bool(g.get("save_plots", False))
    plots_dpi = int(g.get("plots_dpi", 150))
    overwrite = bool(g.get("save_overwrite", False))
    plot_dir = analysis.plot_dir_for_run(run_dir)

    def _plot_one(
        *,
        props: tuple[str, ...],
        name: str,
        bins: float,
        win_size: float,
        cell: Any = None,
        plot_type_override: Optional[str] = None,
    ) -> None:
        try:
            plotting.plot_syn_records(
                cell,
                syn_records,
                plotted_groups=plotted_groups,
                plotted_props=props,
                plot_type=plot_type_override or plot_type,
                bins=bins,
                win_size=win_size,
                fig_sizes=figsize,
            )
            fig = plt.gcf()
            _save_fig(
                fig,
                plot_dir / f"{name}.png",
                enabled=save_enabled,
                dpi=plots_dpi,
                overwrite=overwrite,
            )
        except Exception as exc:
            print(f"Synapse {name} plot failed: {exc}")

    if bool(g.get("extra_synapse_weight_plot", True)):
        _plot_one(
            props=("weight_probability",),
            name="synapse_weight_probability",
            bins=weight_bin,
            win_size=weight_bin,
        )

    if bool(g.get("extra_synapse_distance_plot", True)):
        _plot_one(
            props=("distance_probability",),
            name="synapse_distance_probability",
            bins=distance_bin,
            win_size=distance_bin,
        )

    if bool(g.get("extra_synapse_scatter_plot", True)):
        _plot_one(
            props=("weight", "distance"),
            name="synapse_weight_distance",
            bins=distance_bin,
            win_size=distance_bin,
            plot_type_override="hist",
        )

    if bool(g.get("extra_synapse_density", False)):
        tune_dir = (g.get("CELLS_DIR") / sel["cell"] / sel["tunes"] / sel["model"]).resolve()
        try:
            cell, _, _ = analysis.load_cell_and_geometry(tune_dir)
        except Exception as exc:
            print(f"Synapse density plot skipped: cell/geometry load failed: {exc}")
            return
        _plot_one(
            props=("distance_density",),
            name="synapse_distance_density",
            bins=distance_bin,
            win_size=distance_bin,
            cell=cell,
        )


def run_extra_tables_from_globals(g: Dict[str, Any]) -> None:
    sel, run_dir, res = resolve_single_from_globals(g)
    load_cell = bool(g.get("load_cell_for_analysis"))
    need_cell = bool(g.get("extra_cell_tables") or g.get("extra_geometry_tables") or g.get("extra_synapse_tables"))
    cell = None
    geom = None
    geom_cfg = None

    if load_cell and need_cell:
        tune_dir = (g.get("CELLS_DIR") / sel["cell"] / sel["tunes"] / sel["model"]).resolve()
        try:
            cell, geom, geom_cfg = analysis.load_cell_and_geometry(tune_dir)
        except Exception as exc:
            print("Cell/geometry load failed:", exc)
    elif need_cell and not load_cell:
        if g.get("extra_cell_tables"):
            print("Cell tables skipped: set load_cell_for_analysis=True to enable.")
        if g.get("extra_geometry_tables"):
            print("Geometry tables skipped: set load_cell_for_analysis=True to enable.")

    if g.get("extra_cell_tables"):
        if cell is None:
            print("Cell tables skipped: cell object not available.")
        else:
            cell_sections = analysis.summarize_cell_sections(cell)
            mech_summary = analysis.summarize_mechanisms(cell)
            show_md(analysis.format_section_summary_table(cell_sections, title=f"{analysis.run_label(run_dir)} cell sections"))
            show_md(analysis.format_mechanism_summary_table(mech_summary, title="Mechanisms (per section group)"))
            analysis.save_json(cell_sections, analysis.analysis_dir_for_run(run_dir) / "cell_sections.json", enabled=bool(g.get("save_analysis", False)))
            analysis.save_json(mech_summary, analysis.analysis_dir_for_run(run_dir) / "cell_mechanisms.json", enabled=bool(g.get("save_analysis", False)))

    if g.get("extra_geometry_tables"):
        if geom is None:
            print("Geometry tables skipped: geometry object not available.")
        else:
            geom_summary = analysis.summarize_geometry(geom, geom_config=geom_cfg)
            show_md(analysis.format_geometry_summary_table(geom_summary, title="Geometry distances"))
            analysis.save_json(geom_summary, analysis.analysis_dir_for_run(run_dir) / "geometry_summary.json", enabled=bool(g.get("save_analysis", False)))

    if g.get("extra_synapse_tables"):
        syn_summary = analysis.summarize_synapse_records(
            res.get("syn_records") or {},
            geom=geom,
            duration_ms=analysis._get_duration_ms(res),
        )
        if syn_summary.get("groups"):
            show_md(analysis.format_synapse_summary_table(syn_summary, title="Synapse placement + weights"))
            analysis.save_json(syn_summary, analysis.analysis_dir_for_run(run_dir) / "synapse_summary.json", enabled=bool(g.get("save_analysis", False)))
        else:
            print("No synapse records found for this run.")

    if g.get("extra_recording_tables", True):
        cell_rec_summary = analysis.summarize_cell_recordings(res)
        syn_trace_summary = analysis.summarize_total_synaptic_traces(res)
        any_recordings = False
        if cell_rec_summary.get("available"):
            any_recordings = True
            show_md(analysis.format_cell_recording_summary_table(cell_rec_summary, title="Cell recordings summary"))
            analysis.save_json(
                cell_rec_summary,
                analysis.analysis_dir_for_run(run_dir) / "cell_recording_summary.json",
                enabled=bool(g.get("save_analysis", False)),
            )
        if syn_trace_summary.get("available"):
            any_recordings = True
            show_md(analysis.format_total_synaptic_trace_table(syn_trace_summary, title="Total synaptic trace summary"))
            analysis.save_json(
                syn_trace_summary,
                analysis.analysis_dir_for_run(run_dir) / "syn_trace_summary.json",
                enabled=bool(g.get("save_analysis", False)),
            )
        if not any_recordings:
            print("No cell_recordings or total synaptic I/G traces found for this run.")


def run_extra_compare_from_globals(g: Dict[str, Any]) -> None:
    sel = get_selection_from_globals(g)
    tune_dir = (g.get("CELLS_DIR") / sel["cell"] / sel["tunes"] / sel["model"]).resolve()
    if not tune_dir.is_dir():
        print(f"Compare configs failed: target tune directory not found: {tune_dir}")
        return

    run_paths, run_labels, selection_warnings = _selected_config_compare_runs(sel)
    if not run_paths:
        print("Compare configs skipped: no valid run folders selected.")
        if selection_warnings:
            show_md(analysis.format_diff_list_table(selection_warnings, title="Selection warnings", max_items=20))
        return

    apply_raw = g.get("extra_compare_apply") or list(analysis.RESTORE_CONFIG_APPLY_CHOICES)
    apply = [str(tok).strip() for tok in apply_raw if str(tok).strip() in analysis.RESTORE_CONFIG_APPLY_CHOICES]
    if not apply:
        apply = list(analysis.RESTORE_CONFIG_APPLY_CHOICES)
    syn_groups = str(g.get("extra_compare_syn_groups_selector", "all") or "all").strip() or "all"

    diff_only = bool(g.get("extra_compare_diff_only", True))
    if diff_only and len(run_paths) < 2:
        diff_only = False
        print("Only one run selected; disabling diff-only filter to show all checked parameters.")

    report = analysis.compare_config_runs(
        run_paths,
        target_tune=tune_dir,
        labels=run_labels,
        apply=apply,
        syn_groups=syn_groups,
        diff_only=diff_only,
        allow_source_fallback=True,
    )

    print(f"Target tune: {report.get('target_tune')}")
    print(f"Runs compared: {len(report.get('runs') or [])}")
    print(f"Apply: {', '.join(report.get('apply') or [])}; syn_groups={report.get('syn_groups')}")
    print(f"Rows shown: {len(report.get('rows') or [])}")

    if selection_warnings:
        show_md(analysis.format_diff_list_table(selection_warnings, title="Selection warnings", max_items=20))
    if report.get("skipped"):
        show_md(analysis.format_diff_list_table(report.get("skipped", []), title="Skipped runs", max_items=20))
    if report.get("warnings"):
        show_md(analysis.format_diff_list_table(report.get("warnings", []), title="Config compare warnings", max_items=40))

    show_md(
        analysis.format_config_compare_table(
            report,
            title="Config compare (restore-style)",
        )
    )

    if bool(g.get("save_analysis", False)):
        if len(report.get("runs", [])) == 1:
            out_dir = analysis.analysis_dir_for_run(Path(report["runs"][0]["run_dir"]))
        else:
            out_dir = sel["base"] / "_comparisons" / "config_compare_list" / "analysis"
        analysis.save_json(report, out_dir / "config_compare_report.json", enabled=True)


def run_snapshot_compare_from_globals(g: Dict[str, Any]) -> None:
    if not g.get("extra_snapshot_tables"):
        print("Snapshot compare disabled (set extra_snapshot_tables=True).")
        return
    sel, run_a, run_b, _, _ = resolve_compare_from_globals(g)
    if run_b is None:
        print("Snapshot compare disabled (set Compare B to a run name).")
        return

    label_a = analysis.run_label(run_a)
    label_b = analysis.run_label(run_b)
    out_dir = analysis.analysis_dir_for_compare(sel["base"], run_a, run_b)

    report = analysis.snapshot_compare_report(
        run_a,
        run_b,
        labels=(label_a, label_b),
        max_diffs=200,
        diff_only=g.get("extra_snapshot_diff_only"),
        save_table=g.get("save_snapshot_compare_table"),
        table_scope=g.get("snapshot_compare_scope"),
        table_format=g.get("snapshot_compare_format"),
        table_max_depth=g.get("snapshot_compare_max_depth"),
        table_max_list_items=g.get("snapshot_compare_max_list_items"),
        out_dir=out_dir,
        save_report_json=bool(g.get("save_analysis", False)),
    )

    show_md(report.get("snapshot_diff_table") or "")
    if not g.get("extra_snapshot_diff_only"):
        show_md(
            analysis.format_snapshot_compare(
                report["report"]["snapshot_a"],
                report["report"]["snapshot_b"],
                labels=(label_a, label_b),
            )
        )
    show_md(report.get("manifest_diff_table") or "")
    show_md(report.get("deep_diff_table") or "")


def run_iclamp_analysis_from_globals(g: Dict[str, Any]) -> None:
    sel, run_dir, res = resolve_single_from_globals(g)
    mode = res.get("mode")
    if mode != "iclamp":
        print(f"IClamp analysis skipped (mode={mode!r}). Select an iclamp run and re-run this cell.")
        return

    stats = analysis.summarize_iclamp(res)
    if stats is None:
        print("No trace data found for IClamp run.")
        return

    print(f"Baseline Vm: {stats['baseline']:.2f} mV")
    print(f"Peak Vm: {stats['peak']:.2f} mV, Min Vm: {stats['vmin']:.2f} mV")
    if stats["spike_count"] is not None:
        print(f"Spike count during pulse: {stats['spike_count']} (rate {stats['spike_rate']:.2f} Hz)")

    plt.figure(figsize=(6, 4))
    plt.plot(stats["T"], stats["V"], lw=1.5)
    if stats["delay_ms"] is not None:
        plt.axvline(float(stats["delay_ms"]), color="k", ls="--", lw=1)
    if stats["delay_ms"] is not None and stats["dur_ms"] is not None:
        plt.axvline(float(stats["delay_ms"] + stats["dur_ms"]), color="k", ls="--", lw=1)
    plt.xlabel("Time (ms)")
    plt.ylabel("Vm (mV)")
    plt.title("IClamp Vm trace")
    plt.tight_layout()

    iclamp_payload = res.get("iclamp", {}) or {}
    currents = iclamp_payload.get("I")
    if isinstance(currents, dict) and currents:
        total_I = None
        for arr in currents.values():
            if total_I is None:
                total_I = arr.copy()
            else:
                total_I = total_I + arr
        if total_I is not None:
            plt.figure(figsize=(6, 3))
            plt.plot(stats["T"], total_I, lw=1.0)
            plt.xlabel("Time (ms)")
            plt.ylabel("Total membrane current")
            plt.title("Summed currents (if recorded)")
            plt.tight_layout()

    sel_local = get_selection_from_globals(g)
    if sel_local.get("run_b") not in (None, "none", "None"):
        try:
            run_a = analysis.resolve_run(sel_local["base"], sel_local.get("run_a"))
            run_b = analysis.resolve_run(sel_local["base"], sel_local.get("run_b"))
            res_a = run_sim.load_results(run_a)
            res_b = run_sim.load_results(run_b)
        except Exception as exc:
            print(f"IClamp compare failed to load runs: {exc}")
        else:
            if res_a.get("mode") != "iclamp" or res_b.get("mode") != "iclamp":
                print("IClamp compare skipped: both runs must be mode='iclamp'.")
            else:
                stats_a = analysis.summarize_iclamp(res_a)
                stats_b = analysis.summarize_iclamp(res_b)
                if stats_a is None or stats_b is None:
                    print("IClamp compare skipped: missing traces.")
                else:
                    label_a = analysis.run_label(run_a)
                    label_b = analysis.run_label(run_b)
                    print(f"Compare A: {label_a}")
                    print(f"  Baseline {stats_a['baseline']:.2f} mV, Peak {stats_a['peak']:.2f} mV, Min {stats_a['vmin']:.2f} mV")
                    if stats_a["spike_count"] is not None:
                        print(f"  Spikes {stats_a['spike_count']} (rate {stats_a['spike_rate']:.2f} Hz)")
                    print(f"Compare B: {label_b}")
                    print(f"  Baseline {stats_b['baseline']:.2f} mV, Peak {stats_b['peak']:.2f} mV, Min {stats_b['vmin']:.2f} mV")
                    if stats_b["spike_count"] is not None:
                        print(f"  Spikes {stats_b['spike_count']} (rate {stats_b['spike_rate']:.2f} Hz)")

__all__ = [
    "HELP_EXTRA",
    "build_extra_ui",
    "run_extra_compare_from_globals",
    "run_extra_tables_from_globals",
    "run_iclamp_analysis_from_globals",
    "run_input_sampling_from_globals",
    "run_snapshot_compare_from_globals",
    "run_synapse_plots_from_globals",
]
