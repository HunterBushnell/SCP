from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from modules_local import analysis, plotting, run_sim


def compare_enabled(selection: Dict[str, Any]) -> bool:
    run_a_path = selection.get("run_a_path")
    run_b_path = selection.get("run_b_path")
    if run_a_path and run_b_path:
        return True
    return selection.get("run_b") not in (None, "none", "", "None")


def _coerce_run_path(path_val: Any, base_dir: Path) -> Optional[Path]:
    if path_val in (None, "", "none", "None"):
        return None
    p = Path(str(path_val)).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


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
    if run_a_path is not None and run_b_path is not None:
        run_a = run_a_path
        run_b = run_b_path
    else:
        run_a = analysis.resolve_run(base_dir, selection["run_a"])
        run_b = analysis.resolve_run(base_dir, selection["run_b"])
    res_a = run_sim.load_results(run_a)
    res_b = run_sim.load_results(run_b)
    return run_a, run_b, res_a, res_b


def _save_fig(fig, out_path, *, enabled: bool, dpi: int) -> None:
    analysis.save_figure(fig, out_path, enabled=enabled, dpi=dpi)


def _save_json(data: dict, out_path, *, enabled: bool) -> None:
    analysis.save_json(data, out_path, enabled=enabled)


def _stim_window(sim_cfg: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    stim_start = sim_cfg.get("stim_start_ms")
    stim_stop = sim_cfg.get("stim_stop_ms")
    stim_dur = sim_cfg.get("stim_duration_ms")
    if stim_stop is None and stim_start is not None and stim_dur is not None:
        stim_stop = float(stim_start) + float(stim_dur)
    return (
        float(stim_start) if stim_start is not None else None,
        float(stim_stop) if stim_stop is not None else None,
    )


def run_output_plots(
    selection: Dict[str, Any],
    opts: Dict[str, Any],
    *,
    save_plots: bool = False,
    save_analysis: bool = False,
    plots_dpi: int = 150,
) -> None:
    if compare_enabled(selection):
        run_a, run_b, res_a, res_b = resolve_compare(selection)
        if run_b is None or res_a is None or res_b is None:
            print("Comparison disabled (set Compare B to a run name).")
            return
        label_a = analysis.run_label(run_a)
        label_b = analysis.run_label(run_b)
        smooth_mode = (res_a.get("sim_cfg", {}) or {}).get("avg_rate_curve_smooth_mode", "center")
        output_norms = None
        if opts.get("output_curve_mode", "raw") == "normalized":
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
                norm_a = analysis.normalize_output_curve(
                    curve_a,
                    res_a.get("sim_cfg", {}) or {},
                    mode="normalized",
                    norm_mode=opts.get("output_norm_mode", "avg"),
                    baseline_ms=opts.get("output_baseline_ms", 100.0),
                    baseline_mode=opts.get("output_baseline_mode", "window"),
                    norm_window=opts.get("output_norm_window", "stim"),
                )
                norm_b = analysis.normalize_output_curve(
                    curve_b,
                    res_b.get("sim_cfg", {}) or {},
                    mode="normalized",
                    norm_mode=opts.get("output_norm_mode", "avg"),
                    baseline_ms=opts.get("output_baseline_ms", 100.0),
                    baseline_mode=opts.get("output_baseline_mode", "window"),
                    norm_window=opts.get("output_norm_window", "stim"),
                )
                output_norms = (
                    {
                        "baseline_mean": norm_a.get("baseline_mean"),
                        "norm_scale": norm_a.get("norm_scale"),
                    },
                    {
                        "baseline_mean": norm_b.get("baseline_mean"),
                        "norm_scale": norm_b.get("norm_scale"),
                    },
                )

        if opts.get("plot_outputs", True):
            fig_cmp, _ = plotting.plot_compare_side_by_side(
                res_a,
                res_b,
                labels=(label_a, label_b),
                win_size=opts.get("win_size", 25),
                bin_ms=opts.get("output_bin_ms"),
                plot_window=opts.get("plot_window", (None, None)),
                smooth_mode=smooth_mode,
                output_norms=output_norms,
                layout=opts.get("compare_output_layout", "side-by-side"),
            )
            _save_fig(
                fig_cmp,
                analysis.plot_dir_for_compare(selection["base"], run_a, run_b) / "compare_outputs.png",
                enabled=save_plots,
                dpi=plots_dpi,
            )

        if opts.get("plot_output_curve", True):
            curve_a = analysis.compute_output_curve_from_results(
                res_a,
                bin_ms=opts.get("output_bin_ms"),
                smooth_ms=opts.get("output_smooth_ms"),
                smooth_mode=opts.get("output_smooth_mode", "causal"),
            )
            curve_b = analysis.compute_output_curve_from_results(
                res_b,
                bin_ms=opts.get("output_bin_ms"),
                smooth_ms=opts.get("output_smooth_ms"),
                smooth_mode=opts.get("output_smooth_mode", "causal"),
            )
            scatter_curve = analysis.load_scatter_curve_optional(
                enabled=opts.get("output_scatter_enabled", False),
                path=opts.get("output_scatter_path", ""),
                time_unit=opts.get("output_scatter_time_unit", "s"),
                bin_ms=opts.get("output_bin_ms"),
                smooth_ms=opts.get("output_smooth_ms"),
                smooth_mode=opts.get("output_smooth_mode", "causal"),
                shift_ms=opts.get("output_scatter_shift_ms"),
                quiet=True,
            )
            if curve_a and curve_b:
                curve_a = analysis.normalize_output_curve(
                    curve_a,
                    res_a.get("sim_cfg", {}) or {},
                    mode=opts.get("output_curve_mode", "raw"),
                    norm_mode=opts.get("output_norm_mode", "avg"),
                    baseline_ms=opts.get("output_baseline_ms", 100.0),
                    baseline_mode=opts.get("output_baseline_mode", "window"),
                    norm_window=opts.get("output_norm_window", "stim"),
                )
                curve_b = analysis.normalize_output_curve(
                    curve_b,
                    res_b.get("sim_cfg", {}) or {},
                    mode=opts.get("output_curve_mode", "raw"),
                    norm_mode=opts.get("output_norm_mode", "avg"),
                    baseline_ms=opts.get("output_baseline_ms", 100.0),
                    baseline_mode=opts.get("output_baseline_mode", "window"),
                    norm_window=opts.get("output_norm_window", "stim"),
                )
                stim_start, stim_stop = _stim_window(res_a.get("sim_cfg", {}) or {})
                fig_curve, _ = plotting.plot_compare_output_curves(
                    curve_a,
                    curve_b,
                    labels=(label_a, label_b),
                    plot_window=opts.get("plot_window", (None, None)),
                    stim_start=stim_start,
                    stim_stop=stim_stop,
                    title="Output curve compare",
                )
                if scatter_curve:
                    ax = fig_curve.axes[0] if fig_curve.axes else None
                    if ax is not None:
                        ax.plot(
                            np.asarray(scatter_curve.get("t_ms", []), dtype=float),
                            np.asarray(scatter_curve.get("rate_hz", []), dtype=float),
                            color=opts.get("output_scatter_color", "0.4"),
                            lw=2,
                            ls="--",
                            label=opts.get("output_scatter_label", "Bio scatter"),
                        )
                        ax.legend()
                _save_fig(
                    fig_curve,
                    analysis.plot_dir_for_compare(selection["base"], run_a, run_b) / "compare_output_curve.png",
                    enabled=save_plots,
                    dpi=plots_dpi,
                )
            else:
                print("Output curve compare skipped: missing spikes in one run.")

        if opts.get("plot_spike_stats", False):
            print("Spike stats are only shown for single runs.")
        return

    run_dir, res = resolve_single(selection)
    smooth_mode = (res.get("sim_cfg", {}) or {}).get("avg_rate_curve_smooth_mode", "center")

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
                plot_window=opts.get("plot_window", (None, None)),
            )
            if opts.get("output_bin_ms") is not None:
                sim_params["bins"] = float(opts.get("output_bin_ms"))
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
                    norm_curve = analysis.normalize_output_curve(
                        curve_norm,
                        res.get("sim_cfg", {}) or {},
                        mode="normalized",
                        norm_mode=opts.get("output_norm_mode", "avg"),
                        baseline_ms=opts.get("output_baseline_ms", 100.0),
                        baseline_mode=opts.get("output_baseline_mode", "window"),
                        norm_window=opts.get("output_norm_window", "stim"),
                    )
                    output_norm = {
                        "baseline_mean": norm_curve.get("baseline_mean"),
                        "norm_scale": norm_curve.get("norm_scale"),
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
            )
            _save_fig(
                plt.gcf(),
                analysis.plot_dir_for_run(run_dir) / "output_plot.png",
                enabled=save_plots,
                dpi=plots_dpi,
            )
        else:
            fig_out = plotting.plot_results(
                res,
                syn_records=res.get("syn_records"),
                in_vivo_curve=in_vivo_curve,
                win_size=opts.get("win_size", 25),
                raster_style=opts.get("raster_style", "dot"),
                plot_raster=opts.get("plot_raster", True),
                plot_window=opts.get("plot_window", (None, None)),
                smooth_mode=smooth_mode,
                bin_ms=opts.get("output_bin_ms"),
            )
            _save_fig(
                fig_out,
                analysis.plot_dir_for_run(run_dir) / "output_plot.png",
                enabled=save_plots,
                dpi=plots_dpi,
            )

    if opts.get("plot_output_curve", True):
        curve_single = analysis.compute_output_curve_from_results(
            res,
            bin_ms=opts.get("output_bin_ms"),
            smooth_ms=opts.get("output_smooth_ms"),
            smooth_mode=opts.get("output_smooth_mode", "causal"),
        )
        scatter_curve = analysis.load_scatter_curve_optional(
            enabled=opts.get("output_scatter_enabled", False),
            path=opts.get("output_scatter_path", ""),
            time_unit=opts.get("output_scatter_time_unit", "s"),
            bin_ms=opts.get("output_bin_ms"),
            smooth_ms=opts.get("output_smooth_ms"),
            smooth_mode=opts.get("output_smooth_mode", "causal"),
            shift_ms=opts.get("output_scatter_shift_ms"),
            quiet=True,
        )
        if curve_single:
            curve_single = analysis.normalize_output_curve(
                curve_single,
                res.get("sim_cfg", {}) or {},
                mode=opts.get("output_curve_mode", "raw"),
                norm_mode=opts.get("output_norm_mode", "avg"),
                baseline_ms=opts.get("output_baseline_ms", 100.0),
                baseline_mode=opts.get("output_baseline_mode", "window"),
                norm_window=opts.get("output_norm_window", "stim"),
            )
            stim_start, stim_stop = _stim_window(res.get("sim_cfg", {}) or {})
            fig_curve, _ = plotting.plot_output_curve(
                curve_single,
                label=analysis.run_label(run_dir),
                color=(res.get("sim_cfg", {}) or {}).get("color", None),
                plot_window=opts.get("plot_window", (None, None)),
                stim_start=stim_start,
                stim_stop=stim_stop,
                title="Output curve (avg)",
            )
            if scatter_curve:
                ax = fig_curve.axes[0] if fig_curve.axes else None
                if ax is not None:
                    ax.plot(
                        np.asarray(scatter_curve.get("t_ms", []), dtype=float),
                        np.asarray(scatter_curve.get("rate_hz", []), dtype=float),
                        color=opts.get("output_scatter_color", "0.4"),
                        lw=2,
                        ls="--",
                        label=opts.get("output_scatter_label", "Bio scatter"),
                    )
                    ax.legend()
            _save_fig(
                fig_curve,
                analysis.plot_dir_for_run(run_dir) / "output_curve.png",
                enabled=save_plots,
                dpi=plots_dpi,
            )
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
        )
        _save_json(
            stats_single,
            analysis.analysis_dir_for_run(run_dir) / "spike_stats.json",
            enabled=save_analysis,
        )


def run_input_plots(
    selection: Dict[str, Any],
    opts: Dict[str, Any],
    *,
    save_plots: bool = False,
    save_analysis: bool = False,
    plots_dpi: int = 150,
) -> None:
    if compare_enabled(selection):
        run_a, run_b, res_a, res_b = resolve_compare(selection)
        if run_b is None or res_a is None or res_b is None:
            print("Comparison disabled (set Compare B to a run name).")
            return
        label_a = analysis.run_label(run_a)
        label_b = analysis.run_label(run_b)
        group_colors = analysis.merge_group_colors(res_a, res_b)

        if opts.get("plot_inputs_mean", True):
            summary_a = analysis.summarize_inputs_from_results(
                res_a,
                groups=opts.get("input_groups"),
                bin_ms=opts.get("input_bin_ms"),
                smooth_ms=opts.get("input_smooth_ms"),
            )
            summary_b = analysis.summarize_inputs_from_results(
                res_b,
                groups=opts.get("input_groups"),
                bin_ms=opts.get("input_bin_ms"),
                smooth_ms=opts.get("input_smooth_ms"),
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
                group_colors=group_colors,
            )
            _save_fig(
                fig_cmp_in,
                analysis.plot_dir_for_compare(selection["base"], run_a, run_b) / "compare_inputs.png",
                enabled=save_plots,
                dpi=plots_dpi,
            )
        if opts.get("plot_input_raster", False):
            print("Input raster is only available for single runs.")
        return

    run_dir, res = resolve_single(selection)
    group_colors = analysis.group_colors_from_results(res)

    if opts.get("plot_inputs_mean", True):
        summary_single = analysis.summarize_inputs_from_results(
            res,
            groups=opts.get("input_groups"),
            bin_ms=opts.get("input_bin_ms"),
            smooth_ms=opts.get("input_smooth_ms"),
        )
        curve_single = (res.get("meta") or {}).get("avg_rate_curve")
        fig_in, _ = plotting.plot_input_means(
            summary_single,
            label=analysis.run_label(run_dir),
            groups=opts.get("input_groups"),
            show_std=opts.get("show_input_std", False),
            output_curve=curve_single,
            group_colors=group_colors,
        )
        _save_fig(
            fig_in,
            analysis.plot_dir_for_run(run_dir) / "inputs_mean.png",
            enabled=save_plots,
            dpi=plots_dpi,
        )
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
                plot_window=opts.get("input_plot_window", (None, None)),
                plot_raster=True,
            )
            _save_fig(
                plt.gcf(),
                analysis.plot_dir_for_run(run_dir) / "inputs_raster.png",
                enabled=save_plots,
                dpi=plots_dpi,
            )
        else:
            print("No saved inputs available for raster plot.")


def run_output_metrics(
    selection: Dict[str, Any],
    opts: Dict[str, Any],
    *,
    save_analysis: bool = False,
) -> Optional[Dict[str, Any]]:
    if compare_enabled(selection):
        print("Output metrics are only available for single runs.")
        return None
    run_dir, res = resolve_single(selection)
    curve = analysis.compute_output_curve_from_results(
        res,
        bin_ms=opts.get("output_bin_ms"),
        smooth_ms=opts.get("output_smooth_ms"),
        smooth_mode=opts.get("output_smooth_mode", "causal"),
    )
    if not curve:
        print("Output metrics skipped: missing spikes in this run.")
        return None
    curve = analysis.normalize_output_curve(
        curve,
        res.get("sim_cfg", {}) or {},
        mode=opts.get("output_curve_mode", "raw"),
        norm_mode=opts.get("output_norm_mode", "avg"),
        baseline_ms=opts.get("output_baseline_ms", 100.0),
        baseline_mode=opts.get("output_baseline_mode", "window"),
        norm_window=opts.get("output_norm_window", "stim"),
    )
    metrics = analysis.compute_output_metrics(
        curve,
        res.get("sim_cfg", {}) or {},
        peak_window_ms=opts.get("output_peak_window_ms", 100.0),
        drop_window_ms=opts.get("output_drop_window_ms", 100.0),
        rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
        auc_window=opts.get("output_auc_window", "stim"),
    )
    _save_json(
        metrics,
        analysis.analysis_dir_for_run(run_dir) / "output_metrics.json",
        enabled=save_analysis,
    )
    return metrics


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


def format_kv_table(data: dict, *, title: str = "Output metrics") -> str:
    lines = [f"### {title}", "| Metric | Value |", "| --- | --- |"]
    for key, val in data.items():
        lines.append(f"| {key} | {val} |")
    return "\n".join(lines)


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
        run_single = g["run_single_dd"].value
        run_a = g["run_a_dd"].value
        run_b = g["run_b_dd"].value
        comp_a_widget = g.get("compare_a_path_txt")
        comp_b_widget = g.get("compare_b_path_txt")
        comp_a = comp_a_widget.value if comp_a_widget is not None else ""
        comp_b = comp_b_widget.value if comp_b_widget is not None else ""
    else:
        cell = g.get("cell_name")
        tunes = g.get("tunes_dir")
        model = g.get("model_dir")
        run_single = g.get("run_single_stem")
        run_a = g.get("run_compare_a")
        run_b = g.get("run_compare_b")
        comp_a = g.get("compare_a_path", "")
        comp_b = g.get("compare_b_path", "")

    base_dir = g.get("CELLS_DIR") / cell / tunes / model / "output_data"
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
    }


def output_opts_from_globals(g: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plot_outputs": g.get("plot_outputs"),
        "plot_output_curve": g.get("plot_output_curve"),
        "plot_spike_stats": g.get("plot_spike_stats"),
        "plot_raster": g.get("plot_raster"),
        "raster_style": g.get("raster_style"),
        "plot_window": g.get("plot_window"),
        "compare_output_layout": g.get("compare_output_layout"),
        "win_size": g.get("win_size"),
        "multi_plot_type": g.get("multi_plot_type"),
        "multi_shade_mode": g.get("multi_shade_mode"),
        "multi_norm_fr": g.get("multi_norm_fr"),
        "multi_use_bio_curve": g.get("multi_use_bio_curve"),
        "bio_curve_path": g.get("bio_curve_path"),
        "bio_curve_time_col": g.get("bio_curve_time_col"),
        "bio_curve_rate_col": g.get("bio_curve_rate_col"),
        "bio_curve_time_unit": g.get("bio_curve_time_unit"),
        "bio_curve_t_min": g.get("bio_curve_t_min"),
        "bio_curve_delay_ms": g.get("bio_curve_delay_ms"),
        "bio_curve_shift_ms": g.get("bio_curve_shift_ms"),
        "output_curve_mode": g.get("output_curve_mode"),
        "output_norm_mode": g.get("output_norm_mode"),
        "output_baseline_ms": g.get("output_baseline_ms"),
        "output_baseline_mode": g.get("output_baseline_mode"),
        "output_norm_window": g.get("output_norm_window"),
        "output_bin_ms": g.get("output_bin_ms"),
        "output_smooth_ms": g.get("output_smooth_ms"),
        "output_smooth_mode": g.get("output_smooth_mode"),
        "output_scatter_enabled": g.get("output_scatter_enabled"),
        "output_scatter_path": g.get("output_scatter_path"),
        "output_scatter_time_unit": g.get("output_scatter_time_unit"),
        "output_scatter_label": g.get("output_scatter_label"),
        "output_scatter_color": g.get("output_scatter_color"),
        "output_scatter_shift_ms": g.get("output_scatter_shift_ms"),
    }


def input_opts_from_globals(g: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plot_inputs_mean": g.get("plot_inputs_mean"),
        "plot_input_raster": g.get("plot_input_raster"),
        "show_input_std": g.get("show_input_std"),
        "input_groups": g.get("input_groups"),
        "input_bin_ms": g.get("input_bin_ms"),
        "input_smooth_ms": g.get("input_smooth_ms"),
        "input_raster_trial_idx": g.get("input_raster_trial_idx"),
        "input_raster_max_trains": g.get("input_raster_max_trains"),
        "input_raster_win_size": g.get("input_raster_win_size"),
        "input_raster_style": g.get("input_raster_style"),
        "input_plot_window": g.get("input_plot_window"),
        "compare_input_layout": g.get("compare_input_layout"),
        "compare_show_input_std": g.get("compare_show_input_std"),
    }


def run_output_plots_from_globals(g: Dict[str, Any]) -> None:
    sel = get_selection_from_globals(g)
    run_output_plots(
        sel,
        output_opts_from_globals(g),
        save_plots=bool(g.get("save_plots", False)),
        save_analysis=bool(g.get("save_analysis", False)),
        plots_dpi=int(g.get("plots_dpi", 150)),
    )


def run_input_plots_from_globals(g: Dict[str, Any]) -> None:
    sel = get_selection_from_globals(g)
    run_input_plots(
        sel,
        input_opts_from_globals(g),
        save_plots=bool(g.get("save_plots", False)),
        save_analysis=bool(g.get("save_analysis", False)),
        plots_dpi=int(g.get("plots_dpi", 150)),
    )


def run_output_metrics_from_globals(g: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sel = get_selection_from_globals(g)
    opts = output_opts_from_globals(g)
    opts.update({
        "output_peak_window_ms": g.get("output_peak_window_ms"),
        "output_drop_window_ms": g.get("output_drop_window_ms"),
        "output_rebound_window_ms": g.get("output_rebound_window_ms"),
        "output_auc_window": g.get("output_auc_window"),
    })
    return run_output_metrics(
        sel,
        opts,
        save_analysis=bool(g.get("save_analysis", False)),
    )


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


def build_selection_ui(g: Dict[str, Any]) -> None:
    if not g.get("use_widgets", True):
        print("Widgets disabled (use_widgets=False).")
        return
    widgets = _maybe_import_widgets()
    display, _ = _maybe_import_display()
    if widgets is None or display is None:
        print("Widgets not enabled or ipywidgets unavailable.")
        return

    base_dir = g.get("BASE_DIR")
    cells = analysis.list_cells(base_dir) or [g.get("cell_name")]
    cell_dd = widgets.Dropdown(options=cells, value=g.get("cell_name"), description="Cell")
    tunes = analysis.list_tunes(base_dir, cell_dd.value) or [g.get("tunes_dir")]
    tunes_dd = widgets.Dropdown(options=tunes, value=g.get("tunes_dir"), description="Tunes")
    models = analysis.list_models(base_dir, cell_dd.value, tunes_dd.value) or [g.get("model_dir")]
    model_dd = widgets.Dropdown(options=models, value=g.get("model_dir"), description="Model")

    run_single_dd = widgets.Dropdown(options=["latest", "previous"], value=g.get("run_single_stem"), description="Single")
    run_a_dd = widgets.Dropdown(options=["latest", "previous"], value=g.get("run_compare_a"), description="Compare A")
    run_b_dd = widgets.Dropdown(options=["none", "latest", "previous"], value=g.get("run_compare_b"), description="Compare B")

    compare_a_path_txt = widgets.Text(value=g.get("compare_a_path", ""), description="Compare A path")
    compare_b_path_txt = widgets.Text(value=g.get("compare_b_path", ""), description="Compare B path")

    def _refresh_runs(*_):
        base = g.get("CELLS_DIR") / cell_dd.value / tunes_dd.value / model_dd.value / "output_data"
        names = [analysis.run_label(p) for p in analysis.collect_run_candidates(base)]
        options = ["latest", "previous"] + names
        run_single_dd.options = options
        run_a_dd.options = options
        run_b_dd.options = ["none"] + options

    def _refresh_models(*_):
        models = analysis.list_models(base_dir, cell_dd.value, tunes_dd.value) or [g.get("model_dir")]
        model_dd.options = models
        if model_dd.value not in model_dd.options:
            model_dd.value = model_dd.options[0]
        _refresh_runs()

    cell_dd.observe(_refresh_models, names="value")
    tunes_dd.observe(_refresh_models, names="value")
    model_dd.observe(_refresh_runs, names="value")

    _refresh_runs()

    save_plots_cb = widgets.Checkbox(value=bool(g.get("save_plots")), description="Save plots")
    save_analysis_cb = widgets.Checkbox(value=bool(g.get("save_analysis")), description="Save analysis JSON")

    g["cell_dd"] = cell_dd
    g["tunes_dd"] = tunes_dd
    g["model_dd"] = model_dd
    g["run_single_dd"] = run_single_dd
    g["run_a_dd"] = run_a_dd
    g["run_b_dd"] = run_b_dd
    g["compare_a_path_txt"] = compare_a_path_txt
    g["compare_b_path_txt"] = compare_b_path_txt
    g["save_plots_cb"] = save_plots_cb
    g["save_analysis_cb"] = save_analysis_cb

    display(widgets.HBox([cell_dd, tunes_dd, model_dd, run_single_dd]))
    display(widgets.HBox([run_a_dd, run_b_dd, compare_a_path_txt, compare_b_path_txt]))
    display(widgets.HBox([save_plots_cb, save_analysis_cb]))


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

    outputs_cb = widgets.Checkbox(value=g.get("plot_outputs"), description="Outputs")
    output_curve_cb = widgets.Checkbox(value=g.get("plot_output_curve"), description="Output curve")
    spikes_cb = widgets.Checkbox(value=g.get("plot_spike_stats"), description="Spike stats")

    outputs_raster_cb = widgets.Checkbox(value=g.get("plot_raster"), description="Output raster")
    outputs_style_dd = widgets.Dropdown(options=["dot", "line"], value=g.get("raster_style"), description="Raster style")
    outputs_win_txt = widgets.FloatText(value=g.get("win_size"), description="Win size")
    window_start_txt = widgets.Text(value="" if g.get("plot_window")[0] is None else str(g.get("plot_window")[0]), description="tstart")
    window_end_txt = widgets.Text(value="" if g.get("plot_window")[1] is None else str(g.get("plot_window")[1]), description="tstop")

    output_curve_mode_dd = widgets.Dropdown(options=["raw", "normalized"], value=g.get("output_curve_mode"), description="Curve mode")
    output_norm_mode_dd = widgets.Dropdown(options=["avg", "peak"], value=g.get("output_norm_mode"), description="Norm mode")
    output_baseline_txt = widgets.FloatText(value=g.get("output_baseline_ms"), description="Baseline ms")
    output_baseline_mode_dd = widgets.Dropdown(options=["point", "window"], value=g.get("output_baseline_mode"), description="Baseline mode")
    output_norm_window_dd = widgets.Dropdown(options=["stim", "full"], value=g.get("output_norm_window"), description="Norm window")
    output_bin_txt = widgets.Text(value="" if g.get("output_bin_ms") is None else str(g.get("output_bin_ms")), description="Curve bin ms")
    output_smooth_txt = widgets.Text(value="" if g.get("output_smooth_ms") is None else str(g.get("output_smooth_ms")), description="Curve smooth ms")
    output_smooth_mode_dd = widgets.Dropdown(options=["causal", "center"], value=g.get("output_smooth_mode"), description="Curve smooth mode")

    output_scatter_cb = widgets.Checkbox(value=g.get("output_scatter_enabled"), description="Scatter curve")
    output_scatter_path_txt = widgets.Text(value=g.get("output_scatter_path"), description="Scatter path")
    output_scatter_unit_dd = widgets.Dropdown(options=["s", "ms"], value=g.get("output_scatter_time_unit"), description="Scatter unit")
    output_scatter_label_txt = widgets.Text(value=g.get("output_scatter_label"), description="Scatter label")
    output_scatter_color_txt = widgets.Text(value=g.get("output_scatter_color"), description="Scatter color")
    output_scatter_shift_txt = widgets.Text(value="" if g.get("output_scatter_shift_ms") is None else str(g.get("output_scatter_shift_ms")), description="Scatter shift ms")

    outputs_plot_type_dd = widgets.Dropdown(options=["line", "hist", "both"], value=g.get("multi_plot_type"), description="Plot type")
    shade_val = g.get("multi_shade_mode")
    outputs_shade_dd = widgets.Dropdown(options=["none", "sem", "std"], value="none" if shade_val is None else shade_val, description="Shade")
    outputs_norm_txt = widgets.Text(value="" if g.get("multi_norm_fr") is None else str(g.get("multi_norm_fr")), description="Norm FR")
    outputs_compare_layout_dd = widgets.Dropdown(options=["side-by-side", "stacked", "overlay"], value=g.get("compare_output_layout"), description="Compare layout")

    outputs_bio_cb = widgets.Checkbox(value=g.get("multi_use_bio_curve"), description="Bio curve")
    outputs_bio_path_txt = widgets.Text(value=g.get("bio_curve_path"), description="Bio path")
    outputs_bio_shift_txt = widgets.Text(value="" if g.get("bio_curve_shift_ms") is None else str(g.get("bio_curve_shift_ms")), description="Bio shift")

    outputs_btn = widgets.Button(description="Run output plots")

    def _on_outputs(_):
        sync_common_from_globals(g)
        g["plot_outputs"] = outputs_cb.value
        g["plot_output_curve"] = output_curve_cb.value
        g["plot_spike_stats"] = spikes_cb.value
        g["plot_raster"] = outputs_raster_cb.value
        g["raster_style"] = outputs_style_dd.value
        g["win_size"] = float(outputs_win_txt.value)
        g["plot_window"] = (
            analysis.parse_optional_float(window_start_txt.value),
            analysis.parse_optional_float(window_end_txt.value),
        )
        g["output_curve_mode"] = output_curve_mode_dd.value
        g["output_norm_mode"] = output_norm_mode_dd.value
        g["output_baseline_ms"] = float(output_baseline_txt.value)
        g["output_baseline_mode"] = output_baseline_mode_dd.value
        g["output_norm_window"] = output_norm_window_dd.value
        g["output_bin_ms"] = analysis.parse_optional_float(output_bin_txt.value)
        g["output_smooth_ms"] = analysis.parse_optional_float(output_smooth_txt.value)
        g["output_smooth_mode"] = output_smooth_mode_dd.value
        g["output_scatter_enabled"] = output_scatter_cb.value
        g["output_scatter_path"] = output_scatter_path_txt.value.strip()
        g["output_scatter_time_unit"] = output_scatter_unit_dd.value
        g["output_scatter_label"] = output_scatter_label_txt.value.strip()
        g["output_scatter_color"] = output_scatter_color_txt.value.strip()
        g["output_scatter_shift_ms"] = analysis.parse_optional_float(output_scatter_shift_txt.value)
        g["multi_plot_type"] = outputs_plot_type_dd.value
        shade_val_local = outputs_shade_dd.value
        g["multi_shade_mode"] = None if shade_val_local in ("none", "", None) else shade_val_local
        g["multi_norm_fr"] = analysis.parse_optional_float(outputs_norm_txt.value)
        g["compare_output_layout"] = outputs_compare_layout_dd.value
        g["multi_use_bio_curve"] = outputs_bio_cb.value
        g["bio_curve_path"] = outputs_bio_path_txt.value.strip()
        g["bio_curve_shift_ms"] = analysis.parse_optional_float(outputs_bio_shift_txt.value)

        with out_outputs:
            out_outputs.clear_output()
            run_output_plots_from_globals(g)

    outputs_btn.on_click(_on_outputs)

    g["out_outputs"] = out_outputs
    g["_on_outputs"] = _on_outputs

    display(
        widgets.VBox([
            widgets.HBox([outputs_btn, outputs_cb, output_curve_cb, spikes_cb]),
            widgets.HBox([outputs_raster_cb, outputs_style_dd, outputs_win_txt]),
            widgets.HBox([window_start_txt, window_end_txt]),
            widgets.HBox([output_curve_mode_dd, output_norm_mode_dd, output_baseline_txt, output_baseline_mode_dd, output_norm_window_dd]),
            widgets.HBox([output_bin_txt, output_smooth_txt, output_smooth_mode_dd]),
            widgets.HBox([output_scatter_cb, output_scatter_path_txt, output_scatter_unit_dd]),
            widgets.HBox([output_scatter_label_txt, output_scatter_color_txt, output_scatter_shift_txt]),
            widgets.HBox([outputs_plot_type_dd, outputs_shade_dd, outputs_norm_txt]),
            widgets.HBox([outputs_compare_layout_dd]),
            widgets.HBox([outputs_bio_cb, outputs_bio_path_txt, outputs_bio_shift_txt]),
            out_outputs,
        ])
    )

    if g.get("auto_run_outputs"):
        _on_outputs(None)


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

    inputs_mean_cb = widgets.Checkbox(value=g.get("plot_inputs_mean"), description="Inputs mean")
    inputs_raster_cb = widgets.Checkbox(value=g.get("plot_input_raster"), description="Inputs raster")
    inputs_std_cb = widgets.Checkbox(value=g.get("show_input_std"), description="Show std")

    input_groups = g.get("input_groups")
    inputs_groups_txt = widgets.Text(value="" if input_groups is None else ",".join(input_groups), description="Groups")
    inputs_bin_txt = widgets.Text(value="" if g.get("input_bin_ms") is None else str(g.get("input_bin_ms")), description="Bin ms")
    inputs_smooth_txt = widgets.Text(value="" if g.get("input_smooth_ms") is None else str(g.get("input_smooth_ms")), description="Smooth ms")

    raster_trial_txt = widgets.IntText(value=g.get("input_raster_trial_idx"), description="Raster trial")
    raster_max_txt = widgets.IntText(value=g.get("input_raster_max_trains"), description="Max trains")
    raster_win_txt = widgets.FloatText(value=g.get("input_raster_win_size"), description="Raster win")
    raster_style_dd = widgets.Dropdown(options=["dot", "line"], value=g.get("input_raster_style"), description="Raster style")

    input_window = g.get("input_plot_window")
    input_window_start_txt = widgets.Text(value="" if input_window[0] is None else str(input_window[0]), description="tstart")
    input_window_end_txt = widgets.Text(value="" if input_window[1] is None else str(input_window[1]), description="tstop")

    compare_layout_dd = widgets.Dropdown(options=["side-by-side", "stacked", "overlay"], value=g.get("compare_input_layout"), description="Layout")
    compare_std_cb = widgets.Checkbox(value=g.get("compare_show_input_std"), description="Compare std")

    inputs_btn = widgets.Button(description="Run input plots")

    def _on_inputs(_):
        sync_common_from_globals(g)
        g["plot_inputs_mean"] = inputs_mean_cb.value
        g["plot_input_raster"] = inputs_raster_cb.value
        g["show_input_std"] = inputs_std_cb.value
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
        g["compare_input_layout"] = compare_layout_dd.value
        g["compare_show_input_std"] = compare_std_cb.value

        with out_inputs:
            out_inputs.clear_output()
            run_input_plots_from_globals(g)

    inputs_btn.on_click(_on_inputs)

    g["out_inputs"] = out_inputs
    g["_on_inputs"] = _on_inputs

    display(
        widgets.VBox([
            widgets.HBox([inputs_btn, inputs_mean_cb, inputs_raster_cb, inputs_std_cb]),
            widgets.HBox([inputs_groups_txt, inputs_bin_txt, inputs_smooth_txt]),
            widgets.HBox([raster_trial_txt, raster_max_txt, raster_win_txt, raster_style_dd]),
            widgets.HBox([input_window_start_txt, input_window_end_txt]),
            widgets.HBox([compare_layout_dd, compare_std_cb]),
            out_inputs,
        ])
    )

    if g.get("auto_run_inputs"):
        _on_inputs(None)


def run_extra_tables_from_globals(g: Dict[str, Any]) -> None:
    sel, run_dir, res = resolve_single_from_globals(g)
    if not g.get("load_cell_for_analysis"):
        print("Cell loading disabled; enable load_cell_for_analysis to run this section.")
        return

    tune_dir = (g.get("CELLS_DIR") / sel["cell"] / sel["tunes"] / sel["model"]).resolve()
    try:
        cell, geom, geom_cfg = analysis.load_cell_and_geometry(tune_dir)
    except Exception as exc:
        print("Cell/geometry load failed:", exc)
        return

    if g.get("extra_cell_tables"):
        cell_sections = analysis.summarize_cell_sections(cell)
        mech_summary = analysis.summarize_mechanisms(cell)
        show_md(analysis.format_section_summary_table(cell_sections, title=f"{analysis.run_label(run_dir)} cell sections"))
        show_md(analysis.format_mechanism_summary_table(mech_summary, title="Mechanisms (per section group)"))
        analysis.save_json(cell_sections, analysis.analysis_dir_for_run(run_dir) / "cell_sections.json", enabled=bool(g.get("save_analysis", False)))
        analysis.save_json(mech_summary, analysis.analysis_dir_for_run(run_dir) / "cell_mechanisms.json", enabled=bool(g.get("save_analysis", False)))

    if g.get("extra_geometry_tables"):
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


def run_extra_compare_from_globals(g: Dict[str, Any]) -> None:
    sel, run_a, run_b, res_a, res_b = resolve_compare_from_globals(g)
    if run_b is None:
        print("Comparison disabled (set Compare B to a run name).")
        return

    label_a = analysis.run_label(run_a)
    label_b = analysis.run_label(run_b)
    tune_dir = (g.get("CELLS_DIR") / sel["cell"] / sel["tunes"] / sel["model"]).resolve()

    cell = None
    geom = None
    geom_cfg = None
    if g.get("load_cell_for_analysis") and (
        g.get("extra_compare_cell_tables")
        or g.get("extra_compare_geometry_tables")
        or g.get("extra_compare_synapse_tables")
    ):
        try:
            cell, geom, geom_cfg = analysis.load_cell_and_geometry(tune_dir)
        except Exception as exc:
            print("Cell/geometry load failed:", exc)

    if g.get("extra_compare_cell_tables") and cell is not None:
        cell_sections = analysis.summarize_cell_sections(cell)
        mech_summary = analysis.summarize_mechanisms(cell)
        show_md(
            analysis.format_section_summary_compare(
                cell_sections,
                cell_sections,
                labels=(label_a, label_b),
                diff_only=g.get("extra_compare_diff_only"),
                title="Cell sections",
            )
        )
        show_md(
            analysis.format_mechanism_summary_compare(
                mech_summary,
                mech_summary,
                labels=(label_a, label_b),
                diff_only=g.get("extra_compare_diff_only"),
                title="Mechanisms",
            )
        )

    if g.get("extra_compare_geometry_tables") and geom is not None:
        geom_summary = analysis.summarize_geometry(geom, geom_config=geom_cfg)
        show_md(
            analysis.format_geometry_summary_compare(
                geom_summary,
                geom_summary,
                labels=(label_a, label_b),
                diff_only=g.get("extra_compare_diff_only"),
                title="Geometry distances",
            )
        )

    syn_summary_a = analysis.summarize_synapse_records(
        res_a.get("syn_records") or {},
        geom=geom,
        duration_ms=analysis._get_duration_ms(res_a),
    )
    syn_summary_b = analysis.summarize_synapse_records(
        res_b.get("syn_records") or {},
        geom=geom,
        duration_ms=analysis._get_duration_ms(res_b),
    )

    if g.get("extra_compare_synapse_tables") and (syn_summary_a.get("groups") or syn_summary_b.get("groups")):
        show_md(
            analysis.format_synapse_summary_compare(
                syn_summary_a,
                syn_summary_b,
                labels=(label_a, label_b),
                diff_only=g.get("extra_compare_diff_only"),
                title="Synapse summary",
            )
        )

    if g.get("extra_synapse_weight_plot") or g.get("extra_synapse_distance_plot"):
        syn_groups = g.get("extra_synapse_groups")
        vals_w_a = analysis.extract_synapse_values(res_a.get("syn_records") or {}, "weight", syn_groups)
        vals_w_b = analysis.extract_synapse_values(res_b.get("syn_records") or {}, "weight", syn_groups)
        vals_d_a = analysis.extract_synapse_values(res_a.get("syn_records") or {}, "distance", syn_groups)
        vals_d_b = analysis.extract_synapse_values(res_b.get("syn_records") or {}, "distance", syn_groups)

        if g.get("extra_synapse_weight_plot"):
            fig_w = plotting.plot_synapse_compare_hist(
                vals_w_a,
                vals_w_b,
                labels=(label_a, label_b),
                bin_width=g.get("extra_synapse_weight_bin"),
                xlabel="Synaptic weight",
                title="Synapse weight distribution",
                density=g.get("extra_synapse_density"),
            )
            if fig_w is not None:
                analysis.save_figure(
                    fig_w,
                    analysis.plot_dir_for_compare(sel["base"], run_a, run_b) / "syn_weight_compare.png",
                    enabled=bool(g.get("save_plots", False)),
                )

        if g.get("extra_synapse_distance_plot"):
            fig_d = plotting.plot_synapse_compare_hist(
                vals_d_a,
                vals_d_b,
                labels=(label_a, label_b),
                bin_width=g.get("extra_synapse_distance_bin"),
                xlabel="Distance from soma (um)",
                title="Synapse distance distribution",
                density=g.get("extra_synapse_density"),
            )
            if fig_d is not None:
                analysis.save_figure(
                    fig_d,
                    analysis.plot_dir_for_compare(sel["base"], run_a, run_b) / "syn_distance_compare.png",
                    enabled=bool(g.get("save_plots", False)),
                )


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
