from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt

from modules_local import analysis, plotting, run_sim


def compare_enabled(selection: Dict[str, Any]) -> bool:
    return selection.get("run_b") not in (None, "none", "", "None")


def resolve_single(selection: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    run_dir = analysis.resolve_run(selection["base"], selection["run_single"])
    res = run_sim.load_results(run_dir)
    return run_dir, res


def resolve_compare(
    selection: Dict[str, Any],
) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not compare_enabled(selection):
        return None, None, None, None
    run_a = analysis.resolve_run(selection["base"], selection["run_a"])
    run_b = analysis.resolve_run(selection["base"], selection["run_b"])
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

        if opts.get("plot_outputs", True):
            fig_cmp, _ = plotting.plot_compare_side_by_side(
                res_a,
                res_b,
                labels=(label_a, label_b),
                win_size=opts.get("win_size", 25),
                plot_window=opts.get("plot_window", (None, None)),
                smooth_mode=smooth_mode,
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
            if curve_a and curve_b:
                curve_a = analysis.normalize_output_curve(
                    curve_a,
                    res_a.get("sim_cfg", {}) or {},
                    mode=opts.get("output_curve_mode", "raw"),
                    norm_mode=opts.get("output_norm_mode", "avg"),
                    baseline_ms=opts.get("output_baseline_ms", 100.0),
                    norm_window=opts.get("output_norm_window", "stim"),
                )
                curve_b = analysis.normalize_output_curve(
                    curve_b,
                    res_b.get("sim_cfg", {}) or {},
                    mode=opts.get("output_curve_mode", "raw"),
                    norm_mode=opts.get("output_norm_mode", "avg"),
                    baseline_ms=opts.get("output_baseline_ms", 100.0),
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
            plot_bio = None
            if in_vivo_curve is not None:
                plot_bio = (True, in_vivo_curve[0], in_vivo_curve[1])
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
        if curve_single:
            curve_single = analysis.normalize_output_curve(
                curve_single,
                res.get("sim_cfg", {}) or {},
                mode=opts.get("output_curve_mode", "raw"),
                norm_mode=opts.get("output_norm_mode", "avg"),
                baseline_ms=opts.get("output_baseline_ms", 100.0),
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
        norm_window=opts.get("output_norm_window", "stim"),
    )
    metrics = analysis.compute_output_metrics(
        curve,
        res.get("sim_cfg", {}) or {},
        peak_window_ms=opts.get("output_peak_window_ms", 100.0),
        drop_window_ms=opts.get("output_drop_window_ms", 100.0),
        auc_window=opts.get("output_auc_window", "stim"),
    )
    _save_json(
        metrics,
        analysis.analysis_dir_for_run(run_dir) / "output_metrics.json",
        enabled=save_analysis,
    )
    return metrics
