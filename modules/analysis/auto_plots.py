"""Automatic plot saving used by Step 5 result saving.

This module is the explicit boundary between the simulation backend and the
heavier analysis/plotting package. The simulation pipeline should import this
module when `sim_config["save_plots"]` is enabled, rather than importing the
general analysis helper module directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import json
import matplotlib.pyplot as plt

from .analysis import (
    group_colors_from_results,
    resolve_scp_root_for_results,
    save_figure,
    summarize_inputs_from_results,
)


def _close_figure(fig: Any) -> None:
    if fig is not None:
        try:
            plt.close(fig)
        except Exception:
            pass


def save_default_plots(
    results: Dict[str, Any],
    run_dir: Union[str, Path],
    *,
    save_inputs: bool = True,
    save_synapses: bool = False,
    win_size: float = 25.0,
    input_bin_ms: Optional[float] = None,
    input_smooth_ms: Optional[float] = 25.0,
    raster_style: str = "dot",
    plot_mode: str = "default",
    single_plot_preset: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> Dict[str, Path]:
    """
    Save a small set of default plots into <run_dir>/plots.

    plot_mode:
      - "default": standard output/input/synapse plots
      - "single_plot": compact composite plot via preset JSON

    Returns a dict of plot name -> file path.
    """
    from . import plotting  # local import to avoid circular deps

    run_dir = Path(run_dir)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, Path] = {}
    mode = str(plot_mode or "default").strip().lower()

    if mode in {"single_plot", "single-panel", "single_plot_panel", "single-plot-panel"}:
        from . import single_plot_panel  # local import to avoid circular deps

        repo_root = resolve_scp_root_for_results(results, run_dir)
        preset_path = (
            repo_root / "modules" / "analysis" / "analysis_presets" / "single_plot.json"
            if single_plot_preset in (None, "", False)
            else Path(str(single_plot_preset)).expanduser()
        )
        if not preset_path.is_absolute():
            preset_path = (repo_root / preset_path).resolve()

        panel_cfg: Dict[str, Any] = {}
        try:
            payload = json.loads(preset_path.read_text())
            if isinstance(payload, dict):
                defaults = payload.get("defaults", payload)
                if isinstance(defaults, dict):
                    panel_cfg = dict(defaults)
        except Exception as exc:
            print(f"save_plots single_plot preset load failed ({preset_path}): {exc}")

        panel_cfg.setdefault("export_path", "single_plot_panel")
        panel_cfg.setdefault("export_formats", ["svg"])
        panel_cfg.setdefault("dpi", 150)
        panel_cfg.setdefault("export_overwrite", bool(overwrite))

        panel_result = single_plot_panel.plot_single_plot_panel_from_results(
            results,
            run_dir=run_dir,
            **panel_cfg,
        )
        fig = panel_result.get("fig")
        for w in panel_result.get("warnings", []) or []:
            print(f"save_plots single_plot warning: {w}")

        exported = [Path(p) for p in (panel_result.get("exported_paths") or [])]
        requested = [Path(p) for p in (panel_result.get("requested_export_paths") or [])]
        if exported:
            for i, path in enumerate(exported):
                key = "single_plot" if i == 0 else f"single_plot_{i}"
                saved[key] = path
        elif not requested:
            if fig is not None:
                fallback_path = plot_dir / "single_plot.png"
                saved_path = save_figure(
                    fig,
                    fallback_path,
                    enabled=True,
                    dpi=150,
                    overwrite=overwrite,
                )
                if saved_path is not None:
                    saved["single_plot"] = saved_path
        _close_figure(fig)
        return saved

    if mode != "default":
        print(f"save_default_plots: unknown plot_mode={plot_mode!r}; using default mode.")

    # Output plot
    fig_out = plotting.plot_results(
        results,
        syn_records=results.get("syn_records"),
        win_size=win_size,
        raster_style=raster_style,
        plot_window=(None, None),
    )
    out_path = plot_dir / "output_plot.png"
    fig_out = fig_out[0] if isinstance(fig_out, tuple) else fig_out
    saved_path = save_figure(
        fig_out,
        out_path,
        enabled=True,
        dpi=150,
        overwrite=overwrite,
    )
    _close_figure(fig_out)
    if saved_path is not None:
        saved["output_plot"] = saved_path

    # Input mean curves
    if save_inputs:
        try:
            summary = summarize_inputs_from_results(
                results,
                bin_ms=input_bin_ms,
                smooth_ms=input_smooth_ms,
            )
            group_colors = group_colors_from_results(results)
            fig_in, _ = plotting.plot_input_means(
                summary,
                label="inputs",
                groups=None,
                show_std=False,
                output_curve=(results.get("meta") or {}).get("avg_rate_curve"),
                group_colors=group_colors,
            )
            in_path = plot_dir / "inputs_mean.png"
            saved_path = save_figure(
                fig_in,
                in_path,
                enabled=True,
                dpi=150,
                overwrite=overwrite,
            )
            _close_figure(fig_in)
            if saved_path is not None:
                saved["inputs_mean"] = saved_path
        except Exception:
            pass

    # Synapse plots (optional)
    if save_synapses:
        syn_recs = results.get("syn_records") or {}
        if syn_recs:
            plotted_groups = list(syn_recs.keys())
            plotting.plot_syn_records(
                results.get("cell", None),
                syn_recs,
                plotted_groups=plotted_groups,
                plotted_props=["weight_probability"],
                plot_type="hist",
                bins=0.1,
                win_size=0.1,
            )
            syn_path = plot_dir / "syn_weight_prob.png"
            fig_syn = plt.gcf()
            saved_path = save_figure(
                fig_syn,
                syn_path,
                enabled=True,
                dpi=150,
                overwrite=overwrite,
            )
            _close_figure(fig_syn)
            if saved_path is not None:
                saved["syn_weight_prob"] = saved_path

    return saved
