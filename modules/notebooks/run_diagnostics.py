"""Notebook display helpers for Step 5 run diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np


def _spike_counts(spikes: Any) -> list[int]:
    if spikes is None:
        return []
    if isinstance(spikes, list):
        return [len(np.asarray(trial)) for trial in spikes]
    return [len(np.asarray(spikes))]


def _plot_vm_summary(
    results: dict[str, Any],
    *,
    cell_name: Optional[str] = None,
    tune_name: Optional[str] = None,
    max_traces: int = 3,
    trial_idx: Optional[int] = None,
    plot_window: Any = None,
    figsize: tuple[float, float] = (8.0, 4.0),
) -> Optional[Any]:
    """Plot saved Vm traces only, returning the Matplotlib figure if available."""
    import matplotlib.pyplot as plt

    traces = results.get("traces", {}) or {}
    time_ms = traces.get("T")
    voltage = traces.get("V")

    if time_ms is None or voltage is None:
        print("No Vm trace stored. Increase n_traces_to_save or cell_recording settings if needed.")
        return None

    fig = plt.figure(figsize=figsize)
    if isinstance(voltage, list):
        if trial_idx is None:
            selected = list(enumerate(voltage[: max(1, int(max_traces))]))
        else:
            idx = min(max(int(trial_idx), 0), max(0, len(voltage) - 1))
            selected = [(idx, voltage[idx])] if voltage else []
        for idx, trace in selected:
            plt.plot(time_ms, trace, lw=1.0, label=f"trial {idx}")
        if len(selected) > 1:
            plt.legend()
    else:
        plt.plot(time_ms, voltage, lw=1.2)

    plt.xlabel("Time (ms)")
    plt.ylabel("Vm (mV)")
    label_bits = [bit for bit in (cell_name, tune_name, results.get("mode")) if bit]
    plt.title(" / ".join(str(bit) for bit in label_bits) if label_bits else "Vm trace")
    if isinstance(plot_window, (list, tuple)) and len(plot_window) >= 2:
        x0, x1 = plot_window[:2]
        if x0 is not None or x1 is not None:
            plt.xlim(left=x0, right=x1)
    plt.tight_layout()
    return fig


def _print_summary(results: dict[str, Any]) -> dict[str, Any]:
    mode = results.get("mode")
    counts = _spike_counts(results.get("spikes"))

    print("Mode:", mode)
    if counts:
        if len(counts) <= 20:
            print("Spike counts:", counts)
        else:
            print("Trials:", len(counts))
            print(
                "Spike-count range:",
                f"{min(counts)}–{max(counts)}",
                "| total:",
                sum(counts),
            )
        print("Mean spikes/trial:", float(np.mean(counts)))
    elif mode == "iclamp":
        meta = results.get("meta", {}) or {}
        frequency = meta.get("frequency_hz")
        if frequency is not None:
            print("IClamp frequency (Hz):", float(frequency))
        else:
            print("No spike vector found; this is expected for some IClamp runs.")
    else:
        print("No spike vector found.")

    return {"mode": mode, "spike_counts": counts}


def _standard_plot(
    results: dict[str, Any],
    *,
    include_inputs: bool = True,
    plot_window: Any = None,
    win_size: Optional[float] = None,
    raster_style: Optional[str] = None,
    max_traces: int = 3,
    cell_name: Optional[str] = None,
    tune_name: Optional[str] = None,
) -> Any:
    """Use the standard analysis plotting wrapper, adding saved Vm traces for multi runs."""
    result_mode = results.get("mode")
    if result_mode == "iclamp":
        return _plot_vm_summary(
            results,
            cell_name=cell_name,
            tune_name=tune_name,
            max_traces=max_traces,
        )

    from modules.analysis import plotting

    sim_cfg = results.get("sim_cfg", {}) or {}
    if win_size is None:
        win_size = float(sim_cfg.get("plots_win_size", 25.0))
    if raster_style is None:
        raster_style = str(sim_cfg.get("plots_raster_style", "dot"))

    effective_include_inputs = bool(include_inputs)
    if result_mode == "single" and not results.get("syn_records"):
        effective_include_inputs = False

    analysis_plot = plotting.plot_results(
        results,
        syn_records=results.get("syn_records"),
        win_size=win_size,
        rate_style="line" if effective_include_inputs else None,
        raster_style=raster_style if effective_include_inputs else None,
        plot_window=plot_window or (None, None),
        bin_ms=sim_cfg.get("plots_input_bin_ms", None),
    )
    if result_mode == "multi":
        vm_fig = _plot_vm_summary(
            results,
            cell_name=cell_name,
            tune_name=tune_name,
            max_traces=max_traces,
        )
        return {"analysis_plot": analysis_plot, "vm_fig": vm_fig}
    return analysis_plot


def _single_plot_diagnostic(
    results: dict[str, Any],
    *,
    repo_root: Optional[Union[str, Path]] = None,
    preset_path: Optional[Union[str, Path]] = None,
    include_inputs: bool = True,
    plot_options: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Use the single-plot panel helper without exporting files."""
    if results.get("mode") == "iclamp":
        fig = _plot_vm_summary(results)
        return {"fig": fig, "warnings": ["single-plot diagnostics unavailable for IClamp"]}

    from modules.analysis import single_plot_panel

    preset = single_plot_panel.load_single_plot_preset(
        repo_root=repo_root,
        preset_path=preset_path,
    )
    panel_cfg = dict(preset.get("config", {}) or {})
    warnings = list(preset.get("warnings", []) or [])

    allowed_overrides = {
        "trial_idx",
        "top_input_groups",
        "raster_input_groups",
        "input_bin_ms",
        "input_smooth_ms",
        "input_raster_style",
        "include_input_rate",
        "include_input_raster",
        "include_vm",
        "include_output_rate",
        "include_output_raster",
        "output_raster_style",
        "output_recompute_bin_ms",
        "output_recompute_smooth_ms",
        "plot_window",
        "auto_plot_window_from_stim",
        "plot_window_adjustment_ms",
        "show_stim_lines",
        "figsize",
    }
    for key, value in dict(plot_options or {}).items():
        if key in allowed_overrides and value is not None:
            panel_cfg[key] = value

    panel_cfg["export_path"] = None
    panel_cfg["export_formats"] = None
    panel_cfg["include_input_rate"] = bool(
        include_inputs and panel_cfg.get("include_input_rate", True)
    )
    panel_cfg["include_input_raster"] = bool(
        include_inputs and panel_cfg.get("include_input_raster", False)
    )
    panel_cfg["show_input_legend"] = bool(
        include_inputs and panel_cfg.get("show_input_legend", False)
    )
    if not include_inputs:
        panel_cfg["top_input_groups"] = []
        panel_cfg["raster_input_groups"] = []

    result = single_plot_panel.plot_single_plot_panel_from_results(
        results, **panel_cfg
    )
    result["warnings"] = warnings + list(result.get("warnings", []) or [])
    for warning in result["warnings"]:
        print(f"diagnostic single-plot warning: {warning}")
    return result


def _custom_plot_diagnostic(
    results: dict[str, Any],
    *,
    plots: Sequence[str],
    include_inputs: bool,
    cell_name: Optional[str],
    tune_name: Optional[str],
    repo_root: Optional[Union[str, Path]],
    preset_path: Optional[Union[str, Path]],
    plot_options: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = {str(value) for value in plots}
    options = dict(plot_options or {})
    if not selected:
        return {"fig": None, "warnings": ["No diagnostic plots selected."]}

    if results.get("mode") == "iclamp":
        unsupported = selected - {"membrane_voltage"}
        warnings = []
        if unsupported:
            warnings.append(
                "IClamp results currently support the membrane-voltage panel "
                "in the compact notebook; use Step 6 for specialized analysis."
            )
        fig = None
        if "membrane_voltage" in selected:
            fig = _plot_vm_summary(
                results,
                cell_name=cell_name,
                tune_name=tune_name,
                trial_idx=int(options.get("trial_idx", 0)),
                plot_window=options.get("plot_window"),
                figsize=tuple(options.get("figsize", (8.0, 4.0))),
            )
        for warning in warnings:
            print(f"diagnostic plot warning: {warning}")
        return {"fig": fig, "warnings": warnings}

    options.update(
        {
            "include_input_rate": bool(
                include_inputs and "input_rate" in selected
            ),
            "include_input_raster": bool(
                include_inputs and "input_raster" in selected
            ),
            "include_vm": "membrane_voltage" in selected,
            "include_output_rate": "output_rate" in selected,
            "include_output_raster": "output_raster" in selected,
        }
    )
    return _single_plot_diagnostic(
        results,
        repo_root=repo_root,
        preset_path=preset_path,
        include_inputs=include_inputs,
        plot_options=options,
    )


def show_run_diagnostics(
    results: dict[str, Any],
    *,
    diagnostic_plot: Optional[str] = "standard",
    include_inputs: bool = True,
    cell_name: Optional[str] = None,
    tune_name: Optional[str] = None,
    max_traces: int = 3,
    plot_window: Any = None,
    repo_root: Optional[Union[str, Path]] = None,
    single_plot_preset_path: Optional[Union[str, Path]] = None,
    diagnostic_plots: Optional[Sequence[str]] = None,
    plot_options: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """
    Show Step 5 notebook diagnostics.

    `diagnostic_plot` options:
      - "summary": spike counts + Vm trace only
      - "standard": Step 5 standard plot; multi runs also show saved Vm traces
      - "single_plot": compact composite plot using the single_plot preset
      - "custom": selected compact panels using `diagnostic_plots` and
        `plot_options`
      - None/"off"/False: print counts only
    """
    summary = _print_summary(results)
    mode = "off" if diagnostic_plot in (None, "", False) else str(diagnostic_plot).strip().lower()

    payload: dict[str, Any] = {"summary": summary, "plot_mode": mode}
    if mode in {"off", "none", "false"}:
        return payload

    try:
        if mode == "summary":
            payload["fig"] = _plot_vm_summary(
                results,
                cell_name=cell_name,
                tune_name=tune_name,
                max_traces=max_traces,
            )
        elif mode == "standard":
            payload["fig"] = _standard_plot(
                results,
                include_inputs=include_inputs,
                plot_window=plot_window,
                max_traces=max_traces,
                cell_name=cell_name,
                tune_name=tune_name,
            )
        elif mode == "single_plot":
            payload["single_plot"] = _single_plot_diagnostic(
                results,
                repo_root=repo_root,
                preset_path=single_plot_preset_path,
                include_inputs=include_inputs,
                plot_options=plot_options,
            )
        elif mode == "custom":
            payload["custom_plot"] = _custom_plot_diagnostic(
                results,
                plots=list(diagnostic_plots or []),
                include_inputs=include_inputs,
                cell_name=cell_name,
                tune_name=tune_name,
                repo_root=repo_root,
                preset_path=single_plot_preset_path,
                plot_options=plot_options,
            )
        else:
            print(f"Unknown diagnostic_plot={diagnostic_plot!r}; using summary plot.")
            payload["plot_mode"] = "summary"
            payload["fig"] = _plot_vm_summary(
                results,
                cell_name=cell_name,
                tune_name=tune_name,
                max_traces=max_traces,
            )
    except Exception as exc:
        print(f"Diagnostic plot failed ({mode}): {exc}")
        payload["plot_error"] = str(exc)
        payload["fig"] = _plot_vm_summary(
            results,
            cell_name=cell_name,
            tune_name=tune_name,
            max_traces=max_traces,
        )

    return payload
