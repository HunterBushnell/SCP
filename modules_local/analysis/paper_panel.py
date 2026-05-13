from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

from . import analysis


def _resolve_stim_window(sim_cfg: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
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


def _coerce_plot_window(plot_window: Any) -> tuple[Optional[float], Optional[float]]:
    if isinstance(plot_window, (list, tuple)) and len(plot_window) >= 2:
        try:
            x0 = float(plot_window[0]) if plot_window[0] is not None else None
        except Exception:
            x0 = None
        try:
            x1 = float(plot_window[1]) if plot_window[1] is not None else None
        except Exception:
            x1 = None
        return x0, x1
    return None, None


def _resolve_effective_plot_window(
    sim_cfg: Dict[str, Any],
    *,
    plot_window: Optional[tuple[Optional[float], Optional[float]]],
    auto_plot_window_from_stim: bool,
    plot_window_adjustment_ms: float,
    warnings: list[str],
) -> tuple[Optional[float], Optional[float]]:
    manual = _coerce_plot_window(plot_window)
    if not bool(auto_plot_window_from_stim):
        return manual

    try:
        adjust = abs(float(plot_window_adjustment_ms))
    except Exception:
        adjust = 100.0

    stim_start, stim_stop = _resolve_stim_window(sim_cfg)
    if stim_start is None or stim_stop is None:
        warnings.append(
            "Paper panel: auto plot window enabled but stim start/stop unavailable; using manual plot_window."
        )
        return manual

    if stim_stop < stim_start:
        stim_start, stim_stop = stim_stop, stim_start
    return float(stim_start) - adjust, float(stim_stop) + adjust


def _smooth_rate(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bin_ms: float,
    smooth_ms: Optional[float],
    smooth_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if smooth_ms in (None, 0):
        return x, y
    try:
        smooth_ms_f = float(smooth_ms)
    except Exception:
        return x, y
    if smooth_ms_f <= 0 or bin_ms <= 0:
        return x, y

    k = int(round(smooth_ms_f / float(bin_ms)))
    if k <= 1 or y.size < k:
        return x, y

    mode = str(smooth_mode or "center").lower()
    if mode == "center" and k % 2 == 0:
        k += 1

    kernel = np.ones(k, dtype=float) / float(k)
    if mode == "causal":
        y_s = np.convolve(np.pad(y, (k - 1, 0)), kernel, mode="valid")
        x_s = x[: len(y_s)]
        return x_s, y_s

    y_s = np.convolve(y, kernel, mode="valid")
    drop = (len(x) - len(y_s)) // 2
    if drop < 0:
        return x, y
    x_s = x[drop : drop + len(y_s)]
    return x_s, y_s


def _coerce_trials(spikes: Any) -> list[np.ndarray]:
    if spikes is None:
        return []
    if isinstance(spikes, np.ndarray):
        if spikes.dtype == object:
            if spikes.ndim == 0:
                return [np.asarray(spikes.item(), dtype=float).ravel()]
            return [np.asarray(tr, dtype=float).ravel() for tr in spikes.tolist()]
        if spikes.ndim <= 1:
            return [np.asarray(spikes, dtype=float).ravel()]
        return [np.asarray(tr, dtype=float).ravel() for tr in spikes]
    if isinstance(spikes, (list, tuple)):
        if not spikes:
            return []
        first = spikes[0]
        if np.isscalar(first):
            return [np.asarray(spikes, dtype=float).ravel()]
        return [np.asarray(tr, dtype=float).ravel() for tr in spikes]
    return [np.asarray(spikes, dtype=float).ravel()]


def _resolve_groups(
    requested: Optional[Sequence[str]],
    available: Sequence[str],
    *,
    panel_name: str,
    warnings: list[str],
) -> list[str]:
    avail = [str(g) for g in available]
    if requested is None or len(requested) == 0:
        if avail:
            warnings.append(f"{panel_name}: group list empty, auto-using all groups ({len(avail)}).")
        return avail
    selected = [str(g) for g in requested if str(g) in avail]
    missing = [str(g) for g in requested if str(g) not in avail]
    if missing:
        warnings.append(f"{panel_name}: missing groups skipped: {missing}")
    if not selected and avail:
        warnings.append(f"{panel_name}: no requested groups matched, auto-using all groups.")
        return avail
    return selected


def _build_top_summary(
    results: Dict[str, Any],
    *,
    top_input_mode: str,
    trial_idx: int,
    groups: Optional[Sequence[str]],
    input_bin_ms: Optional[float],
    input_smooth_ms: Optional[float],
    input_source: str,
    warnings: list[str],
) -> Optional[Dict[str, Any]]:
    mode = str(top_input_mode or "average").lower()
    if mode not in ("average", "single_trial"):
        warnings.append(f"top_input_mode={top_input_mode!r} is invalid; using 'average'.")
        mode = "average"

    group_filter: Optional[Sequence[str]] = groups
    if groups is not None and len(groups) == 0:
        group_filter = None

    if mode == "single_trial":
        payload = analysis.select_inputs_payload(results, trial_idx=trial_idx)
        if payload is None:
            warnings.append("Top input panel: no saved input payload for selected trial.")
            return None
        single_like = {
            "sim_cfg": results.get("sim_cfg", {}) or {},
            "inputs": payload,
        }
        return analysis.summarize_inputs_from_results(
            single_like,
            groups=group_filter,
            bin_ms=input_bin_ms,
            smooth_ms=input_smooth_ms,
            input_source="saved",
            std_mode="std",
        )

    return analysis.summarize_inputs_from_results(
        results,
        groups=group_filter,
        bin_ms=input_bin_ms,
        smooth_ms=input_smooth_ms,
        input_source=input_source,
        std_mode="std",
    )


def _resolve_vm_trace(
    results: Dict[str, Any],
    *,
    trial_idx: int,
    warnings: list[str],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    traces = results.get("traces", {}) or {}
    T_raw = traces.get("T")
    if T_raw is None:
        warnings.append("Vm panel: traces.T is missing.")
        return None, None
    T = np.asarray(T_raw, dtype=float)
    V_raw = traces.get("V")
    if V_raw is None:
        warnings.append("Vm panel: traces.V is missing.")
        return None, None

    if isinstance(V_raw, list):
        if not V_raw:
            warnings.append("Vm panel: traces.V list is empty.")
            return None, None
        idx = int(max(0, min(trial_idx, len(V_raw) - 1)))
        if idx != trial_idx:
            warnings.append(f"Vm panel: trial_idx={trial_idx} not saved; using trial {idx}.")
        return T, np.asarray(V_raw[idx], dtype=float)

    V = np.asarray(V_raw, dtype=float)
    if V.ndim == 1:
        return T, V
    if V.ndim >= 2:
        idx = int(max(0, min(trial_idx, V.shape[0] - 1)))
        if idx != trial_idx:
            warnings.append(f"Vm panel: trial_idx={trial_idx} not saved; using trial {idx}.")
        return T, np.asarray(V[idx], dtype=float)

    warnings.append("Vm panel: traces.V has unsupported shape.")
    return None, None


def _resolve_output_curve(
    results: Dict[str, Any],
    *,
    source: str,
    recompute_bin_ms: Optional[float],
    recompute_smooth_ms: Optional[float],
    recompute_smooth_mode: str,
    warnings: list[str],
) -> Optional[Dict[str, Any]]:
    src = str(source or "meta").lower()
    meta_curve = (results.get("meta") or {}).get("avg_rate_curve")

    if src == "meta":
        if meta_curve is not None:
            return dict(meta_curve)
        warnings.append("Output curve source 'meta' missing; recomputing from spikes.")
        src = "recompute"

    if src == "auto":
        if meta_curve is not None:
            return dict(meta_curve)
        src = "recompute"

    if src == "recompute":
        curve = analysis.compute_output_curve_from_results(
            results,
            bin_ms=recompute_bin_ms,
            smooth_ms=recompute_smooth_ms,
            smooth_mode=recompute_smooth_mode,
        )
        if curve is None:
            warnings.append("Output panel: failed to compute output curve from spikes.")
        return curve

    warnings.append(f"Output curve source {source!r} is invalid.")
    return None


def _compute_output_band(
    results: Dict[str, Any],
    curve: Dict[str, Any],
    *,
    band_mode: str,
    output_mode: str,
    output_norm_kind: str,
    output_baseline_ms: float,
    output_baseline_mode: str,
    output_baseline_center_ms: Optional[float],
    output_norm_window: str,
    warnings: list[str],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    mode = str(band_mode or "").lower()
    if mode not in ("std", "sem"):
        return None, None

    x_target = np.asarray(curve.get("t_ms", []) or [], dtype=float)
    if x_target.size == 0:
        warnings.append("Output band: curve time axis is empty.")
        return None, None

    sim_cfg = results.get("sim_cfg", {}) or {}
    tstart = float(sim_cfg.get("tstart", 0.0))
    tstop = sim_cfg.get("tstop")
    if tstop is None:
        warnings.append("Output band: sim_cfg.tstop missing.")
        return None, None
    tstop = float(tstop)

    bin_ms = float(curve.get("bin_ms") or sim_cfg.get("bins", 25.0))
    if bin_ms <= 0:
        warnings.append("Output band: invalid bin_ms.")
        return None, None
    bins = np.arange(tstart, tstop + bin_ms, bin_ms, dtype=float)
    if bins.size < 2:
        warnings.append("Output band: insufficient bins.")
        return None, None

    centers = bins[:-1] + 0.5 * bin_ms
    bw_s = bin_ms / 1000.0
    smooth_ms = curve.get("smooth_ms")
    smooth_mode = str(curve.get("smooth_mode", "center") or "center")
    trials = _coerce_trials(results.get("spikes"))
    if not trials:
        warnings.append("Output band: no spikes available.")
        return None, None

    stack = []
    for tr in trials:
        counts, _ = np.histogram(np.asarray(tr, dtype=float), bins=bins)
        y = counts / bw_s
        x_s, y_s = _smooth_rate(
            centers.copy(),
            np.asarray(y, dtype=float),
            bin_ms=bin_ms,
            smooth_ms=smooth_ms,
            smooth_mode=smooth_mode,
        )
        trial_curve = {
            "t_ms": x_s.tolist(),
            "rate_hz": y_s.tolist(),
            "bin_ms": bin_ms,
            "smooth_ms": smooth_ms if smooth_ms is not None else 0.0,
            "smooth_mode": smooth_mode,
        }
        if str(output_mode or "raw").lower() != "raw":
            trial_curve = analysis.normalize_output_curve(
                trial_curve,
                sim_cfg,
                mode="normalized",
                norm_mode=output_norm_kind,
                baseline_ms=output_baseline_ms,
                baseline_mode=output_baseline_mode,
                baseline_center_ms=output_baseline_center_ms,
                norm_window=output_norm_window,
            )

        tx = np.asarray(trial_curve.get("t_ms", []) or [], dtype=float)
        ty = np.asarray(trial_curve.get("rate_hz", []) or [], dtype=float)
        if tx.size == 0 or ty.size == 0:
            continue
        if tx.size != ty.size:
            continue
        y_interp = np.interp(x_target, tx, ty, left=np.nan, right=np.nan)
        stack.append(y_interp)

    if len(stack) < 2:
        warnings.append("Output band: fewer than 2 trials after processing.")
        return None, None

    arr = np.vstack(stack)
    spread = np.nanstd(arr, axis=0, ddof=1)
    if mode == "sem":
        spread = spread / np.sqrt(arr.shape[0])
    mean = np.nanmean(arr, axis=0)
    return mean - spread, mean + spread


def _resolve_export_paths(
    export_path: Optional[Union[str, Path]],
    *,
    run_dir: Optional[Union[str, Path]],
    export_formats: Optional[Sequence[str]],
) -> list[Path]:
    if export_path in (None, ""):
        return []

    out = Path(str(export_path)).expanduser()
    if not out.is_absolute():
        if run_dir is not None:
            run_dir_path = Path(run_dir).expanduser().resolve()
            out = analysis.plot_dir_for_run(run_dir_path) / out
        else:
            out = Path.cwd() / out

    if out.suffix:
        return [out]

    fmts = [str(f).strip(".").lower() for f in (export_formats or ["svg"])]
    fmts = [f for f in fmts if f]
    if not fmts:
        fmts = ["svg"]
    return [out.with_suffix(f".{fmt}") for fmt in fmts]


def resolve_export_paths(
    export_path: Optional[Union[str, Path]],
    *,
    run_dir: Optional[Union[str, Path]],
    export_formats: Optional[Sequence[str]],
) -> list[Path]:
    """
    Public wrapper for resolving requested paper-panel export paths.
    """
    return _resolve_export_paths(
        export_path,
        run_dir=run_dir,
        export_formats=export_formats,
    )


def choose_preview_export_path(paths: Sequence[Union[str, Path]]) -> Optional[Path]:
    """
    Choose a preferred preview file from exported/saved paths.
    """
    path_list = [Path(p) for p in (paths or [])]
    if not path_list:
        return None
    for suffix in (".png", ".svg", ".jpg", ".jpeg", ".webp", ".gif"):
        for p in path_list:
            if p.suffix.lower() == suffix:
                return p
    return path_list[0]


def load_single_plot_preset(
    *,
    repo_root: Optional[Union[str, Path]] = None,
    preset_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Load single-plot preset defaults from JSON.

    Returns:
      {
        "config": dict,
        "preset_path": str,
        "repo_root": str,
        "warnings": list[str],
      }
    """
    warnings: list[str] = []

    if repo_root is None:
        try:
            root = analysis.find_scp_root(Path.cwd())
        except Exception:
            root = Path.cwd()
    else:
        root = Path(repo_root).expanduser().resolve()

    path = (
        root / "modules_local" / "analysis" / "analysis_presets" / "single_plot.json"
        if preset_path in (None, "", False)
        else Path(str(preset_path)).expanduser()
    )
    if not path.is_absolute():
        path = (root / path).resolve()

    cfg: Dict[str, Any] = {}
    try:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            defaults = payload.get("defaults", payload)
            if isinstance(defaults, dict):
                cfg = dict(defaults)
            else:
                warnings.append(f"Single-plot preset defaults missing/invalid in {path}")
        else:
            warnings.append(f"Single-plot preset JSON root must be object (got {type(payload).__name__})")
    except Exception as exc:
        warnings.append(f"Single-plot preset load failed ({path}): {exc}")

    return {
        "config": cfg,
        "preset_path": str(path),
        "repo_root": str(root),
        "warnings": warnings,
    }


def plot_paper_panel_from_results(
    results: Dict[str, Any],
    *,
    run_dir: Optional[Union[str, Path]] = None,
    trial_idx: int = 0,
    top_input_groups: Optional[Sequence[str]] = None,
    raster_input_groups: Optional[Sequence[str]] = None,
    top_input_mode: str = "average",  # "average" | "single_trial"
    top_layout: str = "overlay",  # "overlay" | "stacked"
    top_fill_under: bool = False,
    top_fill_alpha: float = 0.28,
    input_bin_ms: Optional[float] = None,
    input_smooth_ms: Optional[float] = 25.0,
    input_source: str = "auto",  # "saved" | "stats" | "auto"
    input_raster_style: str = "dot",  # "dot" | "line"
    input_raster_dot_size: float = 5.0,
    include_input_raster: bool = True,
    include_output_raster: bool = False,
    output_raster_style: str = "dot",  # "dot" | "line"
    output_raster_dot_size: float = 5.0,
    output_curve_source: str = "meta",  # "meta" | "recompute" | "auto"
    output_recompute_bin_ms: Optional[float] = None,
    output_recompute_smooth_ms: Optional[float] = None,
    output_recompute_smooth_mode: str = "center",
    output_mode: str = "raw",  # "raw" | "normalized"
    output_norm_kind: str = "peak",  # "peak" | "avg"
    output_baseline_ms: float = 100.0,
    output_baseline_mode: str = "window",
    output_baseline_center_ms: Optional[float] = None,
    output_norm_window: str = "stim",
    output_fill_under: bool = False,
    output_fill_alpha: float = 0.28,
    output_band_mode: Optional[str] = None,  # None | "std" | "sem"
    output_band_alpha: float = 0.20,
    plot_window: Optional[tuple[Optional[float], Optional[float]]] = None,
    auto_plot_window_from_stim: bool = False,
    plot_window_adjustment_ms: float = 100.0,
    show_stim_lines: bool = True,
    show_y_axes: bool = True,
    show_y_axis_titles: bool = True,
    show_input_legend: bool = True,
    figsize: tuple[float, float] = (8.0, 10.0),
    panel_height_ratios: Optional[Union[Sequence[float], Dict[str, float]]] = None,
    show_panel_titles: bool = False,
    export_path: Optional[Union[str, Path]] = None,
    export_formats: Optional[Sequence[str]] = ("svg",),
    export_overwrite: bool = False,
    dpi: int = 300,
    vm_color: Optional[str] = None,
    output_color: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a paper-style stacked panel figure from one run's saved results.

    Returns a payload dict with:
      - fig, axes
      - warnings
      - run_dir
      - exported_paths
      - used_groups_top / used_groups_raster
    """
    warnings: list[str] = []
    sim_cfg = results.get("sim_cfg", {}) or {}
    group_colors = analysis.group_colors_from_results(results)
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])

    try:
        input_raster_dot_size_f = float(input_raster_dot_size)
        if input_raster_dot_size_f <= 0:
            raise ValueError
    except Exception:
        input_raster_dot_size_f = 5.0
        warnings.append(
            f"input_raster_dot_size={input_raster_dot_size!r} invalid; using 5.0."
        )
    try:
        output_raster_dot_size_f = float(output_raster_dot_size)
        if output_raster_dot_size_f <= 0:
            raise ValueError
    except Exception:
        output_raster_dot_size_f = 5.0
        warnings.append(
            f"output_raster_dot_size={output_raster_dot_size!r} invalid; using 5.0."
        )

    # Top input summary
    try:
        summary_top = _build_top_summary(
            results,
            top_input_mode=top_input_mode,
            trial_idx=trial_idx,
            groups=top_input_groups,
            input_bin_ms=input_bin_ms,
            input_smooth_ms=input_smooth_ms,
            input_source=input_source,
            warnings=warnings,
        )
    except Exception as exc:
        summary_top = None
        warnings.append(f"Top input panel: failed to build summary ({exc}).")

    top_avail = list((summary_top or {}).get("groups", {}).keys())
    used_groups_top = _resolve_groups(top_input_groups, top_avail, panel_name="Top input panel", warnings=warnings)
    top_layout_mode = str(top_layout or "overlay").lower()
    if top_layout_mode not in ("overlay", "stacked"):
        warnings.append(f"top_layout={top_layout!r} invalid; using 'overlay'.")
        top_layout_mode = "overlay"
    n_top_axes = 1 if top_layout_mode == "overlay" else max(1, len(used_groups_top))

    # Raster payload for input spikes
    payload_raster = None
    used_groups_raster: list[str] = []
    if include_input_raster:
        payload_raster = analysis.select_inputs_payload(results, trial_idx=trial_idx)
        if payload_raster is None:
            warnings.append("Input raster panel: no saved input payload.")
            raster_avail = []
        else:
            raster_avail = list(payload_raster.keys())
        used_groups_raster = _resolve_groups(
            raster_input_groups,
            raster_avail,
            panel_name="Input raster panel",
            warnings=warnings,
        )

    # Panel layout
    n_panels = n_top_axes + 2 + (1 if include_input_raster else 0) + (1 if include_output_raster else 0)
    if isinstance(panel_height_ratios, dict):
        top_h = float(panel_height_ratios.get("top", 1.0))
        in_r_h = float(panel_height_ratios.get("input_raster", 1.2))
        vm_h = float(panel_height_ratios.get("vm", 1.2))
        out_h = float(panel_height_ratios.get("output_rate", 1.3))
        out_r_h = float(panel_height_ratios.get("output_raster", 1.0))
        ratios = [top_h] * n_top_axes
        if include_input_raster:
            ratios.append(in_r_h)
        ratios.extend([vm_h, out_h])
        if include_output_raster:
            ratios.append(out_r_h)
    elif isinstance(panel_height_ratios, (list, tuple)) and len(panel_height_ratios) == n_panels:
        ratios = [float(v) for v in panel_height_ratios]
    else:
        ratios = [1.0] * n_top_axes
        if include_input_raster:
            ratios.append(1.2)
        ratios.extend([1.2, 1.3])
        if include_output_raster:
            ratios.append(1.0)

    fig, axes_arr = plt.subplots(
        n_panels,
        1,
        sharex=True,
        figsize=figsize,
        gridspec_kw={"height_ratios": ratios},
    )
    axes = np.atleast_1d(axes_arr).tolist()

    # Base colors
    cell_color = sim_cfg.get("color", None)
    vm_col = vm_color or cell_color or "k"
    out_col = output_color or cell_color or "k"

    # Top panel(s)
    panel_idx = 0
    if summary_top is None or not used_groups_top:
        ax_top = axes[panel_idx]
        ax_top.text(0.01, 0.5, "No input summary available", transform=ax_top.transAxes, va="center")
        panel_idx += n_top_axes
    else:
        x_top = np.asarray(summary_top.get("t_ms") or [], dtype=float)
        if x_top.size == 0:
            axes[panel_idx].text(0.01, 0.5, "Input summary time axis empty", transform=axes[panel_idx].transAxes, va="center")
            panel_idx += n_top_axes
        elif top_layout_mode == "overlay":
            ax = axes[panel_idx]
            panel_idx += 1
            for i, g in enumerate(used_groups_top):
                gdata = (summary_top.get("groups") or {}).get(g, {}) or {}
                y = np.asarray(gdata.get("mean_rate", []), dtype=float)
                if y.size != x_top.size:
                    continue
                col = group_colors.get(g, palette[i % len(palette)])
                ax.plot(x_top, y, color=col, lw=1.8, label=g)
                if top_fill_under:
                    ax.fill_between(x_top, 0.0, y, color=col, alpha=float(top_fill_alpha), linewidth=0)
            if show_input_legend and len(used_groups_top) > 1:
                leg = ax.legend(frameon=False, loc="upper right")
                # Keep legend from forcing subplot resizing in tight/compact figures.
                try:
                    leg.set_in_layout(False)
                except Exception:
                    pass
            if show_y_axes and show_y_axis_titles:
                ax.set_ylabel("Input\nHz/syn")
            if show_panel_titles:
                ax.set_title("Input rate")
        else:
            for i, g in enumerate(used_groups_top):
                ax = axes[panel_idx]
                panel_idx += 1
                gdata = (summary_top.get("groups") or {}).get(g, {}) or {}
                y = np.asarray(gdata.get("mean_rate", []), dtype=float)
                col = group_colors.get(g, palette[i % len(palette)])
                if y.size == x_top.size:
                    ax.plot(x_top, y, color=col, lw=1.8)
                    if top_fill_under:
                        ax.fill_between(x_top, 0.0, y, color=col, alpha=float(top_fill_alpha), linewidth=0)
                txt = blended_transform_factory(ax.transAxes, ax.transAxes)
                ax.text(-0.015, 0.5, g, color=col, ha="right", va="center", transform=txt, clip_on=False)
                if show_y_axes and show_y_axis_titles:
                    ax.set_ylabel("Hz/syn")
                if show_panel_titles and i == 0:
                    ax.set_title("Input rate")

    # Optional input raster panel
    if include_input_raster:
        ax_in_raster = axes[panel_idx]
        panel_idx += 1
        if payload_raster is None or not used_groups_raster:
            ax_in_raster.text(0.01, 0.5, "No input raster payload", transform=ax_in_raster.transAxes, va="center")
        else:
            style = str(input_raster_style or "dot").lower()
            if style not in ("dot", "line"):
                warnings.append(f"input_raster_style={input_raster_style!r} invalid; using 'dot'.")
                style = "dot"
            y_cursor = 0
            group_mids: dict[str, float] = {}
            for i, g in enumerate(used_groups_raster):
                gdata = payload_raster.get(g, {}) or {}
                trains = [np.asarray(tr, dtype=float) for tr in (gdata.get("spike_trains") or [])]
                if not trains:
                    continue
                y_start = y_cursor + 1
                col = group_colors.get(g, palette[i % len(palette)])
                for tr in trains:
                    y = y_cursor + 1
                    if style == "line":
                        ax_in_raster.vlines(tr, y - 0.40, y + 0.40, color=col, lw=0.7)
                    else:
                        ax_in_raster.scatter(
                            tr,
                            np.full_like(tr, y),
                            color=col,
                            s=input_raster_dot_size_f,
                            marker=".",
                        )
                    y_cursor += 1
                group_mids[g] = 0.5 * (y_start + y_cursor)

            ax_in_raster.set_yticks([])
            if show_y_axes and show_y_axis_titles:
                ax_in_raster.set_ylabel("Input\ntrains")
            if show_panel_titles:
                ax_in_raster.set_title("Input raster")

            txt_trans = blended_transform_factory(ax_in_raster.transAxes, ax_in_raster.transData)
            for i, g in enumerate(used_groups_raster):
                if g not in group_mids:
                    continue
                col = group_colors.get(g, palette[i % len(palette)])
                ax_in_raster.text(
                    -0.015,
                    group_mids[g],
                    g,
                    color=col,
                    ha="right",
                    va="center",
                    transform=txt_trans,
                    clip_on=False,
                )

    # Vm panel
    ax_vm = axes[panel_idx]
    panel_idx += 1
    T_vm, V_vm = _resolve_vm_trace(results, trial_idx=trial_idx, warnings=warnings)
    if T_vm is None or V_vm is None:
        ax_vm.text(0.01, 0.5, "No Vm trace available", transform=ax_vm.transAxes, va="center")
    else:
        ax_vm.plot(T_vm, V_vm, color=vm_col, lw=1.2)
    if show_y_axes and show_y_axis_titles:
        ax_vm.set_ylabel("Vm (mV)")
    if show_panel_titles:
        ax_vm.set_title("Membrane voltage")

    # Output rate panel
    ax_out = axes[panel_idx]
    panel_idx += 1
    curve = _resolve_output_curve(
        results,
        source=output_curve_source,
        recompute_bin_ms=output_recompute_bin_ms,
        recompute_smooth_ms=output_recompute_smooth_ms,
        recompute_smooth_mode=output_recompute_smooth_mode,
        warnings=warnings,
    )
    if curve is None:
        ax_out.text(0.01, 0.5, "No output curve available", transform=ax_out.transAxes, va="center")
    else:
        if str(output_mode or "raw").lower() != "raw":
            curve = analysis.normalize_output_curve(
                curve,
                sim_cfg,
                mode="normalized",
                norm_mode=output_norm_kind,
                baseline_ms=output_baseline_ms,
                baseline_mode=output_baseline_mode,
                baseline_center_ms=output_baseline_center_ms,
                norm_window=output_norm_window,
            )
        x_out = np.asarray(curve.get("t_ms", []) or [], dtype=float)
        y_out = np.asarray(curve.get("rate_hz", []) or [], dtype=float)
        if x_out.size and y_out.size and x_out.size == y_out.size:
            ax_out.plot(x_out, y_out, color=out_col, lw=1.8)
            if output_fill_under:
                ax_out.fill_between(x_out, 0.0, y_out, color=out_col, alpha=float(output_fill_alpha), linewidth=0)
            if output_band_mode in ("std", "sem"):
                lo, hi = _compute_output_band(
                    results,
                    curve,
                    band_mode=str(output_band_mode),
                    output_mode=output_mode,
                    output_norm_kind=output_norm_kind,
                    output_baseline_ms=output_baseline_ms,
                    output_baseline_mode=output_baseline_mode,
                    output_baseline_center_ms=output_baseline_center_ms,
                    output_norm_window=output_norm_window,
                    warnings=warnings,
                )
                if lo is not None and hi is not None:
                    ax_out.fill_between(x_out, lo, hi, color=out_col, alpha=float(output_band_alpha), linewidth=0)
        else:
            ax_out.text(0.01, 0.5, "Output curve is empty", transform=ax_out.transAxes, va="center")

    if show_y_axes and show_y_axis_titles:
        ylab = "Rate (Hz)" if str(output_mode or "raw").lower() == "raw" else "Rate (norm)"
        ax_out.set_ylabel(ylab)
    if show_panel_titles:
        ax_out.set_title("Output average rate")

    # Optional output raster panel
    if include_output_raster:
        ax_out_r = axes[panel_idx]
        style = str(output_raster_style or "dot").lower()
        if style not in ("dot", "line"):
            warnings.append(f"output_raster_style={output_raster_style!r} invalid; using 'dot'.")
            style = "dot"
        trials = _coerce_trials(results.get("spikes"))
        if not trials:
            ax_out_r.text(0.01, 0.5, "No output spikes available", transform=ax_out_r.transAxes, va="center")
        else:
            for i, tr in enumerate(trials):
                y = i + 1
                if style == "line":
                    ax_out_r.vlines(tr, y - 0.40, y + 0.40, color=out_col, lw=0.7)
                else:
                    ax_out_r.scatter(
                        tr,
                        np.full_like(tr, y),
                        color=out_col,
                        s=output_raster_dot_size_f,
                        marker=".",
                    )
            ax_out_r.set_ylim(0.5, len(trials) + 0.5)
        if show_y_axes and show_y_axis_titles:
            ax_out_r.set_ylabel("Trial")
        if show_panel_titles:
            ax_out_r.set_title("Output raster")

    # Global axis styling
    for ax in axes:
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if not show_y_axes:
            ax.spines["left"].set_visible(False)
            ax.set_yticks([])
            ax.set_ylabel("")

    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("Time (ms)")

    effective_plot_window = _resolve_effective_plot_window(
        sim_cfg,
        plot_window=plot_window,
        auto_plot_window_from_stim=auto_plot_window_from_stim,
        plot_window_adjustment_ms=plot_window_adjustment_ms,
        warnings=warnings,
    )

    if effective_plot_window is not None:
        try:
            x0, x1 = effective_plot_window
            for ax in axes:
                ax.set_xlim(x0, x1)
        except Exception:
            warnings.append(f"plot_window={effective_plot_window!r} is invalid; expected (xmin, xmax).")

    if show_stim_lines:
        stim_start, stim_stop = _resolve_stim_window(sim_cfg)
        for vline in (stim_start, stim_stop):
            if vline is None:
                continue
            for ax in axes:
                ax.axvline(vline, color="k", lw=0.9, alpha=0.85)

    fig.tight_layout()

    # Export
    requested_export_paths = _resolve_export_paths(
        export_path,
        run_dir=run_dir,
        export_formats=export_formats,
    )
    exported_paths: list[Path] = []
    for out_path in requested_export_paths:
        if out_path.exists() and not bool(export_overwrite):
            warnings.append(f"Export skipped (exists, overwrite disabled): {out_path}")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
        exported_paths.append(out_path)

    return {
        "fig": fig,
        "axes": axes,
        "warnings": warnings,
        "run_dir": Path(run_dir).resolve() if run_dir is not None else None,
        "requested_export_paths": [str(p) for p in requested_export_paths],
        "exported_paths": [str(p) for p in exported_paths],
        "used_groups_top": used_groups_top,
        "used_groups_raster": used_groups_raster,
    }


def resolve_single_run_for_paper_panel(selection: Dict[str, Any]) -> tuple[Path, list[str]]:
    """
    Resolve one run path from selection for paper panel usage.

    Behavior:
      - If compare_list has entries, choose the most recent resolvable run.
      - Else use selection['run_single'] token/path.
    """
    warnings: list[str] = []
    base = selection.get("base")
    if base is None:
        raise ValueError("selection is missing 'base'")
    base_path = Path(base)

    compare_items = list(selection.get("compare_list") or [])
    candidates: list[Path] = []
    for item in compare_items:
        try:
            p = analysis.resolve_run(base_path, item)
            candidates.append(p)
        except Exception:
            continue

    if candidates:
        if len(candidates) > 1:
            warnings.append(
                f"Multiple selected runs found ({len(candidates)}); using most recent by mtime."
            )
        chosen = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
        return chosen, warnings

    run_single = selection.get("run_single", "latest")
    return analysis.resolve_run(base_path, run_single), warnings


def run_single_plot_from_selection(
    selection: Dict[str, Any],
    *,
    repo_root: Optional[Union[str, Path]] = None,
    preset_path: Optional[Union[str, Path]] = None,
    prefer_saved_export: bool = True,
) -> Dict[str, Any]:
    """
    Resolve one run from analysis selection and produce a single paper panel.

    When prefer_saved_export=True and export_overwrite=False, if all requested export
    files already exist, figure regeneration is skipped and saved files are reused.
    """
    warnings: list[str] = []
    run_dir, run_warnings = resolve_single_run_for_paper_panel(selection)
    warnings.extend(run_warnings)

    preset_info = load_single_plot_preset(
        repo_root=repo_root,
        preset_path=preset_path,
    )
    warnings.extend(list(preset_info.get("warnings") or []))
    panel_cfg = dict(preset_info.get("config") or {})

    requested = resolve_export_paths(
        panel_cfg.get("export_path"),
        run_dir=run_dir,
        export_formats=panel_cfg.get("export_formats", ("svg",)),
    )
    requested_paths = [Path(p) for p in requested]
    existing_paths = [p for p in requested_paths if p.exists()]
    export_overwrite = bool(panel_cfg.get("export_overwrite", False))
    use_saved = (
        bool(prefer_saved_export)
        and (not export_overwrite)
        and bool(requested_paths)
        and (len(existing_paths) == len(requested_paths))
    )

    if use_saved:
        preview = choose_preview_export_path(existing_paths)
        return {
            "fig": None,
            "axes": [],
            "warnings": warnings,
            "run_dir": Path(run_dir).resolve(),
            "preset_path": str(preset_info.get("preset_path") or ""),
            "panel_cfg": panel_cfg,
            "used_existing_exports": True,
            "requested_export_paths": [str(p) for p in requested_paths],
            "existing_export_paths": [str(p) for p in existing_paths],
            "exported_paths": [],
            "preview_path": (str(preview) if preview is not None else None),
        }

    from modules_local import run_sim  # local import to keep module dependencies light

    results = run_sim.load_results(run_dir)
    panel_result = plot_paper_panel_from_results(
        results,
        run_dir=run_dir,
        **panel_cfg,
    )

    exported_paths = [Path(p) for p in (panel_result.get("exported_paths") or [])]
    if requested_paths:
        existing_after = [p for p in requested_paths if p.exists()]
    else:
        existing_after = [Path(p) for p in (panel_result.get("requested_export_paths") or []) if Path(p).exists()]

    preview = choose_preview_export_path(exported_paths or existing_after)
    merged_warnings = warnings + list(panel_result.get("warnings") or [])

    out = dict(panel_result)
    out["warnings"] = merged_warnings
    out["preset_path"] = str(preset_info.get("preset_path") or "")
    out["panel_cfg"] = panel_cfg
    out["used_existing_exports"] = False
    out["existing_export_paths"] = [str(p) for p in existing_after]
    out["preview_path"] = (str(preview) if preview is not None else None)
    return out


def display_single_plot_result(
    result: Dict[str, Any],
    *,
    show_warnings: bool = True,
    show_paths: bool = True,
) -> Optional[Union[Path, Any]]:
    """
    Notebook-friendly display helper for run_single_plot_from_selection results.

    Returns the displayed preview path, displayed fig, or None.
    """
    if show_warnings:
        for w in (result.get("warnings") or []):
            print(f"Warning: {w}")

    used_existing = bool(result.get("used_existing_exports", False))
    exported = [Path(p) for p in (result.get("exported_paths") or [])]
    existing = [Path(p) for p in (result.get("existing_export_paths") or [])]

    if show_paths:
        if used_existing:
            if existing:
                print("Using saved single plot (no regeneration):")
                for p in existing:
                    print("  " + str(p))
        elif exported:
            print("Saved:")
            for p in exported:
                print("  " + str(p))

    preview_raw = result.get("preview_path", None)
    preview = Path(preview_raw) if preview_raw not in (None, "") else choose_preview_export_path(exported or existing)
    if preview is None:
        return None

    try:
        from IPython.display import Image, SVG, display  # type: ignore
    except Exception:
        return preview

    suffix = preview.suffix.lower()
    if suffix == ".svg":
        display(SVG(filename=str(preview)))
        return preview
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        display(Image(filename=str(preview)))
        return preview

    fig = result.get("fig", None)
    if fig is not None:
        display(fig)
        return fig
    return preview
