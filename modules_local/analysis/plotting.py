# ────────────────────────────────────────────────────────────────────────────
#  Moving-average helper
# ────────────────────────────────────────────────────────────────────────────
import pickle
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from matplotlib.lines import Line2D
from itertools import cycle, chain
from collections import defaultdict

from neuron import h

# ---------------------------------------------------------------------------
def moving_average(arr, 
                #    *,
                   win_size: float,
                   bin_width: float,
                   mode: str = 'center'):
    """Centred or causal boxcar moving–average."""
    if win_size is None or win_size <= bin_width or arr.size < win_size/bin_width:
        return np.asarray(arr, dtype=float)

    k = int(round(win_size / bin_width))
    if k % 2 == 0 and mode == 'center':
        k += 1
    k = max(k, 1)

    kernel = np.ones(k) / k
    arr    = np.asarray(arr, dtype=float)

    if mode == 'center':
        y_full = np.convolve(arr, kernel, mode='valid')
        return y_full

    elif mode == 'causal':
        pad = (k - 1, 0)
        arr_pad = np.convolve(np.pad(arr, pad), kernel, mode='valid')
        return arr_pad

    else:
        raise ValueError("mode must be 'center' or 'causal'")
    
def _align_centers(centers, y_smooth, mode='center'):
    """Trim/shift centers to match y_smooth length based on smoothing mode."""
    if len(y_smooth) == len(centers):
        return centers
    if mode == 'center':  # symmetric 'valid' convolution
        drop = (len(centers) - len(y_smooth)) // 2
        return centers[drop: drop + len(y_smooth)]
    elif mode == 'causal':  # left-aligned boxcar
        return centers[:len(y_smooth)]
    else:
        return centers[:len(y_smooth)]

def _apply_legend(ax, legend_loc=None):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if legend_loc is None or str(legend_loc).strip() == "":
        ax.legend(handles=handles, labels=labels)
        return
    loc_val = str(legend_loc).strip()
    if loc_val.lower() in ("none", "off", "false"):
        return
    ax.legend(handles=handles, labels=labels, loc=loc_val)


def _legend_safe_label(label):
    if label is None:
        return None
    txt = str(label)
    if not txt:
        return None
    # Matplotlib ignores labels that start with "_"
    if txt.startswith("_"):
        return f" {txt}"
    return txt


def resolve_export_paths(export_path, export_formats=None):
    """
    Resolve a base export path plus formats into concrete figure file paths.

    Examples
    --------
    - export_path='plots/vm_traces', export_formats=('svg','png')
      -> ['plots/vm_traces.svg', 'plots/vm_traces.png']
    - export_path='plots/vm_traces.png'
      -> ['plots/vm_traces.png']
    """
    if export_path in (None, ""):
        return []

    out = Path(str(export_path)).expanduser()
    if out.suffix:
        return [out]

    fmts = [str(f).strip(".").lower() for f in (export_formats or ["svg"])]
    fmts = [f for f in fmts if f]
    if not fmts:
        fmts = ["svg"]
    return [out.with_suffix(f".{fmt}") for fmt in fmts]


def save_figure_exports(
    fig,
    *,
    export_path=None,
    export_formats=("svg",),
    export_overwrite=False,
    dpi=300,
    verbose=True,
):
    """
    Save a matplotlib figure to one or more paths/formats with overwrite guard.

    Returns
    -------
    dict with keys:
      - requested_export_paths: list[str]
      - exported_paths: list[str]
      - warnings: list[str]
    """
    if fig is None or not hasattr(fig, "savefig"):
        raise TypeError("save_figure_exports requires a matplotlib figure (obj with savefig).")

    requested = [p.resolve() for p in resolve_export_paths(export_path, export_formats)]
    exported = []
    warnings = []

    for out_path in requested:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not bool(export_overwrite):
            msg = f"Warning: Export skipped (exists, overwrite disabled): {out_path}"
            warnings.append(msg)
            if verbose:
                print(msg)
            continue
        fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
        exported.append(out_path)

    if verbose and exported:
        print("Saved:")
        for p in exported:
            print("  " + str(p))

    return {
        "requested_export_paths": [str(p) for p in requested],
        "exported_paths": [str(p) for p in exported],
        "warnings": warnings,
    }


def collect_trace_rows(fig):
    """
    Collect plotted line traces from a matplotlib figure.

    Returns a list of dict rows, one row per trace.
    """
    if fig is None or not hasattr(fig, "axes"):
        raise TypeError("collect_trace_rows requires a matplotlib figure.")

    rows = []
    for ax_idx, ax in enumerate(fig.axes):
        ax_title = str(ax.get_title() or "")
        x_label = str(ax.get_xlabel() or "")
        y_label = str(ax.get_ylabel() or "")

        for line_idx, line in enumerate(ax.get_lines()):
            label = str(line.get_label() or "")
            if label.startswith("_"):
                label = ""
            x = np.asarray(line.get_xdata(), dtype=float).tolist()
            y = np.asarray(line.get_ydata(), dtype=float).tolist()
            rows.append(
                {
                    "axis_index": ax_idx,
                    "trace_index": line_idx,
                    "label": label,
                    "axis_title": ax_title,
                    "x_label": x_label,
                    "y_label": y_label,
                    "n_points": len(y),
                    "x_json": json.dumps(x),
                    "y_json": json.dumps(y),
                }
            )
    return rows


def save_trace_rows_csv(
    fig,
    csv_path,
    *,
    overwrite=False,
    verbose=True,
):
    """
    Save plotted traces from a figure to CSV with one row per trace.
    """
    if fig is None or not hasattr(fig, "axes"):
        raise TypeError("save_trace_rows_csv requires a matplotlib figure.")

    out_path = Path(str(csv_path)).expanduser()
    if not out_path.suffix:
        out_path = out_path.with_suffix(".csv")
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not bool(overwrite):
        msg = f"Warning: CSV export skipped (exists, overwrite disabled): {out_path}"
        if verbose:
            print(msg)
        return {"path": str(out_path), "saved": False, "warnings": [msg], "rows": 0}

    rows = collect_trace_rows(fig)
    fieldnames = [
        "axis_index",
        "trace_index",
        "label",
        "axis_title",
        "x_label",
        "y_label",
        "n_points",
        "x_json",
        "y_json",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        print(f"Saved CSV: {out_path}")

    return {"path": str(out_path), "saved": True, "warnings": [], "rows": len(rows)}

# ────────────────────────────────────────────────────────────────────────────
#  Normalization helper
# ────────────────────────────────────────────────────────────────────────────
def normalize(data,
              norm_offset,
              ):
    
    norm_data = []
    norm_max = max(data) + norm_offset

    for val in data:
        val = val + norm_offset # Normalize base data to 0
        val = val / norm_max # Normalize peak data value to 1
        norm_data.append(val)

    return norm_data


def plot_output_curve(
    curve,
    *,
    label=None,
    color=None,
    plot_window=None,
    stim_start=None,
    stim_stop=None,
    title="Output rate curve",
    line_width=2.0,
    stim_linewidth=1.0,
):
    t_ms = np.asarray(curve.get("t_ms", []) or [], dtype=float)
    y = np.asarray(curve.get("rate_hz", []) or [], dtype=float)
    units = curve.get("units", "Hz")
    y_label = "Rate (Hz)" if units == "Hz" else "Rate (normalized)"
    label_plot = _legend_safe_label(label)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_ms, y, lw=float(line_width), color=color, label=label_plot)
    if label_plot:
        _apply_legend(ax)
    if plot_window is not None:
        ax.set_xlim(plot_window[0], plot_window[1])
    if stim_start is not None:
        ax.axvline(float(stim_start), color="k", linestyle="-", linewidth=float(stim_linewidth))
    if stim_stop is not None:
        ax.axvline(float(stim_stop), color="k", linestyle="-", linewidth=float(stim_linewidth))
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    return fig, ax


def plot_compare_output_curves(
    curve_a,
    curve_b,
    *,
    labels=("A", "B"),
    colors=None,
    plot_window=None,
    stim_start=None,
    stim_stop=None,
    title="Output curve compare",
    line_width=2.0,
    stim_linewidth=1.0,
):
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_a = colors[0 % len(colors)] if colors else None
    color_b = colors[1 % len(colors)] if colors else None
    same_color = (color_a is not None and color_b is not None and color_a == color_b)
    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, curve in enumerate((curve_a, curve_b)):
        if not curve:
            continue
        t_ms = np.asarray(curve.get("t_ms", []) or [], dtype=float)
        y = np.asarray(curve.get("rate_hz", []) or [], dtype=float)
        lab = labels[idx] if labels and idx < len(labels) else None
        lab = _legend_safe_label(lab)
        ls = "--" if same_color and idx == 1 else "-"
        ax.plot(t_ms, y, lw=float(line_width), color=colors[idx % len(colors)], linestyle=ls, label=lab)
    units = (curve_a or curve_b or {}).get("units", "Hz")
    y_label = "Rate (Hz)" if units == "Hz" else "Rate (normalized)"
    if labels:
        _apply_legend(ax)
    if plot_window is not None:
        ax.set_xlim(plot_window[0], plot_window[1])
    if stim_start is not None:
        ax.axvline(float(stim_start), color="k", linestyle="-", linewidth=float(stim_linewidth))
    if stim_stop is not None:
        ax.axvline(float(stim_stop), color="k", linestyle="-", linewidth=float(stim_linewidth))
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    return fig, ax

# ────────────────────────────────────────────────────────────────────────────
#  Input plotting (Step 5.4.2)
# ────────────────────────────────────────────────────────────────────────────
def plot_inputs_by_group(
        inputs_by_group,
        sim_cfg,
        *,
        groups=None,
        bin_ms=None,
        win_size=25.0,
        group_colors=None,
        raster_style='dot',
        max_trains_per_group=200,
        seed=0,
        plot_window=None,
        legend_loc=None,
        plot_raster=True,
        line_width=2.0,
        raster_linewidth=0.8,
        stim_linewidth=1.0):
    """
    Plot input rasters + average rate curves for inputs_by_group.
    group_colors: optional dict {group: color} for consistent coloring.

    Returns a dict of per-group stats.
    """
    if groups is None:
        groups = list(inputs_by_group.keys())
    if not groups:
        print("plot_inputs_by_group: no active input groups.")
        return {}

    tstart = float(sim_cfg.get("tstart", 0.0))
    tstop = float(sim_cfg.get("tstop", 0.0))
    bin_ms = float(bin_ms if bin_ms is not None else sim_cfg.get("bins", 5.0))

    stim_start = sim_cfg.get("stim_start_ms")
    stim_stop = sim_cfg.get("stim_stop_ms")
    stim_dur = sim_cfg.get("stim_duration_ms")
    if stim_start is not None and stim_stop is None and stim_dur is not None:
        stim_stop = float(stim_start) + float(stim_dur)

    rng = np.random.default_rng(seed) if seed is not None else None

    if plot_raster:
        fig, (ax_rate, ax_raster) = plt.subplots(
            2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
        )
    else:
        fig, ax_rate = plt.subplots(1, 1, figsize=(7, 3))
        ax_raster = None

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    stats = {}
    y0 = 0

    def _bin_trains(trains, t0, t1, bw):
        edges = np.arange(t0, t1 + bw, bw, dtype=float)
        if edges.size < 2:
            return np.array([], dtype=float), np.array([], dtype=float)
        counts = np.zeros(edges.size - 1, dtype=float)
        for tr in trains:
            if len(tr) == 0:
                continue
            c, _ = np.histogram(tr, bins=edges)
            counts += c
        n_syn = max(len(trains), 1)
        rate = counts / (n_syn * (bw / 1000.0))
        centers = edges[:-1] + bw * 0.5
        return centers, rate

    for i, gname in enumerate(groups):
        gi = inputs_by_group.get(gname)
        if gi is None:
            continue
        if hasattr(gi, "spike_trains"):
            trains = list(gi.spike_trains)
        elif isinstance(gi, dict):
            trains = list(gi.get("spike_trains", []))
        else:
            trains = list(getattr(gi, "spike_trains", []))
        n_trains = len(trains)
        if n_trains == 0:
            continue

        centers, rate = _bin_trains(trains, tstart, tstop, bin_ms)
        if win_size:
            rate_smooth = moving_average(rate, win_size=win_size, bin_width=bin_ms, mode='center')
            x_line = _align_centers(centers, rate_smooth, mode='center')
            y_line = rate_smooth
        else:
            x_line = centers
            y_line = rate

        if group_colors and gname in group_colors:
            col = group_colors[gname]
        else:
            col = colors[i % len(colors)]
        ax_rate.plot(x_line, y_line, color=col, lw=float(line_width), label=gname)

        plot_trains = trains
        if max_trains_per_group is not None and n_trains > max_trains_per_group:
            if rng is None:
                plot_trains = trains[:max_trains_per_group]
            else:
                idx = rng.choice(n_trains, size=max_trains_per_group, replace=False)
                plot_trains = [trains[j] for j in idx]

        if plot_raster and ax_raster is not None:
            for j, tr in enumerate(plot_trains):
                y = y0 + j + 1
                if raster_style == 'line':
                    ax_raster.vlines(tr, y - 0.4, y + 0.4, color=col, lw=float(raster_linewidth))
                else:
                    ax_raster.scatter(tr, np.full_like(tr, y), color=col, s=6, marker='.')

        total_spikes = int(np.sum([len(tr) for tr in trains]))
        duration_ms = max(tstop - tstart, 0.0)
        mean_rate = total_spikes / (max(n_trains, 1) * (duration_ms / 1000.0)) if duration_ms > 0 else 0.0

        stats[gname] = {
            "n_trains": n_trains,
            "total_spikes": total_spikes,
            "mean_rate_hz": mean_rate,
        }

        y0 += len(plot_trains)

    ax_rate.set_ylabel("Rate (Hz per synapse)")
    ax_rate.set_title("Input rate by group")
    if len(groups) > 1:
        _apply_legend(ax_rate, legend_loc)
    ax_rate.grid(True)

    if plot_raster and ax_raster is not None:
        ax_raster.set_ylabel("Input train")
        ax_raster.set_xlabel("Time (ms)")
        ax_raster.set_title("Input raster (subset)")
        ax_raster.grid(axis='x')

    if plot_window is not None:
        ax_rate.set_xlim(plot_window[0], plot_window[1])

    for vline in [stim_start, stim_stop]:
        if vline is not None:
            ax_rate.axvline(x=vline, color='k', linestyle='-', linewidth=float(stim_linewidth))
            if plot_raster and ax_raster is not None:
                ax_raster.axvline(x=vline, color='k', linestyle='-', linewidth=float(stim_linewidth))

    plt.tight_layout()
    return stats


def plot_compare_input_means(
        summary_a,
        summary_b,
        *,
        labels=("Run A", "Run B"),
        groups=None,
        layout="side-by-side",
        show_std=False,
        output_curves=None,
        plot_window=None,
        legend_loc=None,
        group_colors=None,
        figsize=None,
        line_width=2.0,
        shade_alpha=0.2,
        output_linewidth=1.5,
):
    """
    Plot averaged input rate curves for two runs.

    summary_* should be output from analysis.summarize_inputs_from_results.
    output_curves: optional tuple (curve_a, curve_b), each as dict with keys
    't_ms' and 'rate_hz'. If provided, plots on a twin y-axis.
    plot_window: optional tuple (tstart, tstop) to limit the x-axis.
    legend_loc: optional matplotlib legend location (use "none" to hide).
    group_colors: optional dict {group: color} for consistent coloring.
    """
    if summary_a is None or summary_b is None:
        raise ValueError("plot_compare_input_means: summaries must not be None")

    groups_a = set((summary_a.get("groups") or {}).keys())
    groups_b = set((summary_b.get("groups") or {}).keys())
    if groups is None:
        groups = sorted(groups_a.union(groups_b))
    else:
        groups = [g for g in groups if g in groups_a or g in groups_b]

    layout = (layout or "side-by-side").lower()
    if layout in ("stacked", "top-bottom", "vertical"):
        nrows, ncols = 2, 1
    elif layout in ("overlay", "same", "same-plot", "overlap"):
        nrows, ncols = 1, 1
    else:
        nrows, ncols = 1, 2

    if figsize is None:
        figsize = (10, 4) if ncols == 2 else (7, 6)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    if nrows * ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {}
    for i, g in enumerate(groups):
        if group_colors and g in group_colors:
            color_map[g] = group_colors[g]
        else:
            color_map[g] = colors[i % len(colors)]

    summaries = [summary_a, summary_b]
    if layout in ("overlay", "same", "same-plot", "overlap"):
        ax = axes[0]
        ax2 = None
        for idx, summary in enumerate(summaries):
            label = labels[idx] if idx < len(labels) else f"Run {idx+1}"
            linestyle = "-" if idx == 0 else "--"
            base_alpha = float(shade_alpha)
            alpha = base_alpha if idx == 0 else max(base_alpha * 0.72, 0.05)
            t_ms = summary.get("t_ms") or []
            for g in groups:
                gdata = (summary.get("groups") or {}).get(g)
                if not gdata:
                    continue
                mean_rate = np.asarray(gdata.get("mean_rate", []), dtype=float)
                std_rate = np.asarray(gdata.get("std_rate", []), dtype=float)
                if len(t_ms) != len(mean_rate):
                    continue
                ax.plot(
                    t_ms,
                    mean_rate,
                    color=color_map[g],
                    lw=float(line_width),
                    linestyle=linestyle,
                    label=f"{g} ({label})",
                )
                if show_std and std_rate.size:
                    ax.fill_between(
                        t_ms,
                        mean_rate - std_rate,
                        mean_rate + std_rate,
                        color=color_map[g],
                        alpha=alpha,
                    )

            if output_curves and idx < len(output_curves):
                curve = output_curves[idx]
                if curve:
                    out_t = curve.get("t_ms", [])
                    out_r = curve.get("rate_hz", [])
                    if out_t and out_r:
                        if ax2 is None:
                            ax2 = ax.twinx()
                            ax2.set_ylabel("Output rate (Hz)")
                        ax2.plot(
                            out_t,
                            out_r,
                            color="black",
                            lw=float(output_linewidth),
                            linestyle=linestyle,
                            label=f"output ({label})",
                        )

        ax.set_title("Inputs (mean) overlay")
        ax.set_ylabel("Input rate (Hz per synapse)")
        ax.grid(True)
        if groups:
            _apply_legend(ax, legend_loc)
        if plot_window is not None:
            ax.set_xlim(plot_window[0], plot_window[1])
    else:
        for idx, ax in enumerate(axes[:2]):
            summary = summaries[idx]
            label = labels[idx] if idx < len(labels) else f"Run {idx+1}"
            t_ms = summary.get("t_ms") or []
            for g in groups:
                gdata = (summary.get("groups") or {}).get(g)
                if not gdata:
                    continue
                mean_rate = np.asarray(gdata.get("mean_rate", []), dtype=float)
                std_rate = np.asarray(gdata.get("std_rate", []), dtype=float)
                if len(t_ms) != len(mean_rate):
                    continue
                ax.plot(t_ms, mean_rate, color=color_map[g], lw=float(line_width), label=g)
                if show_std and std_rate.size:
                    ax.fill_between(
                        t_ms,
                        mean_rate - std_rate,
                        mean_rate + std_rate,
                        color=color_map[g],
                        alpha=float(shade_alpha),
                    )

            ax.set_title(f"{label} inputs (mean)")
            ax.set_ylabel("Input rate (Hz per synapse)")
            ax.grid(True)
            if len(groups) > 1:
                _apply_legend(ax, legend_loc)
            if plot_window is not None:
                ax.set_xlim(plot_window[0], plot_window[1])

            # Optional output curve overlay
            if output_curves and idx < len(output_curves):
                curve = output_curves[idx]
                if curve:
                    out_t = curve.get("t_ms", [])
                    out_r = curve.get("rate_hz", [])
                    if out_t and out_r:
                        ax2 = ax.twinx()
                        ax2.plot(
                            out_t,
                            out_r,
                            color="black",
                            lw=float(output_linewidth),
                            linestyle="--",
                            label="output",
                        )
                        ax2.set_ylabel("Output rate (Hz)")

    # Shared x label
    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()
    return fig, axes


def plot_input_means(
        summary,
        *,
        label="Run",
        groups=None,
        show_std=False,
        output_curve=None,
        plot_window=None,
        legend_loc=None,
        group_colors=None,
        figsize=None,
        line_width=2.0,
        shade_alpha=0.2,
        output_linewidth=1.5,
):
    """
    Plot averaged input rate curves for a single run.

    summary should be output from analysis.summarize_inputs_from_results.
    output_curve: optional dict with keys 't_ms' and 'rate_hz' to overlay.
    plot_window: optional tuple (tstart, tstop) to limit the x-axis.
    legend_loc: optional matplotlib legend location (use "none" to hide).
    group_colors: optional dict {group: color} for consistent coloring.
    """
    if summary is None:
        raise ValueError("plot_input_means: summary must not be None")

    groups_avail = list((summary.get("groups") or {}).keys())
    if groups is None:
        groups = groups_avail
    else:
        groups = [g for g in groups if g in groups_avail]

    if figsize is None:
        figsize = (8, 4)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {}
    for i, g in enumerate(groups):
        if group_colors and g in group_colors:
            color_map[g] = group_colors[g]
        else:
            color_map[g] = colors[i % len(colors)]

    t_ms = summary.get("t_ms") or []
    for g in groups:
        gdata = (summary.get("groups") or {}).get(g)
        if not gdata:
            continue
        mean_rate = np.asarray(gdata.get("mean_rate", []), dtype=float)
        std_rate = np.asarray(gdata.get("std_rate", []), dtype=float)
        if len(t_ms) != len(mean_rate):
            continue
        ax.plot(t_ms, mean_rate, color=color_map[g], lw=float(line_width), label=g)
        if show_std and std_rate.size:
            ax.fill_between(
                t_ms,
                mean_rate - std_rate,
                mean_rate + std_rate,
                color=color_map[g],
                alpha=float(shade_alpha),
            )

    ax.set_title(f"{label} inputs (mean)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Input rate (Hz per synapse)")
    ax.grid(True)
    if len(groups) > 1:
        _apply_legend(ax, legend_loc)
    if plot_window is not None:
        ax.set_xlim(plot_window[0], plot_window[1])

    if output_curve:
        out_t = output_curve.get("t_ms", [])
        out_r = output_curve.get("rate_hz", [])
        if out_t and out_r:
            ax2 = ax.twinx()
            ax2.plot(out_t, out_r, color="black", lw=float(output_linewidth), linestyle="--", label="output")
            ax2.set_ylabel("Output rate (Hz)")

    plt.tight_layout()
    return fig, ax

# ────────────────────────────────────────────────────────────────────────────
#  Synapse-property plots (distance / weight / distance-density)
# ────────────────────────────────────────────────────────────────────────────
def plot_synapse_compare_hist(
        vals_a,
        vals_b,
        *,
        labels=("Run A", "Run B"),
        bin_width=0.1,
        xlabel="Value",
        title="Synapse distribution",
        density=True,
        figsize=(6, 4),
):
    """
    Overlay two synapse distributions as line histograms.
    vals_*: array-like of numeric values.
    """
    vals_a = np.asarray(vals_a or [], dtype=float)
    vals_b = np.asarray(vals_b or [], dtype=float)
    if vals_a.size == 0 and vals_b.size == 0:
        print(f"No synapse data for {title}.")
        return None

    all_vals = np.concatenate([v for v in (vals_a, vals_b) if v.size])
    if all_vals.size == 0:
        print(f"No synapse data for {title}.")
        return None

    lo, hi = float(all_vals.min()), float(all_vals.max())
    if lo == hi:
        lo -= 0.5 * bin_width
        hi += 0.5 * bin_width
    edges = np.arange(lo, hi + bin_width, bin_width, dtype=float)
    centers = (edges[:-1] + edges[1:]) * 0.5

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if vals_a.size:
        y_a, _ = np.histogram(vals_a, bins=edges, density=density)
        ax.plot(centers, y_a, lw=2, label=labels[0])
    if vals_b.size:
        y_b, _ = np.histogram(vals_b, bins=edges, density=density)
        ax.plot(centers, y_b, lw=2, label=labels[1])

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


# def plot_syn_records(
#         cell,
#         syn_records,
#         plotted_groups, # e.g. ['bg', 'stim']
#         plotted_props=('distance',), # 'distance', 'distance_density', 'weight', or ('weight','distance')
#         # labels=None,
#         plot_type='both', # 'hist' | 'line' | 'both'
#         bins=10.0, # bin width (µm or weight units)
#         win_size=25,
#         fig_sizes=(6, 4)):
#     """
#     syn_records    : dict -> {'bg_exc':[dict,…], 'bg_inh':[dict,…], 'stim':[dict,…], …}
#     plotted_groups : iterable of keys from syn_records
#     plotted_props  : see below
#     """

#     if plotted_groups == ['all']:
#         record_sets = [list(chain.from_iterable(syn_records.values()))]
#     else:
#         record_sets = [syn_records[g] for g in plotted_groups]
#     labels      = plotted_groups  # legend names follow plotted_groups


#     COLOR_CYCLER = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
#     alpha=0.6
#     # props = tuple(plotted_props)

#     # normalize variants like "weight probability" / "weight_probability"
#     def _norm(s):
#         return s.strip().lower().replace(' ', '_')

#     props = tuple(_norm(p) for p in plotted_props)
#     # Accept some aliases
#     if props == ('density_count',):
#         props = ('distance_count',)

#     def _auto_edges(data, bw):
#         lo, hi = data.min(), data.max()
#         return np.arange(lo, hi + bw, bw)

#     if plot_type not in ('hist', 'line', 'both'):
#         raise ValueError("plot_type must be 'hist', 'line', or 'both'")

#     # =============================== Distance / density =====================
#     if props in (('distance',), ('distance_count',), ('distance_probability',),
#                  ('distance_density',), ('density',)):

#         # "distance_density"/"density" keep your per-branch normalisation path
#         use_density_mode = props[0] in ('distance_density', 'density')
#         # when not using per-branch density, choose count vs probability
#         use_prob_mode    = props[0] == 'distance_probability'

#         plt.figure(figsize=fig_sizes)

#         for recs, lab in zip(record_sets, labels):
#             dist  = np.asarray([r['distance'] for r in recs])
#             sects = np.asarray([r['section']  for r in recs])
#             if not dist.size:
#                 continue

#             col    = next(COLOR_CYCLER)
#             # edges  = _auto_edges(dist, bins)
#             # centers = edges[:-1] + bins / 2
#             # counts, _ = np.histogram(dist, bins=edges)

#             if use_density_mode:
#                 # -------- per-branch density ----------
#                 edges = _auto_edges(dist, bins)
#                 len_exposure = np.zeros(len(edges) - 1)
#                 branches_in_bin = [set() for _ in range(len(edges) - 1)]

#                 h.distance(0, cell.soma[0](0.5))  # set origin for distance measurements
#                 for sec in cell.dend:  # or whatever section list you use
#                     for seg in sec:
#                         seg_dist = h.distance(seg)
#                         seg_len  = sec.L / sec.nseg
#                         i = np.digitize(seg_dist, edges) - 1
#                         if 0 <= i < len(len_exposure):
#                             len_exposure[i] += seg_len
#                             branches_in_bin[i].add(sec.name())

#                 # count synapses in bins
#                 counts, _ = np.histogram(dist, bins=edges)

#                 # density per µm of dendrite
#                 with np.errstate(divide='ignore', invalid='ignore'):
#                     density = np.divide(counts, len_exposure, where=len_exposure > 0)
#                 # optional: per branch average
#                 # branch_counts = np.array([len(s) for s in branches_in_bin])
#                 # density = np.divide(density, branch_counts, where=branch_counts > 0)

#                 ylab = f"Synapses / µm (bin={bins}µm)"

#                 if plot_type in ('hist', 'both'):
#                     plt.bar(centers, density, width=bins, alpha=alpha, color=col)

#                 if plot_type in ('line', 'both'):
#                     y_line = density.copy()
#                     y_line = moving_average(y_line, win_size=win_size, bin_width=bins, mode='center')
#                     x_line = _align_centers(centers, y_line, mode='center')
#                     plt.plot(x_line, y_line, lw=1.8, color=col, label=f'{lab} (smoothed)')
                    
#                     # y_line = moving_average(density, bins, win_size)
#                     # c_line = centers[:len(y_line)]
#                     # plt.plot(c_line, y_line, lw=1.8, color=col, label=f'{lab} (smoothed)')

#             else:
#                 # --------- plain distance: counts vs probability ----------
#                 if use_prob_mode:
#                     # probability density: histogram density=True, KDE unscaled
#                     ylab  = "Probability density (1/µm)"
#                     if plot_type in ('hist', 'both'):
#                         plt.hist(dist, bins=edges, density=True, alpha=alpha, color=col)
#                     if plot_type in ('line', 'both'):
#                         dens, _ = np.histogram(dist, bins=edges, density=True)
#                         y_line = dens.copy()
#                         y_line = moving_average(y_line, win_size=win_size, bin_width=bins, mode='center')
#                         x_line = _align_centers(centers, y_line, mode='center')
#                         plt.plot(x_line, y_line, lw=1.8, color=col, label=f'{lab} (smoothed)')
#                         # y_line = moving_average(dens, bins, win_size)
#                         # c_line = centers[:len(y_line)]
#                         # plt.plot(c_line, y_line, lw=1.8, color=col, label=f'{lab} (smoothed)')

#                 else:
#                     # counts per bin
#                     ylab  = f"Synapse count (bin={bins} µm)"
#                     if plot_type in ('hist', 'both'):
#                         plt.bar(centers, counts, width=bins, alpha=alpha, color=col)
#                     if plot_type in ('line', 'both'):
#                         y_line = counts.copy()
#                         y_line = moving_average(y_line, win_size=win_size, bin_width=bins, mode='center')
#                         x_line = _align_centers(centers, y_line, mode='center')
#                         plt.plot(x_line, y_line, lw=1.8, color=col, label=f'{lab} (smoothed)')
#                         # y_line = moving_average(counts, bins, win_size)
#                         # c_line = centers[:len(y_line)]
#                         # plt.plot(c_line, y_line, lw=1.8, color=col, label=f'{lab} (smoothed)')


#         plt.xlabel("Distance from soma (µm)")
#         plt.ylabel(ylab)
#         plt.title("Distance distribution")
#         plt.legend(); plt.tight_layout(); plt.show()
#         return

#     # ================================= Weight only ==========================
#     if props in (('weight',), ('weight_count',), ('weight_probability',)):
#         use_prob_mode = props[0] == 'weight_probability'

#         plt.figure(figsize=fig_sizes)
#         for recs, lab in zip(record_sets, labels):
#             w = np.asarray([(r['weight']) for r in recs])
#             if not w.size:
#                 continue
#             col   = next(COLOR_CYCLER)
#             edges = _auto_edges(w, bins)

#             if use_prob_mode:
#                 ylab = "Probability density"
#                 if plot_type in ('hist', 'both'):
#                     plt.hist(w, bins=edges, density=True, alpha=alpha, color=col)
#                 if plot_type in ('line', 'both'):
#                     dens, _ = np.histogram(w, bins=edges, density=True)
#                     c_line  = (edges[:-1] + edges[1:]) * 0.5
#                     y_line = dens.copy()
#                     y_line = moving_average(y_line, win_size=win_size, bin_width=bins, mode='center')
#                     x_line = _align_centers(centers, y_line, mode='center')
#                     plt.plot(x_line, y_line, lw=1.8, color=col, label=f'{lab} (smoothed)')
#                     # y_line  = moving_average(dens, bins, win_size)
#                     # c_line  = c_line[:len(y_line)]
#                     # plt.plot(c_line, y_line, color=col, lw=1.8, label=f'{lab} (smoothed)')
#             else:
#                 ylab = f"Synapse count (bin={bins})"
#                 if plot_type in ('hist', 'both'):
#                     plt.hist(w, bins=edges, alpha=alpha, color=col)
#                 if plot_type in ('line', 'both'):
#                     cnts, _ = np.histogram(w, bins=edges)
#                     c_line  = (edges[:-1] + edges[1:]) * 0.5
#                     y_line = cnts.copy()
#                     y_line = moving_average(y_line, win_size=win_size, bin_width=bins, mode='center')
#                     x_line = _align_centers(centers, y_line, mode='center')
#                     plt.plot(x_line, y_line, lw=1.8, color=col, label=f'{lab} (smoothed)')
#                     # y_line  = moving_average(cnts, bins, win_size)
#                     # c_line  = c_line[:len(y_line)]
#                     # plt.plot(c_line, y_line, color=col, lw=1.8, label=f'{lab} (smoothed)')

#         plt.xlabel("Synaptic weight")
#         plt.ylabel(ylab)
#         plt.title("Weight distribution")
#         plt.legend(); plt.tight_layout(); plt.show()
#         return

#     # ============================ Weight vs distance ========================
#     if set(props) == {'weight', 'distance'}:
#         plt.figure(figsize=fig_sizes)
#         for recs, lab in zip(record_sets, labels):
#             w = [r['weight']  for r in recs]
#             d = [r['distance'] for r in recs]
#             plt.scatter(d, w, s=5, alpha=1.0, label=lab)

#         plt.xlabel("Distance (µm)")
#         plt.ylabel("Weight (max) (nS)")
#         plt.title("Weight vs Distance")
#         plt.legend(); 
#         plt.tight_layout(); plt.show()
#         return

#     raise ValueError("plotted_props must be "
#                      "('distance',), ('distance_density',), "
#                      "('weight',), or ('weight','distance')")

def _auto_edges(data, bw):
    """
    Compute histogram bin edges given data and bin width bw.
    """
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return np.array([0.0, bw])

    lo, hi = data.min(), data.max()
    if lo == hi:
        # Expand a degenerate range slightly so we get at least one bin.
        lo -= 0.5 * bw
        hi += 0.5 * bw

    return np.arange(lo, hi + bw, bw)


def plot_syn_records(
        cell,
        syn_records,
        plotted_groups,                 # e.g. ['bg', 'stim'] or ['all']
        plotted_props=('distance',),    # 'distance', 'distance_count',
                                        # 'distance_probability', 'distance_density'/'density',
                                        # 'weight', 'weight_count', 'weight_probability',
        color = None,
        plot_type='both',               # 'hist' | 'line' | 'both'
        bins=10.0,                      # bin width (µm or weight units)
        win_size=25,                    # moving-average window (same units as `bins`)
        fig_sizes=(6, 4)):

    # helper to access fields from either dict or SynapseRecord dataclass
    def _rec_field(rec, key):
        if isinstance(rec, dict):
            return rec.get(key)
        return getattr(rec, key)

    # collect records and labels
    if plotted_groups == ['all']:
        record_sets = [list(chain.from_iterable(syn_records.values()))]
    else:
        record_sets = [syn_records[g] for g in plotted_groups]
    labels = plotted_groups

    COLOR_CYCLER = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    alpha = 0.6

    # support both old direct-hoc cells and new LoadedCell wrapper
    hoc_cell = getattr(cell, "h", cell)

    # normalise prop names
    def _norm(s):
        return s.strip().lower().replace(' ', '_')
    props = tuple(_norm(p) for p in plotted_props)
    if props == ('density_count',):     # alias
        props = ('distance_count',)

    if plot_type not in ('hist', 'line', 'both'):
        raise ValueError("plot_type must be 'hist', 'line', or 'both'")

    # =============================== Distance (and density) ===============================
    if props in (('distance',), ('distance_count',), ('distance_probability',),
                 ('distance_density',), ('density',)):

        use_density_mode = props[0] in ('distance_density', 'density')     # per-µm exposure
        use_prob_mode    = props[0] == 'distance_probability'              # PDF (1/µm)

        plt.figure(figsize=fig_sizes)
        ylab = ""   # set per-loop; used after loop

        for recs, lab in zip(record_sets, labels):
            dist  = np.asarray([_rec_field(r, 'distance') for r in recs])
            sects = np.asarray([_rec_field(r, 'section')  for r in recs])
            if not dist.size:
                continue

            if color is not None:
                col = color
            else:
                col = next(COLOR_CYCLER)

            edges   = _auto_edges(dist, bins)
            centers = (edges[:-1] + edges[1:]) * 0.5

            if use_density_mode:
                # ---- per-µm exposure in each distance bin ----
                len_exposure = np.zeros(len(edges) - 1)  # total dendritic length (µm) in each bin
                # Robust distance computation for hoc_cell (supports cell.h wrapper)
                h.distance(0, hoc_cell.soma[0](0.5))
                dend_secs = getattr(hoc_cell, "dend", []) if hasattr(hoc_cell, "dend") else []
                for sec in dend_secs:
                    seg_len = sec.L / max(sec.nseg, 1)
                    for seg in sec:
                        d = h.distance(seg)
                        idx = np.searchsorted(edges, d) - 1
                        if 0 <= idx < len_exposure.size:
                            len_exposure[idx] += seg_len

                counts, _ = np.histogram(dist, bins=edges)
                with np.errstate(divide='ignore', invalid='ignore'):
                    yvals = np.where(len_exposure > 0, counts / len_exposure, 0.0)  # Hz per µm

                ylab = 'Count per µm'
                if plot_type in ('hist', 'both'):
                    plt.bar(centers, yvals, width=bins, alpha=alpha, color=col, label=f'{lab} (per µm)')

                if plot_type in ('line', 'both'):
                    y_line = moving_average(yvals, win_size=win_size, bin_width=bins, mode='center')
                    x_line = _align_centers(centers, y_line, mode='center')
                    plt.plot(x_line, y_line, lw=1.8, color=col, label=f'{lab} (smoothed)')

            else:
                # plain distance: counts or probability
                if use_prob_mode:
                    yvals, edges = np.histogram(dist, bins=edges, density=True)
                    ylab = 'Probability density (1/µm)'
                else:
                    yvals, edges = np.histogram(dist, bins=edges)
                    ylab = 'Count'

                centers = (edges[:-1] + edges[1:]) * 0.5

                if plot_type in ('hist', 'both'):
                    plt.bar(centers, yvals, width=bins, alpha=alpha, color=col, label=f'{lab}')

                if plot_type in ('line', 'both'):
                    y_line = moving_average(yvals, win_size=win_size, bin_width=bins, mode='center')
                    x_line = _align_centers(centers, y_line, mode='center')
                    plt.plot(x_line, y_line, lw=1.8, color=col, label=f'{lab} (smoothed)')

        plt.xlabel('Distance from soma (µm)')
        plt.ylabel(ylab)
        plt.legend()
        plt.grid(True)
        plt.title('Synapse distance distribution')
        plt.show()

    # =============================== Weight distributions ===============================
    elif props in (('weight',), ('weight_count',), ('weight_probability',)):

        use_prob_mode = props[0] == 'weight_probability'

        plt.figure(figsize=fig_sizes)
        ylab = ""

        for recs, lab in zip(record_sets, labels):
            w = np.asarray([_rec_field(r, 'weight') for r in recs])
            if not w.size:
                continue

            if color is not None:
                col = color
            else:
                col = next(COLOR_CYCLER)

            edges   = _auto_edges(w, bins)
            centers = (edges[:-1] + edges[1:]) * 0.5

            if use_prob_mode:
                yvals, edges = np.histogram(w, bins=edges, density=True)
                ylab = 'Probability density (1/weight)'
            else:
                yvals, edges = np.histogram(w, bins=edges)
                ylab = 'Count'

            centers = (edges[:-1] + edges[1:]) * 0.5

            if plot_type in ('hist', 'both'):
                plt.bar(centers, yvals, width=bins, alpha=alpha, color=col, label=f'{lab}')

            if plot_type in ('line', 'both'):
                y_line = moving_average(yvals, win_size=win_size, bin_width=bins, mode='center')
                x_line = _align_centers(centers, y_line, mode='center')
                plt.plot(x_line, y_line, lw=1.8, color=col, label=f'{lab} (smoothed)')

        plt.xlabel('Synaptic weight')
        plt.ylabel(ylab)
        plt.legend()
        plt.grid(True)
        plt.title('Synaptic weight distribution')
        plt.show()

    # =============================== Joint (weight vs distance) ===============================
    elif props in (('weight', 'distance'), ('distance', 'weight')):

        plt.figure(figsize=fig_sizes)

        for recs, lab in zip(record_sets, labels):
            dist = np.asarray([_rec_field(r, 'distance') for r in recs])
            w    = np.asarray([_rec_field(r, 'weight')   for r in recs])
            if not dist.size or not w.size:
                continue

            if color is not None:
                col = color
            else:
                col = next(COLOR_CYCLER)

            plt.scatter(dist, w, alpha=alpha, color=col, label=lab)

        plt.xlabel('Distance from soma (µm)')
        plt.ylabel('Synaptic weight')
        plt.legend()
        plt.grid(True)
        plt.title('Synaptic weight vs distance')
        plt.show()

    else:
        raise ValueError(f"Unsupported plotted_props: {plotted_props!r}")


# ────────────────────────────────────────────────────────────────────────────
#  Plotting wrapper for new run_sim results
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
#  Plotting wrapper for new run_sim results
# ────────────────────────────────────────────────────────────────────────────
_MISSING = object()


def plot_results(
    results,
    syn_records=None,
    *,
    win_size=25,
    rate_style='line',
    raster_style=_MISSING,
    plot_window=None,
    in_vivo_curve=None,
    benchmark_path=None,
    benchmark_label="benchmark",
    plot_bio=None,
    shade_mode=None,
    alpha=0.6,
    line_width=2.0,
    shade_alpha=0.25,
    set_color=None,
    plot_type='line',
    plot_raster=None,
    smooth_mode="center",
    bin_ms=None,
):
    """
    Convenience wrapper for new run_sim results.

    Parameters
    ----------
    results : dict
        Output of run_sim.run_sim:
          {
            'mode': 'single' | 'multi' | 'param',
            'sim_cfg': {...},
            'traces': {...},
            'spikes': ...,
            'meta': {...},
          }
    syn_records : dict or None, optional
        Optional per-group synapse records (for raster/rate panels in single mode).
        For now you can pass None; voltage-only plots still work.
    in_vivo_curve : (t_s, rate_hz) or None
        Optional in-vivo curve to overlay in the rate panel for single-mode plots.
    """
    mode    = results.get("mode", "single")
    sim_cfg = results.get("sim_cfg", {})

    raster_style_set = raster_style is not _MISSING
    raster_style_val = "line" if not raster_style_set else raster_style

    # ── SINGLE: dispatch to plot_single ────────────────────────────────────
    if mode == "single":
        sim_traces = results.get("traces", {})
        if not sim_traces:
            raise ValueError("plot_results(single): results['traces'] is empty.")

        # Prefer explicitly passed syn_records; otherwise use what run_single stored.
        if syn_records is None:
            syn_records = results.get("syn_records", {}) or {}

        # mode-dependent default:
        #   None  => raster ON by default if syn_records present; otherwise OFF
        #   True  => enable raster with given raster_style
        #   False => disable raster
        if plot_raster is None:
            rs = raster_style_val if (raster_style_set or syn_records) else None
        else:
            rs = raster_style_val if plot_raster else None

        # Color: explicit override > sim_cfg["color"] > default None
        col = set_color if set_color is not None else sim_cfg.get("color", None)
        
        return plot_single(
            sim_traces,
            syn_records,
            sim_cfg,
            win_size=win_size,
            rate_style=rate_style,
            raster_style=rs,
            col=col,
            in_vivo_curve=in_vivo_curve,
            plot_window=plot_window or (None, None),
            bins=bin_ms,
            smooth_mode=smooth_mode,
        )

    # --- MULTI: plot trials from run_sim.run_multi ---------------------------
    if mode == "multi":
        # all_param_data: {label -> [spikes_per_trial]}
        spikes_by_trial = results.get("spikes", []) or []
        all_param_data = {"multi": spikes_by_trial}

        # derive basic sim_params from sim_cfg
        sim_params = {
            "tstop": float(sim_cfg.get("tstop", 0.0)),
            "bins": float(sim_cfg.get("bins", 25.0)),
            "delay": float(sim_cfg.get("delay", 0.0)),
            "n_trials": len(spikes_by_trial),
            "color": sim_cfg.get("color", None),
            "stim_start_ms": sim_cfg.get("stim_start_ms"),
            "stim_stop_ms": sim_cfg.get("stim_stop_ms"),
            "stim_duration_ms": sim_cfg.get("stim_duration_ms"),
        }
        if bin_ms is not None:
            sim_params["bins"] = float(bin_ms)

        # mode-dependent default:
        #   None  => raster ON by default for multi
        #   True  => raster ON
        #   False => raster OFF
        plot_rast_flag = True if plot_raster is None else bool(plot_raster)

        # color: explicit override > sim_cfg["color"] > default None
        col = set_color if set_color is not None else sim_cfg.get("color", None)

        # normalize plot_window to the dict form that plot_multi expects
        pw = plot_window
        if pw is not None and not isinstance(pw, dict):
            # assume (xmin, xmax) and let y be auto
            pw = {"x": (pw[0], pw[1]), "y": (None, None)}

        # convert in_vivo_curve -> old plot_bio triple if provided
        plot_bio = None
        if in_vivo_curve is not None:
            try:
                if isinstance(in_vivo_curve, (list, tuple)):
                    if len(in_vivo_curve) == 3:
                        _, t_s, rate = in_vivo_curve
                    elif len(in_vivo_curve) == 2:
                        t_s, rate = in_vivo_curve
                    else:
                        raise ValueError
                else:
                    t_s, rate = in_vivo_curve
                plot_bio = (True, np.asarray(t_s), np.asarray(rate))
            except Exception:
                plot_bio = None


        return plot_multi(
            all_param_data,
            sim_params=sim_params,
            win_size=win_size,
            plot_type=plot_type,
            plot_bio=plot_bio,
            benchmark_path=benchmark_path,
            benchmark_label=benchmark_label,
            plot_raster=plot_rast_flag,
            raster_style=raster_style_val,
            alpha=alpha,
            line_width=line_width,
            plot_window=pw,
            norm_fr=None,
            shade_mode=shade_mode,
            shade_alpha=shade_alpha,
            set_color=col,
            save_curve=False,
            smooth_mode=smooth_mode,
        )

 
    if mode == "param":
        raise NotImplementedError("plot_results: 'param' mode wiring is pending.")

    raise ValueError(f"plot_results: unsupported mode {mode!r}")


# ────────────────────────────────────────────────────────────────────────────
#  Single-simulation plot (Vm, raster, rates)
# ────────────────────────────────────────────────────────────────────────────
def plot_single(
        sim_traces,
        syn_records,
        sim_cfg,
        *,
        win_size     = None,
        rate_style   = 'line',   # 'hist' | 'line' | 'both' | None
        raster_style = 'line',   # 'line' | 'dot' | None
        col          = None,
        in_vivo_curve = None,    # (time_s, rate_hz) or None
        plot_window   = (None, None),
        bins          = None,    # optional override for bin width (ms)
        delay         = None,    # optional override for stimulus delay (ms)
        smooth_mode   = "center",
    ):
    """
    New interface (pipeline-friendly):

    Parameters
    ----------
    sim_traces : dict
        Output of run_sim.run_cell:
          {'T': array, 'V': array, 'I': array, 'G': array, ...}
    syn_records : dict
        Per-group synapse records from 2.4:
          {group_name: [SynapseRecord, ...], ...}
        Also works if entries are dicts with a 'spike_times' key.
    sim_cfg : dict
        Simulation config from 2.3 (normalized "sim" block).
        Must have 'tstop'; may optionally have 'bins' and 'delay'.

    Other arguments keep the same meaning as the original version.
    """

    # ---- unpack traces ----
    T = np.asarray(sim_traces['T'])
    V = np.asarray(sim_traces['V'])

    # ---- simulation timing / binning ----
    sim_duration_ms = float(sim_cfg.get('tstop', T[-1] if T.size else 0.0))

    # bin width for firing-rate histograms
    if bins is not None:
        bin_width = float(bins)
    else:
        bin_width = float(sim_cfg.get('bins', 25.0))  # default 25 ms

    # stimulus window (prefer explicit sim_cfg markers; fall back to legacy delay)
    if delay is not None:
        delay_ms = float(delay)
    else:
        delay_ms = float(sim_cfg.get('delay', 0.0))

    stim_start = sim_cfg.get("stim_start_ms")
    stim_stop = sim_cfg.get("stim_stop_ms")
    stim_dur = sim_cfg.get("stim_duration_ms")

    if stim_start is None:
        stim_start = delay_ms + 100.0
    if stim_stop is None:
        if stim_dur is not None:
            stim_stop = float(stim_start) + float(stim_dur)
        else:
            stim_stop = delay_ms + 550.0

    # ---- helper to extract spike_times from dict or SynapseRecord ----
    def _get_spike_times(rec):
        if isinstance(rec, dict):
            return np.asarray(rec.get('spike_times', []), dtype=float)
        # SynapseRecord dataclass (or similar)
        return np.asarray(getattr(rec, 'spike_times', []), dtype=float)

    # Dynamically determine groups and colours
    if syn_records is None:
        syn_records = {}
    groups = list(syn_records.keys())

    if raster_style and not syn_records:
        print("plot_single: raster_style requested but syn_records is empty; raster will be empty.")

    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    g2col  = {g: next(colors) for g in groups}

    # ------------------------- build raster data ---------------------------
    spikes_by_group = {
        g: (np.concatenate([_get_spike_times(r) for r in recs])
            if recs else np.array([], dtype=float))
        for g, recs in syn_records.items()
    }
    n_syn = {g: len(syn_records[g]) for g in groups}

    # ----------------------- GridSpec layout -------------------------------
    rows, layout, heights = 1, ['V'], [3]
    if raster_style:
        rows += 1; layout.append('R'); heights.append(3)
    if rate_style:
        rows += 1; layout.append('F'); heights.append(3)

    fig = plt.figure(figsize=(6, 3 * rows))
    gs  = fig.add_gridspec(rows, 1, height_ratios=heights, hspace=0.3)

    # -------------------------- A) Vm trace --------------------------------
    peaks, _ = find_peaks(V, height=-20, distance=2)
    spike_times  = T[peaks]
    print(
        f"Detected {len(spike_times)} spikes "
        f"(total avg: {(len(spike_times)/((sim_duration_ms-100)/1000))}) "
        f"at times (ms):", spike_times
    )

    axV = fig.add_subplot(gs[0])

    axV.plot(T, V, color=col)
    axV.scatter(spike_times, V[peaks], s=15, color='k', zorder=5)
    for vline in [stim_start, stim_stop]:
        axV.axvline(x=vline, color='k', linestyle='-', linewidth=1)

    axV.set_xlim(plot_window[0], plot_window[1])
    axV.set_ylabel('Vm (mV)')
    axV.set_title('Cell Output')
    axV.grid()

    row_idx = 1

    # -------------------------- B) raster ----------------------------------
    if 'R' in layout:
        axR = fig.add_subplot(gs[row_idx], sharex=axV)
        row_idx += 1

        def _draw(ax, recs, color, y0):
            for i, r in enumerate(recs):
                y = y0 + i + 1
                t = _get_spike_times(r)
                if t.size == 0:
                    continue
                if raster_style == 'line':
                    ax.vlines(t, y - .4, y + .4, color=color, lw=0.8)
                else:
                    ax.scatter(t, np.full_like(t, y), s=18, marker='.',
                               color=color)

        y0 = 0
        legend_elems = []
        for g in groups:
            _draw(axR, syn_records[g], g2col[g], y0)
            legend_elems.append(Line2D([0], [0], marker='.',
                                       color=g2col[g], linestyle='',
                                       ms=8, label=g))
            y0 += n_syn[g]

        # plot vertical lines for somatic spikes
        for spk in spike_times:
            axR.axvline(x=spk, color='k', linestyle='--', linewidth=1)

        for vline in [stim_start, stim_stop]:
            axR.axvline(x=vline, color='k', linestyle='-', linewidth=1)

        axR.set_xlim(plot_window[0], plot_window[1])
        axR.set_ylim(0.5, y0 + .5)
        axR.set_ylabel('Synapse ID')
        axR.set_title('Cell Input Raster')
        axR.grid(axis='x')
        if len(groups) > 1:
            axR.legend(handles=legend_elems, loc='upper left')

    # -------------------------- C) rates -----------------------------------
    if 'F' in layout:
        axF = fig.add_subplot(gs[row_idx], sharex=axV)
        bins_edges = np.arange(0, sim_duration_ms + bin_width, bin_width)
        centers    = bins_edges[:-1] + .5 * bin_width
        bw_sec     = bin_width / 1000.0

        rates = {}
        for g in groups:
            counts, _ = np.histogram(spikes_by_group[g], bins=bins_edges)
            rate      = counts / (max(n_syn[g], 1) * bw_sec)
            if win_size:
                rate  = moving_average(rate, win_size=win_size,
                                       bin_width=bin_width,
                                       mode=smooth_mode)
            rates[g] = rate

        if win_size:
            centers = _align_centers(centers, next(iter(rates.values())), mode=smooth_mode)

        if rate_style in ('hist', 'both'):
            for g in groups:
                axF.bar(centers, rates[g], width=bin_width,
                        color=g2col[g], alpha=.5, label=g)

        if rate_style in ('line', 'both'):
            for g in groups:
                axF.plot(centers, rates[g], lw=1.8, color=g2col[g], label=g)

        # combined curve: correctly averaged across *all* synapses
        counts_by_g = {
            g: np.histogram(spikes_by_group[g], bins=bins_edges)[0]
            for g in groups
        }

        total_syn    = sum(n_syn.values())
        total_counts = np.sum(list(counts_by_g.values()), axis=0)
        total_rate   = total_counts / (max(total_syn, 1) * bw_sec)

        if win_size:
            total_rate = moving_average(total_rate,
                                        win_size=win_size,
                                        bin_width=bin_width,
                                        mode=smooth_mode)

        # in-vivo data
        if in_vivo_curve is not None:
            try:
                t_invivo = np.asarray(in_vivo_curve[0], dtype=float) * 1000.0
                r_invivo = np.asarray(in_vivo_curve[1], dtype=float)
                axF.plot(t_invivo, r_invivo, 'k', lw=2, label='In-vivo')
            except Exception:
                # fall back silently if malformed
                pass

        # vertical lines for somatic spikes
        for spk in spike_times:
            axF.axvline(x=spk, color='k', linestyle='--', linewidth=1)

        for vline in [stim_start, stim_stop]:
            axF.axvline(x=vline, color='k', linestyle='-', linewidth=1)

        axF.set_ylabel('Rate (Hz / synapse)')
        axF.set_xlabel('Time (ms)')
        axF.set_xlim(plot_window[0], plot_window[1])
        axF.set_ylim(bottom=0)
        axF.set_title('Cell Input Average')
        if len(groups) > 1 or in_vivo_curve is not None:
            axF.legend()
        axF.grid()

    plt.show()


# ────────────────────────────────────────────────────────────────────────────
#  Param-study plotting helper for shaded plot
# ────────────────────────────────────────────────────────────────────────────
def _mean_band_stats(Y, *, shade=None, norm_fr=None, norm_scope='post_mean'):
    """
    Y: (n_trials, T) array
    shade: None | 'sem' | 'std' | float(multiplier of SEM) | (qlo,qhi) in [0,1]
    norm_scope: 'post_mean' (shift/scale mean and band together) or 'per_trial'
    Returns: mean, lo, hi  (lo/hi are None if shade is None)
    """
    Y = np.asarray(Y, float)
    n = Y.shape[0]

    # per-trial normalization (if requested)
    if norm_fr is not None and norm_scope == 'per_trial':
        Y = np.vstack([
            (y + norm_fr) / max(np.max(y + norm_fr), 1e-12)
            for y in Y
        ])

    mean = Y.mean(axis=0)

    # compute band on current Y
    lo = hi = None
    if shade is not None:
        if isinstance(shade, str):
            if shade == 'sem':
                spread = (Y.std(axis=0, ddof=1) / np.sqrt(n)) if n > 1 else np.zeros_like(mean)
                lo, hi = mean - spread, mean + spread
            elif shade == 'std':
                spread = Y.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
                lo, hi = mean - spread, mean + spread
            else:
                raise ValueError("shade must be 'sem', 'std', float, (qlo,qhi), or None")
        elif isinstance(shade, (int, float)):
            spread = (Y.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)) * float(shade)
            lo, hi = mean - spread, mean + spread
        else:  # (qlo, qhi)
            qlo, qhi = 100*shade[0], 100*shade[1]
            lo, hi = np.percentile(Y, [qlo, qhi], axis=0)

    # post-mean normalization (shift/scale mean and band together)
    if norm_fr is not None and norm_scope == 'post_mean':
        mean0 = mean + norm_fr
        scale = max(np.max(mean0), 1e-12)
        mean  = mean0 / scale
        if lo is not None and hi is not None:
            lo = (lo + norm_fr) / scale
            hi = (hi + norm_fr) / scale

    return mean, lo, hi


# ────────────────────────────────────────────────────────────────────────────
# Multi-simulation plot
# ────────────────────────────────────────────────────────────────────────────
def plot_multi(
        all_param_data,
        sim_params={},
        win_size=25,
        plot_type='line',       # 'hist' | 'line' | 'both'
        plot_bio=None,
        benchmark_path=None,
        benchmark_label="benchmark",
        plot_raster=False,
        raster_style='line',    # 'line' | 'dot'
        alpha=0.6,
        line_width=2.0,
        plot_window=None,       # {'x': (xmin,xmax), 'y': (ymin,ymax)} or None
        norm_fr=None,
        shade_mode=None,
        shade_alpha=0.25,
        set_color=None,
        save_curve=False,       # or filename
        smooth_mode="center",
        output_norm=None,
        output_scale=None,
        bio_scale=None,
    ):

    ok_rate = ('hist', 'line', 'both')
    ok_rast = ('line', 'dot')
    if plot_type not in ok_rate:
        raise ValueError(f"plot_type must be {ok_rate}")
    if raster_style not in ok_rast:
        raise ValueError(f"raster_style must be {ok_rast}")

    # basic timing params
    sim_duration_ms = float(sim_params.get('tstop', 0.0))
    bin_width       = float(sim_params.get('bins', 25.0))
    delay_ms        = float(sim_params.get('delay', 0.0))

    stim_start = sim_params.get("stim_start_ms")
    stim_stop = sim_params.get("stim_stop_ms")
    stim_dur = sim_params.get("stim_duration_ms")

    if stim_start is None:
        stim_start = delay_ms + 100.0
    if stim_stop is None:
        if stim_dur is not None:
            stim_stop = float(stim_start) + float(stim_dur)
        else:
            stim_stop = delay_ms + 550.0

    bw_s   = bin_width / 1000.0
    bins   = np.arange(0, sim_duration_ms + bin_width, bin_width)
    centers = bins[:-1] + 0.5 * bin_width

    groups = list(all_param_data.keys())
    scale_val = None
    if output_scale is not None:
        try:
            scale_val = float(output_scale)
        except Exception:
            scale_val = None
    bio_scale_val = None
    if bio_scale is not None:
        try:
            bio_scale_val = float(bio_scale)
        except Exception:
            bio_scale_val = None
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    g2col  = {g: next(colors) for g in groups}

    if plot_raster:
        fig, (axRate, axRaster) = plt.subplots(
            2, 1, figsize=(6, 8),
            sharex=True, gridspec_kw={'height_ratios': [1, 1]}
        )
    else:
        fig, axRate = plt.subplots(figsize=(6, 4))

    # ----------------------------- rate panel ------------------------------
    n_trials_any = 0
    for key in groups:
        trial_spikes = all_param_data.get(key, [])
        n_trials_any = max(n_trials_any, len(trial_spikes))

        per_trial_rates = []
        for tr in trial_spikes:
            tr = np.asarray(tr)
            counts, _ = np.histogram(tr, bins=bins)
            rate = counts / bw_s  # Hz per single trial
            per_trial_rates.append(rate)

        if not per_trial_rates:
            continue

        Y = np.vstack(per_trial_rates)  # shape (N_trials, T)
        x = centers.copy()

        # Moving average
        if win_size:
            Y_smooth = []
            for y in Y:
                ys = moving_average(y, win_size=win_size,
                                    bin_width=bin_width, mode=smooth_mode)
                Y_smooth.append(ys)
            T_new = min(len(y) for y in Y_smooth)
            Y = np.vstack([y[:T_new] for y in Y_smooth])
            drop = (len(x) - T_new) // 2
            x = x[drop: drop + T_new]

        col = set_color if set_color is not None else g2col[key]

        if output_norm:
            baseline_mean = output_norm.get("baseline_mean")
            norm_scale = output_norm.get("norm_scale")
            if baseline_mean is not None:
                Y = Y - float(baseline_mean)
            if norm_scale not in (None, 0):
                Y = Y / float(norm_scale)
        if scale_val not in (None, 1.0):
            Y = Y * scale_val

        norm_fr_local = None if output_norm else norm_fr
        mean, lo, hi = _mean_band_stats(
            Y, shade=shade_mode, norm_fr=norm_fr_local, norm_scope='post_mean'
        )

        if plot_type in ('hist', 'both'):
            axRate.bar(x, mean, width=bin_width, color=col,
                       alpha=alpha, label=key)

        if plot_type in ('line', 'both'):
            axRate.plot(x, mean, color=col, lw=float(line_width), label=key)
            if (shade_mode is not None) and (lo is not None) and (hi is not None):
                axRate.fill_between(x, lo, hi, color=col,
                                    alpha=float(shade_alpha), linewidth=0)

    # optional in-vivo curve
    if plot_bio and plot_bio[0]:
        bio_data = plot_bio[2]
        if norm_fr is not None:
            bio_data = normalize(bio_data, norm_offset=-1 * bio_data[0])
        if bio_scale_val not in (None, 1.0):
            bio_data = np.asarray(bio_data, dtype=float) * bio_scale_val
        axRate.plot(plot_bio[1] * 1000, bio_data, 'k',
                    lw=2, label='In-Vivo Input')

    # optional benchmark curve from old/new results
    if benchmark_path:
        try:
            from modules_local import run_sim as run_sim_mod
            try:
                bench_res = run_sim_mod.load_results(benchmark_path)
                bench_spikes = bench_res.get("spikes", []) if bench_res.get("mode") == "multi" else []
            except Exception:
                bench_res = run_sim_mod.load_old_multi_results(
                    benchmark_path,
                    label=None,
                    tstop=sim_duration_ms,
                    bins=bin_width,
                    delay=delay_ms,
                )
                bench_spikes = bench_res.get("spikes", [])

            if bench_spikes:
                bw_s = bin_width / 1000.0
                bins = np.arange(0, sim_duration_ms + bin_width, bin_width)
                centers = bins[:-1] + 0.5 * bin_width
                per_trial = []
                for tr in bench_spikes:
                    tr = np.asarray(tr)
                    counts, _ = np.histogram(tr, bins=bins)
                    per_trial.append(counts / bw_s)
                Y = np.vstack(per_trial)
                x = centers.copy()
                if win_size:
                    Y_smooth = []
                    for y in Y:
                        ys = moving_average(y, win_size=win_size,
                                            bin_width=bin_width, mode=smooth_mode)
                        Y_smooth.append(ys)
                    T_new = min(len(y) for y in Y_smooth)
                    Y = np.vstack([y[:T_new] for y in Y_smooth])
                    drop = (len(x) - T_new) // 2
                    x = x[drop: drop + T_new]
                mean = Y.mean(axis=0)
                axRate.plot(x, mean, color='k', lw=2, ls='--', label=benchmark_label)
        except Exception:
            pass

    if output_norm or norm_fr is not None:
        axRate.set_ylabel("Rate (normalized)")
    else:
        axRate.set_ylabel("Avg rate (Hz)")
    if len(groups) > 1 or (plot_bio and plot_bio[0]):
        axRate.legend()
    axRate.grid()
    for vline in [stim_start, stim_stop]:
        axRate.axvline(x=vline, color='k', linestyle='-', linewidth=1)

    if plot_window is not None:
        if plot_window.get('x') is not None:
            axRate.set_xlim(plot_window['x'][0], plot_window['x'][1])
        if plot_window.get('y') is not None:
            axRate.set_ylim(plot_window['y'][0], plot_window['y'][1])

    n_trials_label = sim_params.get("n_trials", n_trials_any)
    axRate.set_title(
        f"Multi-Trial Average Firing Rate "
        f"(Trials: {n_trials_label} | Win. = {win_size} ms)"
    )

    # ----------------------------- raster ----------------------------------
    if plot_raster:
        y0 = 0
        handles = []
        for key in groups:
            trial_spikes = all_param_data.get(key, [])

            col = set_color if set_color is not None else g2col[key]

            for tr in trial_spikes:
                tr = np.asarray(tr)
                if raster_style == 'line':
                    axRaster.vlines(tr, y0 + 0.5, y0 + 1.5, color=col, lw=1.0)
                else:
                    axRaster.scatter(tr, np.full_like(tr, y0 + 1),
                                     color=col, s=6, marker='.')
                y0 += 1

            handles.append(
                Line2D(
                    [0], [0],
                    color=col,
                    marker='|' if raster_style == 'line' else '.',
                    linestyle='', markersize=8, label=key
                )
            )

        axRaster.set_ylim(0.5, y0 + 0.5)
        axRaster.set_xlabel("Time (ms)")
        axRaster.set_ylabel("Trial #")
        if len(handles) > 1:
            axRaster.legend(handles=handles, loc='upper left')
        axRaster.grid(axis='x')
        for vline in [stim_start, stim_stop]:
            axRaster.axvline(x=vline, color='k', linestyle='-', linewidth=1)
    else:
        axRate.set_xlabel("Time (ms)")

# ────────────────────────────────────────────────────────────────────────────
#  Param-study simulation plot
# ────────────────────────────────────────────────────────────────────────────
def plot_param(all_param_data,
                    #  *,
                     param_study = {},
                     sim_params = {},
                     win_size = 25,
                     plot_type='line',
                     plot_bio=None,
                     plot_raster=False,
                     raster_style='line',
                     alpha=0.6,
                     plot_window = None,
                     norm_fr = None,
                     shade_mode = None,
                     set_color = None,
                     save_curve = False, #or filename
                     ):
    raise NotImplementedError("Parametric plotting from results is not implemented yet.")


# ────────────────────────────────────────────────────────────────────────────
#  Comparing multiple datasets plotting
# ────────────────────────────────────────────────────────────────────────────
def plot_compare_multi(
        plotted_files,
        sim_params,
        win_size=25, # None
        plot_bio = (False),
        plot_window = None,
        shade_mode = None,
        smooth_mode="center",
        ):
    

    plt.figure(figsize=(6,4))

    for pfs in plotted_files:
        with open(plotted_files[pfs]['data'], "rb") as f:
            all_param_data = pickle.load(f)
        
        bin_width = sim_params['bins']
        sim_dur = plotted_files[pfs]['dur']
        stim_start = sim_params['delay'] + 100 # Start/stop of stim manual for now, could be auto?
        stim_stop = sim_params['delay'] + 550
        norm_fr = plotted_files[pfs]['norm_fr']

        colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        if plotted_files[pfs]['color'] is not None:
            col = plotted_files[pfs]['color']
        else:
            col = next(colors)

        bw_s  = bin_width / 1000
        bins  = np.arange(0, sim_dur + bin_width, bin_width)
        centers = bins[:-1] + .5 * bin_width
        trial_spikes = all_param_data[list(all_param_data.keys())[0]] # Works for only one param right now
        # Could create color gradient based on col for multiple params in groups

        # Build per-trial rate matrix (N, T) for variability data
        per_trial_rates = []
        for tr in trial_spikes:
            tr = np.asarray(tr)
            counts, _ = np.histogram(tr, bins=bins)
            rate = counts / bw_s  # Hz per single trial
            per_trial_rates.append(rate)

        Y = np.vstack(per_trial_rates)  # shape (N_trials, T)
        x = centers.copy()

        # Moving Average
        if win_size:
            Y_smooth = []
            for y in Y:
                ys = moving_average(y, win_size=win_size, bin_width=bin_width, mode=smooth_mode)
                Y_smooth.append(ys)
            T_new = min(len(y) for y in Y_smooth) # After smoothing, length shrinks: align x
            Y = np.vstack([y[:T_new] for y in Y_smooth])
            drop = (len(x) - T_new) // 2
            x = x[drop : drop + T_new]

        # get mean & band with your chosen normalization scope
        mean, lo, hi = _mean_band_stats(
            Y, shade=shade_mode,
            norm_fr=norm_fr,           # None => no normalization
            norm_scope='post_mean'     # or 'per_trial' if you prefer
        )

        # 1) plot the mean line (always)
        plt.plot(x, mean, color=col, lw=2, label=f'{pfs} ({len(trial_spikes)} trials)')

        # 2) add shaded band if requested
        if shade_mode is not None and lo is not None and hi is not None:
            plt.fill_between(x, lo, hi, color=col, alpha=0.25, linewidth=0)

        # if shade_mode is not None:
        #     _mean_band_stats(
        #         plt, x, Y,
        #         label=label, color=col,
        #         shade=shade_mode, alpha=0.25,
        #         norm_fr=norm_fr, norm_scope='post_mean'
        #     )
        # else: # no shading: just the mean line
        #     plt.plot(x, Y.mean(axis=0), color=col, lw=2, label=label)

    #  Plot additional bio curves
    if plot_bio and plot_bio[0]:
        bio_data = plot_bio[2]
        if norm_fr is not None:
            bio_data = normalize(bio_data,norm_offset=-1*bio_data[0])
        plt.plot(plot_bio[1] * 1000, bio_data, 'k', lw=2, label='In-Vivo Input')

    for vline in [stim_start,stim_stop]: plt.axvline(x=vline,color='k',linestyle='-',linewidth=1)
    # Set up fig
    plt.ylabel('Avg Rate (Hz)')
    plt.xlabel('Time (ms)')
    plt.legend()
    plt.grid()
    if plot_window is not None:
        plt.xlim(plot_window['x'][0],plot_window['x'][1])
        plt.ylim(plot_window['y'][0],plot_window['y'][1])
    plt.title(f'Comparison of Average Firing Rates (Win: {win_size} ms)')


# ────────────────────────────────────────────────────────────────────────────
#  Side-by-side comparison for two results
# ────────────────────────────────────────────────────────────────────────────
def _rate_curve_from_results(
    results,
    win_size=25,
    bin_ms=None,
    smooth_mode="center",
    output_norm=None,
):
    sim_cfg = results.get("sim_cfg", {}) or {}
    tstart = float(sim_cfg.get("tstart", 0.0))
    tstop = float(sim_cfg.get("tstop", 0.0))
    if tstop <= tstart:
        return np.array([]), np.array([]), 0

    bin_width = float(bin_ms if bin_ms is not None else sim_cfg.get("bins", 25.0))
    bins = np.arange(tstart, tstop + bin_width, bin_width)
    centers = bins[:-1] + 0.5 * bin_width
    bw_s = bin_width / 1000.0

    spikes = results.get("spikes")
    if spikes is None:
        spikes = []
    if results.get("mode") == "multi":
        if isinstance(spikes, np.ndarray):
            trials = [np.asarray(tr) for tr in spikes.tolist()]
        elif isinstance(spikes, (list, tuple)):
            trials = [np.asarray(tr) for tr in spikes]
        else:
            trials = [np.asarray(spikes)]
    else:
        trials = [np.asarray(spikes)]

    if not trials:
        return centers, np.zeros_like(centers), 0

    rates = []
    for tr in trials:
        counts, _ = np.histogram(tr, bins=bins)
        rates.append(counts / bw_s)

    Y = np.vstack(rates)
    mean = Y.mean(axis=0)

    if win_size:
        mean = moving_average(mean, win_size=win_size, bin_width=bin_width, mode=smooth_mode)
        centers = _align_centers(centers, mean, mode=smooth_mode)

    if output_norm:
        baseline_mean = output_norm.get("baseline_mean")
        norm_scale = output_norm.get("norm_scale")
        if baseline_mean is not None:
            mean = mean - float(baseline_mean)
        if norm_scale not in (None, 0):
            mean = mean / float(norm_scale)

    return centers, mean, len(trials)


def plot_compare_side_by_side(
        results_a,
        results_b,
        *,
        labels=("A", "B"),
        win_size=25,
        bin_ms=None,
        plot_window=None,
        colors=None,
        smooth_mode="center",
        output_norms=None,
        layout="side-by-side",
        output_scale=None,
        stim_start_ms=None,
        stim_stop_ms=None):
    """
    Plot two results using the requested comparison layout.
    """
    layout = (layout or "side-by-side").lower()
    if layout in ("stacked", "top-bottom", "vertical"):
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True, sharey=True)
    elif layout in ("overlay", "same", "same-plot", "overlap"):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        axes = np.array([ax])
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    axes = np.atleast_1d(axes)
    norms = output_norms or (None, None)
    colors_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    curves = []
    scale_val = None
    if output_scale is not None:
        try:
            scale_val = float(output_scale)
        except Exception:
            scale_val = None
    for idx, (res, label, color, norm) in enumerate(
            zip((results_a, results_b), labels, (colors or [None, None]), norms)):
        sim_cfg = res.get("sim_cfg", {}) or {}
        x, mean, n_trials = _rate_curve_from_results(
            res,
            win_size=win_size,
            bin_ms=bin_ms,
            smooth_mode=smooth_mode,
            output_norm=norm,
        )

        col = color
        if col is None:
            col = sim_cfg.get("color", None)
        if col is None:
            col = colors_cycle[idx % len(colors_cycle)]

        stim_start = sim_cfg.get("stim_start_ms")
        stim_stop = sim_cfg.get("stim_stop_ms")
        stim_dur = sim_cfg.get("stim_duration_ms")
        if stim_start is not None and stim_stop is None and stim_dur is not None:
            stim_stop = float(stim_start) + float(stim_dur)
        if stim_start_ms is not None:
            try:
                stim_start = float(stim_start_ms)
            except Exception:
                pass
        if stim_stop_ms is not None:
            try:
                stim_stop = float(stim_stop_ms)
            except Exception:
                pass

        if scale_val not in (None, 1.0):
            mean = mean * scale_val
        curves.append({
            "x": x,
            "mean": mean,
            "n_trials": n_trials,
            "label": label,
            "color": col,
            "stim_start": stim_start,
            "stim_stop": stim_stop,
        })

    if layout in ("overlay", "same", "same-plot", "overlap"):
        ax = axes[0]
        same_color = False
        if len(curves) > 1:
            same_color = curves[0]["color"] == curves[1]["color"]
        for curve in curves:
            if curve["x"].size == 0:
                continue
            label = curve["label"]
            if label:
                label = f"{label} (n={curve['n_trials']})"
            ls = "--" if same_color and curve is curves[1] else "-"
            ax.plot(curve["x"], curve["mean"], color=curve["color"], lw=2, linestyle=ls, label=label)

        base_start = curves[0]["stim_start"] if curves else None
        base_stop = curves[0]["stim_stop"] if curves else None
        for vline in [base_start, base_stop]:
            if vline is not None:
                ax.axvline(x=vline, color='k', linestyle='-', linewidth=1)

        if len(curves) > 1:
            for vline in [curves[1]["stim_start"], curves[1]["stim_stop"]]:
                if vline is not None and vline not in (base_start, base_stop):
                    ax.axvline(
                        x=vline,
                        color=curves[1]["color"],
                        linestyle='--',
                        linewidth=1,
                        alpha=0.7,
                    )

        if any(curve["label"] for curve in curves):
            ax.legend()
        ax.set_title("Compare (overlay)")
        ax.set_xlabel("Time (ms)")
        ax.grid(True)
        if plot_window is not None:
            ax.set_xlim(plot_window[0], plot_window[1])
    else:
        for ax, curve in zip(axes, curves):
            label = curve["label"]
            ax.plot(curve["x"], curve["mean"], color=curve["color"], lw=2, label=label if label else None)
            for vline in [curve["stim_start"], curve["stim_stop"]]:
                if vline is not None:
                    ax.axvline(x=vline, color='k', linestyle='-', linewidth=1)
            ax.set_title(f"{curve['label']} (n={curve['n_trials']})")
            ax.set_xlabel("Time (ms)")
            ax.grid(True)
            if plot_window is not None:
                ax.set_xlim(plot_window[0], plot_window[1])
            if label:
                ax.legend()

    if any(norms):
        axes[0].set_ylabel("Rate (normalized)")
    else:
        axes[0].set_ylabel("Avg Rate (Hz)")
    plt.tight_layout()
    plt.show()
    return fig, axes
        
