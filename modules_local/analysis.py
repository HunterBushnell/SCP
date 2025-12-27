"""
Simple analysis helpers for single-cell simulation results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from collections import Counter
import csv
import json
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt


def _get_duration_ms(results: Dict[str, Any]) -> Optional[float]:
    sim_cfg = results.get("sim_cfg", {}) or {}
    tstart = float(sim_cfg.get("tstart", 0.0))
    tstop = sim_cfg.get("tstop")
    if tstop is None:
        traces = results.get("traces", {}) or {}
        T = traces.get("T")
        if T is not None and len(T) > 0:
            tstop = float(T[-1])
    if tstop is None:
        return None
    return float(tstop) - float(tstart)


def summarize_spike_trials(
    results: Dict[str, Any],
    *,
    plot: bool = True,
    bins: Optional[int] = None,
    figsize: tuple[float, float] = (8.0, 3.0),
    print_summary: bool = True,
) -> Dict[str, Any]:
    """
    Summarize spike counts (and rates if duration is known) per trial.

    Parameters
    ----------
    results : dict
        Output from run_sim.run_sim or run_sim.load_results.
    plot : bool
        If True, plots spike counts per trial and a histogram.
    bins : int, optional
        Histogram bins. If None, uses a sqrt-based heuristic.
    figsize : tuple
        Figure size for the summary plots.
    print_summary : bool
        If True, prints count/rate summaries to stdout.
    """
    spikes = results.get("spikes")
    if spikes is None:
        if print_summary:
            print("No spikes in results.")
        return {
            "n_trials": 0,
            "counts": [],
            "duration_ms": _get_duration_ms(results),
            "rates_hz": None,
            "mean_count": 0.0,
            "std_count": 0.0,
            "min_count": 0,
            "max_count": 0,
            "mean_rate_hz": None,
            "std_rate_hz": None,
        }

    if results.get("mode") == "multi" and isinstance(spikes, (list, tuple)):
        spikes_by_trial = list(spikes)
    else:
        spikes_by_trial = [spikes]

    counts = np.array([len(np.asarray(s)) for s in spikes_by_trial], dtype=float)
    n_trials = len(counts)
    duration_ms = _get_duration_ms(results)

    rates_hz = None
    if duration_ms and duration_ms > 0:
        rates_hz = counts / (duration_ms / 1000.0)

    stats = {
        "n_trials": n_trials,
        "counts": counts.tolist(),
        "duration_ms": duration_ms,
        "mean_count": float(np.mean(counts)) if n_trials else 0.0,
        "std_count": float(np.std(counts)) if n_trials else 0.0,
        "min_count": int(np.min(counts)) if n_trials else 0,
        "max_count": int(np.max(counts)) if n_trials else 0,
        "rates_hz": rates_hz.tolist() if rates_hz is not None else None,
        "mean_rate_hz": float(np.mean(rates_hz)) if rates_hz is not None and n_trials else None,
        "std_rate_hz": float(np.std(rates_hz)) if rates_hz is not None and n_trials else None,
    }

    if print_summary:
        print(f"Trials: {n_trials}")
        print(
            "Spike count per trial: "
            f"mean={stats['mean_count']:.2f}, std={stats['std_count']:.2f}, "
            f"min={stats['min_count']}, max={stats['max_count']}"
        )
        print("Counts (first 10):", stats["counts"][:10])
        if rates_hz is not None:
            print(
                "Avg rate per trial (Hz): "
                f"mean={stats['mean_rate_hz']:.2f}, std={stats['std_rate_hz']:.2f}"
            )

    if plot:
        if n_trials > 1:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            axes[0].plot(range(n_trials), counts, marker="o", linewidth=1)
            axes[0].set_xlabel("Trial")
            axes[0].set_ylabel("Spike count")
            axes[0].set_title("Spikes per trial")

            if bins is None:
                bins = min(20, max(5, int(n_trials ** 0.5)))
            axes[1].hist(counts, bins=bins)
            axes[1].set_xlabel("Spike count")
            axes[1].set_ylabel("Trials")
            axes[1].set_title("Spike count distribution")
            plt.tight_layout()
        else:
            plt.figure(figsize=(max(4.0, figsize[0] / 2), figsize[1]))
            plt.bar([0], counts)
            plt.xticks([0], ["trial_0"])
            plt.ylabel("Spike count")
            plt.title("Spikes per trial")
            plt.tight_layout()

    return stats


def _moving_average(values: np.ndarray, win_bins: int) -> np.ndarray:
    if win_bins <= 1:
        return values
    kernel = np.ones(int(win_bins), dtype=float) / float(win_bins)
    return np.convolve(values, kernel, mode="same")


def _bin_trains(
    trains: Iterable[np.ndarray],
    tstart: float,
    tstop: float,
    bin_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    trains = list(trains)
    edges = np.arange(tstart, tstop + bin_ms, bin_ms, dtype=float)
    if edges.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    counts = np.zeros(edges.size - 1, dtype=float)
    for tr in trains:
        if len(tr) == 0:
            continue
        c, _ = np.histogram(tr, bins=edges)
        counts += c
    n_syn = max(len(trains), 1)
    rate = counts / (n_syn * (bin_ms / 1000.0))
    centers = edges[:-1] + bin_ms * 0.5
    return centers, rate


def _hoc(cell: Any):
    return getattr(cell, "h", cell)


def _collect_sections(h: Any) -> Dict[str, list]:
    return {
        "soma": list(h.soma) if hasattr(h, "soma") else [],
        "dend": list(h.dend) if hasattr(h, "dend") else [],
        "apic": list(h.apic) if hasattr(h, "apic") else [],
        "axon": list(h.axon) if hasattr(h, "axon") else [],
        "all": [sec for sec in h.allsec()],
    }


def _section_stats(sec_list, *, include_names: bool = False) -> Dict[str, Any]:
    sec_list = list(sec_list or [])
    segs = [seg for sec in sec_list for seg in sec]
    n_sections = len(sec_list)
    n_segments = len(segs)

    total_length = float(sum(sec.L for sec in sec_list))
    seg_lengths = [seg.sec.L / max(seg.sec.nseg, 1) for seg in segs]
    diameters = [float(seg.diam) for seg in segs]

    total_area = 0.0
    for seg, seg_len in zip(segs, seg_lengths):
        total_area += math.pi * float(seg.diam) * float(seg_len)

    stats = {
        "n_sections": n_sections,
        "n_segments": n_segments,
        "total_length_um": total_length,
        "mean_section_length_um": (total_length / n_sections) if n_sections else 0.0,
        "mean_segment_length_um": (total_length / n_segments) if n_segments else 0.0,
        "total_area_um2": total_area,
        "mean_diam_um": float(np.mean(diameters)) if diameters else None,
        "min_diam_um": float(np.min(diameters)) if diameters else None,
        "max_diam_um": float(np.max(diameters)) if diameters else None,
    }

    if include_names:
        stats["section_names"] = [sec.name() for sec in sec_list]

    return stats


def summarize_cell_sections(cell: Any, *, include_names: bool = False) -> Dict[str, Any]:
    """
    Summarize section/segment counts, lengths, and diameters per section group.
    """
    h = _hoc(cell)
    sections = _collect_sections(h)
    summary = {
        name: _section_stats(secs, include_names=include_names)
        for name, secs in sections.items()
    }
    return summary


def summarize_mechanisms(cell: Any, *, max_mechs: Optional[int] = 20) -> Dict[str, Any]:
    """
    Summarize density/point mechanisms per section group.
    """
    h = _hoc(cell)
    sections = _collect_sections(h)
    summary: Dict[str, Any] = {}

    for group, sec_list in sections.items():
        mech_counts: Counter[str] = Counter()
        point_counts: Counter[str] = Counter()
        ion_counts: Counter[str] = Counter()

        for sec in sec_list:
            info = sec.psection()
            density = info.get("density_mechs", {}) or {}
            points = info.get("point_mechs", info.get("point_processes", {})) or {}
            ions = info.get("ions", {}) or {}

            for name in density.keys():
                mech_counts[name] += 1
            for name in points.keys():
                point_counts[name] += 1
            for name in ions.keys():
                ion_counts[name] += 1

        def _trim(counter: Counter[str]) -> Dict[str, int]:
            items = counter.most_common()
            if max_mechs is not None:
                items = items[: max(0, int(max_mechs))]
            return {k: int(v) for k, v in items}

        summary[group] = {
            "density_mechs": _trim(mech_counts),
            "point_mechs": _trim(point_counts),
            "ions": _trim(ion_counts),
        }

    return summary


def _distance_stats(distances: np.ndarray) -> Dict[str, Any]:
    if distances.size == 0:
        return {
            "n": 0,
            "min_um": None,
            "max_um": None,
            "mean_um": None,
            "std_um": None,
        }
    return {
        "n": int(distances.size),
        "min_um": float(distances.min()),
        "max_um": float(distances.max()),
        "mean_um": float(distances.mean()),
        "std_um": float(distances.std()),
    }


def _auto_edges(data: np.ndarray, bin_um: float) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return np.array([0.0, float(bin_um)], dtype=float)
    lo, hi = data.min(), data.max()
    if lo == hi:
        lo -= 0.5 * bin_um
        hi += 0.5 * bin_um
    return np.arange(lo, hi + bin_um, bin_um, dtype=float)


def _band_counts(distances: np.ndarray, bands: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for band in bands:
        name = band.get("name", "<band>")
        low = band.get("low")
        high = band.get("high")
        lo = float(low) if low is not None else None
        hi = float(high) if high is not None else None
        mask = np.ones(distances.shape, dtype=bool)
        if lo is not None:
            mask &= distances >= lo
        if hi is not None:
            mask &= distances < hi
        counts[name] = int(mask.sum())
    return counts


def summarize_geometry(
    geom: Dict[str, Any],
    *,
    include_dist_hist: bool = False,
    dist_bin_um: float = 25.0,
    geom_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Summarize geometry groups from define_geometry with distance stats.
    """
    groups = geom.get("groups", {}) or {}
    summary: Dict[str, Any] = {"groups": {}, "meta": geom.get("meta", {}) or {}}

    for name, segs in groups.items():
        distances = np.asarray([float(ref.dist_um) for ref in segs], dtype=float)
        stats = _distance_stats(distances)
        if include_dist_hist:
            edges = _auto_edges(distances, float(dist_bin_um))
            counts, _ = np.histogram(distances, bins=edges)
            centers = (edges[:-1] + edges[1:]) * 0.5
            stats["dist_hist"] = {
                "bin_um": float(dist_bin_um),
                "centers_um": centers.tolist(),
                "counts": counts.tolist(),
            }
        summary["groups"][name] = stats

    if geom_config and geom_config.get("radial_bands") and "all_dend" in groups:
        dists = np.asarray([float(ref.dist_um) for ref in groups["all_dend"]], dtype=float)
        summary["radial_bands"] = _band_counts(dists, geom_config["radial_bands"])

    return summary


def _classify_distance(
    distance: float,
    section: Optional[str],
    thresholds: Optional[Dict[str, Any]],
) -> str:
    if section and "soma" in section:
        return "soma"

    if not thresholds:
        return "unknown"

    prox = thresholds.get("proximal", {}) or {}
    dist = thresholds.get("distal", {}) or {}
    prox_low = prox.get("low")
    prox_high = prox.get("high")
    dist_low = dist.get("low")

    d = float(distance)
    if prox_low is not None and d <= float(prox_low):
        return "soma"
    if prox_high is None or d < float(prox_high):
        return "proximal"
    if dist_low is None or d >= float(dist_low):
        return "distal"
    return "other"


def summarize_synapse_records(
    syn_records: Dict[str, Iterable[Any]],
    *,
    geom: Optional[Dict[str, Any]] = None,
    duration_ms: Optional[float] = None,
    include_spike_stats: bool = True,
) -> Dict[str, Any]:
    """
    Summarize synapse placement and weights from syn_records.
    """
    thresholds = None
    if geom is not None:
        thresholds = (geom.get("meta", {}) or {}).get("thresholds_um")

    summary: Dict[str, Any] = {"groups": {}, "total_n_syn": 0}

    def _rec_field(rec, key):
        if isinstance(rec, dict):
            return rec.get(key)
        return getattr(rec, key, None)

    for group, recs in (syn_records or {}).items():
        rec_list = list(recs or [])
        if not rec_list:
            continue

        weights = np.asarray([_rec_field(r, "weight") for r in rec_list], dtype=float)
        dists = np.asarray([_rec_field(r, "distance") for r in rec_list], dtype=float)
        sections = [str(_rec_field(r, "section")) for r in rec_list]
        spike_counts = np.asarray(
            [len(_rec_field(r, "spike_times") or []) for r in rec_list], dtype=float
        )

        section_counts = Counter(sections)
        placement_counts = Counter()
        if thresholds:
            for dist, sec in zip(dists, sections):
                placement_counts[_classify_distance(dist, sec, thresholds)] += 1

        group_summary = {
            "n_syn": int(len(rec_list)),
            "weight_mean": float(weights.mean()) if weights.size else None,
            "weight_std": float(weights.std()) if weights.size else None,
            "weight_min": float(weights.min()) if weights.size else None,
            "weight_max": float(weights.max()) if weights.size else None,
            "distance_mean": float(dists.mean()) if dists.size else None,
            "distance_std": float(dists.std()) if dists.size else None,
            "distance_min": float(dists.min()) if dists.size else None,
            "distance_max": float(dists.max()) if dists.size else None,
            "section_counts": dict(section_counts),
        }

        if placement_counts:
            group_summary["placement_counts"] = dict(placement_counts)

        if include_spike_stats:
            group_summary["spikes_per_syn_mean"] = float(spike_counts.mean()) if spike_counts.size else 0.0
            group_summary["spikes_per_syn_std"] = float(spike_counts.std()) if spike_counts.size else 0.0
            group_summary["spikes_per_syn_min"] = float(spike_counts.min()) if spike_counts.size else 0.0
            group_summary["spikes_per_syn_max"] = float(spike_counts.max()) if spike_counts.size else 0.0
            if duration_ms and duration_ms > 0:
                rate = spike_counts / (duration_ms / 1000.0)
                group_summary["rate_hz_mean"] = float(rate.mean())
                group_summary["rate_hz_std"] = float(rate.std())

        summary["groups"][group] = group_summary
        summary["total_n_syn"] += int(len(rec_list))

    return summary


def load_inputs_sample(run_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Load inputs_sample.pkl from a run directory (or run/results).
    """
    p = Path(run_dir)
    candidates = [
        p / "inputs_sample.pkl",
        p / "results" / "inputs_sample.pkl",
    ]
    for c in candidates:
        if c.is_file():
            with c.open("rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"inputs_sample.pkl not found under {p}")


def summarize_inputs_from_payload(
    payload: Dict[str, Any],
    sim_cfg: Dict[str, Any],
    *,
    groups: Optional[Iterable[str]] = None,
    bin_ms: Optional[float] = None,
    smooth_ms: Optional[float] = None,
    max_trials: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Summarize saved inputs into mean/std rate curves per group.
    """
    tstart = float(sim_cfg.get("tstart", 0.0))
    tstop = float(sim_cfg.get("tstop", 0.0))
    if bin_ms is None:
        bin_ms = float(sim_cfg.get("bins", 5.0))
    bin_ms = float(bin_ms)

    if bin_ms <= 0:
        raise ValueError("bin_ms must be > 0")

    trial_inputs = []
    if payload.get("inputs_by_trial"):
        trial_inputs = [entry.get("inputs", {}) for entry in payload["inputs_by_trial"]]
    elif payload.get("inputs"):
        trial_inputs = [payload["inputs"]]
    else:
        raise KeyError("inputs payload missing inputs or inputs_by_trial")

    if max_trials is not None:
        trial_inputs = trial_inputs[: max(0, int(max_trials))]

    group_names = set()
    for inputs in trial_inputs:
        group_names.update(inputs.keys())
    if groups is not None:
        group_names = set(groups).intersection(group_names)
    group_names = sorted(group_names)

    summary: Dict[str, Any] = {
        "bin_ms": bin_ms,
        "tstart_ms": tstart,
        "tstop_ms": tstop,
        "t_ms": None,
        "n_trials": len(trial_inputs),
        "groups": {},
    }

    for g in group_names:
        rates = []
        n_syn = None
        centers_ref = None
        for inputs in trial_inputs:
            gdata = inputs.get(g, {}) or {}
            trains = [np.asarray(t, dtype=float) for t in (gdata.get("spike_trains") or [])]
            if not trains:
                continue
            centers, rate = _bin_trains(trains, tstart, tstop, bin_ms)
            if centers_ref is None:
                centers_ref = centers
            rates.append(rate)
            if n_syn is None:
                n_syn = len(trains)

        if not rates:
            continue
        rates_arr = np.vstack(rates)
        mean_rate = rates_arr.mean(axis=0)
        std_rate = rates_arr.std(axis=0)
        if smooth_ms is not None:
            win_bins = int(round(float(smooth_ms) / bin_ms))
            mean_rate = _moving_average(mean_rate, win_bins)
            std_rate = _moving_average(std_rate, win_bins)

        summary["t_ms"] = centers_ref.tolist() if centers_ref is not None else None
        summary["groups"][g] = {
            "mean_rate": mean_rate.tolist(),
            "std_rate": std_rate.tolist(),
            "n_trials": int(rates_arr.shape[0]),
            "n_syn": int(n_syn or 0),
        }

    return summary


def summarize_inputs_from_results(
    results: Dict[str, Any],
    *,
    groups: Optional[Iterable[str]] = None,
    bin_ms: Optional[float] = None,
    smooth_ms: Optional[float] = None,
    max_trials: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper around summarize_inputs_from_payload for loaded results.
    """
    payload: Dict[str, Any] = {}
    if results.get("inputs_by_trial") is not None:
        payload["inputs_by_trial"] = results.get("inputs_by_trial")
    if results.get("inputs") is not None:
        payload["inputs"] = results.get("inputs")
    if not payload:
        raise KeyError("Results missing inputs/inputs_by_trial.")
    sim_cfg = results.get("sim_cfg", {}) or {}
    return summarize_inputs_from_payload(
        payload,
        sim_cfg,
        groups=groups,
        bin_ms=bin_ms,
        smooth_ms=smooth_ms,
        max_trials=max_trials,
    )


def save_default_plots(
    results: Dict[str, Any],
    run_dir: Union[str, Path],
    *,
    save_inputs: bool = True,
    save_synapses: bool = False,
    win_size: float = 50.0,
    input_bin_ms: Optional[float] = None,
    input_smooth_ms: Optional[float] = 50.0,
    raster_style: str = "dot",
) -> Dict[str, Path]:
    """
    Save a small set of default plots into <run_dir>/plots.

    Returns a dict of plot name -> file path.
    """
    from modules_local import plotting  # local import to avoid circular deps

    run_dir = Path(run_dir)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, Path] = {}

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
    fig_out.savefig(out_path, dpi=150)
    saved["output_plot"] = out_path

    # Input mean curves
    if save_inputs:
        try:
            summary = summarize_inputs_from_results(
                results,
                bin_ms=input_bin_ms,
                smooth_ms=input_smooth_ms,
            )
            fig_in, _ = plotting.plot_input_means(
                summary,
                label="inputs",
                groups=None,
                show_std=False,
                output_curve=(results.get("meta") or {}).get("avg_rate_curve"),
            )
            in_path = plot_dir / "inputs_mean.png"
            fig_in.savefig(in_path, dpi=150)
            saved["inputs_mean"] = in_path
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
            plt.gcf().savefig(syn_path, dpi=150)
            saved["syn_weight_prob"] = syn_path

    return saved


def _format_list(values: Iterable[Any], *, max_items: int = 8) -> str:
    items = [str(v) for v in values]
    if not items:
        return "—"
    if max_items and len(items) > max_items:
        items = items[:max_items] + ["..."]
    return ", ".join(items)


def _format_counts(counts: Dict[str, Any], *, max_items: int = 8) -> str:
    if not counts:
        return "—"
    items = [f"{k}={v}" for k, v in counts.items()]
    if max_items and len(items) > max_items:
        items = items[:max_items] + ["..."]
    return ", ".join(items)


def _format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple, set)):
        return _format_list(value)
    if isinstance(value, dict):
        return _format_counts(value)
    return str(value)


def _values_equal(a: Any, b: Any, *, tol: float = 1e-6) -> bool:
    if a is None and b is None:
        return True
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    return a == b


def _format_group_field_compare(
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
    *,
    fields: Iterable[str],
    labels: tuple[str, str],
    groups: Optional[Iterable[str]] = None,
    diff_only: bool = True,
    title: Optional[str] = None,
) -> str:
    group_list = sorted(set(summary_a.keys()) | set(summary_b.keys()))
    if groups is not None:
        group_list = [g for g in group_list if g in set(groups)]

    lines = []
    if title:
        lines.append(f"**{title}**")
    lines.extend(
        [
            f"| Group | Field | {labels[0]} | {labels[1]} |",
            "| --- | --- | --- | --- |",
        ]
    )

    row_count = 0
    for group in group_list:
        stats_a = summary_a.get(group, {}) or {}
        stats_b = summary_b.get(group, {}) or {}
        for field in fields:
            va = stats_a.get(field)
            vb = stats_b.get(field)
            if diff_only and _values_equal(va, vb):
                continue
            lines.append(
                f"| {group} | {field} | {_format_value(va)} | {_format_value(vb)} |"
            )
            row_count += 1

    if row_count == 0:
        lines.append("| (no differences found) | — | — | — |")

    return "\n".join(lines)


def format_section_summary_table(
    summary: Dict[str, Any],
    *,
    title: Optional[str] = None,
    groups: Optional[Iterable[str]] = None,
) -> str:
    groups = list(groups) if groups is not None else ["soma", "dend", "apic", "axon", "all"]
    lines = []
    if title:
        lines.append(f"**{title}**")
    lines.extend(
        [
            "| Group | n_sections | n_segments | total_length_um | mean_diam_um | total_area_um2 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for g in groups:
        stats = summary.get(g, {}) or {}
        lines.append(
            f"| {g} | {_format_value(stats.get('n_sections'))} | "
            f"{_format_value(stats.get('n_segments'))} | "
            f"{_format_value(stats.get('total_length_um'))} | "
            f"{_format_value(stats.get('mean_diam_um'))} | "
            f"{_format_value(stats.get('total_area_um2'))} |"
        )
    return "\n".join(lines)


def format_section_summary_compare(
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
    *,
    labels: tuple[str, str] = ("A", "B"),
    groups: Optional[Iterable[str]] = None,
    diff_only: bool = True,
    title: Optional[str] = None,
) -> str:
    fields = [
        "n_sections",
        "n_segments",
        "total_length_um",
        "mean_section_length_um",
        "mean_segment_length_um",
        "total_area_um2",
        "mean_diam_um",
        "min_diam_um",
        "max_diam_um",
    ]
    return _format_group_field_compare(
        summary_a,
        summary_b,
        fields=fields,
        labels=labels,
        groups=groups,
        diff_only=diff_only,
        title=title,
    )


def format_geometry_summary_table(
    summary: Dict[str, Any],
    *,
    title: Optional[str] = None,
    groups: Optional[Iterable[str]] = None,
) -> str:
    groups = list(groups) if groups is not None else ["soma", "proximal", "distal", "all_dend"]
    group_stats = summary.get("groups", {}) or {}
    lines = []
    if title:
        lines.append(f"**{title}**")
    lines.extend(
        [
            "| Group | n_segments | dist_min_um | dist_mean_um | dist_max_um | dist_std_um |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for g in groups:
        stats = group_stats.get(g, {}) or {}
        lines.append(
            f"| {g} | {_format_value(stats.get('n'))} | "
            f"{_format_value(stats.get('min_um'))} | "
            f"{_format_value(stats.get('mean_um'))} | "
            f"{_format_value(stats.get('max_um'))} | "
            f"{_format_value(stats.get('std_um'))} |"
        )
    return "\n".join(lines)


def format_geometry_summary_compare(
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
    *,
    labels: tuple[str, str] = ("A", "B"),
    groups: Optional[Iterable[str]] = None,
    diff_only: bool = True,
    title: Optional[str] = None,
) -> str:
    fields = ["n", "min_um", "mean_um", "max_um", "std_um"]
    group_a = summary_a.get("groups", {}) or {}
    group_b = summary_b.get("groups", {}) or {}
    return _format_group_field_compare(
        group_a,
        group_b,
        fields=fields,
        labels=labels,
        groups=groups,
        diff_only=diff_only,
        title=title,
    )


def format_synapse_summary_table(
    summary: Dict[str, Any],
    *,
    title: Optional[str] = None,
    groups: Optional[Iterable[str]] = None,
    max_sections: int = 6,
) -> str:
    groups_summary = summary.get("groups", {}) or {}
    groups = list(groups) if groups is not None else sorted(groups_summary.keys())
    lines = []
    if title:
        lines.append(f"**{title}**")
    lines.extend(
        [
            "| Group | n_syn | weight_mean | weight_std | dist_mean | dist_std | section_counts |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for g in groups:
        stats = groups_summary.get(g, {}) or {}
        section_counts = stats.get("section_counts", {}) or {}
        lines.append(
            f"| {g} | {_format_value(stats.get('n_syn'))} | "
            f"{_format_value(stats.get('weight_mean'))} | "
            f"{_format_value(stats.get('weight_std'))} | "
            f"{_format_value(stats.get('distance_mean'))} | "
            f"{_format_value(stats.get('distance_std'))} | "
            f"{_format_counts(section_counts, max_items=max_sections)} |"
        )
    return "\n".join(lines)


def format_synapse_summary_compare(
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
    *,
    labels: tuple[str, str] = ("A", "B"),
    groups: Optional[Iterable[str]] = None,
    diff_only: bool = True,
    title: Optional[str] = None,
) -> str:
    fields = [
        "n_syn",
        "weight_mean",
        "weight_std",
        "weight_min",
        "weight_max",
        "distance_mean",
        "distance_std",
        "distance_min",
        "distance_max",
        "section_counts",
        "placement_counts",
        "spikes_per_syn_mean",
        "spikes_per_syn_std",
        "spikes_per_syn_min",
        "spikes_per_syn_max",
        "rate_hz_mean",
        "rate_hz_std",
    ]
    group_a = summary_a.get("groups", {}) or {}
    group_b = summary_b.get("groups", {}) or {}
    return _format_group_field_compare(
        group_a,
        group_b,
        fields=fields,
        labels=labels,
        groups=groups,
        diff_only=diff_only,
        title=title,
    )


def format_mechanism_summary_table(
    summary: Dict[str, Any],
    *,
    title: Optional[str] = None,
    groups: Optional[Iterable[str]] = None,
    max_mechs: int = 8,
) -> str:
    groups = list(groups) if groups is not None else ["soma", "dend", "apic", "axon", "all"]
    lines = []
    if title:
        lines.append(f"**{title}**")
    lines.extend(
        [
            "| Group | density_mechs | point_mechs | ions |",
            "| --- | --- | --- | --- |",
        ]
    )
    for g in groups:
        stats = summary.get(g, {}) or {}
        density = _format_counts(stats.get("density_mechs", {}), max_items=max_mechs)
        points = _format_counts(stats.get("point_mechs", {}), max_items=max_mechs)
        ions = _format_counts(stats.get("ions", {}), max_items=max_mechs)
        lines.append(f"| {g} | {density} | {points} | {ions} |")
    return "\n".join(lines)


def format_mechanism_summary_compare(
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
    *,
    labels: tuple[str, str] = ("A", "B"),
    groups: Optional[Iterable[str]] = None,
    diff_only: bool = True,
    title: Optional[str] = None,
) -> str:
    fields = ["density_mechs", "point_mechs", "ions"]
    return _format_group_field_compare(
        summary_a,
        summary_b,
        fields=fields,
        labels=labels,
        groups=groups,
        diff_only=diff_only,
        title=title,
    )


def run_snapshot(results: Dict[str, Any], *, label: Optional[str] = None) -> Dict[str, Any]:
    """
    Build a compact snapshot of the run for quick comparison and reporting.
    """
    sim_cfg = results.get("sim_cfg", {}) or {}
    meta = results.get("meta", {}) or {}

    syn_config = meta.get("syn_config", {}) or {}
    syn_groups = sorted(syn_config.keys())
    syn_group_counts = {}
    for g in syn_groups:
        gcfg = syn_config.get(g, {}) or {}
        n_syn = gcfg.get("N_syn_resolved")
        if n_syn is None:
            n_syn = gcfg.get("N_syn")
        if n_syn is not None:
            syn_group_counts[g] = int(n_syn)

    syn_records = results.get("syn_records") or {}
    syn_record_counts = {g: len(syn_records.get(g, []) or []) for g in sorted(syn_records.keys())}

    inputs_by_trial = results.get("inputs_by_trial")
    inputs = results.get("inputs")
    inputs_saved_trials = 0
    inputs_saved_groups = set()
    if inputs_by_trial:
        inputs_saved_trials = len(inputs_by_trial)
        for entry in inputs_by_trial:
            groups = (entry.get("inputs") or {}).keys()
            inputs_saved_groups.update(groups)
    elif inputs:
        inputs_saved_trials = 1
        inputs_saved_groups.update(inputs.keys())

    input_summaries = meta.get("input_summaries") or []
    input_summary_trials = len(input_summaries) if isinstance(input_summaries, list) else 0
    input_summary_groups = set()
    for entry in input_summaries:
        input_summary_groups.update((entry.get("groups") or {}).keys())

    input_stats = meta.get("input_stats") or {}
    input_stats_groups = set((input_stats.get("group_means") or {}).keys())

    avg_curve = meta.get("avg_rate_curve") or {}
    avg_curve_bin_ms = avg_curve.get("bin_ms")
    avg_curve_len = len(avg_curve.get("t_ms", []) or [])

    randomness = meta.get("randomness") or {}
    mech_info = meta.get("mechanisms") or {}
    neuron_state = meta.get("neuron_state") or {}
    versions = meta.get("versions") or {}
    env = meta.get("env") or {}
    snap_cfg = meta.get("snapshot") or {}

    return {
        "label": label,
        "mode": results.get("mode"),
        "n_trials": sim_cfg.get("n_trials", meta.get("n_trials")),
        "n_traces_to_save": sim_cfg.get("n_traces_to_save"),
        "n_inputs_to_save": sim_cfg.get("n_inputs_to_save"),
        "tstart_ms": sim_cfg.get("tstart"),
        "tstop_ms": sim_cfg.get("tstop"),
        "dt": sim_cfg.get("dt"),
        "stim_start_ms": sim_cfg.get("stim_start_ms"),
        "stim_duration_ms": sim_cfg.get("stim_duration_ms"),
        "output_format": sim_cfg.get("output_format"),
        "save_full_results": sim_cfg.get("save_full_results"),
        "save_sidecars": sim_cfg.get("save_sidecars"),
        "save_input_stats": sim_cfg.get("save_input_stats"),
        "input_stats_bin_ms": sim_cfg.get("input_stats_bin_ms"),
        "save_syn_records_sidecar": sim_cfg.get("save_syn_records_sidecar"),
        "save_plots": sim_cfg.get("save_plots"),
        "randomness_base_seed_used": randomness.get("base_seed_used"),
        "randomness_trials_setting": randomness.get("trials_setting"),
        "mechanism_dll": mech_info.get("dll_path"),
        "mechanism_sha256": mech_info.get("dll_sha256"),
        "modfiles_hash": mech_info.get("modfiles_sha256"),
        "neuron_state": neuron_state,
        "versions": versions,
        "env": env,
        "snapshot_deterministic": snap_cfg.get("deterministic_applied"),
        "snapshot_seed": snap_cfg.get("deterministic_seed"),
        "syn_groups": syn_groups,
        "syn_group_counts": syn_group_counts,
        "syn_record_counts": syn_record_counts,
        "inputs_saved_trials": inputs_saved_trials,
        "inputs_saved_groups": sorted(inputs_saved_groups),
        "input_summary_trials": input_summary_trials,
        "input_summary_groups": sorted(input_summary_groups),
        "input_stats_groups": sorted(input_stats_groups),
        "avg_rate_curve_bin_ms": avg_curve_bin_ms,
        "avg_rate_curve_len": avg_curve_len,
    }


def _truncate_text(text: Any, max_len: int) -> str:
    if text is None:
        return ""
    s = str(text)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _summarize_array(arr: Any) -> str:
    try:
        a = np.asarray(arr)
    except Exception:
        return "array(?)"
    shape = a.shape
    dtype = str(a.dtype)
    if a.size == 0:
        return f"array(shape={shape}, dtype={dtype}, empty)"
    if np.issubdtype(a.dtype, np.number):
        try:
            af = a.astype(float)
            return (
                f"array(shape={shape}, dtype={dtype}, min={np.nanmin(af):.6g}, "
                f"max={np.nanmax(af):.6g}, mean={np.nanmean(af):.6g}, std={np.nanstd(af):.6g})"
            )
        except Exception:
            pass
    return f"array(shape={shape}, dtype={dtype})"


def _summarize_list(values: list, *, max_items: int, max_str: int) -> str:
    n = len(values)
    if n == 0:
        return "list(len=0)"
    if n <= max_items and all(not isinstance(v, (dict, list, tuple, np.ndarray)) for v in values):
        return _truncate_text(values, max_str)
    sample = values[:max_items]
    return f"list(len={n}, sample={_truncate_text(sample, max_str)})"


def _summarize_value(val: Any, *, max_list_items: int, max_str: int) -> str:
    if isinstance(val, np.ndarray):
        return _summarize_array(val)
    if isinstance(val, (list, tuple)):
        return _summarize_list(list(val), max_items=max_list_items, max_str=max_str)
    if isinstance(val, dict):
        return f"dict(len={len(val)})"
    if isinstance(val, (float, int, bool)) or val is None:
        return _truncate_text(val, max_str)
    return _truncate_text(val, max_str)


def _flatten_for_compare(
    obj: Any,
    prefix: str,
    out: Dict[str, str],
    *,
    max_depth: int,
    max_list_items: int,
    max_dict_items: int,
    max_str: int,
) -> None:
    if max_depth <= 0:
        out[prefix] = _summarize_value(obj, max_list_items=max_list_items, max_str=max_str)
        return
    if isinstance(obj, dict):
        if len(obj) > max_dict_items:
            out[prefix] = f"dict(len={len(obj)})"
            return
        for key in sorted(obj.keys(), key=lambda k: str(k)):
            _flatten_for_compare(
                obj[key],
                f"{prefix}.{key}",
                out,
                max_depth=max_depth - 1,
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
                max_str=max_str,
            )
        return
    if isinstance(obj, (list, tuple)):
        if len(obj) > max_list_items:
            out[prefix] = _summarize_list(list(obj), max_items=max_list_items, max_str=max_str)
            return
        for idx, item in enumerate(obj):
            _flatten_for_compare(
                item,
                f"{prefix}[{idx}]",
                out,
                max_depth=max_depth - 1,
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
                max_str=max_str,
            )
        return
    out[prefix] = _summarize_value(obj, max_list_items=max_list_items, max_str=max_str)


def build_compare_table(
    run_a: Union[str, Path, Dict[str, Any]],
    run_b: Union[str, Path, Dict[str, Any]],
    *,
    scope: str = "full",
    max_depth: int = 6,
    max_list_items: int = 20,
    max_dict_items: int = 200,
    max_str: int = 160,
) -> list[dict[str, str]]:
    """
    Build a flattened side-by-side comparison table.

    scope:
      - "snapshot": compare run_snapshot(...) output
      - "meta": compare results["meta"]
      - "full": compare full results dict
    """
    res_a = _load_results_any(run_a)
    res_b = _load_results_any(run_b)

    if scope == "snapshot":
        obj_a = run_snapshot(res_a, label="A")
        obj_b = run_snapshot(res_b, label="B")
        prefix = "results.snapshot"
    elif scope == "meta":
        obj_a = res_a.get("meta", {}) or {}
        obj_b = res_b.get("meta", {}) or {}
        prefix = "results.meta"
    else:
        obj_a = res_a
        obj_b = res_b
        prefix = "results"

    flat_a: Dict[str, str] = {}
    flat_b: Dict[str, str] = {}
    _flatten_for_compare(
        obj_a,
        prefix,
        flat_a,
        max_depth=max_depth,
        max_list_items=max_list_items,
        max_dict_items=max_dict_items,
        max_str=max_str,
    )
    _flatten_for_compare(
        obj_b,
        prefix,
        flat_b,
        max_depth=max_depth,
        max_list_items=max_list_items,
        max_dict_items=max_dict_items,
        max_str=max_str,
    )

    keys = sorted(set(flat_a.keys()) | set(flat_b.keys()))
    rows: list[dict[str, str]] = []
    for key in keys:
        a_val = flat_a.get(key, "")
        b_val = flat_b.get(key, "")
        rows.append(
            {
                "path": key,
                "a": a_val,
                "b": b_val,
                "equal": str(a_val == b_val),
            }
        )
    return rows


def save_compare_table(
    run_a: Union[str, Path, Dict[str, Any]],
    run_b: Union[str, Path, Dict[str, Any]],
    out_path: Union[str, Path],
    *,
    scope: str = "full",
    fmt: str = "csv",
    max_depth: int = 6,
    max_list_items: int = 20,
    max_dict_items: int = 200,
    max_str: int = 160,
) -> Dict[str, Path]:
    """
    Save a comparison table to CSV (and optionally XLSX if pandas is available).
    """
    rows = build_compare_table(
        run_a,
        run_b,
        scope=scope,
        max_depth=max_depth,
        max_list_items=max_list_items,
        max_dict_items=max_dict_items,
        max_str=max_str,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}

    fmt = fmt.lower()
    if fmt in ("csv", "both"):
        csv_path = out_path if out_path.suffix.lower() == ".csv" else out_path.with_suffix(".csv")
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "a", "b", "equal"])
            writer.writeheader()
            writer.writerows(rows)
        outputs["csv"] = csv_path

    if fmt in ("xlsx", "excel", "both"):
        try:
            import pandas as pd
        except Exception:
            return outputs
        xlsx_path = out_path if out_path.suffix.lower() in (".xlsx", ".xls") else out_path.with_suffix(".xlsx")
        pd.DataFrame(rows).to_excel(xlsx_path, index=False)
        outputs["xlsx"] = xlsx_path

    return outputs


def format_snapshot_table(
    snapshot: Dict[str, Any],
    *,
    title: Optional[str] = None,
    max_groups: int = 8,
) -> str:
    """
    Return a markdown table of a run snapshot.
    """
    rows = [
        ("mode", snapshot.get("mode")),
        ("n_trials", snapshot.get("n_trials")),
        ("n_traces_to_save", snapshot.get("n_traces_to_save")),
        ("n_inputs_to_save", snapshot.get("n_inputs_to_save")),
        ("inputs_saved_trials", snapshot.get("inputs_saved_trials")),
        ("input_summary_trials", snapshot.get("input_summary_trials")),
        ("tstart_ms", snapshot.get("tstart_ms")),
        ("tstop_ms", snapshot.get("tstop_ms")),
        ("dt", snapshot.get("dt")),
        ("stim_start_ms", snapshot.get("stim_start_ms")),
        ("stim_duration_ms", snapshot.get("stim_duration_ms")),
        ("output_format", snapshot.get("output_format")),
        ("save_full_results", snapshot.get("save_full_results")),
        ("save_sidecars", snapshot.get("save_sidecars")),
        ("save_input_stats", snapshot.get("save_input_stats")),
        ("input_stats_bin_ms", snapshot.get("input_stats_bin_ms")),
        ("save_syn_records_sidecar", snapshot.get("save_syn_records_sidecar")),
        ("save_plots", snapshot.get("save_plots")),
        ("randomness_base_seed_used", snapshot.get("randomness_base_seed_used")),
        ("randomness_trials_setting", snapshot.get("randomness_trials_setting")),
        ("syn_groups", _format_list(snapshot.get("syn_groups", []), max_items=max_groups)),
        ("syn_group_counts", _format_counts(snapshot.get("syn_group_counts", {}), max_items=max_groups)),
        ("syn_record_counts", _format_counts(snapshot.get("syn_record_counts", {}), max_items=max_groups)),
        ("inputs_saved_groups", _format_list(snapshot.get("inputs_saved_groups", []), max_items=max_groups)),
        ("input_summary_groups", _format_list(snapshot.get("input_summary_groups", []), max_items=max_groups)),
        ("input_stats_groups", _format_list(snapshot.get("input_stats_groups", []), max_items=max_groups)),
        ("avg_rate_curve_bin_ms", snapshot.get("avg_rate_curve_bin_ms")),
        ("avg_rate_curve_len", snapshot.get("avg_rate_curve_len")),
    ]

    lines = []
    if title:
        lines.append(f"**{title}**")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    for k, v in rows:
        lines.append(f"| {k} | {_format_value(v)} |")
    return "\n".join(lines)


def format_snapshot_diff(
    snapshot_a: Dict[str, Any],
    snapshot_b: Dict[str, Any],
    *,
    labels: tuple[str, str] = ("A", "B"),
    max_groups: int = 8,
) -> str:
    """
    Return a markdown table listing differences between two snapshots.
    """
    keys = [
        "mode",
        "n_trials",
        "n_traces_to_save",
        "n_inputs_to_save",
        "inputs_saved_trials",
        "input_summary_trials",
        "tstart_ms",
        "tstop_ms",
        "dt",
        "stim_start_ms",
        "stim_duration_ms",
        "output_format",
        "save_full_results",
        "save_sidecars",
        "save_input_stats",
        "input_stats_bin_ms",
        "save_syn_records_sidecar",
        "save_plots",
        "randomness_base_seed_used",
        "randomness_trials_setting",
        "mechanism_dll",
        "mechanism_sha256",
        "modfiles_hash",
        "neuron_state",
        "versions",
        "env",
        "snapshot_deterministic",
        "snapshot_seed",
        "syn_groups",
        "syn_group_counts",
        "syn_record_counts",
        "inputs_saved_groups",
        "input_summary_groups",
        "input_stats_groups",
        "avg_rate_curve_bin_ms",
        "avg_rate_curve_len",
    ]

    def _fmt(key: str, snap: Dict[str, Any]) -> str:
        val = snap.get(key)
        if key in ("syn_groups", "inputs_saved_groups", "input_summary_groups", "input_stats_groups"):
            return _format_list(val or [], max_items=max_groups)
        if key in ("syn_group_counts", "syn_record_counts"):
            return _format_counts(val or {}, max_items=max_groups)
        return _format_value(val)

    lines = [
        f"**Snapshot diff ({labels[0]} vs {labels[1]})**",
        f"| Field | {labels[0]} | {labels[1]} |",
        "| --- | --- | --- |",
    ]
    for key in keys:
        va = snapshot_a.get(key)
        vb = snapshot_b.get(key)
        if va == vb:
            continue
        lines.append(f"| {key} | {_fmt(key, snapshot_a)} | {_fmt(key, snapshot_b)} |")

    if len(lines) == 3:
        lines.append("| (no differences found) | — | — |")

    return "\n".join(lines)


def format_snapshot_compare(
    snapshot_a: Dict[str, Any],
    snapshot_b: Dict[str, Any],
    *,
    labels: tuple[str, str] = ("A", "B"),
    max_groups: int = 8,
) -> str:
    """
    Return a markdown table with full side-by-side snapshot values.
    """
    keys = [
        "mode",
        "n_trials",
        "n_traces_to_save",
        "n_inputs_to_save",
        "inputs_saved_trials",
        "input_summary_trials",
        "tstart_ms",
        "tstop_ms",
        "dt",
        "stim_start_ms",
        "stim_duration_ms",
        "output_format",
        "save_full_results",
        "save_sidecars",
        "save_input_stats",
        "input_stats_bin_ms",
        "save_syn_records_sidecar",
        "save_plots",
        "randomness_base_seed_used",
        "randomness_trials_setting",
        "mechanism_dll",
        "mechanism_sha256",
        "modfiles_hash",
        "neuron_state",
        "versions",
        "env",
        "snapshot_deterministic",
        "snapshot_seed",
        "syn_groups",
        "syn_group_counts",
        "syn_record_counts",
        "inputs_saved_groups",
        "input_summary_groups",
        "input_stats_groups",
        "avg_rate_curve_bin_ms",
        "avg_rate_curve_len",
    ]

    def _fmt(key: str, snap: Dict[str, Any]) -> str:
        val = snap.get(key)
        if key in ("syn_groups", "inputs_saved_groups", "input_summary_groups", "input_stats_groups"):
            return _format_list(val or [], max_items=max_groups)
        if key in ("syn_group_counts", "syn_record_counts"):
            return _format_counts(val or {}, max_items=max_groups)
        return _format_value(val)

    lines = [
        f"**Snapshot compare ({labels[0]} vs {labels[1]})**",
        f"| Field | {labels[0]} | {labels[1]} |",
        "| --- | --- | --- |",
    ]
    for key in keys:
        lines.append(f"| {key} | {_fmt(key, snapshot_a)} | {_fmt(key, snapshot_b)} |")

    return "\n".join(lines)


def _load_results_any(run_or_results: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(run_or_results, dict):
        return run_or_results
    from modules_local import run_sim

    return run_sim.load_results(run_or_results)


def _read_manifest_files(run_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    p = Path(run_path)
    if p.is_dir():
        manifest = p / "run_manifest.json"
        if not manifest.is_file():
            manifest = p / "results" / "run_manifest.json"
    elif p.name == "run_manifest.json":
        manifest = p
    else:
        return None
    if not manifest.is_file():
        return None
    try:
        return json.loads(manifest.read_text()).get("files", {})
    except Exception:
        return None


def _diff_values(
    a: Any,
    b: Any,
    *,
    path: str,
    diffs: list[str],
    max_diffs: int,
    rtol: float,
    atol: float,
) -> None:
    if len(diffs) >= max_diffs:
        return

    # Handle numpy arrays (or array-like)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        aa = np.asarray(a, dtype=object)
        bb = np.asarray(b, dtype=object)
        if aa.shape != bb.shape:
            diffs.append(f"{path}: shape {aa.shape} vs {bb.shape}")
            return
        if aa.size == 0 and bb.size == 0:
            return
        try:
            if np.allclose(aa.astype(float), bb.astype(float), rtol=rtol, atol=atol):
                return
            max_abs = float(np.max(np.abs(aa.astype(float) - bb.astype(float))))
            diffs.append(f"{path}: arrays differ (max_abs_diff={max_abs:.6g})")
        except Exception:
            if not np.array_equal(aa, bb):
                diffs.append(f"{path}: arrays differ")
        return

    # Dicts
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a.keys()) | set(b.keys()), key=lambda k: str(k))
        for k in keys:
            if len(diffs) >= max_diffs:
                return
            if k not in a:
                diffs.append(f"{path}.{k}: only in B")
                continue
            if k not in b:
                diffs.append(f"{path}.{k}: only in A")
                continue
            _diff_values(
                a[k],
                b[k],
                path=f"{path}.{k}",
                diffs=diffs,
                max_diffs=max_diffs,
                rtol=rtol,
                atol=atol,
            )
        return

    # Lists / tuples
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} vs {len(b)}")
            return
        for idx, (va, vb) in enumerate(zip(a, b)):
            if len(diffs) >= max_diffs:
                return
            _diff_values(
                va,
                vb,
                path=f"{path}[{idx}]",
                diffs=diffs,
                max_diffs=max_diffs,
                rtol=rtol,
                atol=atol,
            )
        return

    # Scalars / fallback
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol):
            diffs.append(f"{path}: {a!r} != {b!r}")
        return
    if a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")


def compare_snapshot_runs(
    run_a: Union[str, Path, Dict[str, Any]],
    run_b: Union[str, Path, Dict[str, Any]],
    *,
    labels: tuple[str, str] = ("A", "B"),
    max_diffs: int = 200,
    rtol: float = 0.0,
    atol: float = 0.0,
    print_summary: bool = True,
) -> Dict[str, Any]:
    """
    Deep-compare two snapshot runs (or results dicts) and report differences.

    Returns:
      {
        "equal": bool,
        "n_diffs": int,
        "diffs": [str, ...],
        "snapshot_a": {...},
        "snapshot_b": {...},
        "snapshot_diff_table": str,
        "manifest_files_a": {...} | None,
        "manifest_files_b": {...} | None,
        "manifest_diff": [str, ...],
      }
    """
    res_a = _load_results_any(run_a)
    res_b = _load_results_any(run_b)

    diffs: list[str] = []
    _diff_values(
        res_a,
        res_b,
        path="results",
        diffs=diffs,
        max_diffs=max_diffs,
        rtol=rtol,
        atol=atol,
    )

    snap_a = run_snapshot(res_a, label=labels[0])
    snap_b = run_snapshot(res_b, label=labels[1])
    snap_diff = format_snapshot_diff(snap_a, snap_b, labels=labels)

    files_a = _read_manifest_files(run_a) if not isinstance(run_a, dict) else None
    files_b = _read_manifest_files(run_b) if not isinstance(run_b, dict) else None
    manifest_diff: list[str] = []
    if files_a is not None or files_b is not None:
        files_a = files_a or {}
        files_b = files_b or {}
        keys = sorted(set(files_a.keys()) | set(files_b.keys()))
        for key in keys:
            va = files_a.get(key)
            vb = files_b.get(key)
            if va != vb:
                manifest_diff.append(f"{key}: {va!r} vs {vb!r}")

    if print_summary:
        print(f"Snapshot compare: equal={len(diffs)==0}, n_diffs={len(diffs)}")
        if diffs:
            print("First differences:")
            for line in diffs[: min(10, len(diffs))]:
                print("-", line)

    return {
        "equal": len(diffs) == 0,
        "n_diffs": len(diffs),
        "diffs": diffs,
        "snapshot_a": snap_a,
        "snapshot_b": snap_b,
        "snapshot_diff_table": snap_diff,
        "manifest_files_a": files_a,
        "manifest_files_b": files_b,
        "manifest_diff": manifest_diff,
    }
