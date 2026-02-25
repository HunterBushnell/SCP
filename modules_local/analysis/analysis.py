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
import os
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


def _coerce_spike_trials(spikes: Any) -> list[np.ndarray]:
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


def _resolve_spikes_npz_path(path: Union[str, Path]) -> Path:
    p = Path(path).expanduser().resolve()
    if p.is_file():
        return p
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")

    candidates = [
        p / "spikes.npz",
        p / "results" / "spikes.npz",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Could not find spikes.npz under: {p}")


def _format_float_series(values: np.ndarray, *, precision: int, delimiter: str) -> str:
    if values.size == 0:
        return ""
    fmt = f"{{:.{int(max(1, precision))}g}}"
    return delimiter.join(fmt.format(float(v)) for v in values)


def export_spikes_trials_csv(
    spikes_npz_or_run: Union[str, Path],
    out_csv: Optional[Union[str, Path]] = None,
    *,
    delimiter: str = "|",
    precision: int = 10,
    overwrite: bool = False,
    trial_prefix: str = "trial_",
) -> Path:
    """
    Export spikes as one CSV row per trial.

    Output columns:
      - trial_n: trial label (e.g. trial_0)
      - n_spikes: number of spikes in the trial
      - spike_times_ms: delimiter-separated spike times in ms
    """
    if not delimiter:
        raise ValueError("delimiter must be a non-empty string")

    spikes_path = _resolve_spikes_npz_path(spikes_npz_or_run)
    with np.load(spikes_path, allow_pickle=True) as data:
        if "spikes" in data.files:
            spikes_obj = data["spikes"]
        elif len(data.files) == 1:
            spikes_obj = data[data.files[0]]
        else:
            raise ValueError(
                f"Could not determine spikes array in {spikes_path}; keys={list(data.files)}"
            )

    trials = _coerce_spike_trials(spikes_obj)

    if out_csv is None:
        out_path = spikes_path.with_name(f"{spikes_path.stem}_trials.csv")
    else:
        out_path = Path(out_csv).expanduser()
        if not out_path.suffix:
            out_path = out_path.with_suffix(".csv")
        if not out_path.is_absolute():
            out_path = (Path.cwd() / out_path).resolve()
        else:
            out_path = out_path.resolve()

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output CSV already exists: {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_n", "n_spikes", "spike_times_ms"])
        for idx, tr in enumerate(trials):
            arr = np.asarray(tr, dtype=float).ravel()
            arr = arr[np.isfinite(arr)]
            writer.writerow(
                [
                    f"{trial_prefix}{idx}",
                    int(arr.size),
                    _format_float_series(arr, precision=precision, delimiter=delimiter),
                ]
            )
    return out_path


def _moving_average(values: np.ndarray, win_bins: int) -> np.ndarray:
    if win_bins <= 1:
        return values
    kernel = np.ones(int(win_bins), dtype=float) / float(win_bins)
    return np.convolve(values, kernel, mode="same")


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


def _select_window_mask(
    t_ms: np.ndarray,
    start_ms: Optional[float],
    stop_ms: Optional[float],
) -> np.ndarray:
    mask = np.ones_like(t_ms, dtype=bool)
    if start_ms is not None:
        mask &= t_ms >= float(start_ms)
    if stop_ms is not None:
        mask &= t_ms < float(stop_ms)
    return mask


def group_colors_from_syn_config(syn_config: Dict[str, Any]) -> Dict[str, str]:
    colors: Dict[str, str] = {}
    for group, cfg in (syn_config or {}).items():
        color = (cfg or {}).get("color")
        if color:
            colors[str(group)] = str(color)
    return colors


def group_colors_from_results(results: Dict[str, Any]) -> Dict[str, str]:
    meta = results.get("meta") or {}
    return group_colors_from_syn_config(meta.get("syn_config") or {})


def merge_group_colors(*results: Dict[str, Any]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for res in results:
        if not res:
            continue
        merged.update(group_colors_from_results(res))
    return merged


def normalize_output_curve(
    curve: Dict[str, Any],
    sim_cfg: Dict[str, Any],
    *,
    mode: str = "raw",
    norm_mode: str = "avg",
    baseline_ms: float = 100.0,
    baseline_mode: str = "window",
    baseline_center_ms: Optional[float] = None,
    norm_window: str = "stim",
) -> Dict[str, Any]:
    """
    Normalize an avg_rate_curve with baseline subtraction + avg/peak scaling.
    """
    mode = (mode or "raw").lower()
    norm_mode = (norm_mode or "avg").lower()
    norm_window = (norm_window or "stim").lower()
    baseline_mode = (baseline_mode or "window").lower()

    t_ms = np.asarray(curve.get("t_ms", []) or [], dtype=float)
    rate = np.asarray(curve.get("rate_hz", []) or [], dtype=float)

    out = dict(curve)
    out["t_ms"] = t_ms.tolist()
    out["rate_hz"] = rate.tolist()
    out["normalized"] = False
    out["norm_mode"] = None
    out["norm_window"] = None
    out["baseline_ms"] = None
    out["baseline_mode"] = None
    out["baseline_time_ms"] = None
    out["baseline_center_ms"] = None
    out["baseline_mean"] = None
    out["baseline_subtracted"] = False
    out["rate_hz_baseline_sub"] = None
    out["norm_scale"] = None
    out["units"] = "Hz"

    if mode == "raw":
        return out

    tstart = sim_cfg.get("tstart")
    tstop = sim_cfg.get("tstop")
    stim_start, stim_stop = _resolve_stim_window(sim_cfg)

    baseline_mean = 0.0
    baseline_time = None
    baseline_center_offset = None
    if stim_start is not None:
        baseline_center_offset = float(baseline_center_ms) if baseline_center_ms is not None else float(baseline_ms) * 0.5
        baseline_time = float(stim_start) - baseline_center_offset
        if baseline_mode == "point":
            if t_ms.size:
                baseline_mean = float(
                    np.interp(baseline_time, t_ms, rate, left=rate[0], right=rate[-1])
                )
        else:
            half = float(baseline_ms) * 0.5
            baseline_start = baseline_time - half
            baseline_stop = baseline_time + half
            baseline_mask = _select_window_mask(t_ms, baseline_start, baseline_stop)
            baseline_mean = float(np.mean(rate[baseline_mask])) if baseline_mask.any() else 0.0

    rate_bs = rate - baseline_mean

    if norm_window == "full":
        window_mask = _select_window_mask(t_ms, tstart, tstop)
    else:
        window_mask = _select_window_mask(t_ms, stim_start, stim_stop)
    if not window_mask.any():
        window_mask = np.ones_like(t_ms, dtype=bool)

    if norm_mode == "peak":
        norm_scale = float(np.max(rate_bs[window_mask])) if rate_bs.size else 0.0
    else:
        norm_scale = float(np.mean(rate_bs[window_mask])) if rate_bs.size else 0.0

    if norm_scale == 0.0:
        norm_scale = 1.0

    rate_norm = rate_bs / norm_scale

    out["rate_hz"] = rate_norm.tolist()
    out["normalized"] = True
    out["norm_mode"] = norm_mode
    out["norm_window"] = norm_window
    out["baseline_ms"] = float(baseline_ms)
    out["baseline_mode"] = baseline_mode
    out["baseline_time_ms"] = baseline_time
    out["baseline_center_ms"] = baseline_center_offset
    out["baseline_mean"] = baseline_mean
    out["baseline_subtracted"] = True
    out["rate_hz_baseline_sub"] = rate_bs.tolist()
    out["norm_scale"] = norm_scale
    out["units"] = "normalized"
    return out


def _smooth_rate_curve(
    centers: np.ndarray,
    rates: np.ndarray,
    bin_ms: float,
    smooth_ms: Optional[float],
    *,
    mode: str = "center",
) -> tuple[np.ndarray, np.ndarray]:
    if smooth_ms is None:
        return centers, rates
    try:
        smooth_ms = float(smooth_ms)
    except Exception:
        return centers, rates
    if smooth_ms <= 0 or bin_ms <= 0:
        return centers, rates

    k = int(round(smooth_ms / bin_ms))
    if k <= 1 or rates.size < k:
        return centers, rates
    if mode == "center" and k % 2 == 0:
        k += 1

    kernel = np.ones(k, dtype=float) / float(k)
    if mode == "center":
        y = np.convolve(rates, kernel, mode="valid")
        drop = (len(centers) - len(y)) // 2
        if drop < 0:
            return centers, rates
        return centers[drop : drop + len(y)], y
    if mode == "causal":
        pad = (k - 1, 0)
        y = np.convolve(np.pad(rates, pad), kernel, mode="valid")
        return centers[: len(y)], y
    return centers, rates


def _smooth_curve_nanmean(
    centers: np.ndarray,
    values: np.ndarray,
    bin_ms: float,
    smooth_ms: Optional[float],
    *,
    mode: str = "center",
) -> tuple[np.ndarray, np.ndarray]:
    if smooth_ms is None:
        return centers, values
    try:
        smooth_ms = float(smooth_ms)
    except Exception:
        return centers, values
    if smooth_ms <= 0 or bin_ms <= 0:
        return centers, values

    k = int(round(smooth_ms / bin_ms))
    if k <= 1 or values.size < k:
        return centers, values
    if mode == "center" and k % 2 == 0:
        k += 1

    valid = np.isfinite(values)
    numer = np.where(valid, values, 0.0)
    denom = valid.astype(float)
    kernel = np.ones(k, dtype=float)

    if mode == "center":
        numer_sm = np.convolve(numer, kernel, mode="valid")
        denom_sm = np.convolve(denom, kernel, mode="valid")
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(denom_sm > 0, numer_sm / denom_sm, np.nan)
        drop = (len(centers) - len(out)) // 2
        if drop < 0:
            return centers, values
        return centers[drop : drop + len(out)], out

    if mode == "causal":
        pad = (k - 1, 0)
        numer_sm = np.convolve(np.pad(numer, pad), kernel, mode="valid")
        denom_sm = np.convolve(np.pad(denom, pad), kernel, mode="valid")
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(denom_sm > 0, numer_sm / denom_sm, np.nan)
        return centers[: len(out)], out

    return centers, values


def compute_output_curve_from_results(
    results: Dict[str, Any],
    *,
    bin_ms: Optional[float] = None,
    smooth_ms: Optional[float] = None,
    smooth_mode: str = "causal",
) -> Optional[Dict[str, Any]]:
    sim_cfg = results.get("sim_cfg", {}) or {}
    tstart = float(sim_cfg.get("tstart", 0.0))
    tstop = sim_cfg.get("tstop")
    if tstop is None:
        traces = results.get("traces", {}) or {}
        T = traces.get("T")
        if T is not None and len(T) > 0:
            tstop = float(T[-1])
    if tstop is None or tstop <= tstart:
        return None

    bin_width = float(bin_ms if bin_ms is not None else sim_cfg.get("bins", 25.0))
    if bin_width <= 0:
        return None

    bins = np.arange(tstart, tstop + bin_width, bin_width, dtype=float)
    if bins.size < 2:
        return None
    centers = bins[:-1] + 0.5 * bin_width
    bw_s = bin_width / 1000.0

    spikes = results.get("spikes")
    if spikes is None:
        return None

    if results.get("mode") == "multi":
        if isinstance(spikes, np.ndarray):
            spikes_by_trial = list(spikes.tolist())
        elif isinstance(spikes, (list, tuple)):
            spikes_by_trial = list(spikes)
        else:
            spikes_by_trial = [spikes]
    else:
        spikes_by_trial = [spikes]

    if not spikes_by_trial:
        mean_rate = np.array([], dtype=float)
    else:
        rates = []
        for tr in spikes_by_trial:
            tr = np.asarray(tr, dtype=float)
            counts, _ = np.histogram(tr, bins=bins)
            rates.append(counts / bw_s)
        mean_rate = np.mean(np.vstack(rates), axis=0) if rates else np.array([], dtype=float)

    smooth_mode = str(smooth_mode or "center").lower()
    centers, mean_rate = _smooth_rate_curve(
        centers,
        mean_rate,
        bin_width,
        smooth_ms,
        mode=smooth_mode,
    )

    try:
        smooth_ms_val = float(smooth_ms) if smooth_ms is not None else 0.0
    except Exception:
        smooth_ms_val = 0.0

    return {
        "bin_ms": bin_width,
        "smooth_ms": smooth_ms_val,
        "smooth_mode": smooth_mode,
        "t_ms": centers.tolist(),
        "rate_hz": mean_rate.tolist(),
    }


def compute_output_isi_curve_from_results(
    results: Dict[str, Any],
    *,
    bin_ms: Optional[float] = None,
    smooth_ms: Optional[float] = None,
    smooth_mode: str = "causal",
) -> Optional[Dict[str, Any]]:
    sim_cfg = results.get("sim_cfg", {}) or {}
    tstart = float(sim_cfg.get("tstart", 0.0))
    tstop = sim_cfg.get("tstop")
    if tstop is None:
        traces = results.get("traces", {}) or {}
        T = traces.get("T")
        if T is not None and len(T) > 0:
            tstop = float(T[-1])
    if tstop is None or tstop <= tstart:
        return None

    bin_width = float(bin_ms if bin_ms is not None else sim_cfg.get("bins", 25.0))
    if bin_width <= 0:
        return None

    bins = np.arange(tstart, tstop + bin_width, bin_width, dtype=float)
    if bins.size < 2:
        return None
    centers = bins[:-1] + 0.5 * bin_width
    n_bins = centers.size

    spikes = results.get("spikes")
    if spikes is None:
        return None

    if results.get("mode") == "multi":
        if isinstance(spikes, np.ndarray):
            spikes_by_trial = list(spikes.tolist())
        elif isinstance(spikes, (list, tuple)):
            spikes_by_trial = list(spikes)
        else:
            spikes_by_trial = [spikes]
    else:
        spikes_by_trial = [spikes]

    trial_isis: list[np.ndarray] = []
    for tr in spikes_by_trial:
        sp = np.asarray(tr, dtype=float)
        if sp.size < 2:
            continue
        sp = sp[np.isfinite(sp)]
        if sp.size < 2:
            continue
        sp = np.sort(sp)
        sp = sp[(sp >= tstart) & (sp < tstop)]
        if sp.size < 2:
            continue

        isi = np.diff(sp)
        if isi.size == 0:
            continue
        t_mid = sp[:-1] + 0.5 * isi
        weighted_sum, _ = np.histogram(t_mid, bins=bins, weights=isi)
        counts, _ = np.histogram(t_mid, bins=bins)
        trial_curve = np.full(n_bins, np.nan, dtype=float)
        valid = counts > 0
        if np.any(valid):
            trial_curve[valid] = weighted_sum[valid] / counts[valid]
            trial_isis.append(trial_curve)

    if not trial_isis:
        return None

    mat = np.vstack(trial_isis)
    valid_mask = np.isfinite(mat)
    mean_isi = np.full(mat.shape[1], np.nan, dtype=float)
    counts = np.sum(valid_mask, axis=0)
    has_vals = counts > 0
    if np.any(has_vals):
        numer = np.sum(np.where(valid_mask, mat, 0.0), axis=0)
        mean_isi[has_vals] = numer[has_vals] / counts[has_vals]
    if not np.isfinite(mean_isi).any():
        return None

    smooth_mode = str(smooth_mode or "center").lower()
    centers, mean_isi = _smooth_curve_nanmean(
        centers,
        mean_isi,
        bin_width,
        smooth_ms,
        mode=smooth_mode,
    )

    try:
        smooth_ms_val = float(smooth_ms) if smooth_ms is not None else 0.0
    except Exception:
        smooth_ms_val = 0.0

    return {
        "bin_ms": bin_width,
        "smooth_ms": smooth_ms_val,
        "smooth_mode": smooth_mode,
        "t_ms": centers.tolist(),
        "isi_ms": mean_isi.tolist(),
        "units": "ms",
        "n_trials_with_isi": int(len(trial_isis)),
    }


def load_scatter_curve_optional(
    *,
    enabled: bool,
    path: str,
    time_unit: str = "s",
    bin_ms: Optional[float] = None,
    smooth_ms: Optional[float] = None,
    smooth_mode: str = "center",
    x_col: int = 0,
    y_col: int = 1,
    shift_ms: Optional[float] = None,
    quiet: bool = False,
) -> Optional[Dict[str, Any]]:
    if not enabled:
        return None
    if not path:
        if not quiet:
            print("Scatter curve enabled but path is empty.")
        return None
    try:
        import pandas as pd

        df = pd.read_csv(path, header=None)
        x_raw = pd.to_numeric(df.iloc[:, x_col], errors="coerce")
        y_raw = pd.to_numeric(df.iloc[:, y_col], errors="coerce")
        mask = x_raw.notna() & y_raw.notna()
        x_vals = x_raw[mask].to_numpy(dtype=float)
        y_vals = y_raw[mask].to_numpy(dtype=float)
    except Exception:
        try:
            data = np.genfromtxt(path, delimiter=",")
            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError("scatter CSV must have at least two columns")
            x_vals = data[:, x_col].astype(float)
            y_vals = data[:, y_col].astype(float)
            mask = np.isfinite(x_vals) & np.isfinite(y_vals)
            x_vals = x_vals[mask]
            y_vals = y_vals[mask]
        except Exception as exc:
            if not quiet:
                print("Scatter curve load failed:", exc)
            return None

    if x_vals.size == 0:
        if not quiet:
            print("Scatter curve is empty after parsing.")
        return None

    time_unit = (time_unit or "s").strip().lower()
    if time_unit == "s":
        t_ms = x_vals * 1000.0
    elif time_unit == "ms":
        t_ms = x_vals
    else:
        if not quiet:
            print(f"Scatter time_unit must be 's' or 'ms' (got {time_unit!r}).")
        return None

    if shift_ms is not None:
        t_ms = t_ms + float(shift_ms)

    order = np.argsort(t_ms)
    t_ms = t_ms[order]
    y_vals = y_vals[order]

    if bin_ms is None or float(bin_ms) <= 0:
        return {
            "bin_ms": None,
            "smooth_ms": float(smooth_ms) if smooth_ms else 0.0,
            "smooth_mode": str(smooth_mode),
            "t_ms": t_ms.tolist(),
            "rate_hz": y_vals.tolist(),
        }

    bin_ms = float(bin_ms)
    t_min = float(np.min(t_ms))
    t_max = float(np.max(t_ms))
    edges = np.arange(t_min, t_max + bin_ms, bin_ms, dtype=float)
    if edges.size < 2:
        return None
    centers = edges[:-1] + 0.5 * bin_ms

    bin_idx = np.digitize(t_ms, edges) - 1
    bin_idx = bin_idx[(bin_idx >= 0) & (bin_idx < len(centers))]
    if bin_idx.size == 0:
        return None
    sums = np.bincount(bin_idx, weights=y_vals, minlength=len(centers))
    counts = np.bincount(bin_idx, minlength=len(centers))
    mean = np.full_like(centers, np.nan, dtype=float)
    nonzero = counts > 0
    mean[nonzero] = sums[nonzero] / counts[nonzero]

    if np.isnan(mean).any():
        valid = ~np.isnan(mean)
        if valid.sum() >= 2:
            mean = np.interp(centers, centers[valid], mean[valid])
        elif valid.sum() == 1:
            mean = np.full_like(mean, mean[valid][0], dtype=float)
        else:
            return None

    smooth_mode = str(smooth_mode or "center").lower()
    centers, mean = _smooth_rate_curve(
        centers,
        mean,
        bin_ms,
        smooth_ms,
        mode=smooth_mode,
    )

    try:
        smooth_ms_val = float(smooth_ms) if smooth_ms is not None else 0.0
    except Exception:
        smooth_ms_val = 0.0

    return {
        "bin_ms": bin_ms,
        "smooth_ms": smooth_ms_val,
        "smooth_mode": smooth_mode,
        "t_ms": centers.tolist(),
        "rate_hz": mean.tolist(),
    }


def compute_output_metrics(
    curve: Dict[str, Any],
    sim_cfg: Dict[str, Any],
    *,
    peak_window_ms: float = 100.0,
    drop_window_ms: float = 100.0,
    rebound_window_ms: Optional[float] = 300.0,
    auc_window: str = "stim",
    pdp_mode: Optional[str] = None,
    pdp_window_ms: Optional[float] = None,
    baseline_ms: Optional[float] = None,
    baseline_mode: Optional[str] = None,
    baseline_center_ms: Optional[float] = None,
    stim_start_ms: Optional[float] = None,
    stim_stop_ms: Optional[float] = None,
) -> Dict[str, Any]:
    auc_window = (auc_window or "stim").lower()
    t_ms = np.asarray(curve.get("t_ms", []) or [], dtype=float)
    rate = np.asarray(curve.get("rate_hz", []) or [], dtype=float)
    stim_start, stim_stop = _resolve_stim_window(sim_cfg)
    if stim_start_ms is not None:
        stim_start = float(stim_start_ms)
    if stim_stop_ms is not None:
        stim_stop = float(stim_stop_ms)

    metrics = {
        "peak_window_ms": float(peak_window_ms),
        "drop_window_ms": float(drop_window_ms),
        "auc_window": auc_window,
        "pdp_mode": None,
        "pdp_window_ms": None,
        "stim_start_ms": stim_start,
        "stim_stop_ms": stim_stop,
        "peak_time_ms": None,
        "peak_value": None,
        "peak_rate_hz": None,
        "peak_latency_ms": None,
        "drop_time_ms": None,
        "drop_value": None,
        "drop_pct": None,
        "rebound_window_ms": float(rebound_window_ms) if rebound_window_ms is not None else None,
        "rebound_time_ms": None,
        "rebound_value": None,
        "rebound_pct": None,
        "auc": None,
        "auc_units": f"{curve.get('units', 'Hz')}*s",
        "baseline_ms": None,
        "baseline_mode": None,
        "baseline_time_ms": None,
        "baseline_center_ms": None,
        "baseline_window_start_ms": None,
        "baseline_window_stop_ms": None,
        "baseline_mean": None,
        "norm_mode": curve.get("norm_mode"),
        "norm_window": curve.get("norm_window"),
        "norm_scale": curve.get("norm_scale"),
        "avg_norm_scale": curve.get("norm_scale") if curve.get("norm_mode") == "avg" else None,
    }

    if t_ms.size == 0 or rate.size == 0:
        return metrics

    if stim_start is None:
        return metrics

    baseline_ms_val = baseline_ms
    if baseline_ms_val is None:
        baseline_ms_val = curve.get("baseline_ms", 100.0)
    try:
        baseline_ms_val = float(baseline_ms_val)
    except Exception:
        baseline_ms_val = 100.0
    baseline_mode_val = (baseline_mode or curve.get("baseline_mode") or "window").lower()
    pdp_mode_val = (pdp_mode or curve.get("pdp_mode") or "point").lower()
    pdp_window_val = pdp_window_ms if pdp_window_ms is not None else curve.get("pdp_window_ms")
    try:
        pdp_window_val = float(pdp_window_val) if pdp_window_val is not None else 0.0
    except Exception:
        pdp_window_val = 0.0
    baseline_center_val = baseline_center_ms
    if baseline_center_val is None:
        baseline_center_val = curve.get("baseline_center_ms")
    try:
        baseline_center_val = float(baseline_center_val) if baseline_center_val is not None else float(baseline_ms_val) * 0.5
    except Exception:
        baseline_center_val = float(baseline_ms_val) * 0.5
    baseline_time = curve.get("baseline_time_ms")
    baseline_mean_curve = curve.get("baseline_mean")
    baseline_mean = baseline_mean_curve
    if baseline_time is None:
        baseline_time = float(stim_start) - float(baseline_center_val)
    if baseline_mean is None and t_ms.size:
        if baseline_mode_val == "point":
            baseline_mean = float(
                np.interp(baseline_time, t_ms, rate, left=rate[0], right=rate[-1])
            )
        else:
            half = float(baseline_ms_val) * 0.5
            baseline_start = baseline_time - half
            baseline_stop = baseline_time + half
            baseline_mask = _select_window_mask(t_ms, baseline_start, baseline_stop)
            baseline_mean = float(np.mean(rate[baseline_mask])) if baseline_mask.any() else 0.0

    metrics["baseline_ms"] = baseline_ms_val
    metrics["baseline_mode"] = baseline_mode_val
    metrics["baseline_time_ms"] = baseline_time
    metrics["baseline_center_ms"] = baseline_center_val
    if baseline_mode_val == "window":
        half = float(baseline_ms_val) * 0.5
        metrics["baseline_window_start_ms"] = float(baseline_time) - half
        metrics["baseline_window_stop_ms"] = float(baseline_time) + half
    metrics["baseline_mean"] = baseline_mean
    metrics["pdp_mode"] = pdp_mode_val
    metrics["pdp_window_ms"] = pdp_window_val

    peak_start = stim_start
    peak_stop = stim_start + float(peak_window_ms)
    peak_mask = _select_window_mask(t_ms, peak_start, peak_stop)
    if not peak_mask.any():
        return metrics
    peak_idx = np.argmax(rate[peak_mask])
    peak_indices = np.flatnonzero(peak_mask)
    peak_pos = peak_indices[peak_idx]
    peak_time = float(t_ms[peak_pos])
    peak_val = float(rate[peak_pos])

    metrics["peak_time_ms"] = peak_time
    metrics["peak_value"] = peak_val
    metrics["peak_rate_hz"] = peak_val
    metrics["peak_latency_ms"] = peak_time - float(stim_start)

    rate_bs = np.asarray(curve.get("rate_hz_baseline_sub") or [], dtype=float)
    raw_rate = None
    baseline_for_raw = baseline_mean_curve if baseline_mean_curve is not None else baseline_mean
    if rate_bs.size == rate.size and baseline_for_raw is not None:
        raw_rate = rate_bs + float(baseline_for_raw)
    elif not bool(curve.get("normalized")):
        raw_rate = rate.copy()

    if raw_rate is not None and raw_rate.size == rate.size and peak_mask.any():
        peak_raw_idx = np.argmax(raw_rate[peak_mask])
        peak_raw_pos = peak_indices[peak_raw_idx]
        peak_raw_val = float(raw_rate[peak_raw_pos])
        metrics["peak_value_raw"] = peak_raw_val
        metrics["peak_rate_hz_raw"] = peak_raw_val

    drop_target = peak_time + float(drop_window_ms)
    if pdp_mode_val == "window" and pdp_window_val > 0:
        half = 0.5 * pdp_window_val
        drop_mask = _select_window_mask(t_ms, drop_target - half, drop_target + half)
        if drop_mask.any():
            drop_time = float(drop_target)
            drop_val = float(np.mean(rate[drop_mask]))
        else:
            drop_pos = int(np.argmin(np.abs(t_ms - drop_target)))
            drop_time = float(t_ms[drop_pos])
            drop_val = float(rate[drop_pos])
    else:
        drop_pos = int(np.argmin(np.abs(t_ms - drop_target)))
        drop_time = float(t_ms[drop_pos])
        drop_val = float(rate[drop_pos])
    metrics["drop_time_ms"] = drop_time
    metrics["drop_value"] = drop_val
    metrics["drop_center_ms"] = float(drop_target)
    if pdp_mode_val == "window" and pdp_window_val > 0:
        metrics["drop_window_start_ms"] = float(drop_target) - (0.5 * float(pdp_window_val))
        metrics["drop_window_stop_ms"] = float(drop_target) + (0.5 * float(pdp_window_val))
    if peak_val != 0:
        metrics["drop_pct"] = 100.0 * (peak_val - drop_val) / peak_val

    if rebound_window_ms is not None:
        rebound_target = peak_time + float(rebound_window_ms)
        if pdp_mode_val == "window" and pdp_window_val > 0:
            half = 0.5 * pdp_window_val
            rebound_mask = _select_window_mask(t_ms, rebound_target - half, rebound_target + half)
            if rebound_mask.any():
                rebound_time = float(rebound_target)
                rebound_val = float(np.mean(rate[rebound_mask]))
            else:
                rebound_pos = int(np.argmin(np.abs(t_ms - rebound_target)))
                rebound_time = float(t_ms[rebound_pos])
                rebound_val = float(rate[rebound_pos])
        else:
            rebound_pos = int(np.argmin(np.abs(t_ms - rebound_target)))
            rebound_time = float(t_ms[rebound_pos])
            rebound_val = float(rate[rebound_pos])
        metrics["rebound_time_ms"] = rebound_time
        metrics["rebound_value"] = rebound_val
        metrics["rebound_center_ms"] = float(rebound_target)
        if pdp_mode_val == "window" and pdp_window_val > 0:
            metrics["rebound_window_start_ms"] = float(rebound_target) - (0.5 * float(pdp_window_val))
            metrics["rebound_window_stop_ms"] = float(rebound_target) + (0.5 * float(pdp_window_val))
        if peak_val != 0:
            metrics["rebound_pct"] = 100.0 * (peak_val - rebound_val) / peak_val

    tstart = sim_cfg.get("tstart")
    tstop = sim_cfg.get("tstop")
    if auc_window == "full":
        auc_mask = _select_window_mask(t_ms, tstart, tstop)
    else:
        auc_mask = _select_window_mask(t_ms, stim_start, stim_stop)
    if auc_mask.any():
        metrics["auc"] = float(np.trapz(rate[auc_mask], t_ms[auc_mask] / 1000.0))

    return metrics


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


def _unwrap_object_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object and value.ndim == 0:
        try:
            return value.item()
        except Exception:
            return value
    return value


def _resolve_time_axis(
    results: Dict[str, Any],
    *,
    n_hint: Optional[int] = None,
) -> np.ndarray:
    traces = results.get("traces", {}) or {}
    t_raw = traces.get("T")
    if t_raw is not None:
        try:
            t_ms = np.asarray(t_raw, dtype=float).ravel()
            if t_ms.size:
                return t_ms
        except Exception:
            pass

    sim_cfg = results.get("sim_cfg", {}) or {}
    dt = sim_cfg.get("dt")
    tstart = sim_cfg.get("tstart", 0.0)
    if n_hint is None or n_hint <= 0 or dt is None:
        return np.array([], dtype=float)
    try:
        dt = float(dt)
        tstart = float(tstart)
    except Exception:
        return np.array([], dtype=float)
    if dt <= 0:
        return np.array([], dtype=float)
    return tstart + dt * np.arange(int(n_hint), dtype=float)


def _window_masks(
    t_ms: np.ndarray,
    *,
    stim_start_ms: Optional[float],
    stim_stop_ms: Optional[float],
    baseline_ms: float,
) -> tuple[np.ndarray, np.ndarray, Optional[float], Optional[float], Optional[float], Optional[float]]:
    stim_mask = _select_window_mask(t_ms, stim_start_ms, stim_stop_ms)
    baseline_start = None
    baseline_stop = None
    if stim_start_ms is not None:
        baseline_stop = float(stim_start_ms)
        baseline_start = baseline_stop - float(max(0.0, baseline_ms))
        baseline_mask = _select_window_mask(t_ms, baseline_start, baseline_stop)
    else:
        baseline_mask = np.zeros_like(t_ms, dtype=bool)
    return (
        baseline_mask,
        stim_mask,
        baseline_start,
        baseline_stop,
        stim_start_ms,
        stim_stop_ms,
    )


def _summarize_series_metrics(
    values: np.ndarray,
    t_ms: np.ndarray,
    *,
    baseline_mask: np.ndarray,
    stim_mask: np.ndarray,
) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    metrics: Dict[str, Any] = {
        "n_samples": int(len(values)),
        "n_finite": int(arr.size),
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "mean_abs": None,
        "peak_abs": None,
        "baseline_mean": None,
        "stim_mean": None,
        "stim_delta_from_baseline": None,
        "stim_peak_abs": None,
        "stim_auc": None,
    }
    if arr.size == 0:
        return metrics

    metrics["mean"] = float(np.mean(arr))
    metrics["std"] = float(np.std(arr))
    metrics["min"] = float(np.min(arr))
    metrics["max"] = float(np.max(arr))
    metrics["mean_abs"] = float(np.mean(np.abs(arr)))
    metrics["peak_abs"] = float(np.max(np.abs(arr)))

    n = min(len(values), len(t_ms), len(baseline_mask), len(stim_mask))
    if n <= 0:
        return metrics

    vec = np.asarray(values, dtype=float).ravel()[:n]
    t_use = np.asarray(t_ms, dtype=float).ravel()[:n]
    base = np.asarray(baseline_mask[:n], dtype=bool)
    stim = np.asarray(stim_mask[:n], dtype=bool)

    if np.any(base):
        base_vals = vec[base]
        base_vals = base_vals[np.isfinite(base_vals)]
        if base_vals.size:
            metrics["baseline_mean"] = float(np.mean(base_vals))

    if np.any(stim):
        stim_vals = vec[stim]
        stim_vals = stim_vals[np.isfinite(stim_vals)]
        if stim_vals.size:
            metrics["stim_mean"] = float(np.mean(stim_vals))
            metrics["stim_peak_abs"] = float(np.max(np.abs(stim_vals)))
            baseline_mean = metrics.get("baseline_mean")
            if baseline_mean is not None:
                metrics["stim_delta_from_baseline"] = float(metrics["stim_mean"] - baseline_mean)
        if np.sum(stim) >= 2:
            try:
                metrics["stim_auc"] = float(np.trapz(vec[stim], t_use[stim] / 1000.0))
            except Exception:
                metrics["stim_auc"] = None

    return metrics


def _select_cell_recording_payload(
    results: Dict[str, Any],
    *,
    trial_idx: Optional[int] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[int]]:
    direct = _unwrap_object_scalar(results.get("cell_recordings"))
    if isinstance(direct, dict):
        chosen_trial = 0 if trial_idx is None else int(trial_idx)
        return direct, chosen_trial

    by_trial = _unwrap_object_scalar(results.get("cell_recordings_by_trial"))
    if by_trial is None:
        return None, None

    if isinstance(by_trial, np.ndarray):
        by_trial = list(by_trial.tolist())
    if not isinstance(by_trial, (list, tuple)) or not by_trial:
        return None, None

    entries = list(by_trial)
    chosen: Optional[Any] = None
    chosen_trial_idx: Optional[int] = None

    if trial_idx is not None:
        for entry in entries:
            entry = _unwrap_object_scalar(entry)
            if isinstance(entry, dict):
                eidx = entry.get("trial_idx")
                if eidx is not None and int(eidx) == int(trial_idx):
                    chosen = entry
                    chosen_trial_idx = int(trial_idx)
                    break
        if chosen is None:
            return None, int(trial_idx)
    else:
        chosen = _unwrap_object_scalar(entries[0])

    payload = chosen
    if isinstance(chosen, dict) and "recordings" in chosen:
        payload = chosen.get("recordings")
        if chosen_trial_idx is None and chosen.get("trial_idx") is not None:
            try:
                chosen_trial_idx = int(chosen.get("trial_idx"))
            except Exception:
                chosen_trial_idx = None
    payload = _unwrap_object_scalar(payload)
    if not isinstance(payload, dict):
        return None, chosen_trial_idx
    if chosen_trial_idx is None:
        chosen_trial_idx = 0 if trial_idx is None else int(trial_idx)
    return payload, chosen_trial_idx


def summarize_cell_recordings(
    results: Dict[str, Any],
    *,
    trial_idx: Optional[int] = None,
    baseline_ms: float = 100.0,
    stim_start_ms: Optional[float] = None,
    stim_stop_ms: Optional[float] = None,
    sites: Optional[Iterable[str]] = None,
    vars: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    payload, chosen_trial = _select_cell_recording_payload(results, trial_idx=trial_idx)
    if payload is None:
        return {
            "available": False,
            "trial_idx": chosen_trial,
            "sites": {},
            "site_count": 0,
            "series_count": 0,
            "message": "No cell_recordings payload found in results.",
        }

    site_filter = set(sites) if sites is not None else None
    var_filter = set(vars) if vars is not None else None

    n_hint = None
    for recs in payload.values():
        if isinstance(recs, dict) and recs:
            first = next(iter(recs.values()))
            try:
                n_hint = int(np.asarray(first).size)
            except Exception:
                n_hint = None
            break
    t_ms = _resolve_time_axis(results, n_hint=n_hint)

    sim_cfg = results.get("sim_cfg", {}) or {}
    stim_start, stim_stop = _resolve_stim_window(sim_cfg)
    if stim_start_ms is not None:
        stim_start = float(stim_start_ms)
    if stim_stop_ms is not None:
        stim_stop = float(stim_stop_ms)

    (
        baseline_mask,
        stim_mask,
        baseline_start,
        baseline_stop,
        stim_start,
        stim_stop,
    ) = _window_masks(
        t_ms,
        stim_start_ms=stim_start,
        stim_stop_ms=stim_stop,
        baseline_ms=float(baseline_ms),
    )

    out_sites: Dict[str, Dict[str, Any]] = {}
    series_count = 0
    for site_name in sorted(payload.keys(), key=str):
        if site_filter is not None and site_name not in site_filter:
            continue
        vars_raw = payload.get(site_name)
        if not isinstance(vars_raw, dict):
            continue
        site_out: Dict[str, Any] = {}
        for var_name in sorted(vars_raw.keys(), key=str):
            if var_filter is not None and var_name not in var_filter:
                continue
            values = np.asarray(vars_raw[var_name], dtype=float).ravel()
            if t_ms.size:
                n = min(values.size, t_ms.size)
                vals = values[:n]
                t_use = t_ms[:n]
                base = baseline_mask[:n]
                stim = stim_mask[:n]
            else:
                vals = values
                t_use = np.array([], dtype=float)
                base = np.zeros(vals.shape, dtype=bool)
                stim = np.zeros(vals.shape, dtype=bool)
            site_out[var_name] = _summarize_series_metrics(
                vals,
                t_use,
                baseline_mask=base,
                stim_mask=stim,
            )
            series_count += 1
        if site_out:
            out_sites[site_name] = site_out

    dt_mean = None
    if t_ms.size >= 2:
        dt_mean = float(np.mean(np.diff(t_ms)))

    return {
        "available": bool(out_sites),
        "trial_idx": chosen_trial,
        "site_count": int(len(out_sites)),
        "series_count": int(series_count),
        "time_start_ms": float(t_ms[0]) if t_ms.size else None,
        "time_stop_ms": float(t_ms[-1]) if t_ms.size else None,
        "time_n": int(t_ms.size),
        "time_dt_ms": dt_mean,
        "stim_start_ms": stim_start,
        "stim_stop_ms": stim_stop,
        "baseline_start_ms": baseline_start,
        "baseline_stop_ms": baseline_stop,
        "sites": out_sites,
    }


def _flatten_cell_recording_summary(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    flat: Dict[str, Dict[str, Any]] = {}
    for site, vars_map in (summary.get("sites", {}) or {}).items():
        if not isinstance(vars_map, dict):
            continue
        for var, metrics in vars_map.items():
            key = f"{site} | {var}"
            flat[key] = metrics if isinstance(metrics, dict) else {}
    return flat


def format_cell_recording_summary_table(
    summary: Dict[str, Any],
    *,
    title: Optional[str] = None,
    max_rows: int = 200,
) -> str:
    lines: list[str] = []
    if title:
        lines.append(f"**{title}**")
    lines.extend(
        [
            "| Site | Variable | n | mean | std | min | max | baseline_mean | stim_mean | stim_delta | stim_peak_abs | stim_auc |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    rows = 0
    for site in sorted((summary.get("sites") or {}).keys(), key=str):
        vars_map = (summary.get("sites") or {}).get(site) or {}
        for var in sorted(vars_map.keys(), key=str):
            metrics = vars_map.get(var) or {}
            lines.append(
                f"| {site} | {var} | {_format_value(metrics.get('n_finite'))} | "
                f"{_format_value(metrics.get('mean'))} | {_format_value(metrics.get('std'))} | "
                f"{_format_value(metrics.get('min'))} | {_format_value(metrics.get('max'))} | "
                f"{_format_value(metrics.get('baseline_mean'))} | {_format_value(metrics.get('stim_mean'))} | "
                f"{_format_value(metrics.get('stim_delta_from_baseline'))} | {_format_value(metrics.get('stim_peak_abs'))} | "
                f"{_format_value(metrics.get('stim_auc'))} |"
            )
            rows += 1
            if max_rows and rows >= int(max_rows):
                lines.append("| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |")
                return "\n".join(lines)
    if rows == 0:
        lines.append("| (no cell recordings found) | — | — | — | — | — | — | — | — | — | — | — |")
    return "\n".join(lines)


def format_cell_recording_summary_compare(
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
    *,
    labels: tuple[str, str] = ("A", "B"),
    diff_only: bool = True,
    title: Optional[str] = None,
) -> str:
    fields = [
        "n_finite",
        "mean",
        "std",
        "min",
        "max",
        "baseline_mean",
        "stim_mean",
        "stim_delta_from_baseline",
        "stim_peak_abs",
        "stim_auc",
    ]
    flat_a = _flatten_cell_recording_summary(summary_a)
    flat_b = _flatten_cell_recording_summary(summary_b)
    return _format_group_field_compare(
        flat_a,
        flat_b,
        fields=fields,
        labels=labels,
        groups=None,
        diff_only=diff_only,
        title=title,
    )


def _select_trace_trial_series(value: Any, *, trial_idx: Optional[int] = None) -> Optional[np.ndarray]:
    value = _unwrap_object_scalar(value)
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            if value.ndim == 0:
                return _select_trace_trial_series(value.item(), trial_idx=trial_idx)
            return _select_trace_trial_series(list(value.tolist()), trial_idx=trial_idx)
        if value.ndim == 1:
            return np.asarray(value, dtype=float).ravel()
        idx = 0 if trial_idx is None else int(trial_idx)
        idx = max(0, min(idx, value.shape[0] - 1))
        return np.asarray(value[idx], dtype=float).ravel()
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        first = value[0]
        if np.isscalar(first):
            return np.asarray(value, dtype=float).ravel()
        idx = 0 if trial_idx is None else int(trial_idx)
        idx = max(0, min(idx, len(value) - 1))
        return np.asarray(value[idx], dtype=float).ravel()
    try:
        return np.asarray(value, dtype=float).ravel()
    except Exception:
        return None


def summarize_total_synaptic_traces(
    results: Dict[str, Any],
    *,
    trial_idx: Optional[int] = None,
    baseline_ms: float = 100.0,
    stim_start_ms: Optional[float] = None,
    stim_stop_ms: Optional[float] = None,
) -> Dict[str, Any]:
    traces = results.get("traces", {}) or {}
    i_trace = _select_trace_trial_series(traces.get("I"), trial_idx=trial_idx)
    g_trace = _select_trace_trial_series(traces.get("G"), trial_idx=trial_idx)
    if i_trace is None and g_trace is None:
        return {
            "available": False,
            "trial_idx": trial_idx,
            "series": {},
            "message": "No total synaptic traces (I/G) found in results['traces'].",
        }

    n_hint = 0
    if i_trace is not None:
        n_hint = max(n_hint, int(i_trace.size))
    if g_trace is not None:
        n_hint = max(n_hint, int(g_trace.size))
    t_ms = _resolve_time_axis(results, n_hint=n_hint)
    if t_ms.size == 0 and n_hint > 0:
        t_ms = np.arange(n_hint, dtype=float)

    sim_cfg = results.get("sim_cfg", {}) or {}
    stim_start, stim_stop = _resolve_stim_window(sim_cfg)
    if stim_start_ms is not None:
        stim_start = float(stim_start_ms)
    if stim_stop_ms is not None:
        stim_stop = float(stim_stop_ms)

    (
        baseline_mask,
        stim_mask,
        baseline_start,
        baseline_stop,
        stim_start,
        stim_stop,
    ) = _window_masks(
        t_ms,
        stim_start_ms=stim_start,
        stim_stop_ms=stim_stop,
        baseline_ms=float(baseline_ms),
    )

    series: Dict[str, Any] = {}
    for key, arr in (("I", i_trace), ("G", g_trace)):
        if arr is None:
            continue
        n = min(arr.size, t_ms.size) if t_ms.size else arr.size
        vals = arr[:n]
        t_use = t_ms[:n] if t_ms.size else np.array([], dtype=float)
        base = baseline_mask[:n] if baseline_mask.size else np.zeros(n, dtype=bool)
        stim = stim_mask[:n] if stim_mask.size else np.zeros(n, dtype=bool)
        series[key] = _summarize_series_metrics(
            vals,
            t_use,
            baseline_mask=base,
            stim_mask=stim,
        )

    dt_mean = None
    if t_ms.size >= 2:
        dt_mean = float(np.mean(np.diff(t_ms)))

    return {
        "available": bool(series),
        "trial_idx": trial_idx,
        "time_start_ms": float(t_ms[0]) if t_ms.size else None,
        "time_stop_ms": float(t_ms[-1]) if t_ms.size else None,
        "time_n": int(t_ms.size),
        "time_dt_ms": dt_mean,
        "stim_start_ms": stim_start,
        "stim_stop_ms": stim_stop,
        "baseline_start_ms": baseline_start,
        "baseline_stop_ms": baseline_stop,
        "series": series,
    }


def format_total_synaptic_trace_table(
    summary: Dict[str, Any],
    *,
    title: Optional[str] = None,
) -> str:
    lines: list[str] = []
    if title:
        lines.append(f"**{title}**")
    lines.extend(
        [
            "| Trace | n | mean | std | min | max | baseline_mean | stim_mean | stim_delta | stim_peak_abs | stim_auc |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    rows = 0
    for key in ("I", "G"):
        metrics = (summary.get("series") or {}).get(key)
        if not isinstance(metrics, dict):
            continue
        lines.append(
            f"| {key} | {_format_value(metrics.get('n_finite'))} | {_format_value(metrics.get('mean'))} | "
            f"{_format_value(metrics.get('std'))} | {_format_value(metrics.get('min'))} | {_format_value(metrics.get('max'))} | "
            f"{_format_value(metrics.get('baseline_mean'))} | {_format_value(metrics.get('stim_mean'))} | "
            f"{_format_value(metrics.get('stim_delta_from_baseline'))} | {_format_value(metrics.get('stim_peak_abs'))} | "
            f"{_format_value(metrics.get('stim_auc'))} |"
        )
        rows += 1
    if rows == 0:
        lines.append("| (no total synaptic traces found) | — | — | — | — | — | — | — | — | — | — |")
    return "\n".join(lines)


def format_total_synaptic_trace_compare(
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
    *,
    labels: tuple[str, str] = ("A", "B"),
    diff_only: bool = True,
    title: Optional[str] = None,
) -> str:
    fields = [
        "n_finite",
        "mean",
        "std",
        "min",
        "max",
        "baseline_mean",
        "stim_mean",
        "stim_delta_from_baseline",
        "stim_peak_abs",
        "stim_auc",
    ]
    a_map = (summary_a.get("series") or {}) if isinstance(summary_a, dict) else {}
    b_map = (summary_b.get("series") or {}) if isinstance(summary_b, dict) else {}
    return _format_group_field_compare(
        a_map,
        b_map,
        fields=fields,
        labels=labels,
        groups=None,
        diff_only=diff_only,
        title=title,
    )


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
    std_mode: str = "std",
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
        if str(std_mode).lower() == "sem":
            denom = max(1, rates_arr.shape[0]) ** 0.5
            std_rate = std_rate / denom

        summary["t_ms"] = centers_ref.tolist() if centers_ref is not None else None
        summary["groups"][g] = {
            "mean_rate": mean_rate.tolist(),
            "std_rate": std_rate.tolist(),
            "n_trials": int(rates_arr.shape[0]),
            "n_syn": int(n_syn or 0),
        }

    return summary


def summarize_inputs_from_input_stats(
    input_stats: Dict[str, Any],
    *,
    groups: Optional[Iterable[str]] = None,
    smooth_ms: Optional[float] = None,
    max_trials: Optional[int] = None,
    std_mode: str = "std",
) -> Dict[str, Any]:
    """
    Summarize input_stats (binned inputs saved during simulation) into mean/std curves.
    """
    if not input_stats:
        raise KeyError("input_stats missing or empty")

    tstart = float(input_stats.get("tstart_ms", 0.0))
    tstop = float(input_stats.get("tstop_ms", 0.0))
    bin_ms = float(input_stats.get("bin_ms", 0.0) or 0.0)
    if bin_ms <= 0:
        raise ValueError("input_stats bin_ms must be > 0")

    t_ms = input_stats.get("t_ms") or []
    if not t_ms:
        edges = np.arange(tstart, tstop + bin_ms, bin_ms, dtype=float)
        if edges.size >= 2:
            t_ms = (edges[:-1] + 0.5 * bin_ms).tolist()

    trials = list(input_stats.get("trials") or [])
    if max_trials is not None:
        trials = trials[: max(0, int(max_trials))]

    group_names = set((input_stats.get("group_means") or {}).keys())
    if not group_names:
        for trial in trials:
            group_names.update((trial.get("groups") or {}).keys())
    if groups is not None:
        group_names = set(groups).intersection(group_names)
    group_names = sorted(group_names)

    summary: Dict[str, Any] = {
        "bin_ms": bin_ms,
        "tstart_ms": tstart,
        "tstop_ms": tstop,
        "t_ms": t_ms,
        "n_trials": len(trials),
        "groups": {},
    }

    for g in group_names:
        stack = []
        n_syn = None
        for trial in trials:
            gstats = (trial.get("groups") or {}).get(g)
            if not gstats:
                continue
            rate = gstats.get("rate_hz_by_bin_per_syn")
            if rate is None:
                rate = gstats.get("rate_hz_by_bin_total")
            if rate is None:
                continue
            stack.append(rate)
            if n_syn is None:
                n_syn = gstats.get("n_syn")

        if not stack:
            continue
        rates_arr = np.asarray(stack, dtype=float)
        mean_rate = rates_arr.mean(axis=0)
        std_rate = rates_arr.std(axis=0)
        if smooth_ms is not None:
            win_bins = int(round(float(smooth_ms) / bin_ms))
            mean_rate = _moving_average(mean_rate, win_bins)
            std_rate = _moving_average(std_rate, win_bins)
        if str(std_mode).lower() == "sem":
            denom = max(1, rates_arr.shape[0]) ** 0.5
            std_rate = std_rate / denom

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
    input_source: str = "saved",
    std_mode: str = "std",
) -> Dict[str, Any]:
    """
    Convenience wrapper around summarize_inputs_from_payload for loaded results.
    """
    input_source = (input_source or "saved").lower()
    if input_source not in ("saved", "stats", "auto"):
        input_source = "saved"

    payload: Dict[str, Any] = {}
    if results.get("inputs_by_trial") is not None:
        payload["inputs_by_trial"] = results.get("inputs_by_trial")
    if results.get("inputs") is not None:
        payload["inputs"] = results.get("inputs")

    input_stats = (results.get("meta") or {}).get("input_stats")
    if input_source == "stats" or (input_source == "auto" and not payload and input_stats):
        return summarize_inputs_from_input_stats(
            input_stats,
            groups=groups,
            smooth_ms=smooth_ms,
            max_trials=max_trials,
            std_mode=std_mode,
        )

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
        std_mode=std_mode,
    )


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
) -> Dict[str, Path]:
    """
    Save a small set of default plots into <run_dir>/plots.

    Returns a dict of plot name -> file path.
    """
    from . import plotting  # local import to avoid circular deps

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


# ---------------------------------------------------------------------
# Step 6 helpers (run selection + lightweight analysis glue)
# ---------------------------------------------------------------------

def find_scp_root(start: Path) -> Path:
    """
    Locate the SCP repo root starting from `start`.
    Falls back to `start` if no SCP layout is detected.
    """
    for p in [start] + list(start.parents):
        if (p / "cells").is_dir() and (p / "run_pipeline.py").is_file():
            return p
        if (p / "single_cells" / "cells").is_dir():
            return p / "single_cells"

    try:
        for child in start.iterdir():
            if not child.is_dir():
                continue
            if (child / "cells").is_dir() and (child / "run_pipeline.py").is_file():
                return child
            if (child / "single_cells" / "cells").is_dir():
                return child / "single_cells"
    except Exception:
        pass

    return start


def list_cells(base_dir: Path) -> list[str]:
    cells_dir = base_dir / "cells"
    if not cells_dir.is_dir():
        return []
    return sorted([p.name for p in cells_dir.iterdir() if p.is_dir()])


def list_tunes(base_dir: Path, cell: str) -> list[str]:
    base = base_dir / "cells" / cell
    if not base.is_dir():
        return []
    if (base / "tunes").is_dir():
        return ["tunes"]
    return sorted([p.name for p in base.iterdir() if p.is_dir()])


def list_models(base_dir: Path, cell: str, tunes: str) -> list[str]:
    base = base_dir / "cells" / cell / tunes
    if not base.is_dir():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir()])


def resolve_run_dir(candidate: Path) -> Path:
    if (candidate / "run_manifest.json").is_file():
        return candidate
    nested = candidate / "results"
    if (nested / "run_manifest.json").is_file():
        return nested
    return candidate


def collect_run_dirs(base_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()
    if not base_dir.is_dir():
        return candidates
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        resolved = resolve_run_dir(p)
        if (resolved / "run_manifest.json").is_file():
            key = str(resolved)
            if key not in seen:
                seen.add(key)
                candidates.append(resolved)
    return sorted(candidates, key=lambda p: (p / "run_manifest.json").stat().st_mtime)


def _candidate_mtime(candidate: Path) -> float:
    if candidate.is_file():
        return candidate.stat().st_mtime
    manifest = candidate / "run_manifest.json"
    if manifest.is_file():
        return manifest.stat().st_mtime
    results_manifest = candidate / "results" / "run_manifest.json"
    if results_manifest.is_file():
        return results_manifest.stat().st_mtime
    files = list(candidate.glob("*.pkl")) + list(candidate.glob("*.npz"))
    if len(files) == 1:
        return files[0].stat().st_mtime
    return candidate.stat().st_mtime


def collect_run_candidates(base_dir: Path) -> list[Path]:
    """
    Collect run folders plus legacy outputs (single .pkl/.npz files or folders
    containing a single .pkl/.npz) for selection UIs.
    """
    candidates: list[Path] = []
    seen: set[str] = set()
    if not base_dir.is_dir():
        return candidates

    for p in base_dir.iterdir():
        if p.is_dir():
            resolved = resolve_run_dir(p)
            if (resolved / "run_manifest.json").is_file():
                key = str(resolved)
                if key not in seen:
                    seen.add(key)
                    candidates.append(resolved)
                continue
            files = list(p.glob("*.pkl")) + list(p.glob("*.npz"))
            if len(files) == 1:
                key = str(p)
                if key not in seen:
                    seen.add(key)
                    candidates.append(p)
            continue
        if p.is_file() and p.suffix.lower() in (".pkl", ".npz"):
            key = str(p)
            if key not in seen:
                seen.add(key)
                candidates.append(p)

    return sorted(candidates, key=_candidate_mtime)


def resolve_run(base_dir: Path, stem_or_path: Optional[Union[str, Path]]) -> Path:
    if stem_or_path is None:
        stem_or_path = "latest"
    p = stem_or_path if isinstance(stem_or_path, Path) else Path(str(stem_or_path))
    if p.is_absolute():
        return resolve_run_dir(p)
    runs = collect_run_dirs(base_dir)
    if str(stem_or_path) in (None, "latest"):
        if not runs:
            raise FileNotFoundError(f"No run folders found under {base_dir}")
        return runs[-1]
    if str(stem_or_path) in ("previous", "prev", "latest-1"):
        if len(runs) < 2:
            raise FileNotFoundError(f"Need at least 2 runs under {base_dir}")
        return runs[-2]
    try:
        parts = list(p.parts)
        if "output_data" in parts:
            idx = parts.index("output_data")
            if idx + 1 < len(parts):
                candidate = base_dir / parts[idx + 1]
                if candidate.exists():
                    return resolve_run_dir(candidate)
    except Exception:
        pass
    if p.exists():
        return resolve_run_dir(p)
    candidate = base_dir / str(stem_or_path)
    if candidate.exists():
        return resolve_run_dir(candidate)
    names = [p.name for p in runs]
    raise FileNotFoundError(
        f"Run not found: {stem_or_path!r}. Available under {base_dir}: {names}"
    )


def run_label(run_dir: Path) -> str:
    return run_dir.parent.name if run_dir.name == "results" else run_dir.name


def plot_dir_for_run(run_dir: Path) -> Path:
    return run_dir.parent / "plots" if run_dir.name == "results" else run_dir / "plots"


def analysis_dir_for_run(run_dir: Path) -> Path:
    return run_dir.parent / "analysis" if run_dir.name == "results" else run_dir / "analysis"


def plot_dir_for_compare(base_dir: Path, run_a: Path, run_b: Path) -> Path:
    label = f"{run_label(run_a)}_vs_{run_label(run_b)}"
    return base_dir / "_comparisons" / label / "plots"


def analysis_dir_for_compare(base_dir: Path, run_a: Path, run_b: Path) -> Path:
    label = f"{run_label(run_a)}_vs_{run_label(run_b)}"
    return base_dir / "_comparisons" / label / "analysis"


def parse_groups(text: Optional[str]) -> Optional[list[str]]:
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    return [p.strip() for p in text.split(",") if p.strip()]


def parse_optional_float(text: Optional[Union[str, float, int]]) -> Optional[float]:
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_bio_curve_optional(
    *,
    enabled: bool,
    path: str,
    time_col: str = "Time",
    rate_col: str = "AvgFiringRate",
    t_min: float = 0.0,
    delay_ms: float = 0.0,
    time_unit: str = "s",
    shift_ms: Optional[float] = None,
    quiet: bool = False,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if not enabled:
        return None
    if not path:
        if not quiet:
            print("Bio curve enabled but path is empty.")
        return None
    try:
        from . import bio_curve

        t_s, rate = bio_curve.load_bio_curve(
            path,
            time_col=time_col,
            rate_col=rate_col,
            t_min=t_min,
            delay_ms=delay_ms,
            time_unit=time_unit,
        )
    except Exception as exc:
        if not quiet:
            print("Bio curve load failed:", exc)
        return None
    if shift_ms is not None:
        t_s = t_s + float(shift_ms) / 1000.0
    return (np.asarray(t_s), np.asarray(rate))


def select_inputs_payload(results: Dict[str, Any], *, trial_idx: int = 0) -> Optional[Dict[str, Any]]:
    payload = results.get("inputs")
    if payload is not None:
        return payload
    trials = results.get("inputs_by_trial") or []
    if not trials:
        return None
    idx = min(max(int(trial_idx), 0), len(trials) - 1)
    return (trials[idx] or {}).get("inputs")


def extract_synapse_values(
    syn_records: Dict[str, Any],
    field: str,
    groups: Optional[Iterable[str]] = None,
) -> np.ndarray:
    if not syn_records:
        return np.array([], dtype=float)
    if groups is None or list(groups) == ["all"]:
        groups = list(syn_records.keys())
    vals: list[float] = []
    for g in groups:
        for rec in syn_records.get(g, []) or []:
            if isinstance(rec, dict):
                val = rec.get(field)
            else:
                val = getattr(rec, field, None)
            if val is not None:
                vals.append(float(val))
    return np.asarray(vals, dtype=float)


def build_multi_plot_inputs(
    results: Dict[str, Any],
    *,
    plot_window: Optional[Union[tuple[Optional[float], Optional[float]], Dict[str, Any]]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    spikes_by_trial = results.get("spikes", []) or []
    if isinstance(spikes_by_trial, np.ndarray):
        spikes_by_trial = spikes_by_trial.tolist()
    all_param_data = {"multi": spikes_by_trial}
    sim_cfg = results.get("sim_cfg", {}) or {}
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
    pw = plot_window
    if pw is not None and not isinstance(pw, dict):
        pw = {"x": (pw[0], pw[1]), "y": (None, None)}
    return all_param_data, sim_params, pw


def load_cell_and_geometry(
    tune_dir: Path,
) -> tuple[Any, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    cell_cfg_path = tune_dir / "cell_configs" / "cell_config.json"
    geom_cfg_path = tune_dir / "cell_configs" / "geometry.json"
    cell_cfg = json.loads(cell_cfg_path.read_text())
    geom_cfg = json.loads(geom_cfg_path.read_text()) if geom_cfg_path.is_file() else None

    manifest = cell_cfg.get("paths", {}).get("manifest", "manifest.json")
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = (tune_dir / manifest).resolve()
    if not manifest_path.is_file():
        fallback = (tune_dir / "manifest.json").resolve()
        if fallback.is_file():
            manifest_path = fallback
        else:
            raise FileNotFoundError(
                f"manifest.json not found. Tried {manifest_path} and {fallback} (tune_dir={tune_dir})"
            )
    cell_cfg.setdefault("paths", {})["manifest"] = str(manifest_path)

    from modules_local import load_cell, geometry as geom_mod

    cwd = Path.cwd()
    try:
        os.chdir(tune_dir)
        cell = load_cell(cell_cfg)
        geom = geom_mod.define_geometry(cell, geom_cfg)
    finally:
        os.chdir(cwd)

    return cell, geom, geom_cfg


def resolve_figure(obj=None):
    if obj is None:
        return plt.gcf()
    if isinstance(obj, tuple):
        return resolve_figure(obj[0])
    if hasattr(obj, "savefig"):
        return obj
    if hasattr(obj, "figure"):
        return obj.figure
    return plt.gcf()


def save_figure(fig, out_path: Path, *, enabled: bool = True, dpi: int = 150) -> Optional[Path]:
    if not enabled:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = resolve_figure(fig)
    fig.savefig(out_path, dpi=dpi)
    return out_path


def save_json(data: Dict[str, Any], out_path: Path, *, enabled: bool = True) -> Optional[Path]:
    if not enabled:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))
    return out_path


def format_diff_list_table(diffs: Iterable[str], *, title: str, max_items: int = 20) -> str:
    lines = [f"**{title}**", "| # | Difference |", "| --- | --- |"]
    diffs_list = list(diffs or [])
    if not diffs_list:
        lines.append("| (no differences found) | — |")
        return "\n".join(lines)
    for i, line in enumerate(diffs_list[:max_items], 1):
        lines.append(f"| {i} | {line} |")
    return "\n".join(lines)


def snapshot_compare_report(
    run_a: Union[Path, Dict[str, Any]],
    run_b: Union[Path, Dict[str, Any]],
    *,
    labels: Optional[tuple[str, str]] = None,
    max_diffs: int = 200,
    diff_only: bool = True,
    save_table: bool = False,
    table_scope: str = "full",
    table_format: str = "csv",
    table_max_depth: int = 60,
    table_max_list_items: int = 200,
    out_dir: Optional[Path] = None,
    save_report_json: bool = False,
) -> Dict[str, Any]:
    label_a = labels[0] if labels else "A"
    label_b = labels[1] if labels else "B"
    report = compare_snapshot_runs(
        run_a,
        run_b,
        labels=(label_a, label_b),
        max_diffs=max_diffs,
        rtol=0.0,
        atol=0.0,
        print_summary=False,
    )

    manifest_table = format_diff_list_table(
        report.get("manifest_diff", []),
        title="Manifest differences",
        max_items=40,
    )
    deep_table = format_diff_list_table(
        report.get("diffs", []),
        title="Deep differences (first 20)",
        max_items=20,
    )

    if save_table and out_dir is not None:
        compare_path = out_dir / f"snapshot_compare_{table_scope}"
        save_compare_table(
            run_a,
            run_b,
            compare_path,
            scope=table_scope,
            fmt=table_format,
            max_depth=table_max_depth,
            max_list_items=table_max_list_items,
        )

    if save_report_json and out_dir is not None:
        save_json(report, out_dir / "snapshot_compare.json", enabled=True)

    return {
        "report": report,
        "snapshot_diff_table": report.get("snapshot_diff_table"),
        "manifest_diff_table": manifest_table,
        "deep_diff_table": deep_table,
    }


def summarize_iclamp(results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    traces = results.get("traces", {}) or {}
    T = traces.get("T")
    V = traces.get("V")
    meta = results.get("meta", {}) or {}
    if T is None or V is None:
        return None
    delay = meta.get("delay_ms", None)
    dur = meta.get("dur_ms", None)
    if delay is not None:
        base_mask = T < float(delay)
        baseline = float(V[base_mask].mean()) if base_mask.any() else float(V.mean())
    else:
        baseline = float(V.mean())
    peak = float(V.max())
    vmin = float(V.min())
    spike_count = None
    spike_rate = None
    if delay is not None and dur is not None:
        start = float(delay)
        stop = float(delay + dur)
        seg = (T >= start) & (T <= stop)
        if seg.any():
            vseg = V[seg]
            crossings = ((vseg[:-1] < -20.0) & (vseg[1:] >= -20.0)).sum()
            spike_count = int(crossings)
            spike_rate = crossings / (dur / 1000.0) if dur > 0 else None
    return {
        "T": T,
        "V": V,
        "delay_ms": delay,
        "dur_ms": dur,
        "baseline": baseline,
        "peak": peak,
        "vmin": vmin,
        "spike_count": spike_count,
        "spike_rate": spike_rate,
    }
