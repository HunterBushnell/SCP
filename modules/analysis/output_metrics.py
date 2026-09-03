"""Reusable output-metric calculation, formatting, and persistence.

This module is intentionally independent of the interactive Step 6 UI so the
same metric implementation can run after a simulation, from a notebook cell,
or from the Extra Analysis controls.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import numpy as np

from . import analysis


DEFAULT_PRESET_PATH = "modules/analysis/analysis_presets/output_metrics.json"

OUTPUT_METRIC_VALUE_ORDER = [
    "output_metrics_n_trials",
    "stim_spike_count",
    "stim_spike_count_median",
    "stim_mean_rate_hz",
    "stim_mean_rate_hz_median",
    "stim_response_trials",
    "stim_response_fraction",
    "stim_repetitive_trials",
    "stim_repetitive_fraction",
    "first_spike_contributing_trials",
    "first_spike_time_ms",
    "first_spike_time_median_ms",
    "first_spike_latency_ms",
    "first_spike_latency_median_ms",
    "initial_isi_contributing_trials",
    "first_to_second_isi_ms",
    "first_to_second_isi_median_ms",
    "initial_pair_rate_hz",
    "initial_pair_rate_median_hz",
    "mean_isi_ms",
    "mean_isi_median_ms",
    "min_isi_ms",
    "min_isi_median_ms",
    "peak_within_trial_rate_hz",
    "peak_within_trial_rate_median_hz",
    "baseline_mean",
    "peak_rate_hz_raw",
    "peak_value_raw",
    "peak_rate_hz",
    "peak_value",
    "peak_latency_ms",
    "rise_start_time_ms",
    "rise_start_latency_ms",
    "rise_start_rate_hz",
    "rise_stop_time_ms",
    "rise_stop_latency_ms",
    "rise_stop_rate_hz",
    "rise_time_ms",
    "rise_delta_rate_hz",
    "drop_value",
    "drop_pct",
    "t50_ms",
    "rebound_value",
    "rebound_pct",
    "auc_raw_hz_s",
    "auc_normalized_s",
    "auc",
]

OUTPUT_METRIC_PLOT_DEFAULT_KEYS = [
    "stim_spike_count",
    "stim_response_fraction",
    "stim_repetitive_fraction",
    "first_spike_latency_ms",
    "first_to_second_isi_ms",
    "initial_pair_rate_hz",
    "baseline_mean",
    "peak_rate_hz_raw",
    "peak_latency_ms",
    "rise_start_time_ms",
    "rise_start_rate_hz",
    "rise_stop_time_ms",
    "rise_stop_rate_hz",
    "rise_time_ms",
    "rise_delta_rate_hz",
    "drop_pct",
    "t50_ms",
    "rebound_pct",
    "auc_raw_hz_s",
    "auc_normalized_s",
]

DEFAULT_IMPORTANT_KEYS = [
    "stim_spike_count",
    "stim_mean_rate_hz",
    "stim_response_fraction",
    "stim_repetitive_fraction",
    "first_spike_latency_ms",
    "first_to_second_isi_ms",
    "initial_pair_rate_hz",
    "peak_rate_hz_raw",
]

OUTPUT_PARAM_KEYS = {
    "peak_window_ms",
    "drop_window_ms",
    "rebound_window_ms",
    "auc_window",
    "auc_window_start_ms",
    "auc_window_stop_ms",
    "auc_units",
    "auc_raw_units",
    "auc_normalized_units",
    "auc_normalization_mode",
    "auc_normalization_window",
    "auc_normalization_scale_hz",
    "pdp_mode",
    "pdp_window_ms",
    "t50_mode",
    "stim_start_ms",
    "stim_stop_ms",
    "baseline_ms",
    "baseline_mode",
    "baseline_center_ms",
    "baseline_time_ms",
    "baseline_window_start_ms",
    "baseline_window_stop_ms",
    "peak_time_ms",
    "rise_metric_enabled",
    "rise_start_pct",
    "rise_stop_pct",
    "rise_start_value",
    "rise_stop_value",
    "drop_time_ms",
    "t50_time_ms",
    "rebound_time_ms",
    "t50_value",
    "drop_center_ms",
    "drop_window_start_ms",
    "drop_window_stop_ms",
    "rebound_center_ms",
    "rebound_window_start_ms",
    "rebound_window_stop_ms",
    "norm_mode",
    "norm_window",
    "norm_scale",
    "avg_norm_scale",
    "output_metrics_std_mode",
    "output_metrics_bin_ms",
    "output_metrics_smooth_ms",
    "output_metrics_smooth_mode",
    "output_metrics_curve_mode",
    "output_metrics_norm_mode",
    "output_metrics_norm_window",
    "output_stim_spike_metrics_enabled",
    "output_first_spike_metric_enabled",
    "output_isi_metrics_enabled",
}

_HIGHLIGHT_METRICS = {
    "stim_spike_count",
    "stim_mean_rate_hz",
    "stim_response_fraction",
    "stim_repetitive_fraction",
    "first_spike_latency_ms",
    "first_to_second_isi_ms",
    "initial_pair_rate_hz",
    "peak_within_trial_rate_hz",
    "peak_latency_ms",
    "rise_start_time_ms",
    "rise_start_latency_ms",
    "rise_stop_time_ms",
    "rise_stop_latency_ms",
    "rise_time_ms",
    "rise_start_rate_hz",
    "rise_stop_rate_hz",
    "rise_delta_rate_hz",
    "drop_pct",
    "t50_ms",
    "rebound_pct",
    "auc_raw_hz_s",
    "auc_normalized_s",
    "auc",
}

_RAW_SUMMARY_KEYS = (
    "stim_spike_count",
    "stim_mean_rate_hz",
    "first_spike_time_ms",
    "first_spike_latency_ms",
    "first_to_second_isi_ms",
    "initial_pair_rate_hz",
    "mean_isi_ms",
    "min_isi_ms",
    "peak_within_trial_rate_hz",
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_output_metrics_preset(
    *,
    repo_root: Optional[Union[str, Path]] = None,
    preset_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load the dedicated output-metrics preset and apply explicit overrides."""
    if repo_root is None:
        try:
            root = analysis.find_scp_root(Path.cwd())
        except Exception:
            root = Path.cwd().resolve()
    else:
        root = Path(repo_root).expanduser().resolve()

    path = Path(DEFAULT_PRESET_PATH if preset_path in (None, "", False) else str(preset_path)).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()

    warnings: list[str] = []
    config: Dict[str, Any] = {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        defaults = payload.get("defaults", payload) if isinstance(payload, dict) else None
        if isinstance(defaults, dict):
            config.update(defaults)
        else:
            warnings.append(f"Output-metrics preset defaults missing/invalid in {path}")
    except Exception as exc:
        warnings.append(f"Output-metrics preset load failed ({path}): {exc}")

    if overrides:
        for key, value in overrides.items():
            if not _is_missing(value):
                config[key] = value

    return {
        "config": config,
        "preset_path": str(path),
        "repo_root": str(root),
        "warnings": warnings,
    }


def merge_metric_settings(
    settings: Optional[Dict[str, Any]],
    defaults: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    out = dict(defaults or {})
    for key, value in (settings or {}).items():
        if not _is_missing(value):
            out[key] = value
    return out


def _setting(settings: Dict[str, Any], dedicated: str, legacy: str, default: Any) -> Any:
    value = settings.get(dedicated)
    if _is_missing(value):
        value = settings.get(legacy)
    return default if _is_missing(value) else value


def _std_mode(settings: Dict[str, Any]) -> str:
    mode = str(settings.get("output_metrics_std_mode", "std") or "std").strip().lower()
    return mode if mode in {"std", "sem"} else "std"


def _stim_cfg(sim_cfg: Dict[str, Any], settings: Dict[str, Any], shift_ms: Optional[float]) -> Dict[str, Any]:
    cfg = dict(sim_cfg or {})
    if settings.get("output_stim_start_ms") is not None:
        cfg["stim_start_ms"] = float(settings["output_stim_start_ms"])
    if settings.get("output_stim_stop_ms") is not None:
        cfg["stim_stop_ms"] = float(settings["output_stim_stop_ms"])
    if shift_ms is not None:
        shift = float(shift_ms)
        start, stop = analysis._resolve_stim_window(cfg)
        if start is not None:
            cfg["stim_start_ms"] = float(start) + shift
        if stop is not None:
            cfg["stim_stop_ms"] = float(stop) + shift
    return cfg


def extract_spike_trials(results: Optional[Dict[str, Any]]) -> list[np.ndarray]:
    if not isinstance(results, dict) or results.get("spikes") is None:
        return []
    spikes = results.get("spikes")
    if results.get("mode") != "multi":
        raw_trials: Iterable[Any] = [spikes]
    elif isinstance(spikes, np.ndarray):
        if spikes.dtype == object:
            raw_trials = list(spikes.tolist())
        elif spikes.ndim > 1:
            raw_trials = list(spikes)
        else:
            raw_trials = [spikes]
    elif isinstance(spikes, (list, tuple)):
        if not spikes or isinstance(spikes[0], (int, float, np.integer, np.floating)):
            raw_trials = [spikes]
        else:
            raw_trials = spikes
    else:
        raw_trials = [spikes]

    trials: list[np.ndarray] = []
    for trial in raw_trials:
        arr = np.asarray(trial, dtype=float).ravel()
        arr = np.sort(arr[np.isfinite(arr)])
        trials.append(arr)
    return trials


def _build_base_curve(
    results: Dict[str, Any],
    settings: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    bin_ms = _setting(settings, "output_metrics_bin_ms", "output_bin_ms", None)
    smooth_ms = _setting(settings, "output_metrics_smooth_ms", "output_smooth_ms", None)
    if smooth_ms is None:
        smooth_ms = settings.get("win_size")
    smooth_mode = str(
        _setting(settings, "output_metrics_smooth_mode", "output_smooth_mode", "center")
    )
    return analysis.compute_output_curve_from_results(
        results,
        bin_ms=bin_ms,
        smooth_ms=smooth_ms,
        smooth_mode=smooth_mode,
    )


def _prepare_curve(
    curve: Optional[Dict[str, Any]],
    results: Dict[str, Any],
    settings: Dict[str, Any],
    *,
    mode: str,
    shift_ms: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    if not curve:
        return None

    sim_cfg = _stim_cfg(results.get("sim_cfg", {}) or {}, settings, shift_ms=None)
    curve = analysis.normalize_output_curve(
        curve,
        sim_cfg,
        mode=mode,
        norm_mode=str(_setting(settings, "output_metrics_norm_mode", "output_norm_mode", "peak")),
        baseline_ms=settings.get("output_metric_window_ms", 100.0),
        baseline_mode=settings.get("output_metric_mode", "window"),
        baseline_center_ms=settings.get("output_baseline_center_ms"),
        norm_window=_setting(settings, "output_metrics_norm_window", "output_norm_window", "stim"),
    )
    if shift_ms is not None:
        t_ms = np.asarray(curve.get("t_ms", []) or [], dtype=float)
        if t_ms.size:
            curve = dict(curve)
            curve["t_ms"] = (t_ms + float(shift_ms)).tolist()
    return curve


def _build_curve_variants(
    results: Dict[str, Any],
    settings: Dict[str, Any],
    *,
    shift_ms: Optional[float] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    base_curve = _build_base_curve(results, settings)
    raw_curve = _prepare_curve(
        base_curve,
        results,
        settings,
        mode="raw",
        shift_ms=shift_ms,
    )
    normalized_curve = _prepare_curve(
        base_curve,
        results,
        settings,
        mode="normalized",
        shift_ms=shift_ms,
    )
    selected_mode = str(
        _setting(settings, "output_metrics_curve_mode", "output_curve_mode", "raw")
    ).strip().lower()
    selected_curve = normalized_curve if selected_mode == "normalized" else raw_curve
    return {
        "selected": selected_curve,
        "raw": raw_curve,
        "normalized": normalized_curve,
    }


def _build_curve(
    results: Dict[str, Any],
    settings: Dict[str, Any],
    *,
    shift_ms: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    return _build_curve_variants(results, settings, shift_ms=shift_ms)["selected"]


def _compute_curve_metrics(
    curve: Optional[Dict[str, Any]],
    sim_cfg: Dict[str, Any],
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    metrics = analysis.compute_output_metrics(
        curve or {},
        sim_cfg or {},
        peak_window_ms=settings.get("output_peak_window_ms", 100.0),
        drop_window_ms=settings.get("output_drop_window_ms", 100.0),
        rebound_window_ms=settings.get("output_rebound_window_ms", 300.0),
        auc_window=settings.get("output_auc_window", "stim"),
        t50_mode=settings.get("output_t50_mode", "absolute"),
        pdp_mode=settings.get("output_metric_mode", "point"),
        pdp_window_ms=settings.get("output_metric_window_ms", 0.0),
        baseline_ms=settings.get("output_metric_window_ms", 100.0),
        baseline_mode=settings.get("output_metric_mode", "point"),
        baseline_center_ms=settings.get("output_baseline_center_ms"),
        stim_start_ms=settings.get("output_stim_start_ms"),
        stim_stop_ms=settings.get("output_stim_stop_ms"),
        rise_percent_range=settings.get("output_rise_percent_range", (10.0, 90.0)),
        rise_metric_enabled=bool(settings.get("output_rise_metric_enabled", True)),
    )
    metrics.update({
        "output_metrics_bin_ms": _setting(settings, "output_metrics_bin_ms", "output_bin_ms", None),
        "output_metrics_smooth_ms": _setting(settings, "output_metrics_smooth_ms", "output_smooth_ms", settings.get("win_size")),
        "output_metrics_smooth_mode": _setting(settings, "output_metrics_smooth_mode", "output_smooth_mode", "center"),
        "output_metrics_curve_mode": _setting(settings, "output_metrics_curve_mode", "output_curve_mode", "raw"),
        "output_metrics_norm_mode": _setting(settings, "output_metrics_norm_mode", "output_norm_mode", "peak"),
        "output_metrics_norm_window": _setting(settings, "output_metrics_norm_window", "output_norm_window", "stim"),
        "output_stim_spike_metrics_enabled": bool(settings.get("output_stim_spike_metrics_enabled", True)),
        "output_first_spike_metric_enabled": bool(settings.get("output_first_spike_metric_enabled", True)),
        "output_isi_metrics_enabled": bool(settings.get("output_isi_metrics_enabled", True)),
    })
    return metrics


def _compute_curve_metrics_with_auc_variants(
    selected_curve: Optional[Dict[str, Any]],
    raw_curve: Optional[Dict[str, Any]],
    normalized_curve: Optional[Dict[str, Any]],
    sim_cfg: Dict[str, Any],
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    metrics = _compute_curve_metrics(selected_curve, sim_cfg, settings)
    raw_metrics = (
        metrics
        if selected_curve is raw_curve
        else _compute_curve_metrics(raw_curve, sim_cfg, settings)
    )
    normalized_metrics = (
        metrics
        if selected_curve is normalized_curve
        else _compute_curve_metrics(normalized_curve, sim_cfg, settings)
    )

    metrics["auc_raw_hz_s"] = raw_metrics.get("auc")
    metrics["auc_normalized_s"] = normalized_metrics.get("auc")
    metrics["auc_raw_units"] = raw_metrics.get("auc_units", "Hz*s")
    metrics["auc_normalized_units"] = normalized_metrics.get(
        "auc_units", "normalized*s"
    )
    metrics["auc_normalization_mode"] = (
        normalized_curve or {}
    ).get("norm_mode")
    metrics["auc_normalization_window"] = (
        normalized_curve or {}
    ).get("norm_window")
    metrics["auc_normalization_scale_hz"] = (
        normalized_curve or {}
    ).get("norm_scale")
    return metrics


def _raw_trial_metrics(
    spikes: np.ndarray,
    *,
    stim_start: Optional[float],
    stim_stop: Optional[float],
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "stim_spike_count": None,
        "stim_mean_rate_hz": None,
        "stim_response": None,
        "stim_repetitive_response": None,
        "first_spike_time_ms": None,
        "first_spike_latency_ms": None,
        "first_to_second_isi_ms": None,
        "initial_pair_rate_hz": None,
        "mean_isi_ms": None,
        "min_isi_ms": None,
        "peak_within_trial_rate_hz": None,
    }
    if stim_start is None:
        return out

    mask = spikes >= float(stim_start)
    if stim_stop is not None:
        mask &= spikes <= float(stim_stop)
    stim_spikes = spikes[mask]
    count = int(stim_spikes.size)

    if bool(settings.get("output_stim_spike_metrics_enabled", True)):
        out["stim_spike_count"] = float(count)
        out["stim_response"] = float(count >= 1)
        out["stim_repetitive_response"] = float(count >= 2)
        if stim_stop is not None and float(stim_stop) > float(stim_start):
            out["stim_mean_rate_hz"] = float(count) / ((float(stim_stop) - float(stim_start)) / 1000.0)

    if count and bool(settings.get("output_first_spike_metric_enabled", True)):
        out["first_spike_time_ms"] = float(stim_spikes[0])
        out["first_spike_latency_ms"] = float(stim_spikes[0] - float(stim_start))

    if count >= 2 and bool(settings.get("output_isi_metrics_enabled", True)):
        isi = np.diff(stim_spikes)
        isi = isi[np.isfinite(isi) & (isi > 0.0)]
        if isi.size:
            first_isi = float(isi[0])
            min_isi = float(np.min(isi))
            out["first_to_second_isi_ms"] = first_isi
            out["initial_pair_rate_hz"] = 1000.0 / first_isi
            out["mean_isi_ms"] = float(np.mean(isi))
            out["min_isi_ms"] = min_isi
            out["peak_within_trial_rate_hz"] = 1000.0 / min_isi
    return out


def _numeric(values: Iterable[Any]) -> np.ndarray:
    out: list[float] = []
    for value in values:
        try:
            val = float(value)
        except Exception:
            continue
        if np.isfinite(val):
            out.append(val)
    return np.asarray(out, dtype=float)


def _spread(values: np.ndarray, mode: str) -> Optional[float]:
    if values.size == 0:
        return None
    value = float(np.std(values))
    if mode == "sem":
        value /= float(np.sqrt(values.size))
    return value


def compute_trial_output_metrics(
    results: Optional[Dict[str, Any]],
    settings: Dict[str, Any],
    *,
    shift_ms: Optional[float] = None,
) -> list[Dict[str, Any]]:
    """Calculate curve and raw-spike metrics independently for every trial."""
    if not isinstance(results, dict):
        return []
    sim_cfg = results.get("sim_cfg", {}) or {}
    sim_cfg_metrics = _stim_cfg(sim_cfg, settings, shift_ms)
    stim_start, stim_stop = analysis._resolve_stim_window(sim_cfg_metrics)
    trial_metrics: list[Dict[str, Any]] = []
    for spikes in extract_spike_trials(results):
        trial_result = {"mode": "single", "spikes": spikes, "sim_cfg": _stim_cfg(sim_cfg, settings, None)}
        curves = _build_curve_variants(trial_result, settings, shift_ms=shift_ms)
        metrics = _compute_curve_metrics_with_auc_variants(
            curves["selected"],
            curves["raw"],
            curves["normalized"],
            sim_cfg_metrics,
            settings,
        )
        metrics.update(
            _raw_trial_metrics(
                spikes + (float(shift_ms) if shift_ms is not None else 0.0),
                stim_start=stim_start,
                stim_stop=stim_stop,
                settings=settings,
            )
        )
        trial_metrics.append(metrics)
    return trial_metrics


def compute_output_metrics_from_results(
    results: Dict[str, Any],
    settings: Optional[Dict[str, Any]] = None,
    *,
    curve: Optional[Dict[str, Any]] = None,
    shift_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute the complete output metric set for one saved/in-memory run."""
    cfg = dict(settings or {})
    sim_cfg = results.get("sim_cfg", {}) or {}
    sim_cfg_metrics = _stim_cfg(sim_cfg, cfg, shift_ms)
    curves = _build_curve_variants(results, cfg, shift_ms=shift_ms)
    use_curve = curve if curve is not None else curves["selected"]
    metrics = _compute_curve_metrics_with_auc_variants(
        use_curve,
        curves["raw"],
        curves["normalized"],
        sim_cfg_metrics,
        cfg,
    )
    mode = _std_mode(cfg)
    metrics["output_metrics_std_mode"] = mode

    trial_metrics = compute_trial_output_metrics(results, cfg, shift_ms=shift_ms)
    metrics["output_metrics_n_trials"] = len(trial_metrics)

    # Curve-derived trial spread uses the same per-trial computation as Step 6.
    for key in OUTPUT_METRIC_VALUE_ORDER:
        vals = _numeric(trial.get(key) for trial in trial_metrics)
        if vals.size and key not in _RAW_SUMMARY_KEYS:
            spread = _spread(vals, mode)
            if spread is not None:
                metrics[f"{key}_spread"] = spread

    # Raw-spike quantities are aggregated per trial, never from a pooled PSTH.
    for key in _RAW_SUMMARY_KEYS:
        vals = _numeric(trial.get(key) for trial in trial_metrics)
        if not vals.size:
            metrics.setdefault(key, None)
            continue
        metrics[key] = float(np.mean(vals))
        metrics[f"{key}_spread"] = _spread(vals, mode)
        median_key = {
            "first_spike_time_ms": "first_spike_time_median_ms",
            "first_spike_latency_ms": "first_spike_latency_median_ms",
            "first_to_second_isi_ms": "first_to_second_isi_median_ms",
            "initial_pair_rate_hz": "initial_pair_rate_median_hz",
            "mean_isi_ms": "mean_isi_median_ms",
            "min_isi_ms": "min_isi_median_ms",
            "peak_within_trial_rate_hz": "peak_within_trial_rate_median_hz",
            "stim_spike_count": "stim_spike_count_median",
            "stim_mean_rate_hz": "stim_mean_rate_hz_median",
        }[key]
        metrics[median_key] = float(np.median(vals))

    response_vals = _numeric(trial.get("stim_response") for trial in trial_metrics)
    repeat_vals = _numeric(trial.get("stim_repetitive_response") for trial in trial_metrics)
    first_vals = _numeric(trial.get("first_spike_latency_ms") for trial in trial_metrics)
    isi_vals = _numeric(trial.get("first_to_second_isi_ms") for trial in trial_metrics)
    metrics["stim_response_trials"] = int(np.sum(response_vals)) if response_vals.size else 0
    metrics["stim_response_fraction"] = float(np.mean(response_vals)) if response_vals.size else None
    metrics["stim_repetitive_trials"] = int(np.sum(repeat_vals)) if repeat_vals.size else 0
    metrics["stim_repetitive_fraction"] = float(np.mean(repeat_vals)) if repeat_vals.size else None
    metrics["first_spike_contributing_trials"] = int(first_vals.size)
    metrics["initial_isi_contributing_trials"] = int(isi_vals.size)
    for key in OUTPUT_METRIC_VALUE_ORDER:
        metrics.setdefault(key, None)
    return metrics


def split_output_metrics(metrics: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    params: Dict[str, Any] = {}
    values: Dict[str, Any] = {}
    for key, value in metrics.items():
        if key in OUTPUT_PARAM_KEYS:
            params[key] = value
            if key == "output_metrics_std_mode":
                values[key] = value
        else:
            values[key] = value
    return params, values


def split_output_metrics_columns(
    data_by_label: Dict[str, Dict[str, Any]],
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    params_by_label: Dict[str, Dict[str, Any]] = {}
    values_by_label: Dict[str, Dict[str, Any]] = {}
    for label, metrics in data_by_label.items():
        params_by_label[label], values_by_label[label] = split_output_metrics(metrics)
    return params_by_label, values_by_label


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not np.isfinite(value):
            return ""
        return f"{value:.3g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_value(item) for item in value)
    return str(value)


def _rise_percent_label(data: Optional[Dict[str, Any]]) -> str:
    data = data or {}
    start = _format_value(data.get("rise_start_pct")) or "start"
    stop = _format_value(data.get("rise_stop_pct")) or "stop"
    return f"{start}-{stop}%"


def _rise_endpoint_label(data: Optional[Dict[str, Any]], endpoint: str) -> str:
    value = _format_value((data or {}).get(f"rise_{endpoint}_pct"))
    return f"{value}%" if value else endpoint


_METRIC_LABELS = {
    "output_metrics_n_trials": "Trials",
    "stim_spike_count": "Mean stimulus spikes per trial",
    "stim_spike_count_median": "Median stimulus spikes per trial",
    "stim_mean_rate_hz": "Mean whole-stimulus rate (Hz)",
    "stim_mean_rate_hz_median": "Median whole-stimulus rate (Hz)",
    "stim_response_trials": "Trials with at least one stimulus spike",
    "stim_response_fraction": "Stimulus response fraction (>=1 spike)",
    "stim_repetitive_trials": "Trials with at least two stimulus spikes",
    "stim_repetitive_fraction": "Repetitive-response fraction (>=2 spikes)",
    "first_spike_contributing_trials": "Trials contributing first-spike latency",
    "first_spike_time_ms": "First-spike time (ms)",
    "first_spike_time_median_ms": "Median first-spike time (ms)",
    "first_spike_latency_ms": "First-spike latency (ms)",
    "first_spike_latency_median_ms": "Median first-spike latency (ms)",
    "initial_isi_contributing_trials": "Trials contributing ISI metrics",
    "first_to_second_isi_ms": "First-to-second spike interval (ms)",
    "first_to_second_isi_median_ms": "Median first-to-second spike interval (ms)",
    "initial_pair_rate_hz": "Initial paired-spike rate (Hz)",
    "initial_pair_rate_median_hz": "Median initial paired-spike rate (Hz)",
    "mean_isi_ms": "Mean within-trial ISI (ms)",
    "mean_isi_median_ms": "Median of within-trial mean ISI (ms)",
    "min_isi_ms": "Minimum within-trial ISI (ms)",
    "min_isi_median_ms": "Median minimum within-trial ISI (ms)",
    "peak_within_trial_rate_hz": "Peak within-trial instantaneous rate (Hz)",
    "peak_within_trial_rate_median_hz": "Median peak within-trial instantaneous rate (Hz)",
    "peak_rate_hz_raw": "Peak PSTH rate (Hz)",
    "peak_latency_ms": "Peak PSTH latency (ms)",
    "t50_ms": "T50 (ms)",
    "auc_raw_hz_s": "Raw AUC (Hz*s)",
    "auc_normalized_s": "Normalized AUC (normalized*s)",
    "auc": "Selected-curve AUC",
}


def format_metric_key(key: str, data: Optional[Dict[str, Any]] = None) -> str:
    if key == "rise_start_time_ms":
        label = f"Rise start time ({_rise_endpoint_label(data, 'start')}, ms)"
    elif key == "rise_start_latency_ms":
        label = f"Rise start latency ({_rise_endpoint_label(data, 'start')}, ms)"
    elif key == "rise_stop_time_ms":
        label = f"Rise stop time ({_rise_endpoint_label(data, 'stop')}, ms)"
    elif key == "rise_stop_latency_ms":
        label = f"Rise stop latency ({_rise_endpoint_label(data, 'stop')}, ms)"
    elif key == "rise_time_ms":
        label = f"Rise time ({_rise_percent_label(data)}, ms)"
    elif key == "rise_start_rate_hz":
        label = f"Rise start rate ({_rise_endpoint_label(data, 'start')}, Hz)"
    elif key == "rise_stop_rate_hz":
        label = f"Rise stop rate ({_rise_endpoint_label(data, 'stop')}, Hz)"
    elif key == "rise_delta_rate_hz":
        label = f"Rise rate difference ({_rise_percent_label(data)}, Hz)"
    else:
        label = _METRIC_LABELS.get(key, key)
    return f"**{label}**" if key in _HIGHLIGHT_METRICS else label


def _filter_metrics_for_display(data: Dict[str, Any]) -> Dict[str, Any]:
    filtered = dict(data or {})
    peak_raw = filtered.get("peak_rate_hz_raw", filtered.get("peak_value_raw"))
    if peak_raw is not None:
        for key in ("peak_value", "peak_rate_hz"):
            try:
                if filtered.get(key) is not None and abs(float(filtered[key]) - 1.0) < 1e-6:
                    filtered.pop(key, None)
            except Exception:
                pass
    if "peak_value_raw" in filtered and "peak_rate_hz_raw" in filtered:
        try:
            if abs(float(filtered["peak_value_raw"]) - float(filtered["peak_rate_hz_raw"])) < 1e-6:
                filtered.pop("peak_value_raw", None)
        except Exception:
            pass
    return filtered


def _metric_cell(data: Dict[str, Any], key: str) -> str:
    cell = _format_value(data.get(key))
    spread = _format_value(data.get(f"{key}_spread"))
    if not cell or not spread:
        return cell
    label = "SEM" if str(data.get("output_metrics_std_mode", "std")).lower() == "sem" else "STD"
    return f"{cell} +/- {spread} ({label})"


def _ordered_keys(data: Dict[str, Any], metric_keys: Optional[Sequence[str]]) -> list[str]:
    keys = [
        key
        for key in OUTPUT_METRIC_VALUE_ORDER
        if key in data and not key.endswith("_spread") and key != "output_metrics_std_mode"
    ]
    keys.extend(
        key
        for key in data
        if key not in keys and not key.endswith("_spread") and key != "output_metrics_std_mode"
    )
    if metric_keys is not None:
        return [key for key in metric_keys if key in keys]
    return keys


def format_kv_table(
    data: Dict[str, Any],
    *,
    title: str = "Output metrics",
    metric_keys: Optional[Sequence[str]] = None,
) -> str:
    filtered = _filter_metrics_for_display(data)
    lines = [f"### {title}", "| Metric | Value |", "| --- | --- |"]
    for key in _ordered_keys(filtered, metric_keys):
        lines.append(f"| {format_metric_key(key, filtered)} | {_metric_cell(filtered, key)} |")
    return "\n".join(lines)


def format_kv_table_columns(
    data_by_label: Dict[str, Dict[str, Any]],
    *,
    title: str = "Output metrics",
    reference_label: Optional[str] = None,
    show_deltas: bool = False,
    highlight_best: bool = False,
    metric_keys: Optional[Sequence[str]] = None,
) -> str:
    labels = list(data_by_label)
    if not labels:
        return format_kv_table({}, title=title, metric_keys=metric_keys)
    filtered = {label: _filter_metrics_for_display(data_by_label[label]) for label in labels}
    all_data: Dict[str, Any] = {}
    for label in labels:
        all_data.update(filtered[label])
    keys = _ordered_keys(all_data, metric_keys)
    header_labels = [f"{label} (ref)" if label == reference_label else label for label in labels]
    lines = [
        f"### {title}",
        "| Metric | " + " | ".join(header_labels) + " |",
        "| --- | " + " | ".join(["---"] * len(labels)) + " |",
    ]
    for key in keys:
        row: list[str] = []
        numeric_by_label: Dict[str, float] = {}
        for label in labels:
            try:
                numeric_by_label[label] = float(filtered[label].get(key))
            except Exception:
                pass
        best_label = None
        if highlight_best and reference_label in numeric_by_label:
            choices = {k: v for k, v in numeric_by_label.items() if k != reference_label}
            if choices:
                ref = numeric_by_label[reference_label]
                best_label = min(choices, key=lambda label: abs(choices[label] - ref))
        for label in labels:
            cell = _metric_cell(filtered[label], key)
            if show_deltas and label != reference_label and reference_label in numeric_by_label and label in numeric_by_label:
                ref = numeric_by_label[reference_label]
                delta = numeric_by_label[label] - ref
                delta_text = f"{delta:+.3g}"
                if ref != 0:
                    delta_text += f" ({(delta / ref) * 100.0:+.3g}%)"
                cell = f"{cell} (Delta={delta_text})"
            if label == best_label:
                cell = f"**{cell}**"
            row.append(cell)
        lines.append(f"| {format_metric_key(key, filtered[labels[0]])} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_output_metrics_tables(
    metrics: Dict[str, Any],
    *,
    title: str = "Output metrics",
    show_params: bool = True,
    metric_keys: Optional[Sequence[str]] = None,
) -> str:
    params, values = split_output_metrics(metrics)
    parts: list[str] = []
    if values:
        parts.append(format_kv_table(values, title=title, metric_keys=metric_keys))
    if show_params and params:
        parts.append(format_kv_table(params, title=f"{title} (params)"))
    return "\n\n".join(parts)


def format_output_metrics_tables_columns(
    data_by_label: Dict[str, Dict[str, Any]],
    *,
    title: str = "Output metrics",
    show_params: bool = True,
    reference_label: Optional[str] = None,
    show_deltas: bool = False,
    highlight_best: bool = False,
    metric_keys: Optional[Sequence[str]] = None,
) -> str:
    params_by_label, values_by_label = split_output_metrics_columns(data_by_label)
    parts: list[str] = []
    if any(values_by_label.values()):
        parts.append(
            format_kv_table_columns(
                values_by_label,
                title=title,
                reference_label=reference_label,
                show_deltas=show_deltas,
                highlight_best=highlight_best,
                metric_keys=metric_keys,
            )
        )
    if show_params and any(params_by_label.values()):
        parts.append(format_kv_table_columns(params_by_label, title=f"{title} (params)"))
    return "\n\n".join(parts)


def important_metric_keys(settings: Optional[Dict[str, Any]]) -> list[str]:
    raw = (settings or {}).get("output_metrics_important_keys", DEFAULT_IMPORTANT_KEYS)
    if isinstance(raw, str):
        raw = [item.strip() for item in raw.split(",") if item.strip()]
    if not isinstance(raw, (list, tuple)):
        return list(DEFAULT_IMPORTANT_KEYS)
    return [str(key) for key in raw]


def _write_metrics_csv(metrics: Dict[str, Any], path: Path) -> None:
    params, _ = split_output_metrics(metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["section", "metric", "label", "value", "spread", "spread_mode"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in _ordered_keys(metrics, None):
            writer.writerow({
                "section": "parameter" if key in params else "metric",
                "metric": key,
                "label": format_metric_key(key, metrics).replace("**", ""),
                "value": metrics.get(key),
                "spread": metrics.get(f"{key}_spread"),
                "spread_mode": metrics.get("output_metrics_std_mode") if metrics.get(f"{key}_spread") is not None else "",
            })


def save_output_metrics_artifacts(
    metrics: Dict[str, Any],
    run_dir: Union[str, Path],
    settings: Optional[Dict[str, Any]] = None,
    *,
    formats: Optional[Sequence[str]] = None,
    overwrite: bool = True,
) -> Dict[str, Path]:
    """Save the complete dataset plus the configured compact important view."""
    cfg = dict(settings or {})
    raw_formats = formats if formats is not None else cfg.get("output_metrics_save_formats", ["json", "csv", "md"])
    if isinstance(raw_formats, str):
        raw_formats = [part.strip() for part in raw_formats.split(",") if part.strip()]
    normalized = {str(fmt).strip().lower().lstrip(".") for fmt in (raw_formats or [])}
    out_dir = analysis.analysis_dir_for_run(Path(run_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "output_metrics_json": out_dir / "output_metrics.json",
        "output_metrics_csv": out_dir / "output_metrics.csv",
        "output_metrics_important_md": out_dir / "output_metrics_important.md",
    }
    saved: Dict[str, Path] = {}

    def _allowed(path: Path) -> bool:
        return bool(overwrite) or not path.exists()

    if "json" in normalized and _allowed(paths["output_metrics_json"]):
        paths["output_metrics_json"].write_text(
            json.dumps(metrics, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
        saved["output_metrics_json"] = paths["output_metrics_json"]
    if "csv" in normalized and _allowed(paths["output_metrics_csv"]):
        _write_metrics_csv(metrics, paths["output_metrics_csv"])
        saved["output_metrics_csv"] = paths["output_metrics_csv"]
    if "md" in normalized and _allowed(paths["output_metrics_important_md"]):
        table = format_output_metrics_tables(
            metrics,
            title="Important output metrics",
            show_params=bool(cfg.get("output_metrics_show_params", False)),
            metric_keys=important_metric_keys(cfg),
        )
        paths["output_metrics_important_md"].write_text(table + "\n", encoding="utf-8")
        saved["output_metrics_important_md"] = paths["output_metrics_important_md"]
    return saved


def load_saved_output_metrics(run_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    path = analysis.analysis_dir_for_run(Path(run_dir)) / "output_metrics.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
        return payload["metrics"]
    return payload if isinstance(payload, dict) else None


def _saved_metrics_complete(metrics: Dict[str, Any], settings: Dict[str, Any]) -> bool:
    required = {
        "peak_latency_ms",
        "output_metrics_n_trials",
        "auc_raw_hz_s",
        "auc_normalized_s",
    }
    if bool(settings.get("output_stim_spike_metrics_enabled", True)):
        required.update({"stim_spike_count", "stim_response_fraction", "stim_repetitive_fraction"})
    if bool(settings.get("output_first_spike_metric_enabled", True)):
        required.update({"first_spike_latency_ms", "first_spike_contributing_trials"})
    if bool(settings.get("output_isi_metrics_enabled", True)):
        required.update({"first_to_second_isi_ms", "initial_isi_contributing_trials"})
    return required.issubset(metrics)


def load_or_compute_output_metrics(
    run_dir: Union[str, Path],
    *,
    results: Optional[Dict[str, Any]] = None,
    repo_root: Optional[Union[str, Path]] = None,
    preset_path: Optional[Union[str, Path]] = None,
    settings: Optional[Dict[str, Any]] = None,
    prefer_saved: bool = True,
    save: bool = False,
    formats: Optional[Sequence[str]] = None,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """Notebook-friendly saved-first metric loading with compute fallback."""
    run_path = Path(run_dir).expanduser().resolve()
    preset = load_output_metrics_preset(
        repo_root=repo_root,
        preset_path=preset_path,
        overrides=settings,
    )
    cfg = dict(preset.get("config") or {})
    warnings = list(preset.get("warnings") or [])
    metrics = load_saved_output_metrics(run_path) if prefer_saved else None
    used_saved = metrics is not None
    if metrics is not None and not _saved_metrics_complete(metrics, cfg):
        warnings.append("Saved output metrics are incomplete for the current metric set; recalculating.")
        metrics = None
        used_saved = False
    if metrics is None:
        if results is None:
            from modules import run_sim

            results = run_sim.load_results(run_path)
        metrics = compute_output_metrics_from_results(results, cfg)
    saved_paths: Dict[str, Path] = {}
    if save:
        saved_paths = save_output_metrics_artifacts(
            metrics,
            run_path,
            cfg,
            formats=formats,
            overwrite=overwrite,
        )
    return {
        "metrics": metrics,
        "settings": cfg,
        "run_dir": run_path,
        "preset_path": preset.get("preset_path"),
        "used_saved": used_saved,
        "saved_paths": saved_paths,
        "warnings": warnings,
    }


def save_output_metrics_from_results(
    results: Dict[str, Any],
    run_dir: Union[str, Path],
    *,
    preset_path: Optional[Union[str, Path]] = None,
    formats: Optional[Sequence[str]] = None,
    overwrite: bool = True,
) -> Dict[str, Path]:
    """Simulation-time entry point for automatic output-metric persistence."""
    run_path = Path(run_dir)
    repo_root = analysis.resolve_scp_root_for_results(results, run_path)
    preset = load_output_metrics_preset(repo_root=repo_root, preset_path=preset_path)
    for warning in preset.get("warnings") or []:
        print(f"save_output_metrics warning: {warning}")
    settings = dict(preset.get("config") or {})
    metrics = compute_output_metrics_from_results(results, settings)
    return save_output_metrics_artifacts(
        metrics,
        run_path,
        settings,
        formats=formats,
        overwrite=overwrite,
    )


__all__ = [
    "DEFAULT_IMPORTANT_KEYS",
    "OUTPUT_METRIC_PLOT_DEFAULT_KEYS",
    "OUTPUT_METRIC_VALUE_ORDER",
    "OUTPUT_PARAM_KEYS",
    "compute_output_metrics_from_results",
    "compute_trial_output_metrics",
    "extract_spike_trials",
    "format_metric_key",
    "format_output_metrics_tables",
    "format_output_metrics_tables_columns",
    "important_metric_keys",
    "load_or_compute_output_metrics",
    "load_output_metrics_preset",
    "load_saved_output_metrics",
    "merge_metric_settings",
    "save_output_metrics_artifacts",
    "save_output_metrics_from_results",
    "split_output_metrics",
    "split_output_metrics_columns",
]
