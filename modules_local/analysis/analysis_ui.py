from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import textwrap

from modules_local import input_sampling, inputs, run_sim
from . import analysis, plotting

HELP_SELECTION = textwrap.dedent(
    """
    Selection UI
    - Cell/Tune/Model: choose which output_data tree to browse.
    - Compare list: multi-select run folders to plot/compare.
    - Compare paths: comma-separated curve files. Syntax: path@shift:scale;key=val
      Keys: color, label, linestyle, shift, scale (shift in ms, scale is multiplicative).
    - compare_list_dir_paths (defaults JSON): add folders of CSV curves to the Compare list.
    - Use compare paths toggle: enables/disables Compare paths without deleting them.
    - compare_list_paths entries can be objects with {path, enabled, shift_ms, scale, color, label, linestyle}.
    - Tokens: latest/previous/prev/latest-1 resolve to runs in output_data.
    - Save plots/analysis toggles affect saved figures/JSON.
    Defaults live in modules_local/analysis/analysis_defaults.json.
    """
).strip()

HELP_OUTPUTS = textwrap.dedent(
    """
    Outputs UI
    - Uses the current Selection and Compare list/paths.
    - Plot window, stim start/stop are in ms.
    - Curve mode: raw vs normalized; Norm mode: avg/peak; Norm FR optional override.
    - Curve bin/smooth controls binned rate and moving-average window.
    - Compare layout: overlay/stacked/side-by-side; Shade adds std/sem band.
    - Paper compare toggles presets from analysis_presets/paper_compare.json.
    """
).strip()

HELP_INPUTS = textwrap.dedent(
    """
    Inputs UI
    - Input source: auto (saved spikes if present, else stats), saved, or stats.
    - Std mode: std or sem for shaded bands.
    - Groups: comma-separated synapse groups; leave blank for all.
    - Bin/smooth apply to input rate curves; raster settings affect raster plots.
    - Compare layout controls multi-run plotting.
    - Legend: matplotlib location for input legends (e.g., best, upper left, none).
    """
).strip()

HELP_EXTRA = textwrap.dedent(
    """
    Extra UI
    - Output metrics: summary table for selected runs/curves.
    - Compare configs: diff cell/geom/syn configs across runs.
    - Compare outputs/inputs: reuse plot logic with the current selection.
    - Input sampling: synthesize input curves from synapse configs.
    - Snapshot compare: compare notebook vs slurm outputs.
    """
).strip()


def _print_help(out, text: str) -> None:
    if out is None:
        print(text)
        return
    with out:
        out.clear_output()
        print(text)


def compare_enabled(selection: Dict[str, Any]) -> bool:
    list_entries = _compare_list_entries(selection)
    if len(list_entries) >= 2:
        return True
    run_a_path = selection.get("run_a_path")
    run_b_path = selection.get("run_b_path")
    return bool(run_a_path and run_b_path)


def _coerce_run_path(path_val: Any, base_dir: Path) -> Optional[Path]:
    if path_val in (None, "", "none", "None"):
        return None
    token = str(path_val)
    if token in ("latest", "previous", "prev", "latest-1"):
        try:
            return analysis.resolve_run(base_dir, token)
        except Exception:
            pass
    p = Path(str(path_val)).expanduser()
    if not p.is_absolute():
        repo_root = analysis.find_scp_root(base_dir)
        if p.parts and p.parts[0] in ("external_data", "cells"):
            p = (repo_root / p).resolve()
        else:
            p = (base_dir / p).resolve()
    return p


def _safe_load_results(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return run_sim.load_results(path)
    except FileNotFoundError:
        return None


def _is_curve_path(path: Path) -> bool:
    return path.suffix.lower() in (".csv", ".tsv", ".txt")


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
    run_a = run_a_path if run_a_path is not None else analysis.resolve_run(base_dir, selection["run_a"])
    run_b = run_b_path if run_b_path is not None else analysis.resolve_run(base_dir, selection["run_b"])
    res_a = None if (run_a is not None and _is_curve_path(Path(run_a))) else run_sim.load_results(run_a)
    res_b = None if (run_b is not None and _is_curve_path(Path(run_b))) else run_sim.load_results(run_b)
    return run_a, run_b, res_a, res_b


def _save_fig(fig, out_path, *, enabled: bool, dpi: int) -> None:
    analysis.save_figure(fig, out_path, enabled=enabled, dpi=dpi)


def _save_json(data: dict, out_path, *, enabled: bool) -> None:
    analysis.save_json(data, out_path, enabled=enabled)


def _stim_window(sim_cfg: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
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


def _stim_window_for_opts(sim_cfg: Dict[str, Any], opts: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    stim_start, stim_stop = _stim_window(sim_cfg)
    override_start = opts.get("output_stim_start_ms")
    override_stop = opts.get("output_stim_stop_ms")
    if override_start is not None:
        try:
            stim_start = float(override_start)
        except Exception:
            pass
    if override_stop is not None:
        try:
            stim_stop = float(override_stop)
        except Exception:
            pass
    return stim_start, stim_stop


def _plot_metric_points(
    ax,
    metrics: Dict[str, Any],
    *,
    color: Optional[str] = None,
    label_prefix: Optional[str] = None,
    show_labels: bool = False,
    size: float = 36.0,
) -> None:
    if ax is None or not metrics:
        return
    entries = [
        ("peak_time_ms", "peak_value", "o", "Peak"),
        ("drop_time_ms", "drop_value", "v", "+100ms"),
        ("rebound_time_ms", "rebound_value", "^", "+300ms"),
    ]
    any_label = False
    for t_key, y_key, marker, label in entries:
        t_val = metrics.get(t_key)
        y_val = metrics.get(y_key)
        if t_val is None or y_val is None:
            continue
        lab = None
        if show_labels:
            lab = f"{label_prefix} {label}" if label_prefix else label
            any_label = True
        ax.scatter(
            [t_val],
            [y_val],
            s=size,
            marker=marker,
            color=color,
            edgecolor="k",
            linewidth=0.6,
            zorder=5,
            label=lab,
        )
    if show_labels and any_label:
        ax.legend()


def _apply_output_norm(curve: Optional[Dict[str, Any]], output_norm: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not curve or not output_norm:
        return curve
    rate = np.asarray(curve.get("rate_hz", []) or [], dtype=float)
    baseline_mean = output_norm.get("baseline_mean")
    norm_scale = output_norm.get("norm_scale")
    rate_bs = rate
    if baseline_mean is not None:
        rate_bs = rate - float(baseline_mean)
        rate = rate_bs
    if norm_scale not in (None, 0):
        rate = rate / float(norm_scale)
    updated = dict(curve)
    updated["rate_hz"] = rate.tolist()
    updated["units"] = "normalized"
    updated["normalized"] = True
    if baseline_mean is not None:
        updated["baseline_mean"] = float(baseline_mean)
        updated["baseline_ms"] = output_norm.get("baseline_ms")
        updated["baseline_mode"] = output_norm.get("baseline_mode")
        updated["baseline_time_ms"] = output_norm.get("baseline_time_ms")
        updated["baseline_subtracted"] = True
        updated["rate_hz_baseline_sub"] = rate_bs.tolist()
    updated["norm_scale"] = norm_scale
    if output_norm.get("norm_mode") is not None:
        updated["norm_mode"] = output_norm.get("norm_mode")
    if output_norm.get("norm_window") is not None:
        updated["norm_window"] = output_norm.get("norm_window")
    return updated


def _parse_plot_scale(scale: Any) -> float:
    if scale is None:
        return 1.0
    try:
        return float(scale)
    except Exception:
        return 1.0


def _curve_smooth_ms(opts: Dict[str, Any]) -> Optional[float]:
    smooth = opts.get("win_size")
    if smooth is None:
        smooth = opts.get("output_smooth_ms")
    return smooth


def _unique_compare_colors(n: int) -> list:
    if n <= 0:
        return []
    cmap = plt.cm.tab20 if n <= 20 else plt.cm.hsv
    vals = np.linspace(0.0, 1.0, n, endpoint=False)
    colors = [cmap(v) for v in vals]
    rng = np.random.default_rng()
    rng.shuffle(colors)
    return colors


def _scale_curve_for_plot(curve: Optional[Dict[str, Any]], scale: Any) -> Optional[Dict[str, Any]]:
    if not curve:
        return curve
    scale_val = _parse_plot_scale(scale)
    if scale_val == 1.0:
        return curve
    updated = dict(curve)
    rate = np.asarray(curve.get("rate_hz", []) or [], dtype=float) * scale_val
    updated["rate_hz"] = rate.tolist()
    if curve.get("rate_hz_baseline_sub") is not None:
        rate_bs = np.asarray(curve.get("rate_hz_baseline_sub") or [], dtype=float) * scale_val
        updated["rate_hz_baseline_sub"] = rate_bs.tolist()
    return updated


def _scale_metrics_for_plot(metrics: Optional[Dict[str, Any]], scale: Any) -> Optional[Dict[str, Any]]:
    if not metrics:
        return metrics
    scale_val = _parse_plot_scale(scale)
    if scale_val == 1.0:
        return metrics
    scaled = dict(metrics)
    for key in (
        "peak_value",
        "peak_rate_hz",
        "drop_value",
        "rebound_value",
        "peak_value_raw",
        "peak_rate_hz_raw",
    ):
        val = scaled.get(key)
        if val is not None:
            try:
                scaled[key] = float(val) * scale_val
            except Exception:
                pass
    return scaled


def _plot_metric_window_markers(
    ax,
    metrics: Optional[Dict[str, Any]],
    *,
    color: Optional[str],
    alpha: float = 0.15,
    linewidth: float = 1.0,
) -> None:
    if not metrics or ax is None:
        return
    def _vline(x, ls="--"):
        if x is None:
            return
        ax.axvline(float(x), color=color or "0.4", linestyle=ls, linewidth=linewidth)

    _vline(metrics.get("baseline_time_ms"))
    _vline(metrics.get("drop_center_ms"))
    _vline(metrics.get("rebound_center_ms"))

    for key_start, key_stop in (
        ("baseline_window_start_ms", "baseline_window_stop_ms"),
        ("drop_window_start_ms", "drop_window_stop_ms"),
        ("rebound_window_start_ms", "rebound_window_stop_ms"),
    ):
        start = metrics.get(key_start)
        stop = metrics.get(key_stop)
        if start is None or stop is None:
            continue
        ax.axvspan(float(start), float(stop), color=color or "0.4", alpha=alpha, linewidth=0)


def _metric_window_kwargs(opts: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "alpha": float(opts.get("output_metric_window_alpha", 0.15)),
        "linewidth": float(opts.get("output_metric_linewidth", 1.0)),
    }


def _compute_output_metrics(
    curve: Optional[Dict[str, Any]],
    sim_cfg: Dict[str, Any],
    opts: Dict[str, Any],
    **overrides: Any,
) -> Dict[str, Any]:
    def _pick(name: str, opt_key: str, default: Any) -> Any:
        if name in overrides:
            return overrides[name]
        return opts.get(opt_key, default)

    return analysis.compute_output_metrics(
        curve or {},
        sim_cfg or {},
        peak_window_ms=_pick("peak_window_ms", "output_peak_window_ms", 100.0),
        drop_window_ms=_pick("drop_window_ms", "output_drop_window_ms", 100.0),
        rebound_window_ms=_pick("rebound_window_ms", "output_rebound_window_ms", 300.0),
        auc_window=_pick("auc_window", "output_auc_window", "stim"),
        pdp_mode=_pick("pdp_mode", "output_metric_mode", "point"),
        pdp_window_ms=_pick("pdp_window_ms", "output_metric_window_ms", 0.0),
        baseline_ms=_pick("baseline_ms", "output_metric_window_ms", 100.0),
        baseline_mode=_pick("baseline_mode", "output_metric_mode", "point"),
        baseline_center_ms=_pick("baseline_center_ms", "output_baseline_center_ms", None),
        stim_start_ms=_pick("stim_start_ms", "output_stim_start_ms", None),
        stim_stop_ms=_pick("stim_stop_ms", "output_stim_stop_ms", None),
    )


def _smooth_curve_if_requested(
    curve: Optional[Dict[str, Any]],
    *,
    bin_ms: Optional[float],
    smooth_ms: Optional[float],
    smooth_mode: str,
) -> Optional[Dict[str, Any]]:
    if not curve:
        return curve
    t_ms = np.asarray(curve.get("t_ms", []) or [], dtype=float)
    rate = np.asarray(curve.get("rate_hz", []) or [], dtype=float)
    if t_ms.size < 2 or rate.size < 2:
        return curve
    use_bin_ms = bin_ms
    if use_bin_ms is None:
        diffs = np.diff(t_ms)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            return curve
        use_bin_ms = float(np.median(diffs))
    centers, y = analysis._smooth_rate_curve(
        t_ms,
        rate,
        float(use_bin_ms),
        smooth_ms,
        mode=str(smooth_mode or "center").lower(),
    )
    updated = dict(curve)
    updated["t_ms"] = centers.tolist()
    updated["rate_hz"] = y.tolist()
    return updated


def _default_sim_cfg_for_curve(
    curve: Dict[str, Any],
    *,
    fallback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sim_cfg: Dict[str, Any] = dict(fallback or {})
    t_ms = np.asarray(curve.get("t_ms", []) or [], dtype=float)
    if t_ms.size:
        sim_cfg.setdefault("tstart", float(np.min(t_ms)))
        sim_cfg.setdefault("tstop", float(np.max(t_ms)))
    sim_cfg.setdefault("stim_start_ms", 0.0)
    return sim_cfg


def _load_curve_from_path(
    path: Path,
    opts: Dict[str, Any],
    *,
    fallback_sim_cfg: Optional[Dict[str, Any]] = None,
    shift_ms: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    curve = analysis.load_scatter_curve_optional(
        enabled=True,
        path=str(path),
        time_unit=opts.get("output_scatter_time_unit", "s"),
        shift_ms=shift_ms if shift_ms is not None else opts.get("output_scatter_shift_ms"),
        quiet=True,
    )
    if not curve:
        return None
    curve = _smooth_curve_if_requested(
        curve,
        bin_ms=opts.get("output_bin_ms"),
        smooth_ms=_curve_smooth_ms(opts),
        smooth_mode=opts.get("output_smooth_mode", "causal"),
    )
    if opts.get("output_curve_mode", "raw") == "normalized":
        sim_cfg = _default_sim_cfg_for_curve(curve, fallback=fallback_sim_cfg)
        if opts.get("output_stim_start_ms") is not None:
            sim_cfg["stim_start_ms"] = float(opts.get("output_stim_start_ms"))
        if opts.get("output_stim_stop_ms") is not None:
            sim_cfg["stim_stop_ms"] = float(opts.get("output_stim_stop_ms"))
        curve = analysis.normalize_output_curve(
            curve,
            sim_cfg,
            mode="normalized",
            norm_mode=opts.get("output_norm_mode", "avg"),
            baseline_ms=opts.get("output_metric_window_ms", 100.0),
            baseline_mode=opts.get("output_metric_mode", "point"),
            baseline_center_ms=opts.get("output_baseline_center_ms"),
            norm_window=opts.get("output_norm_window", "stim"),
        )
    return curve


def _curve_from_xy(t_ms: np.ndarray, rate: np.ndarray, *, units: str = "Hz") -> Dict[str, Any]:
    return {
        "t_ms": np.asarray(t_ms, dtype=float).tolist(),
        "rate_hz": np.asarray(rate, dtype=float).tolist(),
        "units": units,
    }


def _parse_figsize(value: Any) -> Optional[tuple[float, float]]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except Exception:
            return None
    return None


def _split_path_and_shift(item: str) -> Tuple[str, Optional[float], Optional[float]]:
    if "@" not in item:
        return item.strip(), None, None
    path_part, spec = item.split("@", 1)
    spec = spec.strip()
    if ";" in spec:
        spec = spec.split(";", 1)[0].strip()
    if not spec:
        return path_part.strip(), None, None
    shift_part = spec
    scale_part = None
    if "," in spec:
        shift_part, scale_part = spec.split(",", 1)
    elif ":" in spec:
        shift_part, scale_part = spec.split(":", 1)
    shift_part = shift_part.strip()
    if shift_part.lower().endswith("ms"):
        shift_part = shift_part[:-2].strip()
    try:
        shift_val = float(shift_part) if shift_part else None
    except Exception:
        shift_val = None
    scale_val = None
    if scale_part is not None:
        scale_part = scale_part.strip()
        if scale_part.lower().endswith("x"):
            scale_part = scale_part[:-1].strip()
        try:
            scale_val = float(scale_part) if scale_part else None
        except Exception:
            scale_val = None
    return path_part.strip(), shift_val, scale_val


def _parse_shift_value(text: str) -> Optional[float]:
    val = text.strip()
    if val.lower().endswith("ms"):
        val = val[:-2].strip()
    try:
        return float(val) if val else None
    except Exception:
        return None


def _parse_scale_value(text: str) -> Optional[float]:
    val = text.strip()
    if val.lower().endswith("x"):
        val = val[:-1].strip()
    try:
        return float(val) if val else None
    except Exception:
        return None


def _parse_compare_list_item_str(raw: str) -> Dict[str, Any]:
    parts = [p.strip() for p in str(raw).split(";") if p.strip()]
    base = parts[0] if parts else str(raw).strip()
    path_part, shift_val, scale_val = _split_path_and_shift(base)
    color = None
    label = None
    linestyle = None
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip().lower()
        val = val.strip()
        if not val:
            continue
        if key in ("color", "colour", "c"):
            color = val
        elif key in ("scale", "scale_x", "gain", "s"):
            scale_val = _parse_scale_value(val)
        elif key in ("shift", "shift_ms", "offset", "tshift"):
            shift_val = _parse_shift_value(val)
        elif key in ("label", "name"):
            label = val
        elif key in ("linestyle", "ls", "style"):
            linestyle = val
    return {
        "path": path_part.strip(),
        "shift_ms": shift_val,
        "scale": scale_val,
        "color": color,
        "label": label,
        "linestyle": linestyle,
    }


def _parse_compare_list_item(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        base = raw.get("path") or raw.get("spec") or raw.get("entry") or ""
        spec = _parse_compare_list_item_str(str(base)) if base else _parse_compare_list_item_str("")
        if raw.get("shift_ms") is not None or raw.get("shift") is not None:
            spec["shift_ms"] = raw.get("shift_ms", raw.get("shift"))
        if raw.get("scale") is not None or raw.get("gain") is not None:
            spec["scale"] = raw.get("scale", raw.get("gain"))
        if raw.get("color") is not None:
            spec["color"] = raw.get("color")
        label_val = raw.get("label") if raw.get("label") is not None else raw.get("name")
        if label_val is not None:
            spec["label"] = label_val
        if raw.get("linestyle") is not None:
            spec["linestyle"] = raw.get("linestyle")
        spec["path"] = str(spec.get("path") or base).strip()
        return spec
    return _parse_compare_list_item_str(str(raw))


def _compare_entry_enabled(entry: Any) -> bool:
    if isinstance(entry, dict):
        return bool(entry.get("enabled", True))
    return True


def _compare_list_paths_text(entries: list[Any]) -> str:
    parts: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            if not _compare_entry_enabled(entry):
                continue
            spec = _parse_compare_list_item(entry)
            path = spec.get("path")
            if not path:
                continue
            token = str(path)
            shift_val = spec.get("shift_ms")
            scale_val = spec.get("scale")
            if shift_val is not None or scale_val is not None:
                shift_str = "" if shift_val is None else str(shift_val)
                scale_str = "" if scale_val is None else str(scale_val)
                if scale_str:
                    token = f"{token}@{shift_str}:{scale_str}"
                else:
                    token = f"{token}@{shift_str}"
            if spec.get("color"):
                token += f";color={spec['color']}"
            if spec.get("label"):
                token += f";label={spec['label']}"
            if spec.get("linestyle"):
                token += f";linestyle={spec['linestyle']}"
            parts.append(token)
        else:
            parts.append(str(entry))
    return ",".join(parts)


def _compare_list_dir_options(g: Dict[str, Any], base_dir: Optional[Path]) -> list[tuple[str, str]]:
    raw_dirs = g.get("compare_list_dir_paths") or []
    if isinstance(raw_dirs, (str, Path)):
        raw_dirs = [raw_dirs]
    root = analysis.find_scp_root(base_dir or Path.cwd())
    options: list[tuple[str, str]] = []
    for raw in raw_dirs:
        if raw in (None, "", "none", "None"):
            continue
        dir_path = Path(str(raw)).expanduser()
        if not dir_path.is_absolute():
            dir_path = (root / dir_path).resolve()
        if not dir_path.is_dir():
            continue
        try:
            files = sorted([p for p in dir_path.iterdir() if p.is_file()])
        except Exception:
            continue
        for file_path in files:
            if file_path.suffix.lower() not in (".csv", ".tsv", ".txt"):
                continue
            label = f"{dir_path.name}/{file_path.name}"
            options.append((label, str(file_path)))
    return options


def _read_compare_preset(path_val: Any, base_dir: Optional[Path]) -> tuple[Optional[Path], Optional[Any]]:
    if path_val in (None, "", "none", "None"):
        return None, None
    preset_path = Path(str(path_val)).expanduser()
    if not preset_path.is_absolute():
        repo_root = analysis.find_scp_root(base_dir or Path.cwd())
        preset_path = (repo_root / preset_path).resolve()
    if not preset_path.exists():
        print(f"Compare preset not found: {preset_path}")
        return None, None
    try:
        payload = json.loads(preset_path.read_text())
    except Exception:
        print(f"Compare preset unreadable: {preset_path}")
        return preset_path, None
    return preset_path, payload


def _load_compare_preset(path_val: Any, base_dir: Optional[Path]) -> list[Dict[str, Any]]:
    preset_path, payload = _read_compare_preset(path_val, base_dir)
    if payload is None:
        return []
    entries = payload.get("entries") if isinstance(payload, dict) else payload if isinstance(payload, list) else []
    if not isinstance(entries, list):
        return []
    out: list[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("enabled", True) is False:
            continue
        path_raw = entry.get("path") or entry.get("run") or entry.get("file")
        if not path_raw:
            continue
        path = _coerce_run_path(path_raw, base_dir or (preset_path.parent if preset_path else Path.cwd()))
        if path is None:
            continue
        if not path.exists():
            print(f"Skipping missing preset path: {path}")
            continue
        out.append({
            "path": path,
            "shift_ms": entry.get("shift_ms") if entry.get("shift_ms") is not None else entry.get("shift"),
            "scale": entry.get("scale"),
            "color": entry.get("color"),
            "linestyle": entry.get("linestyle") or entry.get("ls"),
            "label": entry.get("label"),
        })
    return out


def _load_compare_preset_defaults(path_val: Any, base_dir: Optional[Path]) -> Dict[str, Any]:
    _, payload = _read_compare_preset(path_val, base_dir)
    if not isinstance(payload, dict):
        return {}
    defaults = payload.get("defaults")
    return defaults if isinstance(defaults, dict) else {}


def _is_missing_default(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, tuple)):
        if not value:
            return True
        return all(v is None or (isinstance(v, str) and not v.strip()) for v in value)
    return False


def _merge_preset_defaults(opts: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    if not defaults:
        return opts
    merged = dict(opts)
    for key, val in defaults.items():
        if val is None:
            continue
        if _is_missing_default(merged.get(key)):
            merged[key] = val
    return merged


def _parse_compare_list_paths(text: str) -> list[str]:
    if not text:
        return []

    entries: list[str] = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if not parts:
            continue
        for part in parts:
            entries.append(part)
    return entries


def _compare_list_entries(selection: Dict[str, Any]) -> list[Any]:
    entries: list[Any] = []
    entries.extend(selection.get("compare_list") or [])
    entries.extend(selection.get("compare_list_paths") or [])
    entries = [entry for entry in entries if _compare_entry_enabled(entry)]
    # de-dupe while preserving order (by path + shift + scale + extras)
    seen = set()
    out: list[Any] = []
    for item in entries:
        spec = _parse_compare_list_item(item)
        key = (
            spec.get("path"),
            spec.get("shift_ms"),
            spec.get("scale"),
            spec.get("color"),
            spec.get("label"),
            spec.get("linestyle"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _compare_list_run_paths(selection: Dict[str, Any]) -> list[Path]:
    base_dir = selection.get("base")
    if base_dir is None:
        return []
    entries = _compare_list_entries(selection)
    if not entries:
        return []
    seen: set[Path] = set()
    out: list[Path] = []
    for item in entries:
        spec = _parse_compare_list_item(item)
        path_raw = spec.get("path")
        path = _coerce_run_path(path_raw, base_dir)
        if path is None or not path.exists():
            continue
        if _is_curve_path(path):
            continue
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def run_output_plots(
    selection: Dict[str, Any],
    opts: Dict[str, Any],
    *,
    save_plots: bool = False,
    save_analysis: bool = False,
    plots_dpi: int = 150,
) -> None:
    output_scale = 1.0
    external_scale = 1.0
    base_dir = selection.get("base")
    preset_defaults = _load_compare_preset_defaults(selection.get("compare_preset_path"), base_dir)
    if preset_defaults:
        opts = _merge_preset_defaults(opts, preset_defaults)
    preset_entries = _load_compare_preset(selection.get("compare_preset_path"), base_dir)
    list_entries = _compare_list_entries(selection) if not preset_entries else []
    if preset_entries or list_entries:
        paths: list[Path] = []
        shifts: list[Optional[float]] = []
        scales: list[Optional[float]] = []
        labels_override: list[Optional[str]] = []
        colors_override: list[Optional[str]] = []
        linestyles_override: list[Optional[str]] = []
        paths_from_compare_list: list[bool] = []
        has_curve = False
        if preset_entries:
            for entry in preset_entries:
                path = entry["path"]
                shift_ms = entry.get("shift_ms")
                scale_val = entry.get("scale")
                if _is_curve_path(path):
                    has_curve = True
                paths.append(path)
                shifts.append(shift_ms)
                scales.append(scale_val)
                labels_override.append(entry.get("label"))
                colors_override.append(entry.get("color"))
                linestyles_override.append(entry.get("linestyle"))
                paths_from_compare_list.append(False)
        else:
            compare_list_set = {str(v) for v in (selection.get("compare_list") or [])}
            for item in list_entries:
                spec = _parse_compare_list_item(item)
                path_raw = spec.get("path")
                shift_ms = spec.get("shift_ms")
                scale_val = spec.get("scale")
                path = _coerce_run_path(path_raw, base_dir)
                if path is None:
                    continue
                if not path.exists():
                    print(f"Skipping missing path: {path}")
                    continue
                paths.append(path)
                shifts.append(shift_ms)
                scales.append(scale_val)
                labels_override.append(spec.get("label"))
                colors_override.append(spec.get("color"))
                linestyles_override.append(spec.get("linestyle"))
                from_compare_list = False if isinstance(item, dict) else str(item) in compare_list_set
                paths_from_compare_list.append(from_compare_list)
                if _is_curve_path(path):
                    has_curve = True

        fallback_sim_cfg = None
        for path in paths:
            if not _is_curve_path(path):
                res = _safe_load_results(path)
                fallback_sim_cfg = (res.get("sim_cfg") or {}) if res else None
                break

        if len(paths) == 1 and not has_curve and not preset_entries:
            selection["run_single"] = paths[0]
            selection["compare_list"] = []
            selection["compare_list_paths"] = []
        else:
            curves = []
            labels = []
            colors = []
            linestyles = []
            sim_cfgs = []
            curve_paths = []
            curve_scales = []
            curve_from_compare_list: list[bool] = []
            for idx, (path, shift_ms, scale_val) in enumerate(zip(paths, shifts, scales)):
                if _is_curve_path(path):
                    curve = _load_curve_from_path(path, opts, shift_ms=shift_ms, fallback_sim_cfg=fallback_sim_cfg)
                    if not curve:
                        continue
                    label_override = labels_override[idx] if idx < len(labels_override) else None
                    color_override = colors_override[idx] if idx < len(colors_override) else None
                    linestyle_override = linestyles_override[idx] if idx < len(linestyles_override) else None
                    curves.append(curve)
                    labels.append(label_override or Path(path).stem)
                    colors.append(color_override)
                    linestyles.append(linestyle_override)
                    sim_cfgs.append(_default_sim_cfg_for_curve(curve, fallback=fallback_sim_cfg))
                    curve_paths.append(path)
                    curve_scales.append(scale_val)
                    curve_from_compare_list.append(paths_from_compare_list[idx])
                else:
                    res = _safe_load_results(path)
                    if res is None:
                        print(f"Skipping missing run: {path}")
                        continue
                    curve = analysis.compute_output_curve_from_results(
                        res,
                        bin_ms=opts.get("output_bin_ms"),
                        smooth_ms=_curve_smooth_ms(opts),
                        smooth_mode=opts.get("output_smooth_mode", "causal"),
                    )
                    if not curve:
                        continue
                    curve = analysis.normalize_output_curve(
                        curve,
                        res.get("sim_cfg", {}) or {},
                        mode=opts.get("output_curve_mode", "raw"),
                        norm_mode=opts.get("output_norm_mode", "avg"),
                        baseline_ms=opts.get("output_metric_window_ms", 100.0),
                        baseline_mode=opts.get("output_metric_mode", "point"),
                        baseline_center_ms=opts.get("output_baseline_center_ms"),
                        norm_window=opts.get("output_norm_window", "stim"),
                    )
                    if shift_ms is not None:
                        t_ms = np.asarray(curve.get("t_ms", []) or [], dtype=float) + float(shift_ms)
                        curve = dict(curve)
                        curve["t_ms"] = t_ms.tolist()
                    label_override = labels_override[idx] if idx < len(labels_override) else None
                    color_override = colors_override[idx] if idx < len(colors_override) else None
                    linestyle_override = linestyles_override[idx] if idx < len(linestyles_override) else None
                    curves.append(curve)
                    labels.append(label_override or analysis.run_label(path))
                    colors.append(color_override or (res.get("sim_cfg", {}) or {}).get("color", None))
                    linestyles.append(linestyle_override)
                    sim_cfgs.append(res.get("sim_cfg", {}) or {})
                    curve_paths.append(path)
                    curve_scales.append(scale_val)
                    curve_from_compare_list.append(paths_from_compare_list[idx])

            if not curves:
                print("No valid curves found in compare list.")
                return

            run_indices = [i for i, p in enumerate(curve_paths) if not _is_curve_path(p)]
            selected_run_indices = [
                i for i in run_indices
                if i < len(curve_from_compare_list) and curve_from_compare_list[i]
            ]
            if len(selected_run_indices) >= 2:
                seen_colors: Dict[Any, int] = {}
                override_indices: list[int] = []
                for idx in selected_run_indices:
                    col = colors[idx]
                    if col is None:
                        override_indices.append(idx)
                        continue
                    if col in seen_colors:
                        override_indices.append(idx)
                        override_indices.append(seen_colors[col])
                    else:
                        seen_colors[col] = idx
                if override_indices:
                    uniq_indices = sorted(set(override_indices))
                    rand_colors = _unique_compare_colors(len(uniq_indices))
                    for idx, col in zip(uniq_indices, rand_colors):
                        colors[idx] = col

            layout = (opts.get("compare_output_layout") or "overlay").lower()
            overlay_layouts = {"overlay", "same", "same-plot", "overlap"}
            stacked_layouts = {"stacked", "top-bottom", "vertical"}
            side_layouts = {"side-by-side", "side_by_side", "horizontal"}
            compare_title = "Output curves"
            compare_figsize = _parse_figsize(opts.get("output_compare_figsize"))
            compare_panel = _parse_figsize(opts.get("output_compare_panel_size"))

            stim_start, stim_stop = _stim_window_for_opts(sim_cfgs[0] or {}, opts)
            fig_curve = None
            ax = None
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            y_label = "Rate (Hz)" if (curves[0].get("units", "Hz") == "Hz") else "Rate (normalized)"
            if layout in stacked_layouts or layout in side_layouts:
                n = len(curves)
                if layout in stacked_layouts:
                    if compare_panel:
                        panel_w, panel_h = compare_panel
                        fig_curve, axes = plt.subplots(
                            n, 1, figsize=(panel_w, max(panel_h, panel_h * n)), sharex=True
                        )
                    else:
                        fig_curve, axes = plt.subplots(n, 1, figsize=(6, max(3.2, 3.2 * n)), sharex=True)
                else:
                    if compare_panel:
                        panel_w, panel_h = compare_panel
                        fig_curve, axes = plt.subplots(
                            1, n, figsize=(max(panel_w, panel_w * n), panel_h), sharey=True
                        )
                    else:
                        fig_curve, axes = plt.subplots(1, n, figsize=(max(4.0, 4.2 * n), 4), sharey=True)
                axes = np.atleast_1d(axes)
                for idx, (curve, label) in enumerate(zip(curves, labels)):
                    ax_i = axes[idx] if idx < axes.size else axes[-1]
                    scale_val = curve_scales[idx] if curve_scales[idx] is not None else (external_scale if _is_curve_path(curve_paths[idx]) else output_scale)
                    curve_plot = _scale_curve_for_plot(curve, scale_val)
                    t_ms = np.asarray(curve_plot.get("t_ms", []) or [], dtype=float)
                    y = np.asarray(curve_plot.get("rate_hz", []) or [], dtype=float)
                    col = colors[idx] if colors[idx] is not None else color_cycle[idx % len(color_cycle)]
                    ls = linestyles[idx] if idx < len(linestyles) and linestyles[idx] else "-"
                    ax_i.plot(t_ms, y, lw=float(opts.get("output_linewidth", 2.0)), color=col, linestyle=ls, label=label)
                    stim_start_i, stim_stop_i = _stim_window_for_opts(sim_cfgs[idx] or {}, opts)
                    if opts.get("output_show_metric_points", True):
                        metrics = _compute_output_metrics(
                            curve,
                            sim_cfgs[idx] or {},
                            opts,
                            peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                            drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                            rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                            auc_window=opts.get("output_auc_window", "stim"),
                            stim_start_ms=stim_start_i,
                            stim_stop_ms=stim_stop_i,
                        )
                        metrics_plot = _scale_metrics_for_plot(metrics, scale_val)
                        _plot_metric_points(
                            ax_i,
                            metrics_plot,
                            color=col,
                            label_prefix=label,
                            show_labels=bool(opts.get("output_metric_label_points", False)),
                            size=float(opts.get("output_metric_marker_size", 36.0)),
                        )
                        if opts.get("output_metric_window_markers", False):
                            _plot_metric_window_markers(ax_i, metrics_plot, color=col, **_metric_window_kwargs(opts))
                    if stim_start_i is not None:
                        ax_i.axvline(float(stim_start_i), color="k", linestyle="-", linewidth=float(opts.get("output_stim_linewidth", 1.0)))
                    if stim_stop_i is not None:
                        ax_i.axvline(float(stim_stop_i), color="k", linestyle="-", linewidth=float(opts.get("output_stim_linewidth", 1.0)))
                    if opts.get("plot_window") is not None:
                        ax_i.set_xlim(opts.get("plot_window")[0], opts.get("plot_window")[1])
                    ax_i.set_title(label)
                    if label:
                        ax_i.legend()
                    ax_i.grid(True)
                    if layout in side_layouts:
                        ax_i.set_xlabel("Time (ms)")
                        if idx == 0:
                            ax_i.set_ylabel(y_label)
                    else:
                        ax_i.set_ylabel(y_label)
                        if idx == axes.size - 1:
                            ax_i.set_xlabel("Time (ms)")
                if fig_curve is not None:
                    fig_curve.suptitle(compare_title)
                    plt.tight_layout(rect=[0, 0, 1, 0.96])
            else:
                fig_curve, ax = plt.subplots(figsize=compare_figsize or (6, 4))
                for idx, (curve, label) in enumerate(zip(curves, labels)):
                    scale_val = curve_scales[idx] if curve_scales[idx] is not None else (external_scale if _is_curve_path(curve_paths[idx]) else output_scale)
                    curve_plot = _scale_curve_for_plot(curve, scale_val)
                    t_ms = np.asarray(curve_plot.get("t_ms", []) or [], dtype=float)
                    y = np.asarray(curve_plot.get("rate_hz", []) or [], dtype=float)
                    col = colors[idx] if colors[idx] is not None else color_cycle[idx % len(color_cycle)]
                    ls = linestyles[idx] if idx < len(linestyles) and linestyles[idx] else "-"
                    ax.plot(t_ms, y, lw=float(opts.get("output_linewidth", 2.0)), color=col, linestyle=ls, label=label)
                    if opts.get("output_show_metric_points", True):
                        metrics = _compute_output_metrics(
                            curve,
                            sim_cfgs[idx] or {},
                            opts,
                            peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                            drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                            rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                            auc_window=opts.get("output_auc_window", "stim"),
                            stim_start_ms=stim_start,
                            stim_stop_ms=stim_stop,
                        )
                        metrics_plot = _scale_metrics_for_plot(metrics, scale_val)
                        _plot_metric_points(
                            ax,
                            metrics_plot,
                            color=col,
                            label_prefix=label,
                            show_labels=bool(opts.get("output_metric_label_points", False)),
                            size=float(opts.get("output_metric_marker_size", 36.0)),
                        )
                        if opts.get("output_metric_window_markers", False):
                            _plot_metric_window_markers(ax, metrics_plot, color=col, **_metric_window_kwargs(opts))

                if stim_start is not None:
                    ax.axvline(float(stim_start), color="k", linestyle="-", linewidth=float(opts.get("output_stim_linewidth", 1.0)))
                if stim_stop is not None:
                    ax.axvline(float(stim_stop), color="k", linestyle="-", linewidth=float(opts.get("output_stim_linewidth", 1.0)))
                if opts.get("plot_window") is not None:
                    ax.set_xlim(opts.get("plot_window")[0], opts.get("plot_window")[1])
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel(y_label)
                ax.set_title(compare_title)
                ax.grid(True)
                ax.legend()
                plt.tight_layout()
            _save_fig(
                fig_curve,
                analysis.plot_dir_for_compare(selection["base"], Path("compare_list"), Path("curves")) / "compare_output_curve_list.png",
                enabled=save_plots,
                dpi=plots_dpi,
            )
            return

    if compare_enabled(selection):
        run_a, run_b, res_a, res_b = resolve_compare(selection)
        if run_b is None or res_a is None or res_b is None:
            if run_b is None:
                print("Comparison disabled (set Compare B to a run name).")
                return
        curve_only_a = res_a is None
        curve_only_b = res_b is None
        if curve_only_a and curve_only_b and (run_a is None or run_b is None):
            print("Comparison disabled (set Compare A/B to a run name or curve path).")
            return
        label_a = analysis.run_label(run_a) if not curve_only_a else Path(run_a).stem
        label_b = analysis.run_label(run_b) if not curve_only_b else Path(run_b).stem
        if res_a is not None:
            smooth_mode = (res_a.get("sim_cfg", {}) or {}).get("avg_rate_curve_smooth_mode", "center")
        elif res_b is not None:
            smooth_mode = (res_b.get("sim_cfg", {}) or {}).get("avg_rate_curve_smooth_mode", "center")
        else:
            smooth_mode = "center"
        smooth_mode = opts.get("output_smooth_mode") or smooth_mode
        output_norms = None
        if opts.get("output_curve_mode", "raw") == "normalized":
            if not curve_only_a and not curve_only_b and res_a is not None and res_b is not None:
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
                        baseline_ms=opts.get("output_metric_window_ms", 100.0),
                        baseline_mode=opts.get("output_metric_mode", "point"),
                        baseline_center_ms=opts.get("output_baseline_center_ms"),
                        norm_window=opts.get("output_norm_window", "stim"),
                    )
                    norm_b = analysis.normalize_output_curve(
                        curve_b,
                        res_b.get("sim_cfg", {}) or {},
                        mode="normalized",
                        norm_mode=opts.get("output_norm_mode", "avg"),
                        baseline_ms=opts.get("output_metric_window_ms", 100.0),
                        baseline_mode=opts.get("output_metric_mode", "point"),
                        baseline_center_ms=opts.get("output_baseline_center_ms"),
                        norm_window=opts.get("output_norm_window", "stim"),
                    )
                    output_norms = (
                        {
                            "baseline_mean": norm_a.get("baseline_mean"),
                            "norm_scale": norm_a.get("norm_scale"),
                            "norm_mode": norm_a.get("norm_mode"),
                            "norm_window": norm_a.get("norm_window"),
                            "baseline_ms": norm_a.get("baseline_ms"),
                            "baseline_mode": norm_a.get("baseline_mode"),
                            "baseline_time_ms": norm_a.get("baseline_time_ms"),
                        },
                        {
                            "baseline_mean": norm_b.get("baseline_mean"),
                            "norm_scale": norm_b.get("norm_scale"),
                            "norm_mode": norm_b.get("norm_mode"),
                            "norm_window": norm_b.get("norm_window"),
                            "baseline_ms": norm_b.get("baseline_ms"),
                            "baseline_mode": norm_b.get("baseline_mode"),
                            "baseline_time_ms": norm_b.get("baseline_time_ms"),
                        },
                    )

        if opts.get("plot_outputs", True):
            if curve_only_a or curve_only_b:
                print("Output plots skipped: curve-only compare uses output curve plot only.")
                fig_cmp = None
            else:
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
                    output_scale=output_scale,
                    stim_start_ms=opts.get("output_stim_start_ms"),
                    stim_stop_ms=opts.get("output_stim_stop_ms"),
                )
            if fig_cmp is not None and opts.get("output_show_metric_points", True):
                curve_a_out = analysis.compute_output_curve_from_results(
                    res_a,
                    bin_ms=opts.get("output_bin_ms"),
                    smooth_ms=opts.get("win_size", 25) or None,
                    smooth_mode=smooth_mode,
                )
                curve_b_out = analysis.compute_output_curve_from_results(
                    res_b,
                    bin_ms=opts.get("output_bin_ms"),
                    smooth_ms=opts.get("win_size", 25) or None,
                    smooth_mode=smooth_mode,
                )
                norm_a = output_norms[0] if output_norms else None
                norm_b = output_norms[1] if output_norms else None
                curve_a_out = _apply_output_norm(curve_a_out, norm_a)
                curve_b_out = _apply_output_norm(curve_b_out, norm_b)
                metrics_a = _compute_output_metrics(
                    curve_a_out or {},
                    res_a.get("sim_cfg", {}) or {},
                    opts,
                    peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                    drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                    rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                    auc_window=opts.get("output_auc_window", "stim"),
                ) if curve_a_out else None
                metrics_b = _compute_output_metrics(
                    curve_b_out or {},
                    res_b.get("sim_cfg", {}) or {},
                    opts,
                    peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                    drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                    rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                    auc_window=opts.get("output_auc_window", "stim"),
                ) if curve_b_out else None
                metrics_a_plot = _scale_metrics_for_plot(metrics_a, output_scale)
                metrics_b_plot = _scale_metrics_for_plot(metrics_b, output_scale)

                fig_cmp_res = analysis.resolve_figure(fig_cmp)
                axes = np.atleast_1d(fig_cmp_res.axes)
                if axes.size == 1:
                    ax = axes[0]
                    line_colors = [line.get_color() for line in ax.lines if len(line.get_xdata()) > 2]
                    color_a = line_colors[0] if len(line_colors) > 0 else None
                    color_b = line_colors[1] if len(line_colors) > 1 else None
                    if metrics_a:
                        _plot_metric_points(
                            ax,
                            metrics_a_plot,
                            color=color_a,
                            label_prefix=label_a,
                            show_labels=bool(opts.get("output_metric_label_points", False)),
                            size=float(opts.get("output_metric_marker_size", 36.0)),
                        )
                        if opts.get("output_metric_window_markers", False):
                            _plot_metric_window_markers(ax, metrics_a_plot, color=color_a, **_metric_window_kwargs(opts))
                    if metrics_b:
                        _plot_metric_points(
                            ax,
                            metrics_b_plot,
                            color=color_b,
                            label_prefix=label_b,
                            show_labels=bool(opts.get("output_metric_label_points", False)),
                            size=float(opts.get("output_metric_marker_size", 36.0)),
                        )
                        if opts.get("output_metric_window_markers", False):
                            _plot_metric_window_markers(ax, metrics_b_plot, color=color_b, **_metric_window_kwargs(opts))
                else:
                    if axes.size > 0 and metrics_a:
                        line_colors = [line.get_color() for line in axes[0].lines if len(line.get_xdata()) > 2]
                        color_a = line_colors[0] if line_colors else None
                        _plot_metric_points(
                            axes[0],
                            metrics_a_plot,
                            color=color_a,
                            label_prefix=label_a,
                            show_labels=bool(opts.get("output_metric_label_points", False)),
                            size=float(opts.get("output_metric_marker_size", 36.0)),
                        )
                        if opts.get("output_metric_window_markers", False):
                            _plot_metric_window_markers(axes[0], metrics_a_plot, color=color_a, **_metric_window_kwargs(opts))
                    if axes.size > 1 and metrics_b:
                        line_colors = [line.get_color() for line in axes[1].lines if len(line.get_xdata()) > 2]
                        color_b = line_colors[0] if line_colors else None
                        _plot_metric_points(
                            axes[1],
                            metrics_b_plot,
                            color=color_b,
                            label_prefix=label_b,
                            show_labels=bool(opts.get("output_metric_label_points", False)),
                            size=float(opts.get("output_metric_marker_size", 36.0)),
                        )
                        if opts.get("output_metric_window_markers", False):
                            _plot_metric_window_markers(axes[1], metrics_b_plot, color=color_b, **_metric_window_kwargs(opts))
            if fig_cmp is not None:
                _save_fig(
                    fig_cmp,
                    analysis.plot_dir_for_compare(selection["base"], run_a, run_b) / "compare_outputs.png",
                    enabled=save_plots,
                    dpi=plots_dpi,
                )

        if opts.get("plot_output_curve", True):
            curve_a = _load_curve_from_path(
                Path(run_a),
                opts,
                fallback_sim_cfg=(res_b.get("sim_cfg") if res_b else None),
            ) if curve_only_a else analysis.compute_output_curve_from_results(
                res_a,
                bin_ms=opts.get("output_bin_ms"),
                smooth_ms=_curve_smooth_ms(opts),
                smooth_mode=opts.get("output_smooth_mode", "causal"),
            )
            curve_b = _load_curve_from_path(
                Path(run_b),
                opts,
                fallback_sim_cfg=(res_a.get("sim_cfg") if res_a else None),
            ) if curve_only_b else analysis.compute_output_curve_from_results(
                res_b,
                bin_ms=opts.get("output_bin_ms"),
                smooth_ms=_curve_smooth_ms(opts),
                smooth_mode=opts.get("output_smooth_mode", "causal"),
            )
            scatter_curve = analysis.load_scatter_curve_optional(
                enabled=opts.get("output_scatter_enabled", False),
                path=opts.get("output_scatter_path", ""),
                time_unit=opts.get("output_scatter_time_unit", "s"),
                bin_ms=opts.get("output_bin_ms"),
                smooth_ms=_curve_smooth_ms(opts),
                smooth_mode=opts.get("output_smooth_mode", "causal"),
                shift_ms=opts.get("output_scatter_shift_ms"),
                quiet=True,
            )
            if curve_a and curve_b:
                if not curve_only_a:
                    curve_a = analysis.normalize_output_curve(
                        curve_a,
                        res_a.get("sim_cfg", {}) or {},
                        mode=opts.get("output_curve_mode", "raw"),
                        norm_mode=opts.get("output_norm_mode", "avg"),
                        baseline_ms=opts.get("output_metric_window_ms", 100.0),
                        baseline_mode=opts.get("output_metric_mode", "point"),
                        baseline_center_ms=opts.get("output_baseline_center_ms"),
                        norm_window=opts.get("output_norm_window", "stim"),
                    )
                if not curve_only_b:
                    curve_b = analysis.normalize_output_curve(
                        curve_b,
                        res_b.get("sim_cfg", {}) or {},
                        mode=opts.get("output_curve_mode", "raw"),
                        norm_mode=opts.get("output_norm_mode", "avg"),
                        baseline_ms=opts.get("output_metric_window_ms", 100.0),
                        baseline_mode=opts.get("output_metric_mode", "point"),
                        baseline_center_ms=opts.get("output_baseline_center_ms"),
                        norm_window=opts.get("output_norm_window", "stim"),
                    )
                sim_cfg_a = (res_a.get("sim_cfg") if res_a else _default_sim_cfg_for_curve(curve_a, fallback=(res_b.get("sim_cfg") if res_b else None))) or {}
                stim_start, stim_stop = _stim_window_for_opts(sim_cfg_a, opts)
                scale_a = external_scale if curve_only_a else output_scale
                scale_b = external_scale if curve_only_b else output_scale
                curve_a_plot = _scale_curve_for_plot(curve_a, scale_a)
                curve_b_plot = _scale_curve_for_plot(curve_b, scale_b)
                fig_curve, _ = plotting.plot_compare_output_curves(
                    curve_a_plot,
                    curve_b_plot,
                    labels=(label_a, label_b),
                    plot_window=opts.get("plot_window", (None, None)),
                    stim_start=stim_start,
                    stim_stop=stim_stop,
                    title="Output curves",
                    line_width=float(opts.get("output_linewidth", 2.0)),
                    stim_linewidth=float(opts.get("output_stim_linewidth", 1.0)),
                )
                if opts.get("output_show_metric_points", True):
                    metrics_a = _compute_output_metrics(
                        curve_a,
                        sim_cfg_a,
                        opts,
                        peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                        drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                        rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                        auc_window=opts.get("output_auc_window", "stim"),
                        stim_start_ms=stim_start,
                        stim_stop_ms=stim_stop,
                    )
                    metrics_b = _compute_output_metrics(
                        curve_b,
                        (res_b.get("sim_cfg") if res_b else _default_sim_cfg_for_curve(curve_b, fallback=(res_a.get("sim_cfg") if res_a else None))) or {},
                        opts,
                        peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                        drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                        rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                        auc_window=opts.get("output_auc_window", "stim"),
                        stim_start_ms=stim_start,
                        stim_stop_ms=stim_stop,
                    )
                    metrics_a_plot = _scale_metrics_for_plot(metrics_a, scale_a)
                    metrics_b_plot = _scale_metrics_for_plot(metrics_b, scale_b)
                    ax = fig_curve.axes[0] if fig_curve.axes else None
                    if ax is not None:
                        line_colors = [line.get_color() for line in ax.lines]
                        color_a = line_colors[0] if len(line_colors) > 0 else None
                        color_b = line_colors[1] if len(line_colors) > 1 else None
                        _plot_metric_points(
                            ax,
                            metrics_a_plot,
                            color=color_a,
                            label_prefix=label_a,
                            show_labels=bool(opts.get("output_metric_label_points", False)),
                            size=float(opts.get("output_metric_marker_size", 36.0)),
                        )
                        if opts.get("output_metric_window_markers", False):
                            _plot_metric_window_markers(ax, metrics_a_plot, color=color_a, **_metric_window_kwargs(opts))
                        _plot_metric_points(
                            ax,
                            metrics_b_plot,
                            color=color_b,
                            label_prefix=label_b,
                            show_labels=bool(opts.get("output_metric_label_points", False)),
                            size=float(opts.get("output_metric_marker_size", 36.0)),
                        )
                        if opts.get("output_metric_window_markers", False):
                            _plot_metric_window_markers(ax, metrics_b_plot, color=color_b, **_metric_window_kwargs(opts))
                if scatter_curve:
                    ax = fig_curve.axes[0] if fig_curve.axes else None
                    if ax is not None:
                        scatter_plot = _scale_curve_for_plot(scatter_curve, external_scale)
                        ax.plot(
                            np.asarray(scatter_plot.get("t_ms", []), dtype=float),
                            np.asarray(scatter_plot.get("rate_hz", []), dtype=float),
                            color=opts.get("output_scatter_color", "0.4"),
                            lw=float(opts.get("output_linewidth", 2.0)),
                            ls="--",
                            label=opts.get("output_scatter_label", "External curve"),
                        )
                        if opts.get("output_show_metric_points", True):
                            scatter_metrics = _compute_output_metrics(
                                scatter_curve,
                                sim_cfg_a,
                                opts,
                                peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                                drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                                rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                                auc_window=opts.get("output_auc_window", "stim"),
                                stim_start_ms=stim_start,
                                stim_stop_ms=stim_stop,
                            )
                            scatter_metrics_plot = _scale_metrics_for_plot(scatter_metrics, external_scale)
                            _plot_metric_points(
                                ax,
                                scatter_metrics_plot,
                                color=opts.get("output_scatter_color", "0.4"),
                                label_prefix=opts.get("output_scatter_label", "External curve"),
                                show_labels=bool(opts.get("output_metric_label_points", False)),
                                size=float(opts.get("output_metric_marker_size", 36.0)),
                            )
                            if opts.get("output_metric_window_markers", False):
                                _plot_metric_window_markers(
                                    ax,
                                    scatter_metrics_plot,
                                    color=opts.get("output_scatter_color", "0.4"),
                                    **_metric_window_kwargs(opts),
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
    smooth_mode = opts.get("output_smooth_mode") or smooth_mode

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
            stim_start, stim_stop = _stim_window_for_opts(res.get("sim_cfg", {}) or {}, opts)
            if stim_start is not None:
                sim_params["stim_start_ms"] = stim_start
            if stim_stop is not None:
                sim_params["stim_stop_ms"] = stim_stop
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
                        baseline_ms=opts.get("output_metric_window_ms", 100.0),
                        baseline_mode=opts.get("output_metric_mode", "point"),
                        baseline_center_ms=opts.get("output_baseline_center_ms"),
                        norm_window=opts.get("output_norm_window", "stim"),
                    )
                    output_norm = {
                        "baseline_mean": norm_curve.get("baseline_mean"),
                        "norm_scale": norm_curve.get("norm_scale"),
                        "norm_mode": norm_curve.get("norm_mode"),
                        "norm_window": norm_curve.get("norm_window"),
                        "baseline_ms": norm_curve.get("baseline_ms"),
                        "baseline_mode": norm_curve.get("baseline_mode"),
                        "baseline_time_ms": norm_curve.get("baseline_time_ms"),
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
                output_scale=output_scale,
                bio_scale=external_scale,
            )
            if opts.get("output_show_metric_points", True):
                curve_out = analysis.compute_output_curve_from_results(
                    res,
                    bin_ms=sim_params.get("bins"),
                    smooth_ms=opts.get("win_size", 25) or None,
                    smooth_mode=smooth_mode,
                )
                curve_out = _apply_output_norm(curve_out, output_norm)
                metrics_out = _compute_output_metrics(
                    curve_out or {},
                    res.get("sim_cfg", {}) or {},
                    opts,
                    peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                    drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                    rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                    auc_window=opts.get("output_auc_window", "stim"),
                    stim_start_ms=stim_start,
                    stim_stop_ms=stim_stop,
                ) if curve_out else None
                fig_out = analysis.resolve_figure(None)
                ax_rate = fig_out.axes[0] if fig_out.axes else None
                if ax_rate is not None and metrics_out:
                    line_colors = [line.get_color() for line in ax_rate.lines if len(line.get_xdata()) > 2]
                    color_out = line_colors[0] if line_colors else None
                    metrics_plot = _scale_metrics_for_plot(metrics_out, output_scale)
                    _plot_metric_points(
                        ax_rate,
                        metrics_plot,
                        color=color_out,
                        label_prefix=analysis.run_label(run_dir),
                        show_labels=bool(opts.get("output_metric_label_points", False)),
                        size=float(opts.get("output_metric_marker_size", 36.0)),
                    )
                    if opts.get("output_metric_window_markers", False):
                        _plot_metric_window_markers(ax_rate, metrics_plot, color=color_out, **_metric_window_kwargs(opts))
                if ax_rate is not None and in_vivo_curve is not None:
                    bio_curve = _curve_from_xy(in_vivo_curve[0] * 1000.0, in_vivo_curve[1])
                    bio_metrics = _compute_output_metrics(
                        bio_curve,
                        res.get("sim_cfg", {}) or {},
                        opts,
                        peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                        drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                        rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                        auc_window=opts.get("output_auc_window", "stim"),
                        stim_start_ms=stim_start,
                        stim_stop_ms=stim_stop,
                    )
                    bio_metrics_plot = _scale_metrics_for_plot(bio_metrics, external_scale)
                    _plot_metric_points(
                        ax_rate,
                        bio_metrics_plot,
                        color="k",
                        label_prefix="Bio",
                        show_labels=bool(opts.get("output_metric_label_points", False)),
                        size=float(opts.get("output_metric_marker_size", 36.0)),
                    )
                    if opts.get("output_metric_window_markers", False):
                        _plot_metric_window_markers(ax_rate, bio_metrics_plot, color="k", **_metric_window_kwargs(opts))
            _save_fig(
                plt.gcf(),
                analysis.plot_dir_for_run(run_dir) / "output_plot.png",
                enabled=save_plots,
                dpi=plots_dpi,
            )
        else:
            in_vivo_curve_plot = in_vivo_curve
            if in_vivo_curve_plot is not None and external_scale not in (None, 1.0):
                try:
                    t_s, rate = in_vivo_curve_plot
                    in_vivo_curve_plot = (t_s, np.asarray(rate, dtype=float) * float(external_scale))
                except Exception:
                    pass
            fig_out = plotting.plot_results(
                res,
                syn_records=res.get("syn_records"),
                in_vivo_curve=in_vivo_curve_plot,
                win_size=opts.get("win_size", 25),
                raster_style=opts.get("raster_style", "dot"),
                plot_raster=opts.get("plot_raster", True),
                plot_window=opts.get("plot_window", (None, None)),
                smooth_mode=smooth_mode,
                bin_ms=opts.get("output_bin_ms"),
                line_width=opts.get("output_linewidth", 2.0),
                shade_alpha=opts.get("output_shade_alpha", 0.25),
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
            smooth_ms=_curve_smooth_ms(opts),
            smooth_mode=opts.get("output_smooth_mode", "causal"),
        )
        scatter_curve = analysis.load_scatter_curve_optional(
            enabled=opts.get("output_scatter_enabled", False),
            path=opts.get("output_scatter_path", ""),
            time_unit=opts.get("output_scatter_time_unit", "s"),
            bin_ms=opts.get("output_bin_ms"),
            smooth_ms=_curve_smooth_ms(opts),
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
                baseline_ms=opts.get("output_metric_window_ms", 100.0),
                baseline_mode=opts.get("output_metric_mode", "point"),
                baseline_center_ms=opts.get("output_baseline_center_ms"),
                norm_window=opts.get("output_norm_window", "stim"),
            )
            stim_start, stim_stop = _stim_window_for_opts(res.get("sim_cfg", {}) or {}, opts)
            curve_single_plot = _scale_curve_for_plot(curve_single, output_scale)
            fig_curve, _ = plotting.plot_output_curve(
                curve_single_plot,
                label=analysis.run_label(run_dir),
                color=(res.get("sim_cfg", {}) or {}).get("color", None),
                plot_window=opts.get("plot_window", (None, None)),
                stim_start=stim_start,
                stim_stop=stim_stop,
                title="Output curve (avg)",
                line_width=float(opts.get("output_linewidth", 2.0)),
                stim_linewidth=float(opts.get("output_stim_linewidth", 1.0)),
            )
            if opts.get("output_show_metric_points", True):
                metrics_single = _compute_output_metrics(
                    curve_single,
                    res.get("sim_cfg", {}) or {},
                    opts,
                    peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                    drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                    rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                    auc_window=opts.get("output_auc_window", "stim"),
                    stim_start_ms=stim_start,
                    stim_stop_ms=stim_stop,
                )
                metrics_single_plot = _scale_metrics_for_plot(metrics_single, output_scale)
                ax = fig_curve.axes[0] if fig_curve.axes else None
                if ax is not None:
                    line_color = ax.lines[0].get_color() if ax.lines else None
                    _plot_metric_points(
                        ax,
                        metrics_single_plot,
                        color=line_color,
                        label_prefix=analysis.run_label(run_dir),
                        show_labels=bool(opts.get("output_metric_label_points", False)),
                        size=float(opts.get("output_metric_marker_size", 36.0)),
                    )
                    if opts.get("output_metric_window_markers", False):
                        _plot_metric_window_markers(ax, metrics_single_plot, color=line_color, **_metric_window_kwargs(opts))
            if scatter_curve:
                ax = fig_curve.axes[0] if fig_curve.axes else None
                if ax is not None:
                    scatter_plot = _scale_curve_for_plot(scatter_curve, external_scale)
                    ax.plot(
                        np.asarray(scatter_plot.get("t_ms", []), dtype=float),
                        np.asarray(scatter_plot.get("rate_hz", []), dtype=float),
                        color=opts.get("output_scatter_color", "0.4"),
                        lw=float(opts.get("output_linewidth", 2.0)),
                        ls="--",
                        label=opts.get("output_scatter_label", "External curve"),
                    )
                    if opts.get("output_show_metric_points", True):
                        scatter_metrics = _compute_output_metrics(
                            scatter_curve,
                            res.get("sim_cfg", {}) or {},
                            opts,
                            peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                            drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                            rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                            auc_window=opts.get("output_auc_window", "stim"),
                            stim_start_ms=stim_start,
                            stim_stop_ms=stim_stop,
                        )
                        scatter_metrics_plot = _scale_metrics_for_plot(scatter_metrics, external_scale)
                        _plot_metric_points(
                            ax,
                            scatter_metrics_plot,
                            color=opts.get("output_scatter_color", "0.4"),
                            label_prefix=opts.get("output_scatter_label", "External curve"),
                            show_labels=bool(opts.get("output_metric_label_points", False)),
                            size=float(opts.get("output_metric_marker_size", 36.0)),
                        )
                        if opts.get("output_metric_window_markers", False):
                            _plot_metric_window_markers(
                                ax,
                                scatter_metrics_plot,
                                color=opts.get("output_scatter_color", "0.4"),
                                **_metric_window_kwargs(opts),
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
    compare_runs = _compare_list_run_paths(selection)
    if len(compare_runs) >= 2:
        run_a, run_b = compare_runs[0], compare_runs[1]
        res_a = _safe_load_results(run_a)
        res_b = _safe_load_results(run_b)
        if res_a is None or res_b is None:
            print("Comparison disabled (missing input data in one or more runs).")
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
                input_source=opts.get("input_source", "saved"),
                std_mode=opts.get("input_std_mode", "std"),
            )
            summary_b = analysis.summarize_inputs_from_results(
                res_b,
                groups=opts.get("input_groups"),
                bin_ms=opts.get("input_bin_ms"),
                smooth_ms=opts.get("input_smooth_ms"),
                input_source=opts.get("input_source", "saved"),
                std_mode=opts.get("input_std_mode", "std"),
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
                legend_loc=opts.get("input_legend_loc"),
                group_colors=group_colors,
                line_width=opts.get("input_linewidth", 2.0),
                shade_alpha=opts.get("input_shade_alpha", 0.2),
                output_linewidth=opts.get("input_output_linewidth", 1.5),
                plot_window=opts.get("input_plot_window"),
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
    if len(compare_runs) == 1:
        selection = dict(selection)
        selection["run_single"] = compare_runs[0]
        selection["compare_list"] = []
        selection["compare_list_paths"] = []

    run_dir, res = resolve_single(selection)
    group_colors = analysis.group_colors_from_results(res)

    if opts.get("plot_inputs_mean", True):
        summary_single = analysis.summarize_inputs_from_results(
            res,
            groups=opts.get("input_groups"),
            bin_ms=opts.get("input_bin_ms"),
            smooth_ms=opts.get("input_smooth_ms"),
            input_source=opts.get("input_source", "saved"),
            std_mode=opts.get("input_std_mode", "std"),
        )
        curve_single = (res.get("meta") or {}).get("avg_rate_curve")
        fig_in, _ = plotting.plot_input_means(
            summary_single,
            label=analysis.run_label(run_dir),
            groups=opts.get("input_groups"),
            show_std=opts.get("show_input_std", False),
            output_curve=curve_single,
            plot_window=opts.get("input_plot_window"),
            legend_loc=opts.get("input_legend_loc"),
            group_colors=group_colors,
            line_width=opts.get("input_linewidth", 2.0),
            shade_alpha=opts.get("input_shade_alpha", 0.2),
            output_linewidth=opts.get("input_output_linewidth", 1.5),
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
                legend_loc=opts.get("input_legend_loc"),
                plot_raster=True,
                line_width=opts.get("input_linewidth", 2.0),
                raster_linewidth=opts.get("input_raster_linewidth", 0.8),
                stim_linewidth=opts.get("input_stim_linewidth", 1.0),
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
    base_dir = selection.get("base")
    preset_defaults = _load_compare_preset_defaults(selection.get("compare_preset_path"), base_dir)
    if preset_defaults:
        opts = _merge_preset_defaults(opts, preset_defaults)
    preset_entries = _load_compare_preset(selection.get("compare_preset_path"), base_dir)
    list_entries = _compare_list_entries(selection) if not preset_entries else []
    if preset_entries or list_entries:
        paths: list[Path] = []
        shifts: list[Optional[float]] = []
        scales: list[Optional[float]] = []
        labels_override: list[Optional[str]] = []
        has_curve = False
        if preset_entries:
            for entry in preset_entries:
                path = entry["path"]
                shift_ms = entry.get("shift_ms")
                scale_val = entry.get("scale")
                if _is_curve_path(path):
                    has_curve = True
                paths.append(path)
                shifts.append(shift_ms)
                scales.append(scale_val)
                labels_override.append(entry.get("label"))
        else:
            for item in list_entries:
                spec = _parse_compare_list_item(item)
                path_raw = spec.get("path")
                shift_ms = spec.get("shift_ms")
                scale_val = spec.get("scale")
                path = _coerce_run_path(path_raw, base_dir)
                if path is None:
                    continue
                if not path.exists():
                    print(f"Skipping missing path: {path}")
                    continue
                paths.append(path)
                shifts.append(shift_ms)
                scales.append(scale_val)
                labels_override.append(spec.get("label"))
                if _is_curve_path(path):
                    has_curve = True

        if len(paths) == 1 and not has_curve:
            selection["run_single"] = paths[0]
        else:
            fallback_sim_cfg = None
            for path in paths:
                if not _is_curve_path(path):
                    res = _safe_load_results(path)
                    fallback_sim_cfg = (res.get("sim_cfg") or {}) if res else None
                    break

            metrics_map: Dict[str, Any] = {}
            for idx, (path, shift_ms, _scale_val) in enumerate(zip(paths, shifts, scales)):
                if _is_curve_path(path):
                    curve = _load_curve_from_path(path, opts, shift_ms=shift_ms, fallback_sim_cfg=fallback_sim_cfg)
                    if not curve:
                        continue
                    label_override = labels_override[idx] if idx < len(labels_override) else None
                    label = label_override or Path(path).stem
                    sim_cfg = _default_sim_cfg_for_curve(curve, fallback=fallback_sim_cfg)
                else:
                    res = _safe_load_results(path)
                    if res is None:
                        print(f"Skipping missing run: {path}")
                        continue
                    curve = analysis.compute_output_curve_from_results(
                        res,
                        bin_ms=opts.get("output_bin_ms"),
                        smooth_ms=_curve_smooth_ms(opts),
                        smooth_mode=opts.get("output_smooth_mode", "causal"),
                    )
                    if not curve:
                        continue
                    curve = analysis.normalize_output_curve(
                        curve,
                        res.get("sim_cfg", {}) or {},
                        mode=opts.get("output_curve_mode", "raw"),
                        norm_mode=opts.get("output_norm_mode", "avg"),
                        baseline_ms=opts.get("output_metric_window_ms", 100.0),
                        baseline_mode=opts.get("output_metric_mode", "point"),
                        baseline_center_ms=opts.get("output_baseline_center_ms"),
                        norm_window=opts.get("output_norm_window", "stim"),
                    )
                    if shift_ms is not None:
                        t_ms = np.asarray(curve.get("t_ms", []) or [], dtype=float) + float(shift_ms)
                        curve = dict(curve)
                        curve["t_ms"] = t_ms.tolist()
                    label_override = labels_override[idx] if idx < len(labels_override) else None
                    label = label_override or analysis.run_label(path)
                    sim_cfg = res.get("sim_cfg", {}) or {}

                metrics_map[label] = _compute_output_metrics(
                    curve,
                    sim_cfg,
                    opts,
                    peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                    drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                    rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                    auc_window=opts.get("output_auc_window", "stim"),
                )

            if not metrics_map:
                print("Output metrics skipped: no valid curves in compare list.")
                return None
            _save_json(
                metrics_map,
                analysis.analysis_dir_for_compare(selection["base"], Path("compare_list"), Path("curves")) / "output_metrics_list.json",
                enabled=save_analysis,
            )
            return metrics_map

    if compare_enabled(selection):
        run_a, run_b, res_a, res_b = resolve_compare(selection)
        if run_b is None:
            print("Output metrics skipped: compare B not set.")
            return None
        curve_only_a = res_a is None
        curve_only_b = res_b is None
        curve_a = _load_curve_from_path(
            Path(run_a),
            opts,
            fallback_sim_cfg=(res_b.get("sim_cfg") if res_b else None),
        ) if curve_only_a else analysis.compute_output_curve_from_results(
            res_a,
            bin_ms=opts.get("output_bin_ms"),
            smooth_ms=_curve_smooth_ms(opts),
            smooth_mode=opts.get("output_smooth_mode", "causal"),
        )
        curve_b = _load_curve_from_path(
            Path(run_b),
            opts,
            fallback_sim_cfg=(res_a.get("sim_cfg") if res_a else None),
        ) if curve_only_b else analysis.compute_output_curve_from_results(
            res_b,
            bin_ms=opts.get("output_bin_ms"),
            smooth_ms=_curve_smooth_ms(opts),
            smooth_mode=opts.get("output_smooth_mode", "causal"),
        )
        if not curve_a or not curve_b:
            print("Output metrics skipped: missing curves in compare selection.")
            return None
        if not curve_only_a:
            curve_a = analysis.normalize_output_curve(
                curve_a,
                res_a.get("sim_cfg", {}) or {},
                mode=opts.get("output_curve_mode", "raw"),
                norm_mode=opts.get("output_norm_mode", "avg"),
                baseline_ms=opts.get("output_metric_window_ms", 100.0),
                baseline_mode=opts.get("output_metric_mode", "point"),
                baseline_center_ms=opts.get("output_baseline_center_ms"),
                norm_window=opts.get("output_norm_window", "stim"),
            )
        if not curve_only_b:
            curve_b = analysis.normalize_output_curve(
                curve_b,
                res_b.get("sim_cfg", {}) or {},
                mode=opts.get("output_curve_mode", "raw"),
                norm_mode=opts.get("output_norm_mode", "avg"),
                baseline_ms=opts.get("output_metric_window_ms", 100.0),
                baseline_mode=opts.get("output_metric_mode", "point"),
                baseline_center_ms=opts.get("output_baseline_center_ms"),
                norm_window=opts.get("output_norm_window", "stim"),
            )
        sim_cfg_a = res_a.get("sim_cfg", {}) if res_a else _default_sim_cfg_for_curve(curve_a)
        sim_cfg_b = res_b.get("sim_cfg", {}) if res_b else _default_sim_cfg_for_curve(curve_b)
        metrics_map = {
            analysis.run_label(run_a) if not curve_only_a else Path(run_a).stem: _compute_output_metrics(
                curve_a,
                sim_cfg_a or {},
                opts,
                peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                auc_window=opts.get("output_auc_window", "stim"),
            ),
            analysis.run_label(run_b) if not curve_only_b else Path(run_b).stem: _compute_output_metrics(
                curve_b,
                sim_cfg_b or {},
                opts,
                peak_window_ms=opts.get("output_peak_window_ms", 100.0),
                drop_window_ms=opts.get("output_drop_window_ms", 100.0),
                rebound_window_ms=opts.get("output_rebound_window_ms", 300.0),
                auc_window=opts.get("output_auc_window", "stim"),
            ),
        }
        _save_json(
            metrics_map,
            analysis.analysis_dir_for_compare(selection["base"], Path(run_a), Path(run_b)) / "output_metrics_compare.json",
            enabled=save_analysis,
        )
        return metrics_map
    run_dir, res = resolve_single(selection)
    curve = analysis.compute_output_curve_from_results(
        res,
        bin_ms=opts.get("output_bin_ms"),
        smooth_ms=_curve_smooth_ms(opts),
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
        baseline_ms=opts.get("output_metric_window_ms", 100.0),
        baseline_mode=opts.get("output_metric_mode", "point"),
        baseline_center_ms=opts.get("output_baseline_center_ms"),
        norm_window=opts.get("output_norm_window", "stim"),
    )
    metrics = _compute_output_metrics(
        curve,
        res.get("sim_cfg", {}) or {},
        opts,
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

_HIGHLIGHT_METRICS = {
    "peak_latency_ms",
    "drop_pct",
    "rebound_pct",
    "auc",
    "baseline_mean",
    "peak_rate_hz_raw",
    "peak_value_raw",
}


def _format_metric_key(key: str) -> str:
    if key in _HIGHLIGHT_METRICS:
        return f"**{key}**"
    return key


def _format_metric_value(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float, np.integer, np.floating)):
        try:
            val_f = float(val)
        except Exception:
            return str(val)
        if np.isnan(val_f):
            return ""
        return f"{val_f:.3g}"
    return str(val)


def _filter_metrics_for_display(data: Dict[str, Any]) -> Dict[str, Any]:
    if not data:
        return {}
    filtered = dict(data)
    peak_raw = filtered.get("peak_rate_hz_raw")
    if peak_raw is None:
        peak_raw = filtered.get("peak_value_raw")
    if peak_raw is not None:
        for key in ("peak_value", "peak_rate_hz"):
            val = filtered.get(key)
            try:
                if val is not None and abs(float(val) - 1.0) < 1e-6:
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


_OUTPUT_METRIC_VALUE_ORDER = [
    "baseline_mean",
    "peak_rate_hz_raw",
    "peak_value_raw",
    "peak_rate_hz",
    "peak_value",
    "peak_latency_ms",
    "drop_value",
    "drop_pct",
    "rebound_value",
    "rebound_pct",
    "auc",
]


def _ordered_metric_keys(keys: list[str], order: Optional[list[str]] = None) -> list[str]:
    if not order:
        return keys
    ordered = [key for key in order if key in keys]
    for key in keys:
        if key not in ordered:
            ordered.append(key)
    return ordered


def format_kv_table(
    data: dict,
    *,
    title: str = "Output metrics",
    order: Optional[list[str]] = None,
) -> str:
    lines = [f"### {title}", "| Metric | Value |", "| --- | --- |"]
    data = _filter_metrics_for_display(data)
    for key in _ordered_metric_keys(list(data.keys()), order):
        val = data.get(key)
        lines.append(f"| {_format_metric_key(key)} | {_format_metric_value(val)} |")
    return "\n".join(lines)


def _format_delta(value: Any, ref: Any) -> str:
    try:
        val_f = float(value)
        ref_f = float(ref)
    except Exception:
        return ""
    delta = val_f - ref_f
    delta_str = f"{delta:+.3g}"
    if ref_f == 0:
        return f"Δ={delta_str}"
    pct = (delta / ref_f) * 100.0
    pct_str = f"{pct:+.3g}%"
    return f"Δ={delta_str} ({pct_str})"


def _best_labels_by_metric(
    data_by_label: Dict[str, Dict[str, Any]],
    *,
    reference_label: Optional[str],
) -> Dict[str, str]:
    if not reference_label or reference_label not in data_by_label:
        return {}
    best: Dict[str, str] = {}
    ref_vals = data_by_label.get(reference_label, {})
    for key, ref_val in ref_vals.items():
        try:
            ref_f = float(ref_val)
        except Exception:
            continue
        best_label = None
        best_err = None
        for label, metrics in data_by_label.items():
            if label == reference_label:
                continue
            try:
                val_f = float(metrics.get(key))
            except Exception:
                continue
            err = abs(val_f - ref_f)
            if best_err is None or err < best_err:
                best_err = err
                best_label = label
        if best_label is not None:
            best[key] = best_label
    return best


def format_kv_table_columns(
    data_by_label: Dict[str, Dict[str, Any]],
    *,
    title: str = "Output metrics",
    reference_label: Optional[str] = None,
    show_deltas: bool = False,
    highlight_best: bool = False,
    order: Optional[list[str]] = None,
) -> str:
    labels = list(data_by_label.keys())
    if not labels:
        return format_kv_table({}, title=title)
    filtered = {label: _filter_metrics_for_display(data_by_label[label]) for label in labels}
    metric_keys = list(filtered[labels[0]].keys())
    for label in labels[1:]:
        for key in filtered[label].keys():
            if key not in metric_keys:
                metric_keys.append(key)
    metric_keys = _ordered_metric_keys(metric_keys, order)

    best_by_metric = _best_labels_by_metric(filtered, reference_label=reference_label) if highlight_best else {}
    header_labels = [
        f"{label} (ref)" if reference_label and label == reference_label else label for label in labels
    ]
    header = "| Metric | " + " | ".join(header_labels) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(labels)) + " |"
    lines = [f"### {title}", header, sep]
    for key in metric_keys:
        row = []
        for label in labels:
            val = filtered[label].get(key, "")
            cell = _format_metric_value(val)
            if show_deltas and reference_label and reference_label in filtered and label != reference_label:
                ref_val = filtered[reference_label].get(key, None)
                delta_str = _format_delta(val, ref_val)
                if delta_str:
                    cell = f"{cell} ({delta_str})"
            if highlight_best and best_by_metric.get(key) == label:
                cell = f"**{cell}**"
            row.append(cell)
        lines.append("| " + _format_metric_key(key) + " | " + " | ".join(row) + " |")
    return "\n".join(lines)


_OUTPUT_PARAM_KEYS = {
    "peak_window_ms",
    "drop_window_ms",
    "rebound_window_ms",
    "auc_window",
    "auc_units",
    "pdp_mode",
    "pdp_window_ms",
    "stim_start_ms",
    "stim_stop_ms",
    "baseline_ms",
    "baseline_mode",
    "baseline_center_ms",
    "baseline_time_ms",
    "baseline_window_start_ms",
    "baseline_window_stop_ms",
    "peak_time_ms",
    "drop_time_ms",
    "rebound_time_ms",
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
}


def split_output_metrics(metrics: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    params: Dict[str, Any] = {}
    values: Dict[str, Any] = {}
    for key, val in metrics.items():
        if key in _OUTPUT_PARAM_KEYS:
            params[key] = val
        else:
            values[key] = val
    return params, values


def split_output_metrics_columns(
    data_by_label: Dict[str, Dict[str, Any]],
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    params_by_label: Dict[str, Dict[str, Any]] = {}
    values_by_label: Dict[str, Dict[str, Any]] = {}
    for label, metrics in data_by_label.items():
        params, values = split_output_metrics(metrics)
        params_by_label[label] = params
        values_by_label[label] = values
    return params_by_label, values_by_label


def format_output_metrics_tables(
    metrics: Dict[str, Any],
    *,
    title: str = "Output metrics",
    show_params: bool = True,
) -> str:
    params, values = split_output_metrics(metrics)
    parts = []
    if values:
        parts.append(format_kv_table(values, title=title, order=_OUTPUT_METRIC_VALUE_ORDER))
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
) -> str:
    params_by_label, values_by_label = split_output_metrics_columns(data_by_label)
    parts = []
    if any(values_by_label.values()):
        parts.append(
            format_kv_table_columns(
                values_by_label,
                title=title,
                reference_label=reference_label,
                show_deltas=show_deltas,
                highlight_best=highlight_best,
                order=_OUTPUT_METRIC_VALUE_ORDER,
            )
        )
    if show_params and any(params_by_label.values()):
        parts.append(format_kv_table_columns(params_by_label, title=f"{title} (params)"))
    return "\n\n".join(parts)

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
        run_single = g.get("run_single_stem", "latest")
        run_a = g.get("run_compare_a", "latest")
        run_b = g.get("run_compare_b", "none")
        comp_a = g.get("compare_a_path", "")
        comp_b = g.get("compare_b_path", "")
        compare_list = list(g["compare_list_sel"].value) if g.get("compare_list_sel") is not None else []
        compare_paths_enabled = bool(
            g.get("compare_paths_cb").value if g.get("compare_paths_cb") is not None else g.get("compare_list_paths_enabled", True)
        )
        compare_list_paths = (
            _parse_compare_list_paths(
                g.get("compare_list_paths_txt").value if g.get("compare_list_paths_txt") is not None else ""
            )
            if compare_paths_enabled
            else []
        )
    else:
        cell = g.get("cell_name")
        tunes = g.get("tunes_dir")
        model = g.get("model_dir")
        run_single = g.get("run_single_stem")
        run_a = g.get("run_compare_a")
        run_b = g.get("run_compare_b")
        comp_a = g.get("compare_a_path", "")
        comp_b = g.get("compare_b_path", "")
        compare_list = g.get("compare_list", []) or []
        compare_paths_enabled = bool(g.get("compare_list_paths_enabled", True))
        compare_list_paths = g.get("compare_list_paths", []) or []
        if not compare_paths_enabled:
            compare_list_paths = []

    if compare_list_paths and not compare_list:
        compare_list = ["latest"]

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
        "compare_list": compare_list,
        "compare_list_paths": compare_list_paths,
        "compare_preset_path": g.get("compare_preset_path"),
    }


def output_opts_from_globals(g: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plot_outputs": bool(g.get("plot_outputs", True)),
        "plot_output_curve": bool(g.get("plot_output_curve", True)),
        "plot_spike_stats": g.get("plot_spike_stats"),
        "plot_raster": g.get("plot_raster"),
        "raster_style": g.get("raster_style"),
        "plot_window": g.get("plot_window"),
        "compare_output_layout": g.get("compare_output_layout"),
        "output_compare_figsize": g.get("output_compare_figsize"),
        "output_compare_panel_size": g.get("output_compare_panel_size"),
        "win_size": g.get("win_size"),
        "multi_plot_type": g.get("multi_plot_type"),
        "multi_shade_mode": g.get("multi_shade_mode"),
        "multi_norm_fr": g.get("multi_norm_fr"),
        "output_curve_mode": g.get("output_curve_mode"),
        "output_norm_mode": g.get("output_norm_mode"),
        "output_metric_window_ms": g.get("output_metric_window_ms"),
        "output_metric_mode": g.get("output_metric_mode"),
        "output_baseline_center_ms": g.get("output_baseline_center_ms"),
        "output_metric_window_markers": g.get("output_metric_window_markers", True),
        "output_norm_window": g.get("output_norm_window"),
        "output_stim_start_ms": g.get("output_stim_start_ms"),
        "output_stim_stop_ms": g.get("output_stim_stop_ms"),
        "output_bin_ms": g.get("output_bin_ms"),
        "output_smooth_mode": g.get("output_smooth_mode"),
        "output_peak_window_ms": g.get("output_peak_window_ms"),
        "output_drop_window_ms": g.get("output_drop_window_ms"),
        "output_rebound_window_ms": g.get("output_rebound_window_ms"),
        "output_auc_window": g.get("output_auc_window"),
        "output_show_metric_points": g.get("output_show_metric_points"),
        "output_metric_label_points": g.get("output_metric_label_points"),
        "output_metric_marker_size": g.get("output_metric_marker_size"),
        "output_linewidth": g.get("output_linewidth"),
        "output_stim_linewidth": g.get("output_stim_linewidth"),
        "output_metric_linewidth": g.get("output_metric_linewidth"),
        "output_metric_window_alpha": g.get("output_metric_window_alpha"),
        "output_shade_alpha": g.get("output_shade_alpha"),
    }


def input_opts_from_globals(g: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plot_inputs_mean": g.get("plot_inputs_mean"),
        "plot_input_raster": g.get("plot_input_raster"),
        "show_input_std": g.get("show_input_std"),
        "input_source": g.get("input_source"),
        "input_std_mode": g.get("input_std_mode"),
        "input_groups": g.get("input_groups"),
        "input_bin_ms": g.get("input_bin_ms"),
        "input_smooth_ms": g.get("input_smooth_ms"),
        "input_raster_trial_idx": g.get("input_raster_trial_idx"),
        "input_raster_max_trains": g.get("input_raster_max_trains"),
        "input_raster_win_size": g.get("input_raster_win_size"),
        "input_raster_style": g.get("input_raster_style"),
        "input_plot_window": g.get("input_plot_window"),
        "input_legend_loc": g.get("input_legend_loc"),
        "compare_input_layout": g.get("compare_input_layout"),
        "compare_show_input_std": g.get("compare_show_input_std"),
        "input_linewidth": g.get("input_linewidth"),
        "input_shade_alpha": g.get("input_shade_alpha"),
        "input_output_linewidth": g.get("input_output_linewidth"),
        "input_raster_linewidth": g.get("input_raster_linewidth"),
        "input_stim_linewidth": g.get("input_stim_linewidth"),
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
        "output_metric_mode": g.get("output_metric_mode"),
        "output_metric_window_ms": g.get("output_metric_window_ms"),
    })
    return run_output_metrics(
        sel,
        opts,
        save_analysis=bool(g.get("save_analysis", False)),
    )


def run_spike_stats_from_globals(g: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sel = get_selection_from_globals(g)
    run_dir, res = resolve_single(sel)
    stats_single = analysis.summarize_spike_trials(res, plot=True, print_summary=False)
    _save_fig(
        plt.gcf(),
        analysis.plot_dir_for_run(run_dir) / "spike_stats.png",
        enabled=bool(g.get("save_plots", False)),
        dpi=int(g.get("plots_dpi", 150)),
    )
    _save_json(
        stats_single,
        analysis.analysis_dir_for_run(run_dir) / "spike_stats.json",
        enabled=bool(g.get("save_analysis", False)),
    )
    return stats_single


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

    out_selection = widgets.Output()

    base_dir = g.get("BASE_DIR")
    cells = analysis.list_cells(base_dir) or [g.get("cell_name")]
    cell_dd = widgets.Dropdown(options=cells, value=g.get("cell_name"), description="Cell")
    tunes = analysis.list_tunes(base_dir, cell_dd.value) or [g.get("tunes_dir")]
    tunes_dd = widgets.Dropdown(options=tunes, value=g.get("tunes_dir"), description="Tunes")
    models = analysis.list_models(base_dir, cell_dd.value, tunes_dd.value) or [g.get("model_dir")]
    model_dd = widgets.Dropdown(options=models, value=g.get("model_dir"), description="Model")

    compare_list_sel = widgets.SelectMultiple(
        options=[],
        value=(),
        description="Compare list",
        rows=10,
    )
    compare_list_sel.layout = widgets.Layout(width="80%", height="200px")
    compare_list_paths_txt = widgets.Textarea(
        value=_compare_list_paths_text(g.get("compare_list_paths", []) or []),
        description="Compare paths",
        layout=widgets.Layout(width="80%", height="90px"),
    )
    compare_list_clear_btn = widgets.Button(description="Clear selection")
    compare_paths_cb = widgets.Checkbox(
        value=bool(g.get("compare_list_paths_enabled", True)),
        description="Use compare paths",
    )
    selection_help_btn = widgets.Button(description="Help")
    selection_help_btn.layout = widgets.Layout(width="80px", flex="0 0 auto")

    def _refresh_runs(*_):
        base = g.get("CELLS_DIR") / cell_dd.value / tunes_dd.value / model_dd.value / "output_data"
        names = [analysis.run_label(p) for p in analysis.collect_run_candidates(base)]
        options: list[tuple[str, str]] = [(n, n) for n in names]
        options.extend(_compare_list_dir_options(g, base_dir))
        compare_list_sel.options = options
        valid_vals = {val for _, val in options}
        compare_list_sel.value = tuple([n for n in compare_list_sel.value if n in valid_vals])

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
    g["save_plots_cb"] = save_plots_cb
    g["save_analysis_cb"] = save_analysis_cb
    g["compare_list_sel"] = compare_list_sel
    g["compare_list_paths_txt"] = compare_list_paths_txt
    g["compare_paths_cb"] = compare_paths_cb

    selection_help_btn.on_click(lambda *_: _print_help(out_selection, HELP_SELECTION))
    compare_list_paths_txt.disabled = not compare_paths_cb.value
    compare_paths_cb.observe(lambda *_: setattr(compare_list_paths_txt, "disabled", not compare_paths_cb.value), names="value")

    display(widgets.HBox([cell_dd, tunes_dd, model_dd]))
    display(widgets.HBox([compare_list_sel, compare_list_clear_btn]))
    display(compare_list_paths_txt)
    display(widgets.HBox([save_plots_cb, save_analysis_cb, compare_paths_cb, selection_help_btn]))
    display(out_selection)


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

    outputs_raster_cb = widgets.Checkbox(value=g.get("plot_raster"), description="Output raster")
    outputs_style_dd = widgets.Dropdown(options=["dot", "line"], value=g.get("raster_style"), description="Raster style")
    outputs_win_txt = widgets.FloatText(value=g.get("win_size"), description="Win size")
    window_start_txt = widgets.Text(value="" if g.get("plot_window")[0] is None else str(g.get("plot_window")[0]), description="tstart")
    window_end_txt = widgets.Text(value="" if g.get("plot_window")[1] is None else str(g.get("plot_window")[1]), description="tstop")
    output_stim_start_txt = widgets.Text(
        value="" if g.get("output_stim_start_ms") is None else str(g.get("output_stim_start_ms")),
        description="Stim start",
    )
    output_stim_stop_txt = widgets.Text(
        value="" if g.get("output_stim_stop_ms") is None else str(g.get("output_stim_stop_ms")),
        description="Stim stop",
    )

    output_curve_mode_dd = widgets.Dropdown(options=["raw", "normalized"], value=g.get("output_curve_mode"), description="Curve mode")
    output_norm_mode_dd = widgets.Dropdown(options=["avg", "peak"], value=g.get("output_norm_mode"), description="Norm mode")
    outputs_norm_txt = widgets.Text(value="" if g.get("multi_norm_fr") is None else str(g.get("multi_norm_fr")), description="Norm FR")
    output_bin_txt = widgets.Text(value="" if g.get("output_bin_ms") is None else str(g.get("output_bin_ms")), description="Curve bin ms")
    output_smooth_mode_dd = widgets.Dropdown(options=["causal", "center"], value=g.get("output_smooth_mode"), description="Curve smooth mode")

    shade_val = g.get("multi_shade_mode")
    outputs_shade_dd = widgets.Dropdown(options=["none", "sem", "std"], value="none" if shade_val is None else shade_val, description="Shade")
    outputs_compare_layout_dd = widgets.Dropdown(options=["side-by-side", "stacked", "overlay"], value=g.get("compare_output_layout"), description="Compare layout")

    preset_path_default = g.get("compare_preset_path") or "modules_local/analysis/analysis_presets/paper_compare.json"
    compare_preset_cb = widgets.Checkbox(
        value=bool(g.get("compare_preset_path")),
        description="Paper compare",
    )

    outputs_btn = widgets.Button(description="Run output plots")
    outputs_btn.layout = widgets.Layout(width="160px", flex="0 0 auto")
    outputs_help_btn = widgets.Button(description="Help")
    outputs_help_btn.layout = widgets.Layout(width="80px", flex="0 0 auto")

    def _on_outputs(_):
        sync_common_from_globals(g)
        g["compare_preset_path"] = preset_path_default if compare_preset_cb.value else None
        g["plot_outputs"] = True
        g["plot_output_curve"] = True
        g["plot_raster"] = outputs_raster_cb.value
        g["raster_style"] = outputs_style_dd.value
        g["win_size"] = float(outputs_win_txt.value)
        g["plot_window"] = (
            analysis.parse_optional_float(window_start_txt.value),
            analysis.parse_optional_float(window_end_txt.value),
        )
        g["output_stim_start_ms"] = analysis.parse_optional_float(output_stim_start_txt.value)
        g["output_stim_stop_ms"] = analysis.parse_optional_float(output_stim_stop_txt.value)
        g["output_curve_mode"] = output_curve_mode_dd.value
        g["output_norm_mode"] = output_norm_mode_dd.value
        g["output_bin_ms"] = analysis.parse_optional_float(output_bin_txt.value)
        g["output_smooth_mode"] = output_smooth_mode_dd.value
        shade_val_local = outputs_shade_dd.value
        g["multi_shade_mode"] = None if shade_val_local in ("none", "", None) else shade_val_local
        g["multi_norm_fr"] = analysis.parse_optional_float(outputs_norm_txt.value)
        g["compare_output_layout"] = outputs_compare_layout_dd.value

        with out_outputs:
            out_outputs.clear_output()
            run_output_plots_from_globals(g)

    outputs_btn.on_click(_on_outputs)
    outputs_help_btn.on_click(lambda *_: _print_help(out_outputs, HELP_OUTPUTS))

    g["out_outputs"] = out_outputs
    g["_on_outputs"] = _on_outputs

    display(
        widgets.VBox([
            widgets.HBox([outputs_btn, outputs_help_btn, compare_preset_cb, outputs_raster_cb, outputs_style_dd, outputs_win_txt]),
            widgets.HBox([output_bin_txt, output_smooth_mode_dd, outputs_shade_dd, outputs_compare_layout_dd]),
            widgets.HBox([window_start_txt, window_end_txt, output_stim_start_txt, output_stim_stop_txt]),
            widgets.HBox([output_curve_mode_dd, output_norm_mode_dd, outputs_norm_txt]),
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
    inputs_source_dd = widgets.Dropdown(
        options=["auto", "saved", "stats"],
        value=g.get("input_source", "auto"),
        description="Input source",
    )
    inputs_std_mode_dd = widgets.Dropdown(
        options=["std", "sem"],
        value=g.get("input_std_mode", "std"),
        description="Std mode",
    )
    legend_options = [
        "best",
        "upper right",
        "upper left",
        "lower right",
        "lower left",
        "right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
        "none",
    ]
    legend_default = g.get("input_legend_loc") or "best"
    if legend_default not in legend_options:
        legend_options = [legend_default] + legend_options
    inputs_legend_dd = widgets.Dropdown(
        options=legend_options,
        value=legend_default,
        description="Legend",
    )

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
    inputs_btn.layout = widgets.Layout(width="160px", flex="0 0 auto")
    inputs_help_btn = widgets.Button(description="Help")
    inputs_help_btn.layout = widgets.Layout(width="80px", flex="0 0 auto")

    def _on_inputs(_):
        sync_common_from_globals(g)
        g["plot_inputs_mean"] = inputs_mean_cb.value
        g["plot_input_raster"] = inputs_raster_cb.value
        g["show_input_std"] = inputs_std_cb.value
        g["input_source"] = inputs_source_dd.value
        g["input_std_mode"] = inputs_std_mode_dd.value
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
        g["input_legend_loc"] = inputs_legend_dd.value
        g["compare_input_layout"] = compare_layout_dd.value
        g["compare_show_input_std"] = compare_std_cb.value

        with out_inputs:
            out_inputs.clear_output()
            run_input_plots_from_globals(g)

    inputs_btn.on_click(_on_inputs)
    inputs_help_btn.on_click(lambda *_: _print_help(out_inputs, HELP_INPUTS))

    g["out_inputs"] = out_inputs
    g["_on_inputs"] = _on_inputs

    display(
        widgets.VBox([
            widgets.HBox([inputs_btn, inputs_help_btn, inputs_mean_cb, inputs_raster_cb, inputs_std_cb, inputs_source_dd]),
            widgets.HBox([inputs_groups_txt, inputs_bin_txt, inputs_smooth_txt, inputs_std_mode_dd, inputs_legend_dd]),
            widgets.HBox([raster_trial_txt, raster_max_txt, raster_win_txt, raster_style_dd]),
            widgets.HBox([input_window_start_txt, input_window_end_txt]),
            widgets.HBox([compare_layout_dd, compare_std_cb]),
            out_inputs,
        ])
    )

    if g.get("auto_run_inputs"):
        _on_inputs(None)

def build_extra_ui(g: Dict[str, Any]) -> None:
    if not g.get("use_widgets", True):
        print("Widgets disabled (use_widgets=False).")
        return
    widgets = _maybe_import_widgets()
    display, _ = _maybe_import_display()
    if widgets is None or display is None:
        print("Widgets not enabled or ipywidgets unavailable.")
        return

    mode_dd = widgets.Dropdown(
        options=[
            ("Output metrics (table)", "output_metrics"),
            ("Compare configs (cell/geom/syn)", "compare_configs"),
            ("Compare outputs (plots)", "compare_outputs"),
            ("Compare inputs (plots)", "compare_inputs"),
            ("Input sampling (preview)", "input_sampling"),
            ("Snapshot compare", "snapshot_compare"),
            ("Single-run tables", "single_tables"),
            ("Spike stats", "spike_stats"),
            ("IClamp analysis", "iclamp"),
        ],
        value=g.get("extra_mode", "output_metrics"),
        description="Extra mode",
    )

    cfg_cell_cb = widgets.Checkbox(
        value=bool(g.get("extra_compare_cell_tables", False)),
        description="Cell tables",
    )
    cfg_geom_cb = widgets.Checkbox(
        value=bool(g.get("extra_compare_geometry_tables", False)),
        description="Geometry tables",
    )
    cfg_syn_cb = widgets.Checkbox(
        value=bool(g.get("extra_compare_synapse_tables", True)),
        description="Synapse tables",
    )
    cfg_diff_cb = widgets.Checkbox(
        value=bool(g.get("extra_compare_diff_only", True)),
        description="Diff only",
    )

    syn_weight_cb = widgets.Checkbox(
        value=bool(g.get("extra_synapse_weight_plot", False)),
        description="Synapse weight hist",
    )
    syn_dist_cb = widgets.Checkbox(
        value=bool(g.get("extra_synapse_distance_plot", False)),
        description="Synapse distance hist",
    )
    syn_groups_txt = widgets.Text(
        value=",".join(g.get("extra_synapse_groups") or []),
        description="Syn groups",
        layout=widgets.Layout(width="60%"),
    )
    syn_weight_bin_txt = widgets.Text(
        value=str(g.get("extra_synapse_weight_bin", 0.1)),
        description="Weight bin",
        layout=widgets.Layout(width="200px"),
    )
    syn_dist_bin_txt = widgets.Text(
        value=str(g.get("extra_synapse_distance_bin", 25.0)),
        description="Dist bin",
        layout=widgets.Layout(width="200px"),
    )
    syn_density_cb = widgets.Checkbox(
        value=bool(g.get("extra_synapse_density", True)),
        description="Density",
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
        widgets.HBox([cfg_cell_cb, cfg_geom_cb, cfg_syn_cb, cfg_diff_cb]),
        widgets.HBox([syn_weight_cb, syn_dist_cb, syn_density_cb]),
        widgets.HBox([syn_groups_txt, syn_weight_bin_txt, syn_dist_bin_txt]),
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
    metrics_show_delta_cb = widgets.Checkbox(
        value=bool(g.get("output_metrics_show_delta", True)),
        description="Show deltas",
    )
    metrics_highlight_cb = widgets.Checkbox(
        value=bool(g.get("output_metrics_highlight_best", True)),
        description="Highlight best",
    )
    metrics_box = widgets.VBox([
        widgets.HBox([metrics_ref_dd, metrics_show_params_cb]),
        widgets.HBox([metrics_show_delta_cb, metrics_highlight_cb]),
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
        snap_box.layout.display = "flex" if mode == "snapshot_compare" else "none"

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
        g["extra_compare_cell_tables"] = cfg_cell_cb.value
        g["extra_compare_geometry_tables"] = cfg_geom_cb.value
        g["extra_compare_synapse_tables"] = cfg_syn_cb.value
        g["extra_compare_diff_only"] = cfg_diff_cb.value

        g["extra_synapse_weight_plot"] = syn_weight_cb.value
        g["extra_synapse_distance_plot"] = syn_dist_cb.value
        g["extra_synapse_density"] = syn_density_cb.value

        g["extra_synapse_groups"] = analysis.parse_groups(syn_groups_txt.value)
        weight_bin = analysis.parse_optional_float(syn_weight_bin_txt.value)
        dist_bin = analysis.parse_optional_float(syn_dist_bin_txt.value)
        if weight_bin is not None:
            g["extra_synapse_weight_bin"] = weight_bin
        if dist_bin is not None:
            g["extra_synapse_distance_bin"] = dist_bin

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
        g["output_metrics_show_delta"] = metrics_show_delta_cb.value
        g["output_metrics_highlight_best"] = metrics_highlight_cb.value
        ref_val = metrics_ref_dd.value
        g["output_metrics_ref_label"] = None if ref_val in (None, "", "(none)") else ref_val
        g["input_sample_path"] = sample_path_txt.value.strip() or None
        g["input_sample_source"] = sample_source_dd.value
        g["input_sample_run"] = sample_run_dd.value
        g["input_sample_groups"] = list(sample_groups_sel.value)
        g["input_sample_runs"] = int(sample_runs_txt.value)
        g["input_sample_show_std"] = sample_std_cb.value
        g["input_sample_show_ref"] = sample_ref_cb.value
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
            elif mode == "compare_outputs":
                run_output_plots_from_globals(g)
            elif mode == "compare_inputs":
                run_input_plots_from_globals(g)
            elif mode == "input_sampling":
                run_input_sampling_from_globals(g)
            elif mode == "snapshot_compare":
                g["extra_snapshot_tables"] = True
                run_snapshot_compare_from_globals(g)
            elif mode == "single_tables":
                run_extra_tables_from_globals(g)
            elif mode == "spike_stats":
                run_spike_stats_from_globals(g)
            elif mode == "iclamp":
                run_iclamp_analysis_from_globals(g)
            else:
                metrics = run_output_metrics_from_globals(g)
                if metrics:
                    show_params = bool(g.get("output_metrics_show_params", True))
                    ref_label = g.get("output_metrics_ref_label")
                    show_delta = bool(g.get("output_metrics_show_delta", False))
                    highlight_best = bool(g.get("output_metrics_highlight_best", False))
                    if isinstance(metrics, dict) and all(isinstance(v, dict) for v in metrics.values()):
                        show_md(
                            format_output_metrics_tables_columns(
                                metrics,
                                title="Output metrics",
                                show_params=show_params,
                                reference_label=ref_label,
                                show_deltas=show_delta,
                                highlight_best=highlight_best,
                            )
                        )
                    else:
                        sel = get_selection_from_globals(g)
                        label = analysis.run_label(analysis.resolve_run(sel["base"], sel["run_single"]))
                        show_md(format_output_metrics_tables(metrics, title=f"Output metrics ({label})", show_params=show_params))

    run_btn.on_click(_on_run)
    extra_help_btn.on_click(lambda *_: _print_help(out_extra, HELP_EXTRA))

    g["extra_mode_dd"] = mode_dd
    g["out_extra"] = out_extra

    display(
        widgets.VBox([
            widgets.HBox([mode_dd, run_btn, extra_help_btn]),
            cfg_box,
            metrics_box,
            sample_box,
            snap_box,
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
    def _clear_compare_list(_):
        compare_list_sel.value = ()
        compare_list_paths_txt.value = ""

    compare_list_clear_btn.on_click(_clear_compare_list)
