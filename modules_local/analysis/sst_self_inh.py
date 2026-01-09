"""
Helpers for SST self-inhibition (PN -> SST GABAB curve adjustments).

Intended to back `extra_notebooks/5.2.3.1_PN_SST_GABABinh.ipynb`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import re

import numpy as np
import pandas as pd

TIME_CANDIDATES = [
    r"^time$",
    r"^t$",
    r"^t_?ms$",
    r"^(ms|msec|millisecond)s?$",
    r"^t_?s$",
    r"^(s|sec|second)s?$",
]
RATE_CANDIDATES = [
    r"^(rate|r|firing(_rate)?)$",
    r"^(hz)$",
    r"^(spikes?_per_?s|spikes?/s)$",
]


def find_scp_root(start: Path) -> Path:
    """Walk upward to locate the SCP repo root (contains cells/ and run_pipeline.py)."""
    for p in [start] + list(start.parents):
        if (p / "cells").is_dir() and (p / "run_pipeline.py").is_file():
            return p
    return start


def _find_col(patterns: Sequence[str], cols: Sequence[str]) -> Optional[int]:
    cols_lower = [c.lower() for c in cols]
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for i, c in enumerate(cols_lower):
            if rx.match(c):
                return i
    return None


def inspect_pn_csv(
    csv_path: Union[str, Path],
    time_col: Optional[str] = None,
    rate_col: Optional[str] = None,
    max_rows: int = 5,
) -> Dict[str, Any]:
    """Inspect a PN rate CSV and print basic metadata and inferred dt."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    cols = list(df.columns)

    tidx = cols.index(time_col) if (time_col in cols) else None
    ridx = cols.index(rate_col) if (rate_col in cols) else None

    if tidx is None:
        tidx = _find_col(TIME_CANDIDATES, cols)
    if ridx is None:
        ridx = _find_col(RATE_CANDIDATES, cols)

    if tidx is None and len(df.columns) > 0:
        first = df.iloc[:, 0]
        if np.issubdtype(first.dtype, np.number):
            vals = first.values
            if vals.size >= 3 and np.all(np.diff(vals[: min(len(vals), 1000)]) > 0):
                tidx = 0

    if ridx is None:
        candidate_idxs = [i for i in range(len(cols)) if np.issubdtype(df.iloc[:, i].dtype, np.number)]
        if tidx == 0 and len(candidate_idxs) > 1:
            ridx = 1
        elif candidate_idxs:
            ridx = candidate_idxs[-1]

    t = df.iloc[:, tidx].to_numpy(dtype=float) if tidx is not None else None
    r = df.iloc[:, ridx].to_numpy(dtype=float) if ridx is not None else None

    time_unit = None
    dt_ms = None
    if t is not None and t.size > 1:
        diffs = np.diff(t[: min(len(t), 500)])
        diffs = diffs[np.isfinite(diffs)]
        med_dt = float(np.median(diffs)) if diffs.size else None
        if med_dt is not None:
            if 1e-4 <= med_dt <= 1e-1:
                time_unit = "s"
                dt_ms = med_dt * 1000.0
            else:
                time_unit = "ms"
                dt_ms = med_dt

    print("=== PN CSV Inspection ===")
    print(f"Path: {csv_path}")
    print(f"Columns: {cols}")
    print(f"Detected time column index: {tidx}  ({cols[tidx] if tidx is not None else 'None'})")
    print(f"Detected rate column index: {ridx}  ({cols[ridx] if ridx is not None else 'None'})")
    if t is not None:
        neg = int(np.sum(t < 0))
        print(f"Time unit guess: {time_unit} | dt_ms≈ {None if dt_ms is None else round(dt_ms, 4)}")
        print(f"Time span: {round(t[0],4)} -> {round(t[-1],4)} ({time_unit or 'unknown'})")
        print(f"Negative time samples: {neg}")
        if dt_ms is not None:
            print(f"Suggested source.bin_ms: {round(dt_ms, 4)}")
    else:
        print("No time column detected. We will need dt_ms explicitly.")
    if r is not None:
        print(
            "Rate stats (Hz): "
            f"min={np.nanmin(r):.3f}, max={np.nanmax(r):.3f}, median={np.nanmedian(r):.3f}"
        )
        print("First rows:")
        print(df.head(max_rows).to_string(index=False))
    else:
        print("No rate column detected. Please confirm which column is PN firing rate (Hz).")

    return {
        "df": df,
        "time_idx": tidx,
        "rate_idx": ridx,
        "time_unit": time_unit,
        "dt_ms": dt_ms,
    }


def load_curve(
    path: Union[str, Path],
    *,
    time_col: str = "Time",
    rate_col: str = "AvgFiringRate",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load (time_s, rate_hz) from CSV using the configured columns."""
    path = Path(path)
    df = pd.read_csv(path)
    if time_col not in df or rate_col not in df:
        raise KeyError(
            f"CSV missing required columns: {time_col!r}, {rate_col!r} "
            f"(found: {list(df.columns)})"
        )
    t = df[time_col].to_numpy(dtype=float)
    r = df[rate_col].to_numpy(dtype=float)
    return t, r


def infer_dt_s(t_s: np.ndarray) -> float:
    """Infer dt in seconds using the median diff of time samples."""
    if t_s.size < 2:
        raise ValueError("Cannot infer dt from a single time sample.")
    dt_s = float(np.median(np.diff(t_s[: min(t_s.size, 500)])))
    if not (1e-4 <= dt_s <= 1.0):
        raise ValueError(f"Unexpected dt (s): {dt_s}")
    return dt_s


def save_curve_csv(
    out_path: Union[str, Path],
    t_s: np.ndarray,
    rate_hz: np.ndarray,
    *,
    time_col: str = "Time",
    rate_col: str = "AvgFiringRate",
) -> Path:
    """Save a curve with the same column names as the source."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({time_col: t_s, rate_col: rate_hz})
    out_df.to_csv(out_path, index=False)
    return out_path


def apply_freq_xform(
    rate_hz: np.ndarray,
    freq_scale: Optional[float],
    freq_shift: Optional[float],
    *,
    clip_zero: bool = True,
) -> np.ndarray:
    """Apply per-curve rate scaling/shifting (matches input_modes_core behavior)."""
    if freq_scale is None and freq_shift is None:
        return rate_hz
    try:
        scale = 1.0 if freq_scale is None else float(freq_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"freq_scale must be numeric or None (got {freq_scale!r})") from exc
    try:
        shift = 0.0 if freq_shift is None else float(freq_shift)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"freq_shift must be numeric or None (got {freq_shift!r})") from exc
    if scale != 1.0 or shift != 0.0:
        rate_hz = rate_hz * scale + shift
        if clip_zero:
            rate_hz = np.maximum(rate_hz, 0.0)
    return rate_hz


def apply_gabab_simple(
    r_hz: np.ndarray,
    dt_s: float,
    tau_s: float,
    *,
    init: str = "match",
    alpha: float = 1.0,
    robust_norm: bool = False,
    pctl: float = 99.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Advisor model: integrate S and map I = r * (1 - alpha * S)."""
    if tau_s <= 0:
        raise ValueError("tau_s must be > 0")
    r = np.asarray(r_hz, dtype=float)
    if robust_norm:
        r_ref = np.percentile(r, pctl)
    else:
        r_ref = r.max()
    r_ref = max(r_ref, 1e-12)
    r_norm = r / r_ref

    S = np.zeros_like(r_norm)
    S[0] = r_norm[0] if init == "match" else 0.0

    coef = dt_s / tau_s
    for i in range(1, r.size):
        S[i] = S[i - 1] + coef * (r_norm[i - 1] - S[i - 1])
    S = np.clip(S, 0.0, 1.0)

    I = r * (1.0 - alpha * S)
    I[I < 0] = 0.0
    return I, S


def apply_gabab_delayed(
    r_hz: np.ndarray,
    dt_s: float,
    tau_s: float,
    *,
    delay_ms: float = 0.0,
    alpha: float = 1.0,
    init: str = "match",
    robust_norm: bool = False,
    pctl: float = 99.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advisor model with delayed r_norm drive feeding S(t)."""
    if tau_s <= 0:
        raise ValueError("tau_s must be > 0")
    r = np.asarray(r_hz, dtype=float)
    N = r.size

    r_ref = np.percentile(r, pctl) if robust_norm else r.max()
    r_ref = max(r_ref, 1e-12)
    r_norm = r / r_ref

    k = int(round((delay_ms / 1000.0) / max(dt_s, 1e-12)))
    if k <= 0:
        r_drive = r_norm
    elif k >= N:
        base = r_norm[0] if init == "match" else 0.0
        r_drive = np.full_like(r_norm, base)
    else:
        base = r_norm[0] if init == "match" else 0.0
        r_drive = np.empty_like(r_norm)
        r_drive[:k] = base
        r_drive[k:] = r_norm[:-k]

    S = np.zeros_like(r_norm)
    S[0] = r_norm[0] if init == "match" else 0.0
    coef = dt_s / tau_s
    for i in range(1, N):
        S[i] = S[i - 1] + coef * (r_drive[i - 1] - S[i - 1])
    S = np.clip(S, 0.0, 1.0)

    I = r * (1.0 - alpha * S)
    I[I < 0] = 0.0
    return I, S, r_drive


def apply_legacy_exponential(
    t_s: np.ndarray,
    r_hz: np.ndarray,
    *,
    stim_delay_ms: float = 100.0,
    beta: float = 0.5,
    tau_ms: float = 200.0,
) -> np.ndarray:
    """Legacy exponential suppression rule (pre-ODE reference)."""
    t_ms = np.asarray(t_s, dtype=float) * 1000.0
    r = np.asarray(r_hz, dtype=float)
    out = np.empty_like(r)
    mask = t_ms <= stim_delay_ms
    out[mask] = r[mask]
    if np.any(~mask):
        inh_multi = beta * (1.0 - np.exp(-(t_ms[~mask] - stim_delay_ms) / tau_ms))
        out[~mask] = r[~mask] * (1.0 - inh_multi)
    return out


def plot_gabab_simple_suite(
    t_s: np.ndarray,
    r_hz: np.ndarray,
    *,
    baseline_tau_s: float = 0.01,
    init_mode: str = "match",
    tau_sweep: Optional[Sequence[float]] = None,
    show_alpha_sweep: bool = True,
    alpha_sweep: Optional[Sequence[float]] = None,
    robust_norm: bool = False,
    pctl: float = 99.0,
    title_prefix: str = "PN -> SST",
) -> Dict[str, np.ndarray]:
    """Plot baseline + tau sweep (+ optional alpha sweep) for simple GABAB model."""
    import matplotlib.pyplot as plt

    tau_sweep = list(tau_sweep) if tau_sweep is not None else [0.005, 0.01, 0.025, 0.05, 0.1, 0.2]
    alpha_sweep = list(alpha_sweep) if alpha_sweep is not None else [0.2, 0.4, 0.6, 0.8, 1.0]

    dt_s = infer_dt_s(np.asarray(t_s, dtype=float))
    I_base, S_base = apply_gabab_simple(
        r_hz,
        dt_s,
        baseline_tau_s,
        init=init_mode,
        alpha=1.0,
        robust_norm=robust_norm,
        pctl=pctl,
    )

    plt.figure()
    plt.plot(t_s, r_hz, label="PN original (Hz)")
    plt.plot(t_s, I_base, label=f"PN adjusted (Hz), tau={baseline_tau_s:.3f}s")
    plt.title(f"{title_prefix}: Original vs Adjusted")
    plt.xlabel("Time (s)")
    plt.ylabel("Rate (Hz)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(t_s, S_base, label=f"S(t), tau={baseline_tau_s:.3f}s")
    plt.title("GABA_B-like inhibition state S(t)")
    plt.xlabel("Time (s)")
    plt.ylabel("S (0..1)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    for tau in tau_sweep:
        I_tau, _ = apply_gabab_simple(
            r_hz,
            dt_s,
            tau,
            init=init_mode,
            alpha=1.0,
            robust_norm=robust_norm,
            pctl=pctl,
        )
        plt.plot(t_s, I_tau, label=f"tau={tau:.3f}s")
    plt.title("Effect of tau on adjusted PN -> SST drive")
    plt.xlabel("Time (s)")
    plt.ylabel("Adjusted rate (Hz)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if show_alpha_sweep:
        plt.figure()
        for a in alpha_sweep:
            I_a, _ = apply_gabab_simple(
                r_hz,
                dt_s,
                baseline_tau_s,
                init=init_mode,
                alpha=a,
                robust_norm=robust_norm,
                pctl=pctl,
            )
            plt.plot(t_s, I_a, label=f"alpha={a:.2f}")
        plt.title("Optional mapping strength alpha (I = r * (1 - alpha * S))")
        plt.xlabel("Time (s)")
        plt.ylabel("Adjusted rate (Hz)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {"I_base": I_base, "S_base": S_base}


def plot_gabab_delayed_suite(
    t_s: np.ndarray,
    r_hz: np.ndarray,
    *,
    tau_s: float = 0.01,
    delay_ms: float = 50.0,
    alpha: float = 1.0,
    init_mode: str = "match",
    robust_norm: bool = False,
    pctl: float = 99.0,
    delay_sweep_ms: Optional[Sequence[float]] = None,
    title_prefix: str = "PN -> SST",
) -> Dict[str, np.ndarray]:
    """Plot baseline + delay sweep for delayed GABAB model."""
    import matplotlib.pyplot as plt

    delay_sweep_ms = list(delay_sweep_ms) if delay_sweep_ms is not None else [0.0, 25.0, 50.0, 75.0, 100.0, 150.0]
    dt_s = infer_dt_s(np.asarray(t_s, dtype=float))

    I_base, S_base, r_drive_base = apply_gabab_delayed(
        r_hz,
        dt_s,
        tau_s,
        delay_ms=delay_ms,
        alpha=alpha,
        init=init_mode,
        robust_norm=robust_norm,
        pctl=pctl,
    )

    plt.figure()
    plt.plot(t_s, r_hz, label="PN original (Hz)")
    plt.plot(
        t_s,
        I_base,
        label=f"Adjusted (Hz)  tau={tau_s:.3f}s, delay={delay_ms:.0f}ms, alpha={alpha:.2f}",
    )
    plt.title(f"{title_prefix}: Original vs Adjusted (with delay)")
    plt.xlabel("Time (s)")
    plt.ylabel("Rate (Hz)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(t_s, r_drive_base, label="r_drive (delayed r_norm)")
    plt.plot(t_s, S_base, label="S(t)")
    plt.title("Drive vs inhibition state")
    plt.xlabel("Time (s)")
    plt.ylabel("Unitless")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    for dms in delay_sweep_ms:
        I_d, _, _ = apply_gabab_delayed(
            r_hz,
            dt_s,
            tau_s,
            delay_ms=dms,
            alpha=alpha,
            init=init_mode,
            robust_norm=robust_norm,
            pctl=pctl,
        )
        plt.plot(t_s, I_d, label=f"delay={dms:.0f}ms")
    plt.title(f"Delay sweep (tau={tau_s:.3f}s, alpha={alpha:.2f})")
    plt.xlabel("Time (s)")
    plt.ylabel("Adjusted rate (Hz)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {"I_base": I_base, "S_base": S_base, "r_drive_base": r_drive_base}


def plot_multi_curve_comparison(
    curve_specs: Sequence[Dict[str, Any]],
    *,
    src_path: Union[str, Path],
    repo_root: Optional[Path] = None,
    time_col: str = "Time",
    rate_col: str = "AvgFiringRate",
    show_original: bool = True,
    original_label: str = "PN original",
    default_tau_s: float = 0.01,
    default_delay_ms: float = 0.0,
    default_alpha: float = 1.0,
    default_init: str = "match",
    default_robust_norm: bool = False,
    default_pctl: float = 99.0,
    clip_zero_default: bool = True,
    title: str = "PN -> SST comparison (mixed curves)",
) -> List[Tuple[np.ndarray, np.ndarray, str, Dict[str, Any]]]:
    """Plot a mixed set of precomputed and generated curves using per-entry specs."""
    import matplotlib.pyplot as plt

    if not curve_specs and not show_original:
        raise ValueError("No curves configured; add entries to curve_specs or set show_original=True.")

    src_path = Path(src_path)
    repo_root = Path(repo_root) if repo_root is not None else None

    def _resolve_path(p: Union[str, Path]) -> Path:
        p = Path(p)
        if not p.is_absolute() and repo_root is not None:
            p = repo_root / p
        return p

    curves: List[Tuple[np.ndarray, np.ndarray, str, Dict[str, Any]]] = []

    if show_original:
        t0, r0 = load_curve(src_path, time_col=time_col, rate_col=rate_col)
        curves.append((t0, r0, original_label, {}))

    for spec in curve_specs:
        label = spec.get("label")
        plot_kwargs = dict(spec.get("plot_kwargs", {}) or {})
        time_col_spec = spec.get("time_col", time_col)
        rate_col_spec = spec.get("rate_col", rate_col)
        freq_scale = spec.get("freq_scale", None)
        freq_shift = spec.get("freq_shift", None)
        clip_zero = bool(spec.get("clip_zero", clip_zero_default))

        if spec.get("path"):
            p = _resolve_path(spec["path"])
            t, r = load_curve(p, time_col=time_col_spec, rate_col=rate_col_spec)
            r = apply_freq_xform(r, freq_scale, freq_shift, clip_zero=clip_zero)
            if not label:
                label = p.name
            curves.append((t, r, label, plot_kwargs))
            continue

        base_path = _resolve_path(spec.get("source_path", src_path))
        t_base, r_base = load_curve(base_path, time_col=time_col_spec, rate_col=rate_col_spec)
        if t_base.size < 2:
            raise ValueError(f"Base curve has too few samples: {base_path}")

        dt_s = infer_dt_s(t_base)
        tau_s = float(spec.get("tau_s", default_tau_s))
        delay_ms = float(spec.get("delay_ms", default_delay_ms))
        alpha = float(spec.get("alpha", default_alpha))
        init = str(spec.get("init", default_init))
        robust_norm = bool(spec.get("robust_norm", default_robust_norm))
        pctl = float(spec.get("pctl", default_pctl))

        I, _, _ = apply_gabab_delayed(
            r_base,
            dt_s,
            tau_s,
            delay_ms=delay_ms,
            alpha=alpha,
            init=init,
            robust_norm=robust_norm,
            pctl=pctl,
        )
        I = apply_freq_xform(I, freq_scale, freq_shift, clip_zero=clip_zero)
        if not label:
            label = f"tau={tau_s:.3f}s delay={delay_ms:.0f}ms alpha={alpha:.2f}"
        curves.append((t_base, I, label, plot_kwargs))

    plt.figure()
    for t, r, label, kwargs in curves:
        plt.plot(t, r, label=label, **kwargs)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Rate (Hz)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return curves


def plot_legacy_curve(
    t_s: np.ndarray,
    r_hz: np.ndarray,
    inh_curve: np.ndarray,
    *,
    title: str = "Legacy exponential filter",
) -> None:
    """Plot legacy original vs adjusted curves."""
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(t_s, r_hz, color="g", label="Original")
    plt.plot(t_s, inh_curve, color="r", label="Legacy adjusted")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Rate (Hz)")
    plt.legend()
    plt.tight_layout()
    plt.show()
