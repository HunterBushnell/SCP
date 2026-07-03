"""Built-in homogeneous and inhomogeneous Poisson input modes."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .mode_helpers import (
    _apply_gabab_to_curve,
    _generate_homogeneous_poisson_trains,
    _generate_inhomogeneous_from_curve,
    _get_n_syn,
    _parse_gabab_cfg,
    _resolve_source_path,
)


def _mode_homogeneous_poisson(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
    geometry: Optional[Any],
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Built-in handler for mode == "homogeneous_poisson".

    Semantics:
      - Pure constant-rate Poisson drive.
      - Uses a single rate taken from source["freq"] (Hz).
      - Respects the onset anchor: drives from max(onset, tstart) to tstop.
      - Does not use per-block structure (baseline/source blocks are ignored).

    Requirements:
      - source["freq"] must be float-like; if not, raise ValueError.
      - If freq <= 0 or effective window has no duration, return n_syn empty trains.
      - n_syn is obtained via _get_n_syn(group_cfg).
      - All spikes lie in [tstart, tstop] (ms).
    """
    # Resolve sim window
    try:
        sim_tstart = float(sim_cfg["tstart"])
        sim_tstop  = float(sim_cfg["tstop"])
    except KeyError as exc:
        raise KeyError(
            f"sim_cfg is missing required key {exc!r} for homogeneous_poisson mode"
        ) from exc

    # Resolve anchors from group_cfg["time_cfg"], if present
    time_cfg = group_cfg.get("time_cfg") or {}
    anchors = time_cfg.get("anchors", {}) or {}
    onset = float(anchors.get("onset", sim_tstart))
    source_tstart = anchors.get("source_tstart")
    source_tstop = anchors.get("source_tstop")
    duration = anchors.get("duration_ms")
    jitter_tstart = anchors.get("jitter_tstart_ms", None)

    # Effective window: honor explicit source window or onset+duration when provided.
    t_start_ms = max(onset, sim_tstart)
    if source_tstart is not None:
        try:
            t_start_ms = max(float(source_tstart), sim_tstart)
        except Exception:
            pass
    if jitter_tstart is not None:
        try:
            t_start_ms = max(t_start_ms, float(jitter_tstart))
        except Exception:
            pass
    t_end_ms = sim_tstop
    if source_tstop is not None:
        try:
            t_end_ms = min(float(source_tstop), sim_tstop)
        except Exception:
            pass
    elif duration is not None:
        try:
            t_end_ms = min(t_start_ms + float(duration), sim_tstop)
        except Exception:
            pass

    # Resolve synapse count
    n_syn = _get_n_syn(group_cfg)

    # Resolve constant rate from source["freq"]
    source = (group_cfg or {}).get("source", {}) or {}
    if "freq" not in source:
        raise KeyError(
            "homogeneous_poisson mode requires source['freq'] (Hz); "
            "no 'freq' key found in source config"
        )

    try:
        rate_hz = float(source["freq"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"homogeneous_poisson mode requires source['freq'] to be float-like; "
            f"got {source['freq']!r}"
        ) from exc

    # Degenerate cases: no time or no rate => n_syn empty trains
    if n_syn <= 0 or t_end_ms <= t_start_ms or rate_hz <= 0.0:
        return [np.array([], dtype=float) for _ in range(max(n_syn, 0))]

    # Generate homogeneous Poisson trains over [t_start_ms, t_end_ms]
    trains = _generate_homogeneous_poisson_trains(
        rate_hz=rate_hz,
        t_start_ms=t_start_ms,
        t_end_ms=t_end_ms,
        n_syn=n_syn,
        rng=rng,
    )

    return trains


def _mode_inhomogeneous_poisson(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
    geometry: Optional[Any],
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Inhomogeneous Poisson driven by a rate curve (CSV or array).

    Assumptions for this implementation (current SST use case):
      - source["path"] points to a CSV with columns time_col (seconds) and rate_col (Hz),
        OR source["freq"] supplies an in-memory rate array (requires source["bin_ms"]).
      - time_col defaults to "Time", rate_col defaults to "AvgFiringRate".
      - Times < 0 are dropped; remaining times are shifted so the first sample is at 0 ms.
      - bin_ms is taken from source["bin_ms"] if provided; otherwise inferred from median Δt.
      - Optional source["gabab"] applies a GABAB-like filter to the rate curve
        before freq_scale/freq_shift. Set gabab.history="full" (default) to
        use pre-0 history, or "trimmed" to apply after trimming.
      - Optional source["freq_scale"] and source["freq_shift"] apply as:
        rates_hz = max(0, rates_hz * freq_scale + freq_shift).
      - Baseline blocks use anchors["baseline_rate_hz"] (numeric) if present; otherwise quiescent.
      - Source blocks use the rate curve, truncated/padded to the block duration
        (padding uses baseline_rate_hz or 0.0 if absent).
    """
    time_cfg = (group_cfg or {}).get("time_cfg") or {}
    anchors = time_cfg.get("anchors", {}) or {}
    blocks = time_cfg.get("blocks", []) or []
    jitter_tstart = anchors.get("jitter_tstart_ms", None)

    try:
        sim_tstart = float(sim_cfg["tstart"])
        sim_tstop = float(sim_cfg["tstop"])
    except Exception as exc:
        raise ValueError("sim_cfg must contain tstart and tstop for inhomogeneous_poisson") from exc

    n_syn = _get_n_syn(group_cfg)
    if n_syn <= 0:
        return []

    source = group_cfg.get("source", {}) or {}
    path = source.get("path")
    freq = source.get("freq")
    trim_ms = float(anchors.get("source_trim_ms", 0.0) or 0.0)

    time_col = source.get("time_col") or "Time"
    rate_col = source.get("rate_col") or "AvgFiringRate"
    bin_ms_cfg = source.get("bin_ms", None)
    gabab_cfg = _parse_gabab_cfg(source)

    bin_ms = None

    if path or (isinstance(freq, str) and freq):
        # Load curve from CSV
        import pandas as pd  # local import to avoid forcing pandas on import

        p = _resolve_source_path(str(path or freq), sim_cfg)
        if not p.is_file():
            print(f"inhomogeneous_poisson: missing rate curve file {p}; using empty trains")
            return [np.array([], dtype=float) for _ in range(n_syn)]

        df = pd.read_csv(p)
        if time_col not in df or rate_col not in df:
            raise ValueError(f"Rate curve file {p} missing required columns {time_col!r}/{rate_col!r}")

        times_ms = np.asarray(df[time_col], dtype=float) * 1000.0  # seconds → ms
        rates_hz = np.asarray(df[rate_col], dtype=float)

        if gabab_cfg and gabab_cfg["history"] == "full":
            rates_hz = _apply_gabab_to_curve(times_ms, rates_hz, gabab_cfg)

        # Drop times < 0 and shift so first sample is at 0
        keep = times_ms >= 0.0
        times_ms = times_ms[keep]
        rates_hz = rates_hz[keep]
        if times_ms.size == 0:
            raise ValueError(f"Rate curve {p} has no samples with time >= 0 ms after clipping.")
        times_ms = times_ms - times_ms[0]
    elif freq is not None:
        rates_hz = np.asarray(freq, dtype=float).ravel()
        if rates_hz.size == 0:
            raise ValueError("inhomogeneous_poisson: source['freq'] array is empty")
        if bin_ms_cfg is None:
            raise ValueError("inhomogeneous_poisson: source['bin_ms'] required for array inputs")
        bin_ms = float(bin_ms_cfg)
        if bin_ms <= 0.0:
            raise ValueError(f"bin_ms must be > 0 (got {bin_ms!r})")
        times_ms = np.arange(rates_hz.size, dtype=float) * bin_ms
        if gabab_cfg and gabab_cfg["history"] == "full":
            rates_hz = _apply_gabab_to_curve(times_ms, rates_hz, gabab_cfg)
    else:
        raise ValueError("inhomogeneous_poisson requires source['path'] or source['freq']")

    # Determine bin_ms (CSV inputs)
    if bin_ms is None:
        if bin_ms_cfg is not None:
            try:
                bin_ms = float(bin_ms_cfg)
            except Exception as exc:
                raise ValueError(f"source['bin_ms'] must be numeric (got {bin_ms_cfg!r})") from exc
        else:
            if times_ms.size < 2:
                raise ValueError("Cannot infer bin_ms from a single time sample; specify source['bin_ms'].")
            diffs = np.diff(times_ms)
            bin_ms = float(np.median(diffs))
        if bin_ms <= 0.0:
            raise ValueError(f"bin_ms must be > 0 (got {bin_ms!r})")

    # If the raw source start was earlier than the resolved start, trim the
    # leading portion so the stimulus alignment is preserved.
    if trim_ms > 0.0 and bin_ms > 0.0:
        trim_bins = int(np.floor(trim_ms / bin_ms))
        if trim_bins > 0:
            times_ms = times_ms[trim_bins:]
            rates_hz = rates_hz[trim_bins:]
            if times_ms.size:
                times_ms = times_ms - times_ms[0]

    if gabab_cfg and gabab_cfg["history"] == "trimmed":
        rates_hz = _apply_gabab_to_curve(times_ms, rates_hz, gabab_cfg)

    freq_scale_raw = source.get("freq_scale", None)
    freq_shift_raw = source.get("freq_shift", None)
    try:
        freq_scale = 1.0 if freq_scale_raw is None else float(freq_scale_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"source['freq_scale'] must be numeric or null (got {freq_scale_raw!r})"
        ) from exc
    try:
        freq_shift = 0.0 if freq_shift_raw is None else float(freq_shift_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"source['freq_shift'] must be numeric or null (got {freq_shift_raw!r})"
        ) from exc

    apply_xform = (freq_scale != 1.0) or (freq_shift != 0.0)
    if apply_xform:
        rates_hz = rates_hz * freq_scale + freq_shift
        if rates_hz.size:
            rates_hz = np.maximum(rates_hz, 0.0)

    # Build per-synapse accumulators
    trains: List[List[float]] = [[] for _ in range(n_syn)]
    baseline_rate = anchors.get("baseline_rate_hz", None)
    baseline_spec = anchors.get("baseline_spec", {}) or {}

    def _baseline_from_spec():
        if rates_hz.size == 0:
            return None
        if baseline_rate is not None:
            rate = float(baseline_rate)
            if apply_xform:
                rate = rate * freq_scale + freq_shift
                if rate < 0.0:
                    rate = 0.0
            return float(rate)
        kind = baseline_spec.get("kind")
        if kind == "from_curve":
            where = baseline_spec.get("where")
            if where == "start":
                return float(rates_hz[0])
            if where == "end":
                return float(rates_hz[-1])
            if where == "peak":
                return float(np.max(rates_hz))
        if kind == "from_curve_at":
            t_ms = float(baseline_spec.get("t_ms", 0.0))
            idx = int(np.argmin(np.abs(times_ms - t_ms)))
            return float(rates_hz[idx])
        return None

    baseline_rate = _baseline_from_spec()

    # Precompute truncated/padded rate slices for any source block
    rates_len = rates_hz.size

    for block in blocks:
        kind = block.get("kind")
        t0 = float(block.get("t_start", sim_tstart))
        t1 = float(block.get("t_end", t0))
        if t1 <= t0:
            continue
        # Clamp to simulation window just in case
        t0 = max(t0, sim_tstart)
        t1 = min(t1, sim_tstop)
        if t1 <= t0:
            continue

        if kind == "quiescent":
            continue

        if kind == "baseline":
            if jitter_tstart is not None:
                try:
                    t0 = max(t0, float(jitter_tstart))
                except Exception:
                    pass
                if t1 <= t0:
                    continue
            rate = baseline_rate
            if rate is None or rate <= 0.0:
                continue
            seg_trains = _generate_homogeneous_poisson_trains(
                rate_hz=float(rate),
                t_start_ms=t0,
                t_end_ms=t1,
                n_syn=n_syn,
                rng=rng,
            )
        elif kind == "source":
            duration = t1 - t0
            n_bins_needed = int(np.ceil(duration / bin_ms))
            if n_bins_needed <= 0:
                continue

            # Use available curve up to n_bins_needed bins; pad remainder with baseline or zeros
            avail_bins = min(rates_len, n_bins_needed)
            rates_block = rates_hz[:avail_bins]
            if avail_bins < n_bins_needed:
                pad_rate = float(baseline_rate) if (baseline_rate is not None) else 0.0
                pad = np.full(n_bins_needed - avail_bins, pad_rate, dtype=float)
                rates_block = np.concatenate([rates_block, pad])

            seg_trains = _generate_inhomogeneous_from_curve(
                rates_hz=np.asarray(rates_block, dtype=float),
                t0_ms=t0,
                bin_ms=bin_ms,
                n_syn=n_syn,
                rng=rng,
            )
            # Clip to the block window to avoid spillover beyond t1.
            seg_trains = [
                tr[(tr >= t0) & (tr <= t1)] if tr.size else tr
                for tr in seg_trains
            ]
        else:
            continue

        # Accumulate
        for i in range(n_syn):
            trains[i].extend(seg_trains[i].tolist())

    # Finalize: clip to sim window and sort
    out: List[np.ndarray] = []
    for spikes in trains:
        if not spikes:
            out.append(np.array([], dtype=float))
            continue
        arr = np.asarray(spikes, dtype=float)
        arr = arr[(arr >= sim_tstart) & (arr <= sim_tstop)]
        if arr.size == 0:
            out.append(np.array([], dtype=float))
            continue
        arr.sort()
        out.append(arr)

    return out
