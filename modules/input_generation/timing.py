"""Timing-window resolution for synaptic input generation.

The functions here convert simulation and group timing options into concrete
time anchors and non-overlapping blocks. Mode handlers consume those blocks
without needing to reinterpret raw JSON timing fields.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _calculate_timing(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
    *,
    source_jitter_delay_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute resolved timing information for a single synapse group.

    High-level:
      1) Derive concrete time anchors (ms) for this group from sim_cfg
         and group_cfg (tstart/tstop, onset, source start/stop, baseline).
      2) Build an ordered, non-overlapping list of time blocks of three kinds:
           - "quiescent" : no spikes should be generated.
           - "baseline"  : constant-rate (homogeneous) baseline spikes.
           - "source"    : main source-driven spikes (mode-specific).

    This function is mode-agnostic: it does not generate any spikes and
    does not depend on which mode (homogeneous, precomputed, etc.) is used.
    """
    anchors = _compute_time_anchors(
        sim_cfg,
        group_cfg,
        source_jitter_delay_ms=source_jitter_delay_ms,
    )
    blocks = _build_time_blocks_from_anchors(anchors)

    time_cfg = {
        "anchors": anchors,
        "blocks": blocks,
    }

    return time_cfg


def _compute_time_anchors(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
    *,
    source_jitter_delay_ms: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """
    Derive concrete time anchors for this group.

    Output keys (all in ms, floats or None where applicable):
        "sim_tstart"      : simulation start time (sim_cfg["tstart"])
        "sim_tstop"       : simulation stop time  (sim_cfg["tstop"])
        "onset"           : first time this group is allowed to produce spikes
                           (defaults to sim_tstart).
        "source_tstart"   : start of main source-driven input segment
                           (aligned to stim_tstart_ms and input_stim_tstart_ms).
        "source_tstop"    : end of main source-driven input segment.
        "baseline_rate_hz": resolved baseline rate in Hz for this group,
                           or None if there is no baseline.
    """
    tstart = float(sim_cfg["tstart"])
    tstop = float(sim_cfg["tstop"])

    timing = group_cfg.get("timing", {}) or {}
    source = group_cfg.get("source", {}) or {}

    def _as_float_or_none(val: Any, label: str) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be numeric or null (got {val!r})") from exc

    onset_raw = _as_float_or_none(timing.get("onset_ms"), "timing['onset_ms']")
    stim_tstart_raw = _as_float_or_none(timing.get("stim_tstart_ms"), "timing['stim_tstart_ms']")
    stim_jitter_raw = _as_float_or_none(timing.get("stim_jitter"), "timing['stim_jitter']")
    duration_raw = _as_float_or_none(timing.get("duration_ms"), "timing['duration_ms']")
    input_stim_raw = _as_float_or_none(
        timing.get("input_stim_tstart_ms"), "timing['input_stim_tstart_ms']"
    )
    source_tstart_raw: Optional[float] = None
    if stim_jitter_raw is not None and stim_jitter_raw < 0.0:
        raise ValueError(f"timing['stim_jitter'] must be >= 0 (got {stim_jitter_raw!r})")
    stim_jitter_delay: Optional[float] = None
    if source_jitter_delay_ms is not None:
        try:
            stim_jitter_delay = float(source_jitter_delay_ms)
        except Exception as exc:
            raise ValueError(
                f"resolved source jitter delay must be numeric or null "
                f"(got {source_jitter_delay_ms!r})"
            ) from exc
        if stim_jitter_delay < 0.0:
            raise ValueError(
                f"resolved source jitter delay must be >= 0 (got {stim_jitter_delay!r})"
            )


    def _parse_baseline_spec(
        val: Any,
        label: str = "source['baseline']",
    ) -> Tuple[Dict[str, Any], Optional[float]]:
        """
        Parse baseline specification into:
        - baseline_spec: rich descriptor dict
        - baseline_hz: numeric baseline (float) or None

        baseline_hz is only non-None for simple fixed numeric baselines.
        All tokenized / curve-dependent cases return baseline_hz = None.
        """
        # Null → no baseline
        if val is None:
            return {"kind": "none"}, None

        # Plain numeric → fixed Hz
        if isinstance(val, (int, float)):
            hz = float(val)
            return {"kind": "fixed", "hz": hz}, hz

        # String tokens
        if isinstance(val, str):
            v = val.strip()

            if v == "start":
                return {"kind": "from_curve", "where": "start"}, None

            if v == "end":
                return {"kind": "from_curve", "where": "end"}, None

            if v == "peak":
                return {"kind": "from_curve", "where": "peak"}, None

            if v.startswith("time:"):
                # e.g. "time: 234" → 234 ms
                _, _, num_str = v.partition(":")
                try:
                    t_ms = float(num_str.strip())
                except ValueError as exc:
                    raise ValueError(f"{label} malformed time spec {val!r}") from exc
                return {"kind": "from_curve_at", "t_ms": t_ms}, None

        # Anything else is invalid
        raise ValueError(
            f"{label} must be numeric, null, or a recognized string spec (got {val!r})"
        )

    baseline_spec, baseline_hz = _parse_baseline_spec(
        source.get("baseline"), "source['baseline']"
    )
     
    # Onset: default to sim start, clamped into [tstart, tstop]
    if onset_raw is None:
        onset = tstart
    else:
        onset = max(min(onset_raw, tstop), tstart)

    # stim_tstart: explicit marker in sim time for the stimulus; do not infer.
    if stim_tstart_raw is None:
        stim_tstart = None
    else:
        stim_tstart = max(min(stim_tstart_raw, tstop), tstart)

    # Source start: align source data so that its own stim event occurs
    # at stim_tstart in the simulation.
    source_tstart: Optional[float] = None
    if (stim_tstart is not None) and (input_stim_raw is not None):
        source_tstart_raw = stim_tstart - input_stim_raw
        # Clamp to onset/sim window, but remember how much we trimmed so
        # modes can discard the leading portion to keep stim alignment.
        source_tstart = max(onset, tstart, source_tstart_raw)

    # Clamp source_tstart into the simulation window (but do not force it
    # to be >= onset; that interaction is handled by the block builder).
    if source_tstart is not None:
        if source_tstart < tstart:
            source_tstart = tstart
        if source_tstart > tstop:
            source_tstart = tstop

    # Keep trim bookkeeping tied to alignment/clamping only.
    # A configured source jitter intentionally shifts the source block and
    # should not additionally trim source data.
    source_tstart_for_trim = source_tstart
    if source_tstart is not None and stim_jitter_delay is not None and stim_jitter_delay > 0.0:
        source_tstart = min(source_tstart + stim_jitter_delay, tstop)

    # Source stop: choose a duration for the main source segment.
    source_tstop: Optional[float] = None
    if source_tstart is not None:
        remaining = max(0.0, tstop - source_tstart)
        if remaining <= 0.0:
            source_tstop = source_tstart
        else:
            if duration_raw is not None:
                dur = max(0.0, duration_raw)
                dur = min(dur, remaining)
            else:
                # No explicit duration → run until the end of the simulation window
                dur = remaining

            if dur <= 0.0:
                source_tstop = source_tstart
            else:
                source_tstop = source_tstart + dur

        # Clamp to sim window
        source_tstop = min(source_tstop, tstop)

    # If we ended up with an invalid source window, drop it (baseline/quiescent only)
    if source_tstop is not None and source_tstart is not None and source_tstop <= source_tstart:
        source_tstart = None
        source_tstop = None

    # How much of the source should be trimmed (when raw start < resolved start)
    source_trim_ms: Optional[float] = None
    if source_tstart_for_trim is not None and source_tstart_raw is not None:
        delta = source_tstart_for_trim - source_tstart_raw
        source_trim_ms = max(0.0, float(delta))

    anchors: Dict[str, Optional[float]] = {
        "sim_tstart": tstart,
        "sim_tstop": tstop,
        "onset": onset,
        "source_tstart": source_tstart,
        "source_tstop": source_tstop,
        "baseline_rate_hz": baseline_hz,  # numeric baseline only
        "baseline_spec": baseline_spec,   # tokenized baseline (if any)
        "source_trim_ms": source_trim_ms,
        "jitter_tstart_ms": _as_float_or_none(sim_cfg.get("_jitter_tstart_ms"), "sim['_jitter_tstart_ms']"),
        "stim_jitter_ms": stim_jitter_raw,
        "stim_jitter_delay_ms": stim_jitter_delay,
        # Extra fields for debugging / inspection (not used by block builder):
        "stim_tstart_ms": stim_tstart,
        "input_stim_tstart_ms": input_stim_raw,
        "duration_ms": duration_raw,
        "input_duration_ms": _as_float_or_none(timing.get("input_duration_ms"), "timing['input_duration_ms']"),
    }

    return anchors


def _build_time_blocks_from_anchors(
    anchors: Dict[str, Optional[float]],
) -> List[Dict[str, Any]]:
    """
    Build an ordered, non-overlapping list of time blocks from anchors.

    Inputs (anchors):
        sim_tstart, sim_tstop, onset,
        source_tstart, source_tstop, baseline_rate_hz
    """
    sim_tstart = float(anchors.get("sim_tstart", 0.0))
    sim_tstop = float(anchors.get("sim_tstop", sim_tstart))
    onset = anchors.get("onset")
    source_tstart = anchors.get("source_tstart")
    source_tstop = anchors.get("source_tstop")
    baseline_rate = anchors.get("baseline_rate_hz")
    baseline_spec = anchors.get("baseline_spec") or {}
    baseline_active = (baseline_rate is not None and baseline_rate > 0.0) or (
        baseline_spec.get("kind") not in (None, "none")
    )

    # Clamp helpers
    def _clamp(t: Optional[float]) -> Optional[float]:
        if t is None:
            return None
        t_f = float(t)
        if t_f < sim_tstart:
            return sim_tstart
        if t_f > sim_tstop:
            return sim_tstop
        return t_f

    onset = _clamp(onset) or sim_tstart
    source_tstart = _clamp(source_tstart)
    source_tstop = _clamp(source_tstop)

    blocks: List[Dict[str, Any]] = []

    def _add_block(kind: str, t0: float, t1: float) -> None:
        if t1 <= t0:
            return
        blocks.append({"kind": kind, "t_start": float(t0), "t_end": float(t1)})

    # Case: no valid source window → only quiescent + baseline/quiescent
    if source_tstart is None or source_tstop is None or source_tstop <= source_tstart:
        _add_block("quiescent", sim_tstart, onset)
        kind = "baseline" if baseline_active else "quiescent"
        _add_block(kind, onset, sim_tstop)
        return blocks

    # Ensure ordering for source anchors
    src_start = max(min(source_tstart, sim_tstop), sim_tstart)
    src_stop = max(min(source_tstop, sim_tstop), sim_tstart)
    if src_stop <= src_start:
        _add_block("quiescent", sim_tstart, onset)
        kind = "baseline" if baseline_active else "quiescent"
        _add_block(kind, onset, sim_tstop)
        return blocks

    # 1) steady-state / quiescent before onset
    _add_block("quiescent", sim_tstart, min(onset, src_start))

    # 2) pre-source baseline or quiescent
    pre_start = max(onset, sim_tstart)
    pre_end = min(src_start, sim_tstop)
    pre_kind = "baseline" if baseline_active else "quiescent"
    _add_block(pre_kind, pre_start, pre_end)

    # 3) main source window
    _add_block("source", src_start, src_stop)

    # 4) post-source baseline or quiescent
    post_start = max(src_stop, sim_tstart)
    post_end = sim_tstop
    post_kind = "baseline" if baseline_active else "quiescent"
    _add_block(post_kind, post_start, post_end)

    return blocks
