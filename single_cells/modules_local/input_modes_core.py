from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np



# Modes: ###############################################################
# Shared helpers for modes

def _get_group_time_window(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Determine the active time window [t_start, t_end] for a group in ms.

    Simple rule for v1:
      - onset_ms: if set, start there; otherwise use sim['tstart'].
      - duration_ms: if set, window ends at onset + duration; otherwise sim['tstop'].
      - The window is clipped to [sim['tstart'], sim['tstop']].
    """
    t_global_start = float(sim_cfg["tstart"])
    t_global_stop = float(sim_cfg["tstop"])

    timing = group_cfg.get("timing", {}) or {}
    onset_ms = timing.get("onset_ms")
    duration_ms = timing.get("duration_ms")

    if onset_ms is None:
        t_start = t_global_start
    else:
        t_start = max(float(onset_ms), t_global_start)

    if duration_ms is None:
        t_end = t_global_stop
    else:
        t_end = min(t_start + float(duration_ms), t_global_stop)

    if t_end <= t_start:
        # Degenerate window: treat as no activity.
        return t_start, t_start

    return t_start, t_end


def _get_n_syn(group_cfg: Dict[str, Any]) -> int:
    """
    Read N_syn from group_cfg['syns']['N_syn'], defaulting to 1.
    """
    syns = group_cfg.get("syns", {}) or {}
    n_syn = syns.get("N_syn")
    if n_syn is None:
        return 1
    try:
        n_syn_int = int(n_syn)
    except (TypeError, ValueError):
        raise ValueError(f"N_syn must be an integer (got {n_syn!r})")
    if n_syn_int <= 0:
        return 0
    return n_syn_int


def _generate_homogeneous_poisson_trains(
    rate_hz: float,
    t_start_ms: float,
    t_end_ms: float,
    n_syn: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Generate homogeneous Poisson spike trains for n_syn synapses.

    Times are in ms, in simulation time, and lie within [t_start_ms, t_end_ms].
    """
    window_ms = float(t_end_ms) - float(t_start_ms)
    if n_syn <= 0 or window_ms <= 0.0 or rate_hz <= 0.0:
        return [np.array([], dtype=float) for _ in range(max(n_syn, 0))]

    window_s = window_ms / 1000.0
    lam = float(rate_hz)

    trains: List[np.ndarray] = []
    for _ in range(n_syn):
        t = 0.0
        times_ms: list[float] = []
        while True:
            isi = rng.exponential(1.0 / lam)  # seconds
            t += isi
            if t > window_s:
                break
            times_ms.append(t_start_ms + t * 1000.0)
        trains.append(np.array(times_ms, dtype=float))

    return trains

def _generate_inhomogeneous_from_curve(
    times_ms: np.ndarray,
    rates_hz: np.ndarray,
    t_start_ms: float,
    t_end_ms: float,
    n_syn: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Placeholder for an inhomogeneous Poisson generator driven by a rate curve.

    Contract (for future implementation):
      - times_ms, rates_hz define the rate(t) curve in ms and Hz.
      - Generates n_syn spike trains with rate following the curve.
      - Spikes are in ms, clipped to [t_start_ms, t_end_ms].

    Currently not implemented.
    """
    raise NotImplementedError("Inhomogeneous Poisson generator not yet implemented.")



# Default mode functions

# Mode 1: precomputed
def _mode_precomputed(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
    geometry: Optional[Any],
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Built-in handler for mode == "precomputed".

    v1 behavior:
      - load spike trains either from group_cfg["source"]["trains"]
        (inline) or from a JSON file at group_cfg["source"]["path"],
      - interpret as a list of spike-time lists (ms, sim-time),
      - clip spikes to the group's active window [t_start, t_end].
    """
    source = group_cfg.get("source", {}) or {}

    trains_raw = source.get("trains")
    path = source.get("path")

    if trains_raw is None and path is None:
        raise ValueError(
            "Mode 'precomputed' requires either source['trains'] (inline) "
            "or source['path'] (JSON file with spike trains)."
        )

    if trains_raw is None and path is not None:
        path_obj = Path(path)
        with path_obj.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "trains" in data:
            trains_raw = data["trains"]
        else:
            # Allow the top-level to be a list-of-lists as well
            if isinstance(data, list):
                trains_raw = data
            else:
                raise ValueError(
                    f"Precomputed file {path_obj} must be a list-of-lists or "
                    f"a dict with key 'trains'."
                )

    if not isinstance(trains_raw, list):
        raise TypeError(
            f"source['trains'] must be a list of spike trains (got {type(trains_raw)!r})"
        )

    t_start, t_end = _get_group_time_window(sim_cfg, group_cfg)

    spike_trains: List[np.ndarray] = []
    for i, tr in enumerate(trains_raw):
        if not isinstance(tr, (list, tuple)):
            raise TypeError(
                f"Train {i} in source['trains'] must be a list/tuple of times "
                f"(got {type(tr)!r})"
            )
        arr = np.asarray(tr, dtype=float)
        if arr.size:
            mask = (arr >= t_start) & (arr <= t_end)
            arr = arr[mask]
        spike_trains.append(arr)

    return spike_trains

# Mode 2: homogeneous poisson
def _mode_homogeneous_poisson(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
    geometry: Optional[Any],
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Built-in handler for mode == "homogeneous_poisson".

    v1 behavior:
      - read a numeric rate from group_cfg["source"]["freq"] (Hz),
      - get the group's active time window,
      - generate N_syn independent Poisson trains in that window.
    """
    source = group_cfg.get("source", {}) or {}
    freq = source.get("freq")

    try:
        rate_hz = float(freq)
    except (TypeError, ValueError):
        raise ValueError(
            f"Group '{group_cfg.get('label', '') or ''}' with mode "
            f"'homogeneous_poisson' must have numeric source['freq'] (got {freq!r})"
        )

    t_start, t_end = _get_group_time_window(sim_cfg, group_cfg)
    n_syn = _get_n_syn(group_cfg)

    spike_trains = _generate_homogeneous_poisson_trains(
        rate_hz=rate_hz,
        t_start_ms=t_start,
        t_end_ms=t_end,
        n_syn=n_syn,
        rng=rng,
    )
    return spike_trains

# Mode 3: inhomogeneous poisson (future)
def _mode_inhomogeneous_poisson(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
    geometry: Optional[Any],
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Built-in handler for mode == "inhomogeneous_poisson" (placeholder).

    Intended future behavior:
      - Read rate-curve metadata from group_cfg['source'] (e.g. path, columns).
      - Use timing fields (stim_tstart_ms, input_stim_tstart_ms, input_duration_ms, etc.)
        to align the input domain with simulation time and compute the active window.
      - Call _generate_inhomogeneous_from_curve(...) to produce spike trains.
      - Optionally add homogeneous baseline before/after the data window.

    Currently raises NotImplementedError; use 'precomputed' mode + externally
    generated spike trains until this is implemented.
    """
    raise NotImplementedError("Mode 'inhomogeneous_poisson' is not yet implemented.")







def get_default_mode_registry() -> Dict[str, Any]:
    """
    Return the default mode registry for Step 2.3.
    """
    return {
        "homogeneous_poisson": _mode_homogeneous_poisson,
        "precomputed": _mode_precomputed,
        "inhomogeneous_poisson": _mode_inhomogeneous_poisson,
    }
