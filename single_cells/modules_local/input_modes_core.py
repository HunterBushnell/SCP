from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np


# =====================================================================
# Shared helpers for modes
# =====================================================================


def _get_group_time_window(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Determine the active time window [t_start, t_end] for a group in ms.

    Rules (keep in sync with inputs._get_group_time_window):

      let t_global_start = sim_cfg["tstart"], t_global_stop = sim_cfg["tstop"]

      onset_ms:
        - if None → t_start = t_global_start
        - else    → t_start = max(onset_ms, t_global_start)

      duration_ms:
        - if None → t_end = t_global_stop
        - else    → t_end = min(t_start + duration_ms, t_global_stop)

      If t_end <= t_start, the window is degenerate; we treat it as [t_start, t_start].
      Modes may interpret this as “no activity”.
    """
    t_global_start = float(sim_cfg["tstart"])
    t_global_stop = float(sim_cfg["tstop"])

    timing = group_cfg.get("timing", {}) or {}
    onset_ms = timing.get("onset_ms")
    duration_ms = timing.get("duration_ms")

    if onset_ms is None:
        t_start = t_global_start
    else:
        try:
            t_start = max(float(onset_ms), t_global_start)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Group '{group_cfg.get('label', '')}': timing['onset_ms'] "
                f"must be numeric or null (got {onset_ms!r})"
            ) from exc

    if duration_ms is None:
        t_end = t_global_stop
    else:
        try:
            dur = float(duration_ms)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Group '{group_cfg.get('label', '')}': timing['duration_ms'] "
                f"must be numeric or null (got {duration_ms!r})"
            ) from exc
        t_end = min(t_start + dur, t_global_stop)

    if t_end <= t_start:
        # Degenerate window: caller can interpret as “no activity”.
        return t_start, t_start

    return t_start, t_end


def _get_n_syn(group_cfg: Dict[str, Any]) -> int:
    """
    Read the final synapse count for this group.

    Contract with inputs.py:
      - inputs._resolve_n_syn(...) runs before modes and writes
        syns["N_syn_resolved"] for active groups.
      - Modes should use that value when present.

    Fallback:
      - if N_syn_resolved is absent, fall back to syns["N_syn"] as integer.
    """
    syns = group_cfg.get("syns", {}) or {}

    # Preferred path: use N_syn_resolved if set by inputs._resolve_n_syn
    if "N_syn_resolved" in syns and syns["N_syn_resolved"] is not None:
        try:
            n = int(syns["N_syn_resolved"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"N_syn_resolved must be integer-like "
                f"(got {syns['N_syn_resolved']!r})"
            ) from exc
        if n < 0:
            raise ValueError(f"N_syn_resolved must be >= 0 (got {n})")
        return n

    # Fallback: raw N_syn (for safety / testing)
    n_syn = syns.get("N_syn")
    if n_syn is None:
        # If nothing is specified, treat as 1 synapse-equivalent; this is mainly
        # for testing and should not happen in the normal pipeline where
        # _resolve_n_syn runs first.
        return 1

    try:
        n_syn_int = int(n_syn)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"syns['N_syn'] must be integer-like (got {n_syn!r})"
        ) from exc

    if n_syn_int < 0:
        raise ValueError(f"syns['N_syn'] must be >= 0 (got {n_syn_int})")

    return n_syn_int


# ---------------------------------------------------------------------
# Shared helper: homogeneous Poisson spike train generator
# ---------------------------------------------------------------------

def _generate_homogeneous_poisson_trains(
    rate_hz: float,
    t_start_ms: float,
    t_end_ms: float,
    n_syn: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Generate homogeneous Poisson spike trains for n_syn independent sources.

    Parameters
    ----------
    rate_hz : float
        Constant firing rate (Hz) for all sources.
    t_start_ms : float
        Start time of the active window (ms).
    t_end_ms : float
        End time of the active window (ms). Must satisfy t_end_ms >= t_start_ms.
    n_syn : int
        Number of independent spike trains to generate.
    rng : np.random.Generator
        Random number generator to use (no global np.random).

    Returns
    -------
    trains : list[np.ndarray]
        List of length n_syn, each element a 1D float array of spike times (ms),
        sorted and constrained to [t_start_ms, t_end_ms].
    """
    trains: List[np.ndarray] = []

    if n_syn <= 0:
        return trains

    if rate_hz <= 0.0 or t_end_ms <= t_start_ms:
        # Valid config but no spikes: return n_syn empty trains
        return [np.array([], dtype=float) for _ in range(n_syn)]

    # Mean inter-spike interval (ms)
    mean_isi_ms = 1000.0 / float(rate_hz)

    for _ in range(n_syn):
        t = float(t_start_ms)
        spikes: List[float] = []

        # Standard thinning-free homogeneous Poisson in continuous time
        while True:
            isi = rng.exponential(mean_isi_ms)
            t += isi
            if t > t_end_ms:
                break
            spikes.append(t)

        trains.append(np.asarray(spikes, dtype=float))

    return trains


# ---------------------------------------------------------------------
# Shared helper: inhomogeneous Poisson spike train generator
# ---------------------------------------------------------------------

def _generate_inhomogeneous_from_curve(
    rates_hz: np.ndarray,
    t0_ms: float,
    bin_ms: float,
    n_syn: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Generate inhomogeneous Poisson spike trains from a piecewise-constant
    rate curve.

    Parameters
    ----------
    rates_hz : np.ndarray
        1D array of firing rates (Hz). Element k applies on the interval
        [t0_ms + k*bin_ms, t0_ms + (k+1)*bin_ms).
    t0_ms : float
        Start time (ms) of the first bin.
    bin_ms : float
        Bin width (ms). Must be > 0.
    n_syn : int
        Number of independent spike trains to generate.
    rng : np.random.Generator
        Random number generator to use (no global np.random).

    Returns
    -------
    trains : list[np.ndarray]
        List of length n_syn, each a 1D float array of spike times (ms),
        sorted in ascending order. Spikes lie in
        [t0_ms, t0_ms + len(rates_hz) * bin_ms).

    Notes
    -----
    - For bin k with rate r_k (Hz), the expected spike count in that bin
      is λ_k = r_k * bin_ms / 1000.
    - We draw count_k ~ Poisson(λ_k), then place count_k spikes uniformly
      at random within the bin.
    - This allows >1 spike per bin (unlike your old 0/1 vector helper),
      which is the standard continuous-time Poisson construction.
    """
    rates_hz = np.asarray(rates_hz, dtype=float).ravel()
    trains: List[np.ndarray] = []

    if n_syn <= 0 or rates_hz.size == 0:
        return [np.array([], dtype=float) for _ in range(max(n_syn, 0))]

    if bin_ms <= 0.0:
        raise ValueError(f"bin_ms must be > 0, got {bin_ms!r}")

    bin_ms = float(bin_ms)
    n_bins = rates_hz.size

    # Precompute per-bin Poisson means (expected counts)
    # λ_k = rate_k * bin_ms / 1000 (ms → s)
    lam_per_bin = rates_hz * (bin_ms / 1000.0)

    for _ in range(n_syn):
        spikes: List[float] = []

        for k in range(n_bins):
            lam_k = lam_per_bin[k]
            if lam_k <= 0.0:
                continue

            # Number of spikes in this bin
            count = rng.poisson(lam_k)
            if count <= 0:
                continue

            # Uniformly place spikes within the bin
            bin_start = t0_ms + k * bin_ms
            # shape (count,) array uniform in [0, bin_ms)
            offsets = rng.uniform(0.0, bin_ms, size=count)
            times = bin_start + offsets
            spikes.extend(times.tolist())

        if spikes:
            spikes_arr = np.sort(np.asarray(spikes, dtype=float))
        else:
            spikes_arr = np.array([], dtype=float)

        trains.append(spikes_arr)

    return trains



# =====================================================================
# =====================================================================
# Default mode functions
# =====================================================================
# =====================================================================

# ---------------------------------------------------------------------
# Mode: precomputed
# ---------------------------------------------------------------------
def _mode_precomputed(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
    geometry: Optional[Any],
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Built-in handler for mode == "precomputed".

    SOURCE CONTRACT
    ---------------
    Expected `group_cfg["source"]`:
        {
            "trains": [[...], [...], ...],         # optional inline trains (ms, sim-time)
            "path": "pn_precomputed_trains.json"   # OR JSON file with trains
        }

    Accepted JSON file layouts for source['path']:
        1) {"trains": [[t_11, t_12, ...], [t_21, ...], ...]}
        2) [[t_11, t_12, ...], [t_21, ...], ...]

    SYNAPSE COUNT
    -------------
    The intended number of trains is taken from:
        group_cfg["syns"]["N_syn_resolved"]
    falling back to group_cfg["syns"]["N_syn"] if the resolved field is missing.

    RETURN VALUE
    ------------
    Returns:
        list[np.ndarray]
            One 1D float array of spike times per synapse-equivalent source.

    Rules:
        - len(trains) == min(N_syn_resolved, number of trains provided).
          If fewer trains than N_syn_resolved are provided, we DO NOT pad
          or duplicate; 2.3 will raise on the length mismatch.
        - Each array is sorted, dtype=float, and clipped into the group
          time window [t_start, t_end] as computed by _get_group_time_window.
    """
    source = group_cfg.get("source", {}) or {}

    trains_raw = source.get("trains")
    path = source.get("path")

    if trains_raw is None and path is None:
        raise ValueError(
            "Mode 'precomputed' requires either source['trains'] (inline) "
            "or source['path'] (JSON file with spike trains)."
        )

    # If trains are not provided inline, load from JSON file
    if trains_raw is None and path is not None:
        path_obj = Path(path)
        with path_obj.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "trains" in data:
            trains_raw = data["trains"]
        elif isinstance(data, list):
            trains_raw = data
        else:
            raise ValueError(
                f"Precomputed file {path_obj} must be either "
                f'{{"trains": [...]}} or a raw [[...], [...], ...] list.'
            )

    if not isinstance(trains_raw, list):
        raise TypeError(
            "source['trains'] / loaded trains must be a list of lists of spike times."
        )

    # Resolve number of synapses/trains
    syns = group_cfg.get("syns", {}) or {}
    n_syn = syns.get("N_syn_resolved", syns.get("N_syn", 0))
    if n_syn is None:
        n_syn = 0
    n_syn = int(n_syn)
    if n_syn < 0:
        raise ValueError(f"N_syn_resolved must be >= 0, got {n_syn}")
    if n_syn == 0:
        return []

    # Ensure we have a list-of-lists
    trains_list: List[List[float]] = []
    for idx, tr in enumerate(trains_raw):
        if tr is None:
            trains_list.append([])
        elif isinstance(tr, (list, tuple)):
            trains_list.append(list(tr))
        else:
            raise TypeError(
                f"Train {idx} in precomputed source is not list-like; got {type(tr)!r}"
            )

    # If more trains are provided than needed, truncate
    if len(trains_list) >= n_syn:
        trains_list = trains_list[:n_syn]
    # If fewer trains are provided than required, we return what we have;
    # the 2.3 orchestrator will catch the mismatch in len(trains) vs N_syn_resolved.

    t_start, t_end = _get_group_time_window(sim_cfg, group_cfg)

    trains: List[np.ndarray] = []
    for tr in trains_list:
        arr = np.asarray(tr, dtype=float).ravel()
        if arr.size == 0:
            trains.append(arr)
            continue
        # Clip to [t_start, t_end] and sort
        mask = (arr >= t_start) & (arr <= t_end)
        arr = np.sort(arr[mask])
        trains.append(arr)

    return trains


# ---------------------------------------------------------------------
# Mode: homogeneous_poisson
# ---------------------------------------------------------------------

def _mode_homogeneous_poisson(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
    geometry: Optional[Any],
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Built-in handler for mode == "homogeneous_poisson".

    SOURCE CONTRACT
    ---------------
    Expected `group_cfg["source"]` keys:
        {
            "freq": float,   # required, firing rate in Hz
            ...
        }

    SYNAPSE COUNT
    -------------
    Uses:
        N_syn = group_cfg["syns"]["N_syn_resolved"]
    falling back to group_cfg["syns"]["N_syn"] if the resolved field is missing.

    TIMING
    ------
    Uses _get_group_time_window(sim_cfg, group_cfg) to compute:
        t_start_ms, t_end_ms
    and generates spikes only in that interval.

    RETURN VALUE
    ------------
    Returns:
        list[np.ndarray] (length N_syn)
            One 1D float array of spike times (ms, sorted) per synapse-source.
            All spikes lie within [sim_cfg["tstart"], sim_cfg["tstop"]].
    """
    # --- Resolve frequency ---
    source = group_cfg.get("source", {}) or {}
    freq = source.get("freq", None)

    if freq is None:
        raise ValueError(
            "Mode 'homogeneous_poisson' requires source['freq'] (Hz)."
        )

    try:
        rate_hz = float(freq)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Mode 'homogeneous_poisson' expects numeric source['freq'], got {freq!r}."
        ) from exc

    # --- Resolve number of synapses / trains ---
    n_syn = _get_n_syn(group_cfg)
    if n_syn <= 0:
        return []

    # --- Time window for this group ---
    t_start_ms, t_end_ms = _get_group_time_window(sim_cfg, group_cfg)

    # --- Generate Poisson trains ---
    trains = _generate_homogeneous_poisson_trains(
        rate_hz=rate_hz,
        t_start_ms=t_start_ms,
        t_end_ms=t_end_ms,
        n_syn=n_syn,
        rng=rng,
    )

    # At this point, trains is a list of length n_syn, with properly bounded spikes.
    # 2.3 will still do global sanity checks (e.g., within [sim_cfg["tstart"], tstop]).

    return trains

# ---------------------------------------------------------------------
# Mode: inhomogeneous_poisson (stub)
# ---------------------------------------------------------------------
def _mode_inhomogeneous_poisson(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
    geometry: Optional[Any],
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Built-in handler for mode == "inhomogeneous_poisson".

    This mode consumes a **rate curve** from an external file and will
    eventually generate inhomogeneous Poisson spike trains with optional
    baseline before/after a stim segment. The file parsing and curve
    validation are implemented; the exact time-alignment and spike
    generation will be added once the timing semantics are finalized.

    SOURCE CONTRACT
    ---------------
    Expected `group_cfg["source"]` keys:
        {
            "path": "pn_rate_curve.csv" or "pn_rate_curve.json",   # required
            "time_col": "Time",                                   # required for CSV/JSON
            "rate_col": "AvgFiringRate",                          # required for CSV/JSON
            "bin_ms": 1.0,                                        # optional, default inferred
            "baseline": null | float                              # optional; Hz; default first rate
        }

    Accepted file formats for source['path']:
        1) CSV with header row (comma-separated):
           - `time_col` and `rate_col` must match column names.
           - Example: columns "Time", "AvgFiringRate" used with
             time_col="Time", rate_col="AvgFiringRate".

        2) JSON dict:
           - Must contain numeric arrays under `time_col` and `rate_col`, e.g.:
             {
                 "Time": [...],
                 "AvgFiringRate": [...]
             }

    TIMING / ALIGNMENT (group_cfg["timing"])
    ----------------------------------------
    The following fields are available to map curve time → sim time
    (all optional, design still being finalized):
        - onset_ms
        - duration_ms
        - stim_tstart_ms
        - input_stim_tstart_ms
        - input_duration_ms

    CURRENT STATUS
    --------------
    This function:
        - loads and validates the curve (times_ms, rates_hz),
        - infers bin_ms if not given,
        - resolves a baseline rate.

    It then raises NotImplementedError because the detailed mapping
    and spike generation logic are not yet locked in. This keeps the
    behavior explicit and safe until we finalize the semantics.
    """
    source = group_cfg.get("source", {}) or {}
    path = source.get("path")
    time_col = source.get("time_col")
    rate_col = source.get("rate_col")
    bin_ms = source.get("bin_ms")
    baseline = source.get("baseline")

    if path is None:
        raise ValueError("Mode 'inhomogeneous_poisson' requires source['path'].")

    if not time_col or not rate_col:
        raise ValueError(
            "Mode 'inhomogeneous_poisson' requires source['time_col'] and source['rate_col'] "
            "to identify columns/keys in the rate-curve file."
        )

    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Rate curve file not found: {path_obj}")

    # --- Load rate curve ---

    if path_obj.suffix.lower() in {".csv", ".txt"}:
        # CSV with header; np.genfromtxt gives a structured array with named fields
        data = np.genfromtxt(path_obj, delimiter=",", names=True)
        try:
            times_ms = np.asarray(data[time_col], dtype=float)
            rates_hz = np.asarray(data[rate_col], dtype=float)
        except ValueError as exc:
            raise KeyError(
                f"Columns {time_col!r} and/or {rate_col!r} not found in CSV header of {path_obj}."
            ) from exc
    else:
        # Assume JSON dict
        with path_obj.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"Inhomogeneous rate file {path_obj} must be a dict-like JSON "
                f"containing {time_col!r} and {rate_col!r} arrays."
            )
        try:
            times_ms = np.asarray(data[time_col], dtype=float)
            rates_hz = np.asarray(data[rate_col], dtype=float)
        except KeyError as exc:
            raise KeyError(
                f"JSON in {path_obj} must contain keys {time_col!r} and {rate_col!r}."
            ) from exc

    if times_ms.ndim != 1 or rates_hz.ndim != 1 or times_ms.shape != rates_hz.shape:
        raise ValueError(
            "Rate curve arrays 'time_col' and 'rate_col' must be 1D and of equal length."
        )

    # Sort by time in case the file is not ordered
    order = np.argsort(times_ms)
    times_ms = times_ms[order]
    rates_hz = rates_hz[order]

    # Infer bin width if not provided
    if bin_ms is None:
        if times_ms.size < 2:
            raise ValueError(
                "Cannot infer bin_ms from a single-point rate curve; "
                "please specify source['bin_ms']."
            )
        dt = np.median(np.diff(times_ms))
        bin_ms = float(dt)
    else:
        bin_ms = float(bin_ms)

    # Default baseline: first rate value if not explicitly provided
    if baseline is None:
        baseline = float(rates_hz[0])
    else:
        baseline = float(baseline)

    # At this point we have:
    #   times_ms: 1D array of sample times in ms (monotonic)
    #   rates_hz: 1D array of firing rates in Hz
    #   bin_ms:   bin width for interpreting the curve
    #   baseline: baseline rate in Hz
    # plus timing fields in group_cfg["timing"] and sim_cfg for alignment.
    # The remaining step is to decide:
    #   - how to slice the curve using input_duration_ms / input_start_ms,
    #   - how to map that slice into [sim_cfg["tstart"], sim_cfg["tstop"]],
    #   - how to construct pre-baseline / stim / post-baseline segments,
    #   - how to generate N_syn_resolved spike trains from the resulting rate(t).

    raise NotImplementedError(
        "Mode 'inhomogeneous_poisson' has loaded the rate curve but the "
        "time-alignment and spike-train generation logic are not yet implemented."
    )


# =====================================================================
# Registry
# =====================================================================


def get_default_mode_registry() -> Dict[str, Any]:
    """
    Return the default mode registry for Step 2.3.
    """
    return {
        "homogeneous_poisson": _mode_homogeneous_poisson,
        "precomputed": _mode_precomputed,
        "inhomogeneous_poisson": _mode_inhomogeneous_poisson,
    }
