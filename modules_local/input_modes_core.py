from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np


# =====================================================================
# Shared helpers for modes
# =====================================================================

def _get_n_syn(group_cfg: Dict[str, Any]) -> int:
    """
    Read the final synapse count for this group.

    Contract with inputs.py:
      - inputs._resolve_n_syn(...) runs before modes and writes
        syns["N_syn_resolved"] for active groups.
      - Modes should use that value when present, via this helper.

    Fallback:
      - If N_syn_resolved is absent, fall back to syns["N_syn"] as integer.
        This is mainly for testing / non-geometry cases.
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


def _get_active_window_from_time_cfg(time_cfg: Dict[str, Any]) -> Tuple[float, float]:
    """
    Derive an overall active window [t_start, t_end] in ms from time_cfg.

    Usage:
      - Modes obtain time_cfg from group_cfg["time_cfg"] and pass it here.

    Rules:
      - Use time_cfg["blocks"] if present:
          * collect all blocks with kind != "quiescent"
          * t_start = min(block["t_start"]) over those blocks
          * t_end   = max(block["t_end"])   over those blocks
      - If there are no non-quiescent blocks, fall back to anchors:
          [sim_tstart, sim_tstop].

    Notes:
      - This is mainly a convenience for modes (e.g. precomputed) that only
        need a single “union” active window and don’t care about individual
        blocks; more complex modes should work block-by-block instead.
    """
    anchors = (time_cfg or {}).get("anchors", {}) or {}
    blocks = (time_cfg or {}).get("blocks", []) or []

    non_quiescent = [b for b in blocks if b.get("kind") != "quiescent"]

    if non_quiescent:
        t_start = min(float(b["t_start"]) for b in non_quiescent)
        t_end = max(float(b["t_end"]) for b in non_quiescent)
        # Degenerate protection
        if t_end <= t_start:
            return t_start, t_start
        return t_start, t_end

    # Fallback: anchors only
    sim_tstart = float(anchors.get("sim_tstart", 0.0))
    sim_tstop = float(anchors.get("sim_tstop", sim_tstart))
    if sim_tstop <= sim_tstart:
        return sim_tstart, sim_tstart
    return sim_tstart, sim_tstop


# ---------------------------------------------------------------------
# Shared helper: resolve source paths
# ---------------------------------------------------------------------
def _find_scp_root(start: Path) -> Optional[Path]:
    for p in [start] + list(start.parents):
        if (p / "cells").is_dir() and (p / "run_pipeline.py").is_file():
            return p
    return None


def _resolve_source_path(raw_path: str, sim_cfg: Dict[str, Any]) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p

    tune_dir_raw = sim_cfg.get("tune_dir")
    tune_dir = Path(tune_dir_raw) if tune_dir_raw else Path.cwd()
    tune_dir = tune_dir.resolve()
    repo_root = _find_scp_root(tune_dir)

    if repo_root and p.parts and p.parts[0] in ("external_data", "cells"):
        return (repo_root / p).resolve()

    return (tune_dir / p).resolve()


def _parse_gabab_cfg(source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = (source or {}).get("gabab", None)
    if raw in (None, False):
        return None
    if raw is True:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"source['gabab'] must be dict/bool (got {type(raw)!r})")
    if raw.get("enabled") is False:
        return None

    cfg = dict(raw)
    mode = str(cfg.get("mode", "delayed")).strip().lower()
    if mode not in ("delayed", "simple"):
        raise ValueError(f"source['gabab'].mode must be 'delayed' or 'simple' (got {mode!r})")

    history = str(cfg.get("history", "full")).strip().lower()
    if history not in ("full", "trimmed"):
        raise ValueError(f"source['gabab'].history must be 'full' or 'trimmed' (got {history!r})")

    tau_s_raw = cfg.get("tau_s", None)
    tau_ms_raw = cfg.get("tau_ms", None)
    if tau_s_raw is None and tau_ms_raw is None:
        tau_s = 0.01
    elif tau_s_raw is not None:
        tau_s = float(tau_s_raw)
    else:
        tau_s = float(tau_ms_raw) / 1000.0
    if tau_s <= 0.0:
        raise ValueError(f"source['gabab'].tau_s must be > 0 (got {tau_s!r})")

    delay_ms_raw = cfg.get("delay_ms", cfg.get("delay", 50.0))
    delay_ms = 0.0 if delay_ms_raw is None else float(delay_ms_raw)
    alpha = float(cfg.get("alpha", 1.0))
    init = str(cfg.get("init", "match"))
    robust_norm = bool(cfg.get("robust_norm", False))
    pctl = float(cfg.get("pctl", 99.0))

    return {
        "mode": mode,
        "history": history,
        "tau_s": tau_s,
        "delay_ms": delay_ms,
        "alpha": alpha,
        "init": init,
        "robust_norm": robust_norm,
        "pctl": pctl,
    }


def _apply_gabab_to_curve(
    times_ms: np.ndarray,
    rates_hz: np.ndarray,
    cfg: Dict[str, Any],
) -> np.ndarray:
    if rates_hz.size == 0 or times_ms.size < 2:
        return rates_hz

    dt_ms = float(np.median(np.diff(times_ms[: min(times_ms.size, 500)])))
    if dt_ms <= 0.0:
        raise ValueError(f"GABAB: invalid dt_ms {dt_ms!r}")
    dt_s = dt_ms / 1000.0

    r = np.asarray(rates_hz, dtype=float)
    if cfg["robust_norm"]:
        r_ref = np.percentile(r, cfg["pctl"])
    else:
        r_ref = r.max()
    r_ref = max(float(r_ref), 1e-12)
    r_norm = r / r_ref

    if cfg["mode"] == "simple":
        r_drive = r_norm
    else:
        k = int(round(cfg["delay_ms"] / max(dt_ms, 1e-12)))
        if k <= 0:
            r_drive = r_norm
        elif k >= r.size:
            base = r_norm[0] if cfg["init"] == "match" else 0.0
            r_drive = np.full_like(r_norm, base)
        else:
            base = r_norm[0] if cfg["init"] == "match" else 0.0
            r_drive = np.empty_like(r_norm)
            r_drive[:k] = base
            r_drive[k:] = r_norm[:-k]

    S = np.zeros_like(r_norm)
    S[0] = r_norm[0] if cfg["init"] == "match" else 0.0
    coef = dt_s / cfg["tau_s"]
    for i in range(1, r.size):
        S[i] = S[i - 1] + coef * (r_drive[i - 1] - S[i - 1])
    S = np.clip(S, 0.0, 1.0)

    I = r * (1.0 - cfg["alpha"] * S)
    I[I < 0.0] = 0.0
    return I


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

    This helper is mode-agnostic; modes are responsible for:
      - choosing (rates_hz, t0_ms, bin_ms) consistent with their time_cfg,
      - splitting/combining per-block segments as needed,
      - stitching segments per synapse in chronological order.
    """
    rates_hz = np.asarray(rates_hz, dtype=float).ravel()
    trains: List[np.ndarray] = []

    if n_syn <= 0 or rates_hz.size == 0:
        return [np.array([], dtype=float) for _ in range(max(n_syn, 0))]

    if bin_ms <= 0.0:
        raise ValueError(f"bin_ms must be > 0, got {bin_ms!r}")

    bin_ms = float(bin_ms)
    n_bins = rates_hz.size

    # λ_k = rate_k * bin_ms / 1000 (ms → s)
    lam_per_bin = rates_hz * (bin_ms / 1000.0)

    for _ in range(n_syn):
        spikes: List[float] = []

        for k in range(n_bins):
            lam_k = lam_per_bin[k]
            if lam_k <= 0.0:
                continue

            count = rng.poisson(lam_k)
            if count <= 0:
                continue

            bin_start = t0_ms + k * bin_ms
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
# Core mode functions (Step 5.2.3 input-generation contract)
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
    Precomputed spike trains sampled from stored data.

    Source options:
      - source["trains"]: inline list of spike-time arrays (ms, starting at 0)
      - source["path"]: path to file (or run folder) containing trains. Supported:
          * .pkl/.p: pickled list, or dict (auto-selects spikes/trains or source["key"])
          * .json: either {"trains": [...]} or raw list
          * .npz/.npy: arrays of trains (auto-selects "spikes" or single array)
          * run folder with spikes.npz / inputs_sample.pkl / results.pkl

    Selection:
      - source["selection"] = "sample" (default) or "first_n"
      - source["key"] selects a specific key in pickled dicts

    Timing:
      - Use time_cfg blocks; "source" blocks place sampled trains shifted to block start.
      - Baseline blocks use baseline_rate_hz (anchors) to generate homogeneous Poisson if set.
      - Quiescent blocks add nothing.
    """
    time_cfg = (group_cfg or {}).get("time_cfg") or {}
    anchors = time_cfg.get("anchors", {}) or {}
    blocks = time_cfg.get("blocks", []) or []

    try:
        sim_tstart = float(sim_cfg["tstart"])
        sim_tstop = float(sim_cfg["tstop"])
    except Exception as exc:
        raise ValueError("sim_cfg must contain tstart and tstop for precomputed mode") from exc

    n_syn = _get_n_syn(group_cfg)
    if n_syn <= 0:
        return []

    source = group_cfg.get("source", {}) or {}

    def _normalize_pool(
        obj,
        *,
        key: Optional[str] = None,
        label: str = "precomputed",
    ) -> List[np.ndarray]:
        if isinstance(obj, dict):
            if "inputs_by_trial" in obj or "inputs" in obj:
                pool = _extract_pool_from_inputs(obj)
                if pool:
                    return pool
            if key and key in obj:
                obj = obj[key]
            else:
                for k in ("spikes", "spike_trains", "trains"):
                    if k in obj:
                        obj = obj[k]
                        break
                else:
                    if len(obj) == 1:
                        obj = next(iter(obj.values()))
                    else:
                        raise ValueError(
                            f"{label}: multiple keys found; set source['key'] to select one"
                        )
        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                return [np.asarray(x, dtype=float) for x in obj.tolist()]
            if obj.ndim == 1:
                return [np.asarray(obj, dtype=float)]
            if obj.ndim == 2:
                return [np.asarray(row, dtype=float) for row in obj]
        if isinstance(obj, (list, tuple)):
            return [np.asarray(x, dtype=float) for x in obj]
        raise ValueError(f"{label}: trains must be a list, array, or dict of lists")

    def _select_input_group(inputs_map: Dict[str, Any]) -> str:
        group_key = source.get("group") or source.get("group_name") or source.get("key")
        if group_key and group_key in inputs_map:
            return group_key
        if len(inputs_map) == 1:
            return next(iter(inputs_map.keys()))
        if group_key:
            raise KeyError(
                f"precomputed: group {group_key!r} not found in inputs; "
                f"available={list(inputs_map.keys())}"
            )
        raise KeyError(
            "precomputed: multiple input groups found; set source['group'] "
            "to choose one"
        )

    def _extract_pool_from_inputs(payload: Dict[str, Any]) -> List[np.ndarray]:
        pool: List[np.ndarray] = []

        inputs_by_trial = payload.get("inputs_by_trial")
        if isinstance(inputs_by_trial, list):
            for entry in inputs_by_trial:
                inputs_map = (entry or {}).get("inputs", {}) or {}
                if not inputs_map:
                    continue
                gname = _select_input_group(inputs_map)
                gdata = inputs_map.get(gname, {}) or {}
                for tr in gdata.get("spike_trains", []) or []:
                    pool.append(np.asarray(tr, dtype=float))

        inputs_single = payload.get("inputs")
        if isinstance(inputs_single, dict):
            inputs_map = inputs_single
            if inputs_map:
                gname = _select_input_group(inputs_map)
                gdata = inputs_map.get(gname, {}) or {}
                for tr in gdata.get("spike_trains", []) or []:
                    pool.append(np.asarray(tr, dtype=float))

        return pool

    def _load_trains_from_run_dir(run_dir: Path) -> List[np.ndarray]:
        manifest = run_dir / "run_manifest.json"
        inputs_path: Optional[Path] = None
        results_path: Optional[Path] = None
        spikes_path: Optional[Path] = None

        if manifest.is_file():
            try:
                files = json.loads(manifest.read_text()).get("files", {}) or {}
            except Exception:
                files = {}
            if files.get("inputs_sample"):
                inputs_path = run_dir / files["inputs_sample"]
            if files.get("results_pkl"):
                results_path = run_dir / files["results_pkl"]
            if files.get("spikes"):
                spikes_path = run_dir / files["spikes"]

        if spikes_path is None:
            candidates = [
                run_dir / "spikes.npz",
                run_dir / "spikes.pkl",
                run_dir / "spikes.p",
                run_dir / "spikes.npy",
                run_dir / "results" / "spikes.npz",
                run_dir / "results" / "spikes.pkl",
                run_dir / "results" / "spikes.p",
                run_dir / "results" / "spikes.npy",
            ]
            for cand in candidates:
                if cand.is_file():
                    spikes_path = cand
                    break

        if spikes_path is not None and spikes_path.is_file():
            try:
                pool = _load_trains_from_path(spikes_path)
            except Exception:
                pool = []
            if pool:
                return pool

        if inputs_path is None:
            fallback = run_dir / "inputs_sample.pkl"
            if fallback.is_file():
                inputs_path = fallback

        if inputs_path is not None and inputs_path.is_file():
            import pickle

            with inputs_path.open("rb") as f:
                payload = pickle.load(f)
            try:
                pool = _normalize_pool(
                    payload,
                    key=source.get("key"),
                    label=f"precomputed:{inputs_path}",
                )
            except Exception:
                pool = []
            if pool:
                return pool

        if results_path is not None and results_path.is_file():
            import pickle

            with results_path.open("rb") as f:
                payload = pickle.load(f)
            try:
                pool = _normalize_pool(
                    payload,
                    key=source.get("key"),
                    label=f"precomputed:{results_path}",
                )
            except Exception:
                pool = []
            if pool:
                return pool

        raise ValueError(
            "precomputed: run folder lacks spikes.npz (or inputs_sample.pkl). "
            "Provide a file path to spike trains or enable saving spikes/inputs."
        )

    def _load_trains_from_path(p: Path) -> List[np.ndarray]:
        if p.is_file() and p.name == "run_manifest.json":
            try:
                return _load_trains_from_run_dir(p.parent)
            except Exception:
                p = p.parent

        if p.is_dir():
            if (
                (p / "run_manifest.json").is_file()
                or (p / "inputs_sample.pkl").is_file()
                or (p / "spikes.npz").is_file()
            ):
                try:
                    return _load_trains_from_run_dir(p)
                except Exception:
                    pass

            results_dir = p / "results"
            if results_dir.is_dir():
                for cand in (
                    results_dir / "spikes.npz",
                    results_dir / "spikes.pkl",
                    results_dir / "spikes.p",
                    results_dir / "spikes.npy",
                ):
                    if cand.is_file():
                        return _load_trains_from_path(cand)

            preferred = [p / f"{p.name}{ext}" for ext in (".npz", ".npy", ".pkl", ".p", ".json")]
            candidates: List[Path] = [c for c in preferred if c.is_file()]
            if not candidates:
                candidates = sorted(p.glob("spikes.*"))
            if not candidates:
                candidates = sorted(p.glob("*.pkl"))
            if not candidates:
                candidates = sorted(p.glob("*.p"))
            if not candidates:
                candidates = sorted(p.glob("*.npz"))
            if not candidates:
                candidates = sorted(p.glob("*.npy"))
            if not candidates:
                candidates = sorted(p.glob("*.json"))
            for candidate in candidates:
                return _load_trains_from_path(candidate)
            raise ValueError(
                f"precomputed: no supported trains file found in {p}"
            )

        if not p.is_file():
            raise FileNotFoundError(f"precomputed: file not found {p}")
        suffix = p.suffix.lower()
        if suffix in (".pkl", ".p"):
            import pickle

            with p.open("rb") as f:
                obj = pickle.load(f)
            return _normalize_pool(obj, key=source.get("key"), label=f"precomputed:{p}")
        if suffix in (".npz", ".npy"):
            if suffix == ".npz":
                with np.load(p, allow_pickle=True) as data:
                    obj = {k: data[k] for k in data.files}
            else:
                obj = np.load(p, allow_pickle=True)
            return _normalize_pool(obj, key=source.get("key"), label=f"precomputed:{p}")
        if suffix == ".json":
            with p.open("r") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and "trains" in obj:
                obj = obj["trains"]
            return _normalize_pool(obj, key=source.get("key"), label=f"precomputed:{p}")
        raise ValueError(f"precomputed: unsupported file type {p.suffix} for {p}")

    if source.get("trains") is not None:
        pool = _normalize_pool(source["trains"], key=source.get("key"))
    elif source.get("path"):
        try:
            resolved = _resolve_source_path(str(source["path"]), sim_cfg)
            pool = _load_trains_from_path(resolved)
        except FileNotFoundError:
            print(f"precomputed: missing source path {source.get('path')!r}; using empty trains")
            return [np.array([], dtype=float) for _ in range(n_syn)]
    else:
        raise ValueError("precomputed mode requires source['trains'] or source['path']")

    if not pool:
        return [np.array([], dtype=float) for _ in range(n_syn)]

    selection = str(source.get("selection", "sample")).strip().lower()
    if selection not in ("sample", "first_n"):
        raise ValueError("precomputed: selection must be 'sample' or 'first_n'")

    trim_ms = float(anchors.get("source_trim_ms", 0.0) or 0.0)
    jitter_tstart = anchors.get("jitter_tstart_ms", None)

    # helper: select trains for n_syn
    pool_size = len(pool)
    def _select_trains():
        if selection == "first_n":
            if n_syn <= pool_size:
                idx = list(range(n_syn))
            else:
                reps = n_syn // pool_size
                rem = n_syn % pool_size
                idx = list(range(pool_size)) * reps + list(range(rem))
        elif n_syn <= pool_size:
            idx = rng.choice(pool_size, size=n_syn, replace=False)
        else:
            idx = rng.choice(pool_size, size=n_syn, replace=True)
        return [np.asarray(pool[i], dtype=float).copy() for i in idx]

    baseline_rate = anchors.get("baseline_rate_hz", None)
    baseline_spec = anchors.get("baseline_spec") or {}
    jitter_tstart = anchors.get("jitter_tstart_ms", None)
    if baseline_rate is None and baseline_spec.get("kind") not in (None, "none"):
        print(
            "precomputed mode: baseline spec requires a rate curve; "
            "no baseline will be generated."
        )
    trains_accum: List[List[float]] = [[] for _ in range(n_syn)]

    for block in blocks:
        kind = block.get("kind")
        t0 = float(block.get("t_start", sim_tstart))
        t1 = float(block.get("t_end", t0))
        if t1 <= t0:
            continue
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
            sampled = _select_trains()
            seg_trains = []
            for tr in sampled:
                # Trim any leading portion that would have occurred before the
                # resolved source start, to keep the stimulus alignment when
                # the raw source start was earlier than onset.
                trimmed = tr - trim_ms
                trimmed = trimmed[trimmed >= 0.0]

                shifted = trimmed + t0
                clipped = shifted[(shifted >= t0) & (shifted <= t1)]
                seg_trains.append(clipped)
        else:
            continue

        for i in range(n_syn):
            trains_accum[i].extend(seg_trains[i].tolist())

    out: List[np.ndarray] = []
    for spikes in trains_accum:
        if not spikes:
            out.append(np.array([], dtype=float))
            continue
        arr = np.asarray(spikes, dtype=float)
        arr = arr[(arr >= sim_tstart) & (arr <= sim_tstop)]
        arr.sort()
        out.append(arr)

    return out


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

# ---------------------------------------------------------------------
# Mode: inhomogeneous_poisson
# ---------------------------------------------------------------------
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


# =====================================================================
# Registry
# =====================================================================

def get_default_mode_registry() -> Dict[str, Any]:
    """
    Return the default mode registry for Step 5.2.3.

    All registered handlers obey the 4-argument mode contract:
        handler(sim_cfg, group_cfg, geometry, rng)
    and must return List[np.ndarray] of length N_syn as resolved by
    _get_n_syn(group_cfg).
    """
    return {
        "homogeneous_poisson": _mode_homogeneous_poisson,
        "precomputed": _mode_precomputed,
        "inhomogeneous_poisson": _mode_inhomogeneous_poisson,
    }
