"""Built-in mode for replaying or resampling precomputed input curves/trains."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import json
import numpy as np

from .mode_helpers import (
    _get_n_syn,
    _get_active_window_from_time_cfg,
    _generate_homogeneous_poisson_trains,
    _generate_inhomogeneous_from_curve,
    _resolve_source_path,
)


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
      - Step 5 normally calls this handler once per precomputed input block.
      - The block source crop is passed through anchors["source_trim_ms"].
      - Source blocks place sampled trains shifted to the block start.
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
