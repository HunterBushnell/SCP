from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from scp_pipeline.input_generation.recording_config import resolve_n_syn_recording_trials

from .result_helpers import (
    _aggregate_input_stats,
    _resolve_inputs_to_save,
    _resolve_trace_trials_to_save,
    _smooth_rate_curve,
)
from .result_loading import load_results
from .result_saving import _write_results_file, _write_results_to_run_dir
from scp_pipeline.workspaces import assert_same_workspace, result_workspace_identity


def _ensure_multi_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce a single-trial results dict into multi-trial format for appending.
    """
    mode = results.get("mode", "single")
    if mode == "multi":
        return copy.deepcopy(results)
    if mode != "single":
        raise ValueError(f"append_to: unsupported results mode {mode!r}")

    multi = copy.deepcopy(results)
    multi["mode"] = "multi"
    spikes = multi.get("spikes")
    multi["spikes"] = [spikes if spikes is not None else np.array([], dtype=float)]
    traces = multi.get("traces", {}) or {}
    multi["traces"] = {key: value if key == "T" else [value]
                       for key, value in traces.items()}

    for single_key, trial_key, payload_key in (
        ("inputs", "inputs_by_trial", "inputs"),
        ("cell_recordings", "cell_recordings_by_trial", "recordings"),
        ("syn_recordings", "syn_recordings_by_trial", "recordings"),
    ):
        value = multi.pop(single_key, None)
        if value is not None:
            multi[trial_key] = [{"trial_idx": 0, payload_key: value}]
        else:
            multi.setdefault(trial_key, None)
    if not multi.get("syn_records_by_trial") and multi.get("syn_records") is not None:
        multi["syn_records_by_trial"] = [{"trial_idx": 0, "records": copy.deepcopy(multi["syn_records"])}]

    meta = multi.setdefault("meta", {})
    multi.setdefault("sim_cfg", {})["n_trials"] = 1
    meta["n_trials"] = 1
    meta["trial_ids"] = [0]
    return multi


def _resolve_syn_recording_trials_to_save(sim_cfg: Dict[str, Any], n_trials: int) -> int:
    """Resolve capture retention without importing NEURON-backed runtime code."""
    return resolve_n_syn_recording_trials(sim_cfg, n_trials)


def _merge_syn_recordings_by_trial(
    dst: Dict[str, Any], src: Dict[str, Any], *, trial_offset: int, n_trials: int,
) -> None:
    """Copy recorded trials with global indices and their independent retention cap.

    Mixing different recorded axes would make a pooled result misleading. Reject
    that explicitly, including differing array lengths (which must not broadcast).
    """
    entries = list(dst.get("syn_recordings_by_trial") or [])
    incoming = list(src.get("syn_recordings_by_trial") or [])
    template = None
    for entry in entries + incoming:
        recordings = entry.get("recordings", {})
        if "T" not in recordings:
            raise ValueError("syn_recording_time_mismatch: recording payload has no T axis")
        axis = np.asarray(recordings["T"], dtype=float)
        if axis.ndim != 1:
            raise ValueError("syn_recording_time_mismatch: T must be one-dimensional")
        if template is None:
            template = axis
        elif template.shape != axis.shape or not np.allclose(template, axis, rtol=0, atol=1e-9):
            raise ValueError("syn_recording_time_mismatch: cannot merge different recording time axes")

    if not entries and not incoming:
        return
    cap = _resolve_syn_recording_trials_to_save(dst.get("sim_cfg", {}) or {}, n_trials)
    combined = copy.deepcopy(entries)
    for idx, entry in enumerate(incoming):
        new_entry = copy.deepcopy(entry)
        new_entry["trial_idx"] = trial_offset + int(new_entry.get("trial_idx", idx))
        combined.append(new_entry)
    dst["syn_recordings_by_trial"] = combined[:cap] or None


def _resolve_append_path(
    append_to: Union[str, Path],
    *,
    base_dir: Union[str, Path, None] = None,
) -> Path:
    append_path = Path(append_to)
    if not append_path.is_absolute() and base_dir is not None:
        base_dir = Path(base_dir)
        if append_path.parts and append_path.parts[0] == base_dir.name:
            append_path = base_dir / Path(*append_path.parts[1:])
        else:
            append_path = base_dir / append_path

    # Normalize run_manifest.json to its parent folder
    if append_path.name == "run_manifest.json":
        append_path = append_path.parent
    return append_path


def _validate_append_destination(
    results: Dict[str, Any],
    append_to: Union[str, Path],
    *,
    base_dir: Union[str, Path, None] = None,
) -> list[str]:
    """Reject a foreign workspace before any result directory is created."""
    path = _resolve_append_path(append_to, base_dir=base_dir)
    if path.exists():
        return assert_same_workspace(load_results(path), results, context="append_to")
    return []


def _append_results_to_path(
    results: Dict[str, Any],
    append_to: Union[str, Path],
    *,
    base_dir: Union[str, Path, None] = None,
) -> Optional[Path]:
    """Append trials from the same logical workspace only."""
    append_path = _resolve_append_path(append_to, base_dir=base_dir)

    if append_path.exists():
        base_results = load_results(append_path)
        assert_same_workspace(base_results, results, context="append_to")
        merged = append_multi_results(
            _ensure_multi_results(base_results),
            _ensure_multi_results(results),
        )
    else:
        merged = _ensure_multi_results(results)

    # Decide where/how to write
    if append_path.suffix.lower() in (".pkl", ".npz"):
        append_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = "npz" if append_path.suffix.lower() == ".npz" else "pickle"
        manifest_path = append_path.parent / "run_manifest.json"
        if manifest_path.is_file():
            try:
                files = json.loads(manifest_path.read_text()).get("files", {})
                paired = append_path.name in (files.get("results_pkl"), files.get("results_npz"))
            except (OSError, ValueError, AttributeError):
                paired = False
            if paired:
                # An explicitly paired full-result file is part of the saved
                # run, so its snapshots must change with the appended data.
                return _write_results_to_run_dir(
                    merged, append_path.parent, fmt=fmt, results_name=append_path.name)
        _write_results_file(merged, append_path, fmt)
        return append_path

    # Directory target (existing or new)
    fmt = None
    results_name = None
    manifest = append_path / "run_manifest.json"
    if manifest.is_file():
        try:
            old_manifest = json.loads(manifest.read_text())
            files = old_manifest.get("files", {}) or {}
            if files.get("results_npz"):
                fmt = "npz"
                results_name = files.get("results_npz")
            elif files.get("results_pkl"):
                fmt = "pickle"
                results_name = files.get("results_pkl")
        except Exception:
            fmt = None
            results_name = None

    return _write_results_to_run_dir(merged, append_path, fmt=fmt, results_name=results_name)

def append_multi_results(
    base_results: Dict[str, Any],
    new_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge two multi-trial results dicts in memory.

    - Requires both mode == 'multi'.
    - Concatenates spike trains.
    - Updates sim_cfg['n_trials'].
    - If both have V-traces with matching T, concatenates those too.

    Returns a NEW dict; does not modify inputs in-place.
    """
    if base_results.get("mode") != "multi":
        raise ValueError("append_multi_results: base_results.mode must be 'multi'")
    if new_results.get("mode") != "multi":
        raise ValueError("append_multi_results: new_results.mode must be 'multi'")

    identity_warnings = assert_same_workspace(base_results, new_results, context="append_multi_results")
    base_identity = result_workspace_identity(base_results)
    incoming_identity = result_workspace_identity(new_results)
    identity = {key: base_identity[key] if base_identity[key] is not None else incoming_identity[key]
                for key in base_identity}
    merged = copy.deepcopy(base_results)
    if "workspace_identity" in merged:
        merged["workspace_identity"] = copy.deepcopy(identity)
    meta = merged.setdefault("meta", {})
    meta["workspace_identity"] = copy.deepcopy(identity)
    warnings = meta.setdefault("append_warnings", [])
    warnings.extend(identity_warnings)
    sources = meta.setdefault("append_sources", [])
    if not sources:
        sources.append({"workspace_identity": result_workspace_identity(base_results),
                        "tune_dir": (base_results.get("sim_cfg") or {}).get("tune_dir"),
                        "output": (base_results.get("sim_cfg") or {}).get("output"),
                        "model_provenance": copy.deepcopy((base_results.get("meta") or {}).get("model_provenance"))})
    sources.append({"workspace_identity": result_workspace_identity(new_results),
                    "tune_dir": (new_results.get("sim_cfg") or {}).get("tune_dir"),
                    "output": (new_results.get("sim_cfg") or {}).get("output"),
                    "model_provenance": copy.deepcopy((new_results.get("meta") or {}).get("model_provenance"))})
    # A relocated workspace can retain its logical identity. Fresh archives
    # describe the incoming construction, never an old absolute tune location;
    # earlier constructions remain explicitly recorded in append_sources.
    for key in ("model_provenance", "cell_config", "geometry_config", "model_state"):
        if key in (new_results.get("meta") or {}):
            meta[key] = copy.deepcopy(new_results["meta"][key])
    if "model_state" not in (new_results.get("meta") or {}) and "model_provenance" in (new_results.get("meta") or {}):
        meta.pop("model_state", None)

    base_spikes = list(merged.get("spikes", []) or [])
    new_spikes  = copy.deepcopy(list(new_results.get("spikes", []) or []))
    base_trial_count = len(base_spikes)
    merged_spikes = base_spikes + new_spikes
    merged["spikes"] = merged_spikes

    # update n_trials
    sim_cfg = merged.setdefault("sim_cfg", {})
    sim_cfg["workspace_identity"] = copy.deepcopy(identity)
    sim_cfg["project_name"] = identity["project_name"]
    if not sim_cfg.get("cell"):
        sim_cfg["cell"] = identity["cell_name"]
    if not sim_cfg.get("tune"):
        sim_cfg["tune"] = identity["tune_name"]
    if (new_results.get("sim_cfg") or {}).get("tune_dir"):
        sim_cfg["tune_dir"] = new_results["sim_cfg"]["tune_dir"]
    n_trials = len(merged_spikes)
    sim_cfg["n_trials"] = n_trials
    meta["n_trials"] = n_trials
    meta["trial_ids"] = list(range(n_trials))

    # try to merge stored Vm traces if time axes match
    base_traces = merged.get("traces", {}) or {}
    new_traces  = new_results.get("traces", {}) or {}
    T_base = base_traces.get("T")
    T_new  = new_traces.get("T")
    max_traces = _resolve_trace_trials_to_save(sim_cfg, fallback=1)
    max_inputs = _resolve_inputs_to_save(sim_cfg, n_trials, max_traces)

    if max_traces <= 0:
        merged["traces"] = {}
    elif T_base is None and T_new is not None:
        merged["traces"] = copy.deepcopy(new_traces)
        for key in merged["traces"]:
            if key != "T":
                merged["traces"][key] = list(merged["traces"][key])[:max_traces]
    elif T_base is not None and T_new is not None:
        try:
            if np.shape(T_base) == np.shape(T_new) and np.allclose(np.asarray(T_base), np.asarray(T_new)):
                for key in set(base_traces) | set(new_traces):
                    if key == "T":
                        continue
                    values_base = list(base_traces.get(key, []) or [])
                    values_new = copy.deepcopy(list(new_traces.get(key, []) or []))
                    base_traces[key] = (values_base + values_new)[:max_traces]
                merged["traces"] = base_traces
            else:
                warnings.append("trace_time_mismatch: skipped merging traces")
        except Exception:
            warnings.append("trace_merge_failed: skipped merging traces")

    def _offset_trial_idx(entry: Dict[str, Any], offset: int, fallback: int) -> None:
        if "trial_idx" in entry:
            try:
                entry["trial_idx"] = offset + int(entry["trial_idx"])
                return
            except Exception:
                pass
        entry["trial_idx"] = offset + fallback

    def _merge_inputs_by_trial() -> None:
        if max_inputs <= 0:
            merged["inputs_by_trial"] = None
            return
        dst_inputs = merged.get("inputs_by_trial") or []
        src_inputs = new_results.get("inputs_by_trial") or []
        if not src_inputs:
            merged["inputs_by_trial"] = dst_inputs if dst_inputs else None
            return
        for idx, entry in enumerate(src_inputs):
            if len(dst_inputs) >= max_inputs:
                break
            new_entry = copy.deepcopy(entry)
            if isinstance(new_entry, dict):
                _offset_trial_idx(new_entry, base_trial_count, idx)
            dst_inputs.append(new_entry)
        merged["inputs_by_trial"] = dst_inputs if dst_inputs else None

    def _merge_cell_recordings_by_trial() -> None:
        if max_traces <= 0:
            merged["cell_recordings_by_trial"] = None
            return
        dst_recs = list(merged.get("cell_recordings_by_trial") or [])[:max_traces]
        src_recs = new_results.get("cell_recordings_by_trial") or []
        if not src_recs:
            merged["cell_recordings_by_trial"] = dst_recs if dst_recs else None
            return
        for idx, entry in enumerate(src_recs):
            if len(dst_recs) >= max_traces:
                break
            new_entry = copy.deepcopy(entry)
            if isinstance(new_entry, dict):
                _offset_trial_idx(new_entry, base_trial_count, idx)
            dst_recs.append(new_entry)
        merged["cell_recordings_by_trial"] = dst_recs if dst_recs else None

    def _merge_syn_placements_by_trial() -> None:
        # Placement provenance is separate from sampled variables and has no
        # voltage/synaptic trace retention cap.
        entries = list(merged.get("syn_records_by_trial") or [])
        for idx, entry in enumerate(new_results.get("syn_records_by_trial") or []):
            new_entry = copy.deepcopy(entry)
            if isinstance(new_entry, dict):
                _offset_trial_idx(new_entry, base_trial_count, idx)
            entries.append(new_entry)
        if entries:
            merged["syn_records_by_trial"] = entries
        if "syn_records" not in merged and "syn_records" in new_results:
            merged["syn_records"] = copy.deepcopy(new_results["syn_records"])

    def _merge_input_summaries() -> None:
        src_meta = new_results.get("meta", {}) or {}
        src_summaries = src_meta.get("input_summaries") or []
        if not src_summaries:
            return
        dst_summaries = meta.get("input_summaries") or []
        for idx, entry in enumerate(src_summaries):
            new_entry = copy.deepcopy(entry)
            if isinstance(new_entry, dict):
                _offset_trial_idx(new_entry, base_trial_count, idx)
            dst_summaries.append(new_entry)
        meta["input_summaries"] = dst_summaries

    def _stats_compatible(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        keys = ("bin_ms", "tstart_ms", "tstop_ms")
        for key in keys:
            if a.get(key) != b.get(key):
                return False
        a_t = a.get("t_ms") or []
        b_t = b.get("t_ms") or []
        try:
            return np.allclose(np.asarray(a_t, dtype=float), np.asarray(b_t, dtype=float))
        except Exception:
            return False

    def _merge_input_stats() -> None:
        dst_stats = meta.get("input_stats")
        src_stats = (new_results.get("meta", {}) or {}).get("input_stats")
        if not src_stats:
            return
        if dst_stats is None:
            dst_stats = copy.deepcopy(src_stats)
            for idx, entry in enumerate(dst_stats.get("trials", []) or []):
                if isinstance(entry, dict):
                    _offset_trial_idx(entry, base_trial_count, idx)
            dst_stats["group_means"] = _aggregate_input_stats(dst_stats.get("trials", []))
            meta["input_stats"] = dst_stats
            return
        if not _stats_compatible(dst_stats, src_stats):
            warnings.append("input_stats_mismatch: skipped merging input_stats")
            return
        for idx, entry in enumerate(src_stats.get("trials", []) or []):
            new_entry = copy.deepcopy(entry)
            if isinstance(new_entry, dict):
                _offset_trial_idx(new_entry, base_trial_count, idx)
            dst_stats["trials"].append(new_entry)
        dst_stats["group_means"] = _aggregate_input_stats(dst_stats.get("trials", []))
        meta["input_stats"] = dst_stats

    def _recompute_avg_rate() -> None:
        tstop = float(sim_cfg.get("tstop", 0.0))
        bin_width = float(sim_cfg.get("bins", 25.0) or 25.0)
        if tstop <= 0 or bin_width <= 0:
            warnings.append("avg_rate_skipped: invalid tstop/bin width")
            return
        bins = np.arange(0, tstop + bin_width, bin_width)
        centers = bins[:-1] + 0.5 * bin_width
        bw_s = bin_width / 1000.0
        if merged_spikes:
            per_trial_rates = []
            for tr in merged_spikes:
                tr = np.asarray(tr)
                counts, _ = np.histogram(tr, bins=bins)
                per_trial_rates.append(counts / bw_s)
            mean_rate = np.mean(per_trial_rates, axis=0)
        else:
            mean_rate = np.array([], dtype=float)
        smooth_ms = sim_cfg.get("avg_rate_curve_smooth_ms", 25.0)
        smooth_mode = sim_cfg.get("avg_rate_curve_smooth_mode", "center") or "center"
        centers, mean_rate = _smooth_rate_curve(
            centers,
            mean_rate,
            bin_width,
            smooth_ms,
            mode=str(smooth_mode),
        )
        try:
            smooth_ms_val = float(smooth_ms) if smooth_ms is not None else 0.0
        except Exception:
            smooth_ms_val = 0.0
        meta["avg_rate_curve"] = {
            "bin_ms": bin_width,
            "smooth_ms": smooth_ms_val,
            "smooth_mode": str(smooth_mode),
            "t_ms": centers.tolist(),
            "rate_hz": mean_rate.tolist(),
        }

    # sanity checks on sim_cfg compatibility
    for key in ("tstart", "tstop", "dt", "bins"):
        if key in new_results.get("sim_cfg", {}) and new_results["sim_cfg"].get(key) != sim_cfg.get(key):
            warnings.append(f"sim_cfg_mismatch:{key}")

    if meta.get("syn_config") and (new_results.get("meta", {}) or {}).get("syn_config"):
        if meta.get("syn_config") != new_results["meta"].get("syn_config"):
            warnings.append("syn_config_mismatch")

    # annotate meta
    appended = len(new_spikes)
    meta["appended_trials"] = meta.get("appended_trials", 0) + appended
    _merge_inputs_by_trial()
    _merge_cell_recordings_by_trial()
    _merge_syn_placements_by_trial()
    _merge_syn_recordings_by_trial(merged, new_results, trial_offset=base_trial_count, n_trials=n_trials)
    _merge_input_summaries()
    _merge_input_stats()
    _recompute_avg_rate()

    return merged
