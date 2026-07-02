from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from .result_helpers import (
    _aggregate_input_stats,
    _resolve_inputs_to_save,
    _resolve_trace_trials_to_save,
    _smooth_rate_curve,
)
from .result_paths import (
    _build_output_path,
    _copy_fit_json_sidecar,
    _find_fit_json_path,
    _json_default,
    _resolve_tune_path,
    _sha256_file,
    _write_json,
)


def _save_sidecars(results: Dict[str, Any], run_dir: Path) -> Dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    mode = results.get("mode", "")
    sim_cfg = results.get("sim_cfg", {}) or {}
    meta = copy.deepcopy(results.get("meta", {}) or {})

    files: Dict[str, str] = {}

    _write_json(run_dir / "sim_cfg.json", sim_cfg)
    files["sim_cfg"] = "sim_cfg.json"

    syn_config = meta.pop("syn_config", None)
    cell_config = meta.pop("cell_config", None)
    geometry_config = meta.pop("geometry_config", None)
    avg_rate_curve = meta.pop("avg_rate_curve", None)
    input_stats = meta.pop("input_stats", None)
    input_summaries = meta.pop("input_summaries", None)
    fit_sidecar = _copy_fit_json_sidecar(sim_cfg, run_dir)
    if fit_sidecar is not None:
        meta["fit_json"] = fit_sidecar

    _write_json(run_dir / "meta.json", meta)
    files["meta"] = "meta.json"

    if syn_config is not None:
        _write_json(run_dir / "syn_config.json", syn_config)
        files["syn_config"] = "syn_config.json"
    if cell_config is not None:
        _write_json(run_dir / "cell_config.json", cell_config)
        files["cell_config"] = "cell_config.json"
    if geometry_config is not None:
        _write_json(run_dir / "geometry_config.json", geometry_config)
        files["geometry_config"] = "geometry_config.json"
    if avg_rate_curve is not None:
        _write_json(run_dir / "avg_rate_curve.json", avg_rate_curve)
        files["avg_rate_curve"] = "avg_rate_curve.json"
    if input_stats is not None:
        _write_json(run_dir / "input_stats.json", input_stats)
        files["input_stats"] = "input_stats.json"
    if input_summaries is not None:
        _write_json(run_dir / "input_summaries.json", input_summaries)
        files["input_summaries"] = "input_summaries.json"
    if fit_sidecar is not None:
        files["fit_json"] = fit_sidecar["filename"]

    spikes = results.get("spikes", None)
    if spikes is not None:
        spike_path = run_dir / "spikes.npz"
        if mode == "multi":
            np.savez(spike_path, spikes=np.array(spikes, dtype=object))
        else:
            np.savez(spike_path, spikes=np.asarray(spikes))
        files["spikes"] = "spikes.npz"

    traces = results.get("traces", {}) or {}
    if "T" in traces:
        trace_path = run_dir / "traces.npz"
        if mode == "multi":
            np.savez(
                trace_path,
                T=np.asarray(traces.get("T")),
                V_trials=np.array(traces.get("V", []), dtype=object),
            )
        else:
            np.savez(
                trace_path,
                T=np.asarray(traces.get("T")),
                V=np.asarray(traces.get("V")),
            )
        files["traces"] = "traces.npz"

    cell_recordings = results.get("cell_recordings")
    if cell_recordings is not None:
        cell_rec_path = run_dir / "cell_recordings.pkl"
        with cell_rec_path.open("wb") as f:
            pickle.dump(cell_recordings, f)
        files["cell_recordings"] = "cell_recordings.pkl"

    cell_recordings_by_trial = results.get("cell_recordings_by_trial")
    if cell_recordings_by_trial:
        cell_rec_trial_path = run_dir / "cell_recordings_by_trial.pkl"
        with cell_rec_trial_path.open("wb") as f:
            pickle.dump(cell_recordings_by_trial, f)
        files["cell_recordings_by_trial"] = "cell_recordings_by_trial.pkl"

    inputs_payload = {}
    if results.get("inputs") is not None:
        inputs_payload["inputs"] = results.get("inputs")
    if results.get("inputs_by_trial") is not None:
        inputs_payload["inputs_by_trial"] = results.get("inputs_by_trial")
    if inputs_payload:
        inputs_path = run_dir / "inputs_sample.pkl"
        with inputs_path.open("wb") as f:
            pickle.dump(inputs_payload, f)
        files["inputs_sample"] = "inputs_sample.pkl"

    if sim_cfg.get("save_syn_records_sidecar", True):
        syn_records = results.get("syn_records")
        if syn_records:
            syn_path = run_dir / "syn_records.pkl"
            with syn_path.open("wb") as f:
                pickle.dump(syn_records, f)
            files["syn_records"] = "syn_records.pkl"

    syn_records_by_trial = results.get("syn_records_by_trial")
    if syn_records_by_trial:
        syn_by_trial_path = run_dir / "syn_records_by_trial.pkl"
        with syn_by_trial_path.open("wb") as f:
            pickle.dump(syn_records_by_trial, f)
        files["syn_records_by_trial"] = "syn_records_by_trial.pkl"

    return files



def save_results(
    results: Dict[str, Any],
    base_dir: Union[str, Path] = "output_data",
) -> Optional[Path]:    

    sim_cfg = results.get("sim_cfg", {})
    out_path = _build_output_path(sim_cfg, base_dir=base_dir)
    if out_path is None:
        append_to = sim_cfg.get("append_to")
        if append_to:
            try:
                _append_results_to_path(results, append_to, base_dir=base_dir)
            except Exception as exc:
                print(f"append_to failed: {exc}")
        return None

    fmt = sim_cfg.get("output_format", "pickle")

    run_dir = out_path.parent
    manifest = {
        "format_version": 1,
        "mode": results.get("mode", ""),
        "output_stem": sim_cfg.get("output"),
        "files": {},
    }

    save_sidecars = sim_cfg.get("save_sidecars", True)
    save_full_results = sim_cfg.get("save_full_results", False)
    if not save_sidecars and not save_full_results:
        # ensure at least one artifact is written
        save_sidecars = True
    if save_sidecars:
        manifest["files"].update(_save_sidecars(results, run_dir))

    if save_full_results and fmt == "npz":
        # compact, interoperable: arrays + JSON metadata
        mode = results.get("mode", "")
        meta = results.get("meta", {})
        traces = results.get("traces", {}) or {}
        spikes = results.get("spikes", None)

        payload = {
            "mode": np.array(mode),
            "sim_cfg_json": np.array(json.dumps(sim_cfg, default=_json_default)),
            "meta_json": np.array(json.dumps(meta, default=_json_default)),
        }

        if "T" in traces:
            payload["T"] = traces["T"]

        if mode == "single":
            if "V" in traces:
                payload["V"] = traces["V"]
            if spikes is not None:
                payload["spikes"] = spikes
            if results.get("cell_recordings") is not None:
                payload["cell_recordings"] = np.array(results.get("cell_recordings"), dtype=object)
        elif mode == "multi":
            if spikes is not None:
                payload["spikes"] = np.array(spikes, dtype=object)
            if "V" in traces:
                payload["V_trials"] = np.array(traces["V"], dtype=object)
            if results.get("cell_recordings_by_trial") is not None:
                payload["cell_recordings_by_trial"] = np.array(
                    results.get("cell_recordings_by_trial"), dtype=object
                )

        np.savez(out_path, **payload)
        manifest["files"]["results_npz"] = out_path.name
    elif save_full_results:
        # full Python dict with everything
        with out_path.open("wb") as f:
            pickle.dump(results, f)
        manifest["files"]["results_pkl"] = out_path.name

    _write_json(run_dir / "run_manifest.json", manifest)

    # Optional: auto-save plots into run_dir/plots
    if sim_cfg.get("save_plots", False):
        try:
            from modules.analysis import analysis as analysis_mod

            analysis_mod.save_default_plots(
                results,
                run_dir,
                save_inputs=bool(sim_cfg.get("save_plots_inputs", True)),
                save_synapses=bool(sim_cfg.get("save_plots_synapses", False)),
                win_size=float(sim_cfg.get("plots_win_size", 25.0)),
                input_bin_ms=sim_cfg.get("plots_input_bin_ms", None),
                input_smooth_ms=sim_cfg.get("plots_input_smooth_ms", 25.0),
                raster_style=str(sim_cfg.get("plots_raster_style", "dot")),
                plot_mode=str(sim_cfg.get("save_plots_mode", "default")),
                single_plot_preset=sim_cfg.get("save_plots_single_plot_preset", None),
                overwrite=bool(sim_cfg.get("save_plots_overwrite", False)),
            )
        except Exception as exc:
            print(f"save_plots failed: {exc}")

    append_to = sim_cfg.get("append_to")
    if sim_cfg.get("append_enabled", True) and append_to:
        try:
            _append_results_to_path(results, append_to, base_dir=run_dir.parent)
        except Exception as exc:
            print(f"append_to failed: {exc}")

    return out_path if save_full_results else (run_dir / "run_manifest.json")

def save_results_with_name(
    results: Dict[str, Any],
    output_stem: str,
    base_dir: Union[str, Path] = "output_data",
) -> Optional[Path]:
    """
    Manually save an existing results dict under a given output name,
    regardless of what was set in sim_cfg['output'] originally.

    Example:
        results["sim_cfg"]["color"] = "m"
        save_results_with_name(results, "sst2_seg_tuned_batch1")
    """
    sim_cfg = results.setdefault("sim_cfg", {})
    sim_cfg["output"] = str(output_stem)
    return save_results(results, base_dir=base_dir)


def _ensure_multi_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce a single-trial results dict into multi-trial format for appending.
    """
    mode = results.get("mode", "single")
    if mode == "multi":
        return results
    if mode != "single":
        raise ValueError(f"append_to: unsupported results mode {mode!r}")

    sim_cfg = copy.deepcopy(results.get("sim_cfg", {}))
    traces = results.get("traces", {}) or {}
    inputs = results.get("inputs", None)
    cell_recordings = results.get("cell_recordings", None)

    spikes = results.get("spikes", None)
    spikes_list = [spikes] if spikes is not None else []

    multi = {
        "mode": "multi",
        "sim_cfg": sim_cfg,
        "spikes": spikes_list,
        "traces": {},
        "cell_recordings_by_trial": None,
        "inputs_by_trial": None,
        "meta": copy.deepcopy(results.get("meta", {})),
    }

    if "T" in traces and "V" in traces:
        multi["traces"] = {"T": traces["T"], "V": [traces["V"]]}

    if inputs:
        multi["inputs_by_trial"] = [{"trial_idx": 0, "inputs": inputs}]
    if cell_recordings is not None:
        multi["cell_recordings_by_trial"] = [
            {"trial_idx": 0, "recordings": cell_recordings}
        ]

    meta = multi.setdefault("meta", {})
    meta["n_trials"] = len(spikes_list)
    meta["trial_ids"] = list(range(len(spikes_list)))
    return multi


def _write_results_file(results: Dict[str, Any], out_path: Path, fmt: str) -> None:
    fmt = str(fmt).lower()
    if fmt == "npz":
        mode = results.get("mode", "")
        sim_cfg = results.get("sim_cfg", {})
        meta = results.get("meta", {})
        traces = results.get("traces", {}) or {}
        spikes = results.get("spikes", None)

        payload = {
            "mode": np.array(mode),
            "sim_cfg_json": np.array(json.dumps(sim_cfg, default=_json_default)),
            "meta_json": np.array(json.dumps(meta, default=_json_default)),
        }

        if "T" in traces:
            payload["T"] = traces["T"]

        if mode == "single":
            if "V" in traces:
                payload["V"] = traces["V"]
            if spikes is not None:
                payload["spikes"] = spikes
            if results.get("cell_recordings") is not None:
                payload["cell_recordings"] = np.array(results.get("cell_recordings"), dtype=object)
        elif mode == "multi":
            if spikes is not None:
                payload["spikes"] = np.array(spikes, dtype=object)
            if "V" in traces:
                payload["V_trials"] = np.array(traces["V"], dtype=object)
            if results.get("cell_recordings_by_trial") is not None:
                payload["cell_recordings_by_trial"] = np.array(
                    results.get("cell_recordings_by_trial"), dtype=object
                )

        np.savez(out_path, **payload)
    else:
        with out_path.open("wb") as f:
            pickle.dump(results, f)


def _write_results_to_run_dir(
    results: Dict[str, Any],
    run_dir: Path,
    *,
    fmt: Optional[str] = None,
    results_name: Optional[str] = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    sim_cfg = results.get("sim_cfg", {}) or {}

    if fmt is None:
        fmt = sim_cfg.get("output_format", "pickle")
    fmt = str(fmt).lower()

    if results_name is None:
        cell = sim_cfg.get("cell", "cell")
        tune = sim_cfg.get("tune", "tune")
        stem = sim_cfg.get("output") or "results"
        suffix = ".npz" if fmt == "npz" else ".pkl"
        results_name = f"{cell}_{tune}_{stem}{suffix}"

    out_path = run_dir / results_name

    manifest = {
        "format_version": 1,
        "mode": results.get("mode", ""),
        "output_stem": sim_cfg.get("output"),
        "files": {},
    }

    save_sidecars = sim_cfg.get("save_sidecars", True)
    if save_sidecars:
        manifest["files"].update(_save_sidecars(results, run_dir))

    if fmt == "npz":
        _write_results_file(results, out_path, "npz")
        manifest["files"]["results_npz"] = out_path.name
    else:
        _write_results_file(results, out_path, "pickle")
        manifest["files"]["results_pkl"] = out_path.name

    _write_json(run_dir / "run_manifest.json", manifest)
    return out_path


def _append_results_to_path(
    results: Dict[str, Any],
    append_to: Union[str, Path],
    *,
    base_dir: Union[str, Path, None] = None,
) -> Optional[Path]:
    """
    Append results to an existing file/folder (or create a new one).
    """
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

    if append_path.exists():
        base_results = load_results(append_path)
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

    merged = copy.deepcopy(base_results)
    meta = merged.setdefault("meta", {})
    warnings = meta.setdefault("append_warnings", [])

    base_spikes = list(merged.get("spikes", []) or [])
    new_spikes  = list(new_results.get("spikes", []) or [])
    base_trial_count = len(base_spikes)
    merged_spikes = base_spikes + new_spikes
    merged["spikes"] = merged_spikes

    # update n_trials
    sim_cfg = merged.setdefault("sim_cfg", {})
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
    elif T_base is not None and T_new is not None:
        try:
            if np.allclose(np.asarray(T_base), np.asarray(T_new)):
                V_base = list(base_traces.get("V", []) or [])
                V_new  = list(new_traces.get("V", []) or [])
                base_traces["V"] = (V_base + V_new)[:max_traces]
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
    _merge_input_summaries()
    _merge_input_stats()
    _recompute_avg_rate()

    return merged



def _load_from_manifest(manifest_path: Path) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    run_dir = manifest_path.parent
    files = manifest.get("files", {}) or {}

    pkl_name = files.get("results_pkl")
    if pkl_name:
        pkl_path = run_dir / pkl_name
        if pkl_path.is_file():
            return load_results(pkl_path)

    npz_name = files.get("results_npz")
    if npz_name:
        npz_path = run_dir / npz_name
        if npz_path.is_file():
            return load_results(npz_path)

    mode = manifest.get("mode", "single")

    sim_cfg = {}
    if files.get("sim_cfg"):
        sim_cfg = json.loads((run_dir / files["sim_cfg"]).read_text())

    meta = {}
    if files.get("meta"):
        meta = json.loads((run_dir / files["meta"]).read_text())

    if files.get("syn_config"):
        meta["syn_config"] = json.loads((run_dir / files["syn_config"]).read_text())
    if files.get("cell_config"):
        meta["cell_config"] = json.loads((run_dir / files["cell_config"]).read_text())
    if files.get("geometry_config"):
        meta["geometry_config"] = json.loads((run_dir / files["geometry_config"]).read_text())
    if files.get("avg_rate_curve"):
        meta["avg_rate_curve"] = json.loads((run_dir / files["avg_rate_curve"]).read_text())
    if files.get("input_stats"):
        meta["input_stats"] = json.loads((run_dir / files["input_stats"]).read_text())
    if files.get("input_summaries"):
        meta["input_summaries"] = json.loads((run_dir / files["input_summaries"]).read_text())

    spikes = None
    if files.get("spikes"):
        sp = np.load(run_dir / files["spikes"], allow_pickle=True)
        if "spikes" in sp.files:
            spikes = sp["spikes"]
            if mode == "multi":
                spikes = list(spikes)

    traces: Dict[str, Any] = {}
    if files.get("traces"):
        tr = np.load(run_dir / files["traces"], allow_pickle=True)
        if "T" in tr.files:
            traces["T"] = tr["T"]
        if mode == "multi":
            if "V_trials" in tr.files:
                traces["V"] = list(tr["V_trials"])
        else:
            if "V" in tr.files:
                traces["V"] = tr["V"]

    cell_recordings = None
    if files.get("cell_recordings"):
        with (run_dir / files["cell_recordings"]).open("rb") as f:
            cell_recordings = pickle.load(f)
    cell_recordings_by_trial = None
    if files.get("cell_recordings_by_trial"):
        with (run_dir / files["cell_recordings_by_trial"]).open("rb") as f:
            cell_recordings_by_trial = pickle.load(f)

    inputs_payload = None
    if files.get("inputs_sample"):
        with (run_dir / files["inputs_sample"]).open("rb") as f:
            inputs_payload = pickle.load(f)

    syn_records = None
    if files.get("syn_records"):
        with (run_dir / files["syn_records"]).open("rb") as f:
            syn_records = pickle.load(f)
    syn_records_by_trial = None
    if files.get("syn_records_by_trial"):
        with (run_dir / files["syn_records_by_trial"]).open("rb") as f:
            syn_records_by_trial = pickle.load(f)

    results = {
        "mode": mode,
        "sim_cfg": sim_cfg,
        "traces": traces,
        "spikes": spikes,
        "meta": meta,
    }
    if inputs_payload:
        if "inputs" in inputs_payload:
            results["inputs"] = inputs_payload["inputs"]
        if "inputs_by_trial" in inputs_payload:
            results["inputs_by_trial"] = inputs_payload["inputs_by_trial"]
    if syn_records is not None:
        results["syn_records"] = syn_records
    if syn_records_by_trial is not None:
        results["syn_records_by_trial"] = syn_records_by_trial
    if cell_recordings is not None:
        results["cell_recordings"] = cell_recordings
    if cell_recordings_by_trial is not None:
        results["cell_recordings_by_trial"] = cell_recordings_by_trial

    return results


def load_results(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        manifest = p / "run_manifest.json"
        if manifest.is_file():
            return _load_from_manifest(manifest)
        results_dir = p / "results"
        if results_dir.is_dir():
            return load_results(results_dir)
        candidates = list(p.glob("*.pkl")) + list(p.glob("*.npz"))
        if len(candidates) == 1:
            return load_results(candidates[0])
        raise FileNotFoundError(f"No run_manifest.json or single results file in {p}")

    if p.name == "run_manifest.json":
        return _load_from_manifest(p)

    suffix = p.suffix.lower()

    if suffix == ".npz":
        data = np.load(p, allow_pickle=True)

        mode = str(data["mode"])
        sim_cfg = json.loads(str(data["sim_cfg_json"]))
        meta = json.loads(str(data["meta_json"]))

        traces: Dict[str, Any] = {}
        spikes = None
        cell_recordings = None
        cell_recordings_by_trial = None

        if "T" in data.files:
            traces["T"] = data["T"]

        if mode == "single":
            if "V" in data.files:
                traces["V"] = data["V"]
            if "spikes" in data.files:
                spikes = data["spikes"]
            if "cell_recordings" in data.files:
                try:
                    cell_recordings = data["cell_recordings"].item()
                except Exception:
                    cell_recordings = data["cell_recordings"]
        elif mode == "multi":
            if "V_trials" in data.files:
                traces["V"] = list(data["V_trials"])
            if "spikes" in data.files:
                spikes = list(data["spikes"])
            if "cell_recordings_by_trial" in data.files:
                cell_recordings_by_trial = list(data["cell_recordings_by_trial"])

        out = {
            "mode": mode,
            "sim_cfg": sim_cfg,
            "traces": traces,
            "spikes": spikes,
            "meta": meta,
        }
        if cell_recordings is not None:
            out["cell_recordings"] = cell_recordings
        if cell_recordings_by_trial is not None:
            out["cell_recordings_by_trial"] = cell_recordings_by_trial
        return out

    with p.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "mode" in payload and "sim_cfg" in payload:
        return payload
    try:
        return load_old_multi_results(p)
    except Exception:
        return payload
        

# ---------------------------------------------------------------------
# Compatibility loader for old multi-trial pickle outputs
# ---------------------------------------------------------------------
def load_old_multi_results(
    path: Union[str, Path],
    *,
    label: str = None,
    color: str = None,
    tstop: float = 1200.0,
    bins: float = 5.0,
    delay: float = 0.0,
) -> Dict[str, Any]:
    """
    Load an old multi-trial pickle (e.g. tune1_1000tr1200ms.pkl) and
    wrap it into a new-style `results` dict that `plot_results` can use.

    Parameters
    ----------
    path : str or Path
        Path to the old .pkl file.
    label : str, optional
        Which key in all_param_data to use (e.g. 'base tune').
        If None, use 'base tune' if present, otherwise the first key.
    color : str, optional
        Optional plotting color; stored into sim_cfg['color'].
    tstop, bins, delay : float
        Timing parameters in ms, used to build sim_cfg for plotting.

    Returns
    -------
    results : dict
        New-style results dict with:
          - mode='multi'
          - sim_cfg: contains tstop, bins, delay, n_trials, color, ...
          - spikes: list of spike-time arrays (one per trial)
          - traces: {}
          - meta: includes 'source' and 'label'
    """
    p = Path(path)
    with p.open("rb") as f:
        payload = pickle.load(f)

    def _load_json_cfg(cfg_path: Path) -> Optional[Dict[str, Any]]:
        if not cfg_path.is_file():
            return None
        try:
            return json.loads(cfg_path.read_text())
        except Exception:
            return None

    def _find_tune_cfg(pkl_path: Path) -> Optional[Dict[str, Any]]:
        for parent in pkl_path.parents:
            if parent.name == "output_data":
                tune_dir = parent.parent
                for cfg_path in (
                    tune_dir / "cell_configs" / "sim_config.json",
                    tune_dir / "sim_config.json",
                    tune_dir / "cell_config.json",
                ):
                    cfg = _load_json_cfg(cfg_path)
                    if cfg:
                        return cfg
                break
        return None

    if color is None:
        for cfg_path in (
            p.parent / "cell_config.json",
            p.parent / "sim_cfg.json",
            p.parent / "sim_config.json",
            p.parent.parent / "sim_cfg.json",
            p.parent.parent / "sim_config.json",
            p.parent.parent / "cell_config.json",
        ):
            if not cfg_path.is_file():
                continue
            try:
                cfg = json.loads(cfg_path.read_text())
            except Exception:
                continue
            cfg_color = (cfg or {}).get("color")
            if cfg_color:
                color = cfg_color
                break

    # Try to detect shape:
    #  Case 1: {'all_param_data': {...}, 'param_study': ..., 'sim_params': ...}
    if isinstance(payload, dict) and "all_param_data" in payload:
        all_param_data = payload["all_param_data"]
        param_study_old = payload.get("param_study", {})
        sim_params_old = payload.get("sim_params", {})
    else:
        # Case 2: assume payload itself is the all_param_data dict
        all_param_data = payload
        param_study_old = {}
        sim_params_old = {}

    # Choose which group to use (e.g. 'base tune')
    keys = list(all_param_data.keys())
    if not keys:
        raise ValueError(f"Old results file {p} has no parameter groups.")

    if label is None:
        if "base tune" in all_param_data:
            label = "base tune"
        else:
            label = keys[0]

    if label not in all_param_data:
        raise KeyError(f"Label {label!r} not found in old all_param_data; "
                       f"available keys = {keys}")

    spikes_by_trial = all_param_data[label]
    n_trials = len(spikes_by_trial)

    # Build a minimal sim_cfg suitable for plotting
    sim_cfg = {
        "dt":      float(sim_params_old.get("dt", 0.025)),
        "tstart":  float(sim_params_old.get("tstart", 0.0)),
        "tstop":   float(sim_params_old.get("tstop", tstop)),
        "bins":    float(sim_params_old.get("bins", bins)),
        "delay":   float(sim_params_old.get("delay", delay)),
        "n_trials": n_trials,
        "n_traces_to_save": 0,
        "color":   color,
        "param_study": {
            "input_type": None,
            "param_type": label,
            "param_vals": [None],
            "n_trials": n_trials,
        },
        "output": None,
        "output_format": "pickle",
        "cell": None,
        "tune": None,
        "jitter": None,
        "seed": None,
        "trial_randomness": "synapses",
    }

    tune_cfg = _find_tune_cfg(p)
    if tune_cfg:
        for key in ("stim_start_ms", "stim_duration_ms", "stim_stop_ms", "delay"):
            if sim_cfg.get(key) is None and tune_cfg.get(key) is not None:
                sim_cfg[key] = tune_cfg.get(key)

    results = {
        "mode": "multi",
        "sim_cfg": sim_cfg,
        "spikes": spikes_by_trial,
        "traces": {},  # no Vm available in old file
        "meta": {
            "source": "old_pipeline",
            "label": label,
            "path": str(p),
        },
    }
    return results
