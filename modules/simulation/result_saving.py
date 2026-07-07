from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from .result_paths import (
    _build_output_path,
    _copy_fit_json_sidecar,
    _json_default,
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

    from .result_appending import _append_results_to_path

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

    # Optional: auto-save plots into run_dir/plots
    if sim_cfg.get("save_plots", False):
        try:
            from modules.analysis import auto_plots

            saved_plots = auto_plots.save_default_plots(
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
            plot_files = {}
            for name, path in saved_plots.items():
                path = Path(path)
                try:
                    plot_files[name] = str(path.relative_to(run_dir))
                except ValueError:
                    plot_files[name] = str(path)
            if plot_files:
                manifest["files"]["plots"] = plot_files
        except Exception as exc:
            print(f"save_plots failed: {exc}")

    _write_json(run_dir / "run_manifest.json", manifest)

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
