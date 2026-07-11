from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


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
    raise ValueError(
        f"Unsupported results pickle format in {p}; expected a dict with 'mode' and 'sim_cfg'."
    )
