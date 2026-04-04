#!/usr/bin/env python3
"""
Swap saved exemplar data in an existing SCP run output.

Supports:
  - Vm trace exemplar (`--update vm`)
  - Input raster payload exemplar (`--update inputs`)
  - Both together (`--update both`)

Default mode is dry-run. Add --write to apply updates.
"""

import argparse
import copy
import json
import os
import pickle
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class VmSwapReport:
    def __init__(
        self,
        *,
        target_run: Path,
        source_desc: str,
        dry_run: bool,
        target_trial_idx: int,
        source_trial_idx: int,
        source_run: Optional[Path] = None,
    ) -> None:
        self.target_run = target_run
        self.source_desc = source_desc
        self.dry_run = dry_run
        self.target_trial_idx = target_trial_idx
        self.source_trial_idx = source_trial_idx
        self.source_run = source_run
        self.generated_source_run: Optional[Path] = None
        self.target_mode = ""
        self.source_mode = ""
        self.target_trace_count = 0
        self.source_trace_count = 0
        self.target_time_len = 0
        self.source_time_len = 0
        self.source_input_trial_idx: Optional[int] = None
        self.target_input_trial_idx: Optional[int] = None
        self.target_input_count = 0
        self.source_input_count = 0
        self.touched_files: List[Path] = []
        self.backup_files: List[Path] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _backup_file(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.bak_{stamp}")
    shutil.copy2(path, backup)
    return backup


def _resolve_manifest_path(run_path: Path) -> Path:
    p = run_path.expanduser().resolve()
    if p.is_file():
        if p.name == "run_manifest.json":
            return p
        if p.suffix.lower() in (".pkl", ".npz"):
            for cand in (p.parent / "run_manifest.json", p.parent.parent / "run_manifest.json"):
                if cand.is_file():
                    return cand
            raise FileNotFoundError(f"Could not locate run_manifest.json from file path: {p}")
        raise FileNotFoundError(f"Unsupported file target: {p}")

    if p.is_dir():
        for cand in (p / "run_manifest.json", p / "results" / "run_manifest.json"):
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"No run_manifest.json found under: {p}")

    raise FileNotFoundError(f"Run path not found: {p}")


def _coerce_trace_list(v_raw: Any, mode: str) -> List[np.ndarray]:
    mode = str(mode or "single").lower()

    if mode == "multi":
        if isinstance(v_raw, list):
            return [np.asarray(v, dtype=float).ravel() for v in v_raw]

        arr_obj = np.asarray(v_raw, dtype=object)
        if arr_obj.ndim == 1 and arr_obj.dtype == object:
            return [np.asarray(v, dtype=float).ravel() for v in arr_obj.tolist()]

        arr = np.asarray(v_raw, dtype=float)
        if arr.ndim == 1:
            return [arr.ravel()]
        return [np.asarray(row, dtype=float).ravel() for row in arr]

    return [np.asarray(v_raw, dtype=float).ravel()]


def _validate_trace_shapes(T: np.ndarray, V_list: List[np.ndarray], source_label: str) -> None:
    if T.size == 0:
        raise ValueError(f"{source_label}: trace time axis is empty")
    if not V_list:
        raise ValueError(f"{source_label}: trace list is empty")
    for i, v in enumerate(V_list):
        if v.size != T.size:
            raise ValueError(
                f"{source_label}: trace length mismatch at index {i}: len(T)={T.size}, len(V[{i}])={v.size}"
            )


def _load_trace_npz(npz_path: Path, mode_hint: Optional[str] = None) -> Dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        mode = str(mode_hint or "").strip().lower()
        if "mode" in data.files:
            try:
                mode = str(data["mode"]).strip().lower()
            except Exception:
                pass
        if mode not in ("single", "multi"):
            mode = "multi" if "V_trials" in data.files else "single"

        if "T" not in data.files:
            raise ValueError(f"Missing T in npz: {npz_path}")
        T = np.asarray(data["T"], dtype=float).ravel()

        if mode == "multi":
            if "V_trials" in data.files:
                v_raw = data["V_trials"]
            elif "V" in data.files:
                v_raw = data["V"]
            else:
                raise ValueError(f"Missing V_trials/V in npz: {npz_path}")
        else:
            if "V" in data.files:
                v_raw = data["V"]
            elif "V_trials" in data.files:
                v_raw = data["V_trials"]
                mode = "multi"
            else:
                raise ValueError(f"Missing V/V_trials in npz: {npz_path}")

        V_list = _coerce_trace_list(v_raw, mode)

        sim_cfg = None
        if "sim_cfg_json" in data.files:
            try:
                sim_cfg = json.loads(str(data["sim_cfg_json"]))
            except Exception:
                sim_cfg = None

    _validate_trace_shapes(T, V_list, source_label=str(npz_path))
    return {
        "mode": mode,
        "T": T,
        "V_list": V_list,
        "sim_cfg": sim_cfg,
    }


def _load_trace_pkl(pkl_path: Path, mode_hint: Optional[str] = None) -> Dict[str, Any]:
    with pkl_path.open("rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Pickle payload is not a dict: {pkl_path}")

    mode = str(payload.get("mode", mode_hint or "single")).strip().lower()
    if mode not in ("single", "multi"):
        mode = "single"

    traces = payload.get("traces", {}) or {}
    if "T" not in traces or "V" not in traces:
        raise ValueError(f"Pickle payload missing traces.T or traces.V: {pkl_path}")

    T = np.asarray(traces.get("T"), dtype=float).ravel()
    V_list = _coerce_trace_list(traces.get("V"), mode)
    _validate_trace_shapes(T, V_list, source_label=str(pkl_path))

    sim_cfg = payload.get("sim_cfg") if isinstance(payload.get("sim_cfg"), dict) else None
    return {
        "mode": mode,
        "T": T,
        "V_list": V_list,
        "sim_cfg": sim_cfg,
    }


def _load_trace_bundle(run_path: Path, *, require_traces: bool = True) -> Dict[str, Any]:
    manifest_path = _resolve_manifest_path(run_path)
    manifest = _read_json(manifest_path)
    files = manifest.get("files", {}) or {}
    results_dir = manifest_path.parent

    mode_manifest = str(manifest.get("mode", "single")).strip().lower()
    if mode_manifest not in ("single", "multi"):
        mode_manifest = "single"

    sim_cfg = None
    sim_cfg_rel = files.get("sim_cfg")
    if isinstance(sim_cfg_rel, str):
        sim_cfg_path = (results_dir / sim_cfg_rel).resolve()
        if sim_cfg_path.is_file():
            try:
                sim_cfg_val = _read_json(sim_cfg_path)
                if isinstance(sim_cfg_val, dict):
                    sim_cfg = sim_cfg_val
            except Exception:
                sim_cfg = None

    trace_info = None
    traces_rel = files.get("traces")
    if isinstance(traces_rel, str):
        trace_path = (results_dir / traces_rel).resolve()
        if trace_path.is_file():
            trace_info = _load_trace_npz(trace_path, mode_hint=mode_manifest)

    if trace_info is None:
        npz_rel = files.get("results_npz")
        if isinstance(npz_rel, str):
            npz_path = (results_dir / npz_rel).resolve()
            if npz_path.is_file():
                trace_info = _load_trace_npz(npz_path, mode_hint=mode_manifest)

    if trace_info is None:
        pkl_rel = files.get("results_pkl")
        if isinstance(pkl_rel, str):
            pkl_path = (results_dir / pkl_rel).resolve()
            if pkl_path.is_file():
                trace_info = _load_trace_pkl(pkl_path, mode_hint=mode_manifest)

    if trace_info is None and require_traces:
        raise FileNotFoundError(
            "No trace artifact found in manifest (traces sidecar, results_npz, or results_pkl)."
        )

    if trace_info is not None:
        if sim_cfg is None and isinstance(trace_info.get("sim_cfg"), dict):
            sim_cfg = trace_info.get("sim_cfg")
        mode_out = str(trace_info.get("mode") or mode_manifest)
        T_out = np.asarray(trace_info["T"], dtype=float).ravel()
        V_out = [np.asarray(v, dtype=float).ravel() for v in trace_info["V_list"]]
    else:
        mode_out = str(mode_manifest)
        T_out = np.array([], dtype=float)
        V_out = []

    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "files": files,
        "results_dir": results_dir,
        "mode": mode_out,
        "T": T_out,
        "V_list": V_out,
        "sim_cfg": sim_cfg if isinstance(sim_cfg, dict) else {},
    }


def _load_inputs_payload(bundle: Dict[str, Any]) -> Dict[str, Any]:
    files = bundle.get("files", {}) or {}
    results_dir = bundle.get("results_dir")
    if not isinstance(results_dir, Path):
        raise TypeError("bundle.results_dir must be a Path")

    inputs_sample_path = None
    payload = None
    rel = files.get("inputs_sample")
    if isinstance(rel, str):
        cand = (results_dir / rel).resolve()
        if cand.is_file():
            with cand.open("rb") as f:
                data = pickle.load(f)
            if not isinstance(data, dict):
                raise TypeError(f"inputs_sample payload is not dict: {cand}")
            payload = dict(data)
            inputs_sample_path = cand

    results_pkl_path = None
    rel_pkl = files.get("results_pkl")
    if isinstance(rel_pkl, str):
        cand = (results_dir / rel_pkl).resolve()
        if cand.is_file():
            results_pkl_path = cand
            if payload is None:
                with cand.open("rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    p: Dict[str, Any] = {}
                    if data.get("inputs") is not None:
                        p["inputs"] = data.get("inputs")
                    if data.get("inputs_by_trial") is not None:
                        p["inputs_by_trial"] = data.get("inputs_by_trial")
                    if p:
                        payload = p

    return {
        "payload": payload,
        "inputs_sample_path": inputs_sample_path,
        "results_pkl_path": results_pkl_path,
    }


def _count_inputs_payload(payload: Optional[Dict[str, Any]]) -> int:
    if not isinstance(payload, dict):
        return 0
    trials = payload.get("inputs_by_trial")
    if isinstance(trials, list) and trials:
        return len(trials)
    if payload.get("inputs") is not None:
        return 1
    return 0


def _pick_inputs_for_trial(payload: Dict[str, Any], idx: int, *, label: str) -> Any:
    if not isinstance(payload, dict):
        raise ValueError(f"{label}: missing inputs payload")
    trials = payload.get("inputs_by_trial")
    if isinstance(trials, list) and trials:
        if idx < 0 or idx >= len(trials):
            raise IndexError(f"{label} input trial idx {idx} out of range [0, {len(trials)-1}]")
        entry = trials[idx]
        if isinstance(entry, dict) and entry.get("inputs") is not None:
            return copy.deepcopy(entry.get("inputs"))
        raise ValueError(f"{label}: inputs_by_trial[{idx}] missing 'inputs'")

    if payload.get("inputs") is not None:
        if idx != 0:
            raise IndexError(f"{label} input trial idx {idx} out of range [0, 0]")
        return copy.deepcopy(payload.get("inputs"))

    raise ValueError(f"{label}: payload missing inputs/inputs_by_trial")


def _set_inputs_for_trial(
    target_payload: Optional[Dict[str, Any]],
    *,
    mode: str,
    idx: int,
    new_inputs: Any,
) -> Dict[str, Any]:
    payload = dict(target_payload) if isinstance(target_payload, dict) else {}
    mode = str(mode or "single").lower()

    trials = payload.get("inputs_by_trial")
    if isinstance(trials, list) and trials:
        if idx < 0 or idx >= len(trials):
            raise IndexError(f"target input trial idx {idx} out of range [0, {len(trials)-1}]")
        entry = trials[idx]
        if isinstance(entry, dict):
            entry = dict(entry)
            entry["inputs"] = copy.deepcopy(new_inputs)
            if "trial_idx" not in entry:
                entry["trial_idx"] = int(idx)
            trials[idx] = entry
        else:
            trials[idx] = {"trial_idx": int(idx), "inputs": copy.deepcopy(new_inputs)}
        payload["inputs_by_trial"] = trials
        payload.pop("inputs", None)
        return payload

    if mode == "multi":
        if idx != 0:
            raise IndexError(
                "target has no saved inputs_by_trial list; can only set index 0 (or rerun with more saved inputs)"
            )
        payload["inputs_by_trial"] = [{"trial_idx": 0, "inputs": copy.deepcopy(new_inputs)}]
        payload.pop("inputs", None)
        return payload

    if idx != 0:
        raise IndexError(f"target input trial idx {idx} out of range [0, 0]")
    payload["inputs"] = copy.deepcopy(new_inputs)
    payload.pop("inputs_by_trial", None)
    return payload


def _write_inputs_sidecar(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("wb") as f:
        pickle.dump(payload, f)


def _write_results_pkl_inputs(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"results pickle is not a dict: {path}")

    if payload.get("inputs_by_trial") is not None:
        data["inputs_by_trial"] = payload.get("inputs_by_trial")
        data.pop("inputs", None)
    elif payload.get("inputs") is not None:
        data["inputs"] = payload.get("inputs")
        data.pop("inputs_by_trial", None)
    else:
        data.pop("inputs", None)
        data.pop("inputs_by_trial", None)

    with path.open("wb") as f:
        pickle.dump(data, f)


def _pick_trace(V_list: List[np.ndarray], idx: int, *, label: str) -> np.ndarray:
    if idx < 0 or idx >= len(V_list):
        raise IndexError(f"{label} trial idx {idx} out of range [0, {len(V_list)-1}]")
    return np.asarray(V_list[idx], dtype=float).ravel()


def _write_traces_sidecar(path: Path, mode: str, T: np.ndarray, V_list: List[np.ndarray]) -> None:
    mode = str(mode or "single").lower()
    if mode == "multi":
        np.savez(path, T=np.asarray(T), V_trials=np.array(V_list, dtype=object))
    else:
        np.savez(path, T=np.asarray(T), V=np.asarray(V_list[0], dtype=float))


def _write_results_npz(path: Path, mode: str, T: np.ndarray, V_list: List[np.ndarray]) -> None:
    with np.load(path, allow_pickle=True) as data:
        payload = {k: data[k] for k in data.files}

    payload["T"] = np.asarray(T)
    if str(mode or "single").lower() == "multi":
        payload.pop("V", None)
        payload["V_trials"] = np.array(V_list, dtype=object)
    else:
        payload.pop("V_trials", None)
        payload["V"] = np.asarray(V_list[0], dtype=float)

    np.savez(path, **payload)


def _write_results_pkl(path: Path, mode: str, T: np.ndarray, V_list: List[np.ndarray]) -> None:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"results pickle is not a dict: {path}")

    traces: Dict[str, Any] = payload.get("traces", {}) or {}
    traces["T"] = np.asarray(T)
    if str(mode or "single").lower() == "multi":
        traces["V"] = [np.asarray(v, dtype=float) for v in V_list]
    else:
        traces["V"] = np.asarray(V_list[0], dtype=float)
    payload["traces"] = traces

    with path.open("wb") as f:
        pickle.dump(payload, f)


def _run_one_trial_candidate(
    target_sim_cfg: Dict[str, Any],
    *,
    tune_dir_override: Optional[str],
    temp_output_dir: Optional[str],
    temp_output_stem: Optional[str],
    seed: Optional[int],
    trial_offset: Optional[int],
    verbose: bool,
) -> Path:
    tune_raw = tune_dir_override or target_sim_cfg.get("tune_dir")
    if not tune_raw:
        raise ValueError(
            "Could not infer tune_dir from target run. Pass --tune-dir when using --rerun."
        )

    tune_dir = Path(str(tune_raw)).expanduser().resolve()
    if not tune_dir.is_dir():
        raise FileNotFoundError(f"tune_dir not found: {tune_dir}")

    out_base = Path(temp_output_dir).expanduser().resolve() if temp_output_dir else (tune_dir / "output_data")
    out_base.mkdir(parents=True, exist_ok=True)

    if temp_output_stem:
        stem = str(temp_output_stem)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"__vm_candidate_{stamp}_{os.getpid()}"

    cmd = [
        sys.executable,
        str(ROOT / "run_pipeline.py"),
        "--tune-dir",
        str(tune_dir),
        "--mode",
        "multi",
        "--n-trials",
        "1",
        "--force-save",
        "--output-dir",
        str(out_base),
        "--output-stem",
        stem,
    ]
    if seed is not None:
        cmd.extend(["--seed", str(int(seed))])
    if trial_offset is not None:
        cmd.extend(["--trial-offset", str(int(trial_offset))])

    if verbose:
        print("[swap_vm_trace] rerun command:")
        print("  " + " ".join(cmd))
        subprocess.run(cmd, cwd=str(ROOT), check=True)
    else:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "One-trial rerun failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

    exact = out_base / stem
    if exact.is_dir():
        return exact

    cands = sorted(
        [p for p in out_base.glob(f"{stem}*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if cands:
        return cands[0]

    raise RuntimeError(
        f"Could not locate generated candidate run under {out_base} with stem prefix '{stem}'"
    )


def swap_vm_trace(
    *,
    target_run: Path,
    source_run: Optional[Path] = None,
    rerun: bool = False,
    update: str = "vm",
    source_trial_idx: int = 0,
    target_trial_idx: int = 0,
    source_input_trial_idx: Optional[int] = None,
    target_input_trial_idx: Optional[int] = None,
    write: bool = False,
    backup: bool = True,
    allow_time_mismatch: bool = False,
    tune_dir: Optional[str] = None,
    temp_output_dir: Optional[str] = None,
    temp_output_stem: Optional[str] = None,
    rerun_seed: Optional[int] = None,
    rerun_trial_offset: Optional[int] = None,
    keep_temp_run: bool = False,
    verbose_rerun: bool = False,
) -> VmSwapReport:
    report = VmSwapReport(
        target_run=target_run.expanduser().resolve(),
        source_desc=str(update or "vm"),
        dry_run=not bool(write),
        target_trial_idx=int(target_trial_idx),
        source_trial_idx=int(source_trial_idx),
        source_run=source_run.expanduser().resolve() if source_run is not None else None,
    )

    update = str(update or "vm").strip().lower()
    if update not in ("vm", "inputs", "both"):
        report.errors.append(f"update must be one of vm/inputs/both (got {update!r})")
        return report
    do_vm = update in ("vm", "both")
    do_inputs = update in ("inputs", "both")

    if rerun and source_run is not None:
        report.errors.append("Use either source_run or rerun mode, not both.")
        return report
    if (not rerun) and source_run is None:
        report.errors.append("Provide source_run or set rerun=True.")
        return report

    try:
        target_bundle = _load_trace_bundle(report.target_run, require_traces=do_vm)
    except Exception as exc:
        report.errors.append(f"Failed to load target run: {exc}")
        return report

    target_results_dir = target_bundle.get("results_dir")
    target_files = target_bundle.get("files", {}) or {}
    target_mode = str(target_bundle.get("mode", "single"))

    generated_run: Optional[Path] = None
    source_path: Path
    if rerun:
        try:
            generated_run = _run_one_trial_candidate(
                target_bundle.get("sim_cfg", {}) or {},
                tune_dir_override=tune_dir,
                temp_output_dir=temp_output_dir,
                temp_output_stem=temp_output_stem,
                seed=rerun_seed,
                trial_offset=rerun_trial_offset,
                verbose=verbose_rerun,
            )
            report.generated_source_run = generated_run
            source_path = generated_run
        except Exception as exc:
            report.errors.append(f"Failed to generate rerun candidate: {exc}")
            return report
    else:
        source_path = report.source_run  # type: ignore[assignment]

    try:
        source_bundle = _load_trace_bundle(source_path, require_traces=do_vm)
    except Exception as exc:
        report.errors.append(f"Failed to load source run: {exc}")
        return report

    source_mode = str(source_bundle.get("mode", "single"))

    write_targets: List[Tuple[Path, str]] = []
    unique_touched: List[Path] = []
    touched_seen: set = set()

    T_out = None
    V_out = None
    if do_vm:
        T_target = np.asarray(target_bundle.get("T", []), dtype=float).ravel()
        V_target_list = target_bundle.get("V_list", []) or []
        T_source = np.asarray(source_bundle.get("T", []), dtype=float).ravel()
        V_source_list = source_bundle.get("V_list", []) or []

        report.target_mode = target_mode
        report.source_mode = source_mode
        report.target_trace_count = len(V_target_list)
        report.source_trace_count = len(V_source_list)
        report.target_time_len = int(T_target.size)
        report.source_time_len = int(T_source.size)

        if report.target_trial_idx < 0 or report.target_trial_idx >= len(V_target_list):
            report.errors.append(
                f"target_trial_idx={report.target_trial_idx} out of range for target trace count {len(V_target_list)}"
            )
            return report

        try:
            V_source = _pick_trace(V_source_list, report.source_trial_idx, label="source")
        except Exception as exc:
            report.errors.append(str(exc))
            return report

        if T_target.size != T_source.size or not np.allclose(T_target, T_source):
            if allow_time_mismatch:
                report.warnings.append(
                    "Time axis mismatch detected; replacing target trace time axis with source time axis."
                )
            else:
                report.errors.append(
                    "Time axis mismatch between target and source traces. Use --allow-time-mismatch to override."
                )
                return report

        V_out = [np.asarray(v, dtype=float).copy() for v in V_target_list]
        V_out[report.target_trial_idx] = np.asarray(V_source, dtype=float).copy()
        T_out = np.asarray(T_source if allow_time_mismatch else T_target, dtype=float).copy()

        trace_sidecar_rel = target_files.get("traces")
        results_npz_rel = target_files.get("results_npz")
        results_pkl_rel = target_files.get("results_pkl")

        trace_sidecar = (
            (target_results_dir / trace_sidecar_rel).resolve()
            if isinstance(trace_sidecar_rel, str)
            else None
        )
        results_npz = (
            (target_results_dir / results_npz_rel).resolve()
            if isinstance(results_npz_rel, str)
            else None
        )
        results_pkl = (
            (target_results_dir / results_pkl_rel).resolve()
            if isinstance(results_pkl_rel, str)
            else None
        )

        if trace_sidecar is not None:
            if trace_sidecar.is_file():
                write_targets.append((trace_sidecar, "traces_sidecar"))
            else:
                report.warnings.append(f"Manifest traces sidecar missing: {trace_sidecar}")

        if results_npz is not None:
            if results_npz.is_file():
                write_targets.append((results_npz, "results_npz"))
            else:
                report.warnings.append(f"Manifest results_npz missing: {results_npz}")

        if results_pkl is not None:
            if results_pkl.is_file():
                write_targets.append((results_pkl, "results_pkl_vm"))
            else:
                report.warnings.append(f"Manifest results_pkl missing: {results_pkl}")

    inputs_payload_out = None
    if do_inputs:
        source_input_idx = (
            int(source_input_trial_idx)
            if source_input_trial_idx is not None
            else int(report.source_trial_idx)
        )
        target_input_idx = (
            int(target_input_trial_idx)
            if target_input_trial_idx is not None
            else int(report.target_trial_idx)
        )
        report.source_input_trial_idx = int(source_input_idx)
        report.target_input_trial_idx = int(target_input_idx)

        try:
            target_inputs_state = _load_inputs_payload(target_bundle)
            source_inputs_state = _load_inputs_payload(source_bundle)
        except Exception as exc:
            report.errors.append(f"Failed to load inputs payload: {exc}")
            return report

        target_inputs_payload = target_inputs_state.get("payload")
        source_inputs_payload = source_inputs_state.get("payload")
        report.target_input_count = _count_inputs_payload(target_inputs_payload)
        report.source_input_count = _count_inputs_payload(source_inputs_payload)

        if source_inputs_payload is None:
            report.errors.append("Source run has no saved inputs payload (inputs_sample/results_pkl).")
            return report

        try:
            source_inputs = _pick_inputs_for_trial(
                source_inputs_payload,
                source_input_idx,
                label="source",
            )
        except Exception as exc:
            report.errors.append(str(exc))
            return report

        try:
            inputs_payload_out = _set_inputs_for_trial(
                target_inputs_payload,
                mode=target_mode,
                idx=target_input_idx,
                new_inputs=source_inputs,
            )
        except Exception as exc:
            report.errors.append(str(exc))
            return report

        inputs_sidecar = target_inputs_state.get("inputs_sample_path")
        target_results_pkl = target_inputs_state.get("results_pkl_path")
        if isinstance(inputs_sidecar, Path) and inputs_sidecar.is_file():
            write_targets.append((inputs_sidecar, "inputs_sidecar"))
        if isinstance(target_results_pkl, Path) and target_results_pkl.is_file():
            write_targets.append((target_results_pkl, "results_pkl_inputs"))

        if not (
            isinstance(inputs_sidecar, Path)
            and inputs_sidecar.is_file()
        ) and not (
            isinstance(target_results_pkl, Path)
            and target_results_pkl.is_file()
        ):
            report.errors.append(
                "No writable inputs artifact found in target run (inputs_sample sidecar or results_pkl)."
            )
            return report

    if do_vm and not write_targets:
        report.errors.append(
            "No writable trace artifact found in manifest (expected traces sidecar and/or results file)."
        )
        return report

    for path, _kind in write_targets:
        key = str(path)
        if key not in touched_seen:
            touched_seen.add(key)
            unique_touched.append(path)
    report.touched_files = unique_touched

    if report.dry_run:
        if generated_run is not None and not keep_temp_run:
            report.warnings.append(
                f"Dry-run kept generated candidate run at {generated_run} (rerun not auto-deleted in dry-run)."
            )
        return report

    try:
        backed_up: set = set()
        for path, _kind in write_targets:
            if backup:
                key = str(path)
                if key not in backed_up:
                    backup_path = _backup_file(path)
                    report.backup_files.append(backup_path)
                    backed_up.add(key)

        for path, kind in write_targets:
            if kind == "traces_sidecar" and do_vm and T_out is not None and V_out is not None:
                _write_traces_sidecar(path, target_mode, T_out, V_out)
            elif kind == "results_npz" and do_vm and T_out is not None and V_out is not None:
                _write_results_npz(path, target_mode, T_out, V_out)
            elif kind == "results_pkl_vm" and do_vm and T_out is not None and V_out is not None:
                _write_results_pkl(path, target_mode, T_out, V_out)
            elif kind == "inputs_sidecar" and do_inputs and inputs_payload_out is not None:
                _write_inputs_sidecar(path, inputs_payload_out)
            elif kind == "results_pkl_inputs" and do_inputs and inputs_payload_out is not None:
                _write_results_pkl_inputs(path, inputs_payload_out)

        if generated_run is not None and not keep_temp_run:
            shutil.rmtree(generated_run, ignore_errors=True)
            report.warnings.append(f"Deleted temporary rerun candidate: {generated_run}")

    except Exception as exc:
        report.errors.append(f"Write failed: {exc}")

    return report


def print_report(report: VmSwapReport) -> None:
    mode_text = "DRY-RUN" if report.dry_run else "WRITE"
    print(f"[swap_vm_trace] Mode: {mode_text}")
    print(f"[swap_vm_trace] update: {report.source_desc}")
    print(f"[swap_vm_trace] target_run: {report.target_run}")
    if report.source_run is not None:
        print(f"[swap_vm_trace] source_run: {report.source_run}")
    if report.generated_source_run is not None:
        print(f"[swap_vm_trace] generated_source_run: {report.generated_source_run}")

    if report.target_mode:
        print(
            "[swap_vm_trace] target traces: "
            f"mode={report.target_mode}, n={report.target_trace_count}, len(T)={report.target_time_len}, "
            f"target_idx={report.target_trial_idx}"
        )
    if report.source_mode:
        print(
            "[swap_vm_trace] source traces: "
            f"mode={report.source_mode}, n={report.source_trace_count}, len(T)={report.source_time_len}, "
            f"source_idx={report.source_trial_idx}"
        )
    if report.source_input_trial_idx is not None or report.target_input_trial_idx is not None:
        print(
            "[swap_vm_trace] inputs: "
            f"target_saved={report.target_input_count}, source_saved={report.source_input_count}, "
            f"target_idx={report.target_input_trial_idx}, source_idx={report.source_input_trial_idx}"
        )

    if report.touched_files:
        print("[swap_vm_trace] Files to update:")
        for path in report.touched_files:
            print(f"  - {path}")

    if report.backup_files:
        print("[swap_vm_trace] Backup files:")
        for path in report.backup_files:
            print(f"  - {path}")

    for warning in report.warnings:
        print(f"[swap_vm_trace] Warning: {warning}")

    if report.errors:
        for err in report.errors:
            print(f"[swap_vm_trace] ERROR: {err}")
    else:
        if report.dry_run:
            print("[swap_vm_trace] Dry-run complete. Re-run with --write to apply.")
        else:
            print("[swap_vm_trace] Exemplar data swap complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Swap saved exemplar Vm and/or input payload data in an existing run output. "
            "Default is dry-run; add --write to apply."
        )
    )
    parser.add_argument(
        "--target-run",
        required=True,
        help="Target run path (run dir, results dir, or run_manifest.json).",
    )
    parser.add_argument(
        "--update",
        choices=["vm", "inputs", "both"],
        default="vm",
        help="Which exemplar data to replace: vm trace, inputs payload, or both (default: vm).",
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source-run",
        default=None,
        help="Source run path used to copy Vm trace from.",
    )
    source_group.add_argument(
        "--rerun",
        action="store_true",
        help=(
            "Generate a fresh 1-trial candidate run from the current tune_dir "
            "and use its Vm trace as source."
        ),
    )

    parser.add_argument(
        "--source-trial-idx",
        type=int,
        default=0,
        help="Source trace index in source run for Vm replacement (default: 0).",
    )
    parser.add_argument(
        "--target-trial-idx",
        type=int,
        default=0,
        help="Target trace index to replace for Vm replacement (default: 0).",
    )
    parser.add_argument(
        "--source-input-trial-idx",
        type=int,
        default=None,
        help="Source inputs trial index for input-raster payload replacement (default: source-trial-idx).",
    )
    parser.add_argument(
        "--target-input-trial-idx",
        type=int,
        default=None,
        help="Target inputs trial index for input-raster payload replacement (default: target-trial-idx).",
    )
    parser.add_argument(
        "--allow-time-mismatch",
        action="store_true",
        help="Allow source/target time-axis mismatch and replace target T with source T.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Apply file updates. If omitted, dry-run only.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable backup creation before writing.",
    )

    parser.add_argument(
        "--tune-dir",
        default=None,
        help="Override tune directory for --rerun (default: inferred from target sim_cfg.tune_dir).",
    )
    parser.add_argument(
        "--temp-output-dir",
        default=None,
        help="Output directory for the temporary rerun candidate (default: <tune_dir>/output_data).",
    )
    parser.add_argument(
        "--temp-output-stem",
        default=None,
        help="Output stem for the temporary rerun candidate (default: auto-generated unique stem).",
    )
    parser.add_argument(
        "--rerun-seed",
        type=int,
        default=None,
        help="Optional seed override passed to run_pipeline.py during --rerun.",
    )
    parser.add_argument(
        "--rerun-trial-offset",
        type=int,
        default=None,
        help="Optional trial offset passed to run_pipeline.py during --rerun.",
    )
    parser.add_argument(
        "--keep-temp-run",
        action="store_true",
        help="Keep generated candidate run folder after write when using --rerun.",
    )
    parser.add_argument(
        "--verbose-rerun",
        action="store_true",
        help="Stream run_pipeline output while generating rerun candidate.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    report = swap_vm_trace(
        target_run=Path(args.target_run),
        source_run=Path(args.source_run) if args.source_run else None,
        rerun=bool(args.rerun),
        update=args.update,
        source_trial_idx=int(args.source_trial_idx),
        target_trial_idx=int(args.target_trial_idx),
        source_input_trial_idx=args.source_input_trial_idx,
        target_input_trial_idx=args.target_input_trial_idx,
        write=bool(args.write),
        backup=not bool(args.no_backup),
        allow_time_mismatch=bool(args.allow_time_mismatch),
        tune_dir=args.tune_dir,
        temp_output_dir=args.temp_output_dir,
        temp_output_stem=args.temp_output_stem,
        rerun_seed=args.rerun_seed,
        rerun_trial_offset=args.rerun_trial_offset,
        keep_temp_run=bool(args.keep_temp_run),
        verbose_rerun=bool(args.verbose_rerun),
    )
    print_report(report)
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
