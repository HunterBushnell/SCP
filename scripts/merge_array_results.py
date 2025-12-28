#!/usr/bin/env python3
"""
Merge SLURM array outputs into a single multi-trial results file.

Default behavior:
  - Finds output files matching *slurm_<jobid>_* in the input directory.
  - Loads each result file (single or multi).
  - Merges spikes/traces/input summaries/stats into a single multi result.
  - Saves merged results with output_stem "slurm_merged_<jobid>".
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules_local import run_sim


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge SLURM array results into a single multi-trial file.")
    p.add_argument("--tune-dir", type=str, default=None,
                   help="Tune directory (used to infer output_data).")
    p.add_argument("--input-dir", type=str, default=None,
                   help="Directory containing per-task results (overrides --tune-dir for input).")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory for merged output (overrides --tune-dir for output).")
    p.add_argument("--job-id", type=str, default=None,
                   help="SLURM job id used in default filename pattern.")
    p.add_argument("--pattern", type=str, default=None,
                   help="Glob pattern for results (overrides --job-id).")
    p.add_argument("--output-stem", type=str, default=None,
                   help="Output stem for merged results (default: slurm_merged_<jobid>).")
    return p.parse_args()


def _find_results(output_dir: Path, pattern: Optional[str], job_id: Optional[str]) -> List[Path]:
    paths: List[Path] = []
    if pattern:
        for p in output_dir.glob(pattern):
            if p.is_dir():
                manifest = p / "run_manifest.json"
                if manifest.is_file():
                    paths.append(p)
                else:
                    # fallback to single results file in dir
                    candidates = list(p.glob("*.pkl")) + list(p.glob("*.npz"))
                    if len(candidates) == 1:
                        paths.append(candidates[0])
            else:
                paths.append(p)
    else:
        if not job_id:
            raise ValueError("Provide either --pattern or --job-id.")
        run_dirs = list(output_dir.glob(f"slurm_{job_id}_*"))
        for d in run_dirs:
            if not d.is_dir():
                continue
            manifest = d / "run_manifest.json"
            if manifest.is_file():
                paths.append(d)
            else:
                candidates = list(d.glob("*.pkl")) + list(d.glob("*.npz"))
                if len(candidates) == 1:
                    paths.append(candidates[0])

    def _sort_key(p: Path):
        name = p.name if p.is_dir() else p.stem
        m = re.search(r"slurm_(\d+)_(\d+)", name)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (0, name)

    return sorted(paths, key=_sort_key)


def _ensure_multi(res: Dict[str, Any]) -> Dict[str, Any]:
    mode = res.get("mode", "single")
    if mode == "multi":
        return res
    if mode != "single":
        raise ValueError(f"Unsupported results mode: {mode}")

    sim_cfg = copy.deepcopy(res.get("sim_cfg", {}))
    traces = res.get("traces", {}) or {}
    inputs = res.get("inputs", None)

    multi = {
        "mode": "multi",
        "sim_cfg": sim_cfg,
        "spikes": [np.asarray(res.get("spikes", []), dtype=float)],
        "traces": {},
        "inputs_by_trial": None,
        "meta": copy.deepcopy(res.get("meta", {})),
    }

    if "T" in traces and "V" in traces:
        multi["traces"] = {"T": traces["T"], "V": [traces["V"]]}

    if inputs:
        multi["inputs_by_trial"] = [{"trial_idx": 0, "inputs": inputs}]

    meta = multi.setdefault("meta", {})
    meta["n_trials"] = 1
    meta["trial_ids"] = [0]
    return multi


def _append_traces(dst: Dict[str, Any], src: Dict[str, Any], max_traces: int) -> bool:
    if max_traces <= 0:
        dst["traces"] = {}
        return False
    if not dst.get("traces") or not src.get("traces"):
        return False
    if "T" not in dst["traces"] or "T" not in src["traces"]:
        return False
    if not np.allclose(np.asarray(dst["traces"]["T"]), np.asarray(src["traces"]["T"])):
        return False
    dst_v = list(dst["traces"].get("V", []))
    for v in src["traces"].get("V", []) or []:
        if len(dst_v) >= max_traces:
            break
        dst_v.append(v)
    dst["traces"]["V"] = dst_v
    return True


def _append_inputs(dst: Dict[str, Any], src: Dict[str, Any], max_traces: int) -> None:
    if max_traces <= 0:
        dst["inputs_by_trial"] = None
        return
    src_inputs = src.get("inputs_by_trial")
    if not src_inputs:
        return
    dst_inputs = dst.get("inputs_by_trial") or []
    for entry in src_inputs:
        if len(dst_inputs) >= max_traces:
            break
        dst_inputs.append(entry)
    dst["inputs_by_trial"] = dst_inputs


def _merge_input_stats(dst_meta: Dict[str, Any], src_meta: Dict[str, Any]) -> None:
    dst_stats = dst_meta.get("input_stats")
    src_stats = src_meta.get("input_stats")
    if not src_stats and not dst_stats:
        return
    if dst_stats is None:
        dst_meta["input_stats"] = copy.deepcopy(src_stats)
        return
    if src_stats is None:
        return
    dst_stats["trials"].extend(src_stats.get("trials", []))
    dst_stats["group_means"] = run_sim._aggregate_input_stats(dst_stats["trials"])

def _stats_compatible(template: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    for key in ("bin_ms", "tstart_ms", "tstop_ms"):
        if template.get(key) != candidate.get(key):
            return False
    try:
        return np.allclose(
            np.asarray(template.get("t_ms", []), dtype=float),
            np.asarray(candidate.get("t_ms", []), dtype=float),
        )
    except Exception:
        return False


def _smooth_rate_curve(
    centers: np.ndarray,
    rates: np.ndarray,
    bin_ms: float,
    smooth_ms: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    if smooth_ms is None:
        return centers, rates
    try:
        smooth_ms = float(smooth_ms)
    except Exception:
        return centers, rates
    if smooth_ms <= 0 or bin_ms <= 0:
        return centers, rates

    k = int(round(smooth_ms / bin_ms))
    if k <= 1 or rates.size < k:
        return centers, rates
    if k % 2 == 0:
        k += 1

    kernel = np.ones(k, dtype=float) / float(k)
    y = np.convolve(rates, kernel, mode="valid")
    drop = (len(centers) - len(y)) // 2
    if drop < 0:
        return centers, rates
    return centers[drop : drop + len(y)], y


def _recompute_avg_rate(sim_cfg: Dict[str, Any], spikes_by_trial: List[np.ndarray]) -> Dict[str, Any]:
    tstop = float(sim_cfg.get("tstop", 0.0))
    bin_width = float(sim_cfg.get("bins", 25.0))
    if bin_width <= 0:
        bin_width = 25.0
    bins = np.arange(0, tstop + bin_width, bin_width)
    centers = bins[:-1] + 0.5 * bin_width
    bw_s = bin_width / 1000.0
    if spikes_by_trial:
        per_trial_rates = []
        for tr in spikes_by_trial:
            tr = np.asarray(tr)
            counts, _ = np.histogram(tr, bins=bins)
            per_trial_rates.append(counts / bw_s)
        mean_rate = np.mean(per_trial_rates, axis=0)
    else:
        mean_rate = np.array([], dtype=float)
    smooth_ms = sim_cfg.get("avg_rate_curve_smooth_ms", 25.0)
    centers, mean_rate = _smooth_rate_curve(centers, mean_rate, bin_width, smooth_ms)
    try:
        smooth_ms_val = float(smooth_ms) if smooth_ms is not None else 0.0
    except Exception:
        smooth_ms_val = 0.0
    return {
        "bin_ms": bin_width,
        "smooth_ms": smooth_ms_val,
        "t_ms": centers.tolist(),
        "rate_hz": mean_rate.tolist(),
    }


def main() -> None:
    args = parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    elif args.tune_dir:
        output_dir = Path(args.tune_dir).resolve() / "output_data"
    else:
        raise ValueError("Provide either --tune-dir or --output-dir.")

    input_dir = output_dir
    if args.input_dir:
        input_dir = Path(args.input_dir).resolve()

    paths = _find_results(input_dir, args.pattern, args.job_id)
    if not paths:
        raise FileNotFoundError(f"No results found in {output_dir}")

    results_list = [_ensure_multi(run_sim.load_results(p)) for p in paths]

    base = results_list[0]
    sim_cfg = copy.deepcopy(base.get("sim_cfg", {}))
    max_traces = int(sim_cfg.get("n_traces_to_save", 1))
    total_trials = sum(len(res.get("spikes", []) or []) for res in results_list)
    max_inputs = run_sim._resolve_inputs_to_save(sim_cfg, total_trials, max_traces)

    base_meta = copy.deepcopy(base.get("meta", {}) or {})
    for k in ("n_trials", "trial_ids", "avg_rate_curve", "input_summaries", "input_stats"):
        base_meta.pop(k, None)

    traces = copy.deepcopy(base.get("traces", {}) or {})
    if "V" in traces:
        traces["V"] = []

    merged = {
        "mode": "multi",
        "sim_cfg": sim_cfg,
        "spikes": [],
        "traces": traces,
        "inputs_by_trial": [] if max_inputs > 0 else None,
        "meta": base_meta,
    }

    warnings = merged["meta"].setdefault("merge_warnings", [])
    merged_spikes: List[np.ndarray] = []
    input_summaries: List[Dict[str, Any]] = []
    input_stats_trials: List[Dict[str, Any]] = []
    input_stats_template: Optional[Dict[str, Any]] = None
    trial_offset = 0

    for res_idx, res in enumerate(results_list):
        res_cfg = res.get("sim_cfg", {}) or {}
        for key in ("tstart", "tstop", "dt", "bins"):
            if key in res_cfg and res_cfg.get(key) != sim_cfg.get(key):
                warnings.append(f"sim_cfg_mismatch:{key}:idx{res_idx}")
                break

        res_spikes = list(res.get("spikes", []) or [])
        n_res_trials = len(res_spikes)
        merged_spikes.extend(res_spikes)

        appended = _append_traces(merged, res, max_traces)
        if not appended and max_traces > 0:
            try:
                if res.get("traces") and merged.get("traces"):
                    if "T" in res["traces"] and "T" in merged["traces"]:
                        if not np.allclose(
                            np.asarray(merged["traces"]["T"]),
                            np.asarray(res["traces"]["T"]),
                        ):
                            warnings.append(f"trace_time_mismatch:idx{res_idx}")
            except Exception:
                warnings.append(f"trace_merge_failed:idx{res_idx}")

        # inputs_by_trial (renumber trial_idx)
        if max_inputs > 0 and merged.get("inputs_by_trial") is not None:
            res_inputs = res.get("inputs_by_trial") or []
            for entry in res_inputs:
                if len(merged["inputs_by_trial"]) >= max_inputs:
                    break
                new_entry = copy.deepcopy(entry)
                local_idx = new_entry.get("trial_idx", 0)
                try:
                    local_idx = int(local_idx)
                except Exception:
                    local_idx = 0
                new_entry["trial_idx"] = trial_offset + local_idx
                merged["inputs_by_trial"].append(new_entry)

        src_meta = res.get("meta", {}) or {}
        for entry in src_meta.get("input_summaries", []) or []:
            new_entry = copy.deepcopy(entry)
            if "trial_idx" in new_entry:
                try:
                    new_entry["trial_idx"] = trial_offset + int(new_entry["trial_idx"])
                except Exception:
                    pass
            input_summaries.append(new_entry)

        src_stats = src_meta.get("input_stats")
        if src_stats:
            if input_stats_template is None:
                input_stats_template = {
                    "bin_ms": src_stats.get("bin_ms"),
                    "t_ms": src_stats.get("t_ms"),
                    "tstart_ms": src_stats.get("tstart_ms"),
                    "tstop_ms": src_stats.get("tstop_ms"),
                }
            elif not _stats_compatible(input_stats_template, src_stats):
                warnings.append(f"input_stats_mismatch:idx{res_idx}")
                src_stats = None
            if src_stats:
                for entry in src_stats.get("trials", []) or []:
                    new_entry = copy.deepcopy(entry)
                    if "trial_idx" in new_entry:
                        try:
                            new_entry["trial_idx"] = trial_offset + int(new_entry["trial_idx"])
                        except Exception:
                            pass
                    input_stats_trials.append(new_entry)

        trial_offset += n_res_trials

    merged["spikes"] = merged_spikes
    n_trials = len(merged_spikes)
    merged.setdefault("meta", {})["n_trials"] = n_trials
    merged["meta"]["trial_ids"] = list(range(n_trials))
    merged.setdefault("sim_cfg", {})["n_trials"] = n_trials
    merged["meta"]["avg_rate_curve"] = _recompute_avg_rate(sim_cfg, merged_spikes)

    if input_summaries:
        merged["meta"]["input_summaries"] = input_summaries

    if input_stats_template is not None:
        input_stats = dict(input_stats_template)
        input_stats["trials"] = input_stats_trials
        input_stats["group_means"] = run_sim._aggregate_input_stats(input_stats_trials)
        merged["meta"]["input_stats"] = input_stats

    output_stem = args.output_stem
    if not output_stem:
        output_stem = f"slurm_merged_{args.job_id or 'merged'}"
    merged["sim_cfg"]["output"] = output_stem

    saved = run_sim.save_results(merged, base_dir=output_dir)
    print(f"Merged {len(paths)} result(s) into {saved}")


if __name__ == "__main__":
    main()
