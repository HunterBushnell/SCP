#!/usr/bin/env python
"""
CLI entrypoint to run the PV-SST single-cell pipeline without notebooks.

Minimal usage (from repo root or tune dir):
    python -m run_pipeline --tune-dir cells/SST/tunes/seg_tuned --mode multi --n-trials 10

Defaults:
    - Uses sim_config.json and syn_config.json in --tune-dir (or --tune-dir/cell_configs).
    - If sim_cfg['output'] is empty, assigns a timestamped stem so results are always saved.
    - Saves to --output-dir (default: output_data inside the tune dir).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure local imports work regardless of cwd
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from modules_local import load_cell, geometry, inputs, run_sim, randomness  # noqa: E402


def _timestamp_stem() -> str:
    return datetime.now().strftime("slurm_%Y%m%d_%H%M%S")


def _load_mechanisms(tune_dir: Path) -> None:
    """
    Attempt to load compiled NEURON mechanisms from the tune directory.

    Expects nrnivmodl output at:
      tune_dir/modfiles/x86_64/.libs/libnrnmech.so (or modfiles/x86_64/libnrnmech.so)
    If missing, instruct the user to run `nrnivmodl` in tune_dir/modfiles.
    """
    from neuron import h  # local import so we only pull NEURON if needed

    candidates = [
        tune_dir / "modfiles" / "x86_64" / ".libs" / "libnrnmech.so",
        tune_dir / "modfiles" / "x86_64" / "libnrnmech.so",
    ]
    for dll in candidates:
        if dll.is_file():
            try:
                h.nrn_load_dll(str(dll))
                print(f"Loaded mechanisms from {dll}")
                return
            except Exception as exc:
                # If mechanisms are already present, proceed.
                # Some NEURON builds throw hocobj_call error while printing
                # "user defined name already exists" to stderr.
                already_loaded = False
                for mech in ("AMPA_NMDA_STP", "GABA_A", "GABA_A_STP", "vecstim", "Ih"):
                    if hasattr(h, mech):
                        already_loaded = True
                        break
                if already_loaded:
                    print(f"Mechanisms already loaded (skipping reload of {dll})")
                    return
                raise RuntimeError(f"Found compiled mechanisms at {dll} but failed to load: {exc}") from exc

    raise FileNotFoundError(
        "Compiled mechanisms not found. Run `nrnivmodl` inside the tune directory "
        f"{tune_dir}/modfiles to build modfiles/x86_64/.libs/libnrnmech.so, then rerun."
    )


def _parse_enabled_path(raw: object) -> tuple[bool, Optional[str]]:
    if isinstance(raw, (list, tuple)):
        enabled = bool(raw[0]) if len(raw) >= 1 else False
        path = raw[1] if len(raw) >= 2 else None
        return enabled, path
    if isinstance(raw, dict):
        return bool(raw.get("enabled", False)), raw.get("path")
    if raw in (None, "", False):
        return False, None
    return True, str(raw)


def _finalize_sim_cfg(sim_cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.trial_offset is not None:
        sim_cfg["trial_offset"] = int(args.trial_offset)

    # CLI seed overrides both sim_cfg['seed'] and randomness.global.seed
    if args.seed is not None:
        sim_cfg["seed"] = int(args.seed)
        rand_cfg = sim_cfg.get("randomness", {})
        if not isinstance(rand_cfg, dict):
            rand_cfg = {}
        global_cfg = rand_cfg.get("global")
        if not isinstance(global_cfg, dict):
            global_cfg = {}
        global_cfg["seed"] = int(args.seed)
        rand_cfg["global"] = global_cfg
        sim_cfg["randomness"] = rand_cfg

    # Ensure output saving is enabled (if configured)
    save_output = sim_cfg.get("save_output", True)
    if save_output is None:
        save_output = True
    save_output = bool(save_output)
    if getattr(args, "force_save", False):
        save_output = True
    sim_cfg["save_output"] = save_output

    output_stem = sim_cfg.get("output_stem")
    if output_stem not in (None, ""):
        sim_cfg["output"] = output_stem

    if save_output:
        if not sim_cfg.get("output"):
            sim_cfg["output"] = args.output_stem or _timestamp_stem()
        elif args.output_stem:
            sim_cfg["output"] = args.output_stem
    else:
        if args.output_stem and not sim_cfg.get("output"):
            sim_cfg["output"] = args.output_stem

    return sim_cfg


def _resolve_append_target(sim_cfg_raw: Dict[str, Any], output_base: Path) -> Optional[Path]:
    append_raw = sim_cfg_raw.get("append") if "append" in sim_cfg_raw else sim_cfg_raw.get("append_to")
    enabled, path = _parse_enabled_path(append_raw)
    if not enabled or path in (None, "", False):
        return None
    append_path = Path(str(path))
    if not append_path.is_absolute():
        if append_path.parts and append_path.parts[0] == output_base.name:
            append_path = output_base / Path(*append_path.parts[1:])
        else:
            append_path = output_base / append_path
    return append_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PV-SST pipeline (single or multi) for a tune directory.")
    p.add_argument("--tune-dir", type=str, default=None,
                   help="Path to tune directory containing sim_config.json and syn_config.json "
                        "(optionally under cell_configs/).")
    p.add_argument("--cell", type=str, default=None,
                   help="Cell name (e.g., SST, PV). Used if --tune-dir is not provided.")
    p.add_argument("--tune", type=str, default=None,
                   help="Tune name (e.g., seg_tuned). Used if --tune-dir is not provided.")
    p.add_argument("--mode", choices=["single", "multi"], default=None,
                   help="Run a single trial or multi-trial batch. If omitted, chosen from n_trials.")
    p.add_argument("--n-trials", type=int, default=None,
                   help="Override sim_config['n_trials'] for multi mode.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override sim_config['seed'] / randomness base seed.")
    p.add_argument("--trial-offset", type=int, default=None,
                   help="Global trial index offset (used to align array tasks with sequential runs).")
    p.add_argument("--iclamp", "--current-injection", action="store_true", dest="iclamp",
                   help="Run a simple somatic current injection test instead of synapse-driven sims.")
    p.add_argument("--snapshot", action="store_true",
                   help="Enable full snapshot capture (forces saving and full sidecars).")
    p.add_argument("--force-save", action="store_true",
                   help="Force saving even if sim_config disables output.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory for outputs (default: output_data under tune-dir).")
    p.add_argument("--output-stem", type=str, default=None,
                   help="Override sim_config['output'] stem; default assigns a timestamp if empty.")
    return p.parse_args()


def main() -> None:
    job_start = time.perf_counter()
    args = parse_args()
    if args.tune_dir:
        tune_dir = Path(args.tune_dir).resolve()
    else:
        if not (args.cell and args.tune):
            raise ValueError("Provide either --tune-dir or both --cell and --tune.")
        tune_dir = (HERE / "cells" / args.cell / "tunes" / args.tune).resolve()

    # Paths
    config_root = inputs._resolve_config_root(tune_dir)
    sim_path = config_root / "sim_config.json"
    syn_path = config_root / "syn_config.json"
    manifest_path = tune_dir / "manifest.json"

    if not sim_path.is_file():
        raise FileNotFoundError(f"Missing sim_config.json in {config_root}")
    if not syn_path.is_file():
        raise FileNotFoundError(f"Missing syn_config.json in {config_root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest.json in {tune_dir}")

    # Load compiled mechanisms (nrnivmodl output) before building the cell
    _load_mechanisms(tune_dir)

    # ---------------------- Load cell (2.1) ----------------------
    with sim_path.open("r") as f:
        sim_cfg_preview = json.load(f)

    append_target = _resolve_append_target(sim_cfg_preview, tune_dir / "output_data")
    sim_cfg_override = None
    if append_target is not None:
        if append_target.name == "run_manifest.json":
            append_target = append_target.parent
        if append_target.exists():
            base_results = run_sim.load_results(append_target)
            base_sim_cfg = base_results.get("sim_cfg", {}) or {}
            base_cell = base_sim_cfg.get("cell")
            base_tune = base_sim_cfg.get("tune")
            if base_cell and base_cell != tune_dir.parent.name:
                raise ValueError(
                    f"append_to points to cell {base_cell!r} but tune_dir is {tune_dir.parent.name!r}"
                )
            if base_tune and base_tune != tune_dir.name:
                raise ValueError(
                    f"append_to points to tune {base_tune!r} but tune_dir is {tune_dir.name!r}"
                )
            sim_cfg_override = copy.deepcopy(base_sim_cfg)
            sim_cfg_override["append"] = (
                sim_cfg_preview.get("append")
                if "append" in sim_cfg_preview
                else sim_cfg_preview.get("append_to")
            )
        else:
            print(f"append_to target not found yet: {append_target} (using local sim_config.json)")

    sim_cfg_for_cell = sim_cfg_override or sim_cfg_preview
    snapshot_raw = sim_cfg_for_cell.get("snapshot", None)
    snapshot_enabled = False
    if isinstance(snapshot_raw, dict):
        snapshot_enabled = bool(snapshot_raw.get("enabled", False))
    elif snapshot_raw is True:
        snapshot_enabled = True
    if args.snapshot:
        if not isinstance(snapshot_raw, dict):
            snapshot_raw = {}
        snapshot_raw["enabled"] = True
        sim_cfg_for_cell["snapshot"] = snapshot_raw
        snapshot_enabled = True
        if sim_cfg_override is None:
            sim_cfg_override = sim_cfg_for_cell
    # Align with notebook: load cell_configs/cell_config.json when present
    cell_config_path = tune_dir / "cell_configs" / "cell_config.json"
    if cell_config_path.is_file():
        try:
            cell_cfg = json.loads(cell_config_path.read_text())
        except Exception:
            cell_cfg = {}
    else:
        cell_cfg = {}

    cell_cfg.setdefault("cell_name", tune_dir.parent.name)  # e.g., SST or PV
    paths = cell_cfg.setdefault("paths", {})
    paths.setdefault("manifest", "manifest.json")

    tuning = cell_cfg.setdefault("tuning", {})
    if "soma_diam_multiplier" not in tuning:
        tuning["soma_diam_multiplier"] = sim_cfg_for_cell.get("soma_diam_multiplier", 1.0)

    if sim_cfg_for_cell.get("soma_diam_multiplier") is not None:
        tuning["soma_diam_multiplier"] = sim_cfg_for_cell.get("soma_diam_multiplier", 1.0)
    if sim_cfg_for_cell.get("specimen_id") is not None:
        cell_cfg["specimen_id"] = sim_cfg_for_cell.get("specimen_id")
    if sim_cfg_for_cell.get("model_type") is not None:
        cell_cfg["model_type"] = sim_cfg_for_cell.get("model_type")
    cell = load_cell(cell_cfg)
    # Geometry: use tune-specific geometry config when present (notebook parity)
    geom_config_path = tune_dir / "cell_configs" / "geometry.json"
    if geom_config_path.is_file():
        try:
            geom_config = json.loads(geom_config_path.read_text())
        except Exception:
            geom_config = {}
    else:
        geom_config = {}
    geom = geometry.define_geometry(cell, geom_config)

    # ---------------------- Optional IClamp test -----------------
    iclamp_cfg_raw = (sim_cfg_override or sim_cfg_preview).get("iclamp", {})
    iclamp_enabled = bool(iclamp_cfg_raw.get("enabled", False))
    if snapshot_enabled and (args.iclamp or iclamp_enabled):
        print("Snapshot mode ignored because IClamp is enabled.")
        snapshot_enabled = False
        if isinstance(sim_cfg_for_cell.get("snapshot"), dict):
            sim_cfg_for_cell["snapshot"]["enabled"] = False
    if args.iclamp or iclamp_enabled:
        sim_cfg_raw = sim_cfg_override or sim_cfg_preview
        sim_cfg = inputs._normalize_sim_config(sim_cfg_raw)
        inputs._inject_path_metadata(sim_cfg, config_root)
        if args.n_trials is not None:
            sim_cfg["n_trials"] = int(args.n_trials)
        sim_cfg = _finalize_sim_cfg(sim_cfg, args)

        result = run_sim.run_iclamp_test(cell, sim_cfg, iclamp_cfg=iclamp_cfg_raw)
        out_dir = Path(args.output_dir) if args.output_dir else tune_dir / "output_data"
        saved_path = run_sim.save_results(result, base_dir=out_dir)
        if saved_path is None:
            print("IClamp results not saved (sim_cfg['output'] was empty).")
        else:
            print(f"IClamp results saved to {saved_path}")
        total_elapsed = time.perf_counter() - job_start
        print(f"Total runtime: {total_elapsed:.2f}s")
        return

    # ---------------------- Load inputs (2.3) --------------------
    # seed_override applies to sim_cfg["seed"]
    trial_rng = None
    if args.mode == "single":
        # single trial uses a concrete trial RNG so inputs and synapses differ per run
        # (run_single will build its own trial RNG from the same manager)
        pass

    sim_cfg, groups_cfg, inputs_by_group = inputs.generate_inputs(
        path=tune_dir,
        geometry=geom,
        seed_override=args.seed,
        trial_rng=trial_rng,
        sim_cfg_override=sim_cfg_override,
    )

    sim_cfg = _finalize_sim_cfg(sim_cfg, args)

    # ---------------------- Randomness manager -------------------
    rm = randomness.RandomnessManager(sim_cfg)

    # ---------------------- Choose mode ---------------------------
    mode = args.mode
    # If not provided, decide based on n_trials (after any override)
    if args.n_trials is not None:
        sim_cfg["n_trials"] = int(args.n_trials)
    if mode is None:
        n_trials_eff = int(sim_cfg.get("n_trials", 1) or 1)
        mode = "multi" if n_trials_eff > 1 else "single"

    # ---------------------- Run sim ------------------------------
    if mode == "single":
        result = run_sim.run_single(
            cell=cell,
            geom=geom,
            sim_cfg=sim_cfg,
            groups_cfg=groups_cfg,
            inputs_by_group=inputs_by_group,
            rm=rm,
        )
    else:
        result = run_sim.run_multi(
            cell=cell,
            geom=geom,
            sim_cfg=sim_cfg,
            groups_cfg=groups_cfg,
            inputs_by_group=inputs_by_group,
            rm=rm,
        )

    # Record randomness metadata for all runs (helps debugging diffs)
    result.setdefault("meta", {})["randomness"] = rm.meta().as_dict()
    if snapshot_enabled:
        result["meta"]["cell_config"] = copy.deepcopy(cell_cfg)
        result["meta"]["geometry_config"] = copy.deepcopy(geom_config)

    # ---------------------- Save results -------------------------
    out_dir = Path(args.output_dir) if args.output_dir else tune_dir / "output_data"
    saved_path = run_sim.save_results(result, base_dir=out_dir)
    if saved_path is None:
        print("Results not saved (sim_cfg['output'] was empty).")
    else:
        print(f"Results saved to {saved_path}")

    total_elapsed = time.perf_counter() - job_start
    print(f"Total runtime: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
