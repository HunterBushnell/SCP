#!/usr/bin/env python
"""
CLI entrypoint for SCP Step 5 simulation runs.

Minimal usage from the repo root:
    python run_pipeline.py --tune-dir cells/SST/tunes/seg_tuned --mode multi --n-trials 10

This script intentionally stays thin: notebooks, local CLI runs, and SLURM
wrappers should all use the same backend in `modules.simulation`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from modules.simulation import SimulationOptions, SimulationSession, normalize_tune_dir  # noqa: E402
from modules.simulation.status import build_trial_callback  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SCP Step 5 for a tune directory.")
    p.add_argument(
        "--tune-dir",
        type=str,
        default=None,
        help="Path to tune directory containing sim_config.json and syn_config.json "
        "(optionally under cell_configs/).",
    )
    p.add_argument(
        "--cell",
        type=str,
        default=None,
        help="Cell name (e.g., SST, PV). Used if --tune-dir is not provided.",
    )
    p.add_argument(
        "--tune",
        type=str,
        default=None,
        help="Tune name (e.g., seg_tuned). Used if --tune-dir is not provided.",
    )
    p.add_argument(
        "--mode",
        choices=["single", "multi"],
        default=None,
        help="Run a single trial or multi-trial batch. If omitted, chosen from n_trials.",
    )
    p.add_argument("--n-trials", type=int, default=None, help="Override sim_config['n_trials'].")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override sim_config['seed'] / randomness base seed.",
    )
    p.add_argument(
        "--trial-offset",
        type=int,
        default=None,
        help="Global trial index offset for array-task or append workflows.",
    )
    p.add_argument(
        "--iclamp",
        "--current-injection",
        action="store_true",
        dest="iclamp",
        help="Run a simple somatic current injection test instead of synapse-driven sims.",
    )
    p.add_argument(
        "--snapshot",
        action="store_true",
        help="Enable full snapshot capture (forces saving and full sidecars).",
    )
    p.add_argument(
        "--force-save",
        action="store_true",
        help="Force saving even if sim_config disables output.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for outputs (default: output_data under tune-dir).",
    )
    p.add_argument(
        "--output-stem",
        type=str,
        default=None,
        help="Override sim_config['output'] stem; default assigns a timestamp if saving is enabled.",
    )
    return p.parse_args()


def _resolve_tune_dir(args: argparse.Namespace) -> Path:
    if args.tune_dir:
        tune_dir, note = normalize_tune_dir(args.tune_dir)
        if note:
            print(note)
        return tune_dir

    if not (args.cell and args.tune):
        raise ValueError("Provide either --tune-dir or both --cell and --tune.")
    return (HERE / "cells" / args.cell / "tunes" / args.tune).resolve()


def main() -> None:
    job_start = time.perf_counter()
    args = parse_args()
    tune_dir = _resolve_tune_dir(args)

    options = SimulationOptions(
        mode=args.mode,
        n_trials=args.n_trials,
        seed=args.seed,
        trial_offset=args.trial_offset,
        iclamp=args.iclamp,
        snapshot=args.snapshot,
        force_save=args.force_save,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
    )
    session = SimulationSession.from_tune(tune_dir, options=options)
    session.prepare()
    session.run(trial_callback=build_trial_callback(session))
    saved_path = session.save()

    if saved_path is None:
        prefix = "IClamp results" if session.iclamp_enabled else "Results"
        print(f"{prefix} not saved (sim_cfg['output'] was empty).")
    else:
        prefix = "IClamp results" if session.iclamp_enabled else "Results"
        print(f"{prefix} saved to {saved_path}")

    total_elapsed = time.perf_counter() - job_start
    print(f"Total runtime: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
