#!/usr/bin/env python3
"""
Export spikes.npz to a simple row-per-trial CSV.

Usage (from repo root):
  python scripts/export_spikes_csv.py \
    --input cells/PV/tunes/seg_tuned/output_data/my_run/results/spikes.npz

You may also pass a run directory (containing results/spikes.npz):
  python scripts/export_spikes_csv.py \
    --input cells/PV/tunes/seg_tuned/output_data/my_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules_local.analysis import analysis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export spikes.npz into one CSV row per trial.")
    p.add_argument(
        "--input",
        required=True,
        help="Path to spikes.npz or run/results directory containing spikes.npz.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: alongside spikes.npz as spikes_trials.csv).",
    )
    p.add_argument(
        "--delimiter",
        default="|",
        help="Delimiter used inside spike_times_ms cell for each trial (default: '|').",
    )
    p.add_argument(
        "--precision",
        type=int,
        default=10,
        help="Numeric precision for spike times (default: 10 significant digits).",
    )
    p.add_argument(
        "--trial-prefix",
        default="trial_",
        help="Prefix for trial_n values (default: 'trial_').",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output CSV if it already exists.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_path = analysis.export_spikes_trials_csv(
        args.input,
        out_csv=args.out,
        delimiter=args.delimiter,
        precision=args.precision,
        overwrite=bool(args.overwrite),
        trial_prefix=args.trial_prefix,
    )
    print(f"Wrote spikes CSV: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

