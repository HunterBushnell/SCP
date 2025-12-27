#!/usr/bin/env python
"""
Sample spike-train inputs for a synapse group multiple times and
summarize the mean firing-rate curve.

Usage (from repo root):
  python scripts/sample_inputs.py \
    --tune cells/SST/tunes/seg_tuned \
    --group pn_exc \
    --runs 20 \
    --bin-ms 5 \
    --out pn_exc_inputs.csv \
    --plot pn_exc_inputs.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path so modules_local can be imported when run as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from modules_local import input_sampling


def main():
    ap = argparse.ArgumentParser(description="Sample generated inputs and summarize firing-rate curves.")
    ap.add_argument(
        "--tune",
        required=True,
        help="Path to tune directory containing sim_config.json and syn_config.json "
             "(optionally under cell_configs/).",
    )
    ap.add_argument("--group", required=True, help="Synapse group name to sample")
    ap.add_argument("--runs", type=int, default=10, help="Number of input generations to sample")
    ap.add_argument("--bin-ms", type=float, default=None, help="Bin size in ms (default: source.bin_ms or 5)")
    ap.add_argument("--seed", type=int, default=None, help="Optional base seed for reproducibility")
    ap.add_argument("--out", default=None, help="Output CSV path (default: inputs_sample_<group>.csv)")
    ap.add_argument("--plot", default=None, help="Output PNG path (default: same stem as CSV with .png)")
    args = ap.parse_args()

    tune_dir = Path(args.tune).expanduser().resolve()
    if not tune_dir.is_dir():
        raise FileNotFoundError(f"Tune directory not found: {tune_dir}")

    centers, mean_rate, std_rate, sim_cfg, meta, ref_curve = input_sampling.sample_group_rates(
        tune_dir=tune_dir,
        group=args.group,
        runs=args.runs,
        bin_ms=args.bin_ms,
        seed=args.seed,
    )
    n_syn = meta["n_syn"]
    bin_ms = meta["bin_ms"]

    out_csv = Path(args.out) if args.out else Path(f"inputs_sample_{args.group}.csv")
    out_csv = out_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    header = "time_ms,rate_mean_hz,rate_std_hz,n_runs,n_syn,bin_ms"
    with out_csv.open("w") as f:
        f.write(header + "\n")
        for t, m, s in zip(centers, mean_rate, std_rate):
            f.write(f"{t:.6f},{m:.6f},{s:.6f},{meta['n_runs']},{n_syn},{bin_ms}\n")

    out_png = Path(args.plot) if args.plot else out_csv.with_suffix(".png")
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(centers, mean_rate, label="mean")
        plt.fill_between(centers, mean_rate - std_rate, mean_rate + std_rate, color="blue", alpha=0.2, label="±std")
        if ref_curve:
            ref_t, ref_r = ref_curve
            plt.plot(ref_t, ref_r, color="orange", linestyle="--", linewidth=1.5, label="source curve")
        plt.xlabel("Time (ms)")
        plt.ylabel("Rate (Hz per synapse)")
        plt.title(f"{args.group}: mean of {meta['n_runs']} samples")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
    except Exception:
        pass

    print(f"Wrote summary to {out_csv}")
    if out_png:
        print(f"Wrote plot to {out_png}")


if __name__ == "__main__":
    main()
