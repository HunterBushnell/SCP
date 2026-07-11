# Reproducibility

## Randomness Controls

Public configs should use:

- `randomness_mode`
- `seed`

Modes:

- `fixed`: identical stochastic choices across trials for a given seed.
- `derived`: reproducible trial-varying choices for a given seed.
- `random`: fresh stochastic choices; seeds used are recorded in metadata.

Examples:

```json
"randomness_mode": "fixed",
"seed": 12345
```

```json
"randomness_mode": "derived",
"seed": 12345
```

CLI override:

```bash
python run_pipeline.py --tune-dir cells/PV/tunes/seg_tuned --seed 12345 --force-save
```

## Trial Alignment

`--trial-offset` aligns array jobs with sequential multi-trial runs. This matters
when comparing a single multi-trial run to a SLURM array split across tasks.

`run_slurm.sh` sets the offset automatically when splitting `TOTAL_TRIALS`
across array tasks.

## Snapshot Mode

Use snapshot mode to capture richer comparison metadata:

```json
"snapshot": {
  "enabled": true,
  "n_trials": 1,
  "save_all_inputs": true,
  "save_all_traces": true,
  "save_syn_records_by_trial": true
}
```

CLI:

```bash
python run_pipeline.py --tune-dir cells/PV/tunes/seg_tuned --snapshot
```

Snapshot mode forces saving, full sidecars, input summaries, and trace/input
capture.

## Config Snapshots

Saved runs include sidecars such as:

- `sim_cfg.json`
- `syn_config.json`
- `run_manifest.json`

Use these as the authoritative record of what a run used.

## Append Mode

Append mode can add results to an existing run:

```json
"append": {
  "enabled": true,
  "path": "cells/PV/tunes/seg_tuned/output_data/example_run"
}
```

When comparing runs, prefer saved sidecars over the current working configs,
because the working configs may have changed after the run was created.
