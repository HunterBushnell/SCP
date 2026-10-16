# Reproducibility

Project-owned workflows can retain all scientific inputs and outputs outside
the code checkout. New results record SCP's installed version and available Git
revision, while `.scp_project.json` stores relocatable navigation separately
from tune configs. See the [dependency workflow](../guides/dependency_workflow.md).

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
python run_pipeline.py --tune-dir cells/PV/tunes/tuned --seed 12345 --force-save
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
python run_pipeline.py --tune-dir cells/PV/tunes/tuned --snapshot
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
  "path": "cells/PV/tunes/tuned/output_data/example_run"
}
```

When comparing runs, prefer saved sidecars over the current working configs,
because the working configs may have changed after the run was created.

## Project identity

Project tunes use the same saving and randomness controls as legacy tunes.
Saved `workspace_identity` records `project_name`, `cell_name`, and `tune_name`;
the absolute `tune_dir` remains a runtime location, not a portable namespace.
Appends and compatible-result merges guard against mixing different identities.
Step 6 can still compare explicit run paths across projects, and an explicit
Step 7 destination retains its own identity during restoration.

Matching workspace identity does not prove biological equivalence or matching
parameters, source hashes, conditions, or protocols. Older saved runs remain
readable without new fields. See [Project workspaces](../guides/project_workspaces.md).

## Construction recipes are not complete checkpoints

[Custom models](../guides/custom_models.md) distinguishes live cells from
saved builders and copy/reference source registration. Saved runs record
`model_provenance` and native source hashes. Runtime-only results have no
reconstruction recipe; their `model_state.json` is a descriptive pre-run
inventory, not a complete checkpoint of POINTERs, GLOBALs, random objects,
or callbacks. Unsaved in-memory parameter edits are not replayed by CLI/SLURM.

Reference mode keeps a registration snapshot but loads the current originals.
That baseline is not silently refreshed. Declare all native dependencies for
archiving, prefer copy mode for independent tuning, and materialize archived
external bundles only into a destination tune. Compiler caches are local build
products, not archived model sources.
