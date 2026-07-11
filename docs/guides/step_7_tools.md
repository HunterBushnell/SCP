# Step 7: Extra Tools

Step 7 provides notebook access to small utility scripts for users who prefer
notebooks over terminal commands.

Notebook: `../../7_tools.ipynb`

## Scope

Step 7 is optional. It is not part of the required model-preparation or
simulation path.

Use Step 7 for small, self-contained maintenance/export tasks:

- restore saved run values into tune configs,
- export `spikes.npz` to CSV,
- merge compatible run outputs,
- clear top-level `slurm_*` output folders.

Large workflows should stay in their dedicated notebooks or scripts rather than
being absorbed into Step 7.

## Safety Model

Every destructive or writing tool uses an explicit execution gate:

- `RUN_RESTORE`
- `RUN_EXPORT_SPIKES`
- `RUN_MERGE`
- `RUN_CLEAR_SLURM`

Defaults are dry-run or preview mode. Set the relevant toggle to `True` only
after reviewing the printed paths/options.

## Local and Colab Use

`7_tools.ipynb` uses the same root-notebook bootstrap pattern as the current
public notebooks. It can run locally or in Colab.

Local use:

- launch Jupyter from the repo or set `SCP_ROOT`,
- keep write toggles disabled until paths are verified.

Colab use:

- run the environment setup cell,
- make sure target run/output files exist in the runtime,
- avoid deleting outputs unless they are intentionally staged in Colab.

## Tools

### 7.1 Restore Run State to Tune Configs

Backed by:

```text
scripts/restore_run_state.py
```

Use this to restore values from a saved run into a target tune while preserving
the target file structure.

Common controls:

- `from_run`: saved run folder, results folder, or `run_manifest.json`.
- `to_tune`: target tune directory.
- `apply`: config classes to restore.
- `syn_groups`: `all` or a comma-separated subset.
- `allow_source_fallback`: allow fallback reads from source tune files when run
  sidecars are missing.
- `backup`: write timestamped backups before applying changes.

CLI equivalent:

```bash
python scripts/restore_run_state.py \
  --from-run cells/SST/tunes/seg_tuned/output_data/example_run \
  --to-tune cells/SST/tunes/seg_tuned \
  --apply sim_config,cell_config,geometry,syn_config,syn_groups \
  --syn-groups all
```

Add `--write` to apply changes.

### 7.2 Export Spikes CSV

Backed by:

```text
scripts/export_spikes_csv.py
modules.analysis.analysis.export_spikes_trials_csv
```

Use this to convert `spikes.npz` into one CSV row per trial.

Common controls:

- `spikes_source`: run folder, results folder, or direct `spikes.npz` path.
- `spikes_csv_out`: output path, or `None` for `spikes_trials.csv` beside the
  source file.
- `spikes_time_delimiter`: delimiter inside the spike-times cell.
- `spikes_precision`: significant digits for spike times.
- `spikes_overwrite`: allow replacing an existing CSV.

CLI equivalent:

```bash
python scripts/export_spikes_csv.py \
  --input cells/PV/tunes/seg_tuned/output_data/example_run_a \
  --delimiter "," \
  --precision 10 \
  --trial-prefix trial_
```

### 7.3 Merge Two Outputs

Backed by:

```text
scripts/merge_two_runs.py
```

Use this to merge two compatible run outputs into one multi-trial run.

Common controls:

- `run_a`, `run_b`: source run folders, results folders, or manifests.
- `output_dir`: output-data directory for the merged run.
- `output_stem`: merged run folder name.
- `strict_configs`: block merge on config differences.
- `keep_logs`: write merge reports under `logs/`.
- `max_diffs`: maximum config diff lines per section.

CLI equivalent:

```bash
python scripts/merge_two_runs.py \
  --run-a cells/PV/tunes/seg_tuned/output_data/example_run_a \
  --run-b cells/PV/tunes/seg_tuned/output_data/example_run_b \
  --output-dir cells/PV/tunes/seg_tuned/output_data \
  --output-stem merge_example
```

Add `--write` to save the merged output.

### 7.4 Clear `slurm_*` Runs

Backed by:

```text
scripts/clear_slurm_runs.py
```

Use this to delete top-level entries under `output_data/` whose names match a
prefix, usually `slurm_`.

Common controls:

- `tune_dir`: tune directory containing `output_data/`.
- `output_data_dir`: direct override for the output-data directory.
- `slurm_prefix`: prefix to match.
- `max_items_to_show`: maximum number of matched paths printed.

CLI equivalent:

```bash
python scripts/clear_slurm_runs.py \
  --tune-dir cells/SST/tunes/seg_tuned
```

Add `--write` to delete matching entries.

## Future Tool Criteria

Good Step 7 additions should:

- already have a script or callable backend function,
- be safe in dry-run mode,
- need only a few user-facing options,
- be useful to notebook-first users,
- not duplicate a full workflow from Steps 1-6.

