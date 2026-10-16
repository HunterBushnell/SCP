# Outputs Layout

Runs are saved under the selected tune directory when saving is enabled or
forced:

```text
cells/<CELL>/tunes/<TUNE>/output_data/<output_stem>/
projects/<PROJECT>/cells/<CELL>/tunes/<TUNE>/output_data/<output_stem>/
```

Use `force_save = True` in `5_simulate.ipynb` or `--force-save` in
`run_pipeline.py` when configs keep saving disabled by default.

Both layouts share the same run format. An explicit output directory can still
place results elsewhere. Saved `workspace_identity` contains `project_name`,
`cell_name`, and `tune_name`, separately from the absolute runtime `tune_dir`;
legacy workspaces use a null project. Use saved metadata, not filenames, to
identify a project run. See [Project workspaces](../guides/project_workspaces.md).

## Typical Files

- `run_manifest.json`
- `sim_cfg.json`
- `meta.json`
- `syn_config.json`
- `spikes.npz`
- `traces.npz`

## Optional Sidecars

- `input_stats.json`: detailed per-trial binned input stats when
  `save_input_stats` is enabled. This file can be large and may be omitted from
  curated public examples; loaders skip it safely when absent.
- `input_summaries.json`: compact per-trial input counts and summary metadata.
- `inputs_sample.pkl`: saved input samples.
- `syn_records.pkl`: synapse recordings when requested.
- `syn_records_by_trial.pkl`: per-trial synapse recordings when requested.
- `cell_recordings.pkl`: single-run extra cell recordings.
- `cell_recordings_by_trial.pkl`: multi-run extra cell recordings.
- `<fit-file-name>.json`: fit sidecar when `save_fit_json_sidecar` is enabled and found.
- `cell_config.json` and `geometry_config.json`: tune-state sidecars used for loader-aware reconstruction.
- `model_artifacts/manifest.json`: loader name, tune-relative native-source paths, and SHA-256 hashes.
- `model_artifacts/files/`: loader-declared HOC/Allen sources and tune-local `.mod` sources; compiled outputs are excluded.
- `plots/`: auto plots when enabled by `plots_profile` or lower-level save flags.
- `<cell>_<tune>_<output_stem>.pkl`: full results bundle when `save.full_results` is enabled.

New named-project bundles use
`<project>_<cell>_<tune>_<output_stem>.pkl` (or `.npz` for that full-results
format). Legacy filenames stay unchanged. The run manifest and saved
`workspace_identity` remain authoritative; do not parse filenames for routing
or trial-pooling compatibility.

## Notebook Exports

Some notebooks can write lightweight diagnostics under the selected tune:

```text
cells/<CELL>/tunes/<TUNE>/notebook_exports/
```

These exports are scratch artifacts for notebook review, not Step 5 pipeline
runs. They are intentionally separate from `output_data/`.
Project tunes use the same tune-relative export location.

Common Step 2 passive target exports:

- `step2_passive/allen_nwb_passive_targets.csv`: aggregated passive targets
  extracted from an Allen/ADB NWB file.
- `step2_passive/allen_nwb_passive_sweeps.csv`: per-sweep passive metrics used
  for the aggregate target values.

## ACT Workspace Artifacts

Step 3 optional ACT active tuning writes under:

```text
cells/<CELL>/tunes/<TUNE>/act_workspace/
```

Typical generated review/config files:

- `cell_builder.py`
- `act_active_config.json`
- `target_sf.csv`
- `allen_nwb_fi_curve.csv` when extracting Allen/ADB NWB FI targets
- `metrics_<module>.csv`
- `prediction_<module>.json`

The complete `act_workspace/` directory is generated, machine-specific state and
is ignored by Git. It may contain absolute paths and multiprocessing artifacts;
keep it locally or archive selected review results outside the workspace.
Project tunes also keep `act_workspace/` under their selected tune; project
selection does not change optional ACT execution behavior.

## SLURM Array Runs

Merged arrays:

```text
output_data/<batch_stem>/results/
```

Per-task outputs before merge:

```text
output_data/<batch_stem>/parts/
```

Logs, when copied successfully:

```text
output_data/<batch_stem>/logs/
```

## Manifest

`run_manifest.json` is the authoritative index of files written for a run. Use
it instead of inferring run contents from filenames.

Model-artifact archiving is enabled by default, including for full-results
runs. Restoring these files is a separate, opt-in Step 7 action and remains a
dry run until `--write` is supplied.

## Ignored Generated Artifacts

The repo ignores local/generated data such as:

- `cells/**/*.nwb`
- `cells/**/output_data/`
- `cells/**/notebook_exports/`
- `cells/**/act_workspace/`
- `cells/**/x86_64/`
- `cells/**/modfiles/x86_64/`

Equivalent generated-output, export, backup, ACT, and compiled-product rules
apply under `projects/*/cells/`. Native source files and configs remain
Git-eligible in both layouts. Entire private projects need local exclusions.

## Custom-model provenance

`meta.json` records construction kind, source mode, and recipe availability.
Runtime-only runs additionally write `model_state.json`: a descriptive pre-run
cell-scoped inventory, not a resumable NEURON checkpoint. Their artifact
manifest records that reconstruction is unavailable and does not attribute an
unused workspace recipe. Declared Python/HOC/native bundles retain logical
layout and hashes in `model_artifacts/`; compiled caches are excluded.
See [Custom models](../guides/custom_models.md).
