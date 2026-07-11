# Outputs Layout

Runs are saved under the selected tune directory when saving is enabled or
forced:

```text
cells/<CELL>/tunes/<TUNE>/output_data/<output_stem>/
```

Use `force_save = True` in `5_simulate.ipynb` or `--force-save` in
`run_pipeline.py` when configs keep saving disabled by default.

## Typical Files

- `run_manifest.json`
- `sim_cfg.json`
- `meta.json`
- `syn_config.json`
- `spikes.npz`
- `traces.npz`

## Optional Sidecars

- `input_stats.json`: input summaries when `save_input_stats` is enabled.
- `inputs_sample.pkl`: saved input samples.
- `syn_records.pkl`: synapse recordings when requested.
- `syn_records_by_trial.pkl`: per-trial synapse recordings when requested.
- `cell_recordings.pkl`: single-run extra cell recordings.
- `cell_recordings_by_trial.pkl`: multi-run extra cell recordings.
- `<fit-file-name>.json`: fit sidecar when `save_fit_json_sidecar` is enabled and found.
- `plots/`: auto plots when enabled by `plots_profile` or lower-level save flags.
- `<cell>_<tune>_<output_stem>.pkl`: full results bundle when `save.full_results` is enabled.

## Notebook Exports

Some notebooks can write lightweight diagnostics under the selected tune:

```text
cells/<CELL>/tunes/<TUNE>/notebook_exports/
```

These exports are scratch artifacts for notebook review, not Step 5 pipeline
runs. They are intentionally separate from `output_data/`.

## ACT Workspace Artifacts

Step 3 optional ACT active tuning writes under:

```text
cells/<CELL>/tunes/<TUNE>/act_workspace/
```

Typical small review/config files:

- `cell_builder.py`
- `act_active_config.json`
- `target_sf.csv`
- `allen_nwb_fi_curve.csv` when extracting Allen/ADB NWB FI targets
- `metrics_<module>.csv`
- `prediction_<module>.json`

Heavy module/output folders under `act_workspace/` are generated artifacts and
are ignored by Git.

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

## Ignored Generated Artifacts

The repo ignores local/generated data such as:

- `cells/**/*.nwb`
- `cells/**/output_data/`
- `cells/**/notebook_exports/`
- `cells/**/act_workspace/module_*/`
- `cells/**/act_workspace/output/`
- `cells/**/x86_64/`
- `cells/**/modfiles/x86_64/`
