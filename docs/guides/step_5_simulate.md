# Step 5: Simulate

Step 5 is the primary simulation endpoint for the SCP pipeline. It runs prepared
tune directories from the same backend in notebooks, CLI runs, and SLURM jobs.

Entry points:

- Notebook: `../../5_simulate.ipynb`
- CLI: `../../run_pipeline.py`
- SLURM: `../../run_slurm.sh`

## Scope

Use Step 5 after the selected tune has been prepared by Step 1 and reviewed or
tuned through any needed Steps 2-4.

Step 5 can run:

- synapse-driven simulations from `syn_groups/*.json`,
- current-injection simulations through `sim_config.json`,
- single-trial or multi-trial runs,
- optional synapse-placement previews,
- immediate notebook diagnostics,
- saved output folders for later Step 6 analysis.

Step 5 does not tune cell or synapse parameters. It consumes the prepared tune
state and produces simulation outputs.

## Expected Inputs

A simulation tune directory should contain:

- native model source(s) declared by the selected loader,
- `cell_configs/cell_config.json`,
- `cell_configs/sim_config.json`,
- `cell_configs/geometry.json`,
- optional configured `modfiles/` with compiled mechanisms when custom `.mod` sources exist,
- optional `cell_configs/syn_config.json`,
- optional `cell_configs/syn_groups/*.json`.

Synapse-driven simulations also need valid `input_blocks` and any referenced
external input files. Current-injection runs can skip synapse configs when
`iclamp.enabled` is true.

## Outputs

When saving is enabled, runs are written under:

```text
cells/<CELL>/tunes/<TUNE>/output_data/<output_stem>/
```

Typical files include:

- `run_manifest.json`: run metadata and sidecar index,
- `sim_cfg.json`: resolved simulation config,
- `syn_config.json`: resolved synapse group config when used,
- `spikes.npz`: saved spike trains when requested,
- `traces.npz`: saved traces when requested,
- `model_artifacts/manifest.json`: loader-aware native-source provenance and hashes,
- `<cell>_<tune>_<output_stem>.pkl`: optional full result object when `save.full_results` is enabled,
- `plots/`: optional auto-generated plots.

See `../reference/outputs_layout.md` for the full output layout.

## Local and Colab Use

The root `5_simulate.ipynb` is the current notebook entry point for both local
and Colab use.

Local use:

- install the SCP environment,
- launch Jupyter from the repo or set `SCP_ROOT`,
- ensure custom mechanisms are compiled when the model has `.mod` sources.

Colab use:

- open the root notebook,
- run the environment setup cell,
- allow the notebook to clone SCP and install dependencies when needed,
- ensure any private data or external input files are present in the runtime.

Older separate Colab notebooks are archived and are not the current public
entry point.

## Notebook Workflow

### 5.0 Environment Setup

Run this cell first. It finds or clones SCP, configures imports, and installs
dependencies in Colab when requested.

Important controls:

- `SCP_REPO_URL`: repo URL to clone in a fresh Colab runtime.
- `SCP_REPO_BRANCH`: optional branch name.
- `SCP_REPO_DIR`: clone location in Colab.
- `INSTALL_DEPS`: `None` means install automatically in Colab and skip local
  installs.

### 5.1 Select Tune and Run Options

Choose the target tune and optional runtime overrides.

Important controls:

- `cell_name`: cell folder under `cells/`.
- `tune_name`: tune folder under `cells/<CELL>/tunes/`.
- `run_mode`: `None` uses config defaults; `"single"` or `"multi"` override.
- `n_trials`: trial count override for multi-trial runs.
- `seed`: base random seed override.
- `force_save`: force saving even if `sim_config.json` disables it.
- `output_stem`: optional output folder/run-name stem; `None` uses a timestamped `run_...` folder when saving is forced/enabled.
- `plots_profile`: optional auto-plot profile override.
- timing overrides such as `tstop`, `stim_start_ms`, and
  `stim_duration_ms`.

Leave overrides as `None` unless you are intentionally changing the run without
editing `sim_config.json`.

### 5.2 Prepare Simulation Session

This cell compiles mechanisms when needed, loads configs, builds the cell,
constructs geometry groups, and prepares the simulation session.

There are no normal user options here. If this cell fails, return to Step 1
validation or inspect the selected tune directory.

### 5.3 Optional Synapse-Placement Preview

Preview synapse placement without running the full simulation.

Important controls:

- `preview_synapses`: enable preview.
- `preview_synapse_groups`: `None` previews each active group; `"all"` aggregates
  all groups; a list previews selected group names.
- `preview_max_points`: cap plotted points for readability.
- `preview_include_table`: print a placement summary table.

Use this after editing `geometry.json` or `syn_groups/*.json`.

### 5.4 Run Simulation and Save

Run the prepared session and save according to config/options.

Saving follows this priority:

1. explicit notebook/CLI overrides,
2. `sim_config.json` save fields,
3. backend defaults.

If saving is disabled, the run remains in memory for diagnostics and can still
be manually saved in **5.6**.

### 5.5 Quick Diagnostics

Display lightweight diagnostics for the just-finished run.

Important controls:

- `diagnostic_plot`: `"summary"`, `"standard"`, `"single_plot"`, or `None`.
- `diagnostic_include_inputs`: include input summaries when available.

This is separate from Step 6 analysis. Use it for immediate sanity checks, not
publication-quality analysis.

### 5.6 Optional Manual Save

Use this when you initially ran without saving, reviewed diagnostics, and then
decided to keep the run.

Important controls:

- `manual_save`: set to `True` only when ready to save.
- `manual_output_stem`: output folder/run-name stem.
- `manual_note`: optional note stored with the saved run.

### 5.7 Optional Utilities

Optional helper cells can load a previous saved run or inspect the current
session state. They are convenience utilities and not required for the main
pipeline.

## `sim_config.json` Controls

Step 5 uses `cell_configs/sim_config.json` as the primary run-control file.
Common user-facing fields include:

- `mode`: `single` or `multi`.
- `n_trials`: number of trials for multi-trial runs.
- `tstart` / `tstop`: simulation time window.
- `dt`: NEURON time step.
- `stim_start_ms` / `stim_duration_ms`: shared stimulation timing defaults.
- `randomness_mode`, `seed`, `trial_randomness`: reproducibility controls.
- `save`: output saving settings.
- `plots_profile`: auto-plot profile.
- `iclamp`: current-injection mode.
- `cell_recording`: optional extra cell-variable recording.
- `syn_recording`: optional synapse-recording config scaffold.
- `snapshot`: richer reproducibility/debug sidecars.

See `../reference/configs_reference.md` for the full config reference.

## Synapse-Driven Runs

Synaptic input timing lives in `syn_groups/*.json` under `input_blocks`.

Supported public block modes:

- homogeneous Poisson blocks using `rate_hz`,
- inhomogeneous Poisson blocks using source files and optional transforms,
- precomputed blocks using saved spike trains.

Rules:

- Blocks must lie inside `sim_config.tstart` to `sim_config.tstop`.
- Blocks must not overlap.
- Gaps between blocks are quiescent.
- Source crop duration must match block duration for current public configs.

Step 4 tunes mechanism parameters; Step 5 applies the configured mechanism,
placement, weights, and input blocks to the full simulation.

## IClamp Mode

Enable current injection in `sim_config.json`:

```json
"iclamp": {
  "enabled": true,
  "amp_nA": 0.2,
  "delay_ms": null,
  "dur_ms": null
}
```

`delay_ms` and `dur_ms` fall back to `stim_start_ms` and `stim_duration_ms`
when set to `null`.

CLI:

```bash
python run_pipeline.py --tune-dir cells/PV/tunes/tuned --iclamp --force-save
```

## Saving

`sim_config.json` controls default saving:

```json
"save": {
  "enabled": true,
  "stem": null,
  "format": "pkl",
  "full_results": false
}
```

When `save.enabled` or `force_save` is true and `save.stem` is `null`, SCP
creates a timestamped folder under `output_data/`. Use `output_stem` or
`save.stem` for a specific run name.

CLI overrides:

```bash
python run_pipeline.py \
  --tune-dir cells/PV/tunes/tuned \
  --force-save \
  --output-stem my_run
```

The notebook also includes a manual save utility for keeping a run after it has
already completed.

## Auto Plotting

Use `plots_profile` for the main public interface:

- `off`: no auto plots.
- `basic`: output diagnostic plot.
- `inputs`: output plus input plots.
- `full`: output, input, and synapse-recording plots when data are present.

Quick diagnostic plotting in the notebook is separate from saved-output
analysis in `6_analysis.ipynb`; it is intended for immediate run inspection.

## Snapshot Mode

Snapshot mode captures richer sidecars for debugging and comparison:

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

## CLI

Run one trial:

```bash
python run_pipeline.py --tune-dir cells/PV/tunes/tuned --n-trials 1
```

Run by cell/tune:

```bash
python run_pipeline.py --cell SST --tune tuned --n-trials 5
```

Common flags:

- `--mode single|multi`
- `--n-trials`
- `--seed`
- `--trial-offset`
- `--iclamp`
- `--snapshot`
- `--force-save`
- `--output-dir`
- `--output-stem`

## SLURM

Minimal batch run:

```bash
CELL=SST TUNE=tuned N_TRIALS=10 sbatch run_slurm.sh
```

Split total trials across array tasks:

```bash
TOTAL_TRIALS=100 sbatch --array=0-9 run_slurm.sh
```

See `../advanced/cli_slurm.md`.

## Troubleshooting

### Tune Does Not Load

- Rerun Step 1 validation.
- Confirm `cell_configs/cell_config.json` points to the correct loader/files.
- Confirm custom mechanisms are compiled and loadable when `.mod` sources exist.
- For HOC templates, confirm both explicit runtime conditions are present.

### No Synapses Are Attached

- Confirm `cell_configs/syn_config.json` lists the intended group files.
- Confirm each group has `"state": true`.
- Confirm active input blocks have no `state` field or `"state": true`.
- Use **5.3 Optional Synapse-Placement Preview** to inspect placement.

### Input File Not Found

- Check paths in `input_blocks[].source.path`.
- In Colab, upload or clone any external input data before preparing the
  session.
- Prefer repo-relative paths for portable examples.

### No Outputs Written

- Check `save.enabled` in `sim_config.json`.
- Use `force_save = True` in the notebook or `--force-save` in the CLI.
- If the run was already completed in the notebook, use **5.6 Optional Manual
  Save**.

### Notebook and CLI Results Differ

- Use the same tune directory and config files.
- Check notebook override values from **5.1**.
- Check `seed`, `randomness_mode`, `trial_randomness`, and `trial_offset`.
- Enable `snapshot.enabled` for deeper debugging.
