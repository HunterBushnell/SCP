# Troubleshooting

## First Check

Run the setup checker:

```bash
python scripts/check_setup.py --steps 1 2 3 4 5 --cell PV --tune tuned --compile-modfiles
```

Use `--check-act` or `--check-bmtool` only when you want the checker to require
the corresponding optional external integration.

Run notebook portability checks:

```bash
python scripts/check_notebooks.py
```

## Compact Notebook Action Fails

- Step 1 prevents a second model-construction attempt after a failed load.
  Restart the kernel, correct the reported setup/config issue, and rerun from
  the top.
- If a source fingerprint warning names `cell_config.json`, morphology,
  fit/HOC, or MOD files, restart the kernel. For target, simulation, geometry,
  synapse, or ACT config edits, rerun the action that consumes the file instead.
- Quiet modes keep complete details in `pipeline_ui.step1_load_log`,
  `pipeline_ui.input_preview_log`, and `pipeline_ui.simulation_log`. Inspect the
  relevant string or disable its quiet checkbox before retrying.
- If **Plot Results** is disabled, complete **Run Simulation** successfully
  first; plotting always uses the latest loaded saved manifest.

## Missing Modfiles or NEURON Errors

- A model using only built-in NEURON mechanisms needs no `modfiles/` directory
  or compiled library. Confirm `cell_config.json` omits `paths.modfiles` or sets
  it to `null`.
- If the selected tune declares custom `.mod` sources, prepare or refresh it
  with `1_setup.ipynb`, or build the configured source directory manually:

```bash
cd <configured_modfiles_dir>
nrnivmodl
```

- Or run `scripts/check_setup.py --compile-modfiles`.
- In Colab, rerun the bootstrap cell before compiling/loading mechanisms.
- For ADB all-active bundles, use the Step 1 genome cleanup toggles when the
  fit JSON needs normalization before loading.

## Step 2 Passive Values Do Not Apply

Step 2 is manual-first and does not overwrite model files automatically.

- If Step 2 reports missing passive targets, provide all three passive values
  in `manual_passive_targets`, fill `target_config.json` fields
  `manual.passive.v_rest_mV`, `manual.passive.rin_MOhm`, and
  `manual.passive.tau_ms`, or set `target_source.mode` to `traces`/`allen_nwb`.
- Confirm the ACT values were copied into the correct `*_fit.json` or custom
  model source file.
- Save the edited file before rerunning the notebook.
- Rerun from **2.3 Build Cell** so the NEURON cell is rebuilt from the edited
  model files.
- Check `passive_area_mode` and `passive_area_scale` if ACT estimates look too
  large or too small.
- For passive checks, confirm active conductances are disabled or minimized
  according to your model's passive-tuning convention.

## Step 2 Allen/ADB NWB Passive Targets

For passive targets from a downloaded Allen/ADB ephys NWB file:

- Place the `*_ephys.nwb` file in the selected tune directory, set
  `target_config.json` field `allen_nwb.file`, or set
  `NWB_PASSIVE_PATH` explicitly.
- Start with `allen_nwb.passive.stimulus_names = ["Long Square"]`.
- The default current filter uses negative-current sweeps only:
  `allen_nwb.passive.max_current_pA = -1.0`.
- If no sweeps are found, inspect `allen_nwb.sweep_ids`, `allen_nwb.passive.sweep_ids`,
  `allen_nwb.passive.min_current_pA`, `allen_nwb.passive.max_current_pA`,
  and the stimulus names in the NWB file.
- The notebook writes `notebook_exports/step2_passive/allen_nwb_passive_targets.csv`
  and `notebook_exports/step2_passive/allen_nwb_passive_sweeps.csv` for review.

## Step 3 ACT Target Data Issues

For Allen/ADB NWB targets:

- Place the downloaded `*_ephys.nwb` file in the selected tune directory, set
  `target_config.json` field `allen_nwb.file`, or set `ACT_NWB_PATH`
  explicitly.
- Use `target_source.mode = "allen_nwb"` in `target_config.json` or
  `ACT_TARGET_MODE = "allen_nwb"` in the notebook.
- Start with `allen_nwb.active.stimulus_names = ["Long Square"]`.
- If no sweeps are found, inspect `allen_nwb.active.min_current_pA`,
  `allen_nwb.active.max_current_pA`, and the stimulus names in the NWB file.
- The preparation step writes `act_workspace/allen_nwb_fi_curve.csv` for review
  and `act_workspace/target_sf.csv` for ACT.

For custom FI CSV targets, use a current column such as `amp_pA`, `current_pA`,
`amp_nA`, or `mean_i`, and a frequency column such as `spike_frequency`,
`spike_frequency_hz`, or `frequency_hz`.

For `trace_npy` targets, `traces.active.file` must point to an ACT-compatible
`.npy` array with shape `(n_trials, n_timepoints, n_columns)`, voltage in mV in
column 0, and injected current in nA in column 1. See
`docs/reference/target_trace_formats.md`.

In `0_pipeline.ipynb`, click **Prepare ACT workspace** after changing
`target_config.json`, the ACT config, or any ACT option. Existing or partial
module output is intentionally protected; enable the explicit overwrite option
before rerunning that selection. If a run is cancelled, its process group is
terminated and its retained manifest is marked incomplete. Rerunning an earlier
module can make later predictions stale because the later module was trained
against the earlier proposal.

If compact preparation reports that ACT is unavailable locally, install or
clone ACT and set `ACT_ROOT`/`SCP_ACT_PATH`, then prepare again. Colab retains
the existing automatic ACT clone behavior. Non-PV/SST cells need a complete
`act_workspace/act_active_config.json` prepared and validated in
`3_active.ipynb`; registered non-Allen loaders are accepted experimentally only
after fresh-process cell construction succeeds.

## Missing Configs

Confirm the tune contains:

```text
cell_configs/cell_config.json
cell_configs/sim_config.json
cell_configs/geometry.json
```

For synapse-driven runs, also confirm:

```text
cell_configs/syn_config.json
cell_configs/syn_groups/*.json
```

Use Step 1 config scaffolding with `CONFIG_MODE = "fill"` or:

```bash
python scripts/step1_prepare.py \
  --tune-dir cells/PV/tunes/orig \
  --source-type existing \
  --no-download \
  --no-compile \
  --config-mode fill
```

## Synapse Group Not Used

- Check that the group file is listed in `cell_configs/syn_config.json`.
- Check the group has `"state": true`.
- Check each active block has `"state": true` or no `state` field.
- Check all `input_blocks` lie within `sim_config.tstart` and `sim_config.tstop`.

## Input Block Errors

Common causes:

- block `stop_ms` is not greater than `start_ms`,
- blocks overlap,
- block times fall outside the simulation window,
- a source-driven block crop duration does not match the block duration,
- a source file path is not valid from the repo root or tune context.

Gaps between input blocks are allowed and treated as quiescent.

## Input File Not Found

- Paths in `syn_config.json` resolve relative to `cell_configs/`.
- Paths inside `input_blocks[].source.path` should be checked from the runtime
  working directory used by the notebook/CLI.
- For Colab, confirm the referenced file exists after bootstrap/clone.

## No Outputs Written

- Check `save.enabled` in `sim_config.json`.
- Use `force_save = True` in `5_simulate.ipynb`.
- For CLI runs, use `--force-save`.
- For SLURM, `FORCE_SAVE=1` is the default in `run_slurm.sh`.
- If loading an existing run, check `load.enabled` and `load.path`.

## Auto Plots Missing

- Check `plots_profile` is not `off`.
- Check output saving is enabled or forced.
- Check plot dependencies are installed in the active environment.
- Check `save_plots_single_plot_preset` points to an existing preset file.

## Step 6 Has No Runs to Analyze

Step 6 reads saved Step 5 output folders. If no runs appear:

- rerun Step 5 with `force_save = True` or CLI `--force-save`,
- confirm the selected `cell_name`, `tunes_dir`, and `model_dir`,
- confirm `cells/<CELL>/tunes/<TUNE>/output_data/` exists in the runtime.

## Notebook vs SLURM Mismatch

- Use the same `tune_dir`, config files, and (when applicable) compiled mechanisms.
- Use the same `randomness_mode` and `seed`.
- Confirm SLURM `TOTAL_TRIALS`, `N_TRIALS`, and `--trial-offset` settings.
- Enable `snapshot.enabled` for deeper comparison.

## ACT or BMTool Not Found

Some tuning steps use external tools:

```bash
mkdir -p ../mods
git clone https://github.com/V-Marco/ACT.git ../mods/ACT
git clone https://github.com/cyneuro/bmtool.git ../mods/bmtool
```

If stored elsewhere:

```bash
export SCP_ACT_PATH=/path/to/ACT
export SCP_BMTOOL_PATH=/path/to/bmtool
```

ACT is optional for Step 2 target-derived proposals and Step 3 ACT active
tuning. Core passive sweeps and manual active/FI checks run without ACT.
BMTool is needed only when Step 4 synapse tuning is requested.
