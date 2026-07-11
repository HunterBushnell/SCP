# Troubleshooting

## First Check

Run the setup checker:

```bash
python scripts/check_setup.py --steps 1 2 3 4 5 --cell PV --tune seg_tuned --compile-modfiles
```

Run notebook portability checks:

```bash
python scripts/check_notebooks.py
```

## Missing Modfiles or NEURON Errors

- Prepare or refresh the tune with `1_setup.ipynb`.
- Or build manually:

```bash
cd <tune_dir>/modfiles
nrnivmodl
```

- Or run `scripts/check_setup.py --compile-modfiles`.
- In Colab, rerun the bootstrap cell before compiling/loading mechanisms.
- For ADB all-active bundles, use the Step 1 genome cleanup toggles when the
  fit JSON needs normalization before loading.

## Step 2 Passive Values Do Not Apply

Step 2 is manual-first and does not overwrite model files automatically.

- Confirm the ACT values were copied into the correct `*_fit.json` or custom
  model source file.
- Save the edited file before rerunning the notebook.
- Rerun from **2.3 Build Cell** so the NEURON cell is rebuilt from the edited
  model files.
- Check `passive_area_mode` and `passive_area_scale` if ACT estimates look too
  large or too small.
- For passive checks, confirm active conductances are disabled or minimized
  according to your model's passive-tuning convention.

## Step 3 ACT Target Data Issues

For Allen/ADB NWB targets:

- Place the downloaded `*_ephys.nwb` file in the selected tune directory or set
  `ACT_NWB_PATH` explicitly.
- Use `ACT_TARGET_MODE = "allen_nwb"`.
- Start with `ACT_NWB_STIMULUS_NAMES = ["Long Square"]`.
- If no sweeps are found, inspect `ACT_NWB_MIN_CURRENT_PA`,
  `ACT_NWB_MAX_CURRENT_PA`, and the stimulus names in the NWB file.
- The preparation step writes `act_workspace/allen_nwb_fi_curve.csv` for review
  and `act_workspace/target_sf.csv` for ACT.

For custom FI CSV targets, use a current column such as `amp_pA`, `current_pA`,
`amp_nA`, or `mean_i`, and a frequency column such as `spike_frequency`,
`spike_frequency_hz`, or `frequency_hz`.

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
  --tune-dir cells/PV/tunes/adb_peri \
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

- Use the same `tune_dir`, config files, and compiled mechanisms.
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

ACT is required for Step 2 passive tuning and optional Step 3 ACT active
tuning. Step 3 manual active/FI checks can run without ACT. BMTool is used by
Step 4 synapse workflows.
