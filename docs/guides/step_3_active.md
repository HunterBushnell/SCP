# Step 3: Active Tuning

Step 3 checks and tunes active/channel behavior for a tune prepared by Step 1
and passively checked/tuned in Step 2 when applicable.

Notebook: `../../3_active.ipynb`

## Scope

The root `3_active.ipynb` is the primary Step 3 entry point for both local and
Colab use.

Step 3 is manual-first for model edits:

- select a prepared tune,
- compile/load mechanisms,
- build and inspect the cell,
- optionally prepare and run ACT active tuning from a tune-local workspace,
- run positive current-injection sweeps,
- compute active-spiking metrics,
- inspect voltage traces and optional recorded currents,
- generate and plot an FI curve,
- manually edit active/channel parameters in the model source files,
- rerun from the build/check cells.

ACT active tuning is integrated as an optional helper, not a mandatory path. The
notebook can prepare ACT inputs, run ACT modules, collect predictions, and
evaluate temporary predictions without overwriting model files.

## Expected Inputs

- a tune directory from Step 1,
- compiled or compilable `modfiles/`,
- `cell_configs/cell_config.json`,
- model files referenced by `manifest.json`,
- passive parameters from Step 2 when applicable,
- optional ACT target data for active optimization.

## Outputs

Step 3 does not create Step 5 simulation runs. Optional notebook diagnostics are
scratch/review artifacts.

Manual active/FI exports write under:

```text
cells/<CELL>/tunes/<TUNE>/notebook_exports/step3_active/
```

ACT active-tuning artifacts write under:

```text
cells/<CELL>/tunes/<TUNE>/act_workspace/
```

The ACT workspace can contain:

- `cell_builder.py`: generated importable builder used by ACT multiprocessing,
- `target_sf.csv`: normalized ACT FI target summary features,
- `allen_nwb_fi_curve.csv`: review-friendly FI curve extracted from an
  Allen/ADB NWB file when using `ACT_TARGET_MODE = "allen_nwb"`,
- `act_active_config.json`: shared notebook/CLI config,
- `metrics_<module>.csv`: ACT module metrics,
- `prediction_<module>.json`: ACT conductance predictions,
- `module_<name>/`: ACT training/evaluation arrays and outputs,
- `output/`: optional temporary FI evaluation outputs.

Heavy ACT outputs are ignored by `.gitignore`; small config/target/prediction
files can be reviewed and tracked if desired.

## Manual Editing Boundary

Step 3 does not automatically overwrite model files. Use the active sweep, FI
curve, and optional ACT prediction outputs to decide which active/channel
parameters to adjust. Then edit the model source files manually and rerun from
**3.3 Build Cell**.

For ADB/Allen-style tunes, this usually means reviewing the tune's `*_fit.json`
or related mechanism parameters. For custom loaders, edit the model-specific
active/channel parameter source.

## Notebook Workflow

### 3.1 Select Tune Directory

Choose a tune directory prepared by Step 1. The default example is currently:

```python
cell_name = "SST"
tune_name = "seg_tuned"
```

Use `tune_dir_override` when working with a path outside the standard
`cells/<CELL>/tunes/<TUNE>` layout.

### 3.2 Compile and Load Mechanisms

Compile and/or load the tune's `modfiles/`.

Important controls:

- `RECOMPILE_MODFILES`: force recompilation even if compiled files already
  exist.
- `LOAD_COMPILED_DLL`: load compiled mechanisms into the current NEURON
  session.

### 3.3 Build Cell

Build the NEURON cell using `cell_configs/cell_config.json`, then print section
and area summaries. Rerun this cell after editing model parameters.

### 3.4 ACT Active Tuning Setup

This optional section adapts the selected SCP tune into ACT's active optimizer
without placing project-specific files inside the ACT repo.

#### 3.4.1 Controls and Target Data

Important controls:

- `USE_ACT_ACTIVE_TUNING`: prepare/use ACT active tuning when `True`.
- `RUN_ACT_MODULES`: run ACT modules from the notebook when `True`.
- `EVALUATE_ACT_PREDICTIONS`: run temporary FI evaluation with saved ACT
  predictions when `True`.
- `OVERWRITE_ACT_OUTPUTS`: allow rerunning modules/evaluation by deleting prior
  ACT outputs.
- `ACT_MODULE_TO_RUN`: `"all"`, `"lto"`, `"spiking"`, or `"bursting"`.
- `ACT_WORKSPACE`: tune-local ACT workspace path.
- `ACT_TARGET_MODE`: `"fi_arrays"`, `"fi_csv"`, `"allen_nwb"`, or
  `"trace_npy"`.

Target modes:

- `fi_arrays`: enter current amplitudes and target firing rates directly in the
  notebook.
- `fi_csv`: place a CSV in the workspace and point `ACT_FI_CSV_PATH` to it.
  Accepted current columns include `amp_pA`, `current_pA`, `amp_nA`, or
  `mean_i`; accepted frequency columns include `spike_frequency`,
  `spike_frequency_hz`, or `frequency_hz`.
- `allen_nwb`: place a downloaded Allen/ADB ephys `.nwb` file in the tune
  folder and point `ACT_NWB_PATH` to it. SCP extracts selected current-clamp
  sweeps, writes ACT's `target_sf.csv`, and writes a readable
  `allen_nwb_fi_curve.csv` summary for review. The default extraction uses
  `Long Square` sweeps, non-negative current amplitudes, averaged repeated
  amplitudes, and a `-20 mV` spike threshold.
- `trace_npy`: place an ACT-compatible target trace `.npy` file in the
  workspace and point `ACT_TRACE_NPY_PATH` to it.

For Allen/ADB models, the intended workflow is to download the ephys NWB from
the Allen cell page, place it in the selected tune directory, set:

```python
ACT_TARGET_MODE = "allen_nwb"
ACT_NWB_PATH = context.tune_dir / "<allen_ephys_result>_ephys.nwb"
```

Then run **3.4.3 Prepare ACT Workspace**. The notebook auto-detects a single
`*_ephys.nwb` file in the tune folder when present.

#### 3.4.2 Cell Adapter and Module Settings

These settings define ACT's view of the model:

- `ACT_PASSIVE_NAMES`: passive variable names expected by ACT.
- `ACT_ACTIVE_CHANNELS`: full list of active conductance variables ACT may use.
- `ACT_SIM_PARAMS`: ACT simulation and current-injection timing.
- `ACT_OPTIMIZER`: CPU count, random forest settings, spike threshold, and
  selected training features.
- `ACT_FILTER`: optional filtering of saturated/no-spike traces.
- `ACT_MODULES`: editable module definitions and conductance bounds.

Bundled ADB presets use three active modules:

- `lto`: low-threshold/near-threshold channels such as `gbar_Nap`,
  `gbar_K_T`, and `gbar_Im_v2`.
- `spiking`: fast spiking channels such as `gbar_NaTa` and `gbar_Kd`.
- `bursting`: calcium/high-threshold potassium channels such as
  `gbar_Ca_LVA`, `gbar_Ca_HVA`, `gbar_Kv2like`, and `gbar_Kv3_1`.

Edit these names and bounds for non-ADB models or different mechanism names.

#### 3.4.3 Prepare ACT Workspace

Preparing the workspace writes:

- `act_workspace/cell_builder.py`,
- `act_workspace/target_sf.csv` when using FI target modes,
- `act_workspace/act_active_config.json`.

This step does not run ACT modules.

#### 3.4.4 Run ACT Modules

Running ACT modules can be computationally expensive. Keep
`RUN_ACT_MODULES = False` while editing settings. When ready, set it to `True`
and choose `ACT_MODULE_TO_RUN`.

The same workspace can be run from the terminal:

```bash
python scripts/run_act_active.py --cell SST --tune seg_tuned --run --module all
```

Run one module:

```bash
python scripts/run_act_active.py --cell SST --tune seg_tuned --run --module lto
```

Prepare only:

```bash
python scripts/run_act_active.py --cell SST --tune seg_tuned --prepare --prepare-only
```

#### 3.4.5 Review Predictions and Optional Evaluation

ACT predictions are saved as `prediction_<module>.json`. Treat them as
candidate values for manual review/editing.

Set `EVALUATE_ACT_PREDICTIONS = True` to run a temporary FI check using saved
predictions without changing model files.

### 3.5 Run Active Protocol

Runs positive current steps and reports:

- spike count,
- spike frequency,
- resting voltage,
- peak and minimum voltage during stimulation,
- first-spike latency,
- mean/min interspike interval,
- ISI coefficient of variation,
- adaptation ratio.

Important controls:

- `active_sim_params`: current-injection timing and integration settings.
- `active_sim_amps`: current steps in pA.
- `active_spike_threshold_mv`: spike peak detection threshold.

### 3.6 Plot Active Trace Check

Plots voltage traces and can optionally add a recorded-current subplot for one
selected current step.

Important controls:

- `PLOT_XLIM`, `PLOT_YLIM`: voltage plot limits.
- `PLOT_CURRENTS`: include recorded-current subplot.
- `CURRENT_AMP`: selected amplitude for current traces.
- `CURRENT_NAMES`: selected recorded currents, or `None` for auto-selection.
- `EXPORT_TRACE_FIGURE`: save figure exports to `notebook_exports/`.

### 3.7 FI Curve Check

Runs a current sweep, plots model frequency vs. current, and can overlay bundled
PV/SST biological reference points or user-provided custom reference points.

Important controls:

- `fi_sim_params`: current-injection timing and integration settings.
- `FI_AMP_RANGE`: `(start, stop, step)` pA sweep definition.
- `SHOW_BIO_FI_REFERENCE`: overlay bundled example reference points.
- `CUSTOM_BIO_FI_REFERENCE`: provide custom `(current_pA, frequency_hz)` points.
- `EXPORT_FI_FIGURE`, `EXPORT_FI_CSV`: save optional review artifacts.

## CLI Entry Point for ACT Active Tuning

The notebook and CLI use the same backend and workspace config.

Common commands:

```bash
python scripts/run_act_active.py --cell SST --tune seg_tuned --prepare --prepare-only
python scripts/run_act_active.py --cell SST --tune seg_tuned --run --module all
python scripts/run_act_active.py --cell SST --tune seg_tuned --evaluate
```

Useful options:

- `--workspace`: explicit ACT workspace directory.
- `--config`: explicit `act_active_config.json` or workspace directory.
- `--target-mode fi_arrays|fi_csv|allen_nwb|trace_npy`: target data mode
  during prepare.
- `--fi-currents-pa`, `--fi-frequencies-hz`: comma-separated FI arrays.
- `--fi-csv`: FI target CSV path.
- `--nwb`: Allen/ADB ephys NWB file for FI extraction.
- `--nwb-stimulus-names`: comma-separated NWB stimulus names, default
  `Long Square`.
- `--nwb-min-current-pa`, `--nwb-max-current-pa`: optional current filters.
- `--nwb-keep-repeats`: keep repeated current amplitudes as separate target
  rows instead of averaging them.
- `--trace-npy`: ACT-compatible trace target path.
- `--n-cpus`: override configured ACT CPU count.
- `--overwrite`: rerun by replacing existing ACT outputs.

## Validation Status

The current implementation validates workspace generation and notebook
structure. Full ACT module execution should still be tested on the target model
because ACT runtime behavior depends on mechanism names, model builder behavior,
conductance bounds, multiprocessing, and target data quality.

## Next Step

After active behavior is acceptable, continue to Step 4 for synapse setup/tuning
or Step 5 for current-injection/simulation checks on the prepared tune.
