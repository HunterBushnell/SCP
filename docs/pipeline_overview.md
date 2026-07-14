# Pipeline Overview

SCP is organized as a notebook-first workflow with optional CLI/SLURM entry
points for larger simulations.

## Main Flow

1. `1_setup.ipynb`: prepare a tune directory.
2. `2_passive.ipynb`: tune/check passive cell properties.
3. `3_active.ipynb`: tune/check active/channel properties.
4. `4_synapses.ipynb`: define and tune synapse groups.
5. `5_simulate.ipynb`: run the final simulation workflow.
6. `6_analysis.ipynb`: analyze and compare saved runs.
7. `7_tools.ipynb`: run optional utility scripts from a notebook.

Step 5 is the main simulation destination. Earlier notebooks prepare the cell
and configs; Step 6 is optional post-processing; Step 7 is optional maintenance
and export tooling.

## Step 1 Setup

Step 1 prepares the tune directory contract used by later steps:

- model source files, usually an Allen Database bundle or an existing local model,
- `modfiles/` and compiled NEURON mechanisms,
- `cell_configs/cell_config.json`,
- `cell_configs/sim_config.json`,
- `cell_configs/geometry.json`,
- optional `cell_configs/syn_config.json` and `cell_configs/syn_groups/*.json`,
- validation checks for files, compiled mechanisms, cell loading, and inputs.

Notebook: `1_setup.ipynb`

CLI equivalent:

```bash
python scripts/step1_prepare.py --cell PV --tune orig --specimen-id 484635029 --model-type perisomatic
```

See `guides/step_1_setup.md`.

## Step 2 Passive Tuning

Step 2 uses ACT helper functions to estimate passive membrane values from target
measurements, then leaves model edits under user control. It can run locally or
in Colab from the root notebook.

Passive targets can be entered directly in `target_config.json`, set in the
notebook, or extracted from downloaded Allen/ADB ephys NWB negative-current
sweeps.

Notebook: `2_passive.ipynb`

See `guides/step_2_passive.md`.

## Step 3 Active Tuning

Step 3 provides:

- manual active current-step checks,
- active-spiking metrics,
- voltage/current diagnostic plots,
- FI-curve checks,
- optional ACT active-tuning workspace generation and CLI execution.

ACT target data can be supplied as direct FI arrays, FI CSV, Allen/ADB ephys NWB
files, or ACT-compatible trace `.npy` files. The exact file contracts are in
`docs/reference/target_trace_formats.md`. The currently robust public paths
include passive-target extraction in Step 2 and FI-target CSV generation in
Step 3 from downloaded Allen/ADB NWB files.

Notebook: `3_active.ipynb`

See `guides/step_3_active.md`.

## Step 4 Synapse Tuning

Step 4 is a BMTool-based chemical synapse tuning workflow. It adapts SCP tune
files into BMTool's SynapseTuner flow and prints copyable parameter blocks for
`cell_configs/syn_groups/*.json`.

Notebook: `4_synapses.ipynb`

See `guides/step_4_synapses.md`.

## Step 5 Simulation

Step 5 loads the prepared tune and runs:

- cell loading,
- geometry definition,
- input generation from synapse configs,
- optional synapse-placement preview,
- synapse attachment,
- simulation,
- saving and optional auto-plotting.

Notebook: `5_simulate.ipynb`

CLI:

```bash
python run_pipeline.py --tune-dir cells/PV/tunes/tuned --n-trials 1 --force-save
```

SLURM:

```bash
CELL=SST TUNE=tuned N_TRIALS=10 sbatch run_slurm.sh
```

See `guides/step_5_simulate.md` and `advanced/cli_slurm.md`.

## Step 6 Analysis

Step 6 analyzes saved Step 5 outputs. It includes:

- a compact single-plot workflow,
- advanced output and input plotting UIs,
- extra analysis modes for metrics, config comparisons, input sampling,
  synapse summaries, snapshot comparison, and table exports.

Notebook: `6_analysis.ipynb`

See `guides/analysis.md` for usage and `guides/step_6_analysis.md` for detailed
preset/default-field references.

## Local vs Colab

The root notebooks are the current local and Colab entry points. Local notebooks
assume the environment is installed. Colab notebooks can clone SCP and install
runtime dependencies when needed.

CLI/SLURM entry points are local/HPC-oriented and use the same config files as
the notebooks.

## Step 7 Extra Tools

Step 7 wraps small utility scripts for notebook-first users:

- restore saved run values into tune configs,
- export `spikes.npz` to CSV,
- merge compatible run outputs,
- clear `slurm_*` output folders.

Notebook: `7_tools.ipynb`

See `guides/step_7_tools.md`.

## Data Flow

Inputs:

- `cells/<CELL>/tunes/<TUNE>/manifest.json`
- `cells/<CELL>/tunes/<TUNE>/modfiles/`
- `cells/<CELL>/tunes/<TUNE>/cell_configs/`
- optional `external_data/` sources
- optional local Allen/ADB ephys `.nwb` files for Step 2 passive and Step 3 ACT targets

Outputs:

- `cells/<CELL>/tunes/<TUNE>/output_data/<RUN>/`
- `run_manifest.json`
- sidecars such as `sim_cfg.json`, `syn_config.json`, `spikes.npz`, and `traces.npz`
- notebook-only diagnostics under `notebook_exports/`
- ACT active-tuning artifacts under `act_workspace/`

See `reference/outputs_layout.md`.
