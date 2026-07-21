# Quickstart

This is the shortest path from a bundled tune to a saved run.

## Prerequisites

```bash
conda env create -f environment.yml
conda activate scp-py311
python scripts/check_setup.py --steps 1 2 3 4 5 --cell PV --tune tuned --compile-modfiles
```

Generated mechanism builds are ignored by Git.

## Compact Notebook: Recommended

Open [`0_pipeline.ipynb`](../0_pipeline.ipynb), then choose **Run All** once.
This only renders the interface; it does not load a model or run a simulation.

### Step 1 — Setup and Load

Select `PV` / `tuned` and click **Prepare and load**. Leave **Quiet load**
enabled for the concise summary. The full captured stream remains available as
`pipeline_ui.step1_load_log`.

The compact notebook loads already prepared model sources and safely fills
missing standard configs without replacing existing values. Use
[`1_setup.ipynb`](../1_setup.ipynb) or `scripts/step1_prepare.py` for a new
Allen/ADB download or custom-loader setup.

### Step 2 — Passive Tuning

Optionally click **Compute ACT proposal** to review suggested passive values;
this does not modify the model. Then click **Run passive**. Target fields start
from `target_config.json`, and timing controls are under **Show advanced
options**.

### Step 3 — Active Tuning and FI Curve

Click **Run active protocol** and **Run FI curve** independently. Timing,
thresholds, and ionic-current display controls are under **Show advanced
options**.

ACT active tuning is experimental, review-only, and not release-blocking. It
runs in an isolated, cancellable process and never applies predictions to model
files. Use [`3_active.ipynb`](../3_active.ipynb) for its detailed workflow.

### Step 4 — Synapse Tuning (Optional)

Leave this step disabled for a first run. To use it, enable and click
**Initialize BMTool**, then run **Single Event** or **Interactive Tuner** from
their separate cards. Copy accepted values into the relevant synapse JSON.

### Step 5 — Check Inputs, Simulate, and Plot

1. In **Check Inputs**, choose synapse groups and preview plots, then click its
   run button. The optional seed shown here is also used by simulation.
2. In **Run Simulation**, choose trials, seed, mode, and optional output stem,
   then click the run button. The model runs in a fresh process and is always
   saved under a unique `pipeline_...` stem unless one is supplied.
3. In **Plot Results**, choose panels and a saved trial, then click the plot
   button. Use [`6_analysis.ipynb`](../6_analysis.ipynb) for comparisons,
   metrics, detailed styling, and export.

**Quiet preview** and **Quiet run** retain their complete subprocess streams in
`pipeline_ui.input_preview_log` and `pipeline_ui.simulation_log`. Advanced
widget changes are session-only overrides in `pipeline_settings`; they do not
edit JSON files.

Widget edits update `pipeline_settings`. Rerunning the settings cell pushes
code edits back to unlocked controls. Model selection locks after Step 1.
Changes to `cell_config.json`, morphology, fit/HOC, or MOD sources require a
kernel restart; runtime, target, geometry, and synapse configs reload at the
stage that uses them.

## Detailed Notebooks

Use the numbered path for full preparation, optimization, export, placement,
simulation, and analysis controls:

```text
1_setup.ipynb -> 2_passive.ipynb -> 3_active.ipynb -> 4_synapses.ipynb -> 5_simulate.ipynb -> 6_analysis.ipynb
```

`7_tools.ipynb` contains optional utilities.

## CLI

Run and save one trial:

```bash
python run_pipeline.py --tune-dir cells/PV/tunes/tuned --n-trials 1 --force-save --output-stem quickstart_pv
```

Run by cell/tune labels:

```bash
python run_pipeline.py --cell SST --tune tuned --n-trials 1 --force-save --output-stem quickstart_sst
```

Run a current-injection check:

```bash
python run_pipeline.py --tune-dir cells/PV/tunes/tuned --iclamp --force-save --output-stem pv_iclamp_check
```

## Preparing a Raw ADB Tune

```bash
python scripts/step1_prepare.py --cell PV --tune orig --specimen-id 484635029 --model-type perisomatic
python scripts/step1_prepare.py --cell SST --tune orig --specimen-id 485466109 --model-type "all active"
```

## Outputs

Saved runs are written under:

```text
cells/<CELL>/tunes/<TUNE>/output_data/<RUN>/
```

The compact notebook always saves its fresh-process run. Other entry points save
when enabled in `sim_config.json`, `force_save = True`, or `--force-save` is
used. See the [output layout](reference/outputs_layout.md) for file details.

## Next

- [Pipeline overview](pipeline_overview.md)
- [Step guides](README.md#step-guides)
- [CLI and SLURM](advanced/cli_slurm.md)
- [Troubleshooting](troubleshooting.md)
