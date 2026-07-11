# Quickstart

This is the shortest path to run a bundled tuned example and produce a saved run
for analysis.

## Prerequisites

Install the environment first:

```bash
conda env create -f environment.yml
conda activate scp-py311
```

Optional checks:

```bash
python scripts/check_setup.py --steps 5 --cell PV --tune seg_tuned --compile-modfiles
python scripts/check_notebooks.py
```

`--compile-modfiles` builds missing NEURON mechanisms for the selected tune.
Generated `x86_64/` mechanism folders are ignored by Git.

## Option A: Notebook

1. Open `5_simulate.ipynb`.
2. Select:
   - `cell_name = "PV"`
   - `tune_name = "seg_tuned"`
3. Set `force_save = True` if you want to create an `output_data/` run for Step 6.
4. Run the notebook top to bottom.
5. Open `6_analysis.ipynb` to inspect saved outputs.

For the full preparation/tuning workflow, use the notebooks in order:

```text
1_setup.ipynb -> 2_passive.ipynb -> 3_active.ipynb -> 4_synapses.ipynb -> 5_simulate.ipynb -> 6_analysis.ipynb
```

`7_tools.ipynb` is optional utility tooling.

## Option B: CLI

Run and save one trial:

```bash
python run_pipeline.py --tune-dir cells/PV/tunes/seg_tuned --n-trials 1 --force-save --output-stem quickstart_pv
```

Run by cell/tune labels:

```bash
python run_pipeline.py --cell SST --tune seg_tuned --n-trials 1 --force-save --output-stem quickstart_sst
```

Run a simple current-injection check instead of synapse-driven inputs:

```bash
python run_pipeline.py --tune-dir cells/PV/tunes/seg_tuned --iclamp --force-save --output-stem pv_iclamp_check
```

## Option C: Prepare a Raw ADB Tune

Use Step 1 when creating or refreshing a tune directory:

```bash
python scripts/step1_prepare.py --cell PV --tune adb_peri --specimen-id 484635029 --model-type perisomatic
```

For SST all-active:

```bash
python scripts/step1_prepare.py --cell SST --tune adb_all --specimen-id 485466109 --model-type "all active"
```

## Outputs

Runs are saved only when saving is enabled in `sim_config.json`, `force_save = True`
in the notebook, or `--force-save` is passed to the CLI. Saved runs are written
under:

```text
cells/<CELL>/tunes/<TUNE>/output_data/<RUN>/
```

See `reference/outputs_layout.md` for file details.

## Next

- `pipeline_overview.md`: understand the full workflow.
- `guides/step_1_setup.md`: prepare model/tune directories.
- `guides/step_2_passive.md`: passive-property tuning.
- `guides/step_3_active.md`: active/channel tuning and ACT active targets.
- `guides/step_4_synapses.md`: BMTool synapse tuning.
- `guides/step_5_simulate.md`: simulation controls.
- `guides/analysis.md`: Step 6 analysis workflow.
- `advanced/cli_slurm.md`: batch and SLURM runs.
