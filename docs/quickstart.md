# Quickstart

This is the shortest path to check a bundled tune from passive behavior through
final simulation and produce a saved run for analysis.

## Prerequisites

Install the environment first:

```bash
conda env create -f environment.yml
conda activate scp-py311
```

Optional checks:

```bash
python scripts/check_setup.py --steps 5 --cell PV --tune tuned --compile-modfiles
python scripts/check_notebooks.py
```

`--compile-modfiles` builds missing NEURON mechanisms for the selected tune.
Generated `x86_64/` mechanism folders are ignored by Git.

## Option A: Compact Pipeline Notebook

1. Open `0_pipeline.ipynb`.
2. Select:
   - `cell_name = "PV"`
   - `tune_name = "tuned"`
3. Keep synapse tuning disabled for the first run; it is an optional interactive
   BMTool stage.
4. Run the notebook top to bottom. The final simulation runs in a fresh Python
   process and is always saved under a unique `pipeline_...` name.
5. Review the inline diagnostics or open `6_analysis.ipynb` for the full saved-run
   analysis workflow.

The compact notebook safely fills missing standard configs without replacing
existing values. Set `adb_specimen_id` only when you want it to download and
prepare an Allen/ADB model. Existing custom models must already have a registered
loader; use `1_setup.ipynb` for custom-loader setup.

## Option B: Detailed Notebooks

For the full preparation, optimization, export, and placement controls, use the
numbered notebooks in order:

```text
1_setup.ipynb -> 2_passive.ipynb -> 3_active.ipynb -> 4_synapses.ipynb -> 5_simulate.ipynb -> 6_analysis.ipynb
```

`7_tools.ipynb` is optional utility tooling.

To run only the detailed simulation notebook, open `5_simulate.ipynb` and set
`force_save = True` when you want a saved result.

## Option C: CLI

Run and save one trial:

```bash
python run_pipeline.py --tune-dir cells/PV/tunes/tuned --n-trials 1 --force-save --output-stem quickstart_pv
```

Run by cell/tune labels:

```bash
python run_pipeline.py --cell SST --tune tuned --n-trials 1 --force-save --output-stem quickstart_sst
```

Run a simple current-injection check instead of synapse-driven inputs:

```bash
python run_pipeline.py --tune-dir cells/PV/tunes/tuned --iclamp --force-save --output-stem pv_iclamp_check
```

## Option D: Prepare a Raw ADB Tune

Use Step 1 when creating or refreshing a tune directory:

```bash
python scripts/step1_prepare.py --cell PV --tune orig --specimen-id 484635029 --model-type perisomatic
```

For SST all-active:

```bash
python scripts/step1_prepare.py --cell SST --tune orig --specimen-id 485466109 --model-type "all active"
```

## Outputs

`0_pipeline.ipynb` always saves its final fresh-process run. Other runs are saved
when saving is enabled in `sim_config.json`, `force_save = True` in the detailed
simulation notebook, or `--force-save` is passed to the CLI. Saved runs are
written under:

```text
cells/<CELL>/tunes/<TUNE>/output_data/<RUN>/
```

The compact notebook uses a unique timestamped `pipeline_...` folder. Other
entry points use a timestamped `run_...` folder when no run name is specified.

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
