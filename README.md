# Single Cell Pipeline (SCP)

SCP is a notebook-first workflow for preparing, tuning, simulating, and analyzing
single-cell NEURON models. The current examples focus on PV and SST cells, but
the repo is organized around reusable tune directories and JSON configs so other
cell models can be adapted without changing the main pipeline code.

## Quick Start

1. Install the environment:
   - `conda env create -f environment.yml`
   - `conda activate scp-py311`
2. Optional setup check:
   - `python scripts/check_setup.py --steps 1 2 3 4 5 --cell PV --tune seg_tuned --compile-modfiles`
3. Run an example:
   - Notebook: open `5_simulate.ipynb`, set `force_save = True` if you want a saved run.
   - CLI: `python run_pipeline.py --tune-dir cells/PV/tunes/seg_tuned --n-trials 1 --force-save`
4. Analyze saved runs:
   - open `6_analysis.ipynb` after a run has been saved under `output_data/`.

See `docs/quickstart.md` for the shortest runnable path and
`docs/installation.md` for local/Colab setup.

## Pipeline

- `1_setup.ipynb`: set up a tune directory with model files, compiled mechanisms,
  config templates, and validation.
- `2_passive.ipynb`: passive-parameter tuning workflow.
- `3_active.ipynb`: active-parameter tuning workflow, including optional ACT
  active-tuning workspace support.
- `4_synapses.ipynb`: BMTool-based synapse setup/tuning workflow.
- `5_simulate.ipynb`: primary simulation entry point for local or Colab use.
- `6_analysis.ipynb`: saved-output analysis and comparison workflow.
- `7_tools.ipynb`: optional notebook wrappers for small utility scripts.

Step 5 is the main destination of the pipeline. Earlier steps prepare a cell/tune
for simulation; Step 6 is optional post-processing.

## Optional Notebooks

- `extra_notebooks/act_segmentation.ipynb`: optional ACT-style channel
  segmentation helper. Use it when manually creating segmented modfiles before
  passive/active tuning. It is not required for the numbered workflow.

## Examples

Bundled example tune directories:

- `cells/PV/tunes/adb_peri`: raw ADB perisomatic PV setup example.
- `cells/PV/tunes/seg_tuned`: tuned PV simulation example.
- `cells/SST/tunes/adb_all`: raw ADB all-active SST setup example.
- `cells/SST/tunes/seg_tuned`: tuned SST simulation example.

Each tune uses a `cell_configs/` directory containing:

- `cell_config.json`: cell identity, loader, paths, and tuning metadata.
- `sim_config.json`: simulation timing, saving, plotting, recording, and run options.
- `geometry.json`: segment grouping/distance settings.
- `syn_config.json`: list of enabled synapse-group config files.
- `syn_groups/*.json`: synapse groups and explicit `input_blocks`.

Saved example outputs are not required to use the repo. Generate fresh outputs
with Step 5 when you want to use Step 6 analysis.

## Local and Colab Use

The root notebooks are the current local and Colab entry points:

- `1_setup.ipynb`
- `2_passive.ipynb`
- `3_active.ipynb`
- `4_synapses.ipynb`
- `5_simulate.ipynb`
- `6_analysis.ipynb`
- `7_tools.ipynb`

CLI and SLURM entry points are intended for local/HPC use after the same tune
configs have been prepared.

## Documentation

- `docs/README.md`: documentation index.
- `docs/quickstart.md`: fastest path to run and save an example.
- `docs/installation.md`: environment, external tools, and Colab setup.
- `docs/pipeline_overview.md`: step-by-step workflow map.
- `docs/guides/step_1_setup.md`: Step 1 setup notebook/CLI guide.
- `docs/guides/step_2_passive.md`: Step 2 passive-tuning guide.
- `docs/guides/step_3_active.md`: Step 3 active-tuning guide.
- `docs/guides/step_4_synapses.md`: Step 4 synapse-tuning guide.
- `docs/guides/step_5_simulate.md`: Step 5 simulation guide.
- `docs/guides/analysis.md`: Step 6 analysis guide.
- `docs/guides/step_7_tools.md`: Step 7 utility-notebook guide.
- `docs/reference/configs_reference.md`: current config schema.
- `docs/reference/outputs_layout.md`: saved-output structure.
- `docs/advanced/cli_slurm.md`: CLI and SLURM usage.
- `docs/troubleshooting.md`: common issues.

Contracts in `contracts/` are developer/design references, not the primary user
documentation.
