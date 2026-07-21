# Single Cell Pipeline (SCP)

SCP is a notebook-first workflow for preparing, tuning, simulating, and analyzing
single-cell NEURON models. The current examples focus on PV and SST cells. A
registered loader interface supports both Allen manifest bundles and object-owned
HOC templates through the same cell-scoped pipeline machinery.

## Quick Start

1. Install the environment:
   - `conda env create -f environment.yml`
   - `conda activate scp-py311`
2. Optional setup check:
   - `python scripts/check_setup.py --steps 1 2 3 4 5 --cell PV --tune tuned --compile-modfiles`
3. Run an example:
   - Simple notebook: open [`0_pipeline.ipynb`](0_pipeline.ipynb), choose
     **Run All** to render its panels, then click Steps 1–5 in order.
   - Detailed simulation notebook: open [`5_simulate.ipynb`](5_simulate.ipynb),
     setting `force_save = True` if you want a saved run.
   - CLI: `python run_pipeline.py --tune-dir cells/PV/tunes/tuned --n-trials 1 --force-save`
4. Analyze saved runs:
   - open `6_analysis.ipynb` after a run has been saved under `output_data/`.

See the [quickstart](docs/quickstart.md) for the shortest runnable path and the
[installation guide](docs/installation.md) for local/Colab setup.

## Pipeline

- [`0_pipeline.ipynb`](0_pipeline.ipynb): recommended compact Steps 1–5 front
  door. **Run All** only renders independent per-step cards; users explicitly
  load, tune/check, optionally initialize BMTool, preview inputs, simulate in a
  fresh process, and plot the saved result. Quiet modes retain full logs, and
  advanced widget values are session-only unless copied into JSON. ACT active
  tuning is experimental, review-only, and not release-blocking.
- [`1_setup.ipynb`](1_setup.ipynb): set up a tune directory with model files, optional compiled
  mechanisms when custom `.mod` sources exist, config templates, and validation.
- [`2_passive.ipynb`](2_passive.ipynb): passive-parameter tuning workflow.
- [`3_active.ipynb`](3_active.ipynb): active-parameter tuning workflow, including optional ACT
  active-tuning workspace support.
- [`4_synapses.ipynb`](4_synapses.ipynb): BMTool-based synapse setup/tuning workflow.
- [`5_simulate.ipynb`](5_simulate.ipynb): detailed simulation workflow.
- [`6_analysis.ipynb`](6_analysis.ipynb): saved-output analysis and comparison workflow.
- [`7_tools.ipynb`](7_tools.ipynb): optional notebook wrappers for small utility scripts.

Use `0_pipeline.ipynb` for the shortest end-to-end route. Use the numbered
notebooks when you need the full setup, optimization, export, placement, or
analysis controls. Its Python settings mapping and widgets stay synchronized,
so common choices can be made either way. Step 5 remains the detailed simulation
destination; Step 6 is optional post-processing.

## Optional Notebooks

- `extra_notebooks/act_segmentation.ipynb`: optional ACT-style channel
  segmentation helper. Use it when manually creating segmented modfiles before
  passive/active tuning. It is not required for the numbered workflow.

## Examples

Bundled example tune directories:

- `cells/PV/tunes/orig`: raw ADB perisomatic PV setup example.
- `cells/PV/tunes/tuned`: tuned PV simulation example.
- `cells/SST/tunes/orig`: raw ADB all-active SST setup example.
- `cells/SST/tunes/tuned`: tuned SST simulation example.

Each tune uses a `cell_configs/` directory containing:

- `cell_config.json`: cell identity, loader, paths, and tuning metadata.
- `sim_config.json`: simulation timing, saving, plotting, recording, and run options.
- `target_config.json`: optional passive, FI, and trace targets used by tuning notebooks.
- `geometry.json`: segment grouping/distance settings.
- `syn_config.json`: list of enabled synapse-group config files.
- `syn_groups/*.json`: synapse groups and explicit `input_blocks`.

Saved example outputs are not required to use the repo. Generate fresh outputs
with Step 5 when you want to use Step 6 analysis.

## Local and Colab Use

The root notebooks are the current local and Colab entry points:

- `0_pipeline.ipynb`
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

- [Documentation index](docs/README.md)
- [Quickstart](docs/quickstart.md)
- [Installation](docs/installation.md)
- [Pipeline overview](docs/pipeline_overview.md)
- [Step guides](docs/guides/steps_1-4_overview.md)
- [Configuration reference](docs/reference/configs_reference.md)
- [Model-loader reference](docs/reference/model_loaders.md)
- [Output layout](docs/reference/outputs_layout.md)
- [CLI and SLURM](docs/advanced/cli_slurm.md)
- [Troubleshooting](docs/troubleshooting.md)

Contracts in `contracts/` are developer/design references, not the primary user
documentation.
