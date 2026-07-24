# Single Cell Pipeline (SCP)

SCP is a notebook-first workflow for preparing, tuning, simulating, and analyzing
single-cell NEURON models. The tuned PV cell remains the primary example and
public notebook default. A registered loader interface supports both Allen
manifest bundles and object-owned HOC templates through the same cell-scoped
pipeline machinery; bundled examples also include PN, SST, EUSmn, HYPO, and
PGN cells.

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

ACT and BMTool remain optional. SCP checks for or installs a fresh external
checkout only when an ACT proposal/optimization or BMTool initialization is
actually requested, locally or in Colab; normal setup, protocol, and simulation
actions do not download them. See the
[installation guide](docs/installation.md#external-repositories) for paths and
opt-out controls.

## Optional Notebooks

- `extra_notebooks/act_segmentation.ipynb`: optional ACT-style channel
  segmentation helper. Use it when manually creating segmented modfiles before
  passive/active tuning. It is not required for the numbered workflow.

## Examples

The primary worked example is:

- `cells/PV/tunes/tuned`: tuned Allen-manifest PV simulation example and the
  default selected by public notebooks and CLI checks.

Allen-manifest example families:

- `cells/PV/`: perisomatic PV example.
- `cells/SST/`: all-active SST example.
- `cells/PN/`: all-active projection-neuron example.

These families use `orig`, `tuned`, and prepared `tuned_adb` tune directories.
The `tuned_adb` variants are scaffolds for tune-local Allen/ADB target data;
downloaded NWB files are not tracked.

Object-owned HOC-template example families:

- [`cells/EUSmn/`](cells/EUSmn/README.md): single-compartment EUS
  motor-neuron example derived from the LUT PUD starting template.
- [`cells/HYPO/`](cells/HYPO/README.md): single-compartment sympathetic
  preganglionic-neuron example.
- [`cells/PGN/`](cells/PGN/README.md): single-compartment parasympathetic
  preganglionic-neuron example.

Each HOC family contains an `orig` source tune and a manually tuned derivative.
They demonstrate the generic loader and manual SCP workflow; their inclusion is
not an independent biological-validation claim.

Each tune uses a `cell_configs/` directory. Core and optional files include:

- `cell_config.json`: cell identity, loader, paths, and tuning metadata.
- `sim_config.json`: simulation timing, saving, plotting, recording, and run options.
- `target_config.json`: optional passive, FI, and trace targets used by tuning notebooks.
- `geometry.json`: segment grouping/distance settings.
- `syn_config.json`: optional list of enabled synapse-group config files.
- `syn_groups/*.json`: optional synapse groups and explicit `input_blocks`.

Cell-only/IClamp tunes do not require synapse configuration.

Curated PV and SST saved outputs support analysis demonstrations but are not
required to use the repo. Generate fresh outputs with Step 5 when you want to
use Step 6 analysis.

## Licensing

SCP-authored source code and documentation are available under the root
[MIT license](LICENSE). Bundled model, morphology, and mechanism assets may
have different terms or unresolved upstream licensing; see
[Third-Party and Model-Asset Notices](THIRD_PARTY_NOTICES.md). The root MIT
license does not relicense the assets identified there.

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
- [v0.2.0 release notes](docs/project/release_notes_v0.2.0.md)

Contracts in `contracts/` are developer/design references, not the primary user
documentation.
