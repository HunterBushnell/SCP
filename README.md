# Single Cell Pipeline (SCP)

Developed in the Neural Engineering Laboratory at the University of Missouri
(Mizzou), with the project hosted through [CyNeuro](https://github.com/cyneuro).

Current checkpoint: **v0.3.0**. See the
[release notes](docs/project/release_notes_v0.3.0.md) for the dependency,
portable-notebook, and model/workspace improvements since v0.2.0.

SCP is a notebook-first workflow for preparing, tuning, simulating, and analyzing
single-cell NEURON models. The tuned PV cell remains the primary example and
the default in detailed notebooks and CLI checks. The compact notebook restores
saved navigation or asks you to choose a workspace. A registered loader interface supports Allen
manifest bundles, object-owned HOC templates, and explicit Python builders through the same cell-scoped
pipeline machinery. PN and SST remain legacy worked examples, alongside five
new ACT-sourced families staged for later characterization and manual tuning.

You can also adopt a cell created in a notebook or register external native
sources with copy/reference selection and isolated mechanism builds. See
[Custom cells and external models](docs/guides/custom_models.md) and the
[synthetic walkthrough](extra_notebooks/custom_cells.ipynb). Live cells stay
in their notebook process; saved builders enable CLI/SLURM reconstruction.

Keep bundled examples under `cells/`. For separate projects, use
`projects/<project>/cells/<cell>/tunes/<tune>/` so identically named cells can
have independent workspaces. This is additive: existing paths and PV defaults
do not change. See [Project workspaces](docs/guides/project_workspaces.md).

## Quick Start

Run these examples from the root of your cloned **SCP checkout** (the folder
containing `environment.yml`). In `pip install -e .`, the dot means the terminal's
current directory, which must be SCP—not the project where you keep your notebook.

1. Install the environment, then register SCP itself:
   - `conda env create -f environment.yml`
   - `conda activate scp-py311`
   - `python -m pip install -e .`
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

### Work inside your own project

If your terminal is already in another project, install SCP using its full path:

```bash
python -m pip install -e "/path/to/SCP"
```

Replace `/path/to/SCP` with the location of **your SCP checkout**, not your
working project. This command works from any terminal directory. The environment
YAML installs dependencies; this separate pip command makes SCP importable.
Install once per environment, then copy `0_pipeline.ipynb` into your project.
Its interactive setup can register models or open project-owned workspaces;
configs, snapshots, builds, and results stay with your project. Import tools as
`scp_pipeline`, or use `scp-pipeline setup|run|check`. See the
[dependency workflow](docs/guides/dependency_workflow.md). Detailed notebooks
stay in SCP and can point to the same external tune.

## Pipeline

- [`0_pipeline.ipynb`](0_pipeline.ipynb): recommended compact Steps 1–5 front
  door. **Run All** only renders independent per-step cards; users explicitly
  load, tune/check, optionally initialize BMTool, preview inputs, simulate in a
  fresh process (or in-kernel for an explicit live-cell override), and plot the
  saved result. Quiet modes retain full logs, and
  **Cell setup** separates first-time model setup from everyday cell selection.
  **Save notebook defaults** remembers panel choices in a project-local preset;
  unsaved edits remain session-only and never rewrite model configs. ACT active
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

ACT and BMTool remain optional. For the suggested full notebook environment,
install SCP with `python -m pip install -e '.[act,bmtool]' -c requirements.txt`.
Both tools are then available across projects without checkout paths. Existing
environments can add them separately with `python -m pip install -r requirements-act.txt`
or `python -m pip install -r requirements-bmtool.txt`. ACT installs from a pinned
source revision; do not use an unqualified `pip install act`.
SCP prefers installed packages; cloning either tool requires explicit opt-in. See the
[installation guide](docs/installation.md) for optional-tool setup and overrides.

The lab-use workflow is supported, but setup UI polish is ongoing. Setup text
feedback has real-kernel regression coverage and was confirmed working in the
reported VS Code case after a window reload. This is not broad browser coverage;
see [troubleshooting](docs/troubleshooting.md#compact-notebook-action-fails) if
controls stop updating. Already registered cells do not need rebuilding.

## Optional Notebooks

- [`extra_notebooks/custom_cells.ipynb`](extra_notebooks/custom_cells.ipynb):
  synthetic live-cell, source-copy, and fresh-process Python-builder walkthrough.

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

ACT-sourced native examples:

- [ACT Spiker](cells/ACT_Spiker/README.md), [BursterCa](cells/ACT_BursterCa/README.md),
  and [LAA](cells/ACT_LAA/README.md): HOC starting models, editable working
  copies, and separate known-reference models.
- [ACT L5](cells/ACT_L5/README.md) and [ACT SST](cells/ACT_SST/README.md):
  original Allen bundles and matching ACT tuned variants.

These are **staged and smoke-tested**, not completed SCP walkthroughs or
independently biologically validated examples. Their native sources are pinned
and self-contained; ACT's optimizer/package is not required. See the
[ACT example catalog](docs/guides/act_examples.md) for source variants,
conditions, initial use, and fresh-process requirements.

Research workspaces can live privately under project namespaces. The earlier
LUT HOC examples have moved to the locally excluded LUT project; A1 uses short
project-local cell names. Private project files are not supplied by a fresh
public clone. PV/SST project copies are independent from the temporarily
retained public examples; existing public data and defaults are unchanged.

The public PV/SST defaults use their saved **synaptic baseline** settings while
retaining their tuned intrinsic cells. Input/reference CSVs live with their
cells; private bV1 data remains independent. See [Cell data and baselines](docs/guides/cell_data.md).

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

## Project and Contact

SCP is a project of the Neural Engineering Laboratory at the University of
Missouri (Mizzou), [CyNeuro](https://github.com/cyneuro). Hunter Bushnell is
SCP's original creator and SCP contact: [hrbncv@missouri.edu](mailto:hrbncv@missouri.edu).

The canonical repository is [cyneuro/SCP](https://github.com/cyneuro/SCP);
report reproducible bugs and feature requests through its
[issue tracker](https://github.com/cyneuro/SCP/issues).

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
- [Project workspaces](docs/guides/project_workspaces.md)
- [Model-loader reference](docs/reference/model_loaders.md)
- [Output layout](docs/reference/outputs_layout.md)
- [CLI and SLURM](docs/advanced/cli_slurm.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Repository transfer checklist](docs/project/repository_transfer.md)
- [v0.3.0 release notes](docs/project/release_notes_v0.3.0.md)
- [v0.2.0 release notes](docs/project/release_notes_v0.2.0.md)

Contracts in `contracts/` are developer/design references, not the primary user
documentation.
