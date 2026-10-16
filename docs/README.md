# SCP Documentation

[`0_pipeline.ipynb`](../0_pipeline.ipynb) is the recommended simple workflow.
Choose **Run All** once to render its five panels, then run each enabled step.
Use the numbered notebooks when you need full setup, tuning, simulation, or
analysis controls.

## Start Here

- [Installation](installation.md): local environment, external repositories,
  and Colab notes.
- [Quickstart](quickstart.md): exact button-driven Steps 1–5 sequence.
- [Pipeline overview](pipeline_overview.md): lifecycle, process boundaries, and
  data flow.
- [Troubleshooting](troubleshooting.md): failure recovery and common errors.

## Step Guides

- [Step 1 setup](guides/step_1_setup.md)
- [Step 2 passive tuning](guides/step_2_passive.md)
- [Step 3 active tuning](guides/step_3_active.md)
- [Step 4 synapse tuning](guides/step_4_synapses.md)
- [Step 5 simulation](guides/step_5_simulate.md)
- [Step 6 analysis](guides/analysis.md) and its [detailed controls
  reference](guides/step_6_analysis.md)
- [Step 7 tools](guides/step_7_tools.md)
- [Steps 1–4 overview](guides/steps_1-4_overview.md)

## Reference

- [Configuration schema](reference/configs_reference.md)
- [Model loaders](reference/model_loaders.md)
- [Target trace formats](reference/target_trace_formats.md)
- [Output layout](reference/outputs_layout.md)
- [Reproducibility](reference/reproducibility.md)
- [Glossary](reference/glossary.md)
- [Naming conventions](reference/naming_conventions.md)
- [Third-party and model-asset notices](../THIRD_PARTY_NOTICES.md)

## Recipes and Advanced Interfaces

- [Project-owned dependency workflow](guides/dependency_workflow.md): editable
  installation, portable compact notebook, interactive registration, and APIs.
- [Custom cells and external models](guides/custom_models.md): live objects,
  Python builders, external snapshots, and copy/reference selection.
- [Project workspaces](guides/project_workspaces.md): separate same-named cells,
  select project tunes, and interpret saved workspace identity.
- [Cell data and public baselines](guides/cell_data.md): cell-owned inputs,
  independent project data, historical aliases, and required dependencies.
- [Synaptic recordings](guides/synaptic_recording.md): draft usage guidance for
  the [paused implementation](project/synaptic_recording_update.md).
- [ACT example catalog](guides/act_examples.md): staged HOC/Allen source
  variants, reference models, initial use, and characterization boundaries.
- [Synthetic walkthrough](../extra_notebooks/custom_cells.ipynb)

- [Example saved run](recipes/example_run.md)
- [Configuration cookbook](recipes/config_cookbook.md)
- [CLI and SLURM](advanced/cli_slurm.md)
- [SLURM status panel](advanced/status_panel.md)
- [Optional micro:bit status bridge](advanced/microbit_status.md)
- [Optional notebooks](../extra_notebooks/README.md)

## Project and Developer References

- [Roadmap](project/roadmap.md)
- [Demo outline](project/demo_outline.md)
- [Public release cleanup](project/public_release_cleanup.md)
- [v0.2.0 release notes](project/release_notes_v0.2.0.md)
- [v0.1.0 release notes](project/release_notes_v0.1.0.md)
- [Design contracts](../contracts/README.md)
- [Setup checker](../scripts/check_setup.py)
- [Notebook checker](../scripts/check_notebooks.py)
