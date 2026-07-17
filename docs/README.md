# SCP Documentation

Use this directory as the user-facing documentation index. The numbered notebooks
are the primary workflow; docs explain what each notebook expects, writes, and
how it connects to CLI/SLURM tooling.

## Start Here

- `installation.md`: local environment, external repositories, and Colab notes.
- `quickstart.md`: shortest path to run and save a bundled example.
- `pipeline_overview.md`: full workflow map from setup through analysis.
- `troubleshooting.md`: common setup and runtime issues.

## Step Guides

- `guides/step_1_setup.md`: tune-directory setup, ADB download, target-data staging, config scaffolding, and validation.
- `guides/step_2_passive.md`: passive tuning, ACT target conversion, optional Allen/ADB NWB passive targets, manual model edits, and trace checks.
- `guides/step_3_active.md`: active tuning, optional ACT workspace/CLI integration, Allen/ADB NWB FI targets, manual sweeps, and FI checks.
- `guides/step_4_synapses.md`: synapse tuning with BMTool and SCP synapse-config export.
- `guides/step_5_simulate.md`: simulation notebook, CLI, IClamp, saving, plotting, and outputs.
- `guides/analysis.md`: primary guide for `6_analysis.ipynb`.
- `guides/step_6_analysis.md`: detailed Step 6 defaults/options reference.
- `guides/step_7_tools.md`: optional notebook wrappers for small utility scripts.
- `guides/steps_1-4_overview.md`: compact bridge for early setup/tuning steps.

## Reference

- `reference/configs_reference.md`: current `cell_configs/` schema.
- `reference/model_loaders.md`: loader registry, canonical sections, HOC-template contract, and model artifacts.
- `reference/target_trace_formats.md`: passive trace, active trace, and FI CSV target data contracts.
- `reference/outputs_layout.md`: run folders, manifests, sidecars, plots, and array outputs.
- `reference/reproducibility.md`: seeds, snapshots, append behavior, and array alignment.
- `reference/glossary.md`: common SCP terms.
- `reference/naming_conventions.md`: lightweight naming guidance.

## Recipes

- `recipes/example_run.md`: minimal saved-run example.
- `recipes/config_cookbook.md`: common config edits and recipes.

## Advanced Interfaces

- `advanced/cli_slurm.md`: `scripts/step1_prepare.py`, `run_pipeline.py`, and `run_slurm.sh`.
- `advanced/status_panel.md`: terminal monitor for SLURM status files.
- `advanced/microbit_status.md`: optional hardware status bridge.
- `../extra_notebooks/README.md`: optional non-pipeline notebooks.

## Project

- `project/roadmap.md`: deferred and planned improvements.
- `project/demo_outline.md`: short speaking guide for lab/advisor demos.
- `project/release_notes_v0.1.0.md`: initial public-preview release notes.

## Developer References

- `../contracts/README.md`: developer design contracts and implementation notes.
- `../scripts/check_setup.py`: environment/workspace readiness checks.
- `../scripts/check_notebooks.py`: notebook portability checks.
