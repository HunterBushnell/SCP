# SCP Demo Outline

Use this as a 5-15 minute speaking guide for lab/advisor demos. It is not a
full tutorial; the goal is to explain what the repo is, how it is organized, and
where reviewers should focus feedback.

## 1. Purpose

SCP is a notebook-first workflow for preparing, tuning, simulating, and
analyzing single-cell NEURON models.

Key points:

- Designed for lab/class users who may not be comfortable with terminal-heavy
  workflows.
- Works locally and in Google Colab for the main notebook path.
- Keeps user-facing steps simple while moving most pipeline machinery into
  backend modules.
- Uses tune-local JSON configs so models, targets, synapses, simulations, and
  analysis options are reproducible and easy to inspect.

## 2. Repo Layout

Show these top-level pieces:

- `1_setup.ipynb` through `7_tools.ipynb`: numbered notebook workflow.
- `cells/<CELL>/tunes/<TUNE>/`: self-contained model/tune directories.
- `cell_configs/`: model, target, simulation, geometry, and synapse configs.
- `modules/`: backend code used by notebooks, CLI, and SLURM entry points.
- `docs/`: user guides, references, recipes, troubleshooting, and roadmap.
- `extra_notebooks/`: optional non-core helpers such as ACT segmentation.

Recommended example to show:

```text
cells/PV/tunes/tuned/
```

## 3. Main Workflow

### Step 1: Setup

Purpose:

- Download or stage a cell model.
- Compile mechanisms.
- Generate config templates.
- Set up target data mode.
- Validate the tune folder.

Point out:

- `orig` is the raw/original tune.
- `tuned` is the working tune used through later steps.
- `tuned_adb` is prepared for Allen/ADB NWB-based targets, but NWB files are not
  tracked in Git.

### Step 2: Passive Tuning

Purpose:

- Load passive targets from `target_config.json`.
- Optionally extract passive targets from traces or Allen/ADB NWB files.
- Use ACT passive machinery to calculate suggested passive values.
- Keep final model edits manual for clarity and loader-specific control.

### Step 3: Active Tuning

Purpose:

- Run active current-injection checks.
- Compare FI behavior to target FI curves.
- Plot traces and FI curves for manual active-parameter tuning.
- Provide an ACT active auto-tuning workspace path.

Current caveat:

- Step 3.4 ACT active auto-tuning is exposed, but deeper workflow validation is
  still pending ACT-side review.

### Step 4: Synapses

Purpose:

- Use BMTool through SCP adapters.
- Tune chemical synapse parameters.
- Save selected synapse tuning options in `synapse_tuning_config.json`.
- Export synapse groups for simulations.

Point out:

- Users can bring their own `.mod` synapse mechanisms.
- Suggested mechanisms are documented from the CyNeuro mechanisms library.

### Step 5: Simulate

Purpose:

- Run single or multi-trial simulations from the notebook.
- Save outputs in organized run folders.
- Provide notebook, CLI, and SLURM entry points for the same config-driven
  simulation pipeline.

Useful examples:

```bash
python run_pipeline.py --tune-dir cells/PV/tunes/tuned --n-trials 1 --force-save
```

### Step 6: Analysis

Purpose:

- Load saved outputs.
- Generate single-plot summaries.
- Use output/input plotting UIs for deeper analysis.
- Compare saved runs and biological/reference curves.
- Export plotted data and figures when needed.

Recommended live demo:

- Open `6_analysis.ipynb`.
- Select the bundled PV or SST saved example output.
- Show the single plot and one UI section rather than running a new simulation.

### Step 7: Tools

Purpose:

- Notebook wrappers for utility scripts.
- Useful for users who prefer notebooks over terminal commands.

Current caveat:

- Tool bootstrap has been checked, but individual utilities are tested as
  needed.

## 4. Config Philosophy

Explain the core design:

- Notebooks expose only common controls.
- JSON files hold reproducible model/simulation/analysis settings.
- Backend modules handle loader-specific and simulation-specific details.
- The same tune configs should work from notebooks, CLI, and SLURM.

Important config files:

- `cell_config.json`: cell identity, loader, paths, and display metadata.
- `target_config.json`: passive/FI/trace target source.
- `sim_config.json`: simulation timing, saving, plotting, recording, and run
  options.
- `syn_config.json` and `syn_groups/*.json`: enabled synapse groups and input
  blocks.
- `geometry.json`: section grouping and distance settings.

## 5. Validated Scope

Current status:

- Main notebook workflow has been validated through Step 6 in Colab.
- PV/SST example simulation paths have been validated locally and in Colab.
- CLI and SLURM simulation entry points have been checked on the current
  examples.
- ACT active auto-tuning remains the major intentionally pending validation
  item.

## 6. Feedback To Request

Ask reviewers to focus on:

- Are the notebook steps understandable?
- Are the markdown instructions too sparse or too detailed?
- Are the config files clear enough to edit?
- Is the Colab/local workflow practical for lab and class use?
- Are the examples useful and not too large or slow?
- What would make this easier for a new model or new project?

## 7. Suggested Closing

SCP is currently a public-preview candidate for `v0.1.0`. The core workflow is
usable, but lab feedback should guide final polish before creating a formal
GitHub release/tag.
