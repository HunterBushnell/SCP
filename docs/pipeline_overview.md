# Pipeline Overview

SCP is a config-driven, notebook-first workflow with optional CLI and SLURM
entry points. [`0_pipeline.ipynb`](../0_pipeline.ipynb) is the recommended
simple front door; the numbered notebooks expose the full workflow.

## Compact Lifecycle

**Run All** renders five independent widget panels. It does not load a model or
start a simulation.

```text
Step 1                  Steps 2–4                       Step 5
prepare + build once -> reuse one in-kernel cell -> fresh preview/simulation
       |                       |                           |
       v                       v                           v
 validated tune       passive / active / BMTool      saved run manifest
```

1. **Setup and Load** fills missing standard configs, validates the tune,
   compiles/loads mechanisms when needed, and constructs exactly one shared
   tuning cell.
2. **Passive Tuning** keeps the ACT proposal and passive protocol as separate,
   review-only actions.
3. **Active Tuning and FI Curve** keeps active sweeps, FI checks, and
   experimental ACT active tuning independent.
4. **Synapse Tuning** optionally initializes BMTool around the shared cell, then
   exposes Single Event and Interactive Tuner separately.
5. **Check Inputs, Simulate, and Plot** previews inputs and runs the final model
   in fresh processes, loads the saved manifest, and plots compact diagnostics.

Each card owns its controls, status, and output. There are no tabs, accordions,
or global run button. Advanced options expand only within the relevant card.

## Settings and Config Ownership

`pipeline_settings` and the widgets synchronize in both directions. Rerunning
the settings cell refreshes unlocked widgets; widget edits update the mapping.
Model-selection fields lock after Step 1 because the model may only be built
once per kernel.

The compact UI has two kinds of values:

- **File-backed config:** the durable source of truth under
  `cell_configs/*.json` and `cell_configs/syn_groups/*.json`.
- **Session-only override:** a widget or `pipeline_settings` value used for the
  current notebook session. It does not rewrite JSON automatically.

Missing standard configs are scaffolded in fill mode, preserving existing
values and loader metadata. Runtime, target, geometry, and synapse files are
reloaded by the stage that consumes them.

## One-Cell and Restart Rules

Steps 2–4 reuse the exact cell constructed in Step 1. Source fingerprints guard
the in-memory model against silent changes.

Restart the kernel after changing:

- `cell_config.json` or loader-owned model artifacts,
- morphology or fit/HOC sources,
- MOD sources or their mechanism contract.

A restart is not required for runtime, target, geometry, or synapse JSON edits;
rerun the consuming action instead. ACT target or option edits require
**Prepare ACT workspace** again.

## Fresh-Process Boundary

Input preview and final simulation intentionally avoid the shared tuning cell.
They run through subprocess workers so current JSON files are reloaded and
BMTool/current-clamp state cannot leak into the final run. Step 5 always saves a
unique `pipeline_<timestamp>` result unless an explicit output stem is supplied;
the explicit stem takes precedence over the configured stem.

The optional seed is shared: it is displayed by **Check Inputs** and passed to
the final simulation. Quiet modes capture full merged subprocess streams in:

- `pipeline_ui.step1_load_log`
- `pipeline_ui.input_preview_log`
- `pipeline_ui.simulation_log`

Visible outputs remain limited to SCP summaries, requested tables, and plots.

## Tuning Boundaries

Passive ACT values, active ACT predictions, and BMTool results are proposals.
The compact notebook never applies them automatically to fit, HOC, JSON, or MOD
files.

ACT active tuning is experimental, review-only, and not release-blocking. Its
optimization/evaluation workers are isolated and cancellable. The complete
tune-local `act_workspace/` is generated, machine-specific state ignored by
Git. Use [`3_active.ipynb`](../3_active.ipynb) for detailed ACT setup and
inspection.

## Detailed Flow

Use the numbered notebooks when a stage needs more control:

1. [`1_setup.ipynb`](../1_setup.ipynb): download/stage models, configure
   loaders and targets, compile mechanisms, and validate.
2. [`2_passive.ipynb`](../2_passive.ipynb): detailed passive targets,
   proposals, protocols, and manual tuning checks.
3. [`3_active.ipynb`](../3_active.ipynb): detailed active sweeps, FI targets,
   and ACT workspace controls.
4. [`4_synapses.ipynb`](../4_synapses.ipynb): BMTool setup, placement,
   response, optimization, and exports.
5. [`5_simulate.ipynb`](../5_simulate.ipynb): detailed simulation setup and
   execution.
6. [`6_analysis.ipynb`](../6_analysis.ipynb): saved-run analysis,
   comparisons, metrics, styling, and exports.
7. [`7_tools.ipynb`](../7_tools.ipynb): optional maintenance and conversion
   tools.

See the [step guides](README.md#step-guides) for controls and limitations.

## Data Flow

Inputs:

- loader-owned model sources such as `manifest.json` or a HOC template,
- optional tune-local MOD sources,
- `cells/<CELL>/tunes/<TUNE>/cell_configs/`,
- optional external target/input data.

Durable simulation outputs:

```text
cells/<CELL>/tunes/<TUNE>/output_data/<RUN>/
```

`run_manifest.json` indexes saved arrays, config sidecars, plots, and model
provenance. Notebook diagnostics may use `notebook_exports/`; experimental ACT
uses ignored `act_workspace/` state. See the [output layout](reference/outputs_layout.md).

## CLI and SLURM

The CLI and SLURM paths consume the same prepared tune configs:

```bash
python run_pipeline.py --tune-dir cells/PV/tunes/tuned --n-trials 1 --force-save
CELL=SST TUNE=tuned N_TRIALS=10 sbatch run_slurm.sh
```

See [CLI and SLURM](advanced/cli_slurm.md).
