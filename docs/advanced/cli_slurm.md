# CLI and SLURM

## Installed project-owned entry points

Use `scp-pipeline setup`, `run`, or `check` (also `python -m scp_pipeline`).
`--project-root PATH` uses project-local saved selections; `--workspace-root`
overrides the cell/tune shorthand root. Explicit `--tune-dir` remains
current-directory-relative and takes precedence. This differs from `--project`,
which selects a namespace within an SCP workspace. See the
[dependency workflow](../guides/dependency_workflow.md). SLURM can use the same
environment and an absolute `TUNE_DIR`; notebook objects are never serialized.

## Project selection

Add `--project <name>` to cell/tune shorthand to select
`projects/<project>/cells/<cell>/tunes/<tune>/`. Without it, shorthand retains
the existing `cells/` layout. `--tune-dir` takes precedence and accepts either
layout or another explicitly selected workspace. Relative CLI paths are
resolved from the invocation directory.

For an already prepared synthetic project tune, run from the SCP root:

```bash
python run_pipeline.py --project study_a --cell ExampleCell --tune tuned \
  --iclamp --n-trials 1 --force-save
```

For SLURM, use `PROJECT=study_a CELL=ExampleCell TUNE=tuned`; an explicit
`TUNE_DIR` wins over all shorthand settings. Prefer an absolute `TUNE_DIR` when
submitting from another directory. The existing wrapper's no-selection SST
default is unchanged. See [Project workspaces](../guides/project_workspaces.md)
for setup, Git visibility, and saved namespace identity.

## Step 1 Setup CLI

Prepare a raw PV perisomatic tune:

```bash
python scripts/step1_prepare.py \
  --cell PV \
  --tune orig \
  --specimen-id 484635029 \
  --model-type perisomatic
```

Prepare a raw SST all-active tune:

```bash
python scripts/step1_prepare.py \
  --cell SST \
  --tune orig \
  --specimen-id 485466109 \
  --model-type "all active"
```

Refresh configs without downloading or compiling:

```bash
python scripts/step1_prepare.py \
  --tune-dir cells/PV/tunes/orig \
  --source-type existing \
  --no-download \
  --no-compile \
  --config-mode fill
```

`--no-compile` skips `nrnivmodl` but still loads an existing tune-local
mechanism library so validation is safe. If no compiled library exists, run
once without `--no-compile`; use `--no-validate --no-load-dll` only when you
intentionally want a config-only refresh without constructing the cell.

For an existing HOC-template model, select `--cell-loader hoc_template` and
provide `--hoc-template-file`, `--hoc-template-name`, optional constructor and
section-map JSON, plus explicit `--v-init-mv` and `--celsius-c` conditions. See
[`../guides/step_1_setup.md`](../guides/step_1_setup.md) for the full generic
scaffolding example and contract.

Useful Step 1 flags:

- `--project`, `--cell`, `--tune`, or `--tune-dir`
- `--source-type adb|existing`
- `--cell-loader allen_manifest|hoc_template|python_factory`
- `--hoc-template-file`, `--hoc-template-name`
- `--hoc-constructor-args`, `--hoc-section-map`
- `--v-init-mv`, `--celsius-c`
- `--target-source-mode none|manual|traces|allen_nwb`
- `--include-manual-with-file-source`
- `--no-download`
- `--force-download`
- `--no-compile`
- `--recompile-modfiles`
- `--no-target-config`
- `--no-synapse-configs`
- `--synapse-templates input_blocks|none`
- `--config-mode fill|overwrite|skip`
- `--no-validate`

## Step 5 CLI

Run one existing tuned example:

```bash
python run_pipeline.py --tune-dir cells/PV/tunes/tuned --n-trials 1 --force-save
```

Common flags:

- `--project`, `--cell`, `--tune`, or `--tune-dir`
- `--mode single|multi`
- `--n-trials`
- `--seed`
- `--trial-offset`
- `--iclamp`
- `--snapshot`
- `--force-save`
- `--output-dir`
- `--output-stem`

If `--mode` is omitted, it is inferred from `--n-trials`.

## SLURM

Minimal batch run:

```bash
CELL=SST TUNE=tuned N_TRIALS=10 sbatch run_slurm.sh
```

Array with one trial per task:

```bash
N_TRIALS=1 sbatch --array=0-9 run_slurm.sh
```

Split total trials across tasks:

```bash
TOTAL_TRIALS=100 sbatch --array=0-9 run_slurm.sh
```

Merge controls:

```bash
MERGE_ARRAY=0 sbatch --array=0-9 run_slurm.sh
MERGED_STEM=results_custom sbatch --array=0-9 run_slurm.sh
```

Common environment variables:

- `PROJECT`, `CELL`, `TUNE`, `TUNE_DIR`
- `REPO_ROOT`
- `OUTPUT_DIR`, `OUTPUT_STEM`
- `N_TRIALS`, `TOTAL_TRIALS`, `MODE`
- `SEED`, `BASE_SEED`, `TASKS`
- `BATCH_STEM`, `MERGED_STEM`, `MERGE_PATTERN`, `MERGE_ARRAY`
- `FORCE_SAVE`
- `ICLAMP`, `SNAPSHOT`
- `ROTATE_LOGS`

Notes:

- `run_slurm.sh` resolves the configured MOD source directory and auto-builds it
  when `.mod` files exist and the compiled library is missing; built-in-only
  models skip compilation. It does not force-recompile an existing library.
  After changing `.mod` sources, rebuild once through Step 1 before submitting
  a job, rather than rebuilding concurrently in array tasks.
- Per-task array outputs go under `output_data/<batch>/parts/` before merge.
- Logs are copied into the run folder when possible.
- The pipeline resolves configs from `cell_configs/`.

## Custom Python cells and external registration

CLI/SLURM reconstruct configured Python builders through the common loader.
They reject `runtime_cell` workspaces: register a saved builder instead of
expecting a notebook object to exist in another process. See
[Custom models](../guides/custom_models.md) for full recipes.

Step 1 accepts `--python-factory-file`, `--python-factory-callable`, and
`--loader-config-json` for args/kwargs/section maps. External registration uses
`--model-source-root`, `--model-source-folders`, `--model-source-mode copy|reference`,
and optional `--model-working-directory`. Add `--preview-model-sources` for
an inventory-only operation that does not create the tune or compile anything.
Both modes retain a local snapshot. External MOD builds stay in the tune's
`.scp_build/` cache, not in the original repository.

Step 5 still uses `--tune-dir` for any workspace location and `--iclamp` for
cell-only current injection. Independent workers reconstruct models from disk;
unsaved notebook changes are not inherited. Configure/compile before large
SLURM batches and restart after native source/library changes.
