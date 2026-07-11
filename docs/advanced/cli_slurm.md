# CLI and SLURM

## Step 1 Setup CLI

Prepare a raw PV perisomatic tune:

```bash
python scripts/step1_prepare.py \
  --cell PV \
  --tune adb_peri \
  --specimen-id 484635029 \
  --model-type perisomatic
```

Prepare a raw SST all-active tune:

```bash
python scripts/step1_prepare.py \
  --cell SST \
  --tune adb_all \
  --specimen-id 485466109 \
  --model-type "all active"
```

Refresh configs without downloading or compiling:

```bash
python scripts/step1_prepare.py \
  --tune-dir cells/PV/tunes/adb_peri \
  --source-type existing \
  --no-download \
  --no-compile \
  --config-mode fill
```

Useful Step 1 flags:

- `--source-type adb|existing`
- `--no-download`
- `--force-download`
- `--no-compile`
- `--recompile-modfiles`
- `--no-synapse-configs`
- `--synapse-templates input_blocks|none`
- `--config-mode fill|overwrite|skip`
- `--no-validate`

## Step 5 CLI

Run one existing tuned example:

```bash
python run_pipeline.py --tune-dir cells/PV/tunes/seg_tuned --n-trials 1 --force-save
```

Common flags:

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
CELL=SST TUNE=seg_tuned N_TRIALS=10 sbatch run_slurm.sh
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

- `CELL`, `TUNE`, `TUNE_DIR`
- `REPO_ROOT`
- `OUTPUT_DIR`, `OUTPUT_STEM`
- `N_TRIALS`, `TOTAL_TRIALS`, `MODE`
- `SEED`, `BASE_SEED`, `TASKS`
- `BATCH_STEM`, `MERGED_STEM`, `MERGE_PATTERN`, `MERGE_ARRAY`
- `FORCE_SAVE`
- `ICLAMP`, `SNAPSHOT`
- `ROTATE_LOGS`

Notes:

- `run_slurm.sh` auto-builds modfiles if missing.
- Per-task array outputs go under `output_data/<batch>/parts/` before merge.
- Logs are copied into the run folder when possible.
- The pipeline resolves configs from `cell_configs/`.
