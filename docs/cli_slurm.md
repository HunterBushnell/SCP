CLI and SLURM

Step 0 Prep CLI (scripts/step0_prepare.py)
- Minimal:
  `python scripts/step0_prepare.py --cell PV --tune seg_tuned --specimen-id 484635029`
- Example (scaffold-only refresh):
  `python scripts/step0_prepare.py --tune-dir cells/PV/tunes/seg_tuned --no-download --no-compile --config-mode fill`

CLI (run_pipeline.py)
- Minimal:
  `python run_pipeline.py --tune-dir <tune_dir> --n-trials 1`

- Auto mode selection:
  If `--mode` is omitted, it is inferred from `n_trials`.

Key flags
- `--mode` (single|multi)
- `--n-trials`
- `--seed`
- `--trial-offset` (align array jobs with sequential runs)
- `--iclamp`
- `--snapshot`
- `--force-save`
- `--output-dir`, `--output-stem`

SLURM (run_slurm.sh)
- Minimal:
  `CELL=SST TUNE=seg_tuned N_TRIALS=10 sbatch run_slurm.sh`

Array modes
- Per-task trials:
  `N_TRIALS=1 sbatch --array=0-9 run_slurm.sh`
- Split total trials across tasks:
  `TOTAL_TRIALS=100 sbatch --array=0-9 run_slurm.sh`

Merge control
- Disable merge:
  `MERGE_ARRAY=0 sbatch --array=0-9 run_slurm.sh`
- Override merged output name:
  `MERGED_STEM=results_custom sbatch --array=0-9 run_slurm.sh`

Common env vars
- `CELL`, `TUNE`, `TUNE_DIR`
- `OUTPUT_DIR`
- `OUTPUT_STEM`
- `N_TRIALS`, `TOTAL_TRIALS`, `MODE`, `SEED`, `BASE_SEED`, `TASKS`
- `BATCH_STEM`, `MERGED_STEM`, `MERGE_PATTERN`, `MERGE_ARRAY`
- `FORCE_SAVE` (default on)
- `ICLAMP`, `SNAPSHOT`

Notes
- `run_slurm.sh` auto-builds modfiles if missing.
- When arrays are merged, per-task outputs go to `output_data/<batch>/parts/` and
  are cleaned up after merge.
- Logs are copied into the run folder when possible.
- The pipeline prefers `cell_configs/` when resolving `sim_config.json` and `syn_config.json`.
