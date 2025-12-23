PV-SST single_cells workflow

**0. Download Cell**
- Notebook: `0_download_cell_ADB.ipynb`
- Purpose: download the model and generate required manifests.

**1. Segment Cell**
- Notebook: `1_segment_cell_ADB.ipynb`
- Purpose: segment and prepare morphology.

**2. Passive Tuning**
- Notebook: `2_cell_passive_tuner_ADB.ipynb`
- Purpose: passive parameter tuning.

**3. Active Tuning**
- Notebook: `3_cell_active_tuner_ADB.ipynb`
- Purpose: active parameter tuning.

**4. Synaptic Tuning**
- Notebook: `4_synaptic_tuner_ADB.ipynb`
- Purpose: tune synapse parameters.

**5. Local Pipeline / Manual Tuning**
- Notebooks: `5_local.ipynb`, `5_analysis.ipynb`, `5_0_gen_inputs.ipynb`, `5_colab.ipynb`
- Purpose: `5_local` runs the sim + quick sanity plots; `5_analysis` holds deeper plotting/analysis.

**Sim config conventions**
- `sim_config.json` is kept flat for now, but grouped by section with blank lines.
- Recommended order:
  1) identity (cell/tune/specimen), 2) trials/timing, 3) output + save/plots,
  4) param_study, 5) randomness.
- PV has the organized layout; copy it to other cells/tunes when ready.
- Optional: `randomness_mode` can be set to `fixed`, `derived`, or `random` to
  auto-fill the detailed `randomness` block for simpler usage.
  - `fixed`: identical trials (use a fixed seed for reproducible runs).
  - `derived`: varies per trial, reproducible if a seed is set.
  - `random`: fully random per trial (fresh entropy each run).

**6. Batch / Slurm Runs**
- Entry points:
  - `run_slurm.sh`: SLURM wrapper with defaults and log rotation.
  - `run_pipeline.py`: direct CLI runner (single or multi).
- Required inputs (tune directory):
  - `sim_config.json`
  - `syn_config.json`
  - `manifest.json`
- Compiled mechanisms under `modfiles/x86_64/` (auto-built by `run_slurm.sh` if missing).
- Slurm runner (recommended for batch runs):
```
cd /home/hrbncv/PV-SST/single_cells
CELL=SST2 TUNE=seg_tuned N_TRIALS=10 sbatch run_slurm.sh
```

- SLURM modes (what they do and how to call them):
  - **Single job (no array)**: one job runs all trials in one process.
    ```
    CELL=SST2 TUNE=seg_tuned N_TRIALS=10 sbatch run_slurm.sh
    ```
  - **Array (parallel tasks, per-task trials)**: each task runs `N_TRIALS` trials.
    ```
    N_TRIALS=1 sbatch --array=0-9 run_slurm.sh
    ```
  - **Chunked array (split total trials across tasks)**: divide `TOTAL_TRIALS`
    across the array tasks; first tasks take the remainder.
    ```
    TOTAL_TRIALS=100 sbatch --array=0-9 run_slurm.sh
    ```
  - **Non-merged array**: keep each task’s results separate.
    ```
    MERGE_ARRAY=0 sbatch --array=0-9 run_slurm.sh
    ```
  - Optional overrides (env vars):
    - `CELL`, `TUNE` (used to build `TUNE_DIR`)
    - `TUNE_DIR` (explicit path, overrides `CELL`/`TUNE`)
    - `OUTPUT_DIR` (default: `${TUNE_DIR}/output_data`)
    - `BATCH_STEM` (array jobs only; groups outputs under `output_data/<BATCH_STEM>`; default `slurm_<jobid>` when merged, or `slurm_<jobid>_<taskid>` when not merged)
    - `MODE` (`single` or `multi`; leave empty to auto-pick)
    - `N_TRIALS` (overrides `sim_config.json`)
    - `SEED` (sets randomness base seed)
    - `FORCE_SAVE=1` (default) forces saving even if `save` is disabled in config
- Example with explicit tune dir:
```
TUNE_DIR=/home/hrbncv/PV-SST/single_cells/cells/PV/tunes/seg_tuned \
N_TRIALS=100 sbatch run_slurm.sh
```
- Slurm array (parallel trials, simplest parallel option):
```
sbatch --array=0-9 run_slurm.sh
```
  - Each task runs `--n-trials 1` unless you override `N_TRIALS`.
  - Array runs are grouped under a batch folder:
    `output_data/<BATCH_STEM>/` (defaults to `slurm_<jobid>` when merged).
  - If `BASE_SEED` is set, `SEED = BASE_SEED + SLURM_ARRAY_TASK_ID`.
  - By default, a merge job is submitted after the array completes.
  - Disable merge: `MERGE_ARRAY=0 sbatch --array=0-99 run_slurm.sh`
  - Merged arrays use a temporary `parts/` folder for per-task outputs and delete it after merge.
- Chunked array (cap tasks, split total trials):
```
TOTAL_TRIALS=10 sbatch --array=0-9 run_slurm.sh
```
  - Array size (`0-9`) controls number of tasks.
  - Trials are split evenly; the first tasks take the remainder.
  - If a task gets 0 trials, it exits immediately.
  - `run_slurm.sh` passes a `--trial-offset` so RNG streams line up with a sequential multi-trial run.
- Direct CLI runner:
```
python /home/hrbncv/PV-SST/single_cells/run_pipeline.py \
  --tune-dir /home/hrbncv/PV-SST/single_cells/cells/SST2/tunes/seg_tuned \
  --n-trials 10 \
  --seed 123
```
- Current injection test (cell-only, no synapses/inputs):
  - Set `sim_config.json -> iclamp.enabled: true` to run it.
  - CLI override: add `--iclamp` (alias `--current-injection`).
  - Slurm override: set `ICLAMP=1` in the environment.
  - Slurm default forces saving; set `FORCE_SAVE=0` to respect config.
  - Uses `sim_config.json -> iclamp` block for amp/delay/dur settings.
  - If `--mode` is omitted, it is inferred from `n_trials`.
- Snapshot mode (full debug capture for diffing notebook vs SLURM):
  - Set `sim_config.json -> snapshot.enabled: true` or pass `--snapshot`.
  - Slurm override: set `SNAPSHOT=1` in the environment.
  - Forces full saving: all inputs, all traces, synapse records per trial, sidecars + full results.
- Outputs and logs:
  - Run folder (single job): `{tune_dir}/output_data/<output_stem>/`
  - Run folder (array job, merged): `{tune_dir}/output_data/<BATCH_STEM>/results/`
  - Run folder (array job, no merge): `{tune_dir}/output_data/<BATCH_STEM>/results/`
  - Manifest + sidecars: `run_manifest.json`, `sim_cfg.json`, `meta.json`, `spikes.npz`, `traces.npz`, etc.
  - Optional plots: `plots/` when `save_plots: true` (e.g., `output_plot.png`, `inputs_mean.png`).
  - Full results bundle defaults to off (set `save_full_results: true` to write `{cell}_{tune}_{output_stem}.pkl`).
  - Save/load/append tuple form (optional):
    - `load: [enabled, path]` (e.g., `[true, "slurm_2025_01_01"]`)
    - `save: [enabled, stem, format, full_results]` (e.g., `[true, "nb", "pkl", true]`)
      - 4th entry controls `save_full_results` (single-file bundle).
      - If `full_results` is true and `save_sidecars` is not set, sidecars are disabled by default.
    - `append: [enabled, path]` (e.g., `[false, null]`)
    - Back-compat: `output` and `append_to` are still accepted.
  - Input/trace saving:
    - `n_traces_to_save` (int) controls how many voltage traces are stored.
    - `n_inputs_to_save` (int or `"all"`) controls how many input trials are stored.
  - Plot profiles (optional):
    - `plots_profile: "off" | "basic" | "inputs" | "full"` sets defaults for plot saving flags.
  - Optional append-on-save: set `append` in `sim_config.json` to a results file or folder.
    - Relative paths resolve under `{tune_dir}/output_data/` (e.g., `append: "nb_vs_slurm_test"`).
    - `append` can be a run folder, `run_manifest.json`, or a `.pkl`/`.npz` file; missing targets are created.
    - If `append` points to an existing run, `run_pipeline.py` uses that run’s `sim_cfg` for consistency.
  - Logs (single job): `{tune_dir}/output_data/<output_stem>/logs/`
  - Logs (array job): `{tune_dir}/output_data/<BATCH_STEM>/logs/` (all task logs + merge logs).
  - Log rotation (fallback): older logs may be moved to `PV-SST/single_cells/logs/old/<timestamp>/`.
- Precomputed inputs:
  - `source.path` can point to a file (`.pkl/.json`), a run folder, or a `run_manifest.json`.
  - If a run folder is used, `inputs_sample.pkl` is preferred; otherwise a legacy `<folder>.pkl`/`.json` inside the folder is used.
- Results schema:
  - See `PV-SST/single_cells/contracts/pvsst_results_outputs.contract.v1.md`.
