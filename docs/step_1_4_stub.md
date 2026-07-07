Steps 1-4 (Reference)

Step 1 is implemented with `modules` and prepares tune directories for
later steps. Steps 2-4 use updated notebooks with repo-relative path handling,
while still depending on ACT (Steps 2-3) and bmtool (Step 4).

Local vs Colab behavior
- Local Steps 2-4 assume Step 1 already prepared the tune directory and
  compiled mechanisms. They validate bundle contents and load existing DLLs.
- If you need notebook-only bootstrapping, use `colab_notebooks/2_colab.ipynb` / `colab_notebooks/3_colab.ipynb`
  for Steps 2-3. Step 5 uses the unified `5_simulate.ipynb`.

Prerequisites for Steps 1-4
- Run `python scripts/check_setup.py --steps 1 2 3 4 --cell PV --tune seg_tuned`.
- Notebooks depend on external repos:
  - ACT (`../mods/ACT` or `SCP_ACT_PATH`)
  - bmtool for Step 4 (`../mods/bmtool` or `SCP_BMTOOL_PATH`)

1_setup.ipynb / scripts/step1_prepare.py
- Purpose: bootstrap a tune directory so it is ready for later SCP steps.
- Core implementation: `modules/setup/step1_prepare.py`.
- Inputs:
  - Cell/tune target (`cells/<CELL>/tunes/<TUNE>` or explicit `--tune-dir`)
  - Allen model identity (`specimen_id`, `model_type`)
  - Optional toggles for download/compile/scaffold/validation behavior
- Actions:
  - Download/cache Allen bundle files (`manifest.json`, morphology, fit json, `modfiles/`)
  - Compile modfiles (`nrnivmodl`) and optionally load the DLL
  - Scaffold common files under `cell_configs/`
- Outputs:
  - `manifest.json`
  - `.adb_download_meta.json` (download provenance + one-model-per-tune guard)
  - `modfiles/` (+ compiled `x86_64/.libs/libnrnmech.so` when compiled)
  - `cell_configs/cell_config.json`
  - `cell_configs/sim_config.json`
  - `cell_configs/geometry.json`
  - `cell_configs/syn_config.json`
  - `cell_configs/syn_groups/placeholder_off.json`

- archive/1_segment.ipynb (optional)
- Purpose: archived ACT-derived segmentation reference for building geometry descriptions.
- Outputs: `cell_configs/geometry.json`.

2_passive.ipynb
- Purpose: tune passive parameters against target traces.
- Outputs: tuned parameters saved under `cell_configs/` or model files.

3_active.ipynb
- Purpose: tune active parameters (channels).
- Outputs: updated model parameters.

4_synapses.ipynb
- Purpose: tune synaptic parameters and group configs.
- Outputs: `cell_configs/syn_config.json` and `cell_configs/syn_groups/`.
