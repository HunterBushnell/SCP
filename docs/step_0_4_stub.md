Steps 0-4 (Transition Status)

Step 0 is now updated for the `modules_local` pipeline. Steps 1-4 are still
legacy/stub and will be migrated next.

0_download.ipynb / scripts/step0_prepare.py
- Purpose: bootstrap a tune directory so it is ready for Steps 1-6.
- Core implementation: `modules_local/step0_prepare.py`.
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
  - `modfiles/` (+ compiled `x86_64/.libs/libnrnmech.so` when compiled)
  - `cell_configs/cell_config.json`
  - `cell_configs/sim_config.json`
  - `cell_configs/geometry.json`
  - `cell_configs/syn_config.json`
  - `cell_configs/syn_groups/placeholder_off.json`

1_segment.ipynb
- Purpose: segment the morphology and build a geometry description.
- Outputs (expected): `cell_configs/geometry.json`.

2_passive.ipynb
- Purpose: tune passive parameters against target traces.
- Outputs (expected): tuned parameters saved under `cell_configs/` or model files.

3_active.ipynb
- Purpose: tune active parameters (channels).
- Outputs (expected): updated model parameters.

4_synapses.ipynb
- Purpose: tune synaptic parameters and group configs.
- Outputs (expected): `cell_configs/syn_config.json` and `cell_configs/syn_groups/`.
