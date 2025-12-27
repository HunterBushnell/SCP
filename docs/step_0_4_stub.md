Steps 0-4 (Stub)

These steps will be updated after the PV/SST paper to align with the current
config layout in `cell_configs/`. The notes below are placeholders.

0_download.ipynb
- Purpose: download an AllenDB biophysical bundle into `cells/<CELL>/...`.
- Inputs: `specimen_id`, `model_type`, `tunes_dir`, optional `model_dir`.
- Outputs (expected): Allen bundle files including `manifest.json`, morphologies,
  and `modfiles/` (compiled in 0.2).

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
