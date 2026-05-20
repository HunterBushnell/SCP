Step 5: Local Pipeline

Step 5 is the stable, current pipeline for running tuned single-cell simulations.
The notebook `5_local.ipynb` is the reference implementation.

Sub-steps (documentation labels)
- 5.2.1 Load cell: build NEURON cell from `cell_config.json` and `manifest.json`.
- 5.2.2 Geometry: load `geometry.json` and build segment groups.
- 5.2.3 Inputs: generate spike trains from `syn_config.json` and `syn_groups/`.
- 5.2.4 Synapses: place synapses on the cell using geometry + inputs.
- 5.3 Run: run single or multi trials and save outputs.
- 5.4 Analyze: summarize and compare outputs.

Entry points
- Notebook: `5_local.ipynb`
- CLI: `run_pipeline.py`
- SLURM: `run_slurm.sh`

Inputs (tune directory)
- `cell_configs/cell_config.json`
- `cell_configs/sim_config.json`
- `cell_configs/syn_config.json`
- `cell_configs/syn_groups/`
- `manifest.json`
- `modfiles/` (compiled once per tune)

Outputs
- `output_data/<output_stem>/` with `run_manifest.json` and sidecars.
- Optional plots in `output_data/<output_stem>/plots/`.
- Optional row-per-trial spikes CSV:
  - `python scripts/export_spikes_csv.py --input output_data/<output_stem>/results/spikes.npz`

IClamp mode
- Enable via `cell_configs/sim_config.json -> iclamp.enabled: true`.
- Runs a current injection test without synapses or inputs.

Snapshot mode
- Enable via `cell_configs/sim_config.json -> snapshot.enabled: true` or `--snapshot`.
- Captures full inputs/traces/synapse records for comparison.

Append mode
- `append` can target an existing run folder or manifest and reuse its sim_cfg.

Notes
- Step numbers in older docs may refer to 2.x (2.3, 2.4). In SCP these map
  to 5.2.3 and 5.2.4 to keep Step 5 as the stable pipeline.
- Inhomogeneous input sources can use `source.gabab` and
  `source.freq_scale`/`source.freq_shift` (see `configs_reference.md`).
- `5_old_PV.ipynb` and `5_old_SST.ipynb` are legacy notebooks kept for reference.
- `colab_notebooks/5_colab.ipynb` bootstraps a clean environment when needed.
