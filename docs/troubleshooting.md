Troubleshooting

Missing modfiles or NEURON errors
- Build modfiles: `cd <tune_dir>/modfiles && nrnivmodl`.

Missing configs
- Confirm `cell_configs/` contains `cell_config.json`, `sim_config.json`, `syn_config.json`.
- If using legacy locations, ensure files also exist at the tune root.

No outputs written
- Check `save_output` or `save` settings in `sim_config.json`.
- For SLURM, set `FORCE_SAVE=1` to force saving.

Notebook vs SLURM mismatch
- Enable snapshot mode to capture full metadata for comparison.
- Confirm the same seed, `trial_offset`, and config paths are used.

Input file not found
- Paths in `syn_config.json` are resolved relative to `cell_configs/`.
- Verify precomputed `source.path` files exist on the same machine.
