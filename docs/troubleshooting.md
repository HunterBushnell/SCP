Troubleshooting

First check
- Run `python scripts/check_setup.py --steps 0 1 2 3 4 5 --cell PV --tune seg_tuned`.

Missing modfiles or NEURON errors
- Local notebooks (Steps 2-5) expect precompiled mechanisms from Step 0.
- Prepare/refresh with Step 0 (`0_download.ipynb` or `scripts/step0_prepare.py`)
  if the tune bundle is incomplete.
- Build modfiles: `cd <tune_dir>/modfiles && nrnivmodl`.
- In Colab notebooks, rerun the bootstrap cell first (it installs deps and
  can compile mechanisms automatically).

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

ACT or bmtool not found (legacy Steps 1-4)
- Set `SCP_ACT_PATH` and/or `SCP_BMTOOL_PATH`, or place repos at `../mods/ACT` and `../mods/bmtool`.
