Config Cookbook (Short)

Common tweaks in `cell_configs/sim_config.json`.

1) Run more trials
- Set `n_trials` to a larger value (e.g., 10).

2) Enable snapshot mode
- `snapshot.enabled: true`
- Use for notebook vs SLURM comparisons.

3) IClamp test (no synapses)
- `iclamp.enabled: true`
- Adjust `amp_nA`, `delay_ms`, `dur_ms`.

4) Save full results bundle
- `save_full_results: true`
- Optional: set `output_format: pkl`.

5) Force saving in SLURM
- `FORCE_SAVE=1 sbatch run_slurm.sh`

6) Rate curve transforms (inhomogeneous inputs)
- In a synapse group `source`, you can add `gabab` (auto filter) plus
  `freq_scale`/`freq_shift` for post-filter scaling (see `configs_reference.md`).
