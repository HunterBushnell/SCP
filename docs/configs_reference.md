Configs Reference

All tune-specific configs live under `cell_configs/`.
The pipeline prefers `cell_configs/` but still supports legacy paths during transition.

cell_config.json
- Identity: `cell_name`, `tune`, `color`.
- Model metadata: `specimen_id`, `model_type`.
- Paths: `paths.manifest`.
- Tuning: `tuning.soma_diam_multiplier` and related fields.

sim_config.json
Timing and trials
- `tstart`, `tstop`, `dt`, `jitter`, `stim_start_ms`, `stim_duration_ms`.
- `n_trials`, `n_traces_to_save`, `n_inputs_to_save`.

Save/load/append
- `save_output` (bool), `output` or `output_stem` (string).
- `output_format` (pkl|npz), `save_full_results` (bool).
- `save_sidecars`, `save_input_stats`, `save_syn_records_sidecar`.
- `save_syn_records_by_trial` for per-trial synapse records.
- `load`, `save`, `append` can be list or dict forms.
- Back-compat: `append_to` is still accepted.

Plots and summaries
- `plots_profile`: off|basic|inputs|full.
- `save_plots`, `save_plots_inputs`, `save_plots_synapses`.
- `plots_win_size`, `plots_input_bin_ms`, `plots_input_smooth_ms`, `plots_raster_style`.
- `input_stats_bin_ms` controls input summary binning.

Modes
- `iclamp`: cell-only test mode.
- `snapshot`: full capture for debugging comparisons.

Randomness
- `randomness_mode` (fixed|derived|random).
- `random_seed` (legacy) and `randomness` block (global/trials/inputs/timing/synapses/modes).

syn_config.json
- Defines which synapse group files to include from `syn_groups/`.
- Typically uses `__includes__` with relative paths.

syn_groups/
- One JSON file per group (e.g., pn_exc, bg_inh, sst_inh).
- Each group defines `state`, `mode`, `source`, `timing`, and `syns`.
- Optional rate transforms for inhomogeneous inputs:
  - `source.gabab`: auto GABAB-style filter on the rate curve (bool or dict).
    - Common fields: `enabled`, `mode` (delayed|simple), `tau_s` or `tau_ms`,
      `delay_ms`, `history` (full|trimmed).
  - `source.freq_scale`: multiply rate curve (Hz) after GABAB.
  - `source.freq_shift`: additive rate shift (Hz) after GABAB.

Notes
- Keep cell identity in `cell_config.json` and sim-only settings in
  `sim_config.json` to avoid duplication.
- Paths in `syn_config.json` resolve relative to `cell_configs/`.
