Configs Reference

All tune-specific configs live under `cell_configs/`.
The pipeline prefers `cell_configs/` but still supports legacy paths during transition.

cell_config.json
- Identity: `cell_name`, `tune`, `color`.
- Paths: `paths.manifest`.
- Tuning: `tuning.soma_diam_multiplier` and related fields.

sim_config.json
Timing and trials
- `tstart`, `tstop`, `dt`, `jitter`, `stim_start_ms`, `stim_duration_ms`.
- `n_trials`, `n_inputs_to_save`.
- Trace/sample cap: `cell_recording.n_trials` (legacy top-level `n_traces_to_save` still accepted).

Save/load/append
- `save_output` (bool), `output` or `output_stem` (string).
- `output_format` (pkl|npz), `save_full_results` (bool).
- `save_sidecars`, `save_input_stats`, `save_syn_records_sidecar`, `save_fit_json_sidecar`.
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
- `cell_recording`: optional cell-segment recording during regular sims.
  - `enabled`: bool.
  - `n_trials`: how many trial traces to keep for saved `traces["V"]` and `cell_recordings_by_trial`.
    - Legacy alias: top-level `n_traces_to_save`.
  - `sites`: list of recording locations (JSON-safe):
    - string form: `"soma[0](0.5)"` (also accepts `"soma"` / `"soma[0]"`).
    - dict form: `{"sec":"soma","idx":0,"x":0.5}` plus optional `label`.
  - `vars`: toggles for recorded variable classes:
    - `v`: membrane voltage at each selected site.
    - `i_cap`: capacitive membrane current (`i_cap`).
    - `ion_currents`: built-in ionic currents where present (`ina`, `ik`, `ica`, `ih`).
    - `mech_currents`: mechanism-specific current state variables (`i*`) for inserted density mechs.
    - `ion_concentrations`: ionic concentrations where present (`nai`, `ki`, `cai`, `nao`, `ko`, `cao`).
    - `ion_reversals`: ionic reversal potentials where present (`ena`, `ek`, `eca`).
    - `mech_conductances`: mechanism-specific conductance state variables (`g*`) for inserted density mechs.
    - `mech_states`: other mechanism state variables (neither `i*` nor `g*`).
  - Back-compat aliases are accepted: `rec_sec_list`, `rec_var_toggles` (or `rec_vars`).
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
- `specimen_id` and `model_type` are Step-1/download inputs and are not required
  in `cell_config.json` or `sim_config.json`.
- Paths in `syn_config.json` resolve relative to `cell_configs/`.
