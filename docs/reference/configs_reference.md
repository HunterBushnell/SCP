# Configs Reference

All tune-specific runtime configs live under:

```text
cells/<CELL>/tunes/<TUNE>/cell_configs/
```

Step 1 scaffolds the default files. Step 5 reads the same files from notebooks,
CLI runs, and SLURM jobs.

## Directory Layout

```text
cell_configs/
  cell_config.json
  sim_config.json
  geometry.json
  syn_config.json
  syn_groups/
    <group>.json
```

`syn_config.json` and `syn_groups/` are optional for cell-only or IClamp runs.

## `cell_config.json`

Cell identity, loader selection, and tune-level metadata.

Generated fields:

- `cell_name`: display/selection label, such as `PV` or `SST`.
- `tune`: tune directory name, such as `seg_tuned`.
- `color`: plotting color for the cell.
- `cell_loader`: loader adapter name. Current bundled examples use `allen_manifest`.
- `paths.manifest`: path to `manifest.json`, relative to the tune directory.
- `tuning.soma_diam_multiplier`: soma diameter multiplier used by setup/tuning/loading code.

Example:

```json
{
  "cell_name": "PV",
  "tune": "seg_tuned",
  "color": "royalblue",
  "cell_loader": "allen_manifest",
  "paths": {
    "manifest": "manifest.json"
  },
  "tuning": {
    "soma_diam_multiplier": 6.0
  }
}
```

Step 1 download inputs such as Allen `specimen_id` and `model_type` are setup
inputs, not runtime config identity fields.

## `sim_config.json`

Simulation-level timing, trial, saving, plotting, recording, and debug options.

### Timing

- `tstart`: simulation start time in ms.
- `tstop`: simulation stop time in ms.
- `dt`: NEURON integration time step in ms.
- `bins`: default output/input bin size in ms.
- `stim_start_ms`: stimulus/task marker used for plotting and summaries.
- `stim_duration_ms`: stimulus/task marker duration in ms.
- `jitter`: optional global onset jitter in ms. Use `null` for no global jitter.

Input timing for synaptic groups is defined in each group file with
`input_blocks`, not in `sim_config.json`.

### Trials and Saved Samples

- `n_trials`: number of simulation trials.
- `n_traces_to_save`: number of trial voltage traces to keep.
- `n_inputs_to_save`: number of trial input sets to keep, or `"all"`.
- `save_profile`: optional shortcut for sample caps: `lean`, `standard`, or `full`.

### Save, Load, and Append

- `load.enabled`: load existing output instead of running when supported.
- `load.path`: path to load.
- `save.enabled`: write run outputs.
- `save.stem`: output folder stem under `output_data/`.
- `save.format`: full-results format, `pkl` or `npz`.
- `save.full_results`: save the full Python results bundle.
- `append.enabled`: append/merge into a previous run.
- `append.path`: target run path or manifest.

CLI flags such as `--force-save` and `--output-stem` can override these at run
time.

### Input Summaries

- `save_input_stats`: write binned input summary sidecars.
- `input_stats_bin_ms`: bin size for input stats.
- `avg_rate_curve_smooth_ms`: smoothing window for saved/diagnostic average-rate curves.
- `avg_rate_curve_smooth_mode`: smoothing alignment, usually `center`.

### Auto-Plotting

- `plots_profile`: `off`, `basic`, `inputs`, or `full`.
- `plots_win_size`: output-rate smoothing/window size.
- `plots_input_bin_ms`: input plot bin size; `null` uses `input_stats_bin_ms`.
- `plots_input_smooth_ms`: input plot smoothing window.
- `plots_raster_style`: raster style, usually `dot`.
- `save_plots_mode`: plot backend mode. Current default is `single_plot`.
- `save_plots_single_plot_preset`: preset JSON path for single-plot output.
- `save_plots_overwrite`: overwrite existing auto-plot files.

`plots_profile` expands into the lower-level save flags used by the backend:

- `off`: no auto plots.
- `basic`: output diagnostic plot.
- `inputs`: output plus input plots.
- `full`: output, input, and synapse-recording plots when data are present.

### Randomness

- `randomness_mode`: `fixed`, `derived`, or `random`.
- `seed`: optional base seed.
- `trial_randomness`: optional per-trial variation scope: `inputs`, `synapses`, `both`, or `none`.

Recommended public defaults are:

```json
"randomness_mode": "random",
"seed": null
```

Use `fixed` plus an integer `seed` for reproducible repeated runs. Use `derived`
plus a `seed` for reproducible but trial-varying runs.

### IClamp

Cell-only current injection mode:

- `iclamp.enabled`: run IClamp instead of synapse-driven simulation.
- `iclamp.amp_nA`: current amplitude in nA.
- `iclamp.delay_ms`: delay before current onset; `null` falls back to `stim_start_ms`.
- `iclamp.dur_ms`: current duration; `null` falls back to `stim_duration_ms`.
- `iclamp.tstop_ms`: optional IClamp-specific `tstop`.
- `iclamp.dt_ms`: optional IClamp-specific `dt`.
- `iclamp.record_currents`: record current traces when supported.

### Cell Recording

Optional extra recording sites/variables:

- `cell_recording.enabled`: enable extra cell recordings.
- `cell_recording.n_trials`: number of trial recordings to keep.
- `cell_recording.sites`: list of recording locations.
- `cell_recording.vars`: variable-class toggles.

Site forms:

```json
{"sec": "soma", "idx": 0, "x": 0.5}
```

```json
"soma[0](0.5)"
```

Supported variable toggles:

- `v`
- `i_cap`
- `ion_currents`
- `mech_currents`
- `ion_concentrations`
- `ion_reversals`
- `mech_conductances`
- `mech_states`

### Synapse Recording

Optional synapse variable recording:

- `syn_recording.enabled`: enable synapse recording.
- `syn_recording.default_mode`: default sampling mode, currently `group`.
- `syn_recording.default_sample_per_group`: default sampled synapses per group.
- `syn_recording.default_vars`: variables to record where present.
- `syn_recording.groups`: per-group overrides.

Common variable toggles:

- `i`
- `g`
- `i_AMPA`
- `i_NMDA`
- `g_AMPA`
- `g_NMDA`
- `record_use`
- `record_Pr`

The config schema is present now; detailed synapse-recording machinery remains
a planned Step 5 extension.

### Snapshot

Snapshot mode is a debugging/comparison capture:

- `snapshot.enabled`: enable full capture.
- `snapshot.n_trials`: number of snapshot trials.
- `snapshot.save_all_inputs`: keep all generated inputs.
- `snapshot.save_all_traces`: keep all voltage traces.
- `snapshot.save_syn_records_by_trial`: keep per-trial synapse records.

When enabled, snapshot mode forces saving and richer sidecars.

## `geometry.json`

Segment grouping and distance reference used for synapse placement.

Generated fields:

- `distance_origin.kind`: distance reference type, currently `soma`.
- `distance_origin.x`: reference position on the soma section.
- `thresholds_um.proximal.low`: lower bound for proximal dendrite group.
- `thresholds_um.proximal.high`: upper bound for proximal dendrite group.
- `thresholds_um.distal.low`: lower bound for distal dendrite group.
- `thresholds_um.distal.high`: upper bound for distal group, or `null`.
- `label`: human-readable geometry label.

Example:

```json
{
  "distance_origin": {"kind": "soma", "x": 0.5},
  "thresholds_um": {
    "proximal": {"low": 20.0, "high": 100.0},
    "distal": {"low": 100.0, "high": null}
  },
  "label": "PV_default_geometry"
}
```

## `syn_config.json`

Manifest of enabled synapse-group files.

```json
{
  "group_files": [
    "syn_groups/pn_exc.json",
    "syn_groups/bg_exc.json"
  ]
}
```

Paths resolve relative to `cell_configs/`. A file can exist in `syn_groups/`
without being active; it is only used when listed in `group_files`.

## `syn_groups/*.json`

Each file maps one or more group names to group configs.

Top-level group fields:

- `state`: enable/disable the group.
- `color`: plotting color.
- `input_blocks`: ordered list of explicit input windows.
- `syns`: synapse mechanism, count, placement, and parameter settings.

Example shape:

```json
{
  "pn_exc": {
    "state": true,
    "color": "#1f77b4",
    "syns": {
      "type": "AMPA_NMDA_STP",
      "N_syn": 435,
      "segs": "all",
      "dist_func": {"kind": "uniform", "params": {"c": 2.0}},
      "params": {"wt_mean": 0.35, "wt_std": 0.33}
    },
    "input_blocks": []
  }
}
```

### `input_blocks`

`input_blocks` define the active input windows for the group.

Common fields:

- `name`: unique block label within the group.
- `role`: descriptive role such as `baseline`, `stimulus`, or `background`.
- `start_ms`: block start time in simulation time.
- `stop_ms`: block stop time in simulation time.
- `mode`: input mode.
- `state`: optional per-block enable/disable flag.
- `jitter_ms`: optional per-block onset jitter.

Rules:

- Blocks must lie inside `sim_config.tstart` to `sim_config.tstop`.
- Blocks must not overlap.
- Gaps between blocks are quiescent.
- Source crop duration must match block duration for current public configs.

Homogeneous block:

```json
{
  "name": "pre_baseline",
  "role": "baseline",
  "start_ms": 0.0,
  "stop_ms": 300.0,
  "mode": "homogeneous_poisson",
  "rate_hz": 2.0
}
```

Inhomogeneous block:

```json
{
  "name": "stimulus",
  "role": "stimulus",
  "start_ms": 300.0,
  "stop_ms": 800.0,
  "mode": "inhomogeneous_poisson",
  "source": {
    "path": "external_data/pyrFiringRateAvg.csv",
    "time_col": "Time",
    "rate_col": "AvgFiringRate",
    "bin_ms": 5.0,
    "crop_start_ms": 0.0,
    "crop_stop_ms": 500.0
  }
}
```

Precomputed block:

```json
{
  "name": "stimulus",
  "role": "stimulus",
  "start_ms": 0.0,
  "stop_ms": 1000.0,
  "mode": "precomputed",
  "source": {
    "path": "cells/SST/tunes/seg_tuned/output_data/example/results/spikes.npz",
    "selection": "sample",
    "crop_start_ms": 0.0,
    "crop_stop_ms": 1000.0
  }
}
```

### Source Transforms

Inhomogeneous source blocks can apply simple rate transforms:

- `source.gabab`: GABAB-style rate filtering, either `true` or a dict.
- `source.freq_scale`: multiply rate curve after filtering.
- `source.freq_shift`: add to rate curve after filtering.

Common `gabab` dict fields:

- `enabled`
- `mode`: `delayed` or `simple`
- `tau_s` or `tau_ms`
- `delay_ms`
- `history`: `full` or `trimmed`

### `syns`

Synapse placement/mechanism fields:

- `type`: NEURON point-process/synapse mechanism name.
- `N_syn`: explicit synapse count, or `null` for geometry/density-derived count.
- `segs`: segment selector: `all`, `proximal`, `distal`, or `soma`.
- `dist_func`: distance-density function for placement.
- `params`: mechanism parameters and weight parameters.

Weight parameter patterns:

- Fixed weight: `{"initW": 1.0}`
- Distributed weight: `{"wt_mean": 1.0, "wt_std": 0.0}`

Mechanism-specific entries in `syns.params` are passed through to the synapse
mechanism when supported, so they vary by model and mechanism.

### `dist_func`

Distance-density functions control how many synapses are assigned as a function
of distance from soma.

Supported forms:

- `{"kind": "uniform", "params": {"c": 1.0}}`
- `{"kind": "linear", "params": {"m": -0.015, "b": 4.25}}`
- `{"kind": "polynomial", "params": {"coeffs": [1.0, 0.0]}}`
- `{"kind": "exponential", "params": {"a": 1.0, "tau": 100.0, "b": 0.0}}`
- `{"kind": "gaussian", "params": {"a": 1.0, "mu": 100.0, "sigma": 25.0, "b": 0.0}}`
- `{"kind": "piecewise_linear", "params": {"points": [[0, 1.0], [200, 0.2]]}}`

Common optional params:

- `multi`: multiplier applied to the density.
- `clip_min`: lower clip value; default `0.0`.
- `clip_max`: optional upper clip value.
