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
  target_config.json
  geometry.json
  syn_config.json
  syn_groups/
    <group>.json
```

`target_config.json` is optional for characterization without biological targets.
`syn_config.json` and `syn_groups/` are optional for cell-only or IClamp runs.

## `cell_config.json`

Cell identity, loader selection, and tune-level metadata.

Generated fields:

- `cell_name`: display/selection label, such as `PV` or `SST`.
- `tune`: tune directory name, such as `tuned`.
- `color`: plotting color for the cell.
- `cell_loader`: registered adapter name. Supported values are
  `allen_manifest` (default, with legacy Allen aliases) and `hoc_template`.
- `paths.manifest`: Allen manifest path, relative to the tune directory.
- `paths.hoc_template`: HOC entry source for the HOC-template loader.
- `paths.modfiles`: optional mechanism-source directory; `null` means built-in
  NEURON mechanisms only.
- `tuning.soma_diam_multiplier`: optional Allen-only geometry compatibility
  setting. It defaults to `1.0` for Allen and is not added or applied to HOC
  templates.

Example:

```json
{
  "cell_name": "PV",
  "tune": "tuned",
  "color": "blue",
  "cell_loader": "allen_manifest",
  "paths": {
    "manifest": "manifest.json"
  },
  "tuning": {
    "soma_diam_multiplier": 1.0
  }
}
```

Step 1 download inputs such as Allen `specimen_id` and `model_type` are setup
inputs, not runtime config identity fields.

Generic object-owned HOC example:

```json
{
  "cell_name": "example_cell",
  "tune": "orig",
  "cell_loader": "hoc_template",
  "paths": {
    "hoc_template": "model/CellTemplate.hoc",
    "modfiles": null
  },
  "hoc_template": {
    "template_name": "CellTemplate",
    "constructor_args": [],
    "section_map": {
      "soma": "somatic",
      "dend": ["basal"],
      "apic": ["apical"],
      "axon": ["axonal"],
      "all": "all"
    }
  }
}
```

Mapping values may be a string or list of owner attributes. Optional groups can
be omitted or map to empty section collections; soma must resolve to at least
one section. See `model_loaders.md` for construction and process-lifetime rules.

## `target_config.json`

Optional biological or experimental targets used by tuning notebooks. This file
stores desired model behavior; it does not control Step 5 simulation execution.

The target config is organized by **source mode**:

- `none`: no biological target; Steps 2–3 run intrinsic trace/FI checks only.
- `manual`: user-entered passive values and FI curve values or FI CSV.
- `traces`: user-provided trace files following the SCP trace contract.
- `allen_nwb`: Allen/ADB electrophysiology `.nwb` files.

Generated fields:

- `schema_version`: target-config schema version.
- `target_source.mode`: one of `none`, `manual`, `traces`, or `allen_nwb`.
- `target_source.description`: optional human-readable target-source note.
- `manual.passive`: direct Step 2 targets: `v_rest_mV`, `rin_MOhm`, `tau_ms`.
- `manual.fi_curve`: direct Step 3 FI targets: `currents_pA`, `rates_Hz`, or `csv`.
- `traces.passive`: generic passive trace file contract for Step 2 extraction.
- `traces.active`: ACT-compatible active trace target file contract for Step 3.
- `allen_nwb.file`: tune-local or absolute Allen/ADB NWB file path.
- `allen_nwb.passive`: Step 2 NWB passive sweep/filter settings.
- `allen_nwb.active`: Step 3 NWB FI sweep/filter settings.
- `notes`: free-form target notes.

Example:

```json
{
  "schema_version": 1,
  "target_source": {
    "mode": "manual",
    "description": ""
  },
  "manual": {
    "passive": {
      "v_rest_mV": null,
      "rin_MOhm": null,
      "tau_ms": null
    },
    "fi_curve": {
      "currents_pA": [],
      "rates_Hz": [],
      "csv": null
    }
  },
  "traces": {
    "format": "csv",
    "passive": {
      "file": null,
      "time_column": "time_ms",
      "voltage_column": "voltage_mV",
      "current_column": "current_pA",
      "sweep_column": null,
      "stim_start_ms": null,
      "stim_stop_ms": null,
      "current_pA": null,
      "dt_ms": null,
      "end_margin_ms": 10.0,
      "reducer": "median",
      "tau_field": "tau_avg_ms"
    },
    "active": {
      "file": null,
      "format": "npy",
      "stim_start_ms": null,
      "stim_stop_ms": null,
      "dt_ms": null,
      "spike_threshold_mV": -20.0,
      "refractory_ms": 1.0
    }
  },
  "allen_nwb": {
    "file": null,
    "sweep_ids": [],
    "passive": {
      "stimulus_names": ["Long Square"],
      "sweep_ids": null,
      "min_current_pA": null,
      "max_current_pA": -1.0,
      "end_margin_ms": 10.0,
      "reducer": "median",
      "tau_field": "tau_avg_ms"
    },
    "active": {
      "stimulus_names": ["Long Square"],
      "min_current_pA": 0.0,
      "max_current_pA": null,
      "include_negative_currents": false,
      "average_repeats": true,
      "spike_threshold_mV": -20.0,
      "refractory_ms": 1.0
    }
  },
  "notes": ""
}
```

Step 2 uses `manual.passive`, calculates targets from `traces.passive`, or
calculates targets from `allen_nwb.passive` depending on `target_source.mode`.
Step 3 uses `manual.fi_curve`, `traces.active`, or `allen_nwb.active` the same
way. Manual FI CSV is still considered manual mode because it provides already
summarized FI values rather than raw traces.

Detailed passive trace, active trace, and FI CSV file requirements are in
`docs/reference/target_trace_formats.md`.

For generic passive CSV traces, `time_column` should start at 0 ms for each
trace. Required columns are the configured time and voltage columns. A current
column can be used for automatic stimulus-window detection; otherwise set
`stim_start_ms`, `stim_stop_ms`, and `current_pA` in `traces.passive`.

For active trace targets, `traces.active` currently expects an ACT-compatible
`.npy` file with shape `(n_trials, n_timepoints, n_columns)`, voltage in column
0, and injected current in nA in column 1. SCP currently passes that file
through to ACT; it does not yet convert generic active voltage traces.

## `sim_config.json`

Simulation-level timing, trial, saving, plotting, recording, and debug options.

### Runtime Conditions

- `conditions.v_init_mV`: initialization voltage in mV.
- `conditions.celsius_C`: NEURON temperature in degrees Celsius.

Both values are required and must be finite for new `hoc_template` tunes. SCP
applies them before every passive, active/FI, and Step 5 protocol or trial.
Legacy Allen tunes without this block retain their existing loader/runtime
fallbacks.

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
- `save.stem`: output folder stem under `output_data/`; `null` uses a timestamped `run_...` folder when saving is enabled.
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
    "path": "cells/SST/tunes/tuned/output_data/example/results/spikes.npz",
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
