# Target Trace Formats

`target_config.json` can point Steps 2-3 to target data instead of requiring all
values to be typed manually. The recommended public paths are:

- `manual`: enter passive targets and FI points directly, or use an FI CSV.
- `allen_nwb`: place a downloaded Allen/ADB ephys `.nwb` file in the tune folder.
- `traces`: use a local trace file that matches the contracts below.

## Passive Traces for Step 2

Use this mode when you have current-clamp voltage traces and want SCP to compute
passive targets for ACT.

Required `target_config.json` fields:

```json
{
  "target_source": {"mode": "traces"},
  "traces": {
    "passive": {
      "format": "csv",
      "file": "passive_traces.csv",
      "time_column": "time_ms",
      "voltage_column": "voltage_mV",
      "current_column": "current_pA",
      "sweep_column": "sweep",
      "stim_start_ms": null,
      "stim_stop_ms": null,
      "current_pA": null,
      "dt_ms": null,
      "end_margin_ms": 10.0,
      "reducer": "median",
      "tau_field": "tau_avg_ms"
    }
  }
}
```

CSV contract:

- One row per time point.
- Time is in ms and voltage is in mV.
- Required columns are the configured `time_column` and `voltage_column`.
- `current_column` is optional but recommended; values are in pA.
- `sweep_column` is optional; when present, each sweep is grouped separately.
- If no current column is available, set `stim_start_ms`, `stim_stop_ms`, and
  `current_pA` in `traces.passive`.

NPY contract for passive traces:

- `format` must be `"npy"`.
- The file must contain a 1D voltage trace or a 2D array of voltage traces with
  shape `(n_sweeps, n_timepoints)`.
- Values must be membrane voltage in mV.
- `dt_ms`, `stim_start_ms`, `stim_stop_ms`, and `current_pA` are required because
  the NPY file does not carry time/current columns.
- `current_pA` may be a single value or one value per sweep.

Step 2 writes extracted review files under
`notebook_exports/step2_passive/` before converting targets into ACT passive
parameters.

## Active Traces for Step 3

Current active-trace support is intentionally narrower than passive trace
support. Step 3 does not yet convert generic voltage traces into ACT active
training targets. It only passes an ACT-compatible `.npy` target file directly to
ACT.

Required `target_config.json` fields:

```json
{
  "target_source": {"mode": "traces"},
  "traces": {
    "active": {
      "format": "npy",
      "file": "active_target.npy",
      "stim_start_ms": null,
      "stim_stop_ms": null,
      "dt_ms": null,
      "spike_threshold_mV": -20.0,
      "refractory_ms": 1.0
    }
  }
}
```

ACT `.npy` contract:

- Shape must be `(n_trials, n_timepoints, n_columns)`.
- Column `0` must be membrane voltage in mV.
- Column `1` must be injected current in nA.
- Additional columns are ignored by ACT target feature extraction.
- ACT computes active summary features from the voltage/current arrays.
- ACT's current spike-frequency calculation assumes samples are effectively 1 ms
  apart. If traces use another timestep, use `manual.fi_curve` arrays/CSV or
  `allen_nwb` until SCP has a stable active-trace converter.

The `traces.active.stim_start_ms`, `stim_stop_ms`, `dt_ms`,
`spike_threshold_mV`, and `refractory_ms` fields are scaffolded for future
SCP-managed active trace conversion. In the current implementation, `trace_npy`
mode validates that the file exists and then passes it through to ACT.

## FI CSV for Step 3

If you have active traces but only need FI targets, the simpler path is to
provide summarized current/rate values as `manual.fi_curve.csv`.

Accepted current columns:

- `amp_pA`
- `current_pA`
- `amp_nA`
- `mean_i` in nA

Accepted firing-rate columns:

- `spike_frequency`
- `spike_frequency_hz`
- `frequency_hz`

Manual FI CSV remains `target_source.mode = "manual"` because it provides
already summarized target values, not raw traces.
