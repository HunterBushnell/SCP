# Config Cookbook

Common edits for files under `cell_configs/`.

## Run More Trials

In `sim_config.json`:

```json
"n_trials": 10
```

CLI override:

```bash
python run_pipeline.py --tune-dir cells/PV/tunes/seg_tuned --n-trials 10 --force-save
```

## Make Runs Reproducible

Use fixed randomness with an explicit seed:

```json
"randomness_mode": "fixed",
"seed": 12345
```

Use derived randomness for reproducible trial-to-trial variation:

```json
"randomness_mode": "derived",
"seed": 12345
```

## Run Cell-Only IClamp

In `sim_config.json`:

```json
"iclamp": {
  "enabled": true,
  "amp_nA": 0.2,
  "delay_ms": null,
  "dur_ms": null,
  "tstop_ms": null,
  "dt_ms": null,
  "record_currents": false
}
```

When `delay_ms` or `dur_ms` is `null`, Step 5 uses `stim_start_ms` and
`stim_duration_ms`.

## Save Output from a Notebook Run

In `sim_config.json`:

```json
"save": {
  "enabled": true,
  "stem": "my_test_run",
  "format": "pkl",
  "full_results": false
}
```

Or set `force_save = True` in `5_simulate.ipynb`, or use the manual save cell after a run you want to keep.

## Enable Auto Plots

Use one of the plot profiles:

```json
"plots_profile": "basic"
```

Options:

- `off`: no auto plots.
- `basic`: output diagnostic plot.
- `inputs`: output plus input plots.
- `full`: output, input, and synapse-recording plots when data are present.

## Save More Input Samples

```json
"n_inputs_to_save": "all"
```

For large runs, prefer a small integer to avoid large sidecars:

```json
"n_inputs_to_save": 5
```

## Record Extra Cell Variables

```json
"cell_recording": {
  "enabled": true,
  "n_trials": 1,
  "sites": [
    {"sec": "soma", "idx": 0, "x": 0.5}
  ],
  "vars": {
    "v": true,
    "i_cap": false,
    "ion_currents": true,
    "mech_currents": false,
    "ion_concentrations": false,
    "ion_reversals": false,
    "mech_conductances": false,
    "mech_states": false
  }
}
```

## Add a Homogeneous Baseline Block

In a `syn_groups/*.json` group:

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

## Add an Inhomogeneous Stimulus Block

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

## Scale an Inhomogeneous Rate Curve

Inside the block `source`:

```json
"freq_scale": 0.5,
"freq_shift": 1.0
```

## Switch Weight Style

Fixed weight:

```json
"params": {
  "initW": 1.0
}
```

Distributed weight:

```json
"params": {
  "wt_mean": 1.0,
  "wt_std": 0.0
}
```
