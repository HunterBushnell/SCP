# 6_analysis Guide

This guide covers day-to-day use of `6_analysis.ipynb`.
It is intended to be comprehensive enough for regular analysis work without
going too deep into implementation details.

## Scope

`6_analysis.ipynb` is the end-of-pipeline analysis notebook for SCP.
It is used to:

- plot output and input activity from saved runs
- compare runs and/or external curves
- compute output metrics tables and metric distributions
- generate recording/config/snapshot comparison tables

## What It Reads

- Run folders under `cells/<cell>/tunes/<tune>/output_data/<run_name>/`
- Optional external curves (CSV paths) from compare-path entries
- Defaults from `modules/analysis/analysis_defaults.json`
- Output-plot defaults from `modules/analysis/analysis_presets/output_plotting.json`
  (loaded via `output_plot_preset_path` in `analysis_defaults.json`)
- Input-plot defaults from `modules/analysis/analysis_presets/input_plotting.json`
  (loaded via `input_plot_preset_path` in `analysis_defaults.json`)
- Output-metrics defaults from `modules/analysis/analysis_presets/output_metrics.json`
  (loaded via `output_metrics_preset_path` in `analysis_defaults.json`)
- Extra-analysis defaults from `modules/analysis/analysis_presets/extra_analysis.json`
  (loaded via `extra_preset_path` in `analysis_defaults.json`)
- Optional presets from `modules/analysis/analysis_presets/`

## Typical Workflow

1. Open `6_analysis.ipynb`.
2. Run setup/import cells.
3. Build Selection UI, then choose cell/tune/model and run(s).
4. Use Outputs and Inputs UI to generate plots.
5. Use Extra UI for metrics/config/snapshot tables.
6. Save plots/analysis artifacts when needed.

## UI Sections

### Selection

Use Selection to define what data is in scope for downstream plots/tables.

- Choose `Cell`, `Tune`, and `Model` to set the base `output_data` tree.
- Select runs from compare list, or define run A/run B.
- Add compare curve paths with optional metadata.
- Use run tokens: `latest`, `previous`, `prev`, `latest-1`.
- Toggle `Save plots` / `Save analysis` for generated artifacts.

Compare path syntax:

```text
path/to/curve.csv@shift:scale;color=red;label=Bio;linestyle=--
```

Notes:

- `shift` is in ms.
- `scale` is multiplicative.
- `@shift`, `@shift:scale`, and `@shift,scale` are accepted.
- Style keys include `color`, `label`, `linestyle`, `shift`, `scale`.

### Outputs

Use Outputs for firing-rate / ISI curve plotting and run-to-run overlays.

- Curve mode: `raw` or `normalized`.
- Curve plot mode: rate, ISI, or stacked rate+ISI.
- Norm mode: `avg` or `peak`.
- Compare layout: `overlay`, `stacked`, `side-by-side`.
- Shading options for uncertainty bands (std/sem).
- Plot window controls and optional auto-window around stim.
- Optional plotted-data CSV export (trace rows or long rows).

### Inputs

Use Inputs for presynaptic input summaries and input raster/mean plots.

- Input source mode: auto/saved/stats.
- Group filters (comma-separated synapse groups).
- Bin/smooth controls for input curves.
- Raster controls (trial index, max trains, style).
- Compare layout + optional std display.
- Optional plotted-data CSV export (trace rows or long rows).

### Extra

Extra mode provides analysis utilities beyond standard plots.

Available modes:

- `Output metrics (table)`
- `Compare configs (restore-style)`
- `Compare outputs (plots)`
- `Compare inputs (plots)`
- `Input sampling (preview)`
- `Snapshot compare`
- `Single-run tables`
- `Spike stats`
- `IClamp analysis`

## Output Metrics

Output metrics are computed from the output rate curve for selected runs/curves.

Common metrics shown by default:

- `baseline_mean`
- `peak_rate_hz_raw`
- `peak_latency_ms`
- `Tpeak10` (latency from stim start to first rising-phase point at `>= 10%` of peak)
- `drop_pct` (the +100ms post-peak drop percentage)
- `T50` (time from peak to first 50% drop crossing)
- `rebound_pct`
- `auc`

Notes on definitions:

- `drop_pct` uses the configured post-peak drop evaluation point/window.
- `Tpeak10` uses the first sampled point between stim start and peak where the
  curve reaches `10%` of peak on the rising phase.
- `T50` is computed as time from peak to the first sampled point after peak
  where the curve is `<= 50%` of peak.
  - default mode (`output_t50_mode=absolute`): latency from stim start (ms)
  - optional mode (`output_t50_mode=relative`): time since peak (ms)
- `auc` uses `output_auc_window`, which supports:
  - `stim`: integrate over stim window
  - `full`: integrate over full sim window
  - custom bounds: `[start_ms, stop_ms]` (or `"start,stop"` text)

Key knobs (defaults JSON):

- `output_peak_window_ms`
- `output_drop_window_ms`
- `output_rebound_window_ms`
- `output_auc_window`
- `output_t50_mode` (`absolute` or `relative`)
- `output_metric_mode` (`point` or `window`)
- `output_metric_window_ms`
- `output_metrics_std_mode` (`std` or `sem`)
- `output_metrics_plot_keys`

## What Gets Saved

When save toggles are enabled, `6_analysis` can write:

- Figures into run/compare plot directories
- Output metrics JSON
  - single run: `output_metrics.json`
  - run-vs-run: `output_metrics_compare.json`
  - compare-list mode: `output_metrics_list.json`
- Output metric distribution plot and optional CSV
- Plotted-data CSV exports from Outputs/Inputs panels

## Quick Recipes

### Single-run quality check

1. In Selection, set run to `latest`.
2. In Outputs, plot normalized rate (optionally with ISI).
3. In Extra, run `Output metrics (table)` and inspect `Tpeak10`, `drop_pct`, `T50`, and `auc`.

### Run-vs-run comparison

1. In Selection, choose run A and run B.
2. In Outputs, use `overlay` for direct shape comparison.
3. In Extra, run `Output metrics (table)` with deltas/highlight enabled.

### Compare against external CSV curves

1. Add curve paths in compare-path syntax.
2. Keep `Use compare paths` enabled.
3. In Outputs, render compare curves; in Extra, run output metrics table/distributions.

## Related Docs

- `docs/step_6_analysis.md` (legacy step-oriented reference)
- `docs/pipeline_overview.md`
- `docs/outputs_layout.md`
- `docs/configs_reference.md`
- `modules/analysis/README.md`
