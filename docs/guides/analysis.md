# 6_analysis Guide

This guide covers day-to-day use of `6_analysis.ipynb`.
Use it for saved-output inspection, run comparisons, external curve overlays,
metrics, and optional diagnostic tables.

## Scope

`6_analysis.ipynb` is the post-processing notebook for SCP. It reads saved Step
5 output folders and optional external curves; it does not rerun simulations.

Use Step 6 to:

- make a compact single-run summary figure
- plot model output firing-rate and ISI curves
- plot saved or summarized synaptic input activity
- compare runs and/or external CSV curves
- compute output metrics and metric-distribution plots
- inspect optional saved tables, synapse records, snapshots, and IClamp traces

## What It Reads

- Saved run folders under `cells/<CELL>/<tunes_dir>/<model_dir>/output_data/<run_name>/`
- Optional external CSV curves, usually from `external_data/`
- `modules/analysis/analysis_defaults.json`
- `modules/analysis/analysis_presets/single_plot.json`
- `modules/analysis/analysis_presets/output_plotting.json`
- `modules/analysis/analysis_presets/input_plotting.json`
- `modules/analysis/analysis_presets/output_metrics.json`
- `modules/analysis/analysis_presets/extra_analysis.json`
- Optional compare presets configured through `compare_preset_path`

## Typical Workflow

1. Open `6_analysis.ipynb`.
2. Run **6.0 Environment Setup**.
3. In **6.1 Select Runs**, choose the cell/tune/model and run(s).
4. Use **6.2 Single Plot** for the first single-run quality check.
5. Use **6.3 Output Plots** and **6.4 Input Plots** for deeper plot review.
6. Use **6.5 Extra Analysis** for metrics, config comparisons, tables, or diagnostics.
7. Enable `Save plots` / `Save analysis` only when you want artifacts written to disk.

## Selection

Selection defines the output tree and comparison data used by every downstream
section.

Notebook variables:

- `cell_name`: cell folder under `cells/`, such as `SST` or `PV`
- `tunes_dir`: usually `tunes`
- `model_dir`: tune/model folder, such as `seg_tuned`
- `run_single_stem`: single-run target; `latest` selects the newest run
- `run_compare_a`, `run_compare_b`: optional run-vs-run comparison targets
- `compare_a_path`, `compare_b_path`: explicit path overrides for manual use

Widget controls:

- Cell/tune/model selectors
- Single run selector
- Compare list selector
- Compare path entry box
- Save toggles for plots and analysis artifacts

Run tokens:

- `latest`
- `previous` / `prev`
- `latest-1`, `latest-2`, etc.

Compare path syntax:

```text
path/to/curve.csv@shift:scale;color=red;label=Bio;linestyle=--
```

Notes:

- `shift` is in ms.
- `scale` is multiplicative.
- `@shift`, `@shift:scale`, and `@shift,scale` are accepted.
- Style keys include `color`, `label`, `linestyle`, `shift`, and `scale`.

## 6.2 Single Plot

Single Plot creates the compact all-in-one single-run panel. It is the best
first check after Step 5 because it can combine output rate, Vm, output raster,
input rate, and input raster in one figure.

Notebook options:

- `single_plot_trial_idx`: saved trial used for Vm/raster panels
- `single_plot_window`: optional `[start_ms, stop_ms]` x-axis crop
- `single_plot_include_input_raster`: show/hide input raster panel
- `single_plot_include_output_raster`: show/hide output spike raster panel
- `single_plot_top_input_groups`: input groups shown in the top input-rate panel
- `single_plot_raster_input_groups`: input groups shown in the raster panel
- `single_plot_display_options`: notebook display size only

JSON defaults:

- `top_input_mode`: input summary mode for the top panel
- `top_layout`: input overlay/layout mode
- `include_input_raster`, `include_output_raster`
- `input_bin_ms`, `input_smooth_ms`, `input_source`
- `output_curve_source`, `output_recompute_bin_ms`, `output_recompute_smooth_ms`
- `output_mode`, `output_norm_kind`, `output_norm_window`
- `plot_window`, `auto_plot_window_from_stim`, `plot_window_adjustment_ms`
- `figsize`, `panel_height_ratios`, `export_formats`, `dpi`

Primary file: `modules/analysis/analysis_presets/single_plot.json`.

## 6.3 Output Plots

Output Plots are for firing-rate and ISI plots across one run, two runs, a
compare list, and/or external curves.

UI sections:

- **Run**: full output plot, compact rate/ISI curve, spike stats, output raster
- **Window**: x/y limits, stimulus markers, auto-windowing, zero-origin display
- **Curve**: raw/normalized mode, rate/ISI/stacked mode, binning, smoothing
- **Compare**: overlay/side-by-side/stacked layout and std/sem bands
- **Export**: plotted-data CSV export

Common JSON options:

- `plot_outputs`: full saved-output plot
- `plot_output_curve`: compact output curve
- `plot_spike_stats`: spike count/timing summary for single runs
- `plot_raster`: output spike raster
- `plot_window`, `y_window`
- `output_stim_start_ms`, `output_stim_stop_ms`
- `output_curve_mode`: `raw` or `normalized`
- `output_curve_plot_mode`: `rate`, `isi`, or `rate_isi`
- `output_norm_mode`: `avg` or `peak`
- `output_bin_ms`, `output_smooth_mode`
- `compare_output_layout`: `overlay`, `stacked`, or `side-by-side`
- `multi_shade_mode`: `std`, `sem`, or `null`
- `output_plot_data_auto_save`, `output_plot_data_path`

Primary file: `modules/analysis/analysis_presets/output_plotting.json`.

## 6.4 Input Plots

Input Plots are for synaptic input summaries, input rasters, and input
comparisons. They are useful for checking what presynaptic activity was delivered
to the model.

UI sections:

- **Run**: input source, mean traces, rasters, trial/group filters
- **Window**: x-axis crop and stimulus markers
- **Curve**: binning, smoothing, std/sem display, line styling
- **Raster**: trial index, max displayed trains, dot/line raster style
- **Compare/Export**: compare layout and plotted-data CSV export

Common JSON options:

- `plot_inputs_mean`: input mean/summary traces
- `plot_input_raster`: input raster panel
- `input_source`: `auto`, `saved`, or `stats`
- `input_groups`: group filter; `null` means all available groups
- `input_bin_ms`, `input_smooth_ms`
- `show_input_std`, `input_std_mode`
- `input_raster_trial_idx`, `input_raster_max_trains`
- `input_raster_style`: `dot` or `line`
- `compare_input_layout`
- `input_plot_data_auto_save`, `input_plot_data_path`

Primary file: `modules/analysis/analysis_presets/input_plotting.json`.

## 6.5 Extra Analysis

Extra Analysis contains optional utilities that do not belong in the core
single/output/input plotting path.

Available modes:

- `Output metrics (table)`: output metric table for selected runs/curves
- `Compare configs (restore-style)`: compare configs across runs and the current tune
- `Input sampling (preview)`: sample input curves from synapse configs
- `Synapse plots`: summarize and plot saved Step 5 synapse records
- `Snapshot compare`: compare saved run snapshots/manifests
- `Single-run tables`: summarize cell, geometry, synapse, and recording data
- `IClamp analysis`: summarize current-injection outputs

Common JSON options:

- `input_sample_source`: `selection`, `run`, or `path`
- `input_sample_groups`, `input_sample_runs`, `input_sample_bin_ms`
- `extra_cell_tables`, `extra_geometry_tables`, `extra_synapse_tables`
- `extra_recording_tables`
- `extra_compare_apply`
- `extra_compare_syn_groups_selector`
- `extra_compare_diff_only`
- `extra_snapshot_diff_only`, `save_snapshot_compare_table`
- `extra_synapse_show_table`
- `extra_synapse_weight_plot`, `extra_synapse_distance_plot`
- `extra_synapse_scatter_plot`, `extra_synapse_density`
- `extra_synapse_groups`, `extra_synapse_trial_idx`

Data-dependent behavior:

- `Synapse plots` require Step 5 outputs saved with synapse records.
- `Single-run tables` only show recording/synapse summaries when those records were saved.
- Cell and geometry tables may load the cell model and mechanisms.
- `IClamp analysis` skips non-`iclamp` runs.

Primary file: `modules/analysis/analysis_presets/extra_analysis.json`.

## Output Metrics

Output metrics are computed from the output rate curve.

Default table/plot metrics:

- `baseline_mean`
- `peak_rate_hz_raw`
- `peak_latency_ms`
- `tpeak10_ms`
- `drop_pct`
- `t50_ms`
- `rebound_pct`
- `auc`

Metric definition options:

- `output_baseline_center_ms`
- `output_metric_mode`: `point` or `window`
- `output_metric_window_ms`
- `output_peak_window_ms`
- `output_drop_window_ms`
- `output_rebound_window_ms`
- `output_auc_window`: `stim`, `full`, or custom `[start_ms, stop_ms]`
- `output_t50_mode`: `absolute` or `relative`

Metric display options:

- `output_metrics_show_params`
- `output_metrics_std_mode`: `std` or `sem`
- `output_metrics_ref_label`
- `output_metrics_show_delta`
- `output_metrics_highlight_best`
- `output_metrics_plot_keys`
- `output_metrics_plot_style`: `box` or `bar`

Primary file: `modules/analysis/analysis_presets/output_metrics.json`.

## Saving and Exports

Global save options live in `modules/analysis/analysis_defaults.json`.

- `save_plots`: save generated figures
- `save_analysis`: save generated JSON/table artifacts
- `save_overwrite`: allow replacing existing saved plots/artifacts
- `plots_dpi`: saved figure resolution
- `load_cell_for_analysis`: allow cell loading for richer tables

When save toggles are enabled, Step 6 can write:

- figures into run/compare plot directories
- `output_metrics.json`
- `output_metrics_compare.json`
- `output_metrics_list.json`
- `config_compare_report.json`
- `synapse_summary.json`
- recording summary JSON files
- plotted-data CSV exports from Outputs/Inputs

Plotted-data CSV export formats:

- `trace_rows`: one row per trace with vector-like columns
- `long_rows`: one row per plotted point

If a CSV path is only a filename, it saves under the relevant plot-data output
folder. If blank and auto-save is enabled, Step 6 generates a filename.

## Presets

Default preset paths are configured in `modules/analysis/analysis_defaults.json`.

- `output_plot_preset_path`
- `input_plot_preset_path`
- `output_metrics_preset_path`
- `extra_preset_path`
- `compare_preset_path`

Compare presets are optional. Set `compare_preset_path` to a preset JSON file
when you want to load a predefined set of comparison curves and plot options.

## Quick Recipes

### Single-run quality check

1. In Selection, set run to `latest`.
2. Run Single Plot.
3. In Output Plots, plot normalized rate if you need a focused output figure.
4. In Extra Analysis, run `Output metrics (table)`.

### Run-vs-run comparison

1. In Selection, choose run A and run B or select runs in the compare list.
2. In Output Plots, use `overlay` for direct shape comparison.
3. In Extra Analysis, run `Output metrics (table)` with deltas/highlight enabled.

### Compare against external CSV curves

1. Add curve paths using compare-path syntax.
2. Keep `Use compare paths` enabled.
3. In Output Plots, render compare curves.
4. In Extra Analysis, run output metrics table/distribution plots.

### Inspect synapse placement/weights

1. Select a Step 5 run saved with synapse records.
2. In Extra Analysis, choose `Synapse plots`.
3. Optionally filter groups with `extra_synapse_groups`.
4. Enable `Distance density` only if cell/geometry loading is needed and working.

## Troubleshooting

- If no runs appear, verify the selected `cells/<CELL>/<tunes_dir>/<model_dir>/output_data` path exists.
- If external curves do not appear, check the compare path and CSV columns.
- If synapse plots skip, rerun Step 5 with synapse-record saving enabled.
- If cell/geometry tables fail, verify mechanisms are compiled and cell loading works.
- If Colab widgets do not render, rerun setup/import cells and ensure `ipywidgets` is installed.

## Related Docs

- `step_6_analysis.md`
- `../reference/outputs_layout.md`
- `../reference/configs_reference.md`
- `../../modules/analysis/README.md`
