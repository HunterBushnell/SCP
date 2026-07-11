# Step 6: Analysis Reference

`6_analysis.ipynb` is the analysis notebook for the SCP pipeline. It
turns saved Step 5 outputs into plots, metrics, tables, and comparison reports.

For regular usage, start with `docs/guides/analysis.md`. This file is the more
technical reference for Step 6 inputs, defaults, and utility outputs.

## Notebook Structure

- **6.0 Environment Setup**: local/Colab bootstrap and imports
- **6.1 Select Runs**: output tree, single run, compare runs, external curves
- **6.2 Single Plot**: compact all-in-one single-run figure
- **6.3 Output Plots**: output firing-rate/ISI curves and comparisons
- **6.4 Input Plots**: input summaries and input rasters
- **6.5 Extra Analysis**: metrics, config compare, input sampling, synapse plots, snapshots, tables, IClamp

## Backend Layout

Notebook-facing helpers are exposed through `modules.analysis.analysis_ui` for
backward-compatible imports. Internally, the public UI surface is organized in
`modules.analysis.ui`:

- `selection.py`: run selection and run-resolution helpers
- `outputs.py`: output plotting UI and execution helpers
- `inputs.py`: input plotting UI and execution helpers
- `metrics.py`: output metric tables/distribution helpers
- `extra.py`: optional Extra Analysis helpers

The private `modules.analysis.ui._engine` module contains shared implementation
used by those section modules.

## Defaults Loading Order

`6_analysis.ipynb` first loads `modules/analysis/analysis_defaults.json`, then
loads preset files referenced by these keys:

- `output_plot_preset_path`
- `input_plot_preset_path`
- `output_metrics_preset_path`
- `extra_preset_path`

Preset values update the notebook globals. Notebook cells can then override the
most common options for that interactive session.

## Main Default Files

### `analysis_defaults.json`

Global defaults and preset paths.

Important fields:

- `save_plots`: save generated figures
- `save_overwrite`: replace existing saved outputs
- `save_analysis`: save JSON/table artifacts
- `plots_dpi`: saved figure DPI
- `load_cell_for_analysis`: allow Step 6 to load the cell for richer tables
- `auto_run_outputs`: automatically build Output UI
- `auto_run_inputs`: automatically build Input UI
- `auto_plot_window_from_stim`: crop plots around detected stimulus windows
- `plot_window_adjustment_ms`: padding for auto-windowed plots
- `compare_list_paths_enabled`: include external compare paths by default
- `compare_list_paths`: external CSV curves and style metadata
- `compare_list_dir_paths`: folders scanned for compare curves
- `extra_mode`: default Extra Analysis mode

### `single_plot.json`

Compact all-in-one single-run figure defaults.

Important fields:

- `trial_idx`
- `top_input_groups`, `top_input_mode`, `top_layout`
- `include_input_raster`, `raster_input_groups`
- `include_output_raster`
- `output_curve_source`
- `output_recompute_bin_ms`, `output_recompute_smooth_ms`
- `output_mode`, `output_norm_kind`, `output_norm_window`
- `plot_window`, `auto_plot_window_from_stim`, `plot_window_adjustment_ms`
- `show_stim_lines`, `show_y_axes`, `show_y_axis_titles`
- `figsize`, `panel_height_ratios`
- `export_path`, `export_formats`, `export_overwrite`, `dpi`

### `output_plotting.json`

Output plot UI defaults.

Important fields:

- `plot_outputs`
- `plot_output_curve`
- `plot_spike_stats`
- `plot_raster`
- `output_plot_window_zero_origin`
- `raster_style`
- `plot_window`, `y_window`
- `output_stim_start_ms`, `output_stim_stop_ms`
- `win_size`
- `multi_plot_type`, `multi_shade_mode`
- `compare_output_layout`
- `output_linewidth`, `output_shade_alpha`
- `output_plot_data_path`
- `output_plot_export_type`
- `output_plot_data_format`
- `output_plot_data_auto_save`
- `output_curve_mode`
- `output_curve_plot_mode`
- `output_norm_mode`
- `output_norm_window`
- `output_bin_ms`
- `output_smooth_mode`

### `input_plotting.json`

Input plot UI defaults.

Important fields:

- `plot_inputs_mean`
- `plot_input_raster`
- `show_input_std`
- `input_source`
- `input_std_mode`
- `input_groups`
- `input_bin_ms`
- `input_smooth_ms`
- `input_raster_trial_idx`
- `input_raster_max_trains`
- `input_raster_win_size`
- `input_raster_style`
- `input_plot_window`
- `input_legend_loc`
- `compare_input_layout`
- `compare_show_input_std`
- `input_plot_data_path`
- `input_plot_data_format`
- `input_plot_data_auto_save`

### `output_metrics.json`

Output metric definitions and display defaults.

Important fields:

- `output_baseline_center_ms`
- `output_metric_mode`
- `output_metric_window_ms`
- `output_peak_window_ms`
- `output_drop_window_ms`
- `output_rebound_window_ms`
- `output_auc_window`
- `output_t50_mode`
- `output_show_metric_points`
- `output_metric_label_points`
- `output_metrics_show_params`
- `output_metrics_std_mode`
- `output_metrics_ref_label`
- `output_metrics_show_delta`
- `output_metrics_highlight_best`
- `output_metrics_plot_keys`
- `output_metrics_plot_style`
- `output_metrics_plot_show_points`
- `output_metrics_plot_show_error`
- `output_metrics_plot_save_plot`
- `output_metrics_plot_save_data`

### `extra_analysis.json`

Extra Analysis UI defaults.

Important fields:

- `input_sample_source`
- `input_sample_run`
- `input_sample_groups`
- `input_sample_runs`
- `input_sample_bin_ms`
- `input_sample_seed`
- `extra_cell_tables`
- `extra_geometry_tables`
- `extra_synapse_tables`
- `extra_recording_tables`
- `extra_compare_apply`
- `extra_compare_syn_groups_selector`
- `extra_compare_diff_only`
- `extra_snapshot_diff_only`
- `save_snapshot_compare_table`
- `snapshot_compare_scope`
- `snapshot_compare_format`
- `extra_synapse_show_table`
- `extra_synapse_trial_idx`
- `extra_synapse_weight_plot`
- `extra_synapse_distance_plot`
- `extra_synapse_scatter_plot`
- `extra_synapse_groups`
- `extra_synapse_weight_bin`
- `extra_synapse_distance_bin`
- `extra_synapse_plot_type`
- `extra_synapse_density`

## Extra Analysis Modes

### Output metrics

Computes metrics from output-rate curves. Works for selected single runs,
compare lists, and compatible external curves.

### Compare configs

Compares saved run configs against the currently selected tune. The
`extra_compare_apply` list controls which config families are included.

Supported apply values:

- `sim_config`
- `cell_config`
- `geometry`
- `syn_config`
- `syn_groups`
- `fit_json`

### Input sampling

Samples generated inputs from synapse configs without rerunning Step 5. This is
useful for checking stochastic input-generation behavior.

### Synapse plots

Uses saved Step 5 synapse records to plot placement/weight summaries.

Required data:

- `syn_records`, or
- `syn_records_by_trial`

If those records are absent, the mode prints a skip message instead of failing.

### Snapshot compare

Compares run snapshots/manifests. Use this for debugging reproducibility between
notebook, CLI, and SLURM outputs.

### Single-run tables

Generates optional tables for:

- cell sections
- mechanisms
- geometry distances
- synapse records
- cell recordings
- total synaptic traces

Cell and geometry tables require `load_cell_for_analysis = true`.

### IClamp analysis

Summarizes current-injection runs. It skips regular `multi` runs.

## Plotted-Data CSV Export

The Outputs and Inputs panels include CSV export controls.

Export formats:

- `trace_rows`: one row per plotted trace
- `long_rows`: one row per plotted point

If `CSV path` is blank, Step 6 can generate an automatic path. If it is only a
filename, it saves under the relevant Step 6 plot-data output folder.

Existing files are not overwritten unless overwrite is explicitly enabled.

## Utility Scripts

### Spikes CSV export

Use `scripts/export_spikes_csv.py` to convert `spikes.npz` to a row-per-trial
CSV.

```bash
python scripts/export_spikes_csv.py --input cells/PV/tunes/seg_tuned/output_data/<run_name>/results/spikes.npz
```

You can also pass a run directory:

```bash
python scripts/export_spikes_csv.py --input cells/PV/tunes/seg_tuned/output_data/<run_name>
```

### Vm trace swap utility

Use `scripts/swap_vm_trace.py` to replace saved exemplar data in an existing run
without rerunning a full batch. Default behavior is dry-run; add `--write` to
apply changes.

Examples:

```bash
python scripts/swap_vm_trace.py \
  --target-run cells/SST/tunes/seg_tuned/output_data/<target_run> \
  --source-run cells/SST/tunes/seg_tuned/output_data/<source_run> \
  --update vm
```

```bash
python scripts/swap_vm_trace.py \
  --target-run cells/SST/tunes/seg_tuned/output_data/<target_run> \
  --rerun \
  --update both \
  --write
```

## Compare Path Syntax

```text
path/to/file.csv@shift:scale;color=red;label=Bio;linestyle=--
```

Notes:

- `shift` is in ms.
- `scale` is multiplicative.
- `@shift`, `@shift:scale`, and `@shift,scale` are accepted.
- Style keys are `color`, `label`, `linestyle`, `shift`, and `scale`.

## Common Failure Modes

- Missing run folder: check Selection path and run token.
- External CSV not shown: check path, delimiter, and compare-path syntax.
- Synapse plot skipped: selected run lacks saved synapse records.
- Recording table empty: selected run lacks `cell_recordings` or total synaptic traces.
- Cell/geometry table skipped: `load_cell_for_analysis` is false or cell loading failed.
- Colab widget not shown: rerun setup/import cells and verify `ipywidgets`.
