# Analysis Module (SCP)

This folder holds the analysis helpers used by `6_analysis.ipynb`.

## What lives here
- `analysis.py`: core curve/metrics logic and helpers
- `plotting.py`: plotting routines for outputs/inputs/synapses
- `analysis_ui.py`: Jupyter UI builders and glue code
- `bio_curve.py`: CSV loading helper for bio curves
- `analysis_defaults.json`: default UI/options (used by `6_analysis.ipynb`)
- `analysis_presets/`: optional presets (ex: `paper_compare.json`, `output_plotting.json`, `input_plotting.json`, `output_metrics.json`, `extra_analysis.json`)

## Recording summaries
- `analysis.summarize_cell_recordings(results, ...)`: per-site/per-variable summary for `cell_recordings`.
- `analysis.summarize_total_synaptic_traces(results, ...)`: summary for total synaptic `I`/`G` traces (if saved).
- Formatting helpers:
  - `analysis.format_cell_recording_summary_table(...)`
  - `analysis.format_cell_recording_summary_compare(...)`
  - `analysis.format_total_synaptic_trace_table(...)`
  - `analysis.format_total_synaptic_trace_compare(...)`

## Defaults + presets
- Defaults are loaded in `6_analysis.ipynb` from `modules_local/analysis/analysis_defaults.json`.
- Output plotting defaults are loaded from `modules_local/analysis/analysis_presets/output_plotting.json`
  using `output_plot_preset_path` in `analysis_defaults.json`.
- Input plotting defaults are loaded from `modules_local/analysis/analysis_presets/input_plotting.json`
  using `input_plot_preset_path` in `analysis_defaults.json`.
- Output metrics defaults are loaded from `modules_local/analysis/analysis_presets/output_metrics.json`
  using `output_metrics_preset_path` in `analysis_defaults.json`.
- Extra analysis defaults are loaded from `modules_local/analysis/analysis_presets/extra_analysis.json`
  using `extra_preset_path` in `analysis_defaults.json`.
- Presets can be toggled via the Outputs UI (Paper compare) or set in the JSON.

## Compare paths syntax
When using Compare paths (comma-separated list), each entry can include shift/scale
and optional style metadata:

```
path/to/file.csv@shift:scale;color=red;label=Bio;linestyle=--
```

Notes:
- `shift` is in ms, `scale` is multiplicative.
- `@shift` alone is allowed; `@shift:scale` or `@shift,scale` both work.
- Keys are `color`, `label`, `linestyle`, `shift`, `scale`.

Per-item enable/disable
You can also use object entries in `compare_list_paths` to toggle paths on/off:

```
{
  "path": "/abs/path/file.csv@500",
  "enabled": false,
  "color": "0.4"
}
```

## UI help
Each UI section (Selection, Outputs, Inputs, Extra) includes a **Help** button
that prints a short guide into the cell output.

## JSON-only knobs (examples)
Some options are only in the defaults JSON:
- `output_compare_figsize`, `output_compare_panel_size`
- `output_metric_window_markers`, `output_metric_label_points`
- `compare_preset_path`
- `extra_snapshot_*` settings

Keep this file lightweight; the JSON is the source of truth for defaults.
