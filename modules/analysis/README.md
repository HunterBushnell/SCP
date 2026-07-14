# Analysis Module (SCP)

This folder holds the analysis helpers and defaults used by `6_analysis.ipynb`.

## Files

- `analysis.py`: core curve, metric, table, snapshot, and export helpers.
- `plotting.py`: plotting routines for outputs, inputs, and synapse records.
- `analysis_ui.py`: stable facade for Jupyter widget builders.
- `ui/`: section-organized Step 6 UI package.
- `single_plot_panel.py`: compact single-plot wrapper used by Step 6.
- `bio_curve.py`: external CSV curve loading helper.
- `analysis_defaults.json`: global Step 6 defaults and preset paths.
- `analysis_presets/`: JSON preset files for specific UI sections.

## Presets

`6_analysis.ipynb` loads `analysis_defaults.json`, then loads preset files
referenced by these fields:

- `output_plot_preset_path` -> `analysis_presets/output_plotting.json`
- `input_plot_preset_path` -> `analysis_presets/input_plotting.json`
- `output_metrics_preset_path` -> `analysis_presets/output_metrics.json`
- `extra_preset_path` -> `analysis_presets/extra_analysis.json`

Other presets, such as `analysis_presets/single_plot.json`, are used by specific
notebook sections or when explicitly selected.

## Main UI Builders

- `analysis_ui.build_selection_ui(...)`
- `analysis_ui.build_outputs_ui(...)`
- `analysis_ui.build_inputs_ui(...)`
- `analysis_ui.build_extra_ui(...)`

Each UI section has a Help button that prints a concise guide into the notebook
output.

For backend development, import narrower section modules from
`modules.analysis.ui`:

- `modules.analysis.ui.selection`
- `modules.analysis.ui.outputs`
- `modules.analysis.ui.inputs`
- `modules.analysis.ui.metrics`
- `modules.analysis.ui.extra`

## Recording and Synapse Summaries

- `analysis.summarize_cell_recordings(results, ...)`
- `analysis.summarize_total_synaptic_traces(results, ...)`
- `analysis.summarize_synapse_records(results, ...)`
- `analysis.format_cell_recording_summary_table(...)`
- `analysis.format_total_synaptic_trace_table(...)`
- `analysis.format_synapse_summary_table(...)`

These helpers are data-dependent. They only report records saved by Step 5.

## Compare Path Syntax

Compare paths can include shift/scale and optional style metadata:

```text
path/to/file.csv@shift:scale;color=red;label=Bio;linestyle=--
```

Notes:

- `shift` is in ms.
- `scale` is multiplicative.
- `@shift`, `@shift:scale`, and `@shift,scale` are accepted.
- Keys include `color`, `label`, `linestyle`, `shift`, and `scale`.

Object entries are also supported in `compare_list_paths`:

```json
{
  "path": "external_data/pyrFiringRateAvg.csv@290",
  "enabled": true,
  "color": "k",
  "label": "PN bio"
}
```

## User Docs

- Main guide: `docs/guides/analysis.md`
- Detailed Step 6 reference: `docs/guides/step_6_analysis.md`
