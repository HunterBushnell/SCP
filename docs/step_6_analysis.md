Step 6: Analysis

`6_analysis.ipynb` is the terminal analysis notebook for the SCP pipeline.
It is used for comparing runs, plotting aggregates, and generating figures.

Inputs
- One or more run folders under `output_data/`.
- Optional comparison data under `_comparisons/`.
- Defaults are loaded from `modules_local/analysis/analysis_defaults.json`.
- Optional presets live under `modules_local/analysis/analysis_presets/`.

Common tasks
- Multi-trial averages with optional shaded error bands.
- Normalized firing-rate curves and optional inter-spike-interval (ISI) curves.
- Bio-curve overlays for comparison.
- Snapshot comparisons between notebook and SLURM runs.
- Recording summaries for:
  - `cell_recordings` (site/variable traces).
  - total synaptic traces `traces.I` / `traces.G` (if recorded).

Outputs
- Plots saved by the notebook.
- Optional summary CSVs saved under comparison folders.
- Optional "plotted-data CSV" exports from the Outputs/Inputs widget panels.

Plotted-data CSV export
- The Outputs and Inputs panels each include `CSV path`, `Auto-save CSV`, and `Save plotted CSV`.
- Export includes plotted artists from the current figure(s), including raster and shaded bands when shown.
- `Format` toggle controls layout:
  - `Trace rows`: one row per plotted trace (`trace_name`) with `|`-separated vectors in
    `time_ms`, `value`, `value_low`, `value_high`.
  - `Long rows`: one row per plotted point (tidy format with repeated metadata columns).
- If `CSV path` is only a filename (for example `my_plot`), it saves to:
  `.../output_data/plot_data/my_plot.csv`
- If the path is blank, an automatic name is generated using figure type/mode/timestamps.
- Existing files are never overwritten; export prints a warning and skips save.

Spikes CSV export (row-per-trial)
- Use `scripts/export_spikes_csv.py` to export `spikes.npz` to a simple CSV where each row is one trial.
- Output columns:
  - `trial_n` (for example `trial_0`)
  - `n_spikes`
  - `spike_times_ms` (`|`-separated times in ms)
- Example:
  - `python scripts/export_spikes_csv.py --input cells/PV/tunes/seg_tuned/output_data/<run_name>/results/spikes.npz`
- You can also pass a run directory:
  - `python scripts/export_spikes_csv.py --input cells/PV/tunes/seg_tuned/output_data/<run_name>`

Notebook helper cell (5_local or 6_analysis)
```python
from modules_local.analysis import analysis

out_csv = analysis.export_spikes_trials_csv(
    "cells/PV/tunes/seg_tuned/output_data/<run_name>/results/spikes.npz",
    out_csv=None,          # or "my_spikes_trials.csv"
    delimiter="|",         # delimiter inside spike_times_ms
    precision=10,
    overwrite=False,
)
print(out_csv)
```

Compare paths syntax
Compare list entries can include shift/scale and style metadata:

```
path/to/file.csv@shift:scale;color=red;label=Bio;linestyle=--
```

Notes
- `shift` is in ms, `scale` is multiplicative.
- `@shift` alone is allowed; `@shift:scale` or `@shift,scale` both work.
- Keys are `color`, `label`, `linestyle`, `shift`, `scale`.
- Older references to `5_analysis.ipynb` should be replaced with `6_analysis.ipynb`.

Recording summary helpers
```python
from modules_local.analysis import analysis
from modules_local import run_sim

res = run_sim.load_results("cells/SST/tunes/seg_tuned/output_data/<run_name>")

cell_sum = analysis.summarize_cell_recordings(res)
print(analysis.format_cell_recording_summary_table(cell_sum, title="Cell recordings"))

syn_sum = analysis.summarize_total_synaptic_traces(res)
print(analysis.format_total_synaptic_trace_table(syn_sum, title="Total synaptic I/G"))
```
