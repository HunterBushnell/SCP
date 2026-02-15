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
- Normalized firing-rate curves.
- Bio-curve overlays for comparison.
- Snapshot comparisons between notebook and SLURM runs.

Outputs
- Plots saved by the notebook.
- Optional summary CSVs saved under comparison folders.

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
