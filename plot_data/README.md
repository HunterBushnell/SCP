# SCP plot-data exports

This folder stores exported plot data (`.csv`) and companion figure images (`.png`) for the paper plots.
Each CSV row is one plotted artist sample (line point, vertical line, or metric marker), not one simulation trial.

## Folder layout

- `all_cells/raw` and `all_cells/normalized`: PN+PV+SST comparison overlays.
- `PV/raw` and `PV/normalized`: PV-focused final + parameter sweeps.
- `SST/raw` and `SST/normalized`: SST-focused final + parameter sweeps.

`raw` uses firing-rate units in Hz, while `normalized` stores the normalized version of the same plotting flow.

## CSV schema

All CSV files in this directory tree use the same columns:

- `figure_type`: Plot family (`output` or `input`); these files are `output`.
- `mode`: Plot context (`single`, `compare`, `compare_list`); these files are `compare_list`.
- `plot_name`: Internal figure identifier (here, `compare_output_curve_list`).
- `axis_index`: Zero-based subplot index (all current files use `0`).
- `trace_label`: Trace/legend label (or fallback label for unlabeled artists).
- `series_kind`: Encoded artist type.
- `run_label`: Export context label (currently `compare_list`).
- `units`: Y-axis units (`Rate (Hz)` or `Rate (normalized)`).
- `time_ms`: X-axis coordinate in milliseconds.
- `value`: Y value for point-like artists (`line`, `scatter`).
- `value_low`: Lower bound for interval-like artists (`vline`, shaded ranges).
- `value_high`: Upper bound for interval-like artists (`vline`, shaded ranges).

## `series_kind` used here

- `line`: Main output-rate curve samples.
- `vline`: Vertical stimulus boundary markers (typically around 500 ms and 1000 ms).
- `scatter`: Metric markers plotted on curves (peak, +100 ms drop, +300 ms rebound).

## Quick usage

```python
import pandas as pd

df = pd.read_csv("SCP/plot_data/all_cells/raw/_final_tuned.csv")
curves = df[df["series_kind"] == "line"]
stim_lines = df[df["series_kind"] == "vline"]["time_ms"].unique()

# Example: one trace
sst_vip = curves[curves["trace_label"] == "SST VIPinh"][["time_ms", "value"]]
```

## File-by-file summary (CSV)

### All cells

- `SCP/plot_data/all_cells/raw/_final_base.csv`: Raw-rate all-cell baseline overlay with `PN bio`, `PV base`, and `SST base`.
- `SCP/plot_data/all_cells/raw/_final_tuned.csv`: Raw-rate all-cell tuned overlay with `PN bio`, `PV SSTinh`, `SST VIPinh`, and `SST GABABinh`.
- `SCP/plot_data/all_cells/normalized/_final_base.csv`: Normalized all-cell baseline overlay with `PN bio`, `PV base`, and `SST base`.
- `SCP/plot_data/all_cells/normalized/_final_tuned.csv`: Normalized all-cell tuned overlay with `PN bio`, `PV SSTinh`, `SST VIPinh`, and `SST GABABinh`.

### PV

- `SCP/plot_data/PV/raw/_final_PV.csv`: Raw-rate PV final comparison of `PV bio`, `PV base`, and `PV SSTinh`.
- `SCP/plot_data/PV/raw/_param_Rin_base_PV.csv`: Raw-rate PV Rin sweep comparing `PV base` with `PV High Rin` and `PV Low Rin` (with `PV bio` overlay).
- `SCP/plot_data/PV/raw/_param_STP_base_PV.csv`: Raw-rate PV STP sweep comparing `PV base` with `PV High STP` and `PV NO STP` (with `PV bio` overlay).
- `SCP/plot_data/PV/raw/_param_tau_base_PV.csv`: Raw-rate PV tau sweep comparing `PV base` with `PV High Tau` and `PV Low Tau` (with `PV bio` overlay).
- `SCP/plot_data/PV/normalized/_final_PV.csv`: Normalized PV final comparison of `PV bio`, `PV base`, and `PV SSTinh`.
- `SCP/plot_data/PV/normalized/_param_Rin_base_PV.csv`: Normalized PV Rin sweep comparing `PV base` with `PV High Rin` and `PV Low Rin`.
- `SCP/plot_data/PV/normalized/_param_STP_base_PV.csv`: Normalized PV STP sweep comparing `PV base` with `PV High STP` and `PV NO STP`.
- `SCP/plot_data/PV/normalized/_param_tau_base_PV.csv`: Normalized PV tau sweep comparing `PV base` with `PV High Tau` and `PV Low Tau`.

### SST

- `SCP/plot_data/SST/raw/_final_SST.csv`: Raw-rate SST final comparison across `SST bio`, `SST bio shifted`, `SST base`, `SST VIPinh`, and `SST GABABinh`.
- `SCP/plot_data/SST/raw/_param_NMDA_base_SST.csv`: Raw-rate SST NMDA sweep comparing `SST base` with `SST High NMDA (3.0)` and `SST Low NMDA (0)` plus bio overlays.
- `SCP/plot_data/SST/raw/_param_Rin_base_SST.csv`: Raw-rate SST Rin sweep comparing `SST base` with `SST High Rin` and `SST Low Rin` plus bio overlays.
- `SCP/plot_data/SST/raw/_param_STP_base_SST.csv`: Raw-rate SST STP sweep comparing `SST base` with `SST High STP (2x)` and `SST Low STP (0)` plus bio overlays.
- `SCP/plot_data/SST/raw/_param_tau_base_SST.csv`: Raw-rate SST tau sweep comparing `SST base` with `SST High Tau` and `SST Low Tau` plus bio overlays.
- `SCP/plot_data/SST/normalized/_final_SST.csv`: Normalized SST final comparison across `SST bio`, `SST bio shifted`, `SST base`, `SST VIPinh`, and `SST GABABinh`.
- `SCP/plot_data/SST/normalized/_param_NMDA_base_SST.csv`: Normalized SST NMDA sweep comparing `SST base` with `SST High NMDA (3.0)` and `SST Low NMDA (0)` plus bio overlays.
- `SCP/plot_data/SST/normalized/_param_Rin_base_SST.csv`: Normalized SST Rin sweep comparing `SST base` with `SST High Rin` and `SST Low Rin` plus bio overlays.
- `SCP/plot_data/SST/normalized/_param_STP_base_SST.csv`: Normalized SST STP sweep comparing `SST base` with `SST High STP (2x)` and `SST Low STP (0)` plus bio overlays.
- `SCP/plot_data/SST/normalized/_param_tau_base_SST.csv`: Normalized SST tau sweep comparing `SST base` with `SST High Tau` and `SST Low Tau` plus bio overlays.

## PNG companions

Each folder also includes `.png` figure exports corresponding to the same plot names as the CSV files.
Two PNG filenames currently have small naming mismatches versus their CSV counterparts:

- `SCP/plot_data/PV/normalized/_param_STP_base__PV.png` (extra underscore).
- `SCP/plot_data/SST/raw/_param_SSTP_base_SST.png` (`SSTP` vs `STP`).
