# Step 2: Passive Tuning

Step 2 runs passive current-injection traces and diagnostics for a tune prepared
by Step 1. When targets and ACT are explicitly selected, it can also compute
candidate passive values while leaving final model edits under user control.

Notebook: `../../2_passive.ipynb`

## Purpose

Use Step 2 to:

- build and inspect the selected tune's NEURON cell,
- compile/load the tune's mechanisms when needed,
- optionally compute ACT-compatible passive values from measured targets,
- manually apply those values to the model files,
- rerun a simple current-injection check,
- optionally plot/export trace diagnostics.

Step 2 does **not** overwrite model files automatically. This is intentional:
the correct target fields depend on the cell loader and model source format.

## Expected Inputs

Step 2 expects a tune directory created or validated by Step 1:

```text
cells/<CELL>/tunes/<TUNE>/
├── <loader-owned model sources>
├── <configured MOD directory>/ # optional; selected by paths.modfiles
└── cell_configs/
    ├── cell_config.json
    ├── sim_config.json
    ├── target_config.json       # optional
    └── geometry.json
```

The selected loader resolves native model sources. A target config and ACT are
not required for targetless trace checks. Allen-specific proposal work may also
use the fit JSON referenced by its manifest.

## Local and Colab Use

`2_passive.ipynb` is the primary Step 2 entry point for both local and Colab
use. The notebook can:

- add the SCP checkout to `sys.path`,
- locate or clone ACT only when ACT-dependent work is selected,
- compile mechanisms through `nrnivmodl`,
- load the selected tune,
- run the passive protocol.

The root `2_passive.ipynb` is the workflow to use for local and Colab runs.

## Notebook Workflow

### 2.1 Select Tune Directory

Choose the cell and tune, for example:

```python
cell_name = "PV"
tune_name = "tuned"
```

The selected tune should already satisfy the Step 1 contract.

### 2.2 Compile and Load Mechanisms

Compile the configured `paths.modfiles` only when `.mod` sources exist and
compiled outputs are missing or stale. Models using only built-in mechanisms
skip this phase.

### 2.3 Build Cell

Build the selected NEURON cell and review the section/area summary. This is the
cell object used for current-injection checks and, when requested, the optional
ACT passive calculation.

### 2.4 Enter Passive Targets

Step 2 resolves passive targets from `cell_configs/target_config.json` using
`target_source.mode`:

- `none` or a missing target config: run passive traces without target/ACT proposals.
- `manual`: read direct values from `manual.passive`.
- `traces`: calculate passive values from a user-provided trace file using
  `traces.passive`.
- `allen_nwb`: calculate passive values from an Allen/ADB `.nwb` file using
  `allen_nwb.passive`.

For exact passive trace file requirements, see
`docs/reference/target_trace_formats.md`.

Manual passive fields use user-facing units:

- `manual.passive.rin_MOhm`: input resistance in MΩ,
- `manual.passive.tau_ms`: membrane time constant in ms,
- `manual.passive.v_rest_mV`: resting membrane voltage in mV.

Notebook `manual_passive_targets` values override the config only when a field
is not `None`. Set `COMPUTE_ACT_PASSIVE_PROPOSAL = True` to explicitly request
the ACT proposal calculation; it requires all three passive targets, either
directly or from extraction. Reading or comparing manual targets alone does not
import ACT. Targetless trace checks do not use hidden example defaults.

Generic trace mode expects either CSV or NPY traces. The recommended CSV
contract is:

- `time_ms`: time in ms, starting at 0 for the trace,
- `voltage_mV`: voltage in mV,
- optional `current_pA`: injected current in pA,
- optional `sweep`: sweep/trace identifier for multiple traces.

If `current_pA` is not present, provide `traces.passive.stim_start_ms`,
`traces.passive.stim_stop_ms`, and `traces.passive.current_pA`. For NPY traces,
provide `dt_ms`, `stim_start_ms`, `stim_stop_ms`, and `current_pA`.

For Allen/ADB ephys files, set `target_source.mode = "allen_nwb"`, place the
`.nwb` file in the tune directory, and set:

```json
"allen_nwb": {
  "file": "<allen_ephys_result>_ephys.nwb",
  "sweep_ids": [],
  "passive": {
    "stimulus_names": ["Long Square"],
    "sweep_ids": null,
    "min_current_pA": null,
    "max_current_pA": -1.0,
    "end_margin_ms": 10.0,
    "reducer": "median",
    "tau_field": "tau_avg_ms"
  }
}
```

By default, Allen/NWB extraction uses `Long Square` negative-current sweeps and
writes review CSVs to `notebook_exports/step2_passive/`. Set
`APPLY_EXTRACTED_PASSIVE_TARGETS_TO_CONFIG = True` only when you want extracted
targets written back into `manual.passive`; otherwise they are used only for the
current notebook run.

When `COMPUTE_ACT_PASSIVE_PROPOSAL` is enabled, the notebook converts passive
targets into ACT settable passive properties:

- `e_rev_leak`,
- `g_bar_leak`,
- `Cm`.

Area handling follows ACT guidance:

- `auto`: soma area for simple one-section cells, total area for detailed cells,
- `soma`: soma area explicitly,
- `total`: total reconstructed cell area explicitly,
- `custom`: `custom_passive_area_cm2`.

### 2.5 Run Passive Protocol

Run a small current-injection check using negative current steps by default. The
notebook reports passive diagnostic rows and records voltage traces for plotting.

When passive targets are present, the notebook also displays a comparison table
for:

- resting voltage (`v_rest_mV`),
- input resistance (`rin_MOhm`),
- membrane time constant (`tau_ms`).

The table reports target value, measured value, signed difference, absolute
difference, and percent error for each simulated current step where the metric
can be measured.

### 2.6 Plot and Export Trace Check

The plot cell displays membrane-voltage traces and can optionally export figures
to:

```text
cells/<CELL>/tunes/<TUNE>/notebook_exports/
```

This folder is notebook-only scratch output. It is separate from Step 5
simulation output and Step 6 analysis output.

Important controls:

- `PLOT_XLIM`, `PLOT_YLIM`: voltage plot limits.
- `TRACE_COLOR`: voltage trace color when one current is plotted; multiple
  currents use distinct Matplotlib colors.
- `EXPORT_FIGURE`: save figure exports to `notebook_exports/`.

## Outputs

Step 2's primary output is a manually updated model/config state inside the tune
directory. Optional notebook diagnostics may be written to `notebook_exports/`.

Step 2 does not create a pipeline run under `output_data/`.

## Troubleshooting

- **ACT not found**: this matters only for ACT-based estimates. Install ACT at
  `../mods/ACT`, set `SCP_ACT_PATH`, or let the Colab bootstrap clone it.
- **`nrnivmodl` not found**: activate the project environment, then rerun the
  compile/load cell.
- **Values do not change the trace**: confirm the fit/config file was saved,
  restart the kernel, then rerun from the environment setup through
  **2.3 Build Cell** so NEURON constructs the updated model once. Legacy Allen
  models cannot be constructed twice in one NEURON process.
- **Unexpected passive estimates**: verify `passive_area_mode`,
  `passive_area_scale`, and whether active conductances are disabled for the
  passive check.
- **Benign NEURON warnings**: some ADB loaders print mechanism/ion warnings
  during build. Treat them as non-blocking unless the cell fails to build or run.
