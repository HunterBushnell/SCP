# Step 2: Passive Tuning

Step 2 estimates passive membrane parameters for a tune prepared by Step 1. It
uses ACT's passive helper to compute candidate values, then leaves final model
edits under user control.

Notebook: `../../2_passive.ipynb`

## Purpose

Use Step 2 to:

- build and inspect the selected tune's NEURON cell,
- compile/load the tune's mechanisms when needed,
- compute ACT-compatible passive values from measured targets,
- manually apply those values to the model files,
- rerun a simple current-injection check,
- optionally plot/export trace diagnostics.

Step 2 does **not** overwrite model files automatically. This is intentional:
the correct target fields depend on the cell loader and model source format.

## Expected Inputs

Step 2 expects a tune directory created or validated by Step 1:

```text
cells/<CELL>/tunes/<TUNE>/
├── manifest.json
├── modfiles/
└── cell_configs/
    ├── cell_config.json
    ├── sim_config.json
    └── geometry.json
```

For ADB/Allen examples, Step 2 also expects the model fit JSON referenced by
the manifest, such as `*_fit.json`.

## Local and Colab Use

`2_passive.ipynb` is the primary Step 2 entry point for both local and Colab
use. The notebook can:

- add the SCP checkout to `sys.path`,
- locate or clone ACT when configured,
- compile mechanisms through `nrnivmodl`,
- load the selected tune,
- run the passive protocol.

The root `2_passive.ipynb` is the workflow to use for local and Colab runs.

## Notebook Workflow

### 2.1 Select Tune Directory

Choose the cell and tune, for example:

```python
cell_name = "SST"
tune_name = "seg_tuned"
```

The selected tune should already satisfy the Step 1 contract.

### 2.2 Compile and Load Mechanisms

Compile the tune's `modfiles/` only when compiled mechanism outputs are missing
or stale. The shared mechanism helper searches for `nrnivmodl` on `PATH` and
next to the active Python executable, which covers typical Conda environments.

### 2.3 Build Cell

Build the selected NEURON cell and review the section/area summary. This is the
cell object used for the ACT passive calculation and current-injection check.

### 2.4 Enter Passive Targets

Enter measured passive targets in user-facing units:

- `target_rin_mohm`: input resistance in MΩ,
- `target_tau_ms`: membrane time constant in ms,
- `target_vrest_mv`: resting membrane voltage in mV.

The notebook converts these into settable passive properties:

- `e_rev_leak`,
- `g_bar_leak`,
- `Cm`.

Area handling follows ACT guidance:

- `auto`: soma area for simple one-section cells, total area for detailed cells,
- `soma`: soma area explicitly,
- `total`: total reconstructed cell area explicitly,
- `custom`: `custom_passive_area_cm2`.

Use `passive_area_scale` only when ACT's analytical estimate should use a scaled
effective area without changing the morphology.

### Manual Application

For bundled ADB/Allen-style tune directories:

1. Open the `*_fit.json` file printed by the notebook.
2. During passive tuning, disable or minimize active conductances according to
   your passive-tuning convention while keeping leak/passive terms available.
3. Apply the ACT values to the model's passive fields:
   - `e_rev_leak` maps to leak reversal potential, commonly `e_pas`;
   - `g_bar_leak` maps to leak conductance, commonly `g_pas`;
   - `Cm` maps to membrane capacitance, commonly section `cm`.
4. Preserve the existing JSON structure.
5. Save the file and rerun from **2.3 Build Cell**.

For generic/custom models, apply the same conceptual values wherever that model
defines passive parameters. SCP does not assume a universal file layout for
custom model sources.

### 2.5 Run Passive Protocol

Run a small current-injection check using negative current steps by default. The
notebook reports passive diagnostic rows and records voltage traces for plotting.

### 2.6 Plot and Export Trace Check

The plot cell displays membrane-voltage traces and can optionally export figures
to:

```text
cells/<CELL>/tunes/<TUNE>/notebook_exports/
```

This folder is notebook-only scratch output. It is separate from Step 5
simulation output and Step 6 analysis output.

## Outputs

Step 2's primary output is a manually updated model/config state inside the tune
directory. Optional notebook diagnostics may be written to `notebook_exports/`.

Step 2 does not create a pipeline run under `output_data/`.

## Troubleshooting

- **ACT not found**: install ACT at `../mods/ACT`, set `SCP_ACT_PATH`, or let the
  Colab bootstrap clone it.
- **`nrnivmodl` not found**: activate the project environment, then rerun the
  compile/load cell.
- **Values do not change the trace**: confirm the fit/config file was saved,
  then rerun from **2.3 Build Cell** so NEURON rebuilds the model.
- **Unexpected passive estimates**: verify `passive_area_mode`,
  `passive_area_scale`, and whether active conductances are disabled for the
  passive check.
- **Benign NEURON warnings**: some ADB loaders print mechanism/ion warnings
  during build. Treat them as non-blocking unless the cell fails to build or run.
