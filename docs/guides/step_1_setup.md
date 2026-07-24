# Step 1: Setup

Step 1 prepares a tune directory so later notebooks and Step 5 can load,
simulate, and validate the model consistently.

Primary entry points:

- Notebook: `1_setup.ipynb`
- CLI: `scripts/step1_prepare.py`
- Backend: `modules/setup/`

## Purpose

Use Step 1 when you need to:

- download or stage a cell model,
- compile NEURON mechanisms,
- create or refresh `cell_configs/`,
- create or refresh target-data configuration for Steps 2-3,
- scaffold optional synapse config templates,
- validate that a tune is ready for later pipeline steps.

Step 1 does not tune model parameters and does not run final simulations.

## What Step 1 Produces

For a tune directory such as `cells/PV/tunes/orig`, Step 1 prepares:

```text
<loader-owned model sources>
modfiles/                         # optional
cell_configs/
  cell_config.json
  sim_config.json
  target_config.json              # optional
  geometry.json
  syn_config.json
  syn_groups/
    input_blocks_template.json
```

Synapse configs are optional. They can be skipped for cell-only or IClamp work.

## Main Decisions

Before running Step 1, decide:

- `cell_name`: display/model label, such as `PV`, `SST`, or `PN`.
- `tune_name`: tune folder name, usually `orig` for raw setup or `tuned` for the working model.
- Whether the model is already staged under the tune or must be downloaded from
  the Allen Database.
- For a new Allen download or an ambiguous/custom staged layout, the values in
  the notebook's single `MODEL_SOURCE_OVERRIDES` block.
- `CONFIG_MODE`: `fill`, `overwrite`, or `skip` for generated configs.
- Target source mode: `none`, `manual`, `traces`, or `allen_nwb` for Step 2-3 targets.
- Synapse scaffolding: create `input_blocks_template.json`, create an empty `syn_config.json`, or skip synapse configs.

Recommended default for normal work:

```python
CONFIG_MODE = "fill"
```

Use `overwrite` only when intentionally resetting generated configs.

## Notebook Workflow

Open `1_setup.ipynb` and run the phases in order:

1. Select tune directory.
2. Resolve the staged model source automatically, or provide a source override
   for an Allen download or custom layout.
3. Compile/load mechanisms.
4. Scaffold base configs.
5. Edit tune-specific display/geometry compatibility and runtime-condition
   values in the generated JSON files.
6. Scaffold target config for manual targets, user traces, or Allen/ADB NWB data.
7. Optionally scaffold synapse configs.
8. Validate setup.
9. Review expected paths.
10. Optionally create the `tuned` working copy.
11. Optionally continue to segmentation/segregation.

The notebook runs locally or in Colab. In a fresh Colab session it can clone the
repo and install dependencies using the same environment variables as Step 5.

## Tune Selection

Common bundled setup targets:

- PV perisomatic raw setup: `cells/PV/tunes/orig`
- SST all-active raw setup: `cells/SST/tunes/orig`

Common tuned simulation examples:

- `cells/PV/tunes/tuned`
- `cells/SST/tunes/tuned`

The `orig` tunes are raw setup references. The `tuned` tunes are the working
models used after copying, optional segmentation, and Steps 2-4 tuning.

## Cell Source Types

Section 1.2 uses this precedence:

1. Reuse an existing `cell_configs/cell_config.json` as the tune's source of
   truth.
2. Otherwise, discover exactly one staged `manifest.json`.
3. Otherwise, discover exactly one `begintemplate` declaration in a staged HOC
   file.

A standard tune-local `modfiles/` directory is detected with the HOC source.
Leave `MODEL_SOURCE_OVERRIDES = None` for these common cases. Use the override
only for a new Allen download, multiple candidate sources, a nonstandard path,
constructor arguments, or a noncanonical HOC section map.

### `source_type = "adb"`

Downloads or refreshes an Allen Database model bundle.

Required inputs:

- `specimen_id`
- `model_type`: usually `perisomatic` or `all active`

To find an Allen specimen, use the Allen Cell Types Database search page and
open the electrophysiology page for a selected cell. Direct cell pages follow
this pattern:

```text
https://celltypes.brain-map.org/experiment/electrophysiology/<specimen_id>
```

For example, the PV example is:

```text
https://celltypes.brain-map.org/experiment/electrophysiology/484635029
```

Use the Allen page to review cell metadata and optionally click **Download
Data** to download the ephys `.nwb` file for Step 2 passive targets and Step 3
FI targets. Step 1 itself downloads the biophysical model bundle through
AllenSDK using `specimen_id` and `model_type`; the ephys `.nwb` file is a
separate manual download.

For a new Allen download, put the setup inputs in the one optional block in
section 1.2:

```python
MODEL_SOURCE_OVERRIDES = {
    "source_type": "adb",
    "cell_loader": "allen_manifest",
    "specimen_id": 484635029,
    "model_type": "perisomatic",
}
```

ADB model types can differ structurally. Perisomatic and all-active bundles may
have different manifest/model/fit-file layouts, so keep `model_type` explicit
when preparing ADB tunes.

### `source_type = "existing"`

Uses model files already staged in the tune directory. Construction is then
validated by the selected loader: an Allen manifest bundle for
`allen_manifest`, or an object-owned HOC source/template contract for
`hoc_template`.

Use this when files already exist locally and you only want to compile,
scaffold configs, or validate.

Source acquisition and construction are separate: `existing` never downloads
or rewrites native model sources. See `../reference/model_loaders.md` for the
generic HOC configuration and canonical section contract.

For a simple staged HOC model, no section 1.2 edit is needed. If discovery is
ambiguous or the template needs constructor arguments or custom section names,
use the same JSON-shaped blocks accepted by `cell_config.json`:

```python
MODEL_SOURCE_OVERRIDES = {
    "cell_loader": "hoc_template",
    "paths": {
        "hoc_template": "model/CellTemplate.hoc",
        "modfiles": "modfiles",
    },
    "hoc_template": {
        "template_name": "CellTemplate",
        "constructor_args": [1.0],
        "section_map": {"soma": "body", "all": "all_sections"},
    },
}
```

## Mechanisms

Step 1 can compile NEURON mechanisms from:

```text
<tune_dir>/modfiles/
```

The directory is resolved from `cell_config.paths.modfiles`. It may use another
tune-relative path or be `null`. If there are no `.mod` sources, compilation and
DLL loading are skipped and built-in NEURON mechanisms remain available.

Notebook controls:

- `DO_COMPILE_MODFILES`
- `RECOMPILE_MODFILES`
- `LOAD_COMPILED_DLL`
- `SORT_GENOME_ENTRIES_BY_SECTION`
- `COERCE_GENOME_VALUES_TO_NUMERIC`

All-active ADB bundles sometimes need fit JSON cleanup before validation. Use
the genome cleanup toggles only when needed.

## Base Configs

Base config scaffolding creates or updates:

- `cell_config.json`
- `sim_config.json`
- `geometry.json`

`CONFIG_MODE` controls behavior:

- `fill`: create missing files and fill missing keys in existing files.
- `overwrite`: replace files with defaults.
- `skip`: do not modify files.

Generated defaults include:

- `cell_config.json`: cell label, tune name, loader-owned paths/options, and
  display metadata. New files receive a neutral/default plot color. Allen tunes
  also receive `soma_diam_multiplier: 1.0`; HOC geometry is not rescaled.
- `sim_config.json`: simulation timing, explicit HOC runtime conditions, trial/save settings, plotting, IClamp, recording, randomness, and snapshot defaults.
- `geometry.json`: soma-origin distance reference and proximal/distal thresholds.

The notebook intentionally does not hold cell-specific color, Allen diameter
multiplier, initialization voltage, or temperature values. After section 1.4,
edit the generated tune-local files directly:

- `cell_config.json`: set `color`; for an Allen tune, change
  `tuning.soma_diam_multiplier` only when needed.
- `sim_config.json`: set `conditions.v_init_mV` and `conditions.celsius_C`.

Fresh notebook scaffolds leave both runtime-condition values as `null` so they
cannot silently imply a scientific choice. A new HOC tune must set finite values
before section 1.7. In the normal `fill` mode, rerunning section 1.4 preserves
these edited values. `overwrite` intentionally resets the generated files.

See `../reference/configs_reference.md` for every generated field.

## Target Config

Target config scaffolding creates or updates:

- `target_config.json`

This file tells Steps 2-3 what biological or experimental target data to use.
In the notebook, select only the source mode. Step 1 generates a blank,
mode-specific template; then edit the generated tune-local JSON with that
cell's values or paths. This keeps cell-specific target data out of the setup
notebook.

Source modes:

- `none`: no target data; use Steps 2–3 for intrinsic traces and FI diagnostics.
- `manual`: fill `manual.passive` values and `manual.fi_curve` points directly
  in the generated config, or point `manual.fi_curve.csv` to a summarized FI CSV.
- `traces`: point to user-provided passive or active trace files.
- `allen_nwb`: point to an Allen/ADB electrophysiology `.nwb` file.

`manual` is the blank-template default for both Allen and non-Allen loaders.
For `traces` or `allen_nwb`, set `INCLUDE_MANUAL_WITH_FILE_SOURCE = True` if
the generated template should also contain a blank manual block. File-backed
data remains authoritative while that mode is selected; switch the mode to
`manual` to use manually entered values instead.

New files and `CONFIG_MODE = "overwrite"` contain only the blocks relevant to
the selected mode. The normal `fill` mode preserves all existing blocks and
values, so rerunning Step 1 does not erase target work.

See `../reference/target_trace_formats.md` for passive trace, active trace, and
FI CSV target data requirements.

For Allen/ADB electrophysiology target data:

1. Open the Allen electrophysiology page for the specimen:

```text
https://celltypes.brain-map.org/experiment/electrophysiology/<specimen_id>
```

2. Click **Download Data**.
3. Copy the downloaded `*_ephys.nwb` file into the tune you will actually tune,
   usually `cells/<CELL>/tunes/tuned/`.
4. Set `target_source.mode` to `"allen_nwb"` and `allen_nwb.file` to the
   downloaded filename or path.

If you create a `tuned` working copy after setting up `target_config.json`, the
config is copied with the tune. If the working copy already exists, rerun the
notebook target-config section with `TARGET_CONFIG_TUNE_DIR_OVERRIDE` pointing
to the working tune.

CLI equivalent:

- `--no-target-config`
- `--target-source-mode none|manual|traces|allen_nwb`
- `--include-manual-with-file-source`

## Synapse Configs

Optional synapse scaffolding creates:

- `syn_config.json`
- `syn_groups/input_blocks_template.json`

The generated template is disabled by default:

```json
"state": false
```

Enable and edit it before using it in Step 5.

The public synapse input schema uses explicit `input_blocks`:

- homogeneous baseline/background blocks use `rate_hz`,
- inhomogeneous/precomputed blocks use `source`,
- gaps are quiescent,
- overlaps are invalid.

Template weight fields are controlled by `synapse_weight_style`:

- `distributed`: generates `wt_mean` and `wt_std`.
- `fixed`: generates `initW`.

CLI equivalents:

- `--synapse-templates input_blocks`
- `--synapse-templates none`
- `--no-synapse-configs`
- `--synapse-weight-style distributed|fixed`

## Validation

Validation can check:

- required files,
- compiled mechanisms when `.mod` sources exist,
- cell loading,
- synapse config structure,
- input config normalization.

For `source_type="existing"`, validation reports loader-owned missing paths.
New HOC tunes also require finite `conditions.v_init_mV` and
`conditions.celsius_C`. Targets and synapse configs are not required for
cell-only/IClamp work.

Validation can be skipped with:

```bash
--no-validate
```

Input-specific validation can be skipped with:

```bash
--no-validate-inputs
```

## CLI Usage

### List ADB Models

```bash
python scripts/step1_prepare.py \
  --cell PV \
  --source-type adb \
  --specimen-id 484635029 \
  --list-models-only
```

### Prepare PV Perisomatic

```bash
python scripts/step1_prepare.py \
  --cell PV \
  --tune orig \
  --specimen-id 484635029 \
  --model-type perisomatic
```

### Prepare SST All-Active

```bash
python scripts/step1_prepare.py \
  --cell SST \
  --tune orig \
  --specimen-id 485466109 \
  --model-type "all active"
```

### Prepare PN Perisomatic

```bash
python scripts/step1_prepare.py \
  --cell PN \
  --tune orig \
  --specimen-id 382982932 \
  --model-type perisomatic
```

### Prepare Raw Tune and Create `tuned` Copy

```bash
python scripts/step1_prepare.py \
  --cell PV \
  --tune orig \
  --specimen-id 484635029 \
  --model-type perisomatic \
  --create-tuned-copy
```

Use `--overwrite-tuned-copy` only when intentionally replacing an existing
working tune.

### Refresh Configs Only

```bash
python scripts/step1_prepare.py \
  --tune-dir cells/PV/tunes/orig \
  --source-type existing \
  --no-download \
  --no-compile \
  --config-mode fill
```

This skips `nrnivmodl` while loading the tune's existing compiled mechanism
library for validation. If the library has not been built yet, omit
`--no-compile`. To edit configs without constructing the cell, combine
`--no-compile --no-load-dll --no-validate`.

### Prepare an Existing HOC Template

```bash
python scripts/step1_prepare.py \
  --tune-dir cells/example_cell/tunes/orig \
  --cell example_cell \
  --source-type existing \
  --cell-loader hoc_template \
  --hoc-template-file model/CellTemplate.hoc \
  --hoc-template-name CellTemplate \
  --hoc-constructor-args '[]' \
  --hoc-section-map '{}' \
  --v-init-mv -70 \
  --celsius-c 34 \
  --target-source-mode none \
  --no-synapse-configs
```

Use `--loader-paths-json '{"modfiles": null}'` when the template uses built-in
mechanisms only. Add section-map entries only when common owner aliases do not
match the template.

### Create Empty Synapse Manifest

```bash
python scripts/step1_prepare.py \
  --tune-dir cells/PV/tunes/orig \
  --source-type existing \
  --no-download \
  --no-compile \
  --synapse-templates none
```

### Skip All Config Scaffolding

```bash
python scripts/step1_prepare.py \
  --tune-dir cells/PV/tunes/orig \
  --source-type existing \
  --no-download \
  --no-scaffold
```

## Step 1 vs Later Steps

Step 1 is responsible for setup and validation.

- Steps 2-3 tune passive/active properties.
- Step 4 tunes synapse settings.
- Step 5 runs simulations from the prepared tune.
- Step 6 analyzes saved outputs.

## After Step 1

Typical next actions:

- use the optional working-copy cell to copy the validated `orig` tune to
  `tuned` before editing/tuning if you want to preserve the raw setup reference,
- optionally download Allen/ADB ephys target data, place it in the working
  `tuned` tune, and point `target_config.json` to it,
- optionally run `extra_notebooks/act_segmentation.ipynb` on the copied `tuned`
  tune if your workflow needs ACT-style channel/mechanism segmentation before
  tuning,
- open `2_passive.ipynb` for passive traces/diagnostics and optional ACT-based proposals,
- open `3_active.ipynb` for manual active checks and optional ACT active tuning,
- open `4_synapses.ipynb` for synapse tuning,
- open `5_simulate.ipynb` for IClamp checks or full simulations,
- inspect generated configs under `cell_configs/`.

The notebook's optional working-copy cell copies `orig` to `tuned` and updates
the copied `cell_configs/cell_config.json` automatically. If you do the same
operation manually, copy the tune folder:

```bash
cp -a cells/PV/tunes/orig cells/PV/tunes/tuned
```

Then update the copied `cell_configs/cell_config.json` so its `tune` field is
`"tuned"`. Use the matching cell folder for `SST`, `PN`, or your own cell label.

## Optional Segmentation / Segregation

Step 1 validates the raw downloaded or staged tune. Some ACT-style workflows also
use a segmentation/segregation step before passive and active tuning.

Segmentation is model-specific: it separates or modifies channel/mechanism
activation definitions before tuning. It is not required for standard Step 1-7
usage.

If needed, run `extra_notebooks/act_segmentation.ipynb` after the raw Step 1 tune
validates:

1. Keep the raw ADB/setup tune as a reference, usually named `orig`.
2. Copy it to the tune you plan to modify/tune, usually named `tuned`.
3. Run the segmentation notebook on the copied `tuned` tune.
4. Recompile and validate the copied tune in the segmentation notebook.
5. Continue to Step 2 using `tuned`.

Validating the raw tune first keeps download/setup problems separate from
segmentation edits.
