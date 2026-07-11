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
- scaffold optional synapse config templates,
- validate that a tune is ready for later pipeline steps.

Step 1 does not tune model parameters and does not run final simulations.

## What Step 1 Produces

For a tune directory such as `cells/PV/tunes/adb_peri`, Step 1 prepares:

```text
manifest.json
modfiles/
cell_configs/
  cell_config.json
  sim_config.json
  geometry.json
  syn_config.json
  syn_groups/
    input_blocks_template.json
```

Synapse configs are optional. They can be skipped for cell-only or IClamp work.

## Main Decisions

Before running Step 1, decide:

- `cell_name`: display/model label, such as `PV` or `SST`.
- `tune_name`: tune folder name, such as `adb_peri`, `adb_all`, or a project-specific tune.
- `source_type`: `adb` to download Allen Database files, or `existing` for staged local files.
- `CONFIG_MODE`: `fill`, `overwrite`, or `skip` for generated configs.
- Synapse scaffolding: create `input_blocks_template.json`, create an empty `syn_config.json`, or skip synapse configs.

Recommended default for normal work:

```python
CONFIG_MODE = "fill"
```

Use `overwrite` only when intentionally resetting generated configs.

## Notebook Workflow

Open `1_setup.ipynb` and run the phases in order:

1. Select tune directory.
2. Set up model source files.
3. Compile/load mechanisms.
4. Scaffold base configs.
5. Optionally scaffold synapse configs.
6. Validate setup.
7. Review expected paths.

The notebook runs locally or in Colab. In a fresh Colab session it can clone the
repo and install dependencies using the same environment variables as Step 5.

## Tune Selection

Common bundled setup targets:

- PV perisomatic raw setup: `cells/PV/tunes/adb_peri`
- SST all-active raw setup: `cells/SST/tunes/adb_all`

Common tuned simulation examples:

- `cells/PV/tunes/seg_tuned`
- `cells/SST/tunes/seg_tuned`

The raw ADB tunes are useful as starting points. The tuned examples are the
current runnable Step 5 examples.

## Cell Source Types

### `source_type = "adb"`

Downloads or refreshes an Allen Database model bundle.

Required inputs:

- `specimen_id`
- `model_type`: usually `perisomatic` or `all active`

Examples:

```python
source_type = "adb"
specimen_id = 484635029
model_type = "perisomatic"
```

```python
source_type = "adb"
specimen_id = 485466109
model_type = "all active"
```

ADB model types can differ structurally. Perisomatic and all-active bundles may
have different manifest/model/fit-file layouts, so keep `model_type` explicit
when preparing ADB tunes.

### `source_type = "existing"`

Uses files already staged in the tune directory. Currently this means an
Allen-compatible tune layout with a valid `manifest.json`, model files, and
`modfiles/`.

Use this when files already exist locally and you only want to compile,
scaffold configs, or validate.

Generic/non-ADB model support is a planned extension. The public setup contract
for now is an Allen-compatible model bundle or an already working tune directory.

## Mechanisms

Step 1 can compile NEURON mechanisms from:

```text
<tune_dir>/modfiles/
```

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

- `cell_config.json`: cell label, tune name, color, loader, manifest path, and soma diameter multiplier.
- `sim_config.json`: simulation timing, trial/save settings, plotting, IClamp, recording, randomness, and snapshot defaults.
- `geometry.json`: soma-origin distance reference and proximal/distal thresholds.

See `../reference/configs_reference.md` for every generated field.

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
- compiled mechanisms,
- cell loading,
- synapse config structure,
- input config normalization.

For a newly staged `source_type="existing"` tune without a complete
Allen-compatible bundle, validation will fail until `manifest.json` and model
files are present.

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
  --tune adb_peri \
  --specimen-id 484635029 \
  --model-type perisomatic
```

### Prepare SST All-Active

```bash
python scripts/step1_prepare.py \
  --cell SST \
  --tune adb_all \
  --specimen-id 485466109 \
  --model-type "all active"
```

### Refresh Configs Only

```bash
python scripts/step1_prepare.py \
  --tune-dir cells/PV/tunes/adb_peri \
  --source-type existing \
  --no-download \
  --no-compile \
  --config-mode fill
```

### Create Empty Synapse Manifest

```bash
python scripts/step1_prepare.py \
  --tune-dir cells/PV/tunes/adb_peri \
  --source-type existing \
  --no-download \
  --no-compile \
  --synapse-templates none
```

### Skip All Config Scaffolding

```bash
python scripts/step1_prepare.py \
  --tune-dir cells/PV/tunes/adb_peri \
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

- open `2_passive.ipynb` for ACT passive tuning,
- open `3_active.ipynb` for manual active checks and optional ACT active tuning,
- open `4_synapses.ipynb` for synapse tuning,
- open `5_simulate.ipynb` for IClamp checks or full simulations,
- inspect generated configs under `cell_configs/`.
