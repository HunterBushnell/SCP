# Step 4: Synapse Tuning

Step 4 tunes chemical synapse mechanism parameters for a prepared SCP tune using
BMTool's synaptic tuner.

Notebook: `../../4_synapses.ipynb`

## Scope

The root `4_synapses.ipynb` is the primary Step 4 entry point for both local and
Colab use. The notebook intentionally stays close to BMTool's original chemical
synapse tuner notebook, with a small SCP adapter layer for loading prepared tune
directories.

Step 4 is manual-first for config edits:

- select a prepared post-synaptic cell/tune,
- compile/load the tune's mechanisms,
- build an SCP NEURON cell,
- create or load `cell_configs/synapse_tuning_config.json`,
- read BMTool `general_settings` and connection settings from that config,
- validate the configured point process and canonical section index before
  resolving optional BMTool,
- initialize BMTool's `SynapseTuner` with SCP's cell-scoped BMTool facade,
- run BMTool single-event, interactive, frequency-response, and optional
  optimizer workflows,
- print a copyable `syns.params` block,
- manually paste tuned parameters into the relevant
  `cell_configs/syn_groups/*.json` file.

Step 4 does not automatically overwrite synapse group configs.

The compact `0_pipeline.ipynb` counterpart presents this stage as three
independent bordered cards: **Initialize BMTool**, **Single Event**, and
**Interactive Tuner**. Each action has its own output area. Enablement and
connection selection live with initialization, leaving that card available for
future compact overrides while the detailed notebook remains the full
configuration-oriented workflow.

## Expected Inputs

- a tune directory prepared by Step 1,
- passive/active parameters reviewed in Steps 2-3 when applicable,
- custom compiled/compilable `.mod` sources when the selected point process is not built in,
- `cell_configs/cell_config.json`,
- optional `cell_configs/synapse_tuning_config.json`; Step 4 creates a neutral
  four-entry starting catalog when it is missing,
- BMTool available at `../mods/bmtool` or through `SCP_BMTOOL_PATH`,
- a NEURON synapse point process supplied either by built-in mechanisms or the tune's compiled sources,
- optional existing `cell_configs/syn_groups/*.json` files to receive tuned
  parameters.

## Synapse Mechanism Files

Step 4 does not automatically choose or install custom synapse mechanisms.
Built-in point processes need no MOD directory. For custom mechanisms, make
sure the `.mod` file exists in the directory configured by `paths.modfiles`,
normally:

```text
cells/<CELL>/tunes/<TUNE>/modfiles/
```

Use one of these paths:

1. If you already have a synapse model, place that `.mod` file in the tune's
   configured `paths.modfiles` directory.
2. If you need a starting point, use the cyneuro neuron mechanisms library:
   <https://github.com/cyneuro/neuron-mechanisms-library/tree/main/synaptic-mechanisms>.
3. For SCP-style examples, the recommended starting folder is:
   <https://github.com/cyneuro/neuron-mechanisms-library/tree/main/synaptic-mechanisms/blue-brain>.

Typical starting mechanisms:

- Excitatory: `AMPA_NMDA.mod` or `AMPA_NMDA_STP.mod`.
- Inhibitory: `GABA_A.mod` or `GABA_A_STP.mod`.

After adding or changing `.mod` files:

1. Restart the kernel if mechanisms were already loaded.
2. Set `RECOMPILE_MODFILES = True` in section 4.1.
3. Confirm `connections[*]["spec_settings"]["level_of_detail"]` in
   `synapse_tuning_config.json` matches the NEURON point-process name declared
   inside the `.mod` file, usually the filename without `.mod`.

## Outputs

Step 4 does not create Step 5 simulation runs. Its main output is reviewed
synapse parameter values.

The export cell prints a JSON block shaped for a synapse-group config:

```json
"params": {
  "Dep": 0.0,
  "Fac": 200.0,
  "Use": 0.75,
  "initW": 0.25,
  "tau_d_AMPA": 1.7,
  "tau_r_AMPA": 0.2
}
```

Paste the printed values into:

```text
cells/<CELL>/tunes/<TUNE>/cell_configs/syn_groups/<group>.json
```

under:

```json
{
  "<group_name>": {
    "syns": {
      "params": {}
    }
  }
}
```

## BMTool Relationship

BMTool is the core tuning engine for Step 4. SCP only adapts tune-directory
models to BMTool's expected API.

The adapter lives at:

```text
modules/tuning/bmtool_synapse_adapter.py
```

It handles:

- locating the SCP repo,
- locating or cloning BMTool,
- compiling/loading tune mechanisms,
- building the SCP cell,
- rejecting unavailable point-process names and invalid `sec_id` values before
  importing or cloning BMTool,
- creating BMTool's `SynapseTuner` with a pre-built `hoc_cell`,
- choosing conservative default slider/record variables for common mechanisms,
- printing copyable SCP config parameter blocks.

The BMTool tuning methods remain BMTool methods:

- `tuner.SingleEvent()`
- `tuner.InteractiveTuner()`
- `tuner.stp_frequency_response(...)`
- `SynapseOptimizer(tuner)`

## Notebook Workflow

### 4.1 Prepare SCP Cell

Choose the post-synaptic cell/tune used for tuning. Selection is explicit; the
notebook does not choose a production cell by default:

```python
cell_name = None  # set this, or set tune_dir_override
tune_name = "tuned"
```

Important controls:

- `tunes_parent`: normally `tunes`.
- `tune_dir_override`: optional direct path outside the standard repo layout.
- `RECOMPILE_MODFILES`: force `nrnivmodl` even if compiled mechanisms exist.
- `LOAD_COMPILED_DLL`: load the compiled NEURON mechanism library into the
  current kernel when custom `.mod` sources are present.

The notebook calls:

```python
session = prepare_scp_synapse_tuning(...)
cell = session.cell
tune_dir = session.tune_dir
```

Restart the kernel if NEURON reports conflicts from a previously loaded
mechanism library.

### 4.2 Load Synapse Tuning Config

Create or load:

```text
cells/<CELL>/tunes/<TUNE>/cell_configs/synapse_tuning_config.json
```

The file stores BMTool-style settings for the selected tune. If it is missing,
Step 4 creates a loader-neutral template. Generated `vclamp_amp` values come
from `sim_config.conditions.v_init_mV`, and generated `general_settings.celsius`
comes from `sim_config.conditions.celsius_C`. An existing file is read without
being rewritten unless overwrite is explicitly enabled.

Important controls:

- `OVERWRITE_SYNAPSE_TUNING_CONFIG`: recreate the file from the default
  template. Leave `False` after the first run so manual edits are preserved.
- `CONNECTION_OVERRIDE`: temporarily choose a connection key without editing
  `default_connection` in the config.
- `default_connection`: connection key to use by default.
- `current_name`: synaptic current variable; usually `i`.
- `slider_vars`: optional explicit BMTool slider variables; `null` uses SCP
  defaults for the selected mechanism.
- `other_vars_to_record`: optional mechanism variables to record; `null` uses
  SCP defaults.
- `general_settings`: BMTool protocol settings.
- `connections`: per-connection mechanism/location/parameter settings.

`general_settings` controls BMTool's small tuning protocols, not the full Step 5
simulation. Common fields:

- `vclamp`: start BMTool checks in voltage clamp mode.
- `rise_interval`: fractional interval for rise-time calculation.
- `tstart`: single-event start time in ms.
- `tdur`: post-event simulation duration in ms.
- `threshold`: NetCon threshold in mV.
- `delay`: NetCon delay in ms.
- `weight`: NetCon weight.
- `dt`: BMTool protocol time step.
- `celsius`: BMTool protocol temperature.

Each `connections` entry has:

- `description`: human-readable note for the connection.
- `spec_settings.post_cell`: retained for BMTool compatibility; ignored when
  SCP supplies `hoc_cell`.
- `spec_settings.vclamp_amp`: voltage clamp holding value.
- `spec_settings.sec_id`: section index from `list(cell.all)`.
- `spec_settings.sec_x`: normalized section location.
- `spec_settings.level_of_detail`: NEURON point-process mechanism name.
- `spec_syn_param`: initial mechanism parameters.

The generated starting catalog contains exactly four neutral entries:

- `excitatory_facilitating`
- `excitatory_depressing`
- `inhibitory_static`
- `inhibitory_stp`

These are editable mechanism/parameter starting points, not cell-type
selections. SCP never substitutes `ExpSyn` or another point process when a
configured mechanism is unavailable.

For custom mechanisms, keep the JSON shape and edit the mechanism name,
location, and parameter names to match the `.mod` file. The notebook converts
`connections` into BMTool's `conn_type_settings` structure before initializing
the tuner.

### 4.3 Initialize BMTool SynapseTuner

The notebook creates the BMTool tuner through:

```python
tuner = create_scp_synapse_tuner(
    session,
    conn_type_settings=conn_type_settings,
    connection=connection,
    general_settings=general_settings,
    current_name=current_name,
    other_vars_to_record=resolved_record_vars,
    slider_vars=resolved_slider_vars,
)
```

Important controls:

- `slider_vars`: mechanism parameters exposed as interactive sliders, usually
  configured in `synapse_tuning_config.json`.
- `other_vars_to_record`: optional mechanism variables to plot/record, usually
  configured in `synapse_tuning_config.json`.
- `current_name`: synaptic current variable, usually configured in
  `synapse_tuning_config.json`.

Leave `slider_vars` as `null` to use SCP defaults for common mechanisms. Set it
explicitly in the config for custom mechanisms.

### 4.4 Single Event

Run:

```python
tuner.SingleEvent()
```

This runs a single synaptic event and reports BMTool response metrics such as
amplitude, latency, rise time, decay time, half width, and baseline.

Use this before opening the interactive tuner to confirm the synapse mechanism,
section location, clamp settings, and current recording are valid.

### 4.5 Interactive Tuner

Run:

```python
tuner.InteractiveTuner()
```

This opens BMTool's widget interface for changing selected synapse parameters
and inspecting response changes.

Use this as the primary manual tuning interface. If widgets do not render,
confirm `ipywidgets` is installed and enabled in the active notebook runtime.

### 4.6 Frequency Response

Run:

```python
results = tuner.stp_frequency_response(log_plot=False)
```

This evaluates short-term plasticity behavior across stimulation frequencies.
BMTool returns:

- `frequencies`,
- `ppr`,
- `induction`,
- `recovery`.

These values are useful for checking whether a tuned synapse behaves as
facilitating, depressing, or neutral across the frequency range of interest.

### 4.7 Optional SynapseOptimizer

The optional optimizer uses BMTool's `SynapseOptimizer` to search parameter
bounds against target metrics.

Optimizer defaults live under `optimizer` in `synapse_tuning_config.json`.

Important controls:

- `optimizer.enabled`: run the optimizer when `true`.
- `optimizer.param_bounds`: parameter search ranges.
- `optimizer.target_metrics`: desired BMTool response metrics.
- `optimizer.cost_weights`: relative weights for metric errors.
- `optimizer.run_single_event`: include single-event metrics in optimization.
- `optimizer.run_train_input`: include train-response metrics in optimization.
- `optimizer.train_frequency` and `optimizer.train_delay`: STP train protocol
  controls.

Treat this as optional. Manual tuning is often easier to interpret, and the
optimizer depends strongly on parameter bounds and cost-function choices.

### 4.8 Export Tuned Parameters

After manual tuning or optimization, run the export cell:

```python
tuned_params = get_tuned_synapse_params(tuner)
print_syn_group_param_block(tuned_params)
```

Then manually paste the printed parameter block into the appropriate synapse
group config.

If using distributed weights in Step 5, decide whether the tuned amplitude
belongs in `initW` or should be translated into `wt_mean` / `wt_std`. Step 4
tunes mechanism parameters for a single test synapse; Step 5 applies group-level
placement and weight rules.

## Mapping Step 4 to Step 5 Configs

Step 4 tunes mechanism parameters. Step 5 consumes synapse group configs.

Relevant Step 5 config location:

```text
cells/<CELL>/tunes/<TUNE>/cell_configs/syn_groups/*.json
```

Relevant config section:

```json
{
  "group_name": {
    "syns": {
      "type": "AMPA_NMDA_STP",
      "N_syn": 100,
      "segs": "all",
      "dist_func": {"kind": "uniform", "params": {"c": 1.0}},
      "params": {
        "initW": 0.25,
        "tau_r_AMPA": 0.2,
        "tau_d_AMPA": 1.7,
        "Use": 0.75,
        "Dep": 0.0,
        "Fac": 200.0
      }
    },
    "input_blocks": []
  }
}
```

Field mapping:

- BMTool `spec_settings.level_of_detail` maps to Step 5 `syns.type`.
- BMTool `spec_syn_param` values map to Step 5 `syns.params`.
- BMTool `sec_id` / `sec_x` are test-synapse location controls only; Step 5
  uses `syns.segs` and `syns.dist_func` for group placement.
- BMTool `general_settings` are tuning-protocol controls only; Step 5 uses
  `sim_config.json` and `input_blocks` for full simulations.

See `../reference/configs_reference.md` for the full synapse group schema.

## Local vs Colab

The Step 4 notebook uses the same root-notebook bootstrap pattern as the other
public notebooks.

Local use:

- install SCP's environment,
- clone BMTool to `../mods/bmtool` or set `SCP_BMTOOL_PATH`,
- open `4_synapses.ipynb`.

Colab use:

- open the root `4_synapses.ipynb`,
- run the environment setup cell,
- allow the notebook to clone SCP and BMTool when needed,
- make sure any required tune/model files are available in the runtime.

BMTool resolution order:

1. `SCP_BMTOOL_PATH`, `BMTOOL_PATH`, or `BMTOOL_ROOT`,
2. `../mods/bmtool` relative to SCP,
3. common local `mods/bmtool` locations,
4. automatic clone in Colab when enabled.

## Troubleshooting

### BMTool Not Found

Clone BMTool next to SCP:

```bash
mkdir -p ../mods
git clone https://github.com/cyneuro/bmtool.git ../mods/bmtool
```

Or set:

```bash
export SCP_BMTOOL_PATH=/path/to/bmtool
```

### Mechanism Already Loaded

NEURON cannot always unload/reload mechanism libraries cleanly inside a live
kernel. Restart the kernel, then rerun Step 4 from the top.

### Synapse Mechanism Missing

If `level_of_detail` fails:

- confirm the mechanism name matches the `.mod` point-process name,
- confirm the directory selected by `cell_config.paths.modfiles` contains that
  `.mod` file,
- recompile mechanisms with `RECOMPILE_MODFILES = True`,
- restart the kernel if another mechanism library was loaded first.

### Slider Variable Missing

If BMTool skips a slider:

- confirm the variable exists as a range/global parameter in the `.mod` file,
- add it to `spec_syn_param`,
- or set `slider_vars` to only the variables the mechanism exposes.

### Current Recorder Missing

If BMTool warns that `_ref_i` is missing, set:

```python
current_name = "<mechanism_current_variable>"
```

to the current variable exposed by the synapse mechanism.

### Section Location Is Wrong

`sec_id` indexes `list(cell.all)`, which is useful for a single test synapse but
not necessarily intuitive. Start with soma/proximal test locations, then use
Step 5 placement preview for full group-level placement.
