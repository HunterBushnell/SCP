# Custom Cells and External Models

SCP supports ordinary compartmental NEURON cells without an Allen manifest,
canonical native section names, or model files inside SCP. PV remains the
recommended first example.

## Choose a workflow

| Workflow | Use it for | Another process can rebuild it? |
| --- | --- | --- |
| Live notebook cell | Exploration and in-memory manual edits | No; recreate it or save a builder |
| Saved Python builder | Python-owned cells/custom construction | Yes, from the configured callable |
| HOC template | Native object-owned HOC cells | Yes, from the configured template |
| External copy | Independent tuning without editing production | Yes, using the local snapshot |
| External reference | Inspecting original production sources | Yes, while those inputs remain available |

Run the [synthetic walkthrough](../../extra_notebooks/custom_cells.ipynb).
Its numeric settings are demonstration choices, not biological targets or
scientific defaults.

To keep same-named cells from different projects separate, select a workspace
under `projects/<project>/cells/<cell>/tunes/<tune>/`. All workflows below also
work in that layout. A project namespace does not alter the external source
root or make reference-mode sources independent. See
[Project workspaces](project_workspaces.md); bundled PV defaults stay unchanged.

## The cell contract

`wrap_cell` supplies NEURON's actual `h`, retains the owner, and normalizes
`soma`, `dend`, `apic`, `axon`, and `all`. A nonempty soma is required.
Optional groups may be empty. SCP derives `all` from mapped groups if needed,
never from unrelated global sections.

```python
from neuron import h
from modules.model import wrap_cell, validate_cell

soma = h.Section(name="body")
soma.L = soma.diam = 20.0
soma.insert("pas")
cell = wrap_cell(soma, cell_name="MyCell")
validate_cell(cell)
assert cell.model is soma
assert cell.all == (soma,)
```

For an owner, use `section_map={"soma": "body", "all": "whole_cell"}`, or supply
actual sections with `sections={"soma": soma, "dend": dendrites}`. Do not provide
both. An existing `LoadedCell` can be adopted directly. Wrapping does not change
geometry, segmentation, mechanisms, or parameters. Compatibility is not
biological validation; sectionless artificial cells cannot use these same
somatic passive/FI protocols.

Keep native point processes, vectors, random generators, and POINTER
dependencies alive on the owner. SCP cannot infer missing native setup.
Requested recording references must be valid. Existing current discovery
includes exposed i-prefixed variables; inspect labels because mechanisms can
also expose state variables with such names.

## Create a cell at the top of a notebook

After environment setup and before model preparation, use the optional hook:

```python
from neuron import h
from modules.model import wrap_cell

soma = h.Section(name="body")
soma.insert("pas")
CELL_OVERRIDE = wrap_cell(soma, cell_name="MyCell")
RUNTIME_SIM_CONFIG = {
    "conditions": {"v_init_mV": -70.0, "celsius_C": 34.0}
}
```

Select **your own** cell/workspace in the next settings cell, e.g. `MyCell/tuned`,
or use `tune_dir_override`. Do not accidentally leave PV selected. The example
conditions are synthetic choices; supply your experimental values. Custom
mechanisms must be loaded before creating the cell. Compile/recompile first,
then restart/recreate the object.

No Step 1 run is required. SCP creates only missing generic bookkeeping configs,
with explicit runtime conditions, geometry grouping, and `target_source.mode:
"none"`. It does not download sources, invent targets, or create synaptic inputs.
Existing configs/recipes remain on disk; `RUNTIME_SIM_CONFIG` supplies explicit
session overrides. Leave both hooks `None` for normal configured models.

Steps 2–3 use the adopted cell. Step 4 can hand canonical sections to optional
BMTool after mechanism/index preflight. Choose Current clamp explicitly in
Step 5 unless synaptic inputs are configured. Step 6 can inspect the current
live override; saved-result-only analysis does not require a cell. Step 7
cannot reconstruct an arbitrary notebook object.
Adoption does not extend the synapse mechanism API: SCP-generated synapses
still use its existing `initW`/VecStim contract. A cell can also retain native
inputs (for example, a NetStim/NetCon) on its owner. Built-in ExpSyn/Exp2Syn use
NetCon weights and are not interchangeable with an `initW` mechanism in the
Step 5 input generator.

The compact notebook also supports these hooks: select the workspace in
`pipeline_settings` before preparing. Live previews/runs execute in this kernel;
configured models retain fresh-process execution. The live object is never
serialized into JSON or subprocess arguments. In-memory edits remain active in
later live steps. Do not run concurrent NEURON simulations in one kernel.
NEURON still initializes/runs all objects in its process, even though SCP
records this cell; avoid leaving a whole network instantiated beside it.
Recreate the cell in every separate notebook kernel.

### Code-first simulation

```python
from modules.simulation import SimulationOptions, SimulationSession

session = SimulationSession.from_cell(
    cell, tune_dir="cells/MyCell/tunes/tuned",
    sim_config={"conditions": {"v_init_mV": -70.0, "celsius_C": 34.0}},
    options=SimulationOptions(iclamp=True),
)
results = session.run()
```

Configure timing/amplitude in `sim_config.json` or session overrides. Saving is
opt-in through the normal options/config. No native properties become targets
automatically.

## Save a Python construction recipe

Move construction into a trusted file, e.g. `model/cell.py`:

```python
from neuron import h

def build_cell(size=20.0):
    soma = h.Section(name="body")
    soma.L = soma.diam = size
    soma.insert("pas")
    return soma
```

Configure `cell_configs/cell_config.json`:

```json
{
  "cell_name": "MyCell",
  "tune": "orig",
  "cell_loader": "python_factory",
  "paths": {"python_factory": "model/cell.py", "modfiles": null},
  "python_factory": {
    "callable": "build_cell",
    "args": [],
    "kwargs": {"size": 20.0}
  }
}
```

The callable can be a function or class. Arguments must be finite JSON-compatible
values. Its return follows the wrapping contract; `python_factory.section_map`
handles noncanonical owner names. Python files are not executed during discovery.

Use Step 1 to fill bookkeeping configs and set explicit conditions, or:

```bash
python scripts/step1_prepare.py --cell MyCell --tune orig \
  --cell-loader python_factory --source-type existing \
  --python-factory-file model/cell.py --python-factory-callable build_cell \
  --v-init-mv -70 --celsius-c 34 --target-source-mode none --no-synapse-configs
```

Use `--loader-config-json` for args/kwargs/mappings. Package-contained builders
support relative imports. A declared source root is also the Python import root.
Conflicting imports or changed loaded sources require a fresh process; do not
depend on autoreload to replace NEURON models.

## Register external sources

Select a bounded common root and the required native folders, not a whole network
repository accidentally. Include morphology/data assets and MOD include files;
preserve source headers, native paths, and HOC template names.
Keep selected native inputs disjoint from the workspace and output destinations;
the common root may be the project itself. New CLI registrations require explicit
`--model-source-folders` or `--model-source-files`; SCP does not select a whole repository
when you omit them.

In Step 1.2:

```python
MODEL_SOURCE_OVERRIDES = {
    "cell_loader": "hoc_template",
    "paths": {"hoc_template": "templates/Cells.hoc", "modfiles": "mechanisms"},
    "hoc_template": {"template_name": "ExampleCell", "constructor_args": []},
    "model_source": {
        "mode": "copy",
        "source_root": "/path/to/external/model_components",
        "folders": ["templates", "mechanisms", "morphologies"],
        "working_directory": "."
    }
}
```

The same `model_source` block works with Python builders. With it, loader/MOD
paths are **source-root-relative**. Without it, historical absolute/tune-relative
paths still work, but do not register a complete dependency bundle or reference
snapshot.

Setup previews paths/count/bytes before staging. For a separate read-only review,
use `preview_model_sources(tune_dir, recipe)` or CLI `--preview-model-sources`.
Register with `register_model_sources(tune_dir, recipe, dry_run=False)`.

Every registration keeps a byte-preserving snapshot in `model_sources/` plus
hashes/mappings in `MODEL_SOURCE_PROVENANCE.json`, even in reference mode.
Default copy mode loads that snapshot; reference mode loads originals. The
snapshot is a registration baseline, not a silently refreshed upstream mirror.

```python
from modules.setup import set_model_source_mode
set_model_source_mode(tune_dir, "reference")  # or "copy"; then restart the kernel
```

Normal orig-to-tuned creation remains separate. Copy-mode tunes are independent;
reference-mode tunes intentionally share originals. Tune the copy. Snapshot
refresh requires explicit `refresh=True`, retains backups, and may replace
local edits; review carefully. Normal reruns do not refresh it.

External MOD compilation uses the tune's ignored `.scp_build/` mirror,
including declared dependencies. Its key includes source hashes and toolchain
identity. SCP never compiles/deletes upstream `x86_64`; tune-local legacy
libraries remain compatible. External Python imports do not create upstream
bytecode files.

Execution defaults to the entry-file directory. Set a root-relative
`working_directory` when native code expects a project-level cwd. SCP restores
cwd/import settings afterward. Argument strings are not rewritten: use
bundle-relative paths instead of machine-specific filenames. Dependencies are
explicit, not inferred by parsing arbitrary executable code. Traversal/escaping
symlinks are rejected; declare real folders instead of directory symlinks.

Run only trusted code. Source protection applies to **SCP-managed operations**;
user HOC/Python code is not sandboxed and can perform its own writes.

## Extract one cell from a network model

Write a small builder that imports the cell class or loads its HOC template,
creates one cell, initializes native dependencies, and returns the owner.
Do not import a driver that launches simulations at import time or construct
the entire network before returning one member. Keep network/noise hooks
explicit; SCP does not enable/disable them or choose scientific conditions.
ACT and BMTool remain optional; non-Allen ACT optimization is experimental.

## Saving and restoration

Live saves have `model_provenance.session_only: true` and
`construction_recipe_available: false`. `model_state.json` describes pre-run
geometry/mechanism values. It is not a full checkpoint of native POINTERs,
GLOBAL parameters, random generators, callbacks, or object state. An unused
workspace recipe is not attributed to an overridden cell.

Configured builders/templates archive declared native inputs and hashes in
`model_artifacts/`. A construction recipe does not make unsaved in-memory edits
replayable. Persist edits in copied sources or builder arguments, restart, and
validate before CLI/SLURM. This update does not add parameter overlays or
automatic notebook-code capture.
Saved provenance includes preparation-time input hashes. If a file changes
between construction and saving, the archive labels the mismatch rather than
claiming the new bytes constructed the already-loaded cell.

CLI/SLURM rebuild configured cells but reject runtime-only workspaces with
instructions to register a builder. Compact configured Step 5 runs do not
inherit unsaved notebook edits.

To materialize a declared external archive locally:

```bash
python scripts/restore_run_state.py --from-run /path/to/saved/run \
  --to-tune /path/to/local/tune --apply model_artifacts --materialize-external
```

Create the destination directory first. Review the dry-run before adding
`--write`. Materialization preserves layout, verifies hashes, backs up replaced
files, and sets copy mode; it never writes into upstream sources. Use an empty
destination or one with the same loader. Restart/recompile as reported.
Older archives remain supported; legacy undeclared external artifacts lack
the layout information required for this materialization operation.

Saved-result-only analysis works without original sources. Current-cell
inspection uses the current recipe/live override and may differ from a saved
run. Check saved provenance before interpreting comparisons.
