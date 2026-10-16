# Model Loaders and the Canonical Cell Interface

For project-local notebooks use the installed `scp_pipeline.model` interface;
checkout-local `modules` imports remain compatible. External source declarations
accept individual root-relative `files` alongside `folders`. See the
[dependency workflow](../guides/dependency_workflow.md) for editable installation,
interactive setup, and flat native-source layouts.

SCP constructs configured models through registered loaders or adopts an
existing NEURON cell, then exposes the same cell-scoped interface to Steps 1–7.
`allen_manifest` remains the default and
retains the existing Allen aliases. `hoc_template` is the first object-owned
adapter.

The public entry point is:

```python
from scp_pipeline.model import load_cell

cell = load_cell(cell_config, base_dir=tune_dir)
```

Pass the tune directory explicitly. Current-directory path resolution exists
only for older callers.

The tune may use `cells/<cell>/tunes/<tune>/` or
`projects/<project>/cells/<cell>/tunes/<tune>/`; the loader interface and native
source paths are the same. A project is a workspace namespace, not a new loader
or an isolated NEURON runtime. See
[Project workspaces](../guides/project_workspaces.md).

## Step 1 Source Discovery

The Step 1 notebook keeps common source selection out of per-cell notebook
state. It first reuses an existing `cell_configs/cell_config.json`. For a fresh
staged tune, it then accepts exactly one `manifest.json` or exactly one HOC
`begintemplate` declaration and records the discovered relative path in the
generated config. A standard tune-local `modfiles/` directory is detected for
HOC models.

The released Step 1 notebook starts with a PV Allen/ADB example override. Set
`MODEL_SOURCE_OVERRIDES = None` when reusing an existing/staged tune with one
unambiguous source. Keep or edit the override for a new Allen download, multiple
candidate files/templates, nonstandard paths, HOC constructor arguments, or
explicit section mappings. Ambiguity is reported instead of silently selecting a
source.

For a staged file containing multiple templates, use the read-only helper:

```python
from modules.setup import discover_hoc_templates

choices = discover_hoc_templates(tune_dir)
# [{"path": "model/SharedCells.hoc", "template_name": "ExampleCell"}, ...]
```

Results are sorted by tune-relative path and template name and follow Step 1's
existing discovery exclusions. The helper executes no HOC and changes no files.
Set the chosen pair as `paths.hoc_template` and `hoc_template.template_name` in
`MODEL_SOURCE_OVERRIDES`. Step 1 saves it in `cell_config.json`; set the override
to `None` to use the stored selection on reruns. Preserve the complete native
source instead of splitting or renaming its template definitions. Switching to
a copied source with already-defined template names requires a fresh process.

## Canonical Cell Interface

Every loader returns a `LoadedCell` with:

- `h`: the process-global NEURON runtime,
- `model`: the object that owns the instantiated cell,
- `soma`, `dend`, `apic`, `axon`, and `all`: cell-scoped section collections,
- `loader`, `config`, and `source_artifacts`: loader/provenance metadata,
- `Vinit`: an optional loader-provided initialization voltage.

At least one soma section is required. The other named groups may be empty.
When `all` is not mapped, SCP derives it from the canonical groups. Object-owned
loaders never use process-global `h.allsec()` to fill the collection.

## Allen Manifest Loader

An omitted `cell_loader` still selects `allen_manifest` for compatibility:

```json
{
  "cell_loader": "allen_manifest",
  "paths": {
    "manifest": "manifest.json",
    "modfiles": "modfiles"
  },
  "tuning": {
    "soma_diam_multiplier": 1.0
  }
}
```

`tuning.soma_diam_multiplier` is an Allen-only compatibility option. It is
optional and defaults to `1.0`. Other loaders preserve native geometry.

## HOC Template Loader

The generic HOC contract is:

```json
{
  "cell_loader": "hoc_template",
  "paths": {
    "hoc_template": "model/CellTemplate.hoc",
    "modfiles": "modfiles"
  },
  "hoc_template": {
    "template_name": "CellTemplate",
    "constructor_args": [],
    "section_map": {
      "soma": "somatic",
      "dend": ["basal"],
      "apic": ["apical"],
      "axon": ["axonal"],
      "all": "all"
    }
  }
}
```

Each mapping value is an owner attribute name or a list of names. Omit optional
groups that the model does not expose. An empty `section_map` discovers common
owner names (`soma`/`somatic`, `dend`/`basal`, `apic`/`apical`,
`axon`/`axonal`, and `all`). Explicit mappings are useful when native names are
different.

`constructor_args` are passed to the HOC template constructor in order. SCP
loads each resolved HOC entry source once per process. Loading the same template
name from another source requires restarting the Python or Jupyter process;
this avoids silently using a conflicting definition.

HOC tunes must define explicit runtime conditions in `sim_config.json`:

```json
{
  "conditions": {
    "v_init_mV": -70.0,
    "celsius_C": 34.0
  }
}
```

SCP validates and applies these values after construction and before each Step
2, Step 3, or Step 5 protocol/trial. Legacy Allen tunes without the block keep
their loader/runtime fallback behavior.

Bundled object-owned examples are available under `cells/ACT_Spiker/`,
`cells/ACT_BursterCa/`, and `cells/ACT_LAA/`. Each includes an ACT starting
`orig`, an initially identical untuned working `tuned` copy, and a separate
known-target `reference`. See the [ACT example catalog](../guides/act_examples.md).
These are staged examples, not completed scientific walkthroughs. PV remains
the recommended first example for the complete public workflow. Earlier LUT
examples are now maintained as private project workspaces rather than public
loader examples.

## Optional Mechanisms, Targets, and External Tools

`paths.modfiles` is resolved relative to the tune directory. Set it to `null`
or omit the directory when the model uses only NEURON built-in mechanisms. SCP
compiles and loads a library only when `.mod` sources exist. Loaded libraries
are tracked by resolved path and SHA-256; a rebuilt or colliding library needs a
fresh process.

Targets are optional for intrinsic characterization. Either omit
`target_config.json` or set:

```json
{"target_source": {"mode": "none"}}
```

ACT is imported only for explicitly selected ACT target/proposal/optimization
work. Non-Allen ACT use is experimental. BMTool is also optional and receives a
small facade backed by canonical `soma` and `all` sections.

For a cell-only Step 5 run, enable `sim_config.iclamp`; neither
`syn_config.json` nor synapse group files are needed. Synapse-driven runs still
require synapse configuration, and a positive synapse count cannot target an
empty canonical geometry group.

## Current Recording

Step 2, Step 3/FI, and Step 5 use shared reference handling for mechanism
currents, including built-in `extracellular`. Existing results retain keys such
as `pas.i` and `extracellular.i_membrane`. The latter is total membrane current
density, not an individual channel current, and should not be added to channel
currents as though it were an independent contribution.

The low-level `run_current_injection(cell, sim_params, *, record_currents=True)`
keeps current recording enabled for existing callers. With `False`, it bypasses
current discovery and returns an empty `I` dictionary. Step 5 forwards
`iclamp.record_currents` before recording setup and omits `I` when disabled.
Requested currents with unavailable references or invalid native pointers raise
errors identifying the section, mechanism, and variable; source-model pointer
initialization remains the model's responsibility.

## Saved Model Provenance

Saved runs archive loader-declared native sources plus tune-local `.mod` sources
under `model_artifacts/`. The artifact manifest records the loader, tune-relative
target paths, and SHA-256 hashes; compiled outputs are excluded.

`scripts/restore_run_state.py --apply model_artifacts` is opt-in and dry-run by
default. It verifies hashes, refuses source or target path traversal, creates
backups before replacement, warns that restored `.mod` files need recompilation,
and warns that restored HOC sources need a process restart.

## Current Phase Boundary

Parameter-overlay files and new electrophysiology metrics are not part of this interface. Loader adapters normalize native model
ownership and paths; they do not rewrite model-specific parameters or geometry.

## Python, Live Cells, and External Bundles

See [Custom models](../guides/custom_models.md) for complete recipes and contracts.
`python_factory` uses `paths.python_factory`, a named `callable`, JSON `args`/`kwargs`,
and optional `section_map`. `runtime_cell` records a notebook-only construction
that needs an override; `wrap_cell`/`validate_cell` and `SimulationSession.from_cell`
adopt an existing object without reconstruction.

With `model_source`, paths are relative to the selected root. Copy mode uses
the tune's `model_sources/` snapshot; reference mode uses the originals while
retaining that snapshot. External MOD builds use `.scp_build/`. Runtime saves
are explicitly nonreconstructible. External archive materialization is opt-in
and always tune-local.
