# Model Loaders and the Canonical Cell Interface

SCP constructs every model through a registered loader, then exposes the same
cell-scoped interface to Steps 1–7. `allen_manifest` remains the default and
retains the existing Allen aliases. `hoc_template` is the first object-owned
adapter.

The public entry point is:

```python
from modules.model.load_cell import load_cell

cell = load_cell(cell_config, base_dir=tune_dir)
```

Pass the tune directory explicitly. Current-directory path resolution exists
only for older callers.

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

## Saved Model Provenance

Saved runs archive loader-declared native sources plus tune-local `.mod` sources
under `model_artifacts/`. The artifact manifest records the loader, tune-relative
target paths, and SHA-256 hashes; compiled outputs are excluded.

`scripts/restore_run_state.py --apply model_artifacts` is opt-in and dry-run by
default. It verifies hashes, refuses source or target path traversal, creates
backups before replacement, warns that restored `.mod` files need recompilation,
and warns that restored HOC sources need a process restart.

## Current Phase Boundary

Python-factory loading, parameter-overlay files, and new electrophysiology
metrics are not part of this interface. Loader adapters normalize native model
ownership and paths; they do not rewrite model-specific parameters or geometry.
