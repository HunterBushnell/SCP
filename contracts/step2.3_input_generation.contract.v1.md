
# Step 2.3 – Input Generation Module Contract (v1)

> Historical draft: kept for design history. For current behavior, prefer
> `modules/inputs.py` and `docs/configs_reference.md`.

## 0. Scope

Step 2.3 is responsible for generating **spike-train inputs** for each synapse group defined in a JSON config, without touching NEURON directly.

- It consumes:
  - A JSON synapse configuration (`cell_configs/syn_config.json`).
  - A geometry object (optional; not required by core modes yet).
  - A random number generator and mode registry.
- It produces:
  - A normalized simulation config (`sim_cfg`).
  - A normalized per-group config (`groups_cfg`).
  - A mapping from group name to `GroupInputs`, each containing spike trains and metadata.

All NEURON-specific work (creating synapses, VecStim/NetCon, attaching to sections) is handled in Step 2.4.

---

## 1. Data contracts

### 1.1 `cell_configs/syn_config.json` (input file)

Top-level JSON structure:

```json
{
  "sim": {
    "cell": "SST",
    "tune": "baseline",
    "dt": 0.025,
    "tstart": 200.0,
    "tstop": 1200.0,
    "jitter": null
  },
  "synapse_groups": {
    "group_name_1": { },
    "group_name_2": { }
  }
}
```

#### 1.1.1 `sim` block

Required keys (normalized in `_normalize_sim_config`):

- `cell: str` – label for the cell/tune (not used numerically in 2.3).
- `tune: str` – label for the tuning condition.
- `dt: float` – simulation timestep in ms.
- `tstart: float` – simulation start time [ms].
- `tstop: float` – simulation end time [ms].
- `jitter: float | null` – currently unused in core modes; reserved for future.

#### 1.1.2 `synapse_groups` block

Each group config (raw) has the form:

```json
"group_name": {
  "state": true,
  "mode": "precomputed",
  "source": { },
  "timing": { },
  "syns": { }
}
```

Normalized in `_normalize_group_configs` into:

- `state: bool | null` (default `True` if missing).
- `mode: str | null` (required for active groups).
- `source: dict` – mode-specific input spec:
  - Examples for core modes:
    - `precomputed`:
      - `path: str | null` – JSON file with `{"trains": [...]}` or raw list-of-lists.
    - `homogeneous_poisson`:
      - `freq: float | null` – rate in Hz.
    - `inhomogeneous_poisson`:
      - `path: str | null` – CSV or JSON with rate curve.
      - `time_col: str | null`, `rate_col: str | null`, etc.
- `timing: dict` – common timing fields (ms):
  - `onset_ms: float | null`
  - `duration_ms: float | null`
  - `stim_tstart_ms: float | null`
  - `input_stim_tstart_ms: float | null`
  - `input_duration_ms: float | null`
- `syns: dict` – synapse placement parameters:
  - `type: str | null` – synapse type label (e.g. `"AMPA_NMDA_STP"`, `"GABA_A"`).
  - `N_syn: int | null` – number of synapses; default = 1.
  - `segs: str | dict | null` – segment selection spec (consumed in 2.4).
  - `dist_func: str | null` – distribution function name (2.4).
  - `params: dict | null` – additional parameters (2.4).

---

## 2. Core Python types

### 2.1 `GroupInputs` dataclass

Defined in `modules/inputs.py`:

```python
@dataclass
class GroupInputs:
    name: str                   # synapse group name, e.g. "pn_exc"
    mode: str                   # mode name, e.g. "precomputed"
    spike_trains: list[np.ndarray]  # list of 1D float arrays (times in ms)
    meta: dict[str, Any]        # per-group metadata (see below)
```

#### 2.1.1 `spike_trains` contract

Each element of `spike_trains` must satisfy:

- Type: `np.ndarray` of `dtype=float`.
- Shape: 1D vector `[n_spikes]`.
- Times in units of ms.
- Times sorted in ascending order.
- All times within `[sim_cfg["tstart"], sim_cfg["tstop"]]`.

`len(spike_trains)` is typically `N_syn` (one train per synapse), but 2.4 may support tiling/reuse; this mapping is defined in the 2.4 contract.

#### 2.1.2 `meta` contents (minimum)

At minimum, `meta` should include:

```python
meta = {
    "cfg": group_cfg,          # the normalized group config dict
    "t_window": (t_start_ms, t_end_ms),
    "N_syn": n_syn
    # optional extras:
    # "rng_seed": ...,
    # "source_path": ...,
    # "mode_params": {...}
}
```

---

## 3. Public API (step 2.3)

### 3.1 `check_inputs(...)` – pre-2.3 sanity check

Signature:

```python
def check_inputs(
    syn_config_path: str | Path,
    *,
    verbose: bool = True
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    ...
```

Behavior:

- Loads and parses `cell_configs/syn_config.json`.
- Runs `_load_and_split_syn_config`, `_normalize_sim_config`, `_normalize_group_configs`.
- Prints a compact summary of each group (state, mode, source.path, N_syn) if `verbose`.
- Raises:
  - `FileNotFoundError` if config file missing.
  - `ValueError` if an active group has no `mode`.
- Returns `(sim_cfg, groups_cfg)` (normalized).

Usage:

- Notebook “pre-2.3” cell calls this to verify config before generating inputs.

---

### 3.2 `generate_inputs(...)` – main 2.3 entrypoint

Signature:

```python
def generate_inputs(
    syn_config_path: str | Path,
    geometry: Any | None = None,
    rng: np.random.Generator | None = None,
    mode_registry: Mapping[str, Callable] | None = None,
    precheck_verbose: bool = False
) -> tuple[
    dict[str, Any],                 # sim_cfg
    dict[str, dict[str, Any]],      # groups_cfg
    dict[str, GroupInputs]          # inputs_by_group
]:
    ...
```

Behavior (high-level):

1. **Normalization (2.3.1–2.3.2)**  
   - Calls `check_inputs(syn_config_path, verbose=precheck_verbose)` to obtain:
     - `sim_cfg` – normalized sim block.
     - `groups_cfg` – normalized group configs.

2. **Shared resources (2.3.3)**  
   - Ensures `rng` is a `np.random.Generator` via `_init_rng`.
   - Builds default mode registry via `_build_default_mode_registry` (delegates to `input_modes_core.get_default_mode_registry()`).
   - If a user `mode_registry` is provided, merges it with defaults (user overrides built-ins).

3. **Group processing (2.3.4)**  
   - Calls internal `_process_all_groups(sim_cfg, groups_cfg, geometry, mode_registry, rng)` (see section 4).

4. **Final checks (2.3.5)**  
   - Calls `_finalize_inputs(sim_cfg, groups_cfg, inputs_by_group)`:
     - Ensures every active group has an entry in `inputs_by_group`.
     - Verifies `GroupInputs.mode` matches `group_cfg["mode"]`.
     - Verifies all spike times are within `[tstart, tstop]`.

5. Returns `(sim_cfg, groups_cfg, inputs_by_group)`.

No file I/O is performed inside `generate_inputs`. Saving or logging are handled externally.

---

## 4. Internal loop: `_process_all_groups(...)`

Signature (internal):

```python
def _process_all_groups(
    sim_cfg: dict[str, Any],
    groups_cfg: dict[str, dict[str, Any]],
    geometry: Any | None,
    mode_registry: Mapping[str, Callable],
    rng: np.random.Generator
) -> dict[str, GroupInputs]:
    ...
```

Behavior:

1. Initialize `inputs_by_group: dict[str, GroupInputs] = {}`.
2. For each `(gname, gcfg)` in `groups_cfg.items()`:
   - If `_should_skip_group(gname, gcfg)` → continue.
   - `handler = _resolve_mode_handler(gname, gcfg, mode_registry)`.
   - `trains = handler(sim_cfg, gcfg, geometry, rng)` – see mode contract below.
   - Build `GroupInputs` via `_build_group_inputs(gname, gcfg, trains, sim_cfg, geometry, rng)`:
     - Ensures trains are in the correct format.
     - Populates `meta` with `cfg`, `t_window`, `N_syn`, etc.
   - Store `inputs_by_group[gname] = group_inputs`.

3. Return `inputs_by_group`.

---

## 5. Mode handler contract

Each mode handler is a callable registered in the `mode_registry` mapping.

### 5.1 Function signature

```python
def mode_handler(
    sim_cfg: dict[str, Any],
    group_cfg: dict[str, Any],
    geometry: Any | None,
    rng: np.random.Generator
) -> list[np.ndarray]:
    ...
```

### 5.2 Inputs

- `sim_cfg` – normalized sim config (global; read-only).
- `group_cfg` – normalized config for this group only; includes:
  - `state`, `mode`, `source`, `timing`, `syns`, etc.
- `geometry` – optional geometry object; may be `None` for core modes.
- `rng` – `np.random.Generator` for all random sampling.

Handlers must **not** mutate `sim_cfg` or `group_cfg` in place.

### 5.3 Outputs

- A list of spike trains: `list[np.ndarray]`.
- Each array obeys the `spike_trains` contract from §2.1.1.
- The handler is responsible for:
  - Interpreting `source` and `timing` fields for this group.
  - Deciding how many trains to return (typically `N_syn`).
  - Ensuring spike times are within `[tstart, tstop]` (modes may also clip; `_finalize_inputs` enforces).

Handlers do **not**:

- Create NEURON objects.
- Attach synapses.
- Perform cross-group coordination.

---

## 6. Notebook integration (2.2 → 2.3 → pre-2.4)

### 6.1 Pre-2.3 (notebook)

Assume 2.2 has produced:

- `cell` – built NEURON cell.
- `geom` or `cell.geometry` – geometry object.

Notebook sets tune directory and config path:

```python
from pathlib import Path
from modules import inputs as stim_inputs

REPO_ROOT = Path("<repo_root>")
TUNE_DIR = REPO_ROOT / "cells" / "SST" / "tunes" / "seg_tuned"
syn_cfg_path = TUNE_DIR / "cell_configs" / "syn_config.json"

sim_cfg_preview, groups_cfg_preview = stim_inputs.check_inputs(syn_cfg_path)
```

### 6.2 2.3 call

```python
import numpy as np

rng = np.random.default_rng(123)

sim_cfg, groups_cfg, inputs_by_group = stim_inputs.generate_inputs(
    syn_config_path=syn_cfg_path,
    geometry=geom,   # or cell.geometry
    rng=rng
    # mode_registry=None for built-ins only
)

print("sim_cfg:", sim_cfg)
print("\nGenerated input groups:")
for name, gi in inputs_by_group.items():
    print(f"  - {name:15s} mode={gi.mode!r:18}  n_trains={len(gi.spike_trains)}")
```

### 6.3 Optional: saving generated inputs

A separate helper (outside 2.3) may convert `inputs_by_group` to a JSON/NPZ file, including:

- Global metadata (e.g. creation time, seed, sim_cfg).
- Per-group `meta` plus raw spike trains.

This is not part of the core 2.3 contract, but 2.3’s outputs are designed to be serializable.

---

## 7. Default mode set (v1)

The default registry provided by `input_modes_core.get_default_mode_registry()` must include:

- `"homogeneous_poisson"` – homogeneous rate, uniform over time window.
- `"precomputed"` – pass-through of precomputed trains from JSON.
- `"inhomogeneous_poisson"` – (initially a stub raising `NotImplementedError`, to be implemented later using a rate curve and timing fields).

All modes adhere to the handler contract in §5 and are accessed exclusively via the `mode_registry` (no direct calls from the notebook).
