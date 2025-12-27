# Step 2.3 – Input Generation Module Contract (PV–SST, v3)

This document specifies the design and behavior of **Step 2.3 – Input Generation** in the PV–SST single‑cell pipeline.

Step 2.3 is responsible for generating **spike‑train inputs** for each synapse group defined in a JSON config, *without* touching NEURON directly. All synapse creation and attachment are handled later in **Step 2.4**.

---

## 0. Context in the 2.x pipeline

- **2.1 / 2.2 (cell + geometry)**  
  Assume we already have:
  - `cell` – an Allen cell (e.g. SST) built with morphology and mechanisms.
  - `geometry` (or `cell.geometry`) – a structure describing segment groups (soma, proximal/distal dendrites, etc.), including distances and segment lists.

- **2.3 (this module)**  
  Consumes:
  - A JSON synapse configuration (`cell_configs/syn_config.json`).
  - The `geometry` object (optional but required for density‑based N_syn resolution).
  - A random number generator and a registry of mode handlers.  
  Produces:
  - `sim_cfg` – normalized simulation configuration.
  - `groups_cfg` – normalized per‑group configuration, including a resolved synapse count.
  - `inputs_by_group` – spike trains and metadata for each active synapse group.

- **2.4 (synapses + NEURON, separate module)**  
  Consumes:
  - `geometry`, `sim_cfg`, `groups_cfg`, `inputs_by_group`.  
  Creates NEURON synapses, VecStim/NetCon, places them on geometry, and attaches spike trains.

---

## 1. Data contracts

### 1.1 `cell_configs/syn_config.json` (input file)

Top‑level JSON structure:

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
- `jitter: float | null` – reserved for future jitter logic; currently unused in core modes.

#### 1.1.2 `synapse_groups` block

Each raw group config has the form:

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

- `state: bool | null`  
  - Defaults to `True` (active) if missing.  
  - Treated as inactive if explicitly `False` or `None`.
- `mode: str | null`  
  - Required for active groups; e.g. `"homogeneous_poisson"`, `"precomputed"`, `"inhomogeneous_poisson"`.
- `source: dict` – mode‑specific input spec. Core expectations:
  - For `precomputed`:
    - `path: str | null` – JSON file path.
    - `trains: list | null` – inline list‑of‑lists of spike times.
  - For `homogeneous_poisson`:
    - `freq: float | null` – rate in Hz.
  - For `inhomogeneous_poisson` (planned):
    - `path: str | null` – CSV/JSON with rate curve.
    - `time_col: str | null`, `rate_col: str | null`, `bin_ms`, `baseline`, etc.
- `timing: dict` – common timing fields (ms), all optional:
  - `onset_ms: float | null` – onset time of this group relative to `sim_cfg["tstart"]`.
  - `duration_ms: float | null` – duration of active input; if null, defaults to `tstop - tstart`.
  - `stim_tstart_ms: float | null` – sim time when a “stim segment” starts (for inhomogeneous/bio modes).
  - `input_stim_tstart_ms: float | null` – time within the input data that maps to `stim_tstart_ms`.
  - `input_duration_ms: float | null` – how much of the input curve to use.
- `syns: dict` – synapse placement parameters (see below).

##### 1.1.2.1 `syns` block (with geometry‑based N_syn)

Each normalized `group_cfg["syns"]` has:

- `type: str | null` – synapse mechanism label (e.g. `"AMPA_NMDA_STP"`, `"GABA_A"`); used in 2.4.
- `N_syn: int | null` – user‑requested synapse count; may be `null`.
- `segs: str | dict | null` – segment selection spec (e.g. `"proximal"`, `"distal"`, `"all"`, `"soma"`); interpreted using `geometry`.
- `dist_func: Any` – density function spec over geometry:
  - `None` → uniform density.
  - `number` → constant density.
  - `callable` → direct use.
  - `dict` spec (e.g. `{"kind": "uniform", "params": {"c": 2.0}}`); see `_compile_density_from_spec`.
- `params: dict | null` – additional synapse parameters (for 2.4).
- `N_syn_resolved: int` – final synapse count, **added by 2.3** during preprocessing.

Resolution rules (implemented in `_resolve_n_syn`):

1. If `N_syn` is not `None` and `N_syn >= 0`:
   - `N_syn_resolved = N_syn`.
2. Else (i.e. `N_syn is None`):
   - Require `geometry` and a valid `segs` selector.  
   - Map `segs` into a geometry group (e.g. `"proximal" → "proximal"`).  
   - For each segment reference with length `L_seg` and distance `dist_um`:
     - Evaluate `dens = dens_eq(dist_um)` from `dist_func` spec.
     - Compute `n_seg = floor(dens * L_seg)`.
   - Sum `n_seg` over all segments → `total_n_syn`.
   - Set `N_syn_resolved = total_n_syn` (int, ≥ 0).
3. If neither explicit `N_syn` nor geometry/density is available:
   - Raise an error; the group is ill‑posed.

`N_syn_resolved` is computed **once per group** before any mode handler is called and stored in `group_cfg["syns"]["N_syn_resolved"]` for both 2.3 and 2.4.

---

## 2. Core Python types

### 2.1 `GroupInputs` dataclass

Defined in `modules_local/inputs.py`:

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class GroupInputs:
    name: str                    # synapse group name, e.g. "pn_exc"
    mode: str                    # mode name, e.g. "precomputed"
    spike_trains: list[np.ndarray]  # list of 1D float arrays (times in ms)
    meta: dict[str, Any] = field(default_factory=dict)
```

#### 2.1.1 `spike_trains` contract

Each element of `spike_trains` must satisfy:

- Type: `np.ndarray` of `dtype=float`.
- Shape: 1D vector `[n_spikes]`.
- Times in units of ms.
- Times sorted in ascending order.
- All times within `[sim_cfg["tstart"], sim_cfg["tstop"]]`.

Default assumption:

- `len(spike_trains) == N_syn_resolved` (one train per synapse‑equivalent “source”).  
  2.3 enforces this equality; if a mode wants tiling/reuse later, the contract will be updated explicitly.

#### 2.1.2 `meta` contents (minimum)

At minimum, 2.3 populates:

```python
meta = {
    "t_window_ms": {
        "t_start": t_start_ms,
        "t_end": t_end_ms,
    },
    "N_syn_resolved": N_syn_resolved,
    # optional future keys:
    # "cfg": group_cfg,
    # "rng_seed": ...,
    # "source_path": group_cfg["source"].get("path"),
}
```

`meta` is designed to be serializable and useful for debugging and later analysis.

---

## 3. High‑level phases of Step 2.3

1. **Config normalization (2.3.1–2.3.2)**  
   - Load `cell_configs/syn_config.json`, split into `sim_raw` and `groups_raw`.  
   - Normalize into `sim_cfg` and `groups_cfg` (fill defaults, validate required keys).

2. **Shared resources + registries (2.3.3)**  
   - Initialize `rng: np.random.Generator`.  
   - Build `mode_registry` from core modes and optional user modes.

3. **Group‑level processing (2.3.4)**  
   - For each synapse group:
     - Check if active (`state` + `mode`).
     - Resolve final synapse count via `_resolve_n_syn`, writing `N_syn_resolved`.
     - Compute effective time window via `_get_group_time_window`.
     - Resolve and call a mode handler once per group to generate spike trains.
     - Package result into `GroupInputs`.

4. **Final global checks (2.3.5)**  
   - Ensure every active group has inputs.  
   - Validate consistency between `groups_cfg` and `inputs_by_group`.  
   - Enforce global time bounds `[tstart, tstop]` on all spike times.

---

## 4. Public API (step 2.3)

### 4.1 `check_inputs(...)` – pre‑2.3 sanity check

Signature:

```python
def check_inputs(
    syn_config_path: str | Path,
    *,
    verbose: bool = True,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    ...
```

Behavior:

- Loads and parses `cell_configs/syn_config.json`.
- Runs `_load_and_split_syn_config`, `_normalize_sim_config`, `_normalize_group_configs`.
- Prints a compact summary of each group (state, mode, source.path, N_syn) if `verbose`.
- Raises:
  - `FileNotFoundError` if config file is missing.
  - `ValueError` if an active group has no `mode`.
- Returns `(sim_cfg, groups_cfg)` (normalized).

Usage (notebook “pre‑2.3” cell):

```python
sim_cfg_chk, groups_cfg_chk = stim_inputs.check_inputs(syn_cfg_path)
```

---

### 4.2 `generate_inputs(...)` – main 2.3 entrypoint

Signature:

```python
def generate_inputs(
    syn_config_path: Path | str,
    geometry: Any | None = None,
    rng: np.random.Generator | None = None,
    mode_registry: Mapping[str, Any] | None = None,
) -> tuple[
    dict[str, Any],                 # sim_cfg
    dict[str, dict[str, Any]],      # groups_cfg
    dict[str, GroupInputs],         # inputs_by_group
]:
    ...
```

High‑level behavior:

1. **Normalization (2.3.1–2.3.2)**  
   - Calls `_load_and_split_syn_config` to get `sim_raw`, `groups_raw`.  
   - Normalizes to `sim_cfg`, `groups_cfg`.

2. **Shared resources (2.3.3)**  
   - Ensures `rng` is a `np.random.Generator` via `_init_rng`.  
   - Builds default mode registry via `_build_default_mode_registry()` (delegates to `input_modes_core.get_default_mode_registry()`).
   - If a user `mode_registry` is provided, it can override or extend defaults at the notebook level.

3. **Group processing (2.3.4)**  
   - Calls `_process_all_groups(sim_cfg, groups_cfg, geometry, mode_registry, rng)` (see §6).

4. **Final checks (2.3.5)**  
   - Calls `_finalize_inputs(sim_cfg, groups_cfg, inputs_by_group)`:
     - Ensures every active group has an entry in `inputs_by_group`.
     - Verifies all spike times are within `[tstart, tstop]`.

5. Returns `(sim_cfg, groups_cfg, inputs_by_group)`.

No file I/O (saving outputs) is performed inside `generate_inputs`. Saving/logging are external.

---

## 5. Internal helpers (conceptual)

Core internal helpers (names match the current implementation):

- `_load_and_split_syn_config(path) -> tuple[dict, dict]`  
  Reads JSON and returns `(sim_raw, groups_raw)`.

- `_normalize_sim_config(sim_raw) -> dict`  
  Ensures required keys and types, fills defaults.

- `_normalize_group_configs(groups_raw) -> dict[str, dict]`  
  Ensures presence of `state`, `mode`, `source`, `timing`, `syns` with benign defaults.

- `_init_rng(rng) -> np.random.Generator`  
  Returns an existing RNG or constructs a new one.

- `_build_default_mode_registry() -> dict[str, Any]`  
  Delegates to `input_modes_core.get_default_mode_registry()`.

- `_compile_density_from_spec(dist_spec) -> callable`  
  Converts `dist_func` spec into a density function.

- `_resolve_n_syn(sim_cfg, group_cfg, geometry) -> int`  
  Implements `N_syn_resolved` rules in §1.1.2.1 and writes the result into `group_cfg["syns"]["N_syn_resolved"]`.

- `_get_group_time_window(sim_cfg, group_cfg) -> tuple[float, float]`  
  Uses `sim_cfg["tstart"]`, `sim_cfg["tstop"]` and `group_cfg["timing"]` to compute `(t_start_ms, t_end_ms)`.

- `_should_skip_group(gname, gcfg) -> bool`  
  Returns `True` if group is inactive (e.g. `state is False` or `mode` is missing/empty).

- `_resolve_mode_handler(gname, gcfg, mode_registry) -> Callable`  
  Looks up `gcfg["mode"]` in `mode_registry`; raises if not found.

- `_run_mode_handler(handler, sim_cfg, group_name, group_cfg, geometry, rng) -> list[np.ndarray]`  
  Calls the mode handler and enforces basic type checks.

- `_build_group_inputs(gname, gcfg, trains, t_window) -> GroupInputs`  
  Wraps trains into `GroupInputs`, builds `meta`, and stores timing and `N_syn_resolved`.

- `_finalize_inputs(sim_cfg, groups_cfg, inputs_by_group)`  
  Performs cross‑group consistency checks and global time‑window enforcement.

---

## 6. Internal loop: `_process_all_groups(...)`

Signature:

```python
def _process_all_groups(
    sim_cfg: dict[str, Any],
    groups_cfg: dict[str, dict[str, Any]],
    geometry: Any | None,
    mode_registry: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, GroupInputs]:
    ...
```

Algorithm:

1. Initialize `inputs_by_group: dict[str, GroupInputs] = {}`.

2. For each `(gname, gcfg)` in `groups_cfg.items()`:

   1. **Skip logic**  
      - If `_should_skip_group(gname, gcfg)` → `continue`.

   2. **Resolve N_syn**  
      - Call `_resolve_n_syn(sim_cfg, gcfg, geometry)` to compute and store `N_syn_resolved`.

   3. **Compute group time window**  
      - `t_start_ms, t_end_ms = _get_group_time_window(sim_cfg, gcfg)`.

   4. **Resolve mode handler**  
      - `handler = _resolve_mode_handler(gname, gcfg, mode_registry)`.

   5. **Generate spike trains**  
      - `trains = _run_mode_handler(handler, sim_cfg, gname, gcfg, geometry, rng)`.  
      - Handlers are expected to:
        - Use `N_syn_resolved` to decide how many trains to generate.
        - Use the timing fields to keep spikes within `[t_start_ms, t_end_ms]` (2.3 will validate).

   6. **Consistency check**  
      - If `len(trains) != N_syn_resolved` → raise `ValueError`.

   7. **Wrap into `GroupInputs`**  
      - Build and store:

        ```python
        group_inputs = _build_group_inputs(
            group_name=gname,
            group_cfg=gcfg,
            spike_trains=trains,
            t_window=(t_start_ms, t_end_ms),
        )
        inputs_by_group[gname] = group_inputs
        ```

3. Return `inputs_by_group`.

---

## 7. Mode handler global contract

Each mode handler is a callable registered in the `mode_registry` mapping.

### 7.1 Function signature (all modes, core + user)

```python
def mode_handler(
    sim_cfg: dict[str, Any],
    group_cfg: dict[str, Any],
    geometry: Any | None,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    ...
```

### 7.2 Inputs (shared rules)

- `sim_cfg` – normalized global simulation config (read‑only).
- `group_cfg` – normalized config for this group only, including:
  - `state`, `mode`, `source`, `timing`, `syns`, etc.
  - `syns["N_syn_resolved"]` – final synapse count resolved by 2.3.
- `geometry` – optional geometry object; core modes may ignore it, but it is available for geometry‑dependent modes.
- `rng` – `np.random.Generator` for all random sampling.

Handlers must **not** mutate `sim_cfg` or `group_cfg` in place and must not touch NEURON or create mechanisms; they generate *only* spike times.

### 7.3 Outputs (shared rules)

- A list of spike trains: `list[np.ndarray]`, one per synapse‑equivalent “source”.

Each array must:

- Be a 1D `np.ndarray` of `dtype=float`.
- Contain spike times in ms, sorted ascending.
- Have all times within `[sim_cfg["tstart"], sim_cfg["tstop"]]`.
- The list length must satisfy:

  ```python
  len(trains) == group_cfg["syns"]["N_syn_resolved"]
  ```

2.3 enforces this equality; if tiling/reuse is desired later, the contract will be relaxed explicitly.

Handlers are responsible for:

- Reading `N_syn_resolved` (via `_get_n_syn(group_cfg)` or directly from `group_cfg["syns"]`).
- Interpreting `source` and `timing` for this group (rate, delays, baseline vs stim segments, etc.).
- Applying timing so that returned spike times are already in global simulation time.

---

## 8. Default modes (core library) – specific contracts

The default registry provided by `input_modes_core.get_default_mode_registry()` includes:

- `"precomputed"` – pass‑through of precomputed trains from JSON or inline data (with clipping).
- `"homogeneous_poisson"` – homogeneous rate, uniform over the group’s active time window.
- `"inhomogeneous_poisson"` – **planned** in this contract; currently implemented as a stub raising `NotImplementedError`.

### 8.1 Mode `"precomputed"`

**Name:** `"precomputed"` (aliases such as `"spike_trains"` may be mapped to the same handler).

**Additional input expectations**

From `group_cfg["source"]`:

- `source["trains"]` (optional inline):
  - `List[List[float]]` – each inner list is spike times (ms) in sim time.
- OR `source["path"]`:
  - Path to a JSON file.
  - File must be either:
    - `{"trains": [[...], [...], ...]}` **or**
    - A raw `[[...], [...], ...]`.

From `group_cfg["syns"]`:

- `N_syn_resolved: int >= 0` – final synapse count.

From `group_cfg["timing"]`:

- `onset_ms`, `duration_ms` – define group window `[t_start, t_end]` via `_get_group_time_window`.

**Behavior constraints**

- If **both** `source["trains"]` and `source["path"]` are `None` → raise `ValueError`.
- If `N_syn_resolved == 0` → return `[]`.
- Load `trains_raw` as a Python list of trains:
  - If file data is a dict, require a `"trains"` key.
  - If file data is a list, treat it as trains directly.
- If `len(trains_raw) > N_syn_resolved`:
  - Take `trains_raw[:N_syn_resolved]`.
- If `len(trains_raw) < N_syn_resolved`:
  - Do **not** silently duplicate or pad; return the available trains.
  - 2.3’s length check will raise a mismatch error (helps catch misconfigurations).
- For each train:
  - Convert to `np.ndarray` of `float`.
  - Clip to `[t_start, t_end]` using the time window from `_get_group_time_window`.
- Output: list of `np.ndarray` after clipping.

### 8.2 Mode `"homogeneous_poisson"`

**Name:** `"homogeneous_poisson"`

**Additional input expectations**

From `group_cfg["source"]`:

- `source["freq"]`:
  - Required; must be numeric (Hz).  
  - If not numeric → raise `ValueError`.

From `group_cfg["syns"]`:

- `N_syn_resolved: int >= 0`.

From `group_cfg["timing"]`:

- `onset_ms`, `duration_ms` → define `[t_start, t_end]`.

**Behavior constraints**

- If `N_syn_resolved == 0` → return `[]`.
- If `freq <= 0` or `t_end <= t_start`:
  - Return a list of `N_syn_resolved` empty arrays (no spikes).
- Else:
  - For each synapse (0..`N_syn_resolved-1`), generate a homogeneous Poisson process with:
    - Rate `freq` (Hz).
    - Window `[t_start, t_end]` ms.
    - Using only `rng` for randomness.
- Output: list of `N_syn_resolved` `np.ndarray` spike trains.

### 8.3 Mode `"inhomogeneous_poisson"` (planned)

**Name:** `"inhomogeneous_poisson"`

**Current implementation:** stub that raises `NotImplementedError`.

**Intended input expectations**

From `group_cfg["source"]`:

- Rate curve metadata:
  - `path`: CSV/JSON path with rate data.
  - `time_col`: column name for times (ms or s).
  - `rate_col`: column name for rates (Hz).
  - `bin_ms`: optional bin width.
  - `baseline`: optional baseline rate (Hz) for pre/post.

From `group_cfg["timing"]`:

- Rich timing fields to align input domain ↔ sim domain:
  - `onset_ms`, `duration_ms`.
  - `stim_tstart_ms`.
  - `input_stim_tstart_ms`.
  - `input_duration_ms`.

From `group_cfg["syns"]`:

- `N_syn_resolved: int >= 0`.

**Intended behavior outline**

- Load `(times_ms, rates_hz)` from file.
- Use timing fields to:
  - Choose a slice of the curve.
  - Map that slice into simulation time.
  - Possibly construct:
    - Pre‑baseline segment (homogeneous Poisson at baseline rate).
    - Stim segment (inhomogeneous Poisson from rate curve).
    - Post‑baseline segment (baseline again).
- Use `_generate_inhomogeneous_from_curve(...)` to generate trains.
- Output: list of `N_syn_resolved` `np.ndarray` spike trains.

Until implemented, users should rely on `precomputed` mode with externally generated trains.

---

## 9. User‑defined modes

User modes live in `input_modes_user.py` and are registered via:

```python
def get_user_mode_registry() -> Mapping[str, Any]:
    return {
        "my_custom_mode": my_custom_mode,
        ...
    }
```

Each user mode must **fully** satisfy the global handler contract (§7):

- Same signature as core modes.
- Use `N_syn_resolved` to decide how many trains to return.
- Keep spikes within `[sim_cfg["tstart"], sim_cfg["tstop"]]`.
- Return `list[np.ndarray]` with length `N_syn_resolved`.

Notebook code merges registries, for example:

```python
from modules_local import inputs as stim_inputs
from modules_local import input_modes_user

default_modes = stim_inputs._build_default_mode_registry()
user_modes = input_modes_user.get_user_mode_registry()
mode_registry = {**default_modes, **user_modes}

sim_cfg, groups_cfg, inputs_by_group = stim_inputs.generate_inputs(
    syn_config_path=syn_cfg_path,
    geometry=geom,
    rng=rng,
    mode_registry=mode_registry,
)
```

---

## 10. Notebook integration (2.2 → 2.3 → pre‑2.4)

### 10.1 Pre‑2.3

Assume 2.2 has produced:

- `cell` – built NEURON cell.
- `geom` or `cell.geometry` – geometry object.

Notebook sets tune directory and config path:

```python
from pathlib import Path
from modules_local import inputs as stim_inputs

TUNE_DIR = Path("/home/hrbncv/SCP/cells/SST/tunes/seg_tuned")
syn_cfg_path = TUNE_DIR / "cell_configs" / "syn_config.json"

sim_cfg_preview, groups_cfg_preview = stim_inputs.check_inputs(syn_cfg_path)
```

### 10.2 2.3 call

```python
import numpy as np

rng = np.random.default_rng(123)

sim_cfg, groups_cfg, inputs_by_group = stim_inputs.generate_inputs(
    syn_config_path=syn_cfg_path,
    geometry=geom,   # or cell.geometry
    rng=rng,
)
```

---

## 11. Status (current implementation vs planned)

- Implemented in code (matching this contract):
  - `GroupInputs` dataclass and `inputs.py` orchestrator.
  - `check_inputs(...)`, `generate_inputs(...)`.
  - `_resolve_n_syn` with geometry/density logic (N_syn_resolved).
  - `_get_group_time_window`, `_should_skip_group`, `_resolve_mode_handler`,
    `_run_mode_handler`, `_build_group_inputs`, `_finalize_inputs`.
  - Core modes:
    - `"precomputed"` – loads/clips trains, enforces file/inline requirements.
    - `"homogeneous_poisson"` – homogeneous Poisson trains per synapse.

- Planned:
  - Full `"inhomogeneous_poisson"` mode using a rate curve and timing fields.
  - Optional serialization helpers for `inputs_by_group` + metadata.
