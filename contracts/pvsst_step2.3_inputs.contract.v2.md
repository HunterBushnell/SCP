
# Step 2.3 – Input Generation Module Contract (PV–SST, v2)

This document specifies the design and behavior of **Step 2.3 – Input Generation** in the PV–SST single-cell pipeline.

Step 2.3 is responsible for generating **spike-train inputs** for each synapse group defined in a JSON config, *without* touching NEURON directly. All synapse creation and attachment are handled later in **Step 2.4**.

---

## 0. Context in the 2.x pipeline

- **2.1 / 2.2 (cell + geometry)**  
  - Assume we already have:
    - `cell` – an Allen cell (e.g. SST) built with morphology and mechanisms.
    - `geometry` (or `cell.geometry`) – a structure describing segment groups (soma, proximal/distal dendrites, etc.), including distances and segment lists.

- **2.3 (this module)**  
  - Consumes:
    - A JSON synapse configuration (`cell_configs/syn_config.json`).
    - The `geometry` object (optional but required for density-based N_syn resolution).
    - A random number generator and a registry of mode handlers.
  - Produces:
    - `sim_cfg` – normalized simulation configuration.
    - `groups_cfg` – normalized per-group configuration, including a resolved synapse count.
    - `inputs_by_group` – spike trains and metadata for each active synapse group.

- **2.4 (synapses + NEURON, separate module)**  
  - Consumes:
    - `geometry`, `sim_cfg`, `groups_cfg`, `inputs_by_group`.
  - Creates NEURON synapses, VecStim/NetCon, places them on geometry, and attaches spike trains.

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
- `jitter: float | null` – reserved for future jitter logic; currently unused in core modes.

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

- `state: bool | null`  
  - Defaults to `True` (active) if missing.  
  - Treated as inactive if in `{False, 0, "off"}`.
- `mode: str | null`  
  - Required for active groups; e.g. `"homogeneous_poisson"`, `"precomputed"`, `"inhomogeneous_poisson"`.
- `source: dict` – mode-specific input spec:
  - Examples for core modes:
    - `precomputed`:
      - `path: str | null` – JSON file with `{"trains": [...]}` or raw list-of-lists.
    - `homogeneous_poisson`:
      - `freq: float | null` – rate in Hz.
    - `inhomogeneous_poisson` (planned):
      - `path: str | null` – CSV or JSON with rate curve.
      - `time_col: str | null`, `rate_col: str | null`, etc.
- `timing: dict` – common timing fields (ms):
  - `onset_ms: float | null` – onset time of this group relative to `sim_cfg["tstart"]`.
  - `duration_ms: float | null` – duration of active input; if null, defaults to `tstop - tstart`.
  - `stim_tstart_ms: float | null` – sim time when a “stim segment” starts (for inhomogeneous/bio modes).
  - `input_stim_tstart_ms: float | null` – time within the input data that maps to `stim_tstart_ms`.
  - `input_duration_ms: float | null` – how much of the input curve to use.
- `syns: dict` – synapse placement parameters (see below).

##### 1.1.2.1 `syns` block (with geometry-based N_syn)

Each normalized `group_cfg["syns"]` has:

- `type: str | null` – synapse mechanism label (e.g. `"AMPA_NMDA_STP"`, `"GABA_A"`); used in 2.4.
- `N_syn: int | null` – user-requested synapse count; may be `null`.
- `segs: str | dict | null` – segment selection spec (e.g. `"proximal"`, `"distal"`, `"all"`, `"soma"`); interpreted using `geometry` in 2.2/2.4/2.3.
- `dist_func: str | callable-ref | null` – density function identifier to be applied over geometry (e.g. linear decay vs distance).
- `params: dict | null` – additional synapse parameters (2.4 only).
- `N_syn_resolved: int` – final synapse count, **added by 2.3** during preprocessing:

  Resolution rules:

  1. If `N_syn` is not `None` and `N_syn >= 0`:
     - `N_syn_resolved = N_syn`.
  2. Else, if `N_syn is None` and `segs` and `dist_func` are defined and `geometry` is available:
     - Use the segment set indicated by `segs`.
     - For each segment, evaluate `dist_func(distance)` to get a density (synapses/µm).
     - Multiply by the segment length (µm) and sum over all segments to get an expected total synapse count.
     - Convert that expected count to an integer (e.g. floor or round; implementation-defined but consistent).
     - Set `N_syn_resolved` to that integer.
  3. Otherwise:
     - Raise an error indicating that `N_syn` cannot be resolved for this group.

`N_syn_resolved` is computed **once per group** before any mode handler is called and is stored back into `group_cfg["syns"]["N_syn_resolved"]` so that both 2.3 (input generation) and 2.4 (synapse placement) use the same count.

---

## 2. Core Python types

### 2.1 `GroupInputs` dataclass

Defined in `modules_local/inputs.py`:

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class GroupInputs:
    name: str                    # synapse group name, e.g. "pn_exc"
    mode: str                    # mode name, e.g. "precomputed"
    spike_trains: list[np.ndarray]  # list of 1D float arrays (times in ms)
    meta: dict[str, Any]         # per-group metadata (see below)
```

#### 2.1.1 `spike_trains` contract

Each element of `spike_trains` must satisfy:

- Type: `np.ndarray` of `dtype=float`.
- Shape: 1D vector `[n_spikes]`.
- Times in units of ms.
- Times sorted in ascending order.
- All times within `[sim_cfg["tstart"], sim_cfg["tstop"]]`.

In the typical case:

- `len(spike_trains) == N_syn_resolved` (one train per synapse-equivalent “source”).

2.4 can choose to share or tile trains if needed, but the **default contract** is one train per synapse.

#### 2.1.2 `meta` contents (minimum)

At minimum, `meta` should include:

```python
meta = {
    "cfg": group_cfg,                   # the normalized group config dict
    "t_window": (t_start_ms, t_end_ms), # effective time window used by the mode
    "N_syn": gcfg["syns"]["N_syn_resolved"],
    # optional extras:
    # "rng_seed": ...,
    # "source_path": gcfg["source"].get("path"),
    # "mode_params": {...},
}
```

`meta` is designed to be serializable and useful for debugging and later analysis.

---

## 3. High-level phases of Step 2.3

1. **Config normalization (2.3.1–2.3.2)**  
   - Load `cell_configs/syn_config.json`, split into `sim_raw` and `groups_raw`.  
   - Normalize into `sim_cfg` and `groups_cfg` (fill defaults, validate required keys).

2. **Shared resources + registries (2.3.3)**  
   - Initialize `rng: np.random.Generator`.  
   - Build `mode_registry` from core modes and optional user modes.

3. **Group-level processing (2.3.4)**  
   - For each synapse group, decide whether it is active, resolve its final synapse count using `syns` + `geometry`, compute the effective time window, call the mode handler **once** to generate spike trains, and wrap the result in a `GroupInputs`.

4. **Final global checks (2.3.5)**  
   - Ensure every active group has inputs.  
   - Validate consistency between `groups_cfg` and `inputs_by_group`.  
   - Enforce global time bounds `[tstart, tstop]` on all spike times.

---

## 4. Public API (step 2.3)

### 4.1 `check_inputs(...)` – pre-2.3 sanity check

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
  - `FileNotFoundError` if config file is missing.
  - `ValueError` if an active group has no `mode`.
- Returns `(sim_cfg, groups_cfg)` (normalized).

Usage:

- Notebook “pre-2.3” cell calls this to verify config before generating inputs.

---

### 4.2 `generate_inputs(...)` – main 2.3 entrypoint

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

High-level behavior:

1. **Normalization (2.3.1–2.3.2)**  
   - Calls `check_inputs(syn_config_path, verbose=precheck_verbose)` to obtain:
     - `sim_cfg` – normalized sim block.
     - `groups_cfg` – normalized group configs.

2. **Shared resources (2.3.3)**  
   - Ensures `rng` is a `np.random.Generator` via `_init_rng`.
   - Builds default mode registry via `_build_default_mode_registry` (delegates to `input_modes_core.get_default_mode_registry()`).
   - If a user `mode_registry` is provided, merges it with defaults (user overrides built-ins).

3. **Group processing (2.3.4)**  
   - Calls internal `_process_all_groups(sim_cfg, groups_cfg, geometry, mode_registry, rng)` (see section 5).

4. **Final checks (2.3.5)**  
   - Calls `_finalize_inputs(sim_cfg, groups_cfg, inputs_by_group)`:
     - Ensures every active group has an entry in `inputs_by_group`.
     - Verifies `GroupInputs.mode` matches `group_cfg["mode"]`.
     - Verifies all spike times are within `[tstart, tstop]`.

5. Returns `(sim_cfg, groups_cfg, inputs_by_group)`.

No file I/O is performed inside `generate_inputs`. Saving or logging are handled externally.

---

## 5. Internal helpers (conceptual)

Key internal helpers (names may vary slightly in code):

- `_load_and_split_syn_config(path) -> tuple[dict, dict]`  
  - Reads JSON and returns `(sim_raw, groups_raw)`.

- `_normalize_sim_config(sim_raw) -> dict`  
  - Ensures required keys, fills defaults.

- `_normalize_group_configs(groups_raw) -> dict[str, dict]`  
  - Ensures presence of `state`, `mode`, `source`, `timing`, `syns` with benign defaults.

- `_should_skip_group(gname, gcfg) -> bool`  
  - Returns `True` if group is inactive (e.g. `state in {False, 0, "off"}`).

- `_resolve_n_syn(sim_cfg, group_cfg, geometry) -> int`  
  - Implements the `N_syn_resolved` rules in §1.1.2.1 and writes the result into `group_cfg["syns"]["N_syn_resolved"]`.

- `_get_group_time_window(sim_cfg, group_cfg) -> tuple[float, float]`  
  - Uses `sim_cfg["tstart"]`, `sim_cfg["tstop"]` and `group_cfg["timing"]` fields to compute an effective time window `(t_start_ms, t_end_ms)` for that group.

- `_get_n_syn(group_cfg) -> int`  
  - Returns `group_cfg["syns"]["N_syn_resolved"]` and performs basic sanity checks.

- `_resolve_mode_handler(gname, gcfg, mode_registry) -> Callable`  
  - Looks up `gcfg["mode"]` in `mode_registry`; raises clear error if not found.

- `_build_group_inputs(gname, gcfg, trains, sim_cfg, geometry, rng) -> GroupInputs`  
  - Wraps trains into a `GroupInputs`, builds the `meta` dict, and enforces basic per-group checks.

- `_finalize_inputs(sim_cfg, groups_cfg, inputs_by_group)`  
  - Performs cross-group consistency checks and global time-window enforcement.

---

## 6. Internal loop: `_process_all_groups(...)`

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

Algorithm:

1. Initialize `inputs_by_group: dict[str, GroupInputs] = {}`.

2. For each `(gname, gcfg)` in `groups_cfg.items()`:

   1. **Skip logic**  
      - If `_should_skip_group(gname, gcfg)` indicates the group is inactive (e.g. `state in {False, 0, "off"}`) → `continue`.

   2. **Resolve N_syn using geometry and density**  
      - Call `_resolve_n_syn(sim_cfg, gcfg, geometry)` to compute `N_syn_resolved` according to §1.1.2.1.  
      - Store it in `gcfg["syns"]["N_syn_resolved"]`.

   3. **Compute group time window**  
      - Use `_get_group_time_window(sim_cfg, gcfg)` to obtain `(t_start_ms, t_end_ms)` based on `sim_cfg` and `gcfg["timing"]`.

   4. **Resolve mode handler**  
      - `handler = _resolve_mode_handler(gname, gcfg, mode_registry)`.  
      - If `gcfg["mode"]` is unknown, raise an error.

   5. **Generate spike trains (single call per group)**  
      - Call `trains = handler(sim_cfg, gcfg, geometry, rng)`.  
      - The handler uses `N_syn_resolved` (via `_get_n_syn(group_cfg)`) and the time window (via `_get_group_time_window`) to generate exactly `N_syn_resolved` trains, each within `[t_start_ms, t_end_ms]`.

   6. **Wrap into `GroupInputs`**  
      - Build:

        ```python
        group_inputs = GroupInputs(
            name=gname,
            mode=gcfg["mode"],
            spike_trains=trains,
            meta={
                "cfg": gcfg,
                "t_window": (t_start_ms, t_end_ms),
                "N_syn": gcfg["syns"]["N_syn_resolved"],
                # optional: "source_path": gcfg["source"].get("path"),
            },
        )
        ```

      - Store `inputs_by_group[gname] = group_inputs`.

3. Return `inputs_by_group`.

---

## 7. Mode handler contract

Each mode handler is a callable registered in the `mode_registry` mapping.

### 7.1 Function signature

```python
def mode_handler(
    sim_cfg: dict[str, Any],
    group_cfg: dict[str, Any],
    geometry: Any | None,
    rng: np.random.Generator
) -> list[np.ndarray]:
    ...
```

### 7.2 Inputs

- `sim_cfg` – normalized global simulation config (read-only).
- `group_cfg` – normalized config for this group only, including:
  - `state`, `mode`, `source`, `timing`, `syns`, etc.
  - `syns["N_syn_resolved"]` – final synapse count resolved by 2.3.
- `geometry` – optional geometry object; core modes may ignore it, but it is available for geometry-dependent modes.
- `rng` – `np.random.Generator` for all random sampling.

Handlers must **not** mutate `sim_cfg` or `group_cfg` in place.

### 7.3 Outputs

- A list of spike trains: `list[np.ndarray]`, one per synapse-equivalent “source”.

Each array must:

- Be a 1D `np.ndarray` of `dtype=float`.
- Contain spike times in ms, sorted ascending.
- Have all times within `[t_start_ms, t_end_ms]`, where the window is derived from `sim_cfg` and `group_cfg["timing"]`.

The handler is responsible for:

- Reading `N_syn_resolved` via `_get_n_syn(group_cfg)`, and generating exactly that many spike trains.
- Interpreting `source` and `timing` for this group (e.g. rate, delay, baseline vs stim segments).
- Applying any mode-specific timing transformations (delays, baseline, inhomogeneous segments) so that returned spike times are already in global simulation time.

Handlers do **not**:

- Recompute synapse counts or densities.
- Create NEURON objects or synapse mechanisms.
- Perform cross-group coordination.

Cross-group validation is handled by `_finalize_inputs`.

---

## 8. Default mode set (core library)

The default registry provided by `input_modes_core.get_default_mode_registry()` includes:

- `"homogeneous_poisson"` – homogeneous rate, uniform over the group time window.
- `"precomputed"` – pass-through of precomputed trains from JSON (with optional clipping to the group time window).
- `"inhomogeneous_poisson"` – initially a stub raising `NotImplementedError`; to be implemented based on a rate curve and timing fields.

All modes adhere to the handler contract in §7 and are accessed exclusively via the `mode_registry` (no direct calls from the notebook).

---

## 9. Notebook integration (2.2 → 2.3 → pre-2.4)

### 9.1 Pre-2.3 (notebook)

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

### 9.2 2.3 call

```python
import numpy as np

rng = np.random.default_rng(123)

sim_cfg, groups_cfg, inputs_by_group = stim_inputs.generate_inputs(
    syn_config_path=syn_cfg_path,
    geometry=geom,   # or cell.geometry
    rng=rng,
    # mode_registry=None for built-ins only
)

print("sim_cfg:", sim_cfg)
print("\nGenerated input groups:")
for name, gi in inputs_by_group.items():
    print(f"  - {name:15s} mode={gi.mode!r:18}  n_trains={len(gi.spike_trains)}")
```

### 9.3 Optional: saving generated inputs

A separate helper (outside 2.3) may convert `inputs_by_group` to a JSON/NPZ file, including:

- Global metadata (e.g. creation time, seed, sim_cfg).
- Per-group `meta` plus raw spike trains.

This is not part of the core 2.3 contract, but 2.3’s outputs are designed to be serializable.

---

## 10. Status (current implementation vs planned)

- Implemented (in code as of current state):
  - `GroupInputs` dataclass and `inputs.py` orchestrator structure.
  - `check_inputs(...)` and `generate_inputs(...)` entrypoints.
  - Basic `_get_group_time_window`, `_get_n_syn`, `_mode_homogeneous_poisson`, `_mode_precomputed`, and core `mode_registry`.

- Planned / to be implemented or refined:
  - Full `_resolve_n_syn` behavior with geometry-based `N_syn_resolved` as specified above.
  - Full `"inhomogeneous_poisson"` mode using a rate curve and detailed timing fields.
  - Optional helper(s) for saving `inputs_by_group` + metadata to disk.
