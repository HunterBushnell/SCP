# PV–SST Step 2.3 – Input Generation Contract (v4)

> Historical draft: kept for design history. For current behavior, prefer
> `modules_local/inputs.py` and `docs/configs_reference.md`.

This document fixes the contract for **Step 2.3 – synaptic input generation** as implemented in `inputs.py` plus the default mode handlers in `input_modes_core.py`.

It is written as an interface / behavior spec, not as exact code.

---

## 0. Scope

Step 2.3 takes a **synapse configuration JSON** and (optionally) a **geometry object** from Step 2.2, and produces:

- a normalized simulation config `sim_cfg`,
- a normalized per-group config dict `groups_cfg`,
- per-group spike trains `inputs_by_group` (as `GroupInputs` objects).

Step 2.3 is responsible for:

- validating / normalizing `sim` and `synapse_groups` JSON;
- resolving synapse counts (`N_syn_resolved`) including optional geometry/density-based counts;
- resolving timing into **mode-agnostic time anchors and blocks** (`time_cfg`);
- dispatching to mode handlers to generate spike trains;
- basic global sanity checks on spike times.

It is **not** responsible for building NEURON synapses or attaching these spike trains to hoc/Cell objects (that is Step 2.4).

---

## 1. Top-level API

### 1.1 `generate_inputs(...)`

```python
def generate_inputs(
    syn_config_path: Path | str,
    geometry: Optional[Any] = None,
    rng: Optional[np.random.Generator] = None,
    mode_registry: Optional[Mapping[str, Any]] = None,
) -> tuple[dict, dict[str, dict], dict[str, GroupInputs]]
```

**Inputs**

- `syn_config_path`: path to a JSON file with top-level keys:
  - `"sim"`: dict
  - `"synapse_groups"`: dict[str, dict]
- `geometry` (optional): geometry / segment-group object from Step 2.2; required only if any group has `syns["N_syn"] is None` (geometry/density-based synapse counts).
- `rng` (optional): NumPy `Generator`. If `None`, 2.3 will construct one (see §3.1.4).
- `mode_registry` (optional): mapping `mode_name: handler`. If `None`, 2.3 uses `input_modes_core.get_default_mode_registry()`.

**Outputs**

- `sim_cfg: dict` – normalized sim-level config (see §3.1).
- `groups_cfg: dict[str, dict]` – normalized per-group configs (see §3.2–3.4).
- `inputs_by_group: dict[str, GroupInputs]` – per-group inputs (see §5). Only **active** groups appear here.

### 1.2 `check_inputs(...)` (optional preview)

```python
def check_inputs(
    syn_config_path: str | Path,
    *,
    verbose: bool = True,
) -> tuple[dict, dict[str, dict]]
```

- Loads the same JSON and runs the same normalization as `generate_inputs`.
- Prints a short summary when `verbose=True`.
- **Not** called by `generate_inputs`; this is a notebook convenience only.

---

## 2. Synapse configuration JSON structure

The JSON loaded from `syn_config_path` must have:

```jsonc
{
  "sim": { ... },
  "synapse_groups": {
    "group_name_1": { ... },
    "group_name_2": { ... }
  }
}
```

### 2.1 `sim` block

Required numeric keys:

- `dt`      – float, simulation time step (ms).
- `tstart`  – float, global simulation start time (ms).
- `tstop`   – float, global simulation stop time (ms).

Optional keys:

- `cell`    – string, cell label (pass-through only).
- `tune`    – string, “tuning” label (pass-through only).
- `jitter`  – float or `null`; reserved for future input jittering (currently stored but unused).
- `seed`    – integer or `null`:
  - if `null` (default): Step 2.3 creates `rng = np.random.default_rng()` with system entropy.
  - if integer: Step 2.3 creates `rng = np.random.default_rng(seed)` unless an `rng` is explicitly passed into `generate_inputs`.

All three numeric keys are converted to `float`; errors on non-numeric input.

### 2.2 `synapse_groups` block

A mapping from group name → group config. Each value must be a dict and is normalized to contain the following top-level keys:

- `state: bool`
- `mode: str`
- `source: dict`
- `timing: dict`
- `syns: dict`

Details follow.

---

## 3. Normalized config semantics

### 3.1 Group `state` and `mode`

For a group `gname` with raw config `gcfg_raw`:

- `state`:
  - default: `True` if missing;
  - `None` is treated as `False` (inactive legacy/scratch groups);
  - must be boolean after normalization.
- `mode`:
  - required, must be a non-empty string;
  - used to look up the mode handler in `mode_registry`.

Any group with `state is False` or missing/empty `mode` is considered **inactive** and will be skipped by 2.3.

### 3.2 `source` block

Normalized keys (all present after normalization, defaulting to `None`):

- `freq`
- `baseline`
- `kind`
- `path`
- `time_col`
- `rate_col`
- `bin_ms`
- `ref`
- `key`

Mode-specific expectations (summary only):

- `homogeneous_poisson`:
  - expects `source["freq"]` (Hz, float).
- `precomputed`:
  - uses `source["trains"]` (inline, not normalized here) and/or `source["path"]` (JSON file with trains).
- `inhomogeneous_poisson`:
  - expects:
    - `source["path"]` (CSV/JSON rate curve),
    - `source["time_col"]`, `source["rate_col"]`,
    - optional `source["bin_ms"]` and `source["baseline"]`.

2.3 does **not** validate mode-specific fields at normalization time; that is left to the corresponding mode handler.

### 3.3 `timing` block

Normalized keys (all present, default `None`):

- `onset_ms`
- `stim_tstart_ms`
- `duration_ms`
- `input_stim_tstart_ms`
- `input_duration_ms`

These raw fields are interpreted only inside the timing helper (§4.2). Modes should **not** reinterpret them directly; instead they should use the derived `time_cfg` attached to the group (see §4.2–4.3).

The high-level intent is:

- `onset_ms`: first time this group is **allowed** to generate spikes.
- `stim_tstart_ms`: time of the “event” in the simulation (e.g. stimulus onset).
- `input_stim_tstart_ms`: corresponding event time in the **source** data (e.g. bio experiment curve).
- `duration_ms`: desired duration of the **source-driven** segment in sim time (may be `null` to mean “to sim.tstop or to the end of available input”).
- `input_duration_ms`: duration of the source segment in the input space (used mainly by inhomogeneous modes).

Exact anchor math is encoded in `_compute_time_anchors` (see §4.2).

### 3.4 `syns` block

Normalized keys:

- `type`       – synapse mechanism label (pass-through to Step 2.4).
- `N_syn`      – non-negative int or `null` (see below).
- `segs`       – segment-group selector, one of `"all"`, `"proximal"`, `"distal"`, `"soma"` (or `null` to default).
- `dist_func`  – density specification for geometry-based synapse counts:
  - `None` → uniform density 1.0;
  - numeric `x` → uniform density `x`;
  - dict-spec: `{"kind": "uniform", "params": {"c": float, "multi": float}}`
    - effective density = `c * multi` synapses / µm.
- `params`     – arbitrary dict of synapse mechanism parameters (opaque to 2.3).

#### 3.4.1 Synapse count resolution (`_resolve_n_syn`)

For each active group, 2.3 resolves an **effective synapse count** and writes it to:

- `group_cfg["syns"]["N_syn_resolved"]` (integer, ≥ 0).

Logic:

1. If `syns["N_syn"] is not None`:
   - convert to int; must be ≥ 0;
   - set `N_syn_resolved = N_syn` and return.
2. If `syns["N_syn"] is None`:
   - require a non-`None` `geometry` object;
   - select a geometry group based on `syns["segs"]`:
     - `"all"` → `"all_dend"`,
     - `"proximal"` → `"proximal"`,
     - `"distal"` → `"distal"`,
     - `"soma"` → `"soma"`;
   - for each segment reference in `geometry["groups"][geom_group_name]`:
     - each `ref` must expose:
       - `.sec` (a NEURON Section with `.L` and `.nseg`),
       - `.dist_um` (distance from soma, µm);
     - segment length in µm: `seg_len_um = sec.L / max(sec.nseg, 1)`;
     - density at that distance: `dens = dens_eq(ref.dist_um)` where `dens_eq`
       comes from `_compile_density_from_spec(syns["dist_func"])`;
     - per-segment synapses: `n_seg = floor(dens * seg_len_um)` (>= 0);
   - total synapses: `N_syn_resolved = sum_n_seg`.

If no segments are present for the selected geometry group, `N_syn_resolved` is set to 0.

**Note:** if you change the cell geometry but reuse the same config, you must re-run Step 2.3 (and Step 2.4) so that `N_syn_resolved` and any downstream synapse placement reflect the new morphology.

---

## 4. Per-group orchestration

### 4.1 `_process_all_groups(...)`

For each group `(gname, gcfg)` in `groups_cfg`:

1. **Skip check** – `_should_skip_group(gname, gcfg)`:
   - returns `True` if `state is False` or `mode` is missing/empty → group is ignored.
2. **Resolve synapse count** – `_resolve_n_syn(sim_cfg, gcfg, geometry)`:
   - writes `gcfg["syns"]["N_syn_resolved"]`;
   - returns `n_syn_resolved` (must be ≥ 0).
3. **Compute timing config** – `_calculate_timing(sim_cfg, gcfg)`:
   - returns a `time_cfg` dict (see §4.2–4.3) and is attached to the group:
     - `gcfg["time_cfg"] = time_cfg`.
4. **Resolve mode handler** – `_resolve_mode_handler(gname, gcfg, mode_registry)`:
   - looks up `gcfg["mode"]` in `mode_registry`;
   - raises on unknown mode.
5. **Run handler** – `_run_mode_handler(handler, sim_cfg, gname, gcfg, geometry, rng)`:
   - calls the handler per the mode contract (§6);
   - expects a `list[np.ndarray]` of spike trains.
6. **Length check**:
   - 2.3 compares `len(spike_trains)` and `n_syn_resolved`:
     - if they differ, 2.3 currently raises a `ValueError`.
7. **Package results** – `_build_group_inputs(gname, gcfg, spike_trains)`:
   - constructs a `GroupInputs` object (see §5);
   - stores it in `inputs_by_group[gname]`.

### 4.2 Timing anchors – `_compute_time_anchors(...)`

`_calculate_timing` is split into:

```python
anchors = _compute_time_anchors(sim_cfg, group_cfg)
blocks  = _build_time_blocks_from_anchors(anchors)
time_cfg = {"anchors": anchors, "blocks": blocks}
```

`_compute_time_anchors` is the **only place** where raw `timing` and `source` fields are interpreted. It produces a dict with keys (all in ms, `float` or `None` where applicable):

- `sim_tstart`       – from `sim_cfg["tstart"]`.
- `sim_tstop`        – from `sim_cfg["tstop"]`.
- `onset`            – earliest allowed spike time for this group (default `sim_tstart`).
- `source_tstart`    – start of the main **source-driven** segment in sim time (may be `None`).
- `source_tstop`     – end of the main source-driven segment (may be `None`).
- `baseline_rate_hz` – constant baseline rate (Hz) to use in non-source intervals, or `None` for purely quiescent non-source periods.

Current design (reflecting Hunter’s experiments):

- `onset` defaults to `sim_cfg["tstart"]` if `timing["onset_ms"] is None`.
- `source_tstart` is conceptually computed as:
  - “simulation stim time” minus “input stim delay”:
    - `timing["stim_tstart_ms"] - timing["input_stim_tstart_ms"]`,
  - with fallbacks if these are `None` (details encoded in the helper).
- `source_tstop` is based on:
  - explicit `timing["duration_ms"]`, if provided, or
  - the duration of available source data (for inhomogeneous modes), or
  - the remaining sim window up to `sim_tstop`.
- `baseline_rate_hz` comes from:
  - `source["baseline"]` if not `None`,
  - otherwise, a mode-specific default (e.g. first point of a rate curve for inhomogeneous modes, or `None` for pure homogeneous background groups).

The precise corner-case handling is implemented in the helper and may be refined. Modes treat anchors as authoritative.

### 4.3 Time blocks – `_build_time_blocks_from_anchors(...)`

Given validated anchors, 2.3 builds a list of **non-overlapping time blocks** in `time_cfg["blocks"]`:

```python
blocks = [
    {"kind": "quiescent" | "baseline" | "source",
     "t_start": float,
     "t_end": float},
    ...
]
```

Conceptual rules:

- All times are clamped to `[sim_tstart, sim_tstop]`.
- If `source_tstart`/`source_tstop` are invalid or absent, we get:
  1. `[sim_tstart, onset)` → `"quiescent"`;
  2. `[onset, sim_tstop)` → `"baseline"` if `baseline_rate_hz > 0`, else `"quiescent"`.
- With a valid source window, we get up to four segments:
  1. `[sim_tstart, onset)`        → `"quiescent"`;
  2. `[onset, source_tstart)`     → `"baseline"` or `"quiescent"`;
  3. `[source_tstart, source_tstop)` → `"source"`;
  4. `[source_tstop, sim_tstop)`  → `"baseline"` or `"quiescent"`.

Zero-length or negative-length intervals are dropped.

`time_cfg` is attached to `group_cfg` and passed into mode handlers; they can use `blocks` and `anchors` to decide where to generate baseline versus source-driven spikes.

---

## 5. `GroupInputs` object

```python
@dataclass
class GroupInputs:
    name: str
    mode: str
    spike_trains: list[np.ndarray] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
```

For each active group, 2.3 constructs a `GroupInputs` instance with:

- `name` – group name (string key in `synapse_groups`).
- `mode` – mode name (copied from `group_cfg["mode"]`).
- `spike_trains` – list of 1D NumPy arrays (dtype float) of spike times (ms).
- `meta` – snapshot metadata; currently includes:

  - `meta["N_syn_resolved"]` – effective synapse count (int), if present in `syns`.
  - `meta["time_anchors_ms"]` – dict with a subset of anchors:
    - `sim_tstart`, `sim_tstop`, `onset`, `source_tstart`, `source_tstop`.
  - `meta["time_blocks"]` – the list of `time_cfg["blocks"]` (see §4.3).

The full normalized `groups_cfg` (per-group config) is returned separately as the second output of `generate_inputs` and is **not** duplicated inside `GroupInputs.meta`.

---

## 6. Mode handler contract

A **mode handler** is any callable with signature:

```python
def mode_handler(
    sim_cfg: dict,
    group_cfg: dict,
    geometry: Optional[Any],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    ...
```

2.3 treats the handler as a pure function of these arguments (no side effects are assumed, but not enforced).

### 6.1 Required behavior

- Must return a `list` of NumPy arrays (`np.ndarray`) of dtype float.
- Each array must be 1D and contain spike times in **simulation time (ms)**.
- Times should lie within `[sim_cfg["tstart"], sim_cfg["tstop"]]` (2.3 enforces this globally in `_finalize_inputs`).
- The handler is expected to respect:
  - `group_cfg["syns"]["N_syn_resolved"]` (or, more generally, `_get_n_syn(group_cfg)` in `input_modes_core`),
  - `group_cfg["time_cfg"]` (anchors + blocks) if present.

Currently 2.3 enforces that:

```python
len(spike_trains) == group_cfg["syns"]["N_syn_resolved"]
```

for active groups; if this is violated, a `ValueError` is raised in `_process_all_groups`.

### 6.2 Access to timing

Although legacy helpers like `_get_group_time_window` still exist in `input_modes_core`, the preferred way for modes to reason about timing is via:

```python
time_cfg = group_cfg.get("time_cfg")
anchors = time_cfg["anchors"]
blocks  = time_cfg["blocks"]
```

For example, an inhomogeneous mode might:

- generate baseline spikes for all blocks with `kind == "baseline"` using the shared homogeneous generator;
- generate source-driven spikes only within blocks with `kind == "source"`;
- leave `"quiescent"` blocks empty.

Homogeneous and precomputed modes may initially ignore `time_cfg` and only use a simpler window; their contracts can be upgraded later without changing Step 2.3 itself.

---

## 7. Finalization checks

After all groups are processed, `_finalize_inputs` performs:

1. **Coverage check**: every group that is **not** skipped by `_should_skip_group` must appear in `inputs_by_group`; otherwise a `ValueError` is raised.
2. **Spike-range check**: for every spike train in every active group:
   - if `train.min() < tstart - eps` or `train.max() > tstop + eps` (with `eps ≈ 1e-9`), a `ValueError` is raised.

No further modifications are made to the spike times in this step.

---

## 8. Non-goals / future extensions

Step 2.3 currently **does not**:

- cache spike trains across runs (even if configs and seeds are identical);
- place synapses on morphology; that is Step 2.4;
- support non-uniform `dist_func` kinds beyond `"uniform"` (N_syn resolution will raise on others);
- handle multi-phase or multi-source per-group inputs beyond a single `"source"` window plus optional baseline segments.

These are potential future extensions and should be added in a backward-compatible way (e.g. keeping the `time_cfg` structure stable where possible).
