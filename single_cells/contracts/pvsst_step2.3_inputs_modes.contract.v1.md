Here is a concrete 2.3 mode contract we can treat as canonical.

---

### 1. Function signature

All modes (core and user) use:

```python
def handler(sim_cfg, group_cfg, geometry, rng):
    ...
```

* `sim_cfg`: normalized sim dict from `inputs._normalize_sim_config`.
* `group_cfg`: normalized group dict from `inputs._normalize_group_configs`, including `time_cfg`.
* `geometry`: geometry dict from Step 2.2 (or `None`), same structure used by `_resolve_n_syn`.
* `rng`: `numpy.random.Generator` instance; all stochastic sampling must use this (no new global RNGs).

Return type:

* `List[np.ndarray]` of spike-time arrays.

---

### 2. Required fields and expectations

**`sim_cfg` must at least provide:**

* `tstart: float`
* `tstop: float`
* `dt: float`
* Optional: `jitter`, `seed`, `cell`, `tune` (modes can read but must not modify).

**`group_cfg` must at least provide:**

* `name: str`
* `mode: str` (used only for logging/diagnostics; dispatch already happened)
* `syns: dict` with:

  * `N_syn_resolved: int >= 0` (preferred, set by `_resolve_n_syn`);
  * modes should obtain `n_syn` via `_get_n_syn(group_cfg)` rather than reading the field directly.
* `time_cfg: dict` with:

  * `anchors: dict` (see below)
  * `blocks: list[dict]` (see below)
* `source: dict` with whatever that mode needs (`freq`, `baseline`, `path`, `time_col`, `rate_col`, `bin_ms`, etc.), following the v4 contract.

**`geometry`**

* Either a full geometry dict from Step 2.2 or `None`.
* Modes that require geometry (e.g., future spatially-structured modes) may assert `geometry is not None` and raise a clear error otherwise.
* Modes that only care about timing or rates can ignore it.

**`rng`**

* Must be used for all randomness.
* Modes must not reseed or replace `rng`; just call methods on it.

---

### 3. `time_cfg` structure and semantics

`time_cfg` is produced by `_calculate_timing(sim_cfg, group_cfg)` and has:

* `anchors: dict` with at least:

  * `sim_tstart: float`
  * `sim_tstop: float`
  * `onset: float`
  * `source_tstart: Optional[float]`
  * `source_tstop: Optional[float]`
  * `baseline_rate_hz: Optional[float]`
* `blocks: list[dict]`, where each block has:

  * `kind: Literal["quiescent", "baseline", "source"]`
  * `t_start_ms: float`
  * `t_end_ms: float`

Requirements:

* Blocks are sorted, non-overlapping, and form a full partition of `[sim_tstart, sim_tstop]`.
* Modes must:

  * Never generate spikes outside `[sim_tstart, sim_tstop]`.
  * Treat `"quiescent"` blocks as “no spikes allowed”.
  * Treat `"source"` blocks as the precise windows in which source-driven activity (e.g., precomputed, inhomogeneous curve, etc.) is allowed.
  * `"baseline"` blocks represent windows in which baseline drive (e.g., homogeneous Poisson at `baseline_rate_hz`) may be present; a given mode may either:

    * implement baseline explicitly, or
    * treat them quiescent if that mode conceptually has no baseline.

High-level pattern for nontrivial modes (what we will do inside the mode functions later):

* For each block `b` in `time_cfg["blocks"]`:

  * generate block-local spikes for that block only (using helpers like `_generate_homogeneous_poisson_trains` or `_generate_inhomogeneous_from_curve` with `t_start_ms`, `t_end_ms`),
  * then concatenate per synapse in chronological order so each synapse’s train is `block1 || block2 || ...`.

---

### 4. Output requirements

Every mode must:

* Compute `n_syn = _get_n_syn(group_cfg)` and return exactly `n_syn` trains.
* Return a `list` of length `n_syn`, where each element is a 1D `np.ndarray` of dtype float (or convertible to float).
* Ensure each train is sorted in strictly non-decreasing order (no backwards time).
* Ensure all spike times lie within `[sim_cfg["tstart"], sim_cfg["tstop"]]` (any minor numerical edge cases around boundaries are handled by `_finalize_inputs`, but modes should not knowingly generate outside).

Mutation rules:

* Modes must not modify `sim_cfg`.
* Modes should treat `group_cfg` as read-only for all fields used by the pipeline (`syns`, `time_cfg`, `source`, `timing`, etc.); any temporary values should be local in the function.
