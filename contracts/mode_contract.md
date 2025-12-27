# Mode Contract (draft)

**Signature**
- `handler(sim_cfg: dict, group_cfg: dict, geometry: Optional[Any], rng: np.random.Generator) -> list[np.ndarray]`

**Inputs (normalized upstream)**
- `sim_cfg`: normalized; contains `tstart`, `tstop`, `dt`, etc. Do not mutate.
- `group_cfg`: normalized; includes:
  - `syns["N_syn_resolved"]` (int, required).
  - `time_cfg` with `anchors` (`sim_tstart`, `sim_tstop`, `onset`, `source_tstart`, `source_tstop`, `baseline_rate_hz`) and `blocks` (list of `{kind, t_start, t_end}`).
  - `mode` (str), `source` (mode params), `timing` (raw timing if needed), `syns` (type/params if relevant).
- `geometry`: optional; use only if the mode needs morphology.
- `rng`: per-group/per-trial generator; use exclusively for randomness (no global `np.random`).

**Outputs**
- `list[np.ndarray]` of spike trains (ms, sorted, simulation time), length exactly `N_syn_resolved`.
- All spikes clipped to `[sim_cfg["tstart"], sim_cfg["tstop"]]`.

**Behavior expectations**
- Do not mutate `sim_cfg` or `group_cfg`.
- Use `group_cfg["time_cfg"]` to honor onset/source/baseline windows; stitch segments into a single train per synapse.
- Handle zero/empty-rate cases by returning empty arrays (still length `N_syn_resolved`).
- If drawing from files/curves, validate inputs; ensure consistent per-synapse mapping (1 train or N trains).

**Randomness**
- All stochastic draws go through the provided `rng`. No global state.

**Error handling**
- Raise clear errors on invalid configs (missing mode params, wrong train count, bad file paths).

**Registration**
- Core modes live in `input_modes_core.py`; user modes in `input_modes_user.py` via `get_user_mode_registry()`; names must be unique; user modes override on collision.

**Clipping/sorting**
- Ensure each returned train is sorted and within bounds; jitter/noise added must be clipped.

**Determinism**
- Given the same `rng` state and configs, outputs must be reproducible.
