SCP results and outputs contract (v1)

* **Scope**
  * Covers artifacts produced by running the pipeline via `run_pipeline.py`, `run_slurm.sh`, or notebooks that call `run_sim.*`.
  * Covers saved results, runtime logs, and mechanism build outputs.
  * Includes optional saved plots when `save_plots` is enabled (interactive-only plots are not persisted otherwise).

* **Output locations (defaults)**
  * **Run folders:** `{tune_dir}/output_data/<output_stem>/`
    * `tune_dir` is the cell tune folder, e.g. `<repo_root>/cells/SST/tunes/seg_tuned`.
    * Override: `run_pipeline.py --output-dir ...` or `OUTPUT_DIR=...` in `run_slurm.sh`.
  * **SLURM logs (per run):** `{tune_dir}/output_data/<output_stem>/logs/`
    * `run_slurm.sh` moves each task’s `pvsst_*` logs into its run folder.
    * Rotated logs may still appear under `<repo_root>/logs/old/<YYYYMMDD_HHMMSS>/`.
  * **Compiled mechanisms (nrnivmodl):** `{tune_dir}/modfiles/x86_64/` (e.g. `libnrnmech.so`).

* **Saved outputs**
  * **Run folder:** `{tune_dir}/output_data/<output_stem>/`
  * **Manifest:** `run_manifest.json` (authoritative index)
  * **Sidecars (default):** `sim_cfg.json`, `meta.json`, `syn_config.json`, `spikes.npz`, `traces.npz`, etc.
    * Optional snapshot extras: `syn_records_by_trial.pkl`, `cell_config.json`, `geometry_config.json`.
  * **Plots (optional):** `plots/output_plot.png`, `plots/inputs_mean.png`, etc. (only when `save_plots: true`).
  * **Full results bundle (default off; set `save_full_results: true` to write):**
    * `{cell}_{tune}_{output_stem}.{pkl|npz}`
    * Written only if `save_full_results: true`.
    * `npz` is compact (mode, sim_cfg_json, meta_json, T, V/V_trials, spikes).
      * Does not include `syn_records`, `inputs`, or `inputs_by_trial`.
  * **Uniqueness rule:** if a run folder exists, `_1`, `_2`, ... are appended to `output_stem`.
  * **When files are saved:**
    * `sim_cfg["output"]` must be non-empty.
    * `run_pipeline.py` sets a timestamped `output_stem` if empty.

* **Results dict schema (pkl)**
  * **Common keys**
    * `mode`: `"single"` or `"multi"`.
    * `sim_cfg`: simulation config used for the run.
    * `spikes`:
      * single: `np.ndarray` of spike times (ms).
      * multi: `List[np.ndarray]`, one per trial.
    * `traces`:
      * if `n_traces_to_save > 0`: includes `T` and `V`.
      * else: `{}`.
    * If `sim_cfg["load"]` is set, results are loaded from disk and
      `meta["loaded_from"]` is added with the resolved path.
  * **Single-only**
    * `syn_records`: dict `{group: List[SynapseRecord]}` (see below).
    * `inputs`: dict `{group: {mode, spike_trains, meta}}` if `n_inputs_to_save > 0`, else `None`.
      * `meta` includes timing anchors (`time_anchors_ms`) and `time_blocks`.
  * **Multi-only**
    * `inputs_by_trial`: list of up to `n_inputs_to_save` trials (or all if `"all"`), each:
      * `{"trial_idx": int, "inputs": {group: {mode, spike_trains, meta}}}`.
    * `traces["V"]` is a list of at most `n_traces_to_save` trials (not all trials).
    * `syn_records_by_trial` (snapshot/debug): list of `{trial_idx, records}`.
  * **meta (always present)**
    * `cell`: cell name.
    * `tune`: tune name.
    * `n_trials`: number of trials.
    * `syn_config`: normalized per-group config (expanded includes, resolved `N_syn_resolved`, timing).
      * Includes derived fields such as `time_cfg` (anchors + blocks).
  * **meta (multi-only)**
    * `avg_rate_curve`: `{"bin_ms": float, "smooth_ms": float, "smooth_mode": str, "t_ms": [...], "rate_hz": [...]}`.
      * `smooth_ms` is a centered moving-average window applied to the binned curve (0 disables smoothing).
    * `input_summaries`: list of per-trial summaries (see below).
* **meta (optional)**
  * `input_stats`: present if `save_input_stats` is true (see below).
  * `randomness`: present when a randomness manager was used (includes `base_seed_used`, `trials_setting`, etc.).
  * `neuron_state` (snapshot): NEURON globals (dt, tstop, celsius, cvode_active, etc.).
  * `synapse_param_snapshot` (snapshot): per-group mechanism params actually present on synapses.

* **SynapseRecord structure**
  * `syn_id`: int
  * `group`: str
  * `type`: str (NEURON mechanism name)
  * `weight`: float (synaptic weight)
  * `distance`: float (um)
  * `section`: str
  * `x`: float (segment position)
  * `spike_times`: List[float] (ms)

* **Input summaries (meta.input_summaries)**
  * For each trial:
    * `groups[group] = { "n_syn": int, "total_spikes": int, "sum_spike_times": float }`.
  * Used to detect identical inputs across trials.
  * Printed to logs if `log_input_summary` is true.

* **Input stats (meta.input_stats)**
  * Enabled by `save_input_stats` (default true).
  * Bin size: `input_stats_bin_ms` if set, else `sim_cfg["bins"]`, else 25 ms.
  * Structure:
    * `bin_ms`, `t_ms`, `tstart_ms`, `tstop_ms`.
    * `trials`: list of per-trial group stats:
      * `groups[group] = { n_syn, total_spikes, rate_hz_total, rate_hz_per_syn,
        counts_by_bin, rate_hz_by_bin_total, rate_hz_by_bin_per_syn }`.
    * `group_means`: per-group means across trials with the same fields prefixed by `mean_`.

* **Runtime log output (run_slurm)**
  * `run_slurm.sh` writes stdout/stderr to `logs/pvsst_*.out/.err` via SLURM filename templates.
  * Log rotation is handled only by `run_slurm.sh` (not by `run_pipeline.py`).
  * Logs include:
    * command line,
    * per-trial spike counts + rate + elapsed time,
    * optional per-trial input spike totals,
    * "Results saved to ..." and total runtime.

* **Mechanism build artifacts**
  * If mechanisms are missing, `nrnivmodl modfiles` runs and produces
    compiled files in `{tune_dir}/modfiles/x86_64/`.

* **Output control flags (sim_cfg)**
  * `save_sidecars` (default true): write manifest + sidecar files.
  * `save_full_results` (default false): write `{cell}_{tune}_{output_stem}.pkl/.npz`.
  * `save_syn_records_sidecar` (default true): write `syn_records.pkl`.
  * `save_syn_records_by_trial` (default false): include per-trial synapse records in the results dict.
  * `snapshot.enabled` (default false): forces full capture (all inputs/traces + synapse records).
    * `save_plots` (default false): write plots under `plots/`.
      * `save_plots_inputs` (default true): include input mean plots.
      * `save_plots_synapses` (default false): include synapse distribution plots.
      * `plots_win_size`, `plots_input_bin_ms`, `plots_input_smooth_ms`, `plots_raster_style` tune the plots.
    * `n_traces_to_save` / `n_inputs_to_save` control how many samples are stored.
      * `n_inputs_to_save` can be `"all"` to keep inputs for every trial.
  * `randomness_mode` (optional): `fixed`, `derived`, or `random` to auto-fill `randomness`.
    * `fixed`: identical trials; use a fixed seed for reproducibility.
    * `derived`: varies per trial, reproducible if a seed is set.
    * `random`: fully random per trial (fresh entropy each run).
  * `load`, `save/output`, `append/append_to` can be specified as tuples:
    * `load: [enabled, path]`
    * `save: [enabled, stem, format, full_results]` (if `full_results` is true and `save_sidecars` is unset, sidecars are disabled)
    * `append: [enabled, path]`
    * Back-compat: `output` and `append_to` are still accepted.
    * `run_pipeline.py` uses the target run’s `sim_cfg` when appending to an existing run.
  * `plots_profile` (optional): `"off" | "basic" | "inputs" | "full"`; fills plot-saving flags
    (`save_plots`, `save_plots_inputs`, `save_plots_synapses`) when those keys are omitted.
  * **Precomputed inputs note:** if you want to reuse a new run as
    `source.path` for `precomputed` mode, set `save_full_results: true`
    so a `.pkl` exists, or export trains explicitly.

* **Manifest schema (run_manifest.json)**
  * `format_version`: int
  * `mode`: "single" | "multi"
  * `output_stem`: str
  * `files`: mapping of logical names → filenames within the run folder

* **Units**
  * Times are in ms, rates are in Hz.
  * Spike trains are arrays of spike times in ms.

* **Quick load + plot snippet**
  * Minimal Python example for saved results (single or multi):
```python
from modules_local import run_sim
from modules_local.analysis import plotting

results = run_sim.load_results("cells/SST/tunes/seg_tuned/output_data/slurm_20251218_231500")
plotting.plot_results(results)
```
  * Notebook alternative: `<repo_root>/6_analysis.ipynb`
