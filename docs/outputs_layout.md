Outputs Layout

Runs are saved under the tune directory:
- `{tune_dir}/output_data/<output_stem>/`

Typical structure
- `run_manifest.json`
- `sim_cfg.json`
- `meta.json`
- `syn_config.json`
- `spikes.npz`
- `traces.npz`

Optional sidecars
- `input_stats.json` (if `save_input_stats`)
- `inputs_sample.pkl` (if inputs are saved; used by precomputed sources)
- `syn_records.pkl` (if `save_syn_records_sidecar`)
- `syn_records_by_trial.pkl` (if `save_syn_records_by_trial`)
- `cell_recordings.pkl` (if `cell_recordings` present in single-run results)
- `cell_recordings_by_trial.pkl` (if `cell_recordings_by_trial` present in multi-run results)
- `<specimen_id>_fit.json` (if `save_fit_json_sidecar`, default on, and fit file is found)
- `plots/` (if `save_plots` or a plot profile enables it)
- `{cell}_{tune}_{output_stem}.pkl` (if `save_full_results`)

Array runs
- Merged arrays: `output_data/<batch_stem>/results/`
- Per-task outputs (temporary): `output_data/<batch_stem>/parts/`
- Logs may be copied into `output_data/<batch_stem>/logs/`

Notes
- `run_manifest.json` is the authoritative index of files written for a run.
- Output format is controlled by `output_format` and `save_full_results`.
