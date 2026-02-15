# Status Panel (Terminal UI)

This is a lightweight, live-updating terminal panel you can leave open in a small window. It reads the status file written by `run_slurm.sh` and shows state, metadata, and the tail of the most recent `.err` log.

If per-trial progress is enabled (see below), the panel also shows a progress line like `progress: 3/20 (15%)`.

## Run it
From the server (SSH), in a separate terminal window:

```bash
cd <repo_root>
python scripts/status_panel.py
```

If you submit jobs from a different directory, point at that status file:
```bash
cd <repo_root>
python scripts/status_panel.py \
  --status-file /path/to/submit_dir/logs/status/pvsst_latest.status
```

By default, the panel prefers `pvsst_primary.status` (a stable, task-0/merge view)
and falls back to `pvsst_latest.status` if the primary file does not exist.

## Options
- `--interval 2` refresh interval in seconds
- `--tail-lines 12` how many `.err` lines to show
- `--no-tail` hide log tail section
- `--beep` make a terminal bell on SUCCESS/ERROR transitions
- `--beep-on SUCCESS,ERROR` customize which states beep
- `--no-color` disable colors

Press `q` to quit.

## Notes
- If your terminal supports it, the bell (`--beep`) should play on your local machine even over SSH.
- The panel will try, in order, `run_root/logs`, `output_dir/output_stem/logs`, then the submit-dir `logs/` to find an `.err` to tail.
- Per-trial progress appears once `run_pipeline.py` writes `trial_*` fields into the status file (this is now enabled for multi-trial runs).
