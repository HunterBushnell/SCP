Quickstart

This is the shortest path to run an existing tuned cell.

Option A: Notebook
1) Open `5_local.ipynb`.
2) Set the cell/tune at the top.
3) Run all.
4) Open `6_analysis.ipynb` for analysis/comparisons.

Option B: CLI
- Single run:
  `python /home/hrbncv/SCP/run_pipeline.py --tune-dir /home/hrbncv/SCP/cells/PV/tunes/seg_tuned --n-trials 1`

- If modfiles are missing, build them once per tune:
  `cd /home/hrbncv/SCP/cells/PV/tunes/seg_tuned/modfiles && nrnivmodl`

Outputs land under:
- `{tune_dir}/output_data/<output_stem>/`

See also
- `cli_slurm.md` for batch runs.
- `outputs_layout.md` for output file structure.
