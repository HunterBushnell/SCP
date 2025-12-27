Example Run (Minimal)

Goal: run a single PV tune and inspect outputs.

1) Run one trial:
   `python /home/hrbncv/SCP/run_pipeline.py --tune-dir /home/hrbncv/SCP/cells/PV/tunes/seg_tuned --n-trials 1`

2) Find outputs:
   `{tune_dir}/output_data/<output_stem>/run_manifest.json`

3) Open `6_analysis.ipynb` and point it at the run folder.

If modfiles are missing:
- `cd /home/hrbncv/SCP/cells/PV/tunes/seg_tuned/modfiles && nrnivmodl`
