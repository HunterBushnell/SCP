Example Run (Minimal)

Goal: run a single PV tune and inspect outputs.

1) Verify setup:
   `python scripts/check_setup.py --steps 5 --cell PV --tune seg_tuned`

2) Run one trial:
   `python run_pipeline.py --tune-dir cells/PV/tunes/seg_tuned --n-trials 1`

3) Find outputs:
   `{tune_dir}/output_data/<output_stem>/run_manifest.json`

4) Open `6_analysis.ipynb` and point it at the run folder.

If modfiles are missing:
- `cd cells/PV/tunes/seg_tuned/modfiles && nrnivmodl`
