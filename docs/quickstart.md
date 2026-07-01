# Quickstart

This is the shortest path to run an existing tuned cell.

## Prerequisites
- Complete `docs/installation.md`.
- Run `python scripts/check_setup.py --steps 5 --cell PV --tune seg_tuned`.
- Optional safety check: `python scripts/check_notebooks.py`.

## Option A: Notebook
1. Open `5_simulate.ipynb`.
2. Set the cell/tune at the top.
3. Run all cells.
4. Open `6_analysis.ipynb` for analysis/comparisons.

## Option B: CLI
- Single run:
  `python run_pipeline.py --tune-dir cells/PV/tunes/seg_tuned --n-trials 1`
- If modfiles are missing, build them once per tune:
  `cd cells/PV/tunes/seg_tuned/modfiles && nrnivmodl`

## Outputs
- `{tune_dir}/output_data/<output_stem>/`

## See Also
- `cli_slurm.md` for batch runs.
- `outputs_layout.md` for output file structure.
