Pipeline Overview

SCP is a notebook-first pipeline for single-cell simulations.
Step 0 uses `modules_local`; Steps 1-4 run with updated notebook helpers and
external ACT/bmtool dependencies; Step 5 is the stable run pipeline; Step 6 is analysis.

Before running any step on a new machine, complete `installation.md` and run:
- `python scripts/check_setup.py --steps 0 1 2 3 4 5 --cell PV --tune seg_tuned`

Steps (notebooks)
- 0_download.ipynb: Bootstrap tune directory (download/compile/scaffold/validate).
- scripts/step0_prepare.py: CLI equivalent of Step 0.
- 1_segment.ipynb: Segment and prepare morphology.
- 2_passive.ipynb: Passive parameter tuning.
- 2_colab.ipynb: Colab-friendly Step 2 (auto-bootstrap + passive tuning).
- 3_active.ipynb: Active parameter tuning.
- 3_colab.ipynb: Colab-friendly Step 3 (auto-bootstrap + active tuning).
- 4_synapses.ipynb: Synaptic tuning.
- 5_local.ipynb: Stable local pipeline (inputs -> synapses -> simulation -> outputs).
- 5_colab.ipynb: Colab/Linux pipeline (bootstrapped).
- 6_analysis.ipynb: End-of-pipeline analysis and comparisons.

Step 5 sub-steps (documentation labels)
- 5.2.1: Load cell
- 5.2.2: Define geometry
- 5.2.3: Generate inputs
- 5.2.4: Preview/attach synapses
- 5.3: Run sims and save outputs
- 5.4: Analyze results

Entry points
- Notebook: `5_local.ipynb`
- CLI: `run_pipeline.py`
- SLURM: `run_slurm.sh`

Data flow (high level)
- Inputs: `cell_configs/` + `manifest.json` + modfiles
- Outputs: `output_data/<run>/` with `run_manifest.json` and sidecars

See also
- `step_5_local_pipeline.md` for pipeline details.
- `step_6_analysis.md` for analysis usage.
