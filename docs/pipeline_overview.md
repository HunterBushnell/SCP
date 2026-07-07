Pipeline Overview

SCP is a notebook-first pipeline for single-cell simulations.
Step 1 uses `modules`; Steps 2-4 run with updated notebook helpers and
external ACT/bmtool dependencies; Step 5 is the stable run pipeline; Step 6 is analysis.

Execution model
- Local notebooks (`2_passive.ipynb`, `3_active.ipynb`, `4_synapses.ipynb`, `5_simulate.ipynb`)
  assume the tune bundle is already prepared and mechanisms are already compiled
  (typically by Step 1). They validate and load existing files; they do not
  download/compile mechanisms.
- Colab notebooks (`colab_notebooks/2_colab.ipynb`, `colab_notebooks/3_colab.ipynb`) are
  bootstrapped and can run standalone (auto-clone/install/compile as needed).
- `5_simulate.ipynb` is shared by local and Colab Step 5 workflows.

Before running any step on a new machine, complete `installation.md` and run:
- `python scripts/check_setup.py --steps 1 2 3 4 5 --cell PV --tune seg_tuned`

Steps (notebooks)
- 1_download.ipynb: Bootstrap tune directory (download/compile/scaffold/validate).
- scripts/step1_prepare.py: CLI equivalent of Step 1.
- archive/1_segment.ipynb: Optional ACT-derived segmentation reference.
- 2_passive.ipynb: Passive parameter tuning.
- colab_notebooks/2_colab.ipynb: Colab-friendly Step 2 (auto-bootstrap + passive tuning).
- 3_active.ipynb: Active parameter tuning.
- colab_notebooks/3_colab.ipynb: Colab-friendly Step 3 (auto-bootstrap + active tuning).
- 4_synapses.ipynb: Synaptic tuning.
- 5_simulate.ipynb: Stable simulation pipeline (inputs -> synapses -> simulation -> outputs; local + Colab).
- archive/5_colab.ipynb: Archived Step 5 Colab notebook; use `5_simulate.ipynb`.
- 6_analysis.ipynb: End-of-pipeline analysis and comparisons.

Step 5 sub-steps (documentation labels)
- 5.2.1: Load cell
- 5.2.2: Define geometry
- 5.2.3: Generate inputs
- 5.2.4: Preview/attach synapses
- 5.3: Run sims and save outputs
- 5.4: Analyze results

Entry points
- Notebook: `5_simulate.ipynb`
- CLI: `run_pipeline.py`
- SLURM: `run_slurm.sh`

Data flow (high level)
- Inputs: `cell_configs/` + `manifest.json` + modfiles
- Outputs: `output_data/<run>/` with `run_manifest.json` and sidecars

See also
- `step_5_simulate.md` for simulation details.
- `step_6_analysis.md` for analysis usage.
