Pipeline Overview

SCP is a notebook-first pipeline for single-cell simulations. Steps 0-4 are legacy
and will be updated; Step 5 is the stable pipeline; Step 6 is analysis.

Steps (notebooks)
- 0_download.ipynb: Download AllenDB bundle into `cells/` (stub/min info).
- 1_segment.ipynb: Segment and prepare morphology (stub/min info).
- 2_passive.ipynb: Passive parameter tuning (stub/min info).
- 3_active.ipynb: Active parameter tuning (stub/min info).
- 4_synapses.ipynb: Synaptic tuning (stub/min info).
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
