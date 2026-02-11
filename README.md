# Single Cell Pipeline (SCP)

This folder is the working version of the Single Cell Pipeline (SCP) for PV/SST models.
It is being prepared for extraction into its own repo.

Goals
- Provide a repeatable, notebook-first pipeline for single-cell sims and analysis.
- Keep configs and outputs consistent across notebook and CLI/SLURM runs.
- Keep Step 5 (local pipeline) stable while migrating legacy Steps 0-4.

Quickstart (existing tune)
- Notebook route: open `5_local.ipynb` and run all.
- CLI route (from repo root):
  `python run_pipeline.py --tune-dir cells/PV/tunes/seg_tuned --n-trials 1`

Pipeline map
- 0_download.ipynb: Step-0 bootstrap (download + compile + scaffold + validate)
- scripts/step0_prepare.py: CLI for Step-0 bootstrap
- 1_segment.ipynb: Segment cell (stub/min info for now)
- 2_passive.ipynb: Passive tuning (stub/min info for now)
- 3_active.ipynb: Active tuning (stub/min info for now)
- 4_synapses.ipynb: Synaptic tuning (stub/min info for now)
- 5_local.ipynb: Local pipeline (stable)
- 5_colab.ipynb: Colab/Linux pipeline (bootstrapped)
- 5_old_PV.ipynb, 5_old_SST.ipynb: Legacy notebooks
- 6_analysis.ipynb: End-of-pipeline analysis and comparisons

Step 5 sub-steps (for docs)
- 5.2.1: Load cell
- 5.2.2: Define geometry
- 5.2.3: Generate inputs
- 5.2.4: Preview/attach synapses
- 5.3: Run sims and save outputs
- 5.4: Analyze results

Config layout
- `cell_configs/cell_config.json`: cell identity (cell_name, tune, color, specimen_id, model_type, tuning)
- `cell_configs/sim_config.json`: sim-level config (timing, trials, outputs, randomness)
- `cell_configs/syn_config.json`: synapse group config list (includes `cell_configs/syn_groups/`)

Docs index
- `docs/README.md`
- `docs/quickstart.md`
- `docs/installation.md`
- `docs/pipeline_overview.md`
- `docs/step_0_4_stub.md`
- `docs/step_5_local_pipeline.md`
- `docs/step_6_analysis.md`
- `docs/configs_reference.md`
- `docs/cli_slurm.md`
- `docs/outputs_layout.md`
- `docs/troubleshooting.md`
- `docs/reproducibility.md`
- `docs/glossary.md`
- `docs/naming_conventions.md`
- `docs/roadmap.md`
- `docs/example_run.md`
- `docs/config_cookbook.md`

Contracts
- `contracts/README.md` indexes contract files and maps them to Step 5.2.3.
- Contracts are descriptive; only inputs and outputs should be treated as authoritative.

Status notes
- Step 0 now uses `modules_local` and writes/validates `cell_configs/` scaffolds.
- Steps 1-4 are still legacy and will be migrated next.
- `5_colab.ipynb` bootstraps a clean environment and can compile modfiles as needed.
