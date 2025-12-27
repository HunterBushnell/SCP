Single Cell Pipeline (SCP)

This folder is the working version of the Single Cell Pipeline (SCP) for PV/SST models.
It is being prepared for extraction into its own repo.

Goals
- Provide a repeatable, notebook-first pipeline for single-cell sims and analysis.
- Keep configs and outputs consistent across notebook and CLI/SLURM runs.
- Make Step 5 (local pipeline) stable and reusable while Steps 0-4 are updated later.

Quickstart (existing tune)
- Notebook route: open `5_local.ipynb` and run all.
- CLI route:
  `python /home/hrbncv/SCP/run_pipeline.py --tune-dir /home/hrbncv/SCP/cells/PV/tunes/seg_tuned --n-trials 1`

Pipeline map
- 0_download.ipynb: Download cell (stub/min info for now)
- 1_segment.ipynb: Segment cell (stub/min info for now)
- 2_passive.ipynb: Passive tuning (stub/min info for now)
- 3_active.ipynb: Active tuning (stub/min info for now)
- 4_synapses.ipynb: Synaptic tuning (stub/min info for now)
- 5_local.ipynb: Local pipeline (stable)
- 5_colab.ipynb: Colab-friendly pipeline (planned)
- 5_old_PV.ipynb, 5_old_SST.ipynb: Legacy notebooks
- 6_analysis.ipynb: End-of-pipeline analysis and comparisons

Step 5 sub-steps (for docs)
- 5.1: Load cell
- 5.2: Define geometry
- 5.2.3: Generate inputs (formerly 2.3)
- 5.2.4: Build/attach synapses (formerly 2.4)
- 5.3: Run sims and save outputs

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
- Steps 0-4 will be updated after the PV/SST paper to align with the new config layout.
- `5_colab.ipynb` will be expanded to bootstrap a clean environment and download/compile cells.
