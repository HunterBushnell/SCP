# Single Cell Pipeline (SCP)

This folder is the working version of the Single Cell Pipeline (SCP) for PV/SST models.
It is being prepared for extraction into its own repo.

Goals
- Provide a repeatable, notebook-first pipeline for single-cell sims and analysis.
- Keep configs and outputs consistent across notebook and CLI/SLURM runs.
- Keep Steps 0-6 portable across local and Colab workflows.

Setup (new machine)
1. Clone SCP and enter the repo:
   `git clone <SCP_REPO_URL> && cd SCP`
2. Create the recommended Conda environment:
   `conda env create -f environment.yml`
   `conda activate scp-py311`
3. Register the notebook kernel (optional but recommended):
   `python -m ipykernel install --user --name scp-py311 --display-name "Python (SCP)"`
4. Clone external repos used by steps 1-4:
   `mkdir -p ../mods`
   `git clone https://github.com/V-Marco/ACT.git ../mods/ACT`
   `git clone https://github.com/cyneuro/bmtool.git ../mods/bmtool`
5. Verify setup end-to-end:
   `python scripts/check_setup.py --steps 0 1 2 3 4 5 --cell PV --tune seg_tuned`
6. Lint notebooks for portability/config issues:
   `python scripts/check_notebooks.py`

Notes
- You can override paths with env vars:
  `SCP_ROOT`, `SCP_ACT_PATH`, `SCP_BMTOOL_PATH`.
- If you prefer `venv` + pip instead of Conda, see `docs/installation.md`.

Quickstart (existing tune)
- Notebook route: open `5_local.ipynb` and run all.
- CLI route (from repo root):
  `python run_pipeline.py --tune-dir cells/PV/tunes/seg_tuned --n-trials 1`

Pipeline map
- 0_download.ipynb: Step-0 bootstrap (download + compile + scaffold + validate)
- scripts/step0_prepare.py: CLI for Step-0 bootstrap
- 1_segment.ipynb: Segment cell and build geometry outputs
- 2_passive.ipynb: Passive parameter tuning workflow
- 2_colab.ipynb: Passive tuning (Colab classroom version, bootstrapped)
- 3_active.ipynb: Active parameter tuning workflow
- 3_colab.ipynb: Active tuning (Colab classroom version, bootstrapped)
- 4_synapses.ipynb: Synaptic tuning workflow (including bmtool path)
- 5_local.ipynb: Local pipeline (stable)
- 5_colab.ipynb: Colab/Linux full pipeline (classroom + bootstrapped)
- 5_old_PV.ipynb, 5_old_SST.ipynb: Legacy notebooks
- 6_analysis.ipynb: End-of-pipeline analysis and comparisons

Colab classroom usage (notebook-only)
- `2_colab.ipynb`, `3_colab.ipynb`, and `5_colab.ipynb` are designed for first-time users.
- They can auto-clone SCP and required external repos (ACT) when run in a fresh Colab.
- For private repos, set one of: `SCP_GIT_TOKEN`, `SCP_GITHUB_TOKEN`, or `GITHUB_TOKEN`.
- Optional repo controls:
  - `SCP_REPO_URL`, `SCP_REPO_BRANCH`, `SCP_REPO_DIR`
  - `SCP_ACT_REPO_URL`, `SCP_ACT_REPO_BRANCH`, `SCP_ACT_DIR`
- Recommended class flow: run top-to-bottom once with defaults, then vary one parameter block at a time.

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
- `docs/step_0_4_stub.md` (Step 0-4 reference and prerequisites)
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
- Steps 1-4 run with repo-relative paths and use current notebook helpers.
- `5_colab.ipynb` bootstraps a clean environment and can compile modfiles as needed.
