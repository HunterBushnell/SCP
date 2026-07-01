Naming Conventions (Suggested)

These are lightweight guidelines to keep SCP consistent. They are optional and
can be adjusted later.

Notebooks
- Use numeric prefixes for pipeline order: `0_`, `1_`, ..., `6_`.
- Keep the stable pipeline as `5_simulate.ipynb`.
- Keep optional analysis and utility notebooks as `6_analysis.ipynb` and `7_tools.ipynb`.

Scripts
- Entry points can keep short names:
  - `run_pipeline.py`
  - `run_slurm.sh`
- Optional wrappers (if you want Step 5 naming consistency):
  - `5_run_pipeline.py` -> calls `run_pipeline.py`
  - `5_run_slurm.sh` -> calls `run_slurm.sh`

Configs
- Keep all tune configs under `cell_configs/`.
- Keep ADB download/legacy artifacts outside of `cell_configs/`.

Outputs
- Default output: `output_data/<output_stem>/`.
- For batch runs: `output_data/<batch_stem>/results/`.
