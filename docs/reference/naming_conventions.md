# Naming Conventions

These are lightweight conventions for the public SCP repo.

## Notebooks

Use numeric prefixes for the compact entry point and detailed pipeline order:

- `0_pipeline.ipynb`: compact start-to-finish entry point
- `1_setup.ipynb`
- `2_passive.ipynb`
- `3_active.ipynb`
- `4_synapses.ipynb`
- `5_simulate.ipynb`
- `6_analysis.ipynb`
- `7_tools.ipynb`

`0_pipeline.ipynb` is the simplest end-to-end entry point, while
`5_simulate.ipynb` is the detailed simulation entry point. Optional notebooks
live under `extra_notebooks/` and should not be required by the numbered
workflow.

## Tune Directories

Recommended shape:

```text
cells/<CELL>/tunes/<TUNE>/
```

Examples:

- `cells/PV/tunes/orig`
- `cells/PV/tunes/tuned`
- `cells/SST/tunes/orig`
- `cells/SST/tunes/tuned`

## Configs

Keep tune configs under:

```text
cell_configs/
```

Use stable config filenames:

- `cell_config.json`
- `sim_config.json`
- `geometry.json`
- `syn_config.json`
- `syn_groups/<group>.json`

## Outputs and Workspaces

Default Step 5 output:

```text
output_data/<output_stem>/
```

Notebook scratch exports:

```text
notebook_exports/<step_or_task>/
```

ACT active-tuning workspace:

```text
act_workspace/
```

SLURM batch output:

```text
output_data/<batch_stem>/results/
output_data/<batch_stem>/parts/
output_data/<batch_stem>/logs/
```

## Scripts

Keep public entry points short and stable:

- `scripts/step1_prepare.py`
- `scripts/run_act_active.py`
- `run_pipeline.py`
- `run_slurm.sh`
