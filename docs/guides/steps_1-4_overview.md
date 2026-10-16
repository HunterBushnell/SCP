# Steps 1-4 Overview

This file is retained as a compact bridge for the early pipeline steps. Use the
individual step guides for full instructions.

## Current Docs

- `step_1_setup.md`: setup, ADB download, mechanism compilation, config scaffolding, and validation.
- `step_2_passive.md`: passive traces/checks and optional ACT-based parameter proposals.
- `step_3_active.md`: active tuning, optional ACT workspace/CLI integration, manual sweeps, trace plots, and FI checks.
- `step_4_synapses.md`: synapse tuning with BMTool and SCP synapse-config export.

## External Dependencies

- Step 2 runs passive traces directly; ACT is optional for target-based proposals.
- Step 3 can optionally use ACT active tuning through a tune-local workspace,
  but its manual active sweep/FI checks can run without ACT.
- Step 4 uses BMTool for chemical synapse tuning through a small SCP adapter.

Install these once in the Python environment using SCP's suggested
`.[act,bmtool]` extras or the separate `requirements-act.txt` /
`requirements-bmtool.txt` files. ACT uses a pinned source revision. No fixed
checkout location is required; SCP prefers environment imports, while explicit
`SCP_ACT_PATH` / `SCP_BMTOOL_PATH` overrides and existing legacy checkouts remain
supported. Cloning requires opt-in; normal actions do not install packages.
See [installation](../installation.md) for constraints and Colab instructions.

## Expected Step 1 Contract

Later steps assume the selected tune has:

- loader-owned native model source(s),
- `cell_configs/cell_config.json`,
- `cell_configs/sim_config.json`,
- `cell_configs/geometry.json`,
- an optional directory selected by `paths.modfiles`, with compiled mechanisms
  when custom `.mod` sources exist,
- optional `cell_configs/target_config.json`,
- optional `cell_configs/syn_config.json` and `cell_configs/syn_groups/*.json`.
