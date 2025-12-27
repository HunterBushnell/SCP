Reproducibility

Seeds and randomness
- `randomness_mode` can be `fixed`, `derived`, or `random`.
- `random_seed` and the `randomness` block control trial and synapse variation.
- `--seed` on the CLI overrides both sim_cfg and randomness global seed.

Array alignment
- `--trial-offset` is used to align array jobs with sequential multi-trial runs.
- `run_slurm.sh` sets `trial_offset` when splitting `TOTAL_TRIALS`.

Snapshot mode
- `snapshot.enabled: true` captures full metadata for diffing runs.

Append mode
- `append` can attach new results to an existing run and reuse its sim_cfg.
