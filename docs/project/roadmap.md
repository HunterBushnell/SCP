# Roadmap

This roadmap tracks deferred work after the current notebook-first public
pipeline refactor. It is intentionally short; detailed implementation contracts
belong in `contracts/` or issue trackers.

## Near Term

- Decide whether bundled PV/SST saved outputs or fresh smaller public example
  outputs should ship with the public repo.
- Add the planned synapse-recording machinery exposed by `sim_config.json`.
- Add end-to-end smoke tests for at least one small Step 5 run and one Step 6
  analysis load.
- Keep local and Colab notebook entry points aligned as docs and examples change.

## Later

- Add smaller/faster public example runs that avoid private project data if the
  current saved outputs are not approved for release.
- Improve ACT active tuning support for direct voltage-trace/NWB workflows once
  ACT trace sampling/window contracts are clarified.
- Add optional target trace overlays in Step 3 after the active-trace contract
  can reliably select, align, and label matching target sweeps.
- Add parametric-study machinery after the repo-wide refactor is complete.
- Add optional tuning proposal/apply tooling for Steps 2-3. Current notebooks
  should stay manual-first, but future tooling could save computed ACT values,
  review target model fields, and safely apply loader-specific edits such as
  ADB `*_fit.json` passive-parameter updates.
- Add generic/non-ADB model adapters beyond the current ADB-focused examples.
- Add lightweight tests around config validation and example smoke runs.
