# Roadmap

This roadmap tracks deferred work after the v0.2.0 model-neutral pipeline and
public HOC-example release. It is intentionally short; detailed implementation
contracts belong in `contracts/` or issue trackers.

## Near Term

- Add the planned synapse-recording machinery exposed by `sim_config.json`.
- Extend the current IClamp/analysis smoke coverage to a small synapse-driven
  Step 5 run when a public mechanism fixture is available.
- Keep local and Colab notebook entry points aligned as docs and examples change.
- Maintain provenance and representative smoke coverage for the Allen-manifest
  and public HOC-template example families.

## Later

- Improve ACT active tuning support for direct voltage-trace/NWB workflows once
  ACT trace sampling/window contracts are clarified.
- Add optional target trace overlays in Step 3 after the active-trace contract
  can reliably select, align, and label matching target sweeps.
- Add parametric-study machinery as a separate workflow enhancement.
- Add optional tuning proposal/apply tooling for Steps 2-3. Current notebooks
  should stay manual-first, but future tooling could save computed ACT values,
  review target model fields, and safely apply loader-specific edits such as
  ADB `*_fit.json` passive-parameter updates.
- Add future loader adapters, such as the deferred Python-factory loader, after
  the Allen-manifest and HOC-template contracts have matured.
