# Roadmap

This roadmap tracks deferred work after the v0.3.0 portable-pipeline and
project-workspace checkpoint. It is intentionally short; detailed
implementation contracts belong in `contracts/` or issue trackers.

## Near Term

- Publish the reviewed v0.3.0 checkpoint, then complete the manual
  `HunterBushnell/SCP` → `cyneuro/SCP` transfer and check access, settings, and
  apps. Live canonical-link and Colab verification is post-transfer work.
  Version metadata and release notes do not create a tag or GitHub release.
  See the [transfer checklist](repository_transfer.md).
- Retain setup-feedback regression coverage and extend frontend checks as
  needed. The user confirmed discovery, save, and copy feedback working after
  a VS Code window reload. Extension-host logs had identified an
  Output-widget acknowledgement failure
  (`onKernelSocketMessage`, undefined `promise`) after discovery/copy actions.
  Setup and notebook-defaults text logs now use stock HTML widgets instead of
  OutputModel. A disposable Jupyter kernel verifies real frontend-style comm
  events for discovery, template selection, panel expansion, save, and repeated
  copy preview/create, with no OutputModel updates from those actions. This is
  not a broad browser/VS Code end-to-end test. Preserve already-created tunes,
  reload the VS Code window if its extension state is stuck, and retain
  NEURON's construction/restart safeguards. Rich protocol plots still use
  Output widgets.
- Resolve ACT's pinned-source custom-builder copy bug before claiming a complete
  optimization walkthrough. At revision `2b3fdc190107b8242e3496329cfdf240abd09694`,
  `ACTModule._simulate_cells` calls
  `specific_cell.set_custom_cell_builder(cell._custom_cell_builder)`, nesting
  the stored `(builder, kwargs)` tuple; construction then raises
  `TypeError: 'tuple' object is not callable`. This reproduces from both the
  pristine checkout and its installed wheel. Package imports, passive helpers,
  API checks, and fresh-process SCP cell preparation pass. Keep any ACT fix and
  full optimization validation separate from the install-once dependency update.
- Develop the staged ACT Spiker/BursterCa/LAA and L5/SST families into complete
  SCP characterization and manual-tuning walkthroughs, with justified targets,
  protocols, and curated results. Keep legacy PV defaults until these are ready.
- Keep research project workspaces separate from reusable public examples;
  publication and the eventual legacy-example transition are later decisions.
- Resume and review the paused synaptic-recording implementation before final
  release cleanup. Core work and tests exist, but it is not yet release-approved;
  see [current state and remaining work](synaptic_recording_update.md).
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
- Add further loader adapters only when existing Allen/HOC/Python-factory or
  live-cell adoption workflows cannot represent a concrete new use case.
