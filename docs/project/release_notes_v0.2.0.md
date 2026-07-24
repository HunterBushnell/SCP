# SCP v0.2.0 Release Notes

SCP v0.2.0 expands the initial Allen-focused public preview into a
model-neutral, notebook-first single-cell workflow. The tuned PV cell remains
the primary example and public default.

## Highlights

- Registered model-loader interface with `allen_manifest` compatibility and the
  first object-owned `hoc_template` adapter.
- Cell-scoped canonical soma, dendrite, apical, axon, and all-section
  collections across setup, tuning, simulation, analysis, and artifact tools.
- Explicit runtime voltage and temperature conditions for HOC-template models,
  reapplied for current-injection protocols.
- Loader-aware model artifact discovery, SHA-256 provenance, snapshots, and
  dry-run-first restoration.
- Recommended `0_pipeline.ipynb` front door with five independent,
  button-driven panels. Run All renders the interface without loading or
  simulating a model.
- One shared in-kernel cell for Steps 2–4, with fresh-process input preview and
  final simulation in Step 5.
- Config-backed advanced controls, synchronized Python/widget settings, concise
  output with retained logs, and saved-run diagnostics handoff.
- Auto-first Step 1 model setup: existing loader configuration is reused, while
  unambiguous staged Allen manifests and HOC templates can be discovered.
- Sparse, mode-specific target templates for manual values, user traces, Allen
  NWB data, or explicitly targetless characterization.
- ACT and BMTool remain optional. Their repositories are resolved only when an
  ACT proposal/optimization or BMTool initialization is requested.
- Public EUSmn, HYPO, and PGN HOC-template bundles with `orig` and manually
  tuned variants, tune-local configs, source lineage, and artifact hashes.

## Examples and Compatibility

- `cells/PV/tunes/tuned` remains the primary worked example and default in
  public notebooks.
- PN, PV, and SST retain their Allen-manifest layouts and historical loader
  aliases.
- EUSmn, HYPO, and PGN exercise the same core Steps 1–3, IClamp simulation, and
  loader-aware analysis paths without an Allen manifest.
- Missing targets and synapse configuration are supported for cell-only
  current-injection workflows.
- Existing tune-local configuration remains authoritative on fill-mode reruns;
  cell-specific display, geometry-compatibility, and runtime values can be
  edited directly in JSON.

## Scope and Limitations

- The HOC bundles are reproducible manual-tuning examples, not independent
  biological-validation claims.
- ACT active tuning remains experimental, review-only, and not
  release-blocking. Non-Allen ACT execution is explicitly experimental.
- BMTool remains an optional external integration, and the new HOC examples do
  not establish scientific synapse defaults.
- Python-factory loading, parameter overlays, new electrophysiology metrics,
  and automated application of tuning proposals remain deferred.

## Licensing and Provenance

SCP-authored source code and documentation remain under the root MIT license.
Bundled model and mechanism assets are not necessarily MIT-licensed. See
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) and each HOC tune's
`SOURCE_PROVENANCE.json` for the applicable notices, source lineage, and hashes.
