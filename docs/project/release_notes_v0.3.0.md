# SCP v0.3.0 Release Notes

SCP v0.3.0 records the completed dependency, portable-notebook, and adaptable
workspace work since [v0.2.0](release_notes_v0.2.0.md). These notes describe a
**prepared checkpoint**, not an assertion that a tag, GitHub release, or
repository transfer has already been published. The
[transfer checklist](repository_transfer.md) tracks the separate manual steps.

## Highlights

- Editable installation as `scp-pipeline`, with the supported `scp_pipeline`
  namespace and `scp-pipeline setup|run|check` / `python -m scp_pipeline`
  entry points. A host project's own `modules` package no longer conflicts with
  installed SCP; checkout-local compatibility imports remain available.
- Portable `0_pipeline.ipynb`: copy the notebook into a project while SCP's
  installed code remains elsewhere. Models, configs, snapshots, managed builds,
  and results can stay project-owned. The seven detailed notebooks remain
  SCP-hosted and can point to those external workspaces.
- Compact setup separates navigation, model registration, optional target or
  disabled synapse templates, working-copy creation, and explicit loading.
  Preview is read-only; registration does not construct a cell. Run All renders
  controls without loading a configured cell or simulating.
- Explicitly saved navigation in `.scp_project.json` and notebook-wide panel
  preferences in `.scp_pipeline_settings.json`, separate from native sources and
  scientific configs. Returning users can reopen a workspace without editing
  the notebook's settings dictionary.
- Project-aware `projects/<project>/cells/<cell>/tunes/<tune>` workspaces,
  alongside the original `cells/` layout and external project-owned workspaces.
  Saved identity, routing, and append/merge guards distinguish same-named cells
  across projects without preventing cross-project plotting.
- Common cell adoption through `wrap_cell`, explicit live-cell overrides, and
  the `python_factory` adapter for persisted functions/classes. Canonical
  cell-scoped sections retain ownership without capturing unrelated global
  sections or changing native geometry and parameters.
- External-source registration with byte-preserving source snapshots,
  explicit copy/reference selection, declared dependencies, hash provenance,
  tune-owned mechanism builds, and dry-run-first local restoration. SCP-managed
  operations do not build in or edit upstream sources.
- Multi-template HOC discovery without execution, explicit runtime conditions,
  and current recording compatible with `extracellular.i_membrane`. Current
  recording can be disabled before recorder construction.
- Install-once optional ACT/BMTool integrations prefer installed packages;
  per-project tool checkouts are not required. Public ACT Spiker, BursterCa,
  LAA, L5, and SST source variants are staged and smoke-tested without needing
  ACT's optimizer to exercise their native models.
- Canonical `cyneuro/SCP` references, University of Missouri Neural Engineering
  Laboratory attribution, creator/contact details, and guarded old/new origin
  compatibility in the existing detailed-notebook Colab bootstrap helpers.

## Compatibility and Examples

PV remains the primary worked example; detailed notebook and CLI defaults are
unchanged. Compact navigation restores saved choices or asks for a selection.
Legacy Allen/HOC configurations, result archives, repository scripts, and the
existing environment/dependency versions remain supported. Public PV/SST
synaptic defaults use their saved baseline settings while retaining their
intrinsic tunes; required data lives with the relevant cells/projects.

After updating an editable checkout, run
`python -m pip install -e "/path/to/SCP"` in the existing environment to refresh
installed v0.3.0 package metadata, then restart kernels/workers. No new
environment or dependency-version overhaul is required; see the
[dependency workflow](../guides/dependency_workflow.md).

ACT examples are **staged native-source examples**, not complete SCP scientific
walkthroughs or independently biologically validated releases. HOC `tuned`
starts as an untuned working copy; `reference` is ACT's known target model.
Allen `tuned` contains the imported matching ACT variant. See the
[example catalog](../guides/act_examples.md) for those distinctions. Research
workspaces remain separate from reusable public examples.

## Boundaries and Remaining Work

- A live cell is session-only unless its construction is moved into a saved
  builder. Saved geometry/mechanism summaries are not full NEURON checkpoints.
- Copy mode runs the snapshot; reference mode runs external originals. Neither
  registration nor working-copy creation silently refreshes a saved snapshot.
  Restart after loaded HOC/Python/MOD changes; separate folders do not isolate
  process-global NEURON definitions. Trusted user model code is not sandboxed.
- ACT active optimization remains experimental and review-only. The pinned
  ACT custom-builder copy defect is unresolved; working imports/passive helpers
  do not establish a successful end-to-end optimization walkthrough. See the
  [roadmap](roadmap.md).
- Detailed synaptic recording remains paused and is not release-approved; see
  its [implementation status](synaptic_recording_update.md). This checkpoint
  does not claim that work is complete.
- Setup feedback has synthetic and real-kernel regression coverage, plus the
  reported VS Code reload confirmation—not broad browser/VS Code end-to-end
  coverage. Scientific validation remains specific to each model and protocol.
- No parameter overlays, new electrophysiology metrics, scientific retuning,
  automatic release workflow, or migration of user workspaces is introduced.

SCP-authored code and documentation retain the existing MIT licensing. Native
model assets retain their own notices and source credits; see
[third-party and model-asset notices](../../THIRD_PARTY_NOTICES.md).
