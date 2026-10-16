# Synaptic Recording — Paused Implementation

Status: paused at the user's request on 2026-10-14. Prioritize completing the
public baseline/data organization and notebook cleanup first. Do not treat
the recording work as a finished or release-approved feature.

## What already happened

Implementation started before priorities changed. The work remains uncommitted
in the shared working tree; it was not discarded or made into a separate release.
Synaptic capture is opt-in and remains disabled in existing/public defaults.

- Core configuration normalization, bounded native recorders, and single/multi
  trial plumbing have been implemented.
- Save/load sidecars, pickle/NPZ, append and array/two-run merge plumbing have
  been implemented, with focused regression coverage.
- Measured-trace analysis/plot helpers and initial Step 6 tables have been added.
- The Step 4 manual probe now explicitly identifies its measured synapse rather
  than depending on an implicit first-synapse current recorder.
- Native synthetic tests and fresh-process public PV/SST baseline smoke tests
  have been added. Numerical smoke checks are not biological validation.

These are a useful resume point, not a claim that design/release review is done.
The [draft usage guide](../guides/synaptic_recording.md) documents the current
implementation and must be reconciled with final behavior before release.

## Intended contract

- Keep placement/input metadata in `syn_records*`; measured data belongs in
  `syn_recordings` or `syn_recordings_by_trial`.
- Each measured payload has its own time axis, native units, group coverage,
  sampling seed, selected synapse identities, and memory estimate.
- `group` means exact group aggregates; `sample` means deterministic individual
  traces; `both` combines them. Never label sampled sums as population totals.
- Sum currents/conductances; average dimensionless utilization/probability
  states. Do not multiply native currents by weights a second time.
- Missing variables have explicit coverage; incomplete group aggregates are
  omitted. Invalid exposed pointers or recorder failures remain errors.
- Trial count, sample count, recording interval, and a conservative memory
  budget control allocations independently of voltage retention. Snapshot mode
  must not silently enable capture or lift the budget.
- Preserve measured traces through save/load and merging, with correct trial
  indices, time-axis checks, and destination retention limits.
- Legacy I/G arrays have unknown aggregation; do not retroactively call them
  verified whole-cell totals or rewrite historical results.

## Resume checklist

1. Review core behavior and configuration choices with the user before declaring
   the interface final; especially group/sample defaults and memory controls.
2. Fix the known merge metadata issue: `meta.n_syn_recording_trials` can retain
   the base-run count after appending/array merging or truncation. Recompute it
   from retained payloads and add count/cap regression tests.
3. Audit conversion, append and both merge paths for legacy compatibility,
   independent capture caps, retained time axes, and missing/partial variables.
4. Complete Step 4 probe integration verification and review plotting/UI scope,
   units, sampled identities, and interpretation of partial coverage.
5. Review the fixed-step `run_cell` timestep correction added during native
   testing (`steps_per_ms` synchronized with `dt`); ensure regression coverage
   and decide whether to retain it as a separately documented generic fix.
6. Re-run native pointer/memory tests, generated-input synapse tests, save/load
   round trips, append/merge tests, fresh PV/SST regressions, and the full suite.
   Fixed-interval recordings may end one sampling interval before `tstop`; their
   own time axis is authoritative and must not be forced to match Vm endpoints.
7. Finish documentation review. Only then remove paused/WIP notices and mark
   recording support complete on the roadmap.
8. Isolate tests from repository workspaces: a UI/setup test fills missing
   defaults and can regenerate a disabled PV input template or expand its
   `sim_config.json`. Tests must use temporary copies, not mutate public assets.

## Main implementation locations

- `modules/input_generation/recording_config.py`
- `modules/simulation/synaptic_recording.py`, `cell_runtime.py`,
  `single_run.py`, `multi_run.py`
- `modules/simulation/result_saving.py`, `result_loading.py`,
  `result_appending.py`; `scripts/merge_array_results.py`, `merge_two_runs.py`
- `modules/analysis/synaptic_recordings.py`, `analysis.py`, `ui/extra.py`
- `modules/tuning/manual_synapse_probe.py`
- `tests/test_synaptic_recording*.py`,
  `tests/test_allen_synaptic_subprocess.py`,
  `tests/fixtures/synaptic_recording/`

## Validation record at pause

Focused recorder, persistence, analysis, baseline, and preflight tests were run.
The initial full-suite log is retained locally under
`.workspace_migration.local/baseline_recording_20261014/full_suite.log`; it ran
394 tests and exposed three test-expectation failures: two stale endpoint
assertions for downsampled traces and the old PV enabled-group selection.
Those expectations have been corrected; the final pause-check results are
recorded separately in that local directory. Do not present the initial log as
a passing suite or confuse passing tests with completion of the review above.

The pause-check rerun ran 394 tests with four failures: a baseline-test helper
incorrectly required a disabled CSV template, its SST assertion confused the
PN stimulus window with the disabled VIP window, and two release-default tests
found the user's current A1/ET selection in the compact notebook. The first
two test assertions were corrected without changing scientific settings.
The user's A1/ET selection is intentionally preserved; reset it to bundled
PV/tuned only when preparing the public notebook snapshot. Kernel/output
normalization is independent of model selection and passes its own check.

Final targeted pause checks passed: 35 baseline/data/preflight/notebook-cleanliness/
UI/documentation tests, plus both fresh-process PV/SST file-driven smoke tests.
The standalone notebook checker passes all 10 notebooks; Python compilation,
shell syntax, and whitespace checks also pass. The full suite was not rerun
again after those targeted corrections; the two release-default assertions
still conflict with the deliberately preserved A1/ET notebook selection.

A local source snapshot is saved at
`.workspace_migration.local/baseline_recording_20261014/synaptic_recording_wip.tar.gz`.
It contains whole working-tree files, including earlier unrelated changes in
shared modules: use it for recovery/review, not as a patch to restore blindly.

The pre-existing `SST5_base` archived morphology/hash discrepancy remains a
separate issue. It was not silently repaired as part of either baseline setup
or this recording work.
