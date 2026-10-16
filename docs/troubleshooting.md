# Troubleshooting

## Project-owned notebook setup

If a copied compact notebook cannot import `scp_pipeline`, select the SCP kernel
and install the checkout once with `python -m pip install -e /path/to/SCP` in that
environment. Do not add the generic `modules` package to a project's imports.
If paths point at the wrong directory, check Project dir: Jupyter's starting
directory need not be the notebook file's directory. Preview errors can be
corrected without restarting; errors after Prepare begins may require a fresh
kernel because NEURON definitions are process-global. See the
[dependency workflow](guides/dependency_workflow.md).

## First Check

Run the setup checker:

```bash
python scripts/check_setup.py --steps 1 2 3 4 5 --cell PV --tune tuned --compile-modfiles
```

Use `--check-act` or `--check-bmtool` only when you want the checker to require
the corresponding optional external integration.

Run notebook portability checks:

```bash
python scripts/check_notebooks.py
```

## Compact Notebook Action Fails

- **Buttons look clickable, but feedback and expanding panels stop updating:**
  VS Code Jupyter can fail in its Output-widget message handler while Python
  callbacks continue, including successful saves/copies. A diagnosed instance
  logged `Cannot read properties of undefined (reading 'promise')` in
  `onKernelSocketMessage` in **Output → Log (Extension Host)**. Setup and
  notebook-defaults text logs now use ordinary HTML widgets, avoiding that
  Output-specific handler. Rich protocol plots still need Jupyter outputs.
  After updating SCP, restart the kernel and Run All. If the frontend was
  already stuck, use **Developer: Reload Window** in VS Code first, then restart
  and Run All; a kernel restart alone may not reset extension state. Do not
  retry Save/Create blindly: check the existing Tune list or files first.
  The workaround has real-kernel comm regression coverage; confirmation in the
  affected live frontend remains part of the [roadmap](project/roadmap.md).
- Preview, registration, and working-copy errors do not lock model loading.
  Correct the input and retry; file/config checks run before the load lock in
  both workspace modes. Each action replaces only its own feedback.
- If a copy already exists after an error, select that existing tune using
  Cell/Tune or Tune path instead of creating it again. A copy can succeed even
  if the subsequent navigation/settings update fails.
- Step 1 prevents a second model-construction attempt after a failed load.
  Restart the kernel, correct the reported setup/config issue, and rerun from
  the top.
- If buttons look clickable but do nothing while the kernel is idle, try a new
  code cell with `print("kernel responds")`. If it runs, check the action status
  and whether the displayed controls belong to the current kernel. After a
  restart, Run All to render fresh controls; saved widget displays are not live
  connections. Restarting does not remove saved tunes or results. If even a
  plain code cell does not run, resolve the Jupyter/kernel connection first.
- If a source fingerprint warning names `cell_config.json`, morphology,
  fit/HOC, or MOD files, restart the kernel. For target, simulation, geometry,
  synapse, or ACT config edits, rerun the action that consumes the file instead.
- Quiet modes keep complete details in `pipeline_ui.step1_load_log`,
  `pipeline_ui.input_preview_log`, and `pipeline_ui.simulation_log`. Inspect the
  relevant string or disable its quiet checkbox before retrying.
- If **Plot Results** is disabled, complete **Run Simulation** successfully
  first; plotting always uses the latest loaded saved manifest.

## Missing Modfiles or NEURON Errors

- A model using only built-in NEURON mechanisms needs no `modfiles/` directory
  or compiled library. Confirm `cell_config.json` omits `paths.modfiles` or sets
  it to `null`.
- If the selected tune declares custom `.mod` sources, detailed Step 5 compiles
  a missing library automatically. To refresh an existing build, restart the
  kernel and set `RECOMPILE_MODFILES = True` in Step 1, 2, 3, 4, or detailed
  Step 5. You can also build the configured source directory manually:

```bash
cd <configured_modfiles_dir>
nrnivmodl
```

- Or run `scripts/check_setup.py --compile-modfiles`.
- In Colab, rerun the bootstrap cell before compiling/loading mechanisms.
- For ADB all-active bundles, use the Step 1 genome cleanup toggles when the
  fit JSON needs normalization before loading.

## Step 2 Passive Values Do Not Apply

Step 2 is manual-first and does not overwrite model files automatically.

- If Step 2 reports missing passive targets, provide all three passive values
  in `manual_passive_targets`, fill `target_config.json` fields
  `manual.passive.v_rest_mV`, `manual.passive.rin_MOhm`, and
  `manual.passive.tau_ms`, or set `target_source.mode` to `traces`/`allen_nwb`.
- Confirm the ACT values were copied into the correct `*_fit.json` or custom
  model source file.
- Save the edited file before rerunning the notebook.
- Rerun from **2.3 Build Cell** so the NEURON cell is rebuilt from the edited
  model files.
- Check `passive_area_mode` and `passive_area_scale` if ACT estimates look too
  large or too small.
- For passive checks, confirm active conductances are disabled or minimized
  according to your model's passive-tuning convention.

## Step 2 Allen/ADB NWB Passive Targets

For passive targets from a downloaded Allen/ADB ephys NWB file:

- Place the `*_ephys.nwb` file in the selected tune directory, set
  `target_config.json` field `allen_nwb.file`, or set
  `NWB_PASSIVE_PATH` explicitly.
- Start with `allen_nwb.passive.stimulus_names = ["Long Square"]`.
- The default current filter uses negative-current sweeps only:
  `allen_nwb.passive.max_current_pA = -1.0`.
- If no sweeps are found, inspect `allen_nwb.sweep_ids`, `allen_nwb.passive.sweep_ids`,
  `allen_nwb.passive.min_current_pA`, `allen_nwb.passive.max_current_pA`,
  and the stimulus names in the NWB file.
- The notebook writes `notebook_exports/step2_passive/allen_nwb_passive_targets.csv`
  and `notebook_exports/step2_passive/allen_nwb_passive_sweeps.csv` for review.

## Step 3 ACT Target Data Issues

For Allen/ADB NWB targets:

- Place the downloaded `*_ephys.nwb` file in the selected tune directory, set
  `target_config.json` field `allen_nwb.file`, or set `ACT_NWB_PATH`
  explicitly.
- Use `target_source.mode = "allen_nwb"` in `target_config.json` or
  `ACT_TARGET_MODE = "allen_nwb"` in the notebook.
- Start with `allen_nwb.active.stimulus_names = ["Long Square"]`.
- If no sweeps are found, inspect `allen_nwb.active.min_current_pA`,
  `allen_nwb.active.max_current_pA`, and the stimulus names in the NWB file.
- The preparation step writes `act_workspace/allen_nwb_fi_curve.csv` for review
  and `act_workspace/target_sf.csv` for ACT.

For custom FI CSV targets, use a current column such as `amp_pA`, `current_pA`,
`amp_nA`, or `mean_i`, and a frequency column such as `spike_frequency`,
`spike_frequency_hz`, or `frequency_hz`.

For `trace_npy` targets, `traces.active.file` must point to an ACT-compatible
`.npy` array with shape `(n_trials, n_timepoints, n_columns)`, voltage in mV in
column 0, and injected current in nA in column 1. See
`docs/reference/target_trace_formats.md`.

In `0_pipeline.ipynb`, click **Prepare ACT workspace** after changing
`target_config.json`, the ACT config, or any ACT option. Existing or partial
module output is intentionally protected; enable the explicit overwrite option
before rerunning that selection. If a run is cancelled, its process group is
terminated and its retained manifest is marked incomplete. Rerunning an earlier
module can make later predictions stale because the later module was trained
against the earlier proposal.

If compact preparation cannot import ACT, install the pinned source package
once with `python -m pip install -r requirements-act.txt` from SCP in the active
environment, then restart the kernel. Do not substitute an unqualified
`pip install act`. Check the underlying exception for missing Python dependencies
or an incompatible API. Correct/unset stale `ACT_ROOT` / `SCP_ACT_PATH` /
`SCP_ACT_DIR` / `ACT_PATH` overrides, which take precedence over the environment
package. Preparation does not run pip or clone ACT by default.
Non-PV/SST cells need a complete
`act_workspace/act_active_config.json` prepared and validated in
`3_active.ipynb`; registered non-Allen loaders are accepted experimentally only
after fresh-process cell construction succeeds.

If optimization reaches ACT but reports `TypeError: 'tuple' object is not
callable`, the pinned ACT revision has a known custom-builder copy bug inside
`ACTModule._simulate_cells`; reinstalling or moving the checkout does not fix
it. See the [installation limitation](installation.md#act-suggested-install-once-source-dependency)
and [roadmap](project/roadmap.md). Passive proposals and model-neutral SCP
protocols do not depend on that optimizer path.

## BMTool import or synapse API unavailable

BMTool is a suggested, install-once environment dependency for Step 4, not a
required sibling checkout. From SCP in the active environment, run
`python -m pip install -r requirements-bmtool.txt`, restart the kernel, and check
`scp-pipeline check --steps 4 --cell PV --tune tuned --check-bmtool`.
For an external workspace, pass its `--tune-dir` instead of the example names.

Check the underlying import exception for missing Python dependencies. An
installed but broken/incompatible BMTool does not trigger a replacement clone.
Correct or unset `SCP_BMTOOL_PATH` / `SCP_BMTOOL_DIR` / `BMTOOL_PATH` /
`BMTOOL_ROOT` if an old override selects another checkout. Do not switch already
imported copies in the same kernel. See [installation](installation.md).

## Missing Configs

Confirm the tune contains:

```text
cell_configs/cell_config.json
cell_configs/sim_config.json
cell_configs/geometry.json
```

For synapse-driven runs, also confirm:

```text
cell_configs/syn_config.json
cell_configs/syn_groups/*.json
```

Use Step 1 config scaffolding with `CONFIG_MODE = "fill"` or:

```bash
python scripts/step1_prepare.py \
  --tune-dir cells/PV/tunes/orig \
  --source-type existing \
  --no-download \
  --no-compile \
  --config-mode fill
```

## Synapse Group Not Used

- Check that the group file is listed in `cell_configs/syn_config.json`.
- Check the group has `"state": true`.
- Check each active block has `"state": true` or no `state` field.
- Check all `input_blocks` lie within `sim_config.tstart` and `sim_config.tstop`.

## Input Block Errors

Common causes:

- block `stop_ms` is not greater than `start_ms`,
- blocks overlap,
- block times fall outside the simulation window,
- a source-driven block crop duration does not match the block duration,
- a source file path is not valid from the repo root or tune context.

Gaps between input blocks are allowed and treated as quiescent.

## HOC Load Reports “Not a MECHANISM”

A HOC file can depend on custom ion channels implemented in separate `.mod`
files. Selecting the HOC/template alone does not supply those channels.
`paths.modfiles: null` means built-in mechanisms only; it is not automatic MOD
discovery. Include the required MOD folder in the registered source bundle and
set `paths.modfiles` relative to its native-source root. Compile/load those
mechanisms before constructing the cell.

For copy-mode sources, pointing at a new path is not enough: the MOD sources
must also be present in the saved snapshot. Review an explicit, backed-up source
refresh, preserving any manually edited native files; opening or copying a tune
does not refresh its snapshot automatically.

NEURON can leave a partially defined template after reporting a failed HOC
load. SCP rejects that failed source even if its template name exists, and does
not attempt construction. Fix the source/MOD setup, then **restart the kernel**
before retrying. Repeated loads in the same process cannot repair native state.
This guard does not sandbox arbitrary native HOC/Python code or prevent every
possible native-code crash.

## Input File Not Found

- Paths in `syn_config.json` resolve relative to `cell_configs/`.
- Paths inside `input_blocks[].source.path` should be checked from the runtime
  working directory used by the notebook/CLI.
- For Colab, confirm the referenced file exists after bootstrap/clone.

## No Outputs Written

- Check `save.enabled` in `sim_config.json`.
- Use `force_save = True` in `5_simulate.ipynb`.
- For CLI runs, use `--force-save`.
- For SLURM, `FORCE_SAVE=1` is the default in `run_slurm.sh`.
- If loading an existing run, check `load.enabled` and `load.path`.

## Auto Plots Missing

- Check `plots_profile` is not `off`.
- Check output saving is enabled or forced.
- Check plot dependencies are installed in the active environment.
- Check `save_plots_single_plot_preset` points to an existing preset file.

## Step 6 Has No Runs to Analyze

Step 6 reads saved Step 5 output folders. If no runs appear:

- rerun Step 5 with `force_save = True` or CLI `--force-save`,
- confirm the selected `cell_name`, `tunes_dir`, and `model_dir`,
- confirm `cells/<CELL>/tunes/<TUNE>/output_data/` exists in the runtime.

For project workspaces, also select `project_name` and check
`projects/<PROJECT>/cells/<CELL>/tunes/<TUNE>/output_data/`. An explicit
`tune_dir_override` takes precedence over the shorthand settings. Relative
notebook workspace paths are SCP-root-relative; relative CLI paths are
invocation-directory-relative.

## Project Workspace or Identity Errors

- Use `project_name = None` for legacy/bundled `cells/` selection; do not use a
  UI label such as "Bundled / legacy" as a literal project directory name.
- Project/cell/tune shorthand accepts directory names, not complete paths. Use
  `tune_dir_override`, `--tune-dir`, or SLURM `TUNE_DIR` for a complete path.
- If stored `project_name` contradicts a project path, verify which workspace
  you intended. Do not resolve the error by silently moving existing sources.
- Append/merge rejection can mean two runs have the same cell/tune labels but
  different projects. Compare them in Step 6 instead of combining their trials.
- A project directory does not isolate HOC or mechanism definitions. Restart
  after switching conflicting native definitions.
- Generated-output ignores do not hide model/config files. Entire private
  projects need an exact local Git exclusion; tracked files remain visible.

See [Project workspaces](guides/project_workspaces.md) for the full workflow.

## Notebook vs SLURM Mismatch

- Use the same `tune_dir`, config files, and (when applicable) compiled mechanisms.
- Use the same `randomness_mode` and `seed`.
- Confirm SLURM `TOTAL_TRIALS`, `N_TRIALS`, and `--trial-offset` settings.
- Enable `snapshot.enabled` for deeper comparison.

## ACT or BMTool Not Found

Both integrations are optional and resolved only when requested. For BMTool,
use the environment installation described under
[BMTool import or synapse API unavailable](#bmtool-import-or-synapse-api-unavailable)
above. It needs no sibling checkout; cloning is off by default.

ACT also supports install-once environment use: run
`python -m pip install -r requirements-act.txt`, restart, and validate with
`scp-pipeline check --steps 2 3 --cell PV --tune tuned --check-act` (or select
your external tune explicitly). See [installation](installation.md) for the
pinned source revision, constraints, and Colab instructions.

For an intentional development checkout instead:

```bash
mkdir -p ../mods
git clone https://github.com/V-Marco/ACT.git ../mods/ACT
git -C ../mods/ACT checkout 2b3fdc190107b8242e3496329cfdf240abd09694
python -m pip install -e ../mods/ACT -c requirements.txt
```

If stored elsewhere:

```bash
export SCP_ACT_PATH=/path/to/ACT
```

ACT is optional for Step 2 target-derived proposals and Step 3 ACT active
tuning. Core passive sweeps and manual active/FI checks run without ACT.
BMTool is needed only when Step 4 synapse tuning is requested.

Optional legacy checkout provisioning:

```bash
export SCP_ACT_DIR=/path/for/new/ACT
export SCP_AUTO_CLONE_ACT=1
```

Cloning is disabled by default; the last variable explicitly opts in. It is not
the pinned installation recipe. The setup checker never clones. If ACT reports
missing `scikit-learn` or `timeout-decorator`, update the environment from
`environment.yml`; cloning ACT does not install Python dependencies.

## Custom cells and external sources

- **runtime_cell cannot be rebuilt:** recreate `CELL_OVERRIDE` in this kernel,
  or register a saved Python/HOC builder for CLI/SLURM.
- **Missing canonical soma/all:** use `wrap_cell` with an explicit owner map
  or actual section collections. Do not substitute global `h.allsec()`.
- **Relative asset/import missing:** declare the full native source folders and
  appropriate root-relative working directory; constructor arguments are not
  rewritten. Avoid importing a network driver that runs simulations on import.
- **Source snapshot differs:** normal reruns do not refresh it. Review the
  source preview; an explicit refresh makes backups and may replace local edits.
- **Changed source or conflicting template/module/library:** restart the Python
  process and rebuild/recreate the cell. Autoreload cannot safely replace loaded
  NEURON definitions. A live object must be created after mechanism preparation.
- **Current inspection differs from a saved run:** inspect saved source hashes
  and provenance. A live summary is descriptive, not a full replay checkpoint.

See [Custom models](guides/custom_models.md) for end-to-end guidance. SCP-managed
reference operations do not edit upstream sources, but trusted user model code
is not sandboxed.
