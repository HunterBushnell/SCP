# Use SCP from your own project

SCP supplies code; your project owns its notebook, models, configs, targets,
source snapshots, builds, and results. You do not need to move your project
into SCP or copy SCP's implementation into every project.

## Install once

From the SCP checkout, create/activate its normal environment and register it:

```bash
conda env create -f environment.yml
conda activate scp-py311
python -m pip install -e '.[act,bmtool]' -c requirements.txt
```

The suggested command includes ACT and BMTool for optional tuning. For cell-only
use, omit the extra: `python -m pip install -e .`. Existing environments can add
ACT with `python -m pip install -r /path/to/SCP/requirements-act.txt` and/or
BMTool with `python -m pip install -r /path/to/SCP/requirements-bmtool.txt`.
ACT uses a pinned source revision, not an unqualified PyPI `act` installation.
Both are installed once per environment and imported across projects: no `mods/`
layout or per-project tool path is required. See the
[installation guide](../installation.md) for development-checkout overrides.

The dot in `-e .` means the terminal's current directory; the commands above
assume you are inside the SCP checkout. From your working project or any other
directory, use `python -m pip install -e "/path/to/SCP"`, replacing the placeholder
with your SCP checkout's location—not the project containing your notebook.
The environment YAML installs dependencies; the pip command registers SCP itself.

Editable installation
uses that checkout's code from anywhere; it does not make another copy per
project. Base packaging does not replace the pinned scientific environment or
install ACT/BMTool; the explicit `[act,bmtool]` extras add the optional tools and
their declared dependencies, constrained by `-c requirements.txt` in the
suggested setup. Neither is installed automatically during project preparation.
Import `scp_pipeline`, not the generic `modules` name, which may
belong to your project. SCP may live beside your project or be a separately
managed dependency inside it.

Update the checkout with `git pull --ff-only`, then restart kernels/workers.
Reinstall if its location or installation metadata changes. Pin the SCP revision
for reproducibility; one editable checkout does not provide independent versions
per project. New saves record the version and available Git revision.

## Portable interactive notebook

Copy `0_pipeline.ipynb` into your project, select `scp-py311`, and Run All to
render the panels. Rendering does not construct a cell. Installed project mode
never automatically clones SCP or changes the environment.

Hosted Colab initially selects SCP workspaces; choose Project-owned workspace
explicitly to work on project files there. The directory remains an explicit
selection because notebook file locations are not reliably available to kernels.

The expandable **Field guide** is first in Step 1, above the workspace summary.
Location, Cell/Tune, and Results dir are in the main view, followed by one
**Save project settings** button. **Cell setup** contains model
registration and advanced setup, not a second set of project settings.

Registration works in both location modes. Inside SCP, keep **SCP workspaces**
and select Bundled cells or a project namespace; new cells use that namespace's
`cells/<cell>/tunes/<tune>` tree. **Project-owned workspace** is for a directory
you select yourself, not a requirement for HOC or Python models. The loader
choice is independent of where the workspace lives.

For first use:

1. Select **Project-owned workspace**, check **Project dir**, and use **Open /
   reload project** to apply it. The initial directory is the kernel's working
   directory, not an inferred notebook filename. Reload before editing saved
   navigation; replacing unsaved selections requires confirmation.
2. Set **Project name**, **Workspaces** (`.` uses Project dir), and optional
   **Results dir**. **Save project settings** remembers these choices even
   before a cell exists. It does not require a complete model or write one.
3. Open **Cell setup**, choose **Register new model**, and enter Cell/Tune names
   in the text fields directly below Action. Select the model
   type and entry file. Discover/select a HOC template or explicitly enter a
   Python callable; discovery never executes model code.
4. Review source root, MOD folder, copy/reference mode, and explicit voltage /
   temperature. Constructor arguments, section mappings, extra dependencies,
   and working directory are advanced. See native-source guidance below.
5. **Preview model setup → Save cell setup → Prepare and load**. Preview is
   read-only. Save stages sources/configs and navigation but does not construct
   a cell. Prepare compiles/loads mechanisms and constructs the cell.

For returning work, management starts collapsed: review the project/source /
conditions summary, choose Location and Cell/Tune, and optionally set Results dir.
**Save project settings** remembers all these navigation/output choices in one
action; **Prepare and load** constructs the cell separately. Expand management
only to add a cell or supply an explicit **Tune path** (which overrides
Cell/Tune). **Loading options** contains Quiet load and Recompile mechanisms.
The compact notebook's optional `pipeline_settings` starts with
`project_name`, `cell_name`, and `tune_name` set to Python `None`. Existing
`.scp_project.json` navigation restores a remembered selection; otherwise choose
Cell/Tune in the UI. No model is implicitly selected by a blank field. Project
`None` means the top-level **Bundled cells** namespace, not a research project.
In SCP workspaces, **Save project settings** also remembers the selected project
namespace for the next start in the SCP root's optional `selected_project`
navigation field, independently of the bundled cells' remembered tune.
Project-owned notebooks only use their selected
project directory's settings and never search neighboring projects.
Cell/Tune dropdowns show all available names, even when one is already selected.
When registering a model or adopting a live cell, enter names directly under
Action in **Cell setup**. The dropdowns above remain existing-workspace navigation;
there is no need to scroll back to them to create a new name. Saving registration
refreshes those dropdowns and selects the newly registered workspace.
Remembered selections do not pin later Cell/Tune choices. Switching SCP project
namespaces reloads that namespace's navigation and results location; an explicitly
entered **Tune path** still takes precedence until cleared. **Save project settings**
does not load a cell and is optional before **Prepare and load**.
Existing scientific configs remain authoritative and source snapshots are not
refreshed on opening. Model selection locks once loading begins; restart before
changing model sources or selecting another cell. Live-cell work uses the
optional code hook and **Use supplied live cell**, not a serialized object.

Each action has its own status and output beside its control: reload, HOC
discovery, preview, save, and prepare/load. Rerunning an action replaces only
that action's feedback; the other feedback remains visible during the session.
A preview describes the inputs at the time it ran, not later edits. Rerun
**Preview model setup** after changing selections and before saving.

**Discover HOC templates** reads the path already entered in **Entry file**;
it does not open a file chooser. Missing paths and invalid setup fields can be
corrected and retried in the same kernel before loading. Discovery is repeatable,
replaces only its own feedback, and never executes the model. Restart safeguards
apply once model code or mechanisms may have loaded, not to ordinary input errors.

These feedback areas are session UI, not permanent provenance or saved
scientific run records. Reload, discovery, and preview do not save a simulation
run. After restarting the kernel, rerun actions as needed; do not rely on saved
notebook widget output being restored.

For a tidy project, Project dir can be a subdirectory such as `single_cells/SCP`,
with **Workspaces** left as `.`. The notebook can remain in `single_cells/`;
SCP's settings and `cells/` tree then stay inside `single_cells/SCP/`.

New registrations are targetless with no synapse scaffolds by default. In
**Cell setup → Optional configuration generation**, independently opt into initial
target entry or disabled synapse templates. **Preview model setup** validates all
selected operations without writing or running models. **Save cell setup** creates
only missing files, including on an existing tune; it never overwrites scientific
configs or compiles/loads mechanisms. Lasting edits belong in the tune configs.
Without a synapse configuration, select **Current clamp** in Step 5.
Allen acquisition is explicit: Preview reports it, Save downloads without cell
construction. The remaining pipeline panels work as usual.

### Optional targets and working copies

Target generation defaults to `manual` when enabled. Enter passive values and FI
arrays (JSON lists of equal length) **or** a CSV, or leave a blank template for
later. Select `traces` or `allen_nwb` for file-backed targets; only that mode's
settings appear, optionally alongside manual targets. Explicit `none` creates a
targetless config. Entered files must exist, with relative paths interpreted
from the **tune**, not Project dir or notebook location; Preview lists resolved
locations. Active traces require existing ACT-compatible NPY arrays, not arbitrary
experimental files. Setup does not extract targets, run ACT, or add converters.
Existing target files are read-only here, with their location shown for direct
editing. Setup drafts are not session target overrides or notebook-default profiles.
Before leaving an unsaved draft, save it or explicitly confirm its discard.
The same guard runs before **Prepare and load**, so an unsaved form cannot be
silently locked without its target/synapse files being generated. Preview and
Save also validate project navigation and destination paths before creating files.

The required `sim_config.json` provides initialization conditions for Steps 2–3;
unused Step 5 settings do not mean Step 5 must be run. Disabled synapse templates
are optional even for a fully registered model; generating them does not create
inputs, instantiate synapses, or initialize BMTool.

Before loading, select a saved tune and use **Preview copy → Create working copy**
to create a sibling `tuned` (or another new name). Compact copies include native
sources, configs, saved source snapshots, and required local data—not old results,
exports, archives, compiled products, or caches. A dependency inside an excluded
output directory blocks copying; relocate that required data and update the
source config first. External data references remain external.

Copies use the source tune's **saved snapshot**, never updated upstream files.
For registered models the new tune can inherit copy/reference mode or explicitly
choose either. Copy runs the duplicated snapshot and works without upstream;
reference retains the external dependency and shares manual source edits between
tunes. Unavailable reference inputs are reported, never fetched. No source tune
or snapshot is refreshed. Creation selects the new tune but does not construct
it. After Prepare and load starts, restart before copying or selecting another
tune because NEURON definitions and mechanism libraries are process-global.

### Which native files are used?

The **Tune directory** is the workspace for configs, snapshots, builds, and
results, not necessarily the working model location. Step 1's Model summary and
load summary identify the resolved native entry file, template/callable, source
mode, and MOD sources.

- **Copy** loads the saved `model_sources/` bundle. External edits do not alter it.
- **Reference** loads the external originals. A local snapshot is still retained
  but is not automatically synchronized or used as the working model. MODs are
  compiled in a tune-owned build cache, never in the upstream source directory.

For direct editing of a network project's native templates, one reference-mode
tune per cell is sufficient. Two reference tunes pointing at the same files do
not retain independent parameter versions. To keep an independently runnable
baseline, use `orig` in copy mode and duplicate to `tuned` in reference mode, or
later duplicate a reference tune with **Use sources: copy** to run its saved
snapshot. The source snapshot is not a complete original experiment: runtime
conditions, targets, and protocols are owned by their configs. Saved Step 5 runs
normally archive current native inputs separately under `model_artifacts/`,
including hashes and construction-source mismatch reporting.

Restart after HOC/MOD edits before reloading. Keep `project_name` consistent
between `.scp_project.json` and the cell configs; saving project settings does
not duplicate a model or overwrite its scientific configs.

## Persistence and paths

**Save project settings** writes `.scp_project.json`: format version, project identity,
workspace root, selected tune, optional output directory, and optional analysis
settings. It stores navigation, not duplicate scientific configs or live objects.
Project-contained paths are relative; explicitly selected outside paths remain
external dependencies. Changing settings does not silently change scientific data.

On reopening, select the project directory, then use **Open / reload project**
to apply its saved navigation choices. Reload is separate from
editing, previewing, saving, or loading a model; it never saves the fields you
just edited. Launching Jupyter from an unrelated directory may require selecting
the project again. The default new project
identity is its directory name, editable before saving; relocation does not
rename saved identity. Existing unscoped workspaces/results are not reassigned.
Different project identities can be compared, but cannot be pooled as trials.

Notebook selections are project-relative, loader inputs are native-root-relative,
and existing tune/config-relative fields keep their historical meaning. Explicit
CLI paths are invocation-directory-relative. Output overrides propagate to
fresh-process simulation and saved analysis; otherwise everything stays in the
tune. An external workspace is separate from an external native-source reference.

## Remembered notebook defaults

The compact notebook automatically restores `.scp_pipeline_settings.json` from
the selected project directory. SCP-hosted named workspaces use their namespace
root; bundled examples use the SCP root. It is one notebook-wide preset per
project, shared by copied notebooks using that project—not one preset per cell.
No file is required on first use and no file is written just by rendering.

Use the protocol/simulation panels normally, then **Save notebook defaults**
to remember currents, timing, explicit target/simulation overrides, loading
preferences, and display choices. Save validates current fields even if a
protocol has not run. Merely displaying values from a cell config does not
promote them into overrides. Cell/tune selection stays in project navigation;
native sources, construction recipes, runtime objects, and logs are excluded.
The saved profile is not another copy of model/scientific configuration.

Saved choices override the optional initial code-first defaults; subsequent UI
edits affect the session. Changing cells keeps these notebook choices. Changing
projects restores the other project's defaults and requires Save, Discard, or
Cancel if the current choices are unsaved. Repeated Run All does not restore
over current edits or rebuild the cell. The environment's working directory
still only suggests a project: SCP never guesses a notebook filename or scans
unrelated projects to find a preset.

Expand **Notebook defaults options** to choose a project-relative defaults-file
path for an independent notebook copy, reload, or reset the session to its
initial defaults. Reload/reset require confirmation for unsaved edits; reset
does not delete or overwrite the saved file. Construction controls stay locked
after loading, while protocol/display defaults can still be restored. Invalid
profiles leave current controls unchanged and show an actionable message.

For Colab, **Export defaults JSON** provides a download; **Import JSON** applies
an uploaded profile to the session without writing a file. Save explicitly to
persist it on mounted or local storage. Saving an `.ipynb` does not reliably
restore widget choices. Keep native parameter tuning in native model files;
notebook defaults do not apply to CLI/SLURM automatically. Saved runs continue
to record their effective configurations independently of this convenience file.

Code-first callers can opt in without changing the legacy controller behavior:

```python
from scp_pipeline.notebooks import PipelineNotebookUI

ui = PipelineNotebookUI(project_root="my_project", restore_notebook_settings=True)
# Optional explicit file: notebook_settings_path="preferences/notebook.json"
# After editing controls:
# ui.save_notebook_settings()
# ui.load_notebook_settings(discard=True)
```

## Native sources and manual edits

Copy mode defaults to an independent byte-preserving `model_sources/` snapshot.
Reference mode retains that snapshot but constructs from the original inputs.
Refresh requires an explicit, backed-up operation. Copy-mode reconstruction does
not need the upstream tree after staging.

The entry file is included automatically. Declare companion files with **Files
JSON** (for example `["parameters.dat"]`) and folders with **Folders JSON**.
MOD folder paths are source-root-relative and included as dependencies. For
Python packages include the package folder and assets. Allen bundles need the
complete manifest-declared native sources and matching mechanisms. SCP does not
guess every file that arbitrary model code might open.

A broad project source root is allowed when selected inputs are disjoint from
the workspace/output destination. Do not select the entire project folder when
it contains the tune. Traversal and escaping symlinks are rejected; caches,
compiled libraries, and generated results are excluded. External mechanisms
compile only in a tune-owned mirror. SCP operations do not modify upstream
sources; trusted user HOC/Python is not sandboxed.

In copy mode edit your tune's working sources, not the unused upstream original.
In reference mode manual upstream edits affect subsequent reconstruction. Native
source/mechanism changes after loading require a fresh process. Current-model
inspection can differ from the model archived with an older run.

## Python, CLI, analysis, and process boundaries

Supported namespaces are `scp_pipeline.setup`, `.model`, `.protocols`,
`.simulation`, `.analysis`, and `.notebooks`. For an already registered cell:

```python
from pathlib import Path
from scp_pipeline.protocols import (
    prepare_pipeline_notebook, run_passive_stage,
    run_active_protocol_stage, run_fi_curve_stage,
)

state = prepare_pipeline_notebook(
    project_root=Path.cwd(), cell_name="MyCell", tune_name="orig",
)
passive = run_passive_stage(state)
active = run_active_protocol_stage(state)
fi = run_fi_curve_stage(state)
```

Review the existing protocol defaults for your model. Use
`setup.preview_workspace_registration`/`register_workspace` for registration and
`simulation.SimulationSession` for prepare/run/save. Internal helpers are not a
promise of a stable public API.

```bash
scp-pipeline run --project-root . --iclamp --n-trials 1 --force-save
scp-pipeline check --project-root . --steps 2 3 5
python -m scp_pipeline run --tune-dir cells/MyCell/tunes/orig --iclamp
```

`--project-root` uses the saved selection unless cell/tune or an explicit path
is supplied. `--tune-dir` wins; `--output-dir` overrides saved output settings.
Existing repository scripts and SLURM wrappers remain supported. On SLURM use
the same environment and an explicit absolute `TUNE_DIR`.

Load saved results with `scp_pipeline.analysis.load_results` without constructing
a cell. For project analysis defaults use
`scp_pipeline.projects.load_analysis_settings(project_root)`; register a JSON
override with `save_project_settings(project_root, analysis_settings="analysis_settings.json")`.
Packaged defaults are read-only, and project mode does not enable the bundled
PV/SST scientific comparisons by default.

Live cells use the `CELL_OVERRIDE` code-cell hook and execute in-process. They
must be recreated after restarting; their saved descriptive summaries are not
NEURON checkpoints. Use a saved Python builder for notebook/CLI/SLURM
reconstruction. Do not select a network driver merely because it is Python.
See [custom models](custom_models.md) for builder/section ownership requirements.
Conflicting NEURON definitions require fresh processes; there is no parallel
in-process NEURON execution.

Only `0_pipeline.ipynb` is portable. Keep detailed Steps 1–7 in SCP and point
them at your external tune. ACT/BMTool remain optional, and non-Allen ACT
optimization remains experimental.

Project-owned does not mean everything must be committed. Track the notebook,
settings, configs, and sources according to your project policy; ignore build
caches and large generated outputs. SCP does not edit your project's Git policy
or stage/commit files for you.
