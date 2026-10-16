# Project Workspaces

Use a project namespace when separate projects contain cells with the same name.
The namespace groups SCP workspaces; it does not change how NEURON cells are
constructed or adopt an entire network model.

## Additive layout

Both layouts are supported:

The project pattern is `projects/<project>/cells/<cell>/tunes/<tune>/`.

```text
cells/PV/tunes/tuned/                             # bundled/default example
projects/study_a/cells/ExampleCell/tunes/orig/     # project baseline
projects/study_a/cells/ExampleCell/tunes/tuned/    # project working copy
projects/study_b/cells/ExampleCell/tunes/tuned/    # a separate workspace
```

Every tune keeps the ordinary `cell_configs/`, native sources, optional
`model_sources/`, and `output_data/` layout. Project selection introduces no new
loader, targets, scientific defaults, or tuning behavior. Existing `cells/`
assets, saved runs, and PV notebook defaults remain unchanged; there is no
automatic migration. The SLURM wrapper retains its historical SST default when
no selection is supplied.

Project, cell, and tune shorthand values are single directory names, not paths.
Use `tune_dir_override` or `--tune-dir` for a complete workspace path. Traversal
and shorthand that escapes a namespace through a symlink are rejected.

## Select a workspace

In the compact notebook's settings mapping:

```python
pipeline_settings = {
    "project_name": "study_a",
    "cell_name": "ExampleCell",
    "tune_name": "tuned",
    "tune_dir_override": None,
}
```

In a detailed notebook's selection cell:

```python
project_name = "study_a"
cell_name = "ExampleCell"
tune_name = "tuned"
tune_dir_override = None
```

Step 6 uses its existing `model_dir` variable for the selected tune and
`tunes_dir = "tunes"` for its parent. Set `project_name` there too. Widgets
offer the legacy/bundled namespace alongside discovered projects.

With `project_name = None`, selection stays under `cells/`. A complete path
takes precedence over all shorthand settings:

```python
tune_dir_override = "projects/study_b/cells/ExampleCell/tunes/tuned"
```

Notebook relative workspace paths are anchored to SCP. CLI relative paths
remain relative to the directory from which you invoke the command; the
examples below assume the SCP root. Selection determines effective workspace
identity even if notebook shorthand still contains PV defaults. It does not
rewrite existing config files merely to add namespace metadata.

## Setup, CLI, and SLURM

Stage a model or register external sources in the chosen `orig` tune, then use
the ordinary Step 1 workflow. For an already staged, synthetic Python builder:

```bash
python scripts/step1_prepare.py \
  --project study_a --cell ExampleCell --tune orig \
  --source-type existing --cell-loader python_factory \
  --python-factory-file model/cell.py --python-factory-callable build_cell \
  --v-init-mv -70 --celsius-c 34 --target-source-mode none --no-synapse-configs
```

Here `model/cell.py` must already exist in the selected tune and return a valid
compartmental cell. The voltage/temperature are demonstration choices, not
scientific defaults. See [Custom models](custom_models.md) for a complete builder
and live-cell recipes.

Create the usual sibling `tuned` working copy before editing baseline sources.
No step silently moves or copies old `cells/` assets into a project.

Run a prepared project tune:

```bash
python run_pipeline.py --project study_a --cell ExampleCell --tune tuned \
  --iclamp --n-trials 1 --force-save
```

The explicit-path equivalent is:

```bash
python run_pipeline.py \
  --tune-dir projects/study_a/cells/ExampleCell/tunes/tuned \
  --iclamp --n-trials 1 --force-save
```

Setup checks and optional configured integrations also accept project selection
or an explicit tune path. Saved Python/HOC/Allen recipes reconstruct normally;
CLI/SLURM still cannot reconstruct a notebook-only live object.

SLURM shorthand:

```bash
PROJECT=study_a CELL=ExampleCell TUNE=tuned ICLAMP=1 N_TRIALS=1 \
  sbatch run_slurm.sh
```

`TUNE_DIR` overrides `PROJECT`, `CELL`, and `TUNE`. Prefer an absolute `TUNE_DIR`
for scheduler jobs submitted outside SCP:

```bash
TUNE_DIR=/path/to/SCP/projects/study_a/cells/ExampleCell/tunes/tuned \
  ICLAMP=1 N_TRIALS=1 sbatch run_slurm.sh
```

## Namespace is separate from model source

`project_name` selects the SCP workspace. `model_source` selects the native
construction inputs. They are independent: a project tune can hold an ordinary
staged model, a saved Python builder, a live-cell workspace, or an external
copy/reference registration.

Keep external production repositories in their own locations. An external
registration still saves a snapshot under the selected tune's `model_sources/`.
Copy mode uses that local bundle; reference mode deliberately uses the original
sources. Creating another project directory does not make reference-mode
sources independent. Copy mode and separate tune copies are appropriate when
you need independent manual edits. See [Custom models](custom_models.md).

Project directories do not isolate process-global HOC templates, Python imports,
or mechanism definitions. Use a fresh process when switching conflicting
definitions, even if the two models live in different projects.

## Shared Python helpers

The dependency-light workspace helpers discover directories without importing
NEURON or executing model files. From the SCP root:

```python
from pathlib import Path
from modules.workspaces import (
    discover_projects, discover_cells, discover_tunes, resolve_workspace,
)

root = Path.cwd()
print(discover_projects(root))
print(discover_cells(root, project_name="study_a"))
print(discover_tunes(root, "ExampleCell", project_name="study_a"))
workspace = resolve_workspace(
    repo_root=root, project_name="study_a",
    cell_name="ExampleCell", tune_name="tuned",
)
print(workspace.tune_dir, workspace.identity)
```

Resolution does not create a workspace. Existing notebook path helpers remain
compatibility wrappers. Use `relative_to=Path.cwd()` when applying CLI-style
relative explicit paths in your own entry point. Namespace symlinks must not
alias another project, and cell/tune symlinks must stay inside their namespace.

## Configuration and saved identity

New project config scaffolds can include the optional namespace field:

```json
{
  "project_name": "study_a",
  "cell_name": "ExampleCell",
  "tune": "tuned"
}
```

This is an identity excerpt, not a complete loader configuration. Legacy
configs can omit `project_name`. Stored namespace metadata must not contradict
a recognized project workspace path; resolve the discrepancy explicitly rather
than letting a config relabel another project's directory. Normal readers do
not rewrite files to add this field.

Saved runs record the effective logical namespace separately from the absolute
`tune_dir` used at runtime:

```json
{
  "workspace_identity": {
    "project_name": "study_a",
    "cell_name": "ExampleCell",
    "tune_name": "tuned"
  }
}
```

Legacy workspaces use `project_name: null`. Inspect the run manifest and saved
metadata rather than guessing identity from result filenames. Workspace identity
does not establish biological equivalence: matching names do not prove that source
hashes, parameters, conditions, or protocols are equal.

Appends and compatible-result merges guard against mixing different project,
cell, or tune identities. Older saved runs remain readable; missing historical
namespace information is not evidence that a run belongs to a newly selected
project. Step 6 comparison is different from merging trials and can compare
explicit saved-run paths across projects.

Step 7 can restore into another explicitly selected tune. Review the dry run;
restoration keeps the destination's workspace identity rather than turning it
into the source project. Backups, hash checks, native-source protection, and
restart/recompile requirements still apply. Model-artifact materialization
remains an explicit action, not an automatic response to changing a project.

## Git visibility and private work

Project native model files and configs are Git-eligible by default, just as
under `cells/`. Generated outputs, notebook/tuning exports, ACT workspaces,
restore backups, NWB downloads, and compiled products are ignored. The ignored
`.scp_build/` cache applies in either layout.

To keep an entire project private, find the local exclusion file from the SCP
root:

```bash
git rev-parse --git-path info/exclude
```

Open the reported file in your editor and add an exact project rule, for example:

```text
/projects/private_study/
```

This changes only your checkout's local exclusions. Do not add a private project
to tracked `.gitignore` unless you intentionally want that repository policy.
Check the rule with `git check-ignore -v <path>` before committing. Ignore rules
do not hide files already tracked by Git, and adding a project does not establish
redistribution permission for its model sources.

Cleanup/export tools keep their explicit selected-workspace scope; they do not
automatically expand an old `cells/` cleanup into every project. Existing curated
PV/SST outputs and presets are left in place.
