# Public Release Cleanup

Use this checklist after a research cycle and before publishing a new SCP
snapshot. The cleanup commands are dry-run by default so the archive inventory
can be reviewed before any files move or are deleted.

## Repository transfer does not require archive cleanup

The prepared v0.3.0 checkpoint records completed workflow improvements and
readiness for `HunterBushnell/SCP` → `cyneuro/SCP`. It does not publish itself,
create a release, or perform the transfer. Follow the
[transfer checklist](repository_transfer.md): commit/push the reviewed checkpoint,
manually transfer, verify access/settings/apps, then update the local remote.
Live links and Colab entry points still need post-transfer checks.

Do not run archive cleanup for this ownership change. Leave A1, production and
private workspaces, scientific assets, generated research data, and licensing
files unchanged.

## Lightweight lab-use commit checkpoint

A code/documentation update does not require archiving research runs or deleting
compiled caches. Leave ignored project data and production sources in place.
Before committing:

- Back up locally edited notebooks, then clear saved outputs/execution counts,
  use the `scp-py311` kernelspec, and remove local research selections from the
  public copies. Compact selection may remain blank; detailed notebooks use PV.
- Keep checkout-local `.scp_project.json` and `.scp_pipeline_settings.json`
  private. Project-owned repositories can track their own settings separately.
- Include new package modules, resources, tests, docs, and intended example/data
  changes, not just files already tracked by Git. Review intentional removals
  from earlier project migrations; do not force-add ignored private destinations.
- Run the checks in section 4, the installed CLI help paths, and the copied
  notebook/package tests. The optional ACT/BMTool integrations need validation
  only when used; core single-cell characterization must remain independent.
- Read the [roadmap](roadmap.md) for known limitations. The user confirmed
  discovery, save, and copy feedback working after reloading VS Code; this does
  not establish broad browser coverage. ACT active optimization remains
  experimental with a known upstream defect, and detailed synaptic recording
  remains paused/default-off. Passing unit tests do not close those reviews.

Do not run the archive cleanup below solely to make `git status` shorter.
This checkpoint does not commit, tag, push, or create a release.

## 1. Review the worktree

```bash
git status --short --branch
python scripts/check_notebooks.py
```

Commit or intentionally preserve any existing source/configuration changes
before cleaning generated data.

## 2. Preview the archive cleanup

```bash
python scripts/prepare_public_cleanup.py \
  --archive-root archive/public_cleanup_YYYYMMDD
```

The command inventories all non-SLURM output runs, curated public runs, local
research/tune artifacts, SLURM working directories, compiled mechanisms, and
Python caches. Its archive destination is ignored by Git.

## 3. Apply the reviewed cleanup

```bash
python scripts/prepare_public_cleanup.py \
  --archive-root archive/public_cleanup_YYYYMMDD \
  --apply
```

The applied command:

- archives every non-SLURM run while preserving its repository-relative path;
- retains only the four curated PV/SST public runs in active `output_data/`;
- deletes disposable `slurm_*` working directories;
- archives known historical/private tune directories, NWB files, and logs;
- removes rebuildable Python and NEURON compilation caches;
- removes public-run logs and large `input_stats.json` sidecars; and
- rewrites public-run JSON paths so they are portable.

The archive contains `manifest.json` with the action, source, destination,
file count, and byte count for each processed item.

The `archive/` directory is an ignored local safety copy, not an off-machine
backup. Copy important archives to durable storage before deleting the local
repository or cleaning the archive itself.

Verify an archive at any time with:

```bash
python scripts/verify_public_archive.py archive/public_cleanup_YYYYMMDD
```

## 4. Validate the public tree

```bash
python scripts/check_notebooks.py
python -m unittest discover -s tests -v
python -m compileall -q modules scripts run_pipeline.py
git diff --check
git status --short --ignored
```

Also inspect the four curated run folders and confirm no user-specific absolute
paths, local logs, NWB files, or large generated sidecars remain visible.
