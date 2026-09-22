# Public Release Cleanup

Use this checklist after a research cycle and before publishing a new SCP
snapshot. The cleanup commands are dry-run by default so the archive inventory
can be reviewed before any files move or are deleted.

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
