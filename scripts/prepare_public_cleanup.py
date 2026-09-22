#!/usr/bin/env python3
"""Archive local SCP research artifacts and retain curated public examples.

This command is dry-run by default. It archives every non-SLURM run, removes
disposable SLURM working directories, moves known private or historical local
artifacts into an ignored archive, and deletes rebuildable caches. Use
``--apply`` only after reviewing the printed plan.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    from .sanitize_public_runs import DEFAULT_PUBLIC_RUNS, sanitize_run
except ImportError:
    from sanitize_public_runs import DEFAULT_PUBLIC_RUNS, sanitize_run


ARCHIVE_PATHS = (
    "PUBLIC_RELEASE_TODO.local.md",
    "cells/IND-",
    "cells/INM-",
    "cells/PUD-",
    "cells/PV_paper",
    "cells/SST_paper",
    "cells/PV/tunes/tuned_oldampa",
    "cells/SST/tunes/orig_peri",
    "cells/SST/tunes/tuned_allactive_newampa",
    "cells/SST/tunes/tuned_allactive_oldampa",
    "cells/SST/tunes/tuned_ampa_tune5.0_dantune2",
    "cells/SST/tunes/tuned_peri_dantune1",
    "cells/PV/tunes/tuned/restore_backups",
    "cells/SST/tunes.svg",
    "cells/SST/tunes/tuned.svg",
    "cells/SST/tunes/tuned/485466109_fit_5.json",
)


@dataclass
class ArchiveRecord:
    action: str
    source: str
    destination: str | None
    files: int
    bytes: int


def _tree_inventory(path: Path) -> dict[str, tuple[str, int | str]]:
    if path.is_symlink():
        return {".": ("symlink", str(path.readlink()))}
    if path.is_file():
        return {".": ("file", path.stat().st_size)}
    inventory: dict[str, tuple[str, int | str]] = {}
    for child in path.rglob("*"):
        relative = child.relative_to(path).as_posix()
        if child.is_symlink():
            inventory[relative] = ("symlink", str(child.readlink()))
        elif child.is_dir():
            inventory[relative] = ("directory", 0)
        elif child.is_file():
            inventory[relative] = ("file", child.stat().st_size)
    return inventory


def _tree_stats(path: Path) -> tuple[int, int]:
    inventory = _tree_inventory(path)
    files = sum(1 for kind, _value in inventory.values() if kind in {"file", "symlink"})
    total = sum(int(value) for kind, value in inventory.values() if kind == "file")
    return files, total


def _copy_verified(source: Path, destination: Path) -> tuple[int, int]:
    source_inventory = _tree_inventory(source)
    files, total = _tree_stats(source)
    if destination.exists():
        raise FileExistsError(f"Archive destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        destination.symlink_to(source.readlink(), target_is_directory=source.is_dir())
    elif source.is_dir():
        shutil.copytree(source, destination, copy_function=shutil.copy2, symlinks=True)
    else:
        shutil.copy2(source, destination)
    if source_inventory != _tree_inventory(destination):
        raise RuntimeError(
            f"Archive inventory verification failed for {source}. "
            "The copied relative paths, object types, or sizes differ."
        )
    return files, total


def _remove(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _find_repo_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "run_pipeline.py").is_file() and (candidate / "modules").is_dir():
            return candidate
    raise FileNotFoundError("Run this command from inside the SCP repository.")


def _non_slurm_runs(repo_root: Path) -> list[Path]:
    runs: list[Path] = []
    for output_dir in sorted(repo_root.glob("cells/**/output_data")):
        runs.extend(
            path
            for path in sorted(output_dir.iterdir())
            if path.is_dir() and not path.name.startswith("slurm_")
        )
    return runs


def _slurm_runs(repo_root: Path) -> list[Path]:
    return sorted(
        path
        for output_dir in repo_root.glob("cells/**/output_data")
        for path in output_dir.iterdir()
        if path.is_dir() and path.name.startswith("slurm_")
    )


def _log_artifacts(repo_root: Path) -> list[Path]:
    logs = repo_root / "logs"
    if not logs.is_dir():
        return []
    artifacts: list[Path] = []
    for path in sorted(logs.iterdir()):
        if path.name == ".gitignore":
            continue
        if path.is_dir() and (path / ".gitkeep").is_file():
            artifacts.extend(child for child in sorted(path.iterdir()) if child.name != ".gitkeep")
        else:
            artifacts.append(path)
    return artifacts


def _cache_artifacts(repo_root: Path) -> list[Path]:
    archive_root = (repo_root / "archive").resolve()
    paths = {
        path
        for path in repo_root.rglob("__pycache__")
        if archive_root not in path.resolve().parents
    }
    paths.update(repo_root.glob("cells/**/x86_64"))
    paths.update(repo_root.glob("cells/**/modfiles/x86_64"))
    return sorted(paths, key=lambda path: (len(path.parts), str(path)), reverse=True)


def _write_manifest(archive_root: Path, records: list[ArchiveRecord]) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "records": [asdict(record) for record in records],
    }
    (archive_root / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_public_runs(public_runs: set[Path]) -> None:
    missing = sorted(path for path in public_runs if not path.is_dir())
    if missing:
        joined = "\n  - ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Curated public run directories are missing:\n  - {joined}")


def _validate_archive_root(repo_root: Path, archive_root: Path) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", str(archive_root)],
        cwd=repo_root,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(
            f"Archive destination is not ignored by Git: {archive_root}. "
            "Add an appropriate ignore rule before cleanup."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        default=f"archive/public_cleanup_{datetime.now().strftime('%Y%m%d')}",
        help="Ignored archive directory, relative to the repository by default.",
    )
    parser.add_argument("--apply", action="store_true", help="Apply the reviewed cleanup plan.")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd())
    archive_root = Path(args.archive_root).expanduser()
    if not archive_root.is_absolute():
        archive_root = repo_root / archive_root
    archive_root = archive_root.resolve()
    try:
        archive_root.relative_to((repo_root / "archive").resolve())
    except ValueError as exc:
        raise ValueError("Archive root must be inside the repository's ignored archive/ directory.") from exc
    if archive_root.exists():
        raise FileExistsError(f"Choose a new archive root; path already exists: {archive_root}")
    _validate_archive_root(repo_root, archive_root)

    public_runs = {(repo_root / value).resolve() for value in DEFAULT_PUBLIC_RUNS}
    _validate_public_runs(public_runs)
    non_slurm_runs = _non_slurm_runs(repo_root)
    slurm_runs = _slurm_runs(repo_root)
    archive_paths = [repo_root / value for value in ARCHIVE_PATHS]
    archive_paths.extend(sorted(repo_root.glob("cells/**/*.nwb")))
    archive_paths.extend(_log_artifacts(repo_root))
    archive_paths = [path for path in archive_paths if path.exists()]
    cache_paths = [path for path in _cache_artifacts(repo_root) if path.exists()]

    print(f"repo: {repo_root}")
    print(f"archive: {archive_root}")
    print(f"non-SLURM runs to archive: {len(non_slurm_runs)}")
    print(f"curated runs retained: {len(public_runs)}")
    print(f"SLURM working directories to delete: {len(slurm_runs)}")
    print(f"local artifact paths to archive: {len(archive_paths)}")
    print(f"cache/compiled paths to delete: {len(cache_paths)}")

    for path in non_slurm_runs:
        action = "retain after archive" if path.resolve() in public_runs else "remove after archive"
        print(f"  run      {path.relative_to(repo_root)} [{action}]")
    for path in slurm_runs:
        print(f"  delete   {path.relative_to(repo_root)}")
    for path in archive_paths:
        print(f"  archive  {path.relative_to(repo_root)}")

    if not args.apply:
        print("\nNo files changed. Re-run with --apply after reviewing this preview.")
        return 0

    records: list[ArchiveRecord] = []
    archive_root.mkdir(parents=True)

    for source in non_slurm_runs:
        relative = source.relative_to(repo_root)
        destination = archive_root / "output_runs" / relative
        files, total = _copy_verified(source, destination)
        records.append(ArchiveRecord("copied_run", str(relative), str(destination.relative_to(repo_root)), files, total))

    for source in non_slurm_runs:
        if source.resolve() not in public_runs:
            files, total = _tree_stats(source)
            _remove(source)
            records.append(ArchiveRecord("removed_archived_run", str(source.relative_to(repo_root)), None, files, total))

    for source in slurm_runs:
        files, total = _tree_stats(source)
        _remove(source)
        records.append(ArchiveRecord("deleted_slurm_run", str(source.relative_to(repo_root)), None, files, total))

    for source in cache_paths:
        if not source.exists():
            continue
        files, total = _tree_stats(source)
        _remove(source)
        records.append(ArchiveRecord("deleted_rebuildable_cache", str(source.relative_to(repo_root)), None, files, total))

    for source in archive_paths:
        if not source.exists():
            continue
        relative = source.relative_to(repo_root)
        destination = archive_root / "local_artifacts" / relative
        files, total = _copy_verified(source, destination)
        _remove(source)
        records.append(ArchiveRecord("moved_local_artifact", str(relative), str(destination.relative_to(repo_root)), files, total))

    for run_dir in sorted(public_runs):
        report = sanitize_run(run_dir, repo_root, apply=True)
        for path in report.removed:
            records.append(ArchiveRecord("removed_public_local_artifact", str(path.relative_to(repo_root)), None, 0, 0))
        for path in report.rewritten:
            records.append(ArchiveRecord("sanitized_public_json", str(path.relative_to(repo_root)), None, 1, path.stat().st_size))

    _write_manifest(archive_root, records)
    print(f"\nCleanup complete. Manifest: {archive_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
