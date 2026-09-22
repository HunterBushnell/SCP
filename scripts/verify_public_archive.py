#!/usr/bin/env python3
"""Verify archived paths against a public-cleanup manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .prepare_public_cleanup import _find_repo_root, _tree_stats
except ImportError:
    from prepare_public_cleanup import _find_repo_root, _tree_stats


def verify_archive(repo_root: Path, archive_root: Path) -> list[str]:
    manifest_path = archive_root / "manifest.json"
    if not manifest_path.is_file():
        return [f"Missing archive manifest: {manifest_path}"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    checked = 0
    for record in manifest.get("records", []):
        destination = record.get("destination")
        if not destination:
            continue
        checked += 1
        path = repo_root / destination
        if not path.exists():
            errors.append(f"Missing archived path: {destination}")
            continue
        actual_files, actual_bytes = _tree_stats(path)
        expected = (int(record["files"]), int(record["bytes"]))
        actual = (actual_files, actual_bytes)
        if actual != expected:
            errors.append(
                f"Archive mismatch for {destination}: expected {expected}, got {actual}"
            )

    if checked == 0:
        errors.append(f"Manifest has no archived destinations: {manifest_path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", help="Archive directory containing manifest.json")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd())
    archive_root = Path(args.archive_root).expanduser()
    if not archive_root.is_absolute():
        archive_root = repo_root / archive_root
    archive_root = archive_root.resolve()

    errors = verify_archive(repo_root, archive_root)
    if errors:
        print("Archive verification failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"Archive verified: {archive_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
