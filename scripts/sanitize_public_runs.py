#!/usr/bin/env python3
"""Sanitize curated simulation runs before committing them publicly.

The command is dry-run by default. Pass ``--apply`` to remove local-only run
artifacts and rewrite portable JSON metadata.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable


DEFAULT_PUBLIC_RUNS = (
    "cells/PV/tunes/tuned/output_data/PV5_base",
    "cells/PV/tunes/tuned/output_data/PV5_tuned",
    "cells/SST/tunes/tuned/output_data/SST5_base",
    "cells/SST/tunes/tuned/output_data/SST5_tuned",
)


@dataclass
class SanitizeReport:
    run_dir: Path
    removed: list[Path] = field(default_factory=list)
    rewritten: list[Path] = field(default_factory=list)


def _repo_relative_path(value: str, repo_root: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        return value
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return value


def _sanitize_value(value: Any, repo_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_value(child, repo_root) for key, child in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(child, repo_root) for child in value]
    if isinstance(value, str):
        return _repo_relative_path(value, repo_root)
    return value


def _absolute_strings(value: Any, location: str = "") -> list[tuple[str, str]]:
    matches: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}" if location else str(key)
            matches.extend(_absolute_strings(child, child_location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            matches.extend(_absolute_strings(child, f"{location}[{index}]"))
    elif isinstance(value, str) and (
        Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
    ):
        matches.append((location or "<root>", value))
    return matches


def _prepare_json(path: Path, repo_root: Path) -> tuple[Any, Any]:
    original = json.loads(path.read_text(encoding="utf-8"))
    sanitized = _sanitize_value(original, repo_root)

    if path.name == "run_manifest.json":
        files = sanitized.get("files", {})
        files.pop("input_stats", None)
        plots = files.get("plots", {})
        for key, value in list(plots.items()):
            plots[key] = f"../plots/{Path(value).name}"

    unresolved = _absolute_strings(sanitized)
    if unresolved:
        details = ", ".join(f"{location}={value!r}" for location, value in unresolved)
        raise ValueError(
            f"Refusing to publish unresolved absolute paths in {path}: {details}"
        )
    return original, sanitized


def sanitize_run(run_dir: Path, repo_root: Path, *, apply: bool = False) -> SanitizeReport:
    """Remove local-only artifacts and make saved JSON paths portable."""

    run_dir = run_dir.resolve()
    repo_root = repo_root.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Public run directory not found: {run_dir}")

    report = SanitizeReport(run_dir=run_dir)
    prepared_json: list[tuple[Path, Any]] = []
    for path in sorted(run_dir.rglob("*.json")):
        if path == run_dir / "results" / "input_stats.json":
            continue
        original, sanitized = _prepare_json(path, repo_root)
        if sanitized != original:
            report.rewritten.append(path)
            prepared_json.append((path, sanitized))

    local_only = [run_dir / "logs", run_dir / "results" / "input_stats.json"]
    for path in local_only:
        if not path.exists():
            continue
        report.removed.append(path)
        if apply:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    if apply:
        for path, sanitized in prepared_json:
            path.write_text(
                json.dumps(sanitized, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    return report


def _find_repo_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "run_pipeline.py").is_file() and (candidate / "modules").is_dir():
            return candidate
    raise FileNotFoundError("Run this command from inside the SCP repository.")


def _resolve_runs(repo_root: Path, values: Iterable[str]) -> list[Path]:
    return [path if path.is_absolute() else repo_root / path for path in map(Path, values)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", default=DEFAULT_PUBLIC_RUNS)
    parser.add_argument("--apply", action="store_true", help="Apply changes instead of previewing them.")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd())
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[{mode}] repo: {repo_root}")
    for run_dir in _resolve_runs(repo_root, args.runs):
        report = sanitize_run(run_dir, repo_root, apply=args.apply)
        print(f"\n{run_dir.relative_to(repo_root)}")
        for path in report.removed:
            print(f"  remove  {path.relative_to(repo_root)}")
        for path in report.rewritten:
            print(f"  rewrite {path.relative_to(repo_root)}")
        if not report.removed and not report.rewritten:
            print("  already clean")

    if not args.apply:
        print("\nNo files changed. Re-run with --apply after reviewing this preview.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
