#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass
class CleanupReport:
    output_data_dir: Path
    prefix: str
    dry_run: bool
    matched: List[Path] = field(default_factory=list)
    removed: List[Path] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _resolve_output_data_dir(
    *,
    tune_dir: Optional[Path] = None,
    output_data_dir: Optional[Path] = None,
) -> Path:
    if output_data_dir is None and tune_dir is None:
        raise ValueError("Provide either tune_dir or output_data_dir.")

    if output_data_dir is not None:
        resolved = Path(output_data_dir).expanduser().resolve()
    else:
        resolved = (Path(tune_dir).expanduser().resolve() / "output_data")

    if not resolved.exists():
        raise FileNotFoundError(f"Output data directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Output data path is not a directory: {resolved}")
    return resolved


def _collect_matches(output_data_dir: Path, prefix: str) -> List[Path]:
    return sorted(
        (path for path in output_data_dir.iterdir() if path.name.startswith(prefix)),
        key=lambda path: path.name,
    )


def clear_slurm_runs(
    *,
    tune_dir: Optional[Path] = None,
    output_data_dir: Optional[Path] = None,
    prefix: str = "slurm_",
    dry_run: bool = True,
) -> CleanupReport:
    prefix_text = str(prefix)
    if not prefix_text:
        raise ValueError("prefix must not be empty.")

    out_dir = _resolve_output_data_dir(
        tune_dir=Path(tune_dir) if tune_dir is not None else None,
        output_data_dir=Path(output_data_dir) if output_data_dir is not None else None,
    )

    report = CleanupReport(
        output_data_dir=out_dir,
        prefix=prefix_text,
        dry_run=bool(dry_run),
    )

    matches = _collect_matches(out_dir, prefix_text)
    report.matched.extend(matches)
    if not matches:
        report.warnings.append(f"No entries found in {out_dir} with prefix '{prefix_text}'.")
        return report

    if report.dry_run:
        return report

    for path in matches:
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
            report.removed.append(path)
        except Exception as exc:
            report.errors.append(f"Failed to remove {path}: {exc}")

    return report


def print_report(report: CleanupReport, *, max_items: int = 80) -> None:
    show_n = max(1, int(max_items))
    mode = "DRY-RUN" if report.dry_run else "WRITE"
    print(f"[clear_slurm_runs] Mode: {mode}")
    print(f"[clear_slurm_runs] output_data_dir: {report.output_data_dir}")
    print(f"[clear_slurm_runs] prefix: {report.prefix!r}")
    print(f"[clear_slurm_runs] matches: {len(report.matched)}")
    print("")

    if report.warnings:
        for warning in report.warnings:
            print(f"[warning] {warning}")
        print("")

    if report.matched:
        label = "would_remove" if report.dry_run else "removed"
        for path in report.matched[:show_n]:
            print(f"[{label}] {path}")
        if len(report.matched) > show_n:
            print(f"[{label}] ... ({len(report.matched) - show_n} more)")
        print("")

    if report.errors:
        for error in report.errors:
            print(f"[error] {error}")
        print("")

    if report.dry_run:
        print("[clear_slurm_runs] Dry-run only. Re-run with --write to delete.")
    elif report.errors:
        print(
            f"[clear_slurm_runs] Completed with errors. Removed {len(report.removed)} / {len(report.matched)}."
        )
    else:
        print(f"[clear_slurm_runs] Removed {len(report.removed)} entries.")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Delete top-level entries under a tune's output_data directory that start with a prefix "
            "(default: slurm_). Default mode is dry-run."
        )
    )
    p.add_argument(
        "--tune-dir",
        default=None,
        help="Tune directory containing output_data (for example: cells/SST/tunes/seg_tuned_all).",
    )
    p.add_argument(
        "--output-data-dir",
        default=None,
        help="Direct path to output_data (overrides --tune-dir when both are provided).",
    )
    p.add_argument(
        "--prefix",
        default="slurm_",
        help="Entry-name prefix to match (default: slurm_).",
    )
    p.add_argument(
        "--max-show",
        type=int,
        default=80,
        help="Maximum number of matched paths to print in the report (default: 80).",
    )
    p.add_argument(
        "--write",
        action="store_true",
        help="Apply deletions. Default is dry-run.",
    )
    args = p.parse_args(argv)
    if not args.tune_dir and not args.output_data_dir:
        p.error("Provide one of --tune-dir or --output-data-dir.")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        report = clear_slurm_runs(
            tune_dir=Path(args.tune_dir) if args.tune_dir else None,
            output_data_dir=Path(args.output_data_dir) if args.output_data_dir else None,
            prefix=args.prefix,
            dry_run=not bool(args.write),
        )
    except Exception as exc:
        print(f"[error] clear_slurm_runs failed: {exc}")
        return 2

    if args.tune_dir and args.output_data_dir:
        print("[warning] Both --tune-dir and --output-data-dir were provided; using --output-data-dir.")

    print_report(report, max_items=max(1, int(args.max_show)))
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
