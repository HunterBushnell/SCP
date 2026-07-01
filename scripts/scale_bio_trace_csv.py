#!/usr/bin/env python3
"""
Scale and vertically shift biotrace CSV values.

By default, this transforms any of the columns that exist:
  - value
  - value_low
  - value_high

Formula options:
  - scale-then-offset: y' = y * scalar + offset
  - offset-then-scale: y' = (y + offset) * scalar

Usage:
  python scripts/scale_bio_trace_csv.py \
    --input path/to/biotrace.csv \
    --scalar 1.2 \
    --offset -0.3 \
    --order scale-then-offset
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_COLUMNS = ("value", "value_low", "value_high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply scalar/offset transforms to numeric columns in a biotrace CSV."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: <input_stem>_scaled.csv next to input).",
    )
    parser.add_argument(
        "--scalar",
        type=float,
        default=1.0,
        help="Multiplicative scale factor (default: 1.0).",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Additive vertical shift value (default: 0.0).",
    )
    parser.add_argument(
        "--order",
        choices=("scale-then-offset", "offset-then-scale"),
        default="scale-then-offset",
        help="Whether scaling or offset is applied first.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=list(DEFAULT_COLUMNS),
        help="Columns to transform (default: value value_low value_high).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output CSV if it exists.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing columns or non-numeric non-empty cells.",
    )
    return parser.parse_args()


def resolve_output_path(input_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg).expanduser().resolve()
    suffix = input_path.suffix if input_path.suffix else ".csv"
    return input_path.with_name(f"{input_path.stem}_scaled{suffix}")


def apply_transform(value: float, scalar: float, offset: float, order: str) -> float:
    if order == "scale-then-offset":
        return value * scalar + offset
    return (value + offset) * scalar


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.15g}"


def main() -> int:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path = resolve_output_path(input_path, args.out)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output CSV already exists: {output_path} (pass --overwrite to replace it)"
        )

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {input_path}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    missing_columns = [col for col in args.columns if col not in fieldnames]
    if missing_columns and args.strict:
        raise ValueError(
            f"Requested columns missing from CSV: {missing_columns}; available columns: {fieldnames}"
        )
    active_columns = [col for col in args.columns if col in fieldnames]
    if not active_columns:
        raise ValueError(
            f"No requested columns were found. Requested: {args.columns}; available: {fieldnames}"
        )

    stats = {
        col: {"transformed": 0, "blank": 0, "non_numeric": 0} for col in active_columns
    }
    non_numeric_errors: list[str] = []

    for row_idx, row in enumerate(rows, start=2):
        for col in active_columns:
            raw = row.get(col, "")
            raw_str = str(raw).strip() if raw is not None else ""
            if raw_str == "":
                stats[col]["blank"] += 1
                continue
            try:
                value = float(raw_str)
            except ValueError:
                stats[col]["non_numeric"] += 1
                if args.strict:
                    non_numeric_errors.append(
                        f"row {row_idx}, column {col!r}, value {raw!r}"
                    )
                continue

            transformed = apply_transform(value, args.scalar, args.offset, args.order)
            row[col] = format_float(transformed)
            stats[col]["transformed"] += 1

    if non_numeric_errors:
        details = "; ".join(non_numeric_errors[:8])
        if len(non_numeric_errors) > 8:
            details += f"; ... ({len(non_numeric_errors)} total)"
        raise ValueError(
            "Found non-numeric values in --strict mode; no output written: "
            + details
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote transformed CSV: {output_path}")
    print(
        "Applied formula: y' = y * scalar + offset"
        if args.order == "scale-then-offset"
        else "Applied formula: y' = (y + offset) * scalar"
    )

    if missing_columns:
        print(
            "Skipped missing columns (use --strict to fail instead): "
            + ", ".join(missing_columns)
        )
    for col in active_columns:
        col_stats = stats[col]
        print(
            f"{col}: transformed={col_stats['transformed']}, "
            f"blank={col_stats['blank']}, non_numeric={col_stats['non_numeric']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
