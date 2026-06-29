#!/usr/bin/env python
"""
Step 0 CLI: prepare a tune directory for SCP Steps 1-6.

Examples:
  python scripts/step0_prepare.py \
    --cell PV --tune seg_tuned --specimen-id 484635029 --model-type perisomatic

  python scripts/step0_prepare.py \
    --tune-dir cells/SST/tunes/seg_tuned --specimen-id 485466109 \
    --no-download --no-compile --config-mode fill
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules import download_cell
from modules.step0_prepare import (
    guess_cell_color,
    guess_soma_multiplier,
    guess_specimen_from_cell,
    prepare_tune,
)


def _infer_cell_tune_from_path(tune_dir: Path) -> tuple[str, str]:
    """Best-effort infer (cell_name, tune_name) from .../cells/<cell>/tunes/<tune>."""
    parts = tune_dir.resolve().parts
    cell_name = "UNKNOWN"
    tune_name = tune_dir.name
    if "cells" in parts and "tunes" in parts:
        i_cells = parts.index("cells")
        i_tunes = parts.index("tunes")
        if i_cells + 1 < len(parts):
            cell_name = parts[i_cells + 1]
        if i_tunes + 1 < len(parts):
            tune_name = parts[i_tunes + 1]
    return cell_name, tune_name


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare SCP Step-0 tune directory")

    loc = ap.add_mutually_exclusive_group(required=False)
    loc.add_argument(
        "--tune-dir",
        type=str,
        default=None,
        help="Tune directory path (e.g., cells/PV/tunes/seg_tuned)",
    )

    ap.add_argument("--cell", type=str, default=None, help="Cell label (PV or SST)")
    ap.add_argument("--tune", type=str, default="seg_tuned", help="Tune name under cells/<cell>/tunes/")
    ap.add_argument("--tunes-dir", type=str, default="tunes", help="Parent directory for tune names")

    ap.add_argument("--specimen-id", type=int, default=None, help="Allen specimen_id")
    ap.add_argument("--model-type", type=str, default="perisomatic", help="Allen model type")
    ap.add_argument("--soma-diam-multiplier", type=float, default=None, help="Soma diameter multiplier")
    ap.add_argument("--color", type=str, default=None, help="cell_config color field")

    ap.add_argument("--list-models", action="store_true", help="Print available Allen models before prepare")
    ap.add_argument("--list-models-only", action="store_true", help="Only list available models and exit")

    ap.add_argument("--no-download", dest="do_download", action="store_false", help="Skip Allen download")
    ap.add_argument("--force-download", action="store_true", help="Force cache_data even if target has files")
    ap.add_argument("--cache-stimulus", action="store_true", help="Allow large NWB stimulus cache")

    ap.add_argument("--no-compile", dest="do_compile", action="store_false", help="Skip nrnivmodl compile")
    ap.add_argument("--recompile-modfiles", action="store_true", help="Delete modfiles/x86_64 before compile")
    ap.add_argument("--no-load-dll", dest="load_dll", action="store_false", help="Do not load compiled DLL")
    ap.add_argument(
        "--sort-genome-by-section",
        action="store_true",
        help="Reorder fit JSON genome entries by section before validation",
    )

    ap.add_argument("--no-scaffold", dest="do_scaffold", action="store_false", help="Skip config scaffold")
    ap.add_argument(
        "--config-mode",
        choices=["fill", "overwrite", "skip"],
        default="fill",
        help="How to handle existing config files",
    )
    ap.add_argument(
        "--no-sync-cell-metadata",
        dest="sync_cell_metadata",
        action="store_false",
        help="Do not force specimen/model/soma fields in cell_config.json",
    )

    ap.add_argument("--no-validate", dest="do_validate", action="store_false", help="Skip validation checks")
    ap.add_argument(
        "--no-validate-inputs",
        dest="validate_inputs_cfg",
        action="store_false",
        help="Skip inputs.check_inputs validation",
    )

    ap.set_defaults(
        do_download=True,
        do_compile=True,
        load_dll=True,
        do_scaffold=True,
        sync_cell_metadata=True,
        do_validate=True,
        validate_inputs_cfg=True,
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.tune_dir:
        tune_dir = Path(args.tune_dir).expanduser().resolve()
        inferred_cell, inferred_tune = _infer_cell_tune_from_path(tune_dir)
        cell_name = args.cell or inferred_cell
        tune_name = args.tune or inferred_tune
    else:
        if not args.cell:
            raise ValueError("Provide --cell when --tune-dir is not used")
        cell_name = args.cell
        tune_name = args.tune
        tune_dir = (REPO_ROOT / "cells" / cell_name / args.tunes_dir / tune_name).resolve()

    specimen_id = args.specimen_id
    if specimen_id is None:
        specimen_id = guess_specimen_from_cell(cell_name)
    if specimen_id is None:
        raise ValueError(
            "Could not infer specimen_id. Provide --specimen-id explicitly for this cell."
        )

    if args.list_models or args.list_models_only:
        download_cell.list_ADB_models(specimen_id, filter_type=args.model_type)
        if args.list_models_only:
            return

    soma_mult = args.soma_diam_multiplier
    if soma_mult is None:
        soma_mult = guess_soma_multiplier(cell_name)

    color = args.color if args.color is not None else guess_cell_color(cell_name)

    summary = prepare_tune(
        tune_dir=tune_dir,
        cell_name=cell_name,
        tune_name=tune_name,
        specimen_id=int(specimen_id),
        model_type=args.model_type,
        soma_diam_multiplier=float(soma_mult),
        color=color,
        do_download=bool(args.do_download),
        force_download=bool(args.force_download),
        cache_stimulus=bool(args.cache_stimulus),
        do_compile_modfiles=bool(args.do_compile),
        recompile_modfiles=bool(args.recompile_modfiles),
        load_compiled_dll=bool(args.load_dll),
        sort_genome_entries_by_section=bool(args.sort_genome_by_section),
        do_scaffold_configs=bool(args.do_scaffold),
        config_mode=args.config_mode,
        sync_cell_metadata=bool(args.sync_cell_metadata),
        do_validate=bool(args.do_validate),
        validate_inputs_cfg=bool(args.validate_inputs_cfg),
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
