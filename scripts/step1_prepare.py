#!/usr/bin/env python
"""
Step 1 CLI: set up a tune directory for later SCP steps.

Examples:
  python scripts/step1_prepare.py \
    --cell PV --tune adb_peri --specimen-id 484635029 --model-type perisomatic

  python scripts/step1_prepare.py \
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

from modules.setup.adb import list_ADB_models
from modules.setup.step1_prepare import (
    guess_cell_color,
    guess_soma_multiplier,
    guess_specimen_from_cell,
    prepare_tune,
)


DEFAULT_SYNAPSE_TEMPLATES = "input_blocks"


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


def _default_tune_for_model_type(model_type: str) -> str:
    """Return the raw setup tune name implied by an ADB model type."""
    token = str(model_type or "").strip().lower().replace("_", " ").replace("-", " ")
    if "all" in token and "active" in token:
        return "adb_all"
    return "adb_peri"


def _parse_synapse_templates(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text or text.lower() in {"none", "off", "false"}:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Set up an SCP Step-1 tune directory")

    loc = ap.add_mutually_exclusive_group(required=False)
    loc.add_argument(
        "--tune-dir",
        type=str,
        default=None,
        help="Tune directory path (e.g., cells/PV/tunes/seg_tuned)",
    )

    ap.add_argument("--cell", type=str, default=None, help="Cell label (PV or SST)")
    ap.add_argument(
        "--tune",
        type=str,
        default=None,
        help=(
            "Tune name under cells/<cell>/tunes/. Defaults to adb_peri for "
            "perisomatic models and adb_all for all-active models."
        ),
    )
    ap.add_argument("--tunes-dir", type=str, default="tunes", help="Parent directory for tune names")

    ap.add_argument(
        "--source-type",
        choices=["adb", "existing"],
        default="adb",
        help="Cell source adapter. Use 'adb' to download Allen Database files or 'existing' for staged local files.",
    )
    ap.add_argument("--specimen-id", type=int, default=None, help="Allen specimen_id for --source-type adb")
    ap.add_argument("--model-type", type=str, default="perisomatic", help="Allen model type for --source-type adb")
    ap.add_argument("--soma-diam-multiplier", type=float, default=None, help="Soma diameter multiplier")
    ap.add_argument("--color", type=str, default=None, help="cell_config color field")

    ap.add_argument("--list-models", action="store_true", help="Print available ADB models before setup")
    ap.add_argument("--list-models-only", action="store_true", help="Only list available models and exit")

    ap.add_argument("--no-download", dest="do_download", action="store_false", help="Skip ADB bundle download")
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

    ap.add_argument("--no-scaffold", dest="do_scaffold", action="store_false", help="Skip all config scaffolding")
    ap.add_argument(
        "--no-base-configs",
        dest="do_base_configs",
        action="store_false",
        help="Skip cell_config.json, geometry.json, and sim_config.json scaffolding",
    )
    ap.add_argument(
        "--no-synapse-configs",
        dest="do_synapse_configs",
        action="store_false",
        help="Skip optional syn_config.json and syn_groups/ scaffolding",
    )
    ap.add_argument(
        "--synapse-templates",
        type=str,
        default=DEFAULT_SYNAPSE_TEMPLATES,
        help=(
            "Comma-separated disabled synapse templates to scaffold: "
            "input_blocks; use 'none' for an empty syn_config."
        ),
    )
    ap.add_argument(
        "--synapse-weight-style",
        choices=["distributed", "fixed"],
        default="distributed",
        help="Weight fields used in generated synapse templates.",
    )
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
        do_base_configs=True,
        do_synapse_configs=True,
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
        tune_name = args.tune or _default_tune_for_model_type(args.model_type)
        tune_dir = (REPO_ROOT / "cells" / cell_name / args.tunes_dir / tune_name).resolve()

    specimen_id = args.specimen_id
    if specimen_id is None and args.source_type == "adb":
        specimen_id = guess_specimen_from_cell(cell_name)
    if specimen_id is None and args.source_type == "adb":
        raise ValueError(
            "Could not infer specimen_id for ADB setup. Provide --specimen-id explicitly."
        )

    if args.list_models or args.list_models_only:
        if specimen_id is None:
            raise ValueError("--list-models requires --source-type adb and a specimen_id.")
        list_ADB_models(specimen_id, filter_type=args.model_type)
        if args.list_models_only:
            return

    soma_mult = args.soma_diam_multiplier
    if soma_mult is None:
        soma_mult = guess_soma_multiplier(cell_name)

    color = args.color if args.color is not None else guess_cell_color(cell_name)
    synapse_template_kinds = _parse_synapse_templates(args.synapse_templates)

    summary = prepare_tune(
        tune_dir=tune_dir,
        cell_name=cell_name,
        tune_name=tune_name,
        specimen_id=None if specimen_id is None else int(specimen_id),
        source_type=args.source_type,
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
        do_base_configs=bool(args.do_scaffold and args.do_base_configs),
        do_synapse_configs=bool(args.do_scaffold and args.do_synapse_configs),
        config_mode=args.config_mode,
        sync_cell_metadata=bool(args.sync_cell_metadata),
        synapse_template_kinds=synapse_template_kinds,
        synapse_weight_style=args.synapse_weight_style,
        do_validate=bool(args.do_validate),
        validate_inputs_cfg=bool(args.validate_inputs_cfg),
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
