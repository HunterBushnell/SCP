#!/usr/bin/env python
"""
Step 1 CLI: set up a tune directory for later SCP steps.

Examples:
  python scripts/step1_prepare.py \
    --cell PV --tune orig --specimen-id 484635029 --model-type perisomatic

  python scripts/step1_prepare.py \
    --tune-dir cells/SST/tunes/tuned --specimen-id 485466109 \
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
from modules.loaders import get_cell_loader_name
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
    """Return the default raw setup tune name."""
    return "orig"


def _parse_synapse_templates(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text or text.lower() in {"none", "off", "false"}:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_json_value(raw: str | None, *, option: str, expected_type: type):
    if raw in (None, ""):
        return expected_type()
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option} must be valid JSON: {exc}") from exc
    if not isinstance(value, expected_type):
        raise ValueError(f"{option} must decode to a JSON {expected_type.__name__}")
    return value


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Set up an SCP Step-1 tune directory")

    loc = ap.add_mutually_exclusive_group(required=False)
    loc.add_argument(
        "--tune-dir",
        type=str,
        default=None,
        help="Tune directory path (e.g., cells/PV/tunes/orig)",
    )

    ap.add_argument(
        "--cell",
        type=str,
        default=None,
        help="Cell/model label used for tune paths and display metadata",
    )
    ap.add_argument(
        "--tune",
        type=str,
        default=None,
        help=(
            "Tune name under cells/<cell>/tunes/. Defaults to orig for raw setup."
        ),
    )
    ap.add_argument("--tunes-dir", type=str, default="tunes", help="Parent directory for tune names")

    ap.add_argument(
        "--source-type",
        choices=["adb", "existing"],
        default=None,
        help=(
            "Cell source adapter. Defaults to 'adb' for the Allen loader and "
            "'existing' for other loaders."
        ),
    )
    ap.add_argument(
        "--cell-loader",
        default=None,
        help="Registered model loader (default: allen_manifest, or hoc_template when HOC options are used).",
    )
    ap.add_argument(
        "--loader-paths-json",
        default=None,
        help='Additional cell_config paths as a JSON object, e.g. \'{"model":"model.json"}\'.',
    )
    ap.add_argument(
        "--loader-config-json",
        default=None,
        help="Additional loader-specific top-level cell_config blocks as a JSON object.",
    )
    ap.add_argument(
        "--hoc-template-file",
        "--hoc-file",
        dest="hoc_template_file",
        default=None,
        help="HOC template path, absolute or relative to the tune directory.",
    )
    ap.add_argument("--hoc-template-name", default=None, help="HOC template class name.")
    ap.add_argument(
        "--hoc-constructor-args",
        default=None,
        help='HOC constructor arguments as a JSON list (default: "[]").',
    )
    ap.add_argument(
        "--hoc-section-map",
        default=None,
        help=(
            "Canonical HOC section mapping as a JSON object. Keys may be "
            "soma/dend/apic/axon/all; values are owner attributes, lists, or null."
        ),
    )
    ap.add_argument("--specimen-id", type=int, default=None, help="Allen specimen_id for --source-type adb")
    ap.add_argument("--model-type", type=str, default="perisomatic", help="Allen model type for --source-type adb")
    ap.add_argument(
        "--soma-diam-multiplier",
        type=float,
        default=None,
        help="Allen-loader soma diameter multiplier; omit for neutral 1.0 default",
    )
    ap.add_argument("--color", type=str, default=None, help="cell_config color field")
    ap.add_argument(
        "--v-init-mv",
        type=float,
        default=None,
        help="Explicit simulation initialization voltage (required for new HOC tunes)",
    )
    ap.add_argument(
        "--celsius-c",
        type=float,
        default=None,
        help="Explicit simulation temperature (required for new HOC tunes)",
    )

    ap.add_argument("--list-models", action="store_true", help="Print available ADB models before setup")
    ap.add_argument("--list-models-only", action="store_true", help="Only list available models and exit")

    ap.add_argument("--no-download", dest="do_download", action="store_false", help="Skip ADB bundle download")
    ap.add_argument("--force-download", action="store_true", help="Force cache_data even if target has files")
    ap.add_argument("--cache-stimulus", action="store_true", help="Allow large NWB stimulus cache")

    ap.add_argument("--no-compile", dest="do_compile", action="store_false", help="Skip nrnivmodl compile")
    ap.add_argument(
        "--recompile-modfiles",
        action="store_true",
        help="Delete x86_64 under the configured paths.modfiles directory before compiling",
    )
    ap.add_argument("--no-load-dll", dest="load_dll", action="store_false", help="Do not load compiled DLL")
    mod_requirement = ap.add_mutually_exclusive_group()
    mod_requirement.add_argument(
        "--allow-missing-modfiles",
        dest="allow_missing_modfiles",
        action="store_true",
        help="Allow models that use only built-in NEURON mechanisms.",
    )
    mod_requirement.add_argument(
        "--require-modfiles",
        dest="allow_missing_modfiles",
        action="store_false",
        help="Require configured .mod sources and a compiled mechanism library.",
    )
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
        "--no-target-config",
        dest="do_target_config",
        action="store_false",
        help="Skip target_config.json scaffolding",
    )
    ap.add_argument(
        "--target-source-mode",
        choices=["none", "manual", "traces", "allen_nwb"],
        default=None,
        help="Target source to scaffold. Defaults to manual for Allen and none for other loaders.",
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
        help="Do not sync cell/tune/loader/path metadata in cell_config.json",
    )

    ap.add_argument("--no-validate", dest="do_validate", action="store_false", help="Skip validation checks")
    ap.add_argument(
        "--no-validate-inputs",
        dest="validate_inputs_cfg",
        action="store_false",
        help="Skip inputs.check_inputs validation",
    )
    ap.add_argument(
        "--create-tuned-copy",
        action="store_true",
        help="Copy the prepared tune to a sibling working tune after validation.",
    )
    ap.add_argument(
        "--tuned-tune-name",
        default="tuned",
        help="Sibling working tune name for --create-tuned-copy.",
    )
    ap.add_argument(
        "--overwrite-tuned-copy",
        action="store_true",
        help="Replace an existing tuned working copy when --create-tuned-copy is used.",
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
        allow_missing_modfiles=None,
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    hoc_options_used = any(
        value not in (None, "")
        for value in (
            args.hoc_template_file,
            args.hoc_template_name,
            args.hoc_constructor_args,
            args.hoc_section_map,
        )
    )
    cell_loader = get_cell_loader_name(
        {"cell_loader": args.cell_loader or ("hoc_template" if hoc_options_used else "allen_manifest")}
    )
    source_type = args.source_type or ("adb" if cell_loader == "allen_manifest" else "existing")
    if hoc_options_used and cell_loader != "hoc_template":
        raise ValueError("--hoc-* options require --cell-loader hoc_template.")
    if source_type == "adb" and cell_loader != "allen_manifest":
        raise ValueError("--source-type adb can only be used with --cell-loader allen_manifest.")

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
    if specimen_id is None and source_type == "adb":
        specimen_id = guess_specimen_from_cell(cell_name)
    if specimen_id is None and source_type == "adb":
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
    if soma_mult is None and cell_loader == "allen_manifest":
        soma_mult = guess_soma_multiplier(cell_name)

    color = args.color if args.color is not None else guess_cell_color(cell_name)
    synapse_template_kinds = _parse_synapse_templates(args.synapse_templates)
    loader_paths = _parse_json_value(
        args.loader_paths_json,
        option="--loader-paths-json",
        expected_type=dict,
    )
    loader_config = _parse_json_value(
        args.loader_config_json,
        option="--loader-config-json",
        expected_type=dict,
    )
    if cell_loader == "hoc_template":
        existing_hoc_config = loader_config.get("hoc_template")
        if existing_hoc_config is not None and not isinstance(existing_hoc_config, dict):
            raise ValueError("loader_config.hoc_template must be a JSON object")
        if args.hoc_template_file not in (None, ""):
            loader_paths["hoc_template"] = str(args.hoc_template_file)
        if args.hoc_template_name not in (None, ""):
            hoc_config = loader_config.setdefault("hoc_template", {})
            hoc_config["template_name"] = str(args.hoc_template_name)
        if args.hoc_constructor_args not in (None, ""):
            hoc_config = loader_config.setdefault("hoc_template", {})
            hoc_config["constructor_args"] = _parse_json_value(
                args.hoc_constructor_args,
                option="--hoc-constructor-args",
                expected_type=list,
            )
        if args.hoc_section_map not in (None, ""):
            hoc_config = loader_config.setdefault("hoc_template", {})
            hoc_config["section_map"] = _parse_json_value(
                args.hoc_section_map,
                option="--hoc-section-map",
                expected_type=dict,
            )

    summary = prepare_tune(
        tune_dir=tune_dir,
        cell_name=cell_name,
        tune_name=tune_name,
        specimen_id=None if specimen_id is None else int(specimen_id),
        source_type=source_type,
        cell_loader=cell_loader,
        loader_paths=loader_paths,
        loader_config=loader_config,
        model_type=args.model_type,
        soma_diam_multiplier=None if soma_mult is None else float(soma_mult),
        color=color,
        do_download=bool(args.do_download),
        force_download=bool(args.force_download),
        cache_stimulus=bool(args.cache_stimulus),
        do_compile_modfiles=bool(args.do_compile),
        recompile_modfiles=bool(args.recompile_modfiles),
        load_compiled_dll=bool(args.load_dll),
        allow_missing_modfiles=args.allow_missing_modfiles,
        sort_genome_entries_by_section=bool(args.sort_genome_by_section),
        do_scaffold_configs=bool(args.do_scaffold),
        do_base_configs=bool(args.do_scaffold and args.do_base_configs),
        do_target_config=bool(args.do_scaffold and args.do_target_config),
        do_synapse_configs=bool(args.do_scaffold and args.do_synapse_configs),
        config_mode=args.config_mode,
        sync_cell_metadata=bool(args.sync_cell_metadata),
        v_init_mV=args.v_init_mv,
        celsius_C=args.celsius_c,
        target_source_mode=args.target_source_mode,
        synapse_template_kinds=synapse_template_kinds,
        synapse_weight_style=args.synapse_weight_style,
        do_validate=bool(args.do_validate),
        validate_inputs_cfg=bool(args.validate_inputs_cfg),
        create_tuned_copy=bool(args.create_tuned_copy),
        tuned_tune_name=args.tuned_tune_name,
        overwrite_tuned_copy=bool(args.overwrite_tuned_copy),
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
