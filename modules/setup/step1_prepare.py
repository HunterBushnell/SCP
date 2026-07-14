"""Step 1 tune-directory setup orchestration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from .cell_sources import setup_cell_source
from .defaults import (
    guess_cell_color,
    guess_soma_multiplier,
    guess_specimen_from_cell,
)
from .fit_json import (
    coerce_fit_genome_values_to_numeric,
    sort_genome_by_section,
)
from .mechanisms import compile_modfiles
from .scaffold import scaffold_base_configs, scaffold_synapse_configs
from .target_config import prepare_target_config as _prepare_target_config


def resolve_soma_multiplier(
    cell_name: str,
    soma_diam_multiplier: Optional[float],
) -> float:
    """Resolve an explicit soma diameter multiplier or the neutral default."""
    if soma_diam_multiplier is not None:
        return float(soma_diam_multiplier)
    return float(guess_soma_multiplier(cell_name))


def prepare_cell_source(
    *,
    tune_dir: Path,
    source_type: str = "adb",
    specimen_id: Optional[int] = None,
    model_type: str = "perisomatic",
    do_download: bool = True,
    force_download: bool = False,
    cache_stimulus: bool = False,
    download_match: str = "contains",
) -> Dict[str, Any]:
    """Phase 1: stage or acknowledge model source files."""
    return setup_cell_source(
        tune_dir=tune_dir,
        source_type=source_type,
        specimen_id=specimen_id,
        model_type=model_type,
        do_download=do_download,
        force_download=force_download,
        cache_stimulus=cache_stimulus,
        download_match=download_match,
    )


def prepare_mechanisms(
    *,
    tune_dir: Path,
    do_compile_modfiles: bool = True,
    recompile_modfiles: bool = False,
    load_compiled_dll: bool = True,
    coerce_genome_values_to_numeric: bool = False,
    sort_genome_entries_by_section: bool = False,
) -> Dict[str, Any]:
    """Phase 2: normalize fit JSON if requested, then compile/load modfiles."""
    tune_dir = Path(tune_dir).expanduser().resolve()
    actions: Dict[str, Any] = {}

    if coerce_genome_values_to_numeric:
        actions["coerce_genome_values"] = coerce_fit_genome_values_to_numeric(tune_dir)
    else:
        actions["coerce_genome_values"] = {"status": "skipped"}

    if sort_genome_entries_by_section:
        actions["sort_genome"] = sort_genome_by_section(tune_dir)
    else:
        actions["sort_genome"] = {"status": "skipped"}

    if do_compile_modfiles:
        compile_info = compile_modfiles(
            tune_dir,
            recompile=recompile_modfiles,
            load_dll=load_compiled_dll,
        )
        actions["compile_modfiles"] = {"status": "ok", **compile_info}
    else:
        actions["compile_modfiles"] = {"status": "skipped"}

    return actions


def prepare_base_configs(
    *,
    tune_dir: Path,
    cell_name: str,
    tune_name: str,
    specimen_id: Optional[int],
    model_type: str = "perisomatic",
    soma_diam_multiplier: Optional[float] = None,
    color: Optional[str] = None,
    config_mode: str = "fill",
    sync_cell_metadata: bool = True,
) -> Dict[str, Any]:
    """Phase 3: scaffold first-level cell, geometry, and simulation configs."""
    soma_mult = resolve_soma_multiplier(cell_name, soma_diam_multiplier)
    return scaffold_base_configs(
        tune_dir=tune_dir,
        cell_name=cell_name,
        tune_name=tune_name,
        specimen_id=specimen_id,
        model_type=model_type,
        soma_diam_multiplier=soma_mult,
        color=color,
        config_mode=config_mode,
        sync_cell_metadata=sync_cell_metadata,
    )


def prepare_target_config(
    *,
    tune_dir: Path,
    config_mode: str = "fill",
    target_source_mode: Optional[str] = "manual",
    target_description: Optional[str] = None,
    manual_passive: Optional[Dict[str, Any]] = None,
    manual_fi_curve: Optional[Dict[str, Any]] = None,
    passive_trace: Optional[Dict[str, Any]] = None,
    active_trace: Optional[Dict[str, Any]] = None,
    allen_nwb: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """Phase 4: scaffold optional passive/FI/trace target config."""
    return _prepare_target_config(
        tune_dir=tune_dir,
        config_mode=config_mode,
        target_source_mode=target_source_mode,
        target_description=target_description,
        manual_passive=manual_passive,
        manual_fi_curve=manual_fi_curve,
        passive_trace=passive_trace,
        active_trace=active_trace,
        allen_nwb=allen_nwb,
        notes=notes,
    )


def prepare_synapse_configs(
    *,
    tune_dir: Path,
    config_mode: str = "fill",
    template_kinds: Optional[list[str]] = None,
    weight_style: str = "distributed",
) -> Dict[str, Any]:
    """Phase 5: scaffold optional synapse config files."""
    return scaffold_synapse_configs(
        tune_dir=tune_dir,
        config_mode=config_mode,
        template_kinds=template_kinds,
        weight_style=weight_style,
    )


def validate_setup(
    *,
    tune_dir: Path,
    cell_name: str,
    soma_diam_multiplier: Optional[float] = None,
    validate_modfiles: bool = True,
    validate_load_cell: bool = True,
    validate_inputs_cfg: bool = True,
    validate_synapses: bool = True,
) -> Dict[str, Any]:
    """Phase 5: validate files, loader, optional mechanisms, and inputs."""
    from .validation import validate_tune

    soma_mult = resolve_soma_multiplier(cell_name, soma_diam_multiplier)
    return validate_tune(
        tune_dir=tune_dir,
        cell_name=cell_name,
        soma_diam_multiplier=soma_mult,
        validate_modfiles=validate_modfiles,
        validate_load_cell=validate_load_cell,
        validate_inputs=validate_inputs_cfg,
        validate_synapses=validate_synapses,
    )


def create_working_copy(
    *,
    source_tune_dir: Path,
    target_tune_name: str = "tuned",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Copy a validated setup tune to a sibling working tune directory."""
    source = Path(source_tune_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source tune directory not found: {source}")
    if not target_tune_name:
        raise ValueError("target_tune_name must be non-empty")
    if Path(target_tune_name).name != target_tune_name or target_tune_name in {".", ".."}:
        raise ValueError("target_tune_name must be a single sibling directory name")

    target = source.parent / target_tune_name
    if target == source:
        return {
            "status": "skipped",
            "reason": "target is the source tune",
            "source": str(source),
            "target": str(target),
        }
    if target.exists():
        if not overwrite:
            return {
                "status": "exists",
                "source": str(source),
                "target": str(target),
                "overwritten": False,
            }
        shutil.rmtree(target)

    shutil.copytree(source, target)

    cell_config_path = target / "cell_configs" / "cell_config.json"
    cell_config_updated = False
    if cell_config_path.is_file():
        with cell_config_path.open("r", encoding="utf-8") as handle:
            cell_config = json.load(handle)
        if not isinstance(cell_config, dict):
            raise ValueError(f"Expected JSON object in {cell_config_path}")
        cell_config["tune"] = target_tune_name
        with cell_config_path.open("w", encoding="utf-8") as handle:
            json.dump(cell_config, handle, indent=2)
            handle.write("\n")
        cell_config_updated = True

    return {
        "status": "created",
        "source": str(source),
        "target": str(target),
        "target_tune_name": target_tune_name,
        "overwritten": bool(overwrite),
        "cell_config_updated": cell_config_updated,
    }


def prepare_tune(
    *,
    tune_dir: Path,
    cell_name: str,
    tune_name: str,
    specimen_id: Optional[int] = None,
    model_type: str = "perisomatic",
    source_type: str = "adb",
    soma_diam_multiplier: Optional[float] = None,
    color: Optional[str] = None,
    do_download: bool = True,
    force_download: bool = False,
    cache_stimulus: bool = False,
    download_match: str = "contains",
    do_compile_modfiles: bool = True,
    recompile_modfiles: bool = False,
    load_compiled_dll: bool = True,
    coerce_genome_values_to_numeric: bool = False,
    sort_genome_entries_by_section: bool = False,
    do_scaffold_configs: bool = True,
    do_base_configs: Optional[bool] = None,
    do_target_config: Optional[bool] = None,
    do_synapse_configs: Optional[bool] = None,
    config_mode: str = "fill",
    sync_cell_metadata: bool = True,
    synapse_template_kinds: Optional[list[str]] = None,
    synapse_weight_style: str = "distributed",
    do_validate: bool = True,
    validate_inputs_cfg: bool = True,
    create_tuned_copy: bool = False,
    tuned_tune_name: str = "tuned",
    overwrite_tuned_copy: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end Step 1 entrypoint.

    The notebook uses the phase functions directly. The CLI keeps this combined
    function for one-command setup.
    """
    tune_dir = Path(tune_dir).expanduser().resolve()
    tune_dir.mkdir(parents=True, exist_ok=True)

    soma_mult = resolve_soma_multiplier(cell_name, soma_diam_multiplier)
    if do_base_configs is None:
        do_base_configs = bool(do_scaffold_configs)
    if do_target_config is None:
        do_target_config = bool(do_scaffold_configs)
    if do_synapse_configs is None:
        do_synapse_configs = bool(do_scaffold_configs)

    summary: Dict[str, Any] = {
        "tune_dir": str(tune_dir),
        "cell_name": cell_name,
        "tune_name": tune_name,
        "source_type": source_type,
        "specimen_id": None if specimen_id is None else int(specimen_id),
        "model_type": model_type,
        "soma_diam_multiplier": soma_mult,
        "actions": {},
    }

    summary["actions"]["cell_source"] = prepare_cell_source(
        tune_dir=tune_dir,
        source_type=source_type,
        specimen_id=specimen_id,
        model_type=model_type,
        do_download=do_download,
        force_download=force_download,
        cache_stimulus=cache_stimulus,
        download_match=download_match,
    )

    summary["actions"]["mechanisms"] = prepare_mechanisms(
        tune_dir=tune_dir,
        do_compile_modfiles=do_compile_modfiles,
        recompile_modfiles=recompile_modfiles,
        load_compiled_dll=load_compiled_dll,
        coerce_genome_values_to_numeric=coerce_genome_values_to_numeric,
        sort_genome_entries_by_section=sort_genome_entries_by_section,
    )

    if do_base_configs:
        summary["actions"]["base_configs"] = {
            "status": "ok",
            "files": prepare_base_configs(
                tune_dir=tune_dir,
                cell_name=cell_name,
                tune_name=tune_name,
                specimen_id=specimen_id,
                model_type=model_type,
                soma_diam_multiplier=soma_mult,
                color=color,
                config_mode=config_mode,
                sync_cell_metadata=sync_cell_metadata,
            ),
        }
    else:
        summary["actions"]["base_configs"] = {"status": "skipped"}

    if do_target_config:
        summary["actions"]["target_config"] = {
            "status": "ok",
            "file": prepare_target_config(
                tune_dir=tune_dir,
                config_mode=config_mode,
                target_source_mode=None,
            ),
        }
    else:
        summary["actions"]["target_config"] = {"status": "skipped"}

    if do_synapse_configs:
        summary["actions"]["synapse_configs"] = {
            "status": "ok",
            "files": prepare_synapse_configs(
                tune_dir=tune_dir,
                config_mode=config_mode,
                template_kinds=synapse_template_kinds,
                weight_style=synapse_weight_style,
            ),
        }
    else:
        summary["actions"]["synapse_configs"] = {"status": "skipped"}

    if do_validate:
        summary["actions"]["validate"] = {
            "status": "ok",
            "checks": validate_setup(
                tune_dir=tune_dir,
                cell_name=cell_name,
                soma_diam_multiplier=soma_mult,
                validate_modfiles=do_compile_modfiles,
                validate_load_cell=True,
                validate_inputs_cfg=validate_inputs_cfg,
                validate_synapses=bool(do_synapse_configs),
            ),
        }
    else:
        summary["actions"]["validate"] = {"status": "skipped"}

    if create_tuned_copy:
        summary["actions"]["working_copy"] = create_working_copy(
            source_tune_dir=tune_dir,
            target_tune_name=tuned_tune_name,
            overwrite=overwrite_tuned_copy,
        )
    else:
        summary["actions"]["working_copy"] = {"status": "skipped"}

    return summary
