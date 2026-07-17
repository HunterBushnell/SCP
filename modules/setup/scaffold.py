"""Create or update standard SCP config files in a tune directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import copy

from modules.loaders import get_cell_loader_name

from .defaults import (
    default_cell_config,
    default_geometry_config,
    default_sim_config,
    default_syn_config_for_templates,
    default_syn_group_templates,
    guess_cell_color,
)
from .json_utils import _write_json, _write_scaffold_json
from .paths import resolve_step1_paths
from .target_config import prepare_target_config


def _deep_override(existing: Any, updates: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply explicit nested loader overrides without deleting sibling fields."""
    merged = copy.deepcopy(existing) if isinstance(existing, dict) else {}
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_override(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def scaffold_base_configs(
    *,
    tune_dir: Path,
    cell_name: str,
    tune_name: str,
    specimen_id: Optional[int],
    model_type: str,
    soma_diam_multiplier: Optional[float] = None,
    color: Optional[str] = None,
    config_mode: str = "fill",
    sync_cell_metadata: bool = True,
    cell_loader: Optional[str] = None,
    loader_paths: Optional[Mapping[str, Any]] = None,
    loader_config: Optional[Mapping[str, Any]] = None,
    v_init_mV: Optional[float] = None,
    celsius_C: Optional[float] = None,
) -> Dict[str, Any]:
    """Ensure first-level config files exist under cell_configs/."""
    paths = resolve_step1_paths(tune_dir)
    paths.config_dir.mkdir(parents=True, exist_ok=True)

    statuses: Dict[str, Any] = {}

    cell_cfg_defaults = default_cell_config(
        cell_name=cell_name,
        tune_name=tune_name,
        specimen_id=specimen_id,
        model_type=model_type,
        soma_diam_multiplier=soma_diam_multiplier,
        color=color,
        cell_loader=cell_loader or "allen_manifest",
        loader_paths=loader_paths,
        loader_config=loader_config,
    )
    cell_cfg_path = paths.config_dir / "cell_config.json"
    status, cell_cfg_data = _write_scaffold_json(cell_cfg_path, cell_cfg_defaults, config_mode)

    if sync_cell_metadata:
        before_sync = copy.deepcopy(cell_cfg_data)
        cell_cfg_data.setdefault("paths", {})
        cell_cfg_data["cell_name"] = cell_name
        cell_cfg_data["tune"] = tune_name
        resolved_loader = (
            get_cell_loader_name({"cell_loader": cell_loader})
            if cell_loader not in (None, "")
            else get_cell_loader_name(cell_cfg_data)
        )
        cell_cfg_data["cell_loader"] = resolved_loader
        # Canonical model identity is the tune directory itself.
        cell_cfg_data.pop("specimen_id", None)
        cell_cfg_data.pop("model_type", None)
        if color is not None or "color" not in cell_cfg_data:
            cell_cfg_data["color"] = color if color is not None else guess_cell_color(cell_name)
        if resolved_loader == "allen_manifest":
            cell_cfg_data["paths"].setdefault("manifest", "manifest.json")
        elif resolved_loader == "hoc_template":
            cell_cfg_data["paths"].pop("manifest", None)
        if loader_paths:
            cell_cfg_data["paths"].update(dict(loader_paths))
        if loader_config:
            for key, value in loader_config.items():
                if key in {"cell_name", "tune", "color", "cell_loader", "paths", "tuning"}:
                    raise ValueError(
                        f"loader_config cannot replace reserved cell-config key {key!r}"
                    )
                config_key = str(key)
                if isinstance(value, Mapping):
                    cell_cfg_data[config_key] = _deep_override(
                        cell_cfg_data.get(config_key),
                        value,
                    )
                else:
                    cell_cfg_data[config_key] = copy.deepcopy(value)
        if resolved_loader == "allen_manifest":
            tuning = cell_cfg_data.setdefault("tuning", {})
            if not isinstance(tuning, dict):
                raise TypeError("cell_config.json tuning must be an object when provided.")
            multiplier = 1.0 if soma_diam_multiplier is None else float(soma_diam_multiplier)
            tuning["soma_diam_multiplier"] = multiplier
        if cell_cfg_data != before_sync:
            _write_json(cell_cfg_path, cell_cfg_data)
            if status == "unchanged":
                status = "updated"

    statuses["cell_config"] = {
        "path": str(cell_cfg_path),
        "status": status,
    }

    sim_cfg_defaults = default_sim_config(
        cell_name=cell_name,
        specimen_id=specimen_id,
        model_type=model_type,
        v_init_mV=v_init_mV,
        celsius_C=celsius_C,
    )
    sim_cfg_path = paths.config_dir / "sim_config.json"
    sim_status, sim_cfg_data = _write_scaffold_json(sim_cfg_path, sim_cfg_defaults, config_mode)
    # Keep sim config focused on simulation controls; remove identity metadata.
    if isinstance(sim_cfg_data, dict):
        before_sim = copy.deepcopy(sim_cfg_data)
        sim_cfg_data.pop("specimen_id", None)
        sim_cfg_data.pop("model_type", None)
        sim_cfg_data.pop("soma_diam_multiplier", None)
        conditions = sim_cfg_data.setdefault("conditions", {})
        if not isinstance(conditions, dict):
            conditions = {}
            sim_cfg_data["conditions"] = conditions
        if v_init_mV is not None:
            conditions["v_init_mV"] = float(v_init_mV)
        else:
            conditions.setdefault("v_init_mV", None)
        if celsius_C is not None:
            conditions["celsius_C"] = float(celsius_C)
        else:
            conditions.setdefault("celsius_C", None)
        if sim_cfg_data != before_sim:
            _write_json(sim_cfg_path, sim_cfg_data)
            if sim_status == "unchanged":
                sim_status = "updated"
    statuses["sim_config"] = {
        "path": str(sim_cfg_path),
        "status": sim_status,
    }

    geom_cfg_defaults = default_geometry_config(cell_name=cell_name)
    geom_cfg_path = paths.config_dir / "geometry.json"
    geom_status, _ = _write_scaffold_json(geom_cfg_path, geom_cfg_defaults, config_mode)
    statuses["geometry"] = {
        "path": str(geom_cfg_path),
        "status": geom_status,
    }

    return statuses


def scaffold_synapse_configs(
    *,
    tune_dir: Path,
    config_mode: str = "fill",
    template_kinds: Optional[list[str]] = None,
    weight_style: str = "distributed",
) -> Dict[str, Any]:
    """Ensure optional synapse config files exist under cell_configs/."""
    paths = resolve_step1_paths(tune_dir)
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.syn_groups_dir.mkdir(parents=True, exist_ok=True)

    statuses: Dict[str, Any] = {}

    template_payloads = default_syn_group_templates(
        template_kinds=template_kinds,
        weight_style=weight_style,
    )

    for filename, payload in template_payloads.items():
        template_path = paths.syn_groups_dir / filename
        template_status, _ = _write_scaffold_json(template_path, payload, config_mode)
        statuses[f"syn_group_{template_path.stem}"] = {
            "path": str(template_path),
            "status": template_status,
        }

    if template_payloads:
        syn_cfg_defaults = default_syn_config_for_templates(filenames=template_payloads.keys())
    else:
        syn_cfg_defaults = {"group_files": []}

    syn_cfg_path = paths.config_dir / "syn_config.json"
    syn_status, _ = _write_scaffold_json(syn_cfg_path, syn_cfg_defaults, config_mode)
    statuses["syn_config"] = {
        "path": str(syn_cfg_path),
        "status": syn_status,
    }

    return statuses


def scaffold_common_configs(
    *,
    tune_dir: Path,
    cell_name: str,
    tune_name: str,
    specimen_id: Optional[int],
    model_type: str,
    soma_diam_multiplier: Optional[float] = None,
    color: Optional[str] = None,
    config_mode: str = "fill",
    sync_cell_metadata: bool = True,
    include_synapses: bool = True,
    include_target_config: bool = True,
    synapse_template_kinds: Optional[list[str]] = None,
    synapse_weight_style: str = "distributed",
    cell_loader: Optional[str] = None,
    loader_paths: Optional[Mapping[str, Any]] = None,
    loader_config: Optional[Mapping[str, Any]] = None,
    v_init_mV: Optional[float] = None,
    celsius_C: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Ensure standard config files exist under cell_configs/.

    config_mode:
      - fill: create missing files and fill missing keys in existing files
      - overwrite: replace existing files with defaults
      - skip: do not modify existing files
    """
    statuses = scaffold_base_configs(
        tune_dir=tune_dir,
        cell_name=cell_name,
        tune_name=tune_name,
        specimen_id=specimen_id,
        model_type=model_type,
        soma_diam_multiplier=soma_diam_multiplier,
        color=color,
        config_mode=config_mode,
        sync_cell_metadata=sync_cell_metadata,
        cell_loader=cell_loader,
        loader_paths=loader_paths,
        loader_config=loader_config,
        v_init_mV=v_init_mV,
        celsius_C=celsius_C,
    )
    if include_target_config:
        resolved_loader = get_cell_loader_name(
            {"cell_loader": cell_loader or "allen_manifest"}
        )
        statuses["target_config"] = prepare_target_config(
            tune_dir=tune_dir,
            config_mode=config_mode,
            target_source_mode=(
                "manual" if resolved_loader == "allen_manifest" else "none"
            ),
        )
    if include_synapses:
        statuses.update(
            scaffold_synapse_configs(
                tune_dir=tune_dir,
                config_mode=config_mode,
                template_kinds=synapse_template_kinds,
                weight_style=synapse_weight_style,
            )
        )
    return statuses
