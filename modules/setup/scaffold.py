"""Create or update standard SCP config files in a tune directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import copy

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


def scaffold_base_configs(
    *,
    tune_dir: Path,
    cell_name: str,
    tune_name: str,
    specimen_id: Optional[int],
    model_type: str,
    soma_diam_multiplier: float,
    color: Optional[str] = None,
    config_mode: str = "fill",
    sync_cell_metadata: bool = True,
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
    )
    cell_cfg_path = paths.config_dir / "cell_config.json"
    status, cell_cfg_data = _write_scaffold_json(cell_cfg_path, cell_cfg_defaults, config_mode)

    if sync_cell_metadata:
        before_sync = copy.deepcopy(cell_cfg_data)
        cell_cfg_data.setdefault("paths", {})
        cell_cfg_data["cell_name"] = cell_name
        cell_cfg_data["tune"] = tune_name
        cell_cfg_data.setdefault("cell_loader", "allen_manifest")
        # Canonical model identity is the tune directory itself.
        cell_cfg_data.pop("specimen_id", None)
        cell_cfg_data.pop("model_type", None)
        if color is not None or "color" not in cell_cfg_data:
            cell_cfg_data["color"] = color if color is not None else guess_cell_color(cell_name)
        cell_cfg_data["paths"]["manifest"] = "manifest.json"
        cell_cfg_data.setdefault("tuning", {})
        cell_cfg_data["tuning"]["soma_diam_multiplier"] = float(soma_diam_multiplier)
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
    )
    sim_cfg_path = paths.config_dir / "sim_config.json"
    sim_status, sim_cfg_data = _write_scaffold_json(sim_cfg_path, sim_cfg_defaults, config_mode)
    # Keep sim config focused on simulation controls; remove identity metadata.
    if isinstance(sim_cfg_data, dict):
        before_sim = copy.deepcopy(sim_cfg_data)
        sim_cfg_data.pop("specimen_id", None)
        sim_cfg_data.pop("model_type", None)
        sim_cfg_data.pop("soma_diam_multiplier", None)
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
    soma_diam_multiplier: float,
    color: Optional[str] = None,
    config_mode: str = "fill",
    sync_cell_metadata: bool = True,
    include_synapses: bool = True,
    include_target_config: bool = True,
    synapse_template_kinds: Optional[list[str]] = None,
    synapse_weight_style: str = "distributed",
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
    )
    if include_target_config:
        statuses["target_config"] = prepare_target_config(
            tune_dir=tune_dir,
            config_mode=config_mode,
            target_source_mode=None,
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
