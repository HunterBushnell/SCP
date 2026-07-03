"""Create or update standard SCP config files in a tune directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import copy

from .defaults import (
    default_cell_config,
    default_geometry_config,
    default_placeholder_syn_group,
    default_sim_config,
    default_syn_config,
    guess_cell_color,
)
from .json_utils import _write_json, _write_scaffold_json
from .paths import resolve_step1_paths


def scaffold_common_configs(
    *,
    tune_dir: Path,
    cell_name: str,
    tune_name: str,
    specimen_id: int,
    model_type: str,
    soma_diam_multiplier: float,
    color: Optional[str] = None,
    config_mode: str = "fill",
    sync_cell_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Ensure common pipeline config files exist under cell_configs/.

    config_mode:
      - fill: create missing files and fill missing keys in existing files
      - overwrite: replace existing files with defaults
      - skip: do not modify existing files
    """
    paths = resolve_step1_paths(tune_dir)
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.syn_groups_dir.mkdir(parents=True, exist_ok=True)

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
    sim_cfg_defaults["soma_diam_multiplier"] = float(soma_diam_multiplier)
    sim_cfg_path = paths.config_dir / "sim_config.json"
    sim_status, sim_cfg_data = _write_scaffold_json(sim_cfg_path, sim_cfg_defaults, config_mode)
    # Keep sim config focused on simulation controls; remove identity metadata.
    if isinstance(sim_cfg_data, dict):
        before_sim = copy.deepcopy(sim_cfg_data)
        sim_cfg_data.pop("specimen_id", None)
        sim_cfg_data.pop("model_type", None)
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

    placeholder_group_path = paths.syn_groups_dir / "placeholder_off.json"
    placeholder_defaults = default_placeholder_syn_group(group_name="placeholder_off")
    group_status, _ = _write_scaffold_json(placeholder_group_path, placeholder_defaults, config_mode)
    statuses["syn_group_placeholder"] = {
        "path": str(placeholder_group_path),
        "status": group_status,
    }

    syn_cfg_defaults = default_syn_config(include_path="syn_groups/placeholder_off.json")
    syn_cfg_path = paths.config_dir / "syn_config.json"
    syn_status, _ = _write_scaffold_json(syn_cfg_path, syn_cfg_defaults, config_mode)
    statuses["syn_config"] = {
        "path": str(syn_cfg_path),
        "status": syn_status,
    }

    return statuses
