"""Path discovery and metadata helpers for input-generation configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import json


def _resolve_config_root(path: Optional[Union[Path, str]] = None) -> Path:
    """
    Resolve where sim_config.json and syn_config.json live.

    Priority:
      - If an explicit file is provided, use its parent directory (unless missing
        and a cell_configs/ folder with configs exists).
      - If a directory is provided (or path is None), prefer cell_configs/
        when it contains sim_config.json or syn_config.json.
      - Fall back to the provided directory.
    """
    explicit_file = False
    p = None
    if path is None:
        base = Path.cwd()
    else:
        p = Path(path)
        if p.is_dir():
            base = p
        elif p.is_file():
            base = p.parent
            explicit_file = True
        elif p.suffix == ".json":
            base = p.parent
            explicit_file = True
        else:
            base = p

    if base.name == "cell_configs":
        return base

    candidate = base / "cell_configs"
    if (candidate / "sim_config.json").is_file() or (candidate / "syn_config.json").is_file():
        if not explicit_file:
            return candidate
        if p is not None and not p.is_file():
            return candidate
    if explicit_file:
        return base
    return base


def _inject_path_metadata(sim_cfg: Dict[str, Any], config_root: Path) -> None:
    """
    Populate sim_cfg with tune/cell labels inferred from the config path
    when those fields are missing.
    """
    tune_dir = config_root.parent if config_root.name == "cell_configs" else config_root
    cell_cfg_path = (
        config_root / "cell_config.json"
        if config_root.name == "cell_configs"
        else tune_dir / "cell_configs" / "cell_config.json"
    )
    cell_cfg = {}
    if cell_cfg_path.is_file():
        try:
            cell_cfg = json.loads(cell_cfg_path.read_text())
        except Exception:
            cell_cfg = {}

    from scp_pipeline.workspaces import infer_workspace, apply_workspace_metadata
    workspace = infer_workspace(tune_dir, cell_config=cell_cfg,
        cell_name=sim_cfg.get('cell'), tune_name=sim_cfg.get('tune'))
    apply_workspace_metadata(sim_cfg, workspace)
    if not sim_cfg.get("color") and cell_cfg.get("color") is not None:
        sim_cfg["color"] = cell_cfg.get("color")
