"""
Shared helpers for SCP teaching notebooks.

These helpers keep notebook setup cells concise while preserving the same
runtime behavior used in local and Colab workflows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from modules_local.load_cell import load_cell


def resolve_cell_config_for_notebook(cell_name: str, tune_dir: Path | None = None) -> Dict[str, Any]:
    """
    Load and normalize `cell_config.json` for notebook build steps.

    `cell_configs/cell_config.json` is treated as the canonical source. The
    caller can provide `tune_dir` or rely on the current working directory.
    """
    base = Path(tune_dir) if tune_dir is not None else Path(".")
    cell_cfg_path = base / "cell_configs" / "cell_config.json"

    if cell_cfg_path.is_file():
        cell_config: Dict[str, Any] = json.loads(cell_cfg_path.read_text())
        if not isinstance(cell_config, dict):
            cell_config = {}
    else:
        cell_config = {}

    cell_config.setdefault("cell_name", cell_name)
    paths = cell_config.setdefault("paths", {})
    if not isinstance(paths, dict):
        paths = {}
        cell_config["paths"] = paths
    paths.setdefault("manifest", "manifest.json")

    tuning = cell_config.get("tuning")
    if not isinstance(tuning, dict) or "soma_diam_multiplier" not in tuning:
        raise KeyError(
            "cell_configs/cell_config.json must define tuning.soma_diam_multiplier. "
            "Run Step 0 scaffold or set it manually before Steps 2-5."
        )
    tuning["soma_diam_multiplier"] = float(tuning["soma_diam_multiplier"])

    return cell_config


def build_cell_for_notebook(cell_config: Dict[str, Any]):
    """
    Build a cell via `modules_local.load_cell` and expose common section attrs.

    The returned object is compatible with existing notebook calls that access
    `cell.soma`, `cell.dend`, `cell.apic`, and `cell.axon`.
    """
    loaded = load_cell(cell_config)

    if not hasattr(loaded, "soma"):
        loaded.soma = loaded.h.soma
    if not hasattr(loaded, "dend"):
        loaded.dend = list(loaded.h.dend) if hasattr(loaded.h, "dend") else []
    if not hasattr(loaded, "apic"):
        loaded.apic = list(loaded.h.apic) if hasattr(loaded.h, "apic") else []
    if not hasattr(loaded, "axon"):
        loaded.axon = list(loaded.h.axon) if hasattr(loaded.h, "axon") else []

    return loaded

