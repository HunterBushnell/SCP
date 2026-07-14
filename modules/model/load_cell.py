"""
Step 5.2.1 — Load Cell

Public cell-loading entry point:

    cell = load_cell(cell_config)

Model-specific loading is delegated to `modules.loaders`. The current bundled
examples use `cell_loader="allen_manifest"` for Allen Cell Types/ADB bundles.
"""

from __future__ import annotations

from typing import Any, Dict

from modules.loaders import (
    LoadedCell,
    get_cell_loader_name,
    load_cell_with_registered_loader,
)

__all__ = ["LoadedCell", "get_cell_loader_name", "load_cell"]


def load_cell(cell_config: Dict[str, Any]) -> LoadedCell:
    """Build and return a loaded cell using `cell_config['cell_loader']`."""

    return load_cell_with_registered_loader(cell_config)
