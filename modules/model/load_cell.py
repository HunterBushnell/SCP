"""
Step 5.2.1 — Load Cell

Public cell-loading entry point:

    cell = load_cell(cell_config, base_dir=tune_dir)

Model-specific loading is delegated to `modules.loaders`. SCP includes
`allen_manifest` and object-owned `hoc_template` adapters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from modules.loaders import (
    LoadedCell,
    get_cell_loader_name,
    load_cell_with_registered_loader,
)

__all__ = ["LoadedCell", "get_cell_loader_name", "load_cell"]


def load_cell(
    cell_config: Dict[str, Any],
    *,
    base_dir: Optional[str | Path] = None,
) -> LoadedCell:
    """Build a loaded cell, resolving model paths relative to ``base_dir``."""

    return load_cell_with_registered_loader(cell_config, base_dir=base_dir)
