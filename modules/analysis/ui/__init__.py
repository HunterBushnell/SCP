"""Notebook UI sections for Step 6 analysis.

The public notebook API is organized by section here, while
`modules.analysis.analysis_ui` remains as the stable compatibility facade used
by existing notebooks.
"""

from __future__ import annotations

from .selection import *
from .outputs import *
from .inputs import *
from .metrics import *
from .extra import *
from ._engine import show_md

__all__ = [
    name
    for name in globals()
    if not name.startswith("_") and name not in {"annotations"}
]
