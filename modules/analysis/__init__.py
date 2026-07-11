"""Analysis subpackage (metrics, plotting, and UI helpers).

Submodules are loaded lazily so Step 5 auto-plot saving does not import the
interactive analysis UI unless a notebook explicitly requests it.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SUBMODULES = {
    "analysis",
    "plotting",
    "analysis_ui",
    "auto_plots",
    "bio_curve",
    "single_plot_panel",
    "ui",
}

__all__ = sorted(_SUBMODULES)


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    analysis_module = import_module(f"{__name__}.analysis")
    if hasattr(analysis_module, name):
        value = getattr(analysis_module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
