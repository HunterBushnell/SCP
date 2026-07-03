"""Registry for SCP's built-in input-generation modes."""

from __future__ import annotations

from typing import Any, Dict

from .mode_poisson import _mode_homogeneous_poisson, _mode_inhomogeneous_poisson
from .mode_precomputed import _mode_precomputed


def get_default_mode_registry() -> Dict[str, Any]:
    """Return the built-in mode handlers used by Step 5 input generation."""
    return {
        "homogeneous_poisson": _mode_homogeneous_poisson,
        "precomputed": _mode_precomputed,
        "inhomogeneous_poisson": _mode_inhomogeneous_poisson,
    }
