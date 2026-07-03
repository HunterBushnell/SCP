"""Convenience imports for input-generation config normalization."""

from __future__ import annotations

from .config_paths import _inject_path_metadata, _resolve_config_root
from .group_config import _expand_group_includes, _normalize_group_configs
from .sim_config import _normalize_sim_config

__all__ = [
    "_expand_group_includes",
    "_inject_path_metadata",
    "_normalize_group_configs",
    "_normalize_sim_config",
    "_resolve_config_root",
]
