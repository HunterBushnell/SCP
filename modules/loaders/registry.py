from __future__ import annotations

from typing import Any, Dict


DEFAULT_CELL_LOADER = "allen_manifest"
_ALIASES = {
    "adb": "allen_manifest",
    "allen": "allen_manifest",
    "allen_manifest": "allen_manifest",
    "allen_sdk": "allen_manifest",
    "allensdk": "allen_manifest",
}


def get_cell_loader_name(cell_config: Dict[str, Any]) -> str:
    """Return the normalized loader name from `cell_config`."""

    raw = (
        cell_config.get("cell_loader")
        or cell_config.get("loader")
        or cell_config.get("model_loader")
        or DEFAULT_CELL_LOADER
    )
    key = str(raw).strip().lower()
    return _ALIASES.get(key, key)


def loader_requires_manifest(loader_name: str) -> bool:
    """Return True when a loader requires an Allen-style manifest file."""

    return _ALIASES.get(str(loader_name).strip().lower(), loader_name) == "allen_manifest"


def load_cell_with_registered_loader(cell_config: Dict[str, Any]) -> Any:
    """Dispatch cell construction to the configured loader."""

    loader_name = get_cell_loader_name(cell_config)
    if loader_name == "allen_manifest":
        from modules.loaders.allen_manifest import load_cell

        return load_cell(cell_config)

    raise ValueError(
        f"Unsupported cell_loader={loader_name!r}. "
        "Currently supported loaders: 'allen_manifest'."
    )
