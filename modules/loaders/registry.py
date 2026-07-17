from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, FrozenSet, Mapping, Optional


DEFAULT_CELL_LOADER = "allen_manifest"


@dataclass(frozen=True)
class LoaderSpec:
    """Lazy registry metadata for one model loader adapter."""

    name: str
    module: str
    aliases: tuple[str, ...] = ()
    required_path_keys: tuple[str, ...] = ()
    capabilities: FrozenSet[str] = frozenset()

    def import_module(self) -> ModuleType:
        return import_module(self.module)

    def supports(self, capability: str) -> bool:
        return str(capability).strip().lower() in self.capabilities


_SPECS: Dict[str, LoaderSpec] = {
    "allen_manifest": LoaderSpec(
        name="allen_manifest",
        module="modules.loaders.allen_manifest",
        aliases=("adb", "allen", "allen_sdk", "allensdk"),
        required_path_keys=("manifest",),
        capabilities=frozenset(
            {"manual_tuning", "geometry", "synapses", "simulation", "act_active"}
        ),
    ),
    "hoc_template": LoaderSpec(
        name="hoc_template",
        module="modules.loaders.hoc_template",
        aliases=("hoc", "template", "hoc-template"),
        required_path_keys=("hoc_template",),
        capabilities=frozenset(
            {"manual_tuning", "geometry", "synapses", "simulation", "act_active"}
        ),
    ),
}

_ALIASES: Dict[str, str] = {}
for _canonical_name, _spec in _SPECS.items():
    _ALIASES[_canonical_name] = _canonical_name
    for _alias in _spec.aliases:
        _ALIASES[_alias] = _canonical_name


def _normalize_loader_name(raw: Any) -> str:
    key = str(raw).strip().lower()
    return _ALIASES.get(key, key)


def get_cell_loader_name(cell_config: Mapping[str, Any]) -> str:
    """Return the normalized loader name from ``cell_config``."""

    raw = (
        cell_config.get("cell_loader")
        or cell_config.get("loader")
        or cell_config.get("model_loader")
        or DEFAULT_CELL_LOADER
    )
    return _normalize_loader_name(raw)


def available_cell_loaders() -> tuple[str, ...]:
    """Return registered canonical loader names."""

    return tuple(sorted(_SPECS))


def get_loader_spec(cell_config_or_name: Mapping[str, Any] | str) -> LoaderSpec:
    """Return registry metadata for a config or loader name."""

    if isinstance(cell_config_or_name, Mapping):
        loader_name = get_cell_loader_name(cell_config_or_name)
    else:
        loader_name = _normalize_loader_name(cell_config_or_name)
    try:
        return _SPECS[loader_name]
    except KeyError as exc:
        supported = ", ".join(repr(name) for name in available_cell_loaders())
        raise ValueError(
            f"Unsupported cell_loader={loader_name!r}. Registered loaders: {supported}."
        ) from exc


def loader_requires_manifest(loader_name: str) -> bool:
    """Compatibility helper for older setup/session code."""

    return "manifest" in get_loader_spec(loader_name).required_path_keys


def loader_supports(
    cell_config_or_name: Mapping[str, Any] | str,
    capability: str,
) -> bool:
    """Return whether a registered loader advertises a capability."""

    return get_loader_spec(cell_config_or_name).supports(capability)


def _call_loader_hook(
    spec: LoaderSpec,
    hook_name: str,
    cell_config: Dict[str, Any],
    *,
    base_dir: Optional[str | Path],
) -> Any:
    module = spec.import_module()
    hook = getattr(module, hook_name, None)
    if not callable(hook):
        raise RuntimeError(
            f"Registered loader {spec.name!r} does not define callable {hook_name}()."
        )
    return hook(cell_config, base_dir=base_dir)


def validate_cell_loader_config(
    cell_config: Dict[str, Any],
    *,
    base_dir: Optional[str | Path] = None,
) -> Dict[str, Path]:
    """Validate loader-specific config and return resolved required artifacts."""

    if not isinstance(cell_config, dict):
        raise TypeError("cell_config must be an object/dict.")
    spec = get_loader_spec(cell_config)
    artifacts = _call_loader_hook(
        spec,
        "validate_config",
        cell_config,
        base_dir=base_dir,
    )
    if not isinstance(artifacts, dict):
        raise TypeError(f"{spec.name} validate_config() must return dict[str, Path].")
    return {str(role): Path(path).resolve() for role, path in artifacts.items()}


def discover_cell_source_artifacts(
    cell_config: Dict[str, Any],
    *,
    base_dir: Optional[str | Path] = None,
) -> Dict[str, Path]:
    """Return loader-owned source artifacts as ``role -> absolute path``."""

    if not isinstance(cell_config, dict):
        raise TypeError("cell_config must be an object/dict.")
    spec = get_loader_spec(cell_config)
    module = spec.import_module()
    discover = getattr(module, "discover_source_artifacts", None)
    if discover is None:
        return validate_cell_loader_config(cell_config, base_dir=base_dir)
    if not callable(discover):
        raise RuntimeError(
            f"Registered loader {spec.name!r} has non-callable discover_source_artifacts."
        )
    artifacts = discover(cell_config, base_dir=base_dir)
    if not isinstance(artifacts, dict):
        raise TypeError(
            f"{spec.name} discover_source_artifacts() must return dict[str, Path]."
        )
    return {str(role): Path(path).resolve() for role, path in artifacts.items()}


def load_cell_with_registered_loader(
    cell_config: Dict[str, Any],
    *,
    base_dir: Optional[str | Path] = None,
) -> Any:
    """Dispatch cell construction to the configured lazy loader adapter."""

    if not isinstance(cell_config, dict):
        raise TypeError("cell_config must be an object/dict.")
    spec = get_loader_spec(cell_config)
    return _call_loader_hook(spec, "load_cell", cell_config, base_dir=base_dir)
