from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import os

from modules.loaders.base import (
    CANONICAL_SECTION_GROUPS,
    LoadedCell,
    ensure_section_aliases,
)
from modules.loaders.paths import resolve_loader_path


_LOADED_HOC_SOURCES: set[Path] = set()
_LOADED_TEMPLATE_SOURCES: Dict[str, Path] = {}


@contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(old))


def _hoc_options(cell_config: Dict[str, Any]) -> Dict[str, Any]:
    options = cell_config.get("hoc_template", {})
    if not isinstance(options, dict):
        raise TypeError("cell_config['hoc_template'] must be an object/dict.")
    return options


def _template_name(options: Mapping[str, Any]) -> str:
    raw = options.get("template_name")
    if raw in (None, ""):
        raise KeyError("hoc_template loader requires hoc_template.template_name.")
    name = str(raw).strip()
    if not name:
        raise ValueError("hoc_template.template_name must be non-empty.")
    return name


def _constructor_args(options: Mapping[str, Any]) -> list[Any]:
    args = options.get("constructor_args", [])
    if args is None:
        return []
    if not isinstance(args, (list, tuple)):
        raise TypeError("hoc_template.constructor_args must be a JSON list.")
    return list(args)


def _section_map(options: Mapping[str, Any]) -> Dict[str, Any]:
    raw = options.get("section_map", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError("hoc_template.section_map must be an object/dict.")
    unknown = sorted(set(raw) - set(CANONICAL_SECTION_GROUPS))
    if unknown:
        raise ValueError(
            "hoc_template.section_map contains unsupported canonical group(s): "
            + ", ".join(repr(name) for name in unknown)
        )
    normalized: Dict[str, Any] = {}
    for group, value in raw.items():
        if value is None:
            normalized[group] = None
            continue
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            raise TypeError(
                f"hoc_template.section_map[{group!r}] must be a string, list, or null."
            )
        if any(not isinstance(item, str) or not item.strip() for item in values):
            raise ValueError(
                f"hoc_template.section_map[{group!r}] must contain non-empty strings."
            )
        normalized[group] = value
    return normalized


def _configured_v_init(cell_config: Dict[str, Any], options: Mapping[str, Any]) -> Optional[float]:
    candidates = [
        options.get("v_init"),
        options.get("v_init_mV"),
        cell_config.get("v_init"),
    ]
    conditions = cell_config.get("conditions", {})
    if isinstance(conditions, dict):
        candidates.extend((conditions.get("v_init"), conditions.get("v_init_mV")))
    for raw in candidates:
        if raw not in (None, ""):
            return float(raw)
    return None


def validate_config(
    cell_config: Dict[str, Any],
    *,
    base_dir: Optional[str | Path] = None,
) -> Dict[str, Path]:
    """Validate a HOC-template loader config and resolve its source file."""

    options = _hoc_options(cell_config)
    _template_name(options)
    _constructor_args(options)
    _section_map(options)
    path = resolve_loader_path(
        cell_config,
        "hoc_template",
        base_dir=base_dir,
        loader_name="hoc_template",
    )
    return {"hoc_template": path}


def discover_source_artifacts(
    cell_config: Dict[str, Any],
    *,
    base_dir: Optional[str | Path] = None,
) -> Dict[str, Path]:
    """Return the HOC file owned directly by this loader config."""

    return validate_config(cell_config, base_dir=base_dir)


def _load_template_definition(h: Any, template_path: Path, template_name: str) -> None:
    previous_source = _LOADED_TEMPLATE_SOURCES.get(template_name)
    if previous_source is not None and previous_source != template_path:
        raise RuntimeError(
            f"HOC template {template_name!r} was already loaded from {previous_source}; "
            f"cannot replace it with {template_path} in the same NEURON process. "
            "Restart the Python/Jupyter kernel before loading the other model."
        )

    if previous_source == template_path and hasattr(h, template_name):
        return

    if template_path in _LOADED_HOC_SOURCES:
        if not hasattr(h, template_name):
            raise AttributeError(
                f"HOC source {template_path} was already loaded in this process, "
                f"but it does not expose template {template_name!r}. Correct "
                "hoc_template.template_name; restart the Python/Jupyter kernel "
                "only if the source file itself must be changed."
            )
        _LOADED_TEMPLATE_SOURCES[template_name] = template_path
        return

    if previous_source is None and hasattr(h, template_name):
        raise RuntimeError(
            f"HOC template {template_name!r} is already defined in this NEURON "
            f"process before loading {template_path}. Restart the Python/Jupyter "
            "kernel so SCP can load an unambiguous template definition."
        )

    with _pushd(template_path.parent):
        loaded = h.load_file(str(template_path))
    _LOADED_HOC_SOURCES.add(template_path)
    if int(loaded) != 1 and not hasattr(h, template_name):
        raise RuntimeError(f"NEURON failed to load HOC source: {template_path}")
    if not hasattr(h, template_name):
        raise AttributeError(
            f"HOC source {template_path} did not define template {template_name!r}. "
            "The source has already been executed; correct the configuration and "
            "restart the Python/Jupyter kernel before retrying."
        )
    _LOADED_TEMPLATE_SOURCES[template_name] = template_path


def load_cell(
    cell_config: Dict[str, Any],
    *,
    base_dir: Optional[str | Path] = None,
) -> LoadedCell:
    """Instantiate an object-owned cell from a NEURON HOC template."""

    artifacts = validate_config(cell_config, base_dir=base_dir)
    template_path = artifacts["hoc_template"]
    options = _hoc_options(cell_config)
    template_name = _template_name(options)
    constructor_args = _constructor_args(options)
    section_map = _section_map(options)

    from neuron import h

    h.load_file("stdrun.hoc")
    _load_template_definition(h, template_path, template_name)
    constructor = getattr(h, template_name)
    try:
        model = constructor(*constructor_args)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to instantiate HOC template {template_name!r} from "
            f"{template_path} with constructor_args={constructor_args!r}: {exc}"
        ) from exc

    config = dict(cell_config)
    config["cell_loader"] = "hoc_template"
    paths = dict(config.get("paths", {}) or {})
    paths["hoc_template"] = str(template_path)
    config["paths"] = paths

    cell = LoadedCell(
        h=h,
        Vinit=_configured_v_init(config, options),
        config=config,
        loader="hoc_template",
        model=model,
        source_artifacts={role: str(path) for role, path in artifacts.items()},
    )
    ensure_section_aliases(
        cell,
        owner=model,
        section_map=section_map,
        allow_global_fallback=False,
        require_soma=True,
    )
    cell_name = config.get("cell_name", "<unknown>")
    print(
        f"Loaded HOC-template cell for {cell_name!r} from {template_path}, "
        f"template={template_name!r}, sections={len(cell.all)}, "
        f"Vinit={cell.Vinit}"
    )
    return cell
