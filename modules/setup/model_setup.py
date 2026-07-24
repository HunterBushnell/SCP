"""Auto-first model-source discovery for the Step 1 notebook."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Mapping, Optional

from modules.loaders import get_cell_loader_name

from .defaults import guess_specimen_from_cell
from .mechanisms import load_tune_cell_config


_HOC_TEMPLATE_PATTERN = re.compile(
    r"^\s*begintemplate\s+([A-Za-z_][A-Za-z0-9_]*)\b",
    flags=re.MULTILINE | re.IGNORECASE,
)
_IGNORED_DISCOVERY_PARTS = {
    ".git",
    "__pycache__",
    "cell_configs",
    "model_artifacts",
    "notebook_exports",
    "output",
    "output_data",
    "outputs",
    "runs",
    "x86_64",
}
_OVERRIDE_KEYS = {
    "cell_loader",
    "source_type",
    "specimen_id",
    "model_type",
    "paths",
    "loader_paths",
    "hoc_template",
    "loader_config",
}


def resolve_step1_model_setup(
    tune_dir: str | Path,
    *,
    cell_name: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Resolve source and loader settings from config, staged files, or overrides.

    Existing ``cell_config.json`` is authoritative on reruns. A fresh staged
    tune can be inferred when it has exactly one Allen manifest or exactly one
    HOC template declaration. Explicit overrides remain available for Allen
    downloads and ambiguous/custom layouts.
    """
    tune_root = Path(tune_dir).expanduser().resolve()
    normalized_overrides = _normalize_overrides(overrides)
    stored = load_tune_cell_config(tune_root)

    if stored is not None:
        setup = _setup_from_cell_config(stored)
        discovery_mode = "cell_config"
    elif _declares_new_adb_download(normalized_overrides):
        setup = {
            "source_type": "adb",
            "cell_loader": "allen_manifest",
            "specimen_id": None,
            "model_type": "perisomatic",
            "loader_paths": {"manifest": "manifest.json"},
            "loader_config": {},
        }
        discovery_mode = "overrides"
    elif _declares_complete_hoc_source(normalized_overrides):
        setup = _setup_from_explicit_hoc(tune_root, normalized_overrides)
        discovery_mode = "overrides"
    else:
        setup = _discover_staged_setup(tune_root)
        discovery_mode = str(setup.pop("_discovery_mode"))

    _deep_update(setup, normalized_overrides)
    normalized = _normalize_setup(setup, cell_name=cell_name)
    normalized["discovery"] = discovery_mode
    return normalized


def _normalize_overrides(
    overrides: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping):
        raise TypeError("MODEL_SOURCE_OVERRIDES must be a dictionary or None.")
    unknown = sorted(set(overrides) - _OVERRIDE_KEYS)
    if unknown:
        raise ValueError(
            "Unsupported model-source override field(s): "
            + ", ".join(repr(key) for key in unknown)
        )

    normalized: dict[str, Any] = {}
    for key in ("cell_loader", "source_type", "specimen_id", "model_type"):
        if key in overrides:
            normalized[key] = copy.deepcopy(overrides[key])

    raw_paths = overrides.get("paths", overrides.get("loader_paths"))
    if raw_paths is not None:
        if not isinstance(raw_paths, Mapping):
            raise TypeError("MODEL_SOURCE_OVERRIDES paths must be a dictionary.")
        normalized["loader_paths"] = copy.deepcopy(dict(raw_paths))

    raw_loader_config = overrides.get("loader_config")
    if raw_loader_config is not None:
        if not isinstance(raw_loader_config, Mapping):
            raise TypeError(
                "MODEL_SOURCE_OVERRIDES loader_config must be a dictionary."
            )
        normalized["loader_config"] = copy.deepcopy(dict(raw_loader_config))

    raw_hoc = overrides.get("hoc_template")
    if raw_hoc is not None:
        if not isinstance(raw_hoc, Mapping):
            raise TypeError(
                "MODEL_SOURCE_OVERRIDES hoc_template must be a dictionary."
            )
        loader_config = normalized.setdefault("loader_config", {})
        existing = loader_config.setdefault("hoc_template", {})
        if not isinstance(existing, dict):
            raise TypeError("loader_config.hoc_template must be a dictionary.")
        _deep_update(existing, dict(raw_hoc))
    return normalized


def _setup_from_cell_config(cell_config: Mapping[str, Any]) -> dict[str, Any]:
    loader_name = get_cell_loader_name(dict(cell_config))
    paths = cell_config.get("paths", {}) or {}
    if not isinstance(paths, Mapping):
        raise TypeError("Stored cell_config.json paths must be a dictionary.")
    loader_config: dict[str, Any] = {}
    if loader_name == "hoc_template":
        hoc_options = cell_config.get("hoc_template", {}) or {}
        if not isinstance(hoc_options, Mapping):
            raise TypeError(
                "Stored cell_config.json hoc_template must be a dictionary."
            )
        loader_config["hoc_template"] = copy.deepcopy(dict(hoc_options))
    return {
        "source_type": "existing",
        "cell_loader": loader_name,
        "specimen_id": None,
        "model_type": "perisomatic",
        "loader_paths": copy.deepcopy(dict(paths)),
        "loader_config": loader_config,
    }


def _discover_staged_setup(tune_root: Path) -> dict[str, Any]:
    manifests = _candidate_files(tune_root, "manifest.json")
    hoc_templates = _hoc_template_declarations(tune_root)
    if manifests and hoc_templates:
        raise ValueError(
            "Model-source discovery found both an Allen manifest and HOC template "
            "declarations. Set MODEL_SOURCE_OVERRIDES to choose the intended loader."
        )
    if len(manifests) > 1:
        choices = ", ".join(_relative_path(path, tune_root) for path in manifests)
        raise ValueError(
            "Model-source discovery found multiple manifest.json files: "
            f"{choices}. Set MODEL_SOURCE_OVERRIDES['paths']['manifest']."
        )
    if manifests:
        return {
            "source_type": "existing",
            "cell_loader": "allen_manifest",
            "specimen_id": None,
            "model_type": "perisomatic",
            "loader_paths": {"manifest": _relative_path(manifests[0], tune_root)},
            "loader_config": {},
            "_discovery_mode": "staged_manifest",
        }
    if len(hoc_templates) > 1:
        choices = ", ".join(
            f"{_relative_path(path, tune_root)}:{name}"
            for path, name in hoc_templates
        )
        raise ValueError(
            "Model-source discovery found multiple HOC template declarations: "
            f"{choices}. Set MODEL_SOURCE_OVERRIDES to select one."
        )
    if hoc_templates:
        path, template_name = hoc_templates[0]
        paths: dict[str, Any] = {
            "hoc_template": _relative_path(path, tune_root),
        }
        modfiles = tune_root / "modfiles"
        if modfiles.is_dir():
            paths["modfiles"] = "modfiles"
        return {
            "source_type": "existing",
            "cell_loader": "hoc_template",
            "specimen_id": None,
            "model_type": "perisomatic",
            "loader_paths": paths,
            "loader_config": {
                "hoc_template": {"template_name": template_name},
            },
            "_discovery_mode": "staged_hoc_template",
        }
    raise FileNotFoundError(
        f"Could not infer a model source in {tune_root}. Stage one manifest.json "
        "or one HOC template under the tune, reuse an existing cell_config.json, "
        "or set MODEL_SOURCE_OVERRIDES in Step 1.2."
    )


def _declares_new_adb_download(overrides: Mapping[str, Any]) -> bool:
    source = str(overrides.get("source_type") or "").strip().lower()
    loader = str(overrides.get("cell_loader") or "").strip().lower()
    return source == "adb" or (
        loader in {"allen_manifest", "allen", "adb"}
        and bool(overrides.get("specimen_id"))
    )


def _declares_complete_hoc_source(overrides: Mapping[str, Any]) -> bool:
    loader = str(overrides.get("cell_loader") or "").strip().lower()
    paths = overrides.get("loader_paths", {})
    options = overrides.get("loader_config", {}).get("hoc_template", {})
    return (
        loader in {"hoc_template", "hoc", "template"}
        and isinstance(paths, Mapping)
        and paths.get("hoc_template") not in (None, "")
        and isinstance(options, Mapping)
        and options.get("template_name") not in (None, "")
    )


def _setup_from_explicit_hoc(
    tune_root: Path,
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    paths = copy.deepcopy(dict(overrides.get("loader_paths", {})))
    if "modfiles" not in paths and (tune_root / "modfiles").is_dir():
        paths["modfiles"] = "modfiles"
    return {
        "source_type": "existing",
        "cell_loader": "hoc_template",
        "specimen_id": None,
        "model_type": "perisomatic",
        "loader_paths": paths,
        "loader_config": copy.deepcopy(dict(overrides.get("loader_config", {}))),
    }


def _normalize_setup(
    setup: Mapping[str, Any],
    *,
    cell_name: Optional[str],
) -> dict[str, Any]:
    loader_name = get_cell_loader_name(
        {"cell_loader": setup.get("cell_loader") or "allen_manifest"}
    )
    source_type = str(setup.get("source_type") or "existing").strip().lower()
    if source_type not in {"existing", "adb"}:
        raise ValueError("source_type must be 'existing' or 'adb'.")
    if source_type == "adb" and loader_name != "allen_manifest":
        raise ValueError("source_type='adb' requires cell_loader='allen_manifest'.")

    paths = setup.get("loader_paths", {}) or {}
    loader_config = setup.get("loader_config", {}) or {}
    if not isinstance(paths, Mapping):
        raise TypeError("Resolved loader_paths must be a dictionary.")
    if not isinstance(loader_config, Mapping):
        raise TypeError("Resolved loader_config must be a dictionary.")
    paths = copy.deepcopy(dict(paths))
    loader_config = copy.deepcopy(dict(loader_config))
    if loader_name == "allen_manifest":
        paths.setdefault("manifest", "manifest.json")
    else:
        hoc_options = loader_config.get("hoc_template")
        if not isinstance(hoc_options, Mapping):
            raise KeyError(
                "hoc_template discovery requires hoc_template.template_name. "
                "Set MODEL_SOURCE_OVERRIDES for an ambiguous model."
            )
        if hoc_options.get("template_name") in (None, ""):
            raise KeyError("hoc_template.template_name must be non-empty.")
        if paths.get("hoc_template") in (None, ""):
            raise KeyError("paths.hoc_template must be non-empty.")

    specimen_id = setup.get("specimen_id")
    if specimen_id in (None, "") and source_type == "adb":
        specimen_id = guess_specimen_from_cell(str(cell_name or ""))
    if specimen_id in (None, "") and source_type == "adb":
        raise ValueError(
            "ADB download setup requires specimen_id in MODEL_SOURCE_OVERRIDES."
        )
    return {
        "source_type": source_type,
        "cell_loader": loader_name,
        "specimen_id": None if specimen_id in (None, "") else int(specimen_id),
        "model_type": str(setup.get("model_type") or "perisomatic"),
        "loader_paths": paths,
        "loader_config": loader_config,
    }


def _candidate_files(tune_root: Path, filename: str) -> list[Path]:
    return sorted(
        path.resolve()
        for path in tune_root.rglob(filename)
        if path.is_file() and not _is_ignored(path, tune_root)
    )


def _hoc_template_declarations(tune_root: Path) -> list[tuple[Path, str]]:
    declarations: list[tuple[Path, str]] = []
    for path in sorted(tune_root.rglob("*")):
        if (
            not path.is_file()
            or path.suffix.lower() != ".hoc"
            or _is_ignored(path, tune_root)
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        declarations.extend(
            (path.resolve(), match.group(1))
            for match in _HOC_TEMPLATE_PATTERN.finditer(text)
        )
    return declarations


def _is_ignored(path: Path, tune_root: Path) -> bool:
    try:
        parts = path.relative_to(tune_root).parts[:-1]
    except ValueError:
        return True
    return any(part in _IGNORED_DISCOVERY_PARTS for part in parts)


def _relative_path(path: Path, tune_root: Path) -> str:
    return path.resolve().relative_to(tune_root).as_posix()


def _deep_update(target: dict[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
