"""Archive native model sources used by a saved SCP run."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


ARTIFACT_FORMAT_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_dict(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _tune_dir_from_sim_config(sim_cfg: Mapping[str, Any]) -> Optional[Path]:
    raw = sim_cfg.get("tune_dir")
    if raw not in (None, ""):
        try:
            return Path(str(raw)).expanduser().resolve()
        except Exception:
            return None

    cell = sim_cfg.get("cell")
    tune = sim_cfg.get("tune")
    if cell and tune:
        repo_root = Path(__file__).resolve().parents[2]
        return (repo_root / "cells" / str(cell) / "tunes" / str(tune)).resolve()
    return None


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _mechanism_sources(cell_config: Mapping[str, Any], tune_dir: Path) -> Iterable[Path]:
    paths = cell_config.get("paths", {})
    raw = paths.get("modfiles", "modfiles") if isinstance(paths, Mapping) else "modfiles"
    if raw in (None, ""):
        return []
    root = Path(str(raw)).expanduser()
    if not root.is_absolute():
        root = tune_dir / root
    root = root.resolve()
    if not root.is_dir():
        return []
    return sorted(path.resolve() for path in root.rglob("*.mod") if path.is_file())


def _load_cell_config(tune_dir: Path) -> Dict[str, Any]:
    candidates = (
        tune_dir / "cell_configs" / "cell_config.json",
        tune_dir / "cell_config.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return _read_json_dict(candidate)
    return {}


def archive_model_artifacts(
    sim_cfg: Mapping[str, Any],
    run_dir: Path,
    *,
    cell_config: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Copy loader/model sources into ``run_dir/model_artifacts`` with hashes."""
    if sim_cfg.get("save_model_artifacts", True) is False:
        return None

    tune_dir = _tune_dir_from_sim_config(sim_cfg)
    if tune_dir is None or not tune_dir.is_dir():
        return None

    config = dict(cell_config or _load_cell_config(tune_dir))
    if not config:
        return None

    from modules.loaders import (
        discover_cell_source_artifacts,
        get_cell_loader_name,
    )

    loader_name = get_cell_loader_name(config)
    errors: list[str] = []
    candidates: list[tuple[str, Path]] = []
    try:
        for role, path in discover_cell_source_artifacts(config, base_dir=tune_dir).items():
            candidates.append((str(role), Path(path).expanduser().resolve()))
    except Exception as exc:
        errors.append(f"Loader artifact discovery failed: {exc}")

    candidates.extend(("mechanism_source", path) for path in _mechanism_sources(config, tune_dir))

    unique: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for kind, path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        unique.append((kind, path))

    archive_root = Path(run_dir).resolve() / "model_artifacts"
    artifacts: list[Dict[str, Any]] = []
    for kind, source in unique:
        source_hash = sha256_file(source)
        if _path_within(source, tune_dir):
            target_relative = source.relative_to(tune_dir)
            archive_relative = Path("files") / target_relative
            source_label = target_relative.as_posix()
            restorable = True
        else:
            archive_relative = Path("external") / f"{source_hash[:12]}_{source.name}"
            target_relative = None
            source_label = str(source)
            restorable = False

        archived = archive_root / archive_relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, archived)
        entry: Dict[str, Any] = {
            "kind": kind,
            "source_path": source_label,
            "archive_path": archive_relative.as_posix(),
            "sha256": source_hash,
            "restorable": restorable,
        }
        if target_relative is not None:
            entry["target_relative_path"] = target_relative.as_posix()
        artifacts.append(entry)

    manifest = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "loader": loader_name,
        "tune": tune_dir.name,
        "artifacts": artifacts,
        "errors": errors,
    }
    archive_root.mkdir(parents=True, exist_ok=True)
    manifest_path = archive_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "filename": str(manifest_path.relative_to(Path(run_dir).resolve())),
        "loader": loader_name,
        "artifact_count": len(artifacts),
        "errors": errors,
    }
