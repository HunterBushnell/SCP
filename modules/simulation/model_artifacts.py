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
    from scp_pipeline.workspaces import saved_namespace_tune_dir

    return saved_namespace_tune_dir(dict(sim_cfg))


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _mechanism_sources(cell_config: Mapping[str, Any], tune_dir: Path) -> Iterable[Path]:
    from scp_pipeline.setup.mechanisms import resolve_modfiles_dir
    root = resolve_modfiles_dir(tune_dir, dict(cell_config))
    if root is None:
        return []
    if not root.is_dir():
        return []
    from scp_pipeline.loaders.source_context import EXCLUDED_PARTS
    return sorted(path.resolve() for path in root.rglob("*") if path.is_file() and
                  path.suffix in {'.mod', '.inc'} and
                  not any(part in EXCLUDED_PARTS for part in path.relative_to(root).parts))


def construction_inputs(cell_config, tune_dir):
    """Hash declared native inputs at preparation, not just at save time."""
    from scp_pipeline.loaders import discover_cell_source_artifacts
    candidates = dict(discover_cell_source_artifacts(cell_config, base_dir=tune_dir))
    candidates.update({f'mechanism:{index}': path for index, path in
                       enumerate(_mechanism_sources(cell_config, Path(tune_dir)))})
    unique = {}
    for kind, value in candidates.items():
        path = Path(value).expanduser().absolute()
        if path.is_file():
            unique.setdefault(str(path), {'kind': kind, 'path': str(path), 'sha256': sha256_file(path)})
    return list(unique.values())


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
    prepared_inputs: Optional[list[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Copy loader/model sources into ``run_dir/model_artifacts`` with hashes."""
    if sim_cfg.get("save_model_artifacts", True) is False:
        return None

    from scp_pipeline.workspaces import result_workspace_identity
    identity = result_workspace_identity({"sim_cfg": dict(sim_cfg)})
    tune_dir = _tune_dir_from_sim_config(sim_cfg)
    if tune_dir is None or not tune_dir.is_dir():
        return None

    config = dict(cell_config or _load_cell_config(tune_dir))
    if not config:
        return None

    from scp_pipeline.loaders import (
        discover_cell_source_artifacts,
        get_cell_loader_name,
    )

    loader_name = get_cell_loader_name(config)
    from scp_pipeline.loaders.source_context import effective_source_root, declared_source_directories, original_source_root
    model_root = effective_source_root(config, tune_dir)
    input_hashes = {item['path']: item['sha256'] for item in prepared_inputs or []}
    errors: list[str] = []
    candidates: list[tuple[str, Path]] = []
    try:
        for role, path in discover_cell_source_artifacts(config, base_dir=tune_dir).items():
            candidates.append((str(role), Path(path).expanduser().absolute()))
    except Exception as exc:
        errors.append(f"Loader artifact discovery failed: {exc}")

    candidates.extend(("mechanism_source", path) for path in _mechanism_sources(config, tune_dir))
    model_directories = []
    if model_root is not None:
        try:
            model_directories = declared_source_directories(config, tune_dir)
        except Exception as exc:
            errors.append(f'Native directory discovery failed: {exc}')
    for item in prepared_inputs or []:
        if not Path(item['path']).is_file():
            errors.append(f'Construction input is no longer available: {item["path"]}. The archive is incomplete.')

    unique: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for kind, path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        unique.append((kind, path))

    run_root = Path(run_dir).resolve()
    archive_root = run_root / "model_artifacts"
    if not _path_within(archive_root.resolve(), run_root):
        raise ValueError('Model-artifact destination escapes the selected saved run.')
    # Hash every available source before creating any archive output. A failed
    # source read must not leave an apparently usable partial source snapshot.
    source_hashes = {source: sha256_file(source) for _, source in unique}
    planned = []
    for kind, source in unique:
        source_hash = source_hashes[source]
        if _path_within(source, tune_dir):
            target_relative = source.relative_to(tune_dir)
            archive_relative = Path("files") / target_relative
            source_label = target_relative.as_posix()
            restorable = True
        else:
            archive_relative = (Path('external') / source.relative_to(model_root.resolve())
                                if model_root is not None and _path_within(source, model_root.resolve())
                                else Path("external") / f"{source_hash[:12]}_{source.name}")
            target_relative = None
            source_label = str(source)
            restorable = False

        archived = archive_root / archive_relative
        if not _path_within(archived.resolve(), archive_root.resolve()):
            raise ValueError(f'Model-artifact file destination escapes the source archive: {archived}')
        planned.append((kind, source, source_hash, archive_relative, target_relative, source_label, restorable))

    artifacts: list[Dict[str, Any]] = []
    for kind, source, source_hash, archive_relative, target_relative, source_label, restorable in planned:
        archived = archive_root / archive_relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, archived)
        if sha256_file(archived) != source_hash:
            raise RuntimeError(f'Model input changed while archiving: {source}. Save again after reviewing the inputs.')
        entry: Dict[str, Any] = {
            "kind": kind,
            "source_path": source_label,
            "archive_path": archive_relative.as_posix(),
            "sha256": source_hash,
            "restorable": restorable,
        }
        if prepared_inputs is not None:
            entry['construction_sha256'] = input_hashes.get(str(source))
            entry['matches_construction'] = entry['construction_sha256'] == source_hash
            if not entry['matches_construction']:
                errors.append(f'Source changed since cell preparation: {source}. Archived bytes are not the construction input.')
        if model_root is not None:
            try:
                relative = source.relative_to(model_root.resolve())
                entry['materialize_relative_path'] = (Path('model_sources') / relative).as_posix()
            except ValueError:
                pass
        if target_relative is not None:
            entry["target_relative_path"] = target_relative.as_posix()
        artifacts.append(entry)

    manifest = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "workspace_identity": identity,
        "loader": loader_name,
        "tune": tune_dir.name,
        "artifacts": artifacts,
        "errors": errors,
        "cell_config": config,
        "construction_recipe_available": loader_name != 'runtime_cell',
        "source_mode": 'session_only' if loader_name == 'runtime_cell' else (config.get('model_source') or {}).get('mode', 'tune_local'),
        "construction_inputs": prepared_inputs,
        "model_directories": model_directories,
        "original_source_root": (
            str(original_source_root(config, tune_dir))
            if config.get('model_source') else None),
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
