from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def candidate_base_dirs(
    cell_config: Dict[str, Any],
    *,
    base_dir: Optional[str | Path] = None,
) -> tuple[Path, ...]:
    """Return ordered bases used for loader-relative source paths."""

    raw_bases: list[tuple[Any, bool]] = []
    if base_dir is not None:
        raw_bases.append((base_dir, False))

    explicit_base: Optional[Path] = None
    if base_dir is not None:
        explicit_base = Path(str(base_dir)).expanduser().resolve()

    paths = cell_config.get("paths", {})
    if isinstance(paths, dict):
        raw_bases.extend(
            (paths.get(key), True) for key in ("tune_dir", "base_dir", "root")
        )
    raw_bases.extend(
        (cell_config.get(key), True) for key in ("tune_dir", "base_dir", "root")
    )
    raw_bases.append((Path.cwd(), False))

    seen: set[Path] = set()
    bases: list[Path] = []
    for raw, anchor_to_explicit_base in raw_bases:
        if raw in (None, ""):
            continue
        path = Path(str(raw)).expanduser()
        if anchor_to_explicit_base and explicit_base is not None and not path.is_absolute():
            path = explicit_base / path
        try:
            path = path.resolve()
        except Exception:
            pass
        if path in seen:
            continue
        seen.add(path)
        bases.append(path)
    return tuple(bases)


def resolve_loader_path(
    cell_config: Dict[str, Any],
    path_key: str,
    *,
    default: Optional[str] = None,
    base_dir: Optional[str | Path] = None,
    loader_name: str,
    require_file: bool = True,
) -> Path:
    """Resolve one loader source path with tune-relative compatibility."""

    paths = cell_config.get("paths", {})
    if paths is None:
        paths = {}
    if not isinstance(paths, dict):
        raise TypeError("cell_config['paths'] must be an object/dict.")

    raw = paths.get(path_key, default)
    if raw in (None, ""):
        raise KeyError(
            f"{loader_name} loader requires cell_config['paths'][{path_key!r}]."
        )

    source = Path(str(raw)).expanduser()
    candidates: Iterable[Path]
    if source.is_absolute():
        candidates = (source,)
    else:
        candidates = tuple(base / source for base in candidate_base_dirs(cell_config, base_dir=base_dir))

    resolved_candidates: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        resolved_candidates.append(resolved)
        exists = resolved.is_file() if require_file else resolved.exists()
        if exists:
            return resolved

    attempted = ", ".join(str(path) for path in resolved_candidates)
    kind = "file" if require_file else "path"
    raise FileNotFoundError(
        f"{loader_name} loader: required {kind} for paths.{path_key} was not found. "
        f"Configured value: {raw!r}. Tried: {attempted or source}"
    )
