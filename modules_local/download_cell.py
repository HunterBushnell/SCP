"""
Helpers for downloading Allen Cell Types biophysical bundles.

This module is a maintained copy of the legacy download helper, kept in
`modules_local` so Step 0 can prepare tune directories without relying on
`modules_old`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["list_ADB_models", "download_ADB_cell"]


def _require_allensdk() -> None:
    try:
        from allensdk.api.queries.biophysical_api import BiophysicalApi  # noqa: F401
    except Exception as exc:  # pragma: no cover - import error path
        raise ImportError(
            "The 'allensdk' package is required. Install with:\n"
            "\n"
            "    pip install allensdk\n"
        ) from exc


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _match_name(name: str, target: str, match: str) -> bool:
    name_n = _norm(name)
    target_n = _norm(target)

    if match == "exact":
        ok = name_n == target_n
    elif match == "startswith":
        ok = name_n.startswith(target_n)
    else:
        ok = target_n in name_n

    if ok:
        return True

    synonyms = {
        "perisomatic": ["perisomatic", "peri-somatic"],
        "all active": ["all active", "all-active", "all_active"],
    }
    return any(alias in name_n for alias in synonyms.get(target_n, []))


def list_ADB_models(
    specimen_id: int,
    *,
    filter_type: Optional[str] = None,
    match: str = "contains",
    as_df: bool = False,
    quiet: bool = False,
) -> List[Dict[str, Any]]:
    """
    List available AllenDB neuronal models for a specimen.

    Returns list entries with keys:
    ['model_id','name','description','created','species','structure','tags']
    """
    _require_allensdk()
    from allensdk.api.queries.biophysical_api import BiophysicalApi

    bp = BiophysicalApi()
    models = bp.get_neuronal_models(specimen_id) or []

    rows: List[Dict[str, Any]] = []
    for model in models:
        name = model.get("name", "")
        if filter_type and not _match_name(name, filter_type, match):
            continue
        rows.append(
            {
                "model_id": int(model.get("id")),
                "name": name,
                "description": model.get("description", ""),
                "created": model.get("created_at") or model.get("date_created") or "",
                "species": model.get("species", ""),
                "structure": model.get("structure", ""),
                "tags": model.get("tags", []),
            }
        )

    if not quiet:
        if not rows:
            print(
                f"No models found for specimen_id={specimen_id} "
                f"with filter_type={filter_type!r}."
            )
        else:
            print(f"Models for specimen_id={specimen_id}:")
            for row in rows:
                print(f"  {row['model_id']:>9}  {row['name']}")

    if as_df:
        try:
            import pandas as pd

            return (
                pd.DataFrame(rows)
                .sort_values(["name", "model_id"])
                .reset_index(drop=True)
            )
        except Exception:
            pass

    return rows


def download_ADB_cell(
    specimen_id: int,
    model_type: str = "perisomatic",
    tunes_dir: str = "OriginalFromAllenDB",
    *,
    cache_stimulus: bool = False,
    subdir: Optional[str] = None,
    match: str = "contains",
    quiet: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Download/cache one AllenDB bundle for a specimen/model type.

    Parameters
    ----------
    tunes_dir
        Base output directory.
    subdir
        Optional nested output folder under tunes_dir. If None, writes directly
        into tunes_dir.
    force
        If True, always call cache_data even when files already exist.
    """
    _require_allensdk()
    from allensdk.api.queries.biophysical_api import BiophysicalApi

    bp = BiophysicalApi()
    bp.cache_stimulus = cache_stimulus

    models: List[Dict[str, Any]] = bp.get_neuronal_models(specimen_id) or []
    if not models:
        raise ValueError(f"No neuronal models found for specimen_id={specimen_id}")

    candidates = [m for m in models if _match_name(m.get("name", ""), model_type, match)]
    if not candidates:
        available = [f"{m['id']}: {m.get('name', '')}" for m in models]
        raise ValueError(
            f"No model matched model_type={model_type!r}. Available: {available}"
        )

    chosen = candidates[0]
    model_id = int(chosen["id"])
    model_name = chosen.get("name", "")

    base = Path(tunes_dir)
    target = base / subdir if subdir else base
    target.mkdir(parents=True, exist_ok=True)

    has_files = any(target.iterdir())
    if has_files and not force:
        if not quiet:
            print(f"[download_ADB_cell] Found existing cache at: {target} - skipping download.")
    else:
        if has_files and force and not quiet:
            print(f"[download_ADB_cell] Re-downloading into existing target: {target}")
        bp.cache_data(model_id, working_directory=str(target))
        if not quiet:
            print(
                f"Downloaded model_id={model_id} ({model_name}) "
                f"for specimen_id={specimen_id}"
            )

    files = sorted(str(p) for p in target.rglob("*") if p.is_file())

    return {
        "specimen_id": int(specimen_id),
        "model_id": model_id,
        "model_name": model_name,
        "tunes_dir": str(target),
        "files": files,
    }
