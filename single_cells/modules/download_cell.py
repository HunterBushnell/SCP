
"""
Lightweight helpers for downloading AllenDB biophysical model bundles.

Functions
---------
- list_ADB_models(specimen_id, filter_type=None, match='contains', as_df=False, quiet=False)
- download_ADB_cell(specimen_id, model_type='perisomatic', tunes_dir='OriginalFromAllenDB',
                    cache_stimulus=False, subdir=True, match='contains', quiet=False)

Notes
-----
- Requires `allensdk` (install once: `pip install allensdk`)
- Does NOT require NEURON just to download/cache files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["list_ADB_models", "download_ADB_cell"]


def _require_allensdk():
    try:
        # Lazy import so the module loads even if allensdk isn't installed yet
        from allensdk.api.queries.biophysical_api import BiophysicalApi  # noqa: F401
    except Exception as e:
        raise ImportError(
            "The 'allensdk' package is required. Install with:\n\n"
            "    pip install allensdk\n"
        ) from e


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _match_name(name: str, target: str, match: str) -> bool:
    """Match helper: 'contains' | 'startswith' | 'exact' (+ a couple synonyms)."""
    name = _norm(name)
    target = _norm(target)

    if match == "exact":
        ok = name == target
    elif match == "startswith":
        ok = name.startswith(target)
    else:  # default: contains
        ok = (target in name)

    if ok:
        return True

    # light synonyms for common Allen bundle names
    synonyms = {
        "perisomatic": ["perisomatic", "peri-somatic"],
        "all active": ["all active", "all-active", "all_active"],
    }
    return any(alias in name for alias in synonyms.get(target, []))


def list_ADB_models(
    specimen_id: int,
    *,
    filter_type: Optional[str] = None,   # e.g., "perisomatic" or "all active"
    match: str = "contains",             # 'contains' | 'startswith' | 'exact'
    as_df: bool = False,                 # return pandas DataFrame if available
    quiet: bool = False,
) -> List[Dict[str, Any]]:               # or a DataFrame when as_df=True
    """
    List available AllenDB neuronal model bundles for a specimen.

    Returns a list of dicts with keys:
      ['model_id','name','description','created','species','structure','tags']

    If as_df=True (and pandas is installed), returns a DataFrame.
    """
    _require_allensdk()
    from allensdk.api.queries.biophysical_api import BiophysicalApi  # type: ignore

    bp = BiophysicalApi()
    models = bp.get_neuronal_models(specimen_id) or []

    rows: List[Dict[str, Any]] = []
    for m in models:
        name = m.get("name", "")
        if filter_type and not _match_name(name, filter_type, match):
            continue
        rows.append({
            "model_id": int(m.get("id")),
            "name": name,
            "description": m.get("description", ""),
            "created": m.get("created_at") or m.get("date_created") or "",
            "species": m.get("species", ""),
            "structure": m.get("structure", ""),
            "tags": m.get("tags", []),
        })

    if not quiet:
        if not rows:
            print(f"No models found for specimen_id={specimen_id} "
                  f"with filter_type={filter_type!r}.")
        else:
            print(f"Models for specimen_id={specimen_id}:")
            for r in rows:
                print(f"  {r['model_id']:>9}  {r['name']}")

    if as_df:
        try:
            import pandas as pd  # type: ignore
            return pd.DataFrame(rows).sort_values(["name", "model_id"]).reset_index(drop=True)  # type: ignore[return-value]
        except Exception:
            # fall back to list of dicts
            pass

    return rows


def download_ADB_cell(
    specimen_id: int,
    model_type: str = "perisomatic",       # or "all active"
    tunes_dir: str = "OriginalFromAllenDB", # base folder to write into
    *,
    cache_stimulus: bool = False,          # large NWB; False by default
    subdir: bool = None,                   # put files in tunes_dir/<specimen>_<model_type>
    match: str = "contains",               # 'contains' | 'startswith' | 'exact'
    quiet: bool = False,
) -> Dict[str, Any]:
    """
    Download an AllenDB biophysical model bundle for a given specimen.

    Returns a dict:
      {
        'specimen_id': int,
        'model_id': int,
        'model_name': str,
        'tunes_dir': str,     # target directory where files were cached
        'files': List[str]   # all file paths under tunes_dir (recursive)
      }
    """
    _require_allensdk()
    from allensdk.api.queries.biophysical_api import BiophysicalApi  # type: ignore

    bp = BiophysicalApi()
    bp.cache_stimulus = cache_stimulus

    # 1) Discover models
    models: List[Dict[str, Any]] = bp.get_neuronal_models(specimen_id)
    if not models:
        raise ValueError(f"No neuronal models found for specimen_id={specimen_id}")

    # 2) Choose by model_type + match rule
    candidates = [m for m in models if _match_name(m.get("name", ""), model_type, match)]
    if not candidates:
        available = [f"{m['id']}: {m.get('name','')}" for m in models]
        raise ValueError(
            f"No model matched model_type='{model_type}'. "
            f"Available: {available}"
        )

    chosen = candidates[0]
    model_id   = int(chosen["id"])
    model_name = chosen.get("name", "")

    # 3) Prepare output directory
    base = Path(tunes_dir)
    target = base / subdir if subdir is not None else base
    target.mkdir(parents=True, exist_ok=True)

    # 4) Cache into `target`
    if any(target.iterdir()):
        print(f"[download_ADB_cell] Found existing cache at: {target} — skipping download.")
    
    else:
        bp.cache_data(model_id, working_directory=str(target))
        if not quiet:
            print(f"Downloaded model_id={model_id} ({model_name}) for specimen_id={specimen_id}")
    
    # Get cell model files
    files = sorted(str(p) for p in target.rglob("*") if p.is_file())

    return {
        "specimen_id": specimen_id,
        "model_id": model_id,
        "model_name": model_name,
        "tunes_dir": str(target),
        "files": files,
    }



######## Could add downloading other cell types/databases #########


