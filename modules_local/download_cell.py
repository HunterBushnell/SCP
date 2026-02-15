"""
Helpers for downloading Allen Cell Types biophysical bundles.

This module provides the Step 0 download interface used by the current
`modules_local` pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["list_ADB_models", "download_ADB_cell"]
DOWNLOAD_META_FILENAME = ".adb_download_meta.json"


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


def _canonical_model_type(value: Optional[str]) -> str:
    token = _norm(value)
    if token in {"all active", "all-active", "all_active"}:
        return "all active"
    if token in {"perisomatic", "peri-somatic"}:
        return "perisomatic"
    return token


def _same_model_type(a: Optional[str], b: Optional[str]) -> bool:
    return _canonical_model_type(a) == _canonical_model_type(b)


def _load_download_meta(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _infer_specimen_from_fit_filename(target: Path) -> Optional[int]:
    for fit_json in sorted(target.glob("*_fit.json")):
        stem = fit_json.stem
        if stem.endswith("_fit"):
            specimen = stem[: -len("_fit")]
            if specimen.isdigit():
                return int(specimen)
    return None


def _infer_model_type_from_fit_json(target: Path) -> Optional[str]:
    fit_candidates = sorted(target.glob("*_fit.json"))
    if not fit_candidates:
        return None
    try:
        fit_data = json.loads(fit_candidates[0].read_text())
    except Exception:
        return None

    passive = fit_data.get("passive", [])
    if not isinstance(passive, list) or not passive:
        return None
    p0 = passive[0]
    if not isinstance(p0, dict):
        return None

    has_e_pas = "e_pas" in p0
    cm_cfg = p0.get("cm")
    has_cm = isinstance(cm_cfg, list) and len(cm_cfg) > 0
    if has_e_pas and has_cm:
        return "perisomatic"
    return "all active"


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
    downloaded_now = False
    meta_path = target / DOWNLOAD_META_FILENAME
    existing_meta = _load_download_meta(meta_path)
    if has_files and not force:
        mismatch: List[str] = []
        if existing_meta:
            meta_specimen = existing_meta.get("specimen_id")
            if meta_specimen is not None:
                try:
                    if int(meta_specimen) != int(specimen_id):
                        mismatch.append(
                            f"specimen_id={meta_specimen} (existing) != {int(specimen_id)} (requested)"
                        )
                except Exception:
                    pass
            meta_type = existing_meta.get("model_type")
            if meta_type is not None and not _same_model_type(meta_type, model_type):
                mismatch.append(
                    f"model_type={meta_type!r} (existing) != {model_type!r} (requested)"
                )
        else:
            inferred_specimen = _infer_specimen_from_fit_filename(target)
            inferred_model_type = _infer_model_type_from_fit_json(target)
            if (
                inferred_specimen is not None
                and inferred_specimen != int(specimen_id)
            ):
                mismatch.append(
                    f"specimen_id={inferred_specimen} inferred from existing fit JSON "
                    f"!= {int(specimen_id)} (requested)"
                )
            if (
                inferred_model_type is not None
                and not _same_model_type(inferred_model_type, model_type)
            ):
                mismatch.append(
                    f"model_type={inferred_model_type!r} inferred from existing fit JSON "
                    f"!= {model_type!r} (requested)"
                )

        if mismatch:
            details = "\n".join(f"  - {item}" for item in mismatch)
            raise ValueError(
                "Target tune directory already contains another model bundle:\n"
                f"{details}\n"
                f"Target: {target}\n"
                "Use a different tune folder or rerun with force=True to overwrite."
            )

        if not quiet:
            print(f"[download_ADB_cell] Found existing cache at: {target} - skipping download.")
        if existing_meta is None:
            inferred_specimen = _infer_specimen_from_fit_filename(target)
            inferred_model_type = _infer_model_type_from_fit_json(target)
            if inferred_specimen is not None or inferred_model_type is not None:
                inferred_payload: Dict[str, Any] = {
                    "specimen_id": inferred_specimen if inferred_specimen is not None else int(specimen_id),
                    "model_type": inferred_model_type,
                    "model_id": None,
                    "model_name": None,
                    "inferred_from_fit_json": True,
                }
                try:
                    meta_path.write_text(json.dumps(inferred_payload, indent=2, sort_keys=True))
                    existing_meta = inferred_payload
                except Exception:
                    pass
    else:
        if has_files and force and not quiet:
            print(f"[download_ADB_cell] Re-downloading into existing target: {target}")
        bp.cache_data(model_id, working_directory=str(target))
        downloaded_now = True
        if not quiet:
            print(
                f"Downloaded model_id={model_id} ({model_name}) "
                f"for specimen_id={specimen_id}"
            )
    if downloaded_now:
        meta_payload = {
            "specimen_id": int(specimen_id),
            "model_type": model_type,
            "model_id": model_id,
            "model_name": model_name,
        }
        try:
            meta_path.write_text(json.dumps(meta_payload, indent=2, sort_keys=True))
        except Exception:
            pass

    files = sorted(str(p) for p in target.rglob("*") if p.is_file())

    return {
        "specimen_id": int(specimen_id),
        "model_id": model_id,
        "model_name": model_name,
        "tunes_dir": str(target),
        "meta_path": str(meta_path) if meta_path.is_file() else None,
        "files": files,
    }
