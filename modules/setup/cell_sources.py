"""Cell-source setup helpers for Step 1.

Step 1 supports source-specific adapters for getting model files into a tune
directory. ADB is the first-class bundled adapter; existing local models are
treated as already staged and validated later by the config/loader checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .adb import download_ADB_cell


def setup_cell_source(
    *,
    tune_dir: Path,
    source_type: str = "adb",
    specimen_id: Optional[int] = None,
    model_type: str = "perisomatic",
    do_download: bool = True,
    force_download: bool = False,
    cache_stimulus: bool = False,
    download_match: str = "contains",
) -> Dict[str, Any]:
    """Prepare model source files for a tune directory."""
    tune_dir = Path(tune_dir).expanduser().resolve()
    tune_dir.mkdir(parents=True, exist_ok=True)

    source = str(source_type or "existing").strip().lower()
    if source in {"existing", "local", "none", "skip"}:
        return {
            "status": "ok",
            "source_type": "existing",
            "tune_dir": str(tune_dir),
            "downloaded": False,
        }

    if source != "adb":
        raise ValueError(
            f"Unsupported Step 1 source_type={source_type!r}. "
            "Supported values are 'adb' and 'existing'."
        )

    if not do_download:
        return {
            "status": "skipped",
            "source_type": "adb",
            "tune_dir": str(tune_dir),
            "downloaded": False,
        }

    if specimen_id is None:
        raise ValueError("specimen_id is required when source_type='adb'.")

    dl_info = download_ADB_cell(
        specimen_id=int(specimen_id),
        model_type=model_type,
        tunes_dir=str(tune_dir),
        subdir=None,
        cache_stimulus=cache_stimulus,
        match=download_match,
        quiet=False,
        force=force_download,
    )
    return {
        "status": "ok",
        "source_type": "adb",
        "model_id": int(dl_info.get("model_id")),
        "model_name": dl_info.get("model_name"),
        "n_files": int(len(dl_info.get("files", []))),
        "downloaded": True,
    }
