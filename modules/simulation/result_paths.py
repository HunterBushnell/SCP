from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np


def _sha256_file(path: Path) -> str:
    hsh = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hsh.update(chunk)
    return hsh.hexdigest()


def _resolve_tune_path(sim_cfg: Dict[str, Any]) -> Optional[Path]:
    tune_dir = sim_cfg.get("tune_dir")
    if tune_dir:
        try:
            return Path(str(tune_dir)).expanduser().resolve()
        except Exception:
            return None

    cell = sim_cfg.get("cell")
    tune = sim_cfg.get("tune")
    if cell and tune:
        base = Path(__file__).resolve().parents[2]
        return base / "cells" / str(cell) / "tunes" / str(tune)
    return None


def _find_fit_json_path(sim_cfg: Dict[str, Any]) -> Optional[Path]:
    tune_path = _resolve_tune_path(sim_cfg)
    if tune_path is None or not tune_path.is_dir():
        return None

    manifest_path = tune_path / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest_data = json.loads(manifest_path.read_text())
            biophys = manifest_data.get("biophys", [])
            if isinstance(biophys, list):
                for entry in biophys:
                    if not isinstance(entry, dict):
                        continue
                    model_file = entry.get("model_file")
                    model_file_items = model_file if isinstance(model_file, list) else [model_file]
                    for item in model_file_items:
                        if not isinstance(item, str):
                            continue
                        cand = Path(item)
                        if not cand.name.endswith("_fit.json"):
                            continue
                        cand = (tune_path / cand).resolve() if not cand.is_absolute() else cand.resolve()
                        if cand.is_file():
                            return cand
        except Exception:
            pass

    fit_candidates = sorted(tune_path.glob("*_fit.json"))
    if fit_candidates:
        return fit_candidates[0].resolve()
    return None


def _copy_fit_json_sidecar(sim_cfg: Dict[str, Any], run_dir: Path) -> Optional[Dict[str, str]]:
    if sim_cfg.get("save_fit_json_sidecar", True) is False:
        return None

    fit_path = _find_fit_json_path(sim_cfg)
    if fit_path is None:
        return None

    target = run_dir / fit_path.name
    try:
        shutil.copy2(fit_path, target)
    except Exception:
        return None

    fit_info: Dict[str, str] = {
        "filename": target.name,
        "source_path": str(fit_path),
    }
    try:
        fit_info["sha256"] = _sha256_file(target)
    except Exception:
        pass
    return fit_info


def _build_output_path(
    sim_cfg: Dict[str, Any],
    base_dir: Union[str, Path] = "output_data",
) -> Optional[Path]:
    """
    Build a unique output path based on sim_cfg and base_dir.
    Returns None if sim_cfg['output'] is None/empty.
    1) base_dir / {output_stem}/
    2) filename: {cell}_{tune}_{output_stem}.{suffix}
    3) If run folder exists, append _1, _2, ... to output_stem.
    4) suffix based on sim_cfg['output_format']: 'pickle' -> .pkl
    """
    if sim_cfg.get("save_output") is False:
        return None

    output_stem = sim_cfg.get("output_stem")
    if output_stem not in (None, ""):
        sim_cfg["output"] = output_stem
    output_stem = sim_cfg.get("output")
    if not output_stem:
        return None

    cell = sim_cfg.get("cell", "cell")
    tune = sim_cfg.get("tune", "tune")
    fmt = sim_cfg.get("output_format", "pickle")

    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    suffix = ".npz" if fmt == "npz" else ".pkl"

    run_stem = str(output_stem)
    run_dir = base / run_stem
    idx = 1
    while run_dir.exists():
        run_stem = f"{output_stem}_{idx}"
        run_dir = base / run_stem
        idx += 1

    if run_stem != output_stem:
        sim_cfg["output"] = run_stem

    run_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{cell}_{tune}_{run_stem}"
    return run_dir / (stem + suffix)


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
