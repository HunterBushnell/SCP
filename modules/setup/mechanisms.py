"""Shared NEURON mechanism compilation/loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import hashlib
import json
import shutil
import subprocess
import sys


_LOADED_MECHANISM_DLLS: Dict[Path, str] = {}


def load_tune_cell_config(tune_dir: Path) -> Optional[Dict[str, Any]]:
    """Load tune-local cell metadata when it has already been scaffolded."""
    tune_root = Path(tune_dir).expanduser().resolve()
    for candidate in (
        tune_root / "cell_configs" / "cell_config.json",
        tune_root / "cell_config.json",
    ):
        if not candidate.is_file():
            continue
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Could not parse cell configuration {candidate}: {exc}") from exc
        if not isinstance(value, dict):
            raise TypeError(f"Expected a JSON object in {candidate}")
        return value
    return None


def merge_tune_cell_config(
    tune_dir: Path,
    cell_config: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Merge prospective overrides onto stored metadata for safe reruns."""
    stored = load_tune_cell_config(tune_dir)
    if cell_config is None:
        return stored
    if not isinstance(cell_config, dict):
        raise TypeError("cell_config must be an object/dict when provided.")
    if stored is None:
        return dict(cell_config)

    merged = dict(stored)
    merged.update(cell_config)
    stored_paths = stored.get("paths", {})
    supplied_paths = cell_config.get("paths", {})
    if stored_paths is not None and not isinstance(stored_paths, dict):
        raise TypeError("stored cell_config['paths'] must be an object/dict.")
    if supplied_paths is not None and not isinstance(supplied_paths, dict):
        raise TypeError("cell_config['paths'] must be an object/dict.")
    paths = dict(stored_paths or {})
    paths.update(supplied_paths or {})
    merged["paths"] = paths
    return merged


def resolve_modfiles_dir(
    tune_dir: Path,
    cell_config: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Resolve ``paths.modfiles`` relative to a tune; ``null`` disables it.

    When no config is supplied, an existing tune-local ``cell_config.json`` is
    used. This keeps notebook/CLI reruns aligned with the path that Step 1
    previously scaffolded while preserving ``modfiles`` as the legacy default.
    """
    tune_root = Path(tune_dir).expanduser().resolve()
    cell_config = merge_tune_cell_config(tune_root, cell_config)
    raw: Any = "modfiles"
    if isinstance(cell_config, dict):
        paths = cell_config.get("paths", {})
        if paths is not None and not isinstance(paths, dict):
            raise TypeError("cell_config['paths'] must be an object/dict.")
        if isinstance(paths, dict) and "modfiles" in paths:
            raw = paths["modfiles"]
    if raw in (None, ""):
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = tune_root / path
    return path.resolve()


def find_compiled_mechanism_dll(
    tune_dir: Path,
    *,
    cell_config: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    mod_dir = resolve_modfiles_dir(tune_dir, cell_config)
    if mod_dir is None:
        return None
    candidates = [
        mod_dir / "x86_64" / ".libs" / "libnrnmech.so",
        mod_dir / "x86_64" / "libnrnmech.so",
        mod_dir / "nrnmech.dll",
    ]
    for dll in candidates:
        if dll.is_file():
            return dll
    return None


def find_nrnivmodl() -> Optional[str]:
    """Find `nrnivmodl` on PATH or next to the active Python executable."""
    found = shutil.which("nrnivmodl")
    if found:
        return found

    python_bin = Path(sys.executable).expanduser().resolve().parent
    candidate = python_bin / "nrnivmodl"
    if candidate.is_file():
        return str(candidate)
    return None


def mechanism_dll_sha256(dll_path: str | Path) -> str:
    """Return a content hash used to identify a loaded mechanism library safely."""
    path = Path(dll_path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_compiled_mechanism_library(dll_path: str | Path) -> Dict[str, Any]:
    """Load one mechanism DLL once per process, keyed by resolved path and hash.

    NEURON cannot safely replace already registered mechanisms. A repeated load
    is skipped only when this process previously loaded the exact same bytes
    from the exact same path. All ambiguous duplicate/conflict failures require
    a fresh process instead of relying on model-specific mechanism-name probes.
    """
    dll = Path(dll_path).expanduser().resolve()
    if not dll.is_file():
        raise FileNotFoundError(f"Compiled mechanism library not found: {dll}")
    digest = mechanism_dll_sha256(dll)
    previous_digest = _LOADED_MECHANISM_DLLS.get(dll)
    if previous_digest is not None:
        if previous_digest == digest:
            return {
                "dll": str(dll),
                "dll_sha256": digest,
                "loaded": True,
                "dll_preloaded": True,
            }
        raise RuntimeError(
            f"Compiled mechanism library changed after it was loaded: {dll}. "
            "Restart the Python/Jupyter process before loading the rebuilt library."
        )

    from neuron import h

    h.load_file("stdrun.hoc")
    try:
        h.nrn_load_dll(str(dll))
    except RuntimeError as exc:
        raise RuntimeError(
            f"NEURON could not load compiled mechanisms from {dll}. Another "
            "mechanism library may already have registered the same symbols. "
            "Restart the Python/Jupyter process and load only the intended tune's "
            "mechanism library."
        ) from exc

    _LOADED_MECHANISM_DLLS[dll] = digest
    return {
        "dll": str(dll),
        "dll_sha256": digest,
        "loaded": True,
        "dll_preloaded": False,
    }


def compile_modfiles(
    tune_dir: Path,
    *,
    recompile: bool = False,
    load_dll: bool = True,
    allow_missing: bool = False,
    cell_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compile modfiles (nrnivmodl) and optionally load the produced DLL."""
    tune_dir = Path(tune_dir).expanduser().resolve()
    mod_dir = resolve_modfiles_dir(tune_dir, cell_config)
    if mod_dir is None:
        if allow_missing:
            return _skipped_mechanism_summary(None, "paths.modfiles is disabled")
        raise FileNotFoundError("cell_config paths.modfiles is disabled")
    if not mod_dir.is_dir():
        if allow_missing:
            return _skipped_mechanism_summary(mod_dir, "modfiles directory is absent")
        raise FileNotFoundError(f"Missing modfiles directory: {mod_dir}")

    mod_sources = sorted(mod_dir.glob("*.mod"))
    if not mod_sources:
        if allow_missing:
            return _skipped_mechanism_summary(mod_dir, "no .mod source files are present")
        raise FileNotFoundError(f"No .mod source files found in {mod_dir}")

    compiled_dir = mod_dir / "x86_64"
    if recompile and compiled_dir.exists():
        shutil.rmtree(compiled_dir)

    dll = find_compiled_mechanism_dll(tune_dir, cell_config=cell_config)
    compiled_now = False
    if dll is None:
        nrnivmodl = find_nrnivmodl()
        if nrnivmodl is None:
            raise FileNotFoundError(
                "nrnivmodl not found on PATH or next to the active Python "
                f"executable: {sys.executable}"
            )
        subprocess.check_call([nrnivmodl], cwd=str(mod_dir))
        compiled_now = True
        dll = find_compiled_mechanism_dll(tune_dir, cell_config=cell_config)

    if dll is None:
        raise FileNotFoundError(
            "nrnivmodl finished but compiled mechanism library was not found under "
            f"{compiled_dir}"
        )

    loaded = False
    dll_preloaded = False
    dll_sha256 = mechanism_dll_sha256(dll)
    if load_dll:
        load_summary = load_compiled_mechanism_library(dll)
        loaded = bool(load_summary["loaded"])
        dll_preloaded = bool(load_summary["dll_preloaded"])
        dll_sha256 = str(load_summary["dll_sha256"])

    return {
        "status": "ok",
        "modfiles_dir": str(mod_dir),
        "compiled_dir": str(compiled_dir),
        "dll": str(dll),
        "dll_sha256": dll_sha256,
        "compiled_now": bool(compiled_now),
        "nrnivmodl": find_nrnivmodl(),
        "loaded": bool(loaded),
        "dll_preloaded": bool(dll_preloaded),
    }


def _skipped_mechanism_summary(mod_dir: Optional[Path], reason: str) -> Dict[str, Any]:
    """Return the stable result shape for models using built-in mechanisms only."""
    return {
        "status": "skipped",
        "reason": str(reason),
        "modfiles_dir": None if mod_dir is None else str(mod_dir),
        "compiled_dir": None,
        "dll": None,
        "dll_sha256": None,
        "compiled_now": False,
        "nrnivmodl": find_nrnivmodl(),
        "loaded": False,
        "dll_preloaded": False,
    }
