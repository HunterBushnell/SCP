"""Shared NEURON mechanism compilation/loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import hashlib
import json
import shutil
import subprocess
import sys
import importlib.metadata
import os
import platform

from scp_pipeline.loaders.source_context import (EXCLUDED_PARTS, declared_source_files,
    effective_source_root, sha256_file, source_relative_path, within)


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
        elif cell_config.get('model_source') is not None:
            raw = None
    if raw in (None, ""):
        return None
    path = Path(str(raw)).expanduser()
    model_root = effective_source_root(cell_config or {}, tune_root)
    if model_root is not None:
        path = model_root / source_relative_path(cell_config, path, tune_root)
        if not within(path.resolve(), model_root.resolve()):
            raise ValueError('paths.modfiles escapes the selected model root.')
    if not path.is_absolute():
        path = tune_root / path
    return path.resolve()


def _mechanism_build_context(tune_dir, cell_config=None):
    tune = Path(tune_dir).expanduser().resolve()
    config = merge_tune_cell_config(tune, cell_config) or {}
    source = resolve_modfiles_dir(tune, config)
    if source is None or within(source, tune) or not source.is_dir():
        return source, None
    root = effective_source_root(config, tune) or source
    if config.get('model_source'):
        from scp_pipeline.loaders.source_context import guard_source_destinations
        guard_source_destinations(config, tune)
    elif within(tune, source):
        raise ValueError('The managed build workspace must be outside the external MOD source tree.')
    files = declared_source_files(config, tune)
    if not files:
        files = sorted(p for p in source.rglob('*') if p.is_file() and
                       not any(part in EXCLUDED_PARTS for part in p.relative_to(source).parts) and
                       p.suffix not in {'.so', '.dll', '.dylib', '.o', '.pyc', '.pyo'})
    for path in files:
        if not within(path.resolve(), root.resolve()):
            raise ValueError(f'Mechanism dependency escapes its source root: {path}')
    compiler = shutil.which('c++') or shutil.which('cc')
    tool = find_nrnivmodl()
    try:
        neuron_version = importlib.metadata.version('neuron')
    except importlib.metadata.PackageNotFoundError:
        neuron_version = 'unavailable'
    identity = {'neuron': neuron_version, 'platform': platform.machine(), 'python': sys.version,
                'compiler': str(Path(compiler).resolve()) if compiler else None,
                'compiler_stat': [Path(compiler).stat().st_size, Path(compiler).stat().st_mtime_ns] if compiler else None,
                'nrnivmodl': tool, 'nrnivmodl_sha256': sha256_file(Path(tool)) if tool else None,
                'flags': {name: os.environ.get(name) for name in ('CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS')},
                'sources': [(p.relative_to(root).as_posix(), sha256_file(p)) for p in files]}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = tune / '.scp_build' / digest
    if not within(cache.resolve(), tune):
        raise ValueError('Managed mechanism cache escapes the selected tune.')
    build_dir = cache / 'sources' / source.relative_to(root)
    return build_dir, {'root': root, 'files': files, 'cache': cache, 'identity': identity}


def find_compiled_mechanism_dll(
    tune_dir: Path,
    *,
    cell_config: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    mod_dir, _ = _mechanism_build_context(tune_dir, cell_config)
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


def mechanism_build_directory(tune_dir, cell_config=None):
    """Return the safe build directory without compiling or creating it."""
    return _mechanism_build_context(tune_dir, cell_config)[0]


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


def _compile_modfiles_unlocked(
    tune_dir: Path,
    *,
    recompile: bool = False,
    load_dll: bool = True,
    allow_missing: bool = False,
    compile_missing: bool = True,
    cell_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Ensure a mechanism DLL exists and optionally load it.

    ``compile_missing=False`` supports config-only Step 1 reruns: SCP may load
    an already-built library, but it will never invoke ``nrnivmodl``.
    """
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

    build_dir, external = _mechanism_build_context(tune_dir, cell_config)
    compiled_dir = build_dir / "x86_64"
    if recompile:
        existing_dll = find_compiled_mechanism_dll(
            tune_dir,
            cell_config=cell_config,
        )
        if (
            existing_dll is not None
            and existing_dll.resolve() in _LOADED_MECHANISM_DLLS
        ):
            raise RuntimeError(
                "Cannot recompile a mechanism library that is already loaded in "
                f"this Python/Jupyter process: {existing_dll.resolve()}. Restart "
                "the process, enable recompilation before preparing the model, "
                "and rerun this step."
            )
        if compiled_dir.exists():
            shutil.rmtree(compiled_dir)

    dll = find_compiled_mechanism_dll(tune_dir, cell_config=cell_config)
    compiled_now = False
    if dll is None:
        if not compile_missing:
            raise FileNotFoundError(
                "Compiled mechanism library was not found under "
                f"{compiled_dir}, and mechanism compilation is disabled. "
                "Run Step 1 without --no-compile (or compile this tune's "
                "configured paths.modfiles directory first)."
            )
        nrnivmodl = find_nrnivmodl()
        if nrnivmodl is None:
            raise FileNotFoundError(
                "nrnivmodl not found on PATH or next to the active Python "
                f"executable: {sys.executable}"
            )
        if external is not None:
            hashes = dict(external['identity']['sources'])
            for source in external['files']:
                target = external['cache'] / 'sources' / source.relative_to(external['root'])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target, follow_symlinks=True)
                if sha256_file(target) != hashes[source.relative_to(external['root']).as_posix()]:
                    raise RuntimeError(f'Mechanism sources changed during build staging: {source}. Retry preparation.')
            (external['cache'] / 'build_manifest.json').write_text(
                json.dumps(external['identity'], indent=2) + '\n', encoding='utf-8')
        subprocess.check_call([nrnivmodl], cwd=str(build_dir))
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


def compile_modfiles(tune_dir, *, recompile=False, load_dll=True, allow_missing=False,
                     compile_missing=True, cell_config=None):
    """Prepare mechanisms, serializing shared external builds across workers."""
    tune = Path(tune_dir).expanduser().resolve()
    source = resolve_modfiles_dir(tune, cell_config)
    options = dict(recompile=recompile, load_dll=load_dll, allow_missing=allow_missing,
                   compile_missing=compile_missing, cell_config=cell_config)
    if (source is None or within(source, tune) or not source.is_dir()
            or not any(source.glob('*.mod')) or not compile_missing):
        return _compile_modfiles_unlocked(tune, **options)
    locks = tune / '.scp_build' / '.locks'
    if not within(locks.resolve(), tune):
        raise ValueError('Mechanism build lock escapes the selected tune.')
    locks.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(str(source).encode()).hexdigest() + '.lock'
    with (locks / name).open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            lock.write(b'0')
            lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return _compile_modfiles_unlocked(tune, **options)
        finally:
            if os.name == 'nt':
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


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
