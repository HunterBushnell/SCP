"""Shared model roots, declared source inventories, and scoped execution."""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from threading import RLock
import hashlib
import os
import sys


EXCLUDED_PARTS = frozenset({
    '.git', '.hg', '.svn', '__pycache__', '.pytest_cache', '.mypy_cache',
    '.ipynb_checkpoints', 'x86_64', 'aarch64',
    '.scp_build', 'output_data', 'model_artifacts', 'restore_backups',
    'notebook_exports', 'tuning_exports', 'output', 'outputs', 'results', 'runs',
})
EXCLUDED_FILES = frozenset({'nrnmech.dll', 'special', 'libnrnmech.so', 'libnrnmech.dylib'})
_MODEL_LOCK = RLock()


def within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tune_root(config, base_dir=None) -> Path:
    paths = config.get('paths', {}) or {}
    return Path(base_dir or paths.get('tune_dir') or config.get('tune_dir') or Path.cwd()).expanduser().resolve()


def source_options(config):
    options = config.get('model_source')
    if options is None:
        return None
    if not isinstance(options, dict):
        raise TypeError('model_source must be an object.')
    if options.get('mode', 'copy') not in {'copy', 'reference'}:
        raise ValueError('model_source.mode must be copy or reference.')
    if not options.get('source_root'):
        raise ValueError('model_source.source_root is required.')
    folders, files = options.get('folders', []), options.get('files', [])
    if (not isinstance(folders, list) or not isinstance(files, list) or not (folders or files)
            or any(not isinstance(p, str) or not p for p in folders + files)):
        raise ValueError('model_source.folders/files must declare at least one relative native input.')
    for value in folders + files:
        path = relative_path(value)
        if any(part in EXCLUDED_PARTS for part in path.parts):
            raise ValueError(f'Source folder is excluded metadata, build, or output: {value!r}.')
    return options


def original_source_root(config, base_dir=None):
    root = Path(source_options(config)['source_root']).expanduser()
    return (root if root.is_absolute() else tune_root(config, base_dir) / root).resolve()


def same_source_selection(left, right, base_dir=None):
    if not left or not right:
        return left == right
    if original_source_root({'model_source': left}, base_dir) != original_source_root({'model_source': right}, base_dir):
        return False
    return all({relative_path(p) for p in left.get(key, [])} ==
               {relative_path(p) for p in right.get(key, [])} for key in ('folders', 'files'))


def guard_source_destinations(config, base_dir=None, destinations=()):
    """A broad source root is safe; selected native trees must remain disjoint."""
    options = source_options(config)
    root = original_source_root(config, base_dir)
    tune = tune_root(config, base_dir)
    selected = [(root / relative_path(p)).resolve() for p in options.get('folders', [])]
    selected += [(root / relative_path(p)).resolve() for p in options.get('files', [])]
    for source in selected:
        if not within(source, root):
            raise ValueError(f'Selected native input escapes its source root: {source}')
        for destination in (tune, *(Path(p).resolve() for p in destinations)):
            if within(destination, source) or within(source, destination):
                raise ValueError(f'Source/destination overlap: {source} and workspace/output {destination}. Select narrower native inputs.')


def relative_path(value) -> Path:
    path = Path(str(value))
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Expected a safe root-relative path, got {value!r}.')
    return path


def effective_source_root(config, base_dir=None) -> Path | None:
    options = source_options(config)
    if options is None:
        return None
    if options.get('mode', 'copy') == 'reference':
        guard_source_destinations(config, base_dir)
        root = original_source_root(config, base_dir)
        return root
    return tune_root(config, base_dir) / 'model_sources'


def source_relative_path(config, value, base_dir=None) -> Path:
    options = source_options(config)
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        original = original_source_root(config, base_dir)
        try:
            path = path.resolve().relative_to(original)
        except ValueError as exc:
            raise ValueError(f'Model path {value!r} is outside model_source.source_root {original}.') from exc
    relative = relative_path(path)
    if (not any(within(relative, relative_path(folder)) for folder in options.get('folders', []))
            and relative not in {relative_path(value) for value in options.get('files', [])}):
        raise ValueError(f'Model path {value!r} is outside the declared model_source.folders.')
    if any(part in EXCLUDED_PARTS for part in relative.parts) or relative.name in EXCLUDED_FILES:
        raise ValueError(f'Model source path is an excluded build/output: {value!r}.')
    return relative


def declared_source_files(config, base_dir=None) -> list[Path]:
    root = effective_source_root(config, base_dir)
    if root is None:
        return []
    options = source_options(config)
    result = {}
    for folder in sorted(set(options.get('folders', []))):
        start = root / relative_path(folder)
        if start.is_symlink():
            raise ValueError(f'Directory symlinks are not copied; declare the real source folder: {start}')
        if not within(start.resolve(), root.resolve()):
            raise ValueError(f'Source folder escapes the model root: {folder!r}.')
        if not start.is_dir():
            raise FileNotFoundError(f'Declared model source folder not found: {start}')
        for current, dirs, files in os.walk(start, followlinks=False):
            dirs[:] = sorted(name for name in dirs if name not in EXCLUDED_PARTS)
            for name in list(dirs):
                item = Path(current) / name
                if item.is_symlink():
                    raise ValueError(f'Directory symlinks are not copied; declare their source explicitly: {item}')
            for name in sorted(files):
                path = Path(current) / name
                if name in EXCLUDED_FILES or name in EXCLUDED_PARTS or path.suffix in {'.pyc', '.pyo', '.o', '.so', '.dll', '.dylib'}:
                    continue
                resolved = path.resolve()
                if not within(resolved, root.resolve()):
                    raise ValueError(f'Source symlink escapes the model root: {path}')
                relative = path.relative_to(root)
                if not any(within(resolved.relative_to(root.resolve()), relative_path(f)) for f in options.get('folders', [])):
                    raise ValueError(f'Source symlink targets an undeclared folder: {path}')
                result[relative.as_posix()] = path
    for value in sorted(set(options.get('files', []))):
        relative = relative_path(value)
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f'Declared model source file not found: {path}')
        if path.is_symlink() or not within(path.resolve(), root.resolve()):
            raise ValueError(f'Explicit source file must not escape or alias its root: {path}')
        if (any(part in EXCLUDED_PARTS for part in relative.parts) or path.name in EXCLUDED_FILES
                or path.suffix in {'.pyc', '.pyo', '.o', '.so', '.dll', '.dylib'}):
            raise ValueError(f'Explicit model source is an excluded build/output: {path}')
        result[relative.as_posix()] = path
    return [result[key] for key in sorted(result)]


def working_directory(config, entry_path=None, base_dir=None):
    options = source_options(config)
    root = effective_source_root(config, base_dir)
    if options is not None and options.get('working_directory') is not None:
        result = root / relative_path(options['working_directory'])
        if not within(result.resolve(), root.resolve()):
            raise ValueError('model_source.working_directory escapes its model root.')
    else:
        result = Path(entry_path).parent if entry_path is not None else None
    if result is not None and not result.is_dir():
        raise FileNotFoundError(f'Model working directory not found: {result}')
    return result


def declared_source_directories(config, base_dir=None):
    """Keep empty native directories as well as file-containing layout."""
    declared_source_files(config, base_dir)  # Apply the same symlink/root checks.
    root = effective_source_root(config, base_dir)
    result = set()
    for folder in source_options(config).get('folders', []):
        for current, dirs, _ in os.walk(root / relative_path(folder), followlinks=False):
            dirs[:] = sorted(name for name in dirs if name not in EXCLUDED_PARTS)
            result.add(Path(current).relative_to(root).as_posix())
    for path in declared_source_files(config, base_dir):
        parent = path.relative_to(root).parent
        while parent != Path('.'):
            result.add(parent.as_posix())
            parent = parent.parent
    return sorted(result)


@contextmanager
def model_execution(directory=None, import_root=None):
    """Serialize NEURON work and restore cwd/import/bytecode process settings."""
    with _MODEL_LOCK:
        old_cwd, old_path, old_bytecode = Path.cwd(), list(sys.path), sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            if import_root is not None:
                sys.path.insert(0, str(import_root))
            if directory is not None:
                os.chdir(directory)
            yield
        finally:
            os.chdir(old_cwd)
            sys.path[:] = old_path
            sys.dont_write_bytecode = old_bytecode


def cell_execution(function):
    @wraps(function)
    def scoped(cell, *args, **kwargs):
        with model_execution(getattr(cell, 'working_directory', None), getattr(cell, 'import_root', None)):
            return function(cell, *args, **kwargs)
    return scoped
