"""Explicit trusted Python builders, including package-relative imports."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

from .paths import resolve_loader_path
from .source_context import declared_source_files, effective_source_root, model_execution, sha256_file, working_directory

_FACTORIES = {}
_IMPORT_ROOTS = {}
_FAILED_IMPORTS = set()


def _options(config):
    options = config.get('python_factory', {})
    if not isinstance(options, dict):
        raise TypeError('python_factory must be an object.')
    name = options.get('callable')
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError('python_factory.callable must name one exported function or class.')
    if not isinstance(options.get('args', []), list) or not isinstance(options.get('kwargs', {}), dict):
        raise TypeError('python_factory.args/kwargs must be a JSON list/object.')
    try:
        json.dumps([options.get('args', []), options.get('kwargs', {})], allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError('Python factory arguments must be finite JSON-compatible values.') from exc
    return options


def validate_config(cell_config, *, base_dir=None):
    _options(cell_config)
    path = resolve_loader_path(cell_config, 'python_factory', base_dir=base_dir, loader_name='python_factory')
    if path.suffix != '.py':
        raise ValueError('paths.python_factory must select a Python .py source file.')
    return {'python_factory': path}


def discover_source_artifacts(cell_config, *, base_dir=None):
    return validate_config(cell_config, base_dir=base_dir)


def _namespace(name, directory):
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(directory)]
        module.__package__ = name
        sys.modules[name] = module


def load_cell(cell_config, *, base_dir=None):
    from scp_pipeline.model.runtime_cell import wrap_cell
    from scp_pipeline.simulation.session_setup import load_mechanisms
    from .source_context import tune_root

    options = _options(cell_config)
    path = validate_config(cell_config, base_dir=base_dir)['python_factory']
    root = effective_source_root(cell_config, base_dir)
    if root is None:
        root = path.parent
        while (root / '__init__.py').is_file():
            root = root.parent
    inventory = declared_source_files(cell_config, base_dir) or sorted(root.rglob('*.py'))
    fingerprint = tuple((str(p), sha256_file(p)) for p in inventory)
    previous = _IMPORT_ROOTS.get(root)
    if path in _FAILED_IMPORTS:
        raise RuntimeError(f'Python source was partially imported: {path}. Restart the Python/Jupyter process.')
    if previous is not None and previous != fingerprint:
        raise RuntimeError(f'Python model sources changed after import: {root}. Restart the Python/Jupyter process.')
    for source in inventory:
        if source.suffix != '.py':
            continue
        parts = source.relative_to(root).with_suffix('').parts
        name = '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)
        imported = sys.modules.get(name)
        origin = getattr(imported, '__file__', None)
        if origin is not None and Path(origin).resolve() != source.resolve():
            raise RuntimeError(f'Python import {name!r} is already bound to {origin}, not {source}. Restart the process to avoid conflicting models.')
    load_mechanisms(tune_root(cell_config, base_dir), cell_config)
    directory = working_directory(cell_config, path, base_dir)
    with model_execution(directory, root):
        try:
            if path not in _FACTORIES:
                _IMPORT_ROOTS[root] = fingerprint
                relative = path.relative_to(root).with_suffix('')
                prefix = '_scp_factory_' + hashlib.sha256(str(root).encode()).hexdigest()[:16]
                _namespace(prefix, root)
                parts = relative.parts
                for count in range(1, len(parts)):
                    name = prefix + '.' + '.'.join(parts[:count])
                    package_dir = root.joinpath(*parts[:count])
                    init = package_dir / '__init__.py'
                    if name not in sys.modules and init.is_file():
                        spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[str(package_dir)])
                        package = importlib.util.module_from_spec(spec)
                        sys.modules[name] = package
                        spec.loader.exec_module(package)
                    else:
                        _namespace(name, package_dir)
                name = prefix + '.' + '.'.join(parts)
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                _IMPORT_ROOTS[root] = fingerprint
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    _FAILED_IMPORTS.add(path)
                    raise
                _FACTORIES[path] = module
            builder = getattr(_FACTORIES[path], options['callable'])
            if not callable(builder):
                raise TypeError('Selected Python factory attribute is not callable.')
            model = builder(*options.get('args', []), **options.get('kwargs', {}))
            cell = wrap_cell(model, section_map=options.get('section_map'), cell_name=cell_config.get('cell_name'))
        except Exception as exc:
            if path not in _FACTORIES:
                _FAILED_IMPORTS.add(path)
            raise RuntimeError(f'Python factory {options["callable"]!r} from {path} failed: {exc}. '
                               'Check imports, arguments, returned sections, and restart after a partial import.') from exc
    cell.config = dict(cell_config)
    cell.loader = 'python_factory'
    cell.source_artifacts = {'python_factory': str(path)}
    cell.working_directory, cell.import_root = directory, root
    return cell
