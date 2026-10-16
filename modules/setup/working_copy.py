"""File-only working copies, deliberately independent of current upstream files."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile

from scp_pipeline.loaders.source_context import EXCLUDED_PARTS, EXCLUDED_FILES, within, sha256_file
from scp_pipeline.workspaces import _name, infer_workspace, apply_workspace_metadata

_EXCLUDED = EXCLUDED_PARTS | {'exports', 'archives', 'archive', 'backups', '.cache', '.venv', 'venv'}
_SUFFIXES = {'.pyc', '.pyo', '.o', '.so', '.dll', '.dylib'}
_PATH_KEYS = {'file', 'csv', 'path', 'manifest', 'hoc_template', 'python_factory', 'modfiles',
              'spike_file', 'spikes_file', 'rates_file', 'input_file', 'data_file', 'file_path'}


def _excluded(relative):
    return (any(part in _EXCLUDED for part in relative.parts)
            or relative.name in EXCLUDED_FILES or relative.suffix in _SUFFIXES)


def _config_inputs(value, base, *, prefix='', data_root=None):
    """Declared data paths only; never reinterpret arbitrary constructor strings."""
    if not isinstance(value, dict) or value.get('enabled') is False or value.get('state') is False:
        return
    for key, item in value.items():
        label = f'{prefix}.{key}' if prefix else key
        if key in {'args', 'kwargs', 'constructor_args', 'model_source', 'workspace_identity'}:
            continue
        if isinstance(item, dict):
            yield from _config_inputs(item, base, prefix=label, data_root=data_root)
        elif key in _PATH_KEYS and isinstance(item, str) and item:
            path = Path(item).expanduser()
            if key == 'path' and not path.is_absolute() and data_root and path.parts[0] in {'cells', 'projects', 'external_data'}:
                from scp_pipeline.data_paths import resolve_data_alias
                resolved = resolve_data_alias(path, data_root)
            else:
                resolved = path if path.is_absolute() else base / path
            yield label, resolved.resolve()
        elif key == 'group_files' and isinstance(item, list):
            for name in item:
                if isinstance(name, str):
                    yield label, (base / name).resolve()
        elif isinstance(item, list):
            for nested in item:
                if isinstance(nested, dict):
                    yield from _config_inputs(nested, base, prefix=label, data_root=data_root)


def _remap_paths(value, source, target):
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        if key in {'args', 'kwargs', 'constructor_args', 'model_source'}:
            continue
        if isinstance(item, dict):
            _remap_paths(item, source, target)
        elif key in _PATH_KEYS | {'tune_dir', 'output_dir'} and isinstance(item, str):
            path = Path(item)
            if path.is_absolute() and within(path.resolve(), source):
                value[key] = str(target / path.resolve().relative_to(source))
        elif isinstance(item, list):
            for index, nested in enumerate(item):
                if key == 'group_files' and isinstance(nested, str):
                    path = Path(nested)
                    if path.is_absolute() and within(path.resolve(), source):
                        item[index] = str(target / path.resolve().relative_to(source))
                    continue
                _remap_paths(nested, source, target)


def copy_model_workspace(source_tune_dir, target_tune_name='tuned', *, dry_run=True, source_mode=None):
    """Preview/apply a source/config/data-only sibling copy. Never refresh snapshots."""
    workspace = infer_workspace(source_tune_dir)
    source = workspace.tune_dir
    if not source.is_dir():
        raise FileNotFoundError(f'Source tune directory not found: {source}')
    name = _name(target_tune_name, 'target_tune_name')
    if Path(name).name != name or name in {'.', '..'} or '\\' in name:
        raise ValueError('Working-copy name must be a single safe sibling directory name.')
    target = source.parent / name
    if target.is_symlink() or target.resolve().parent != source.parent or target.exists():
        raise ValueError(f'Working-copy destination already exists or is unsafe: {target}. Choose a new tune name.')
    if source_mode not in {None, 'inherit', 'copy', 'reference'}:
        raise ValueError('Destination source mode must be inherit, copy, or reference.')
    config_path = source / 'cell_configs/cell_config.json'
    if not config_path.is_file():
        config_path = source / 'cell_config.json'
    if not config_path.is_file():
        raise FileNotFoundError('Save the source tune setup before creating a working copy.')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError(f'Expected JSON object in {config_path}')
    options = config.get('model_source')
    if options is not None and not isinstance(options, dict):
        raise ValueError('model_source must be a configuration object.')
    if not options and source_mode in {'copy', 'reference'}:
        raise ValueError('Copy/reference selection applies only to registered external sources.')
    mode = (options or {}).get('mode', 'tune-local') if source_mode in {None, 'inherit'} else source_mode
    if mode not in {'copy', 'reference', 'tune-local'} or options and mode == 'tune-local':
        raise ValueError('model_source.mode must be copy or reference.')
    warnings, records, directories = [], [], []
    for current, dirs, files in os.walk(source, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not _excluded((Path(current) / d).relative_to(source)))
        for directory in dirs:
            if (Path(current) / directory).is_symlink():
                raise ValueError(f'Directory symlinks cannot be duplicated safely: {Path(current) / directory}')
            directories.append((Path(current) / directory).relative_to(source).as_posix())
        for filename in sorted(files):
            path = Path(current) / filename
            relative = path.relative_to(source)
            if _excluded(relative):
                continue
            if not within(path.resolve(), source):
                raise ValueError(f'Working-copy input symlink escapes its source tune: {path}')
            records.append({'path': relative.as_posix(), 'size_bytes': path.stat().st_size,
                            'sha256': sha256_file(path)})
    # A reference tune still owns a saved snapshot. Copy that, not today's upstream.
    if options:
        from scp_pipeline.loaders.source_context import declared_source_files, original_source_root, guard_source_destinations
        guard_source_destinations(config, source, [target])
        snapshot_config = deepcopy(config)
        snapshot_config['model_source']['mode'] = 'copy'
        declared_source_files(snapshot_config, source)
        from scp_pipeline.loaders import validate_cell_loader_config
        validate_cell_loader_config(snapshot_config, base_dir=source)
        if mode == 'reference':
            warnings.append('Both reference tunes use the same external sources; edits affect both. Restart before switching tunes.'
                            if options.get('mode', 'copy') == 'reference' else
                            'The destination references external sources; the source tune retains its independent copy. Restart before switching tunes.')
            upstream = original_source_root(config, source)
            try:
                check = deepcopy(config)
                check['model_source']['mode'] = 'reference'
                declared_source_files(check, source)
            except (ValueError, OSError) as exc:
                warnings.append(f'External dependency unavailable: {upstream}: {exc}. No sources will be fetched.')
    elif config.get('cell_loader') != 'runtime_cell':
        from scp_pipeline.loaders import validate_cell_loader_config
        validate_cell_loader_config(config, base_dir=source)
    paths = {record['path'] for record in records}
    from scp_pipeline.projects import project_for_tune
    owner = config.get('workspace_project_root')
    data_root = ((source / owner).resolve() if owner else project_for_tune(source)) or next(
        (parent for parent in source.parents if (parent / 'cells').is_dir()
         and (parent / 'run_pipeline.py').is_file()), None)
    # Inspect active configuration only, not historical provenance/native assets.
    for relative in sorted(paths):
        path = source / relative
        if path.suffix != '.json' or not (relative.startswith('cell_configs/') or path == config_path):
            continue
        payload = json.loads(path.read_text(encoding='utf-8'))
        if path == config_path and options:
            payload = {key: value for key, value in payload.items() if key != 'paths'}
        base = source / 'cell_configs' if path.name == 'syn_config.json' else source
        for label, dependency in _config_inputs(payload, base, data_root=data_root):
            if within(dependency, source):
                local = dependency.relative_to(source)
                if _excluded(local):
                    raise ValueError(f'{relative}: required input {label} points into excluded generated data: {local}. '
                                     'Move the required data into a tune-local data folder and update the source config before copying.')
                if not dependency.exists():
                    raise FileNotFoundError(f'{relative}: required tune-local input {label} is missing: {dependency}')
    report = {'status': 'preview', 'source': str(source), 'target': str(target),
              'target_tune_name': name, 'content_policy': 'model', 'source_mode': mode,
              'file_count': len(records), 'size_bytes': sum(item['size_bytes'] for item in records),
              'files': records, 'warnings': warnings, 'snapshot_refreshed': False, 'constructs_cell': False}
    report['directories'] = sorted(directories)
    if dry_run:
        return report
    # Recheck content before writing; stage beside destination then rename atomically.
    with tempfile.TemporaryDirectory(prefix='.scp-copy-', dir=source.parent) as temporary:
        staging = Path(temporary) / name
        staging.mkdir()
        for directory in directories:
            (staging / directory).mkdir(parents=True, exist_ok=True)
        for record in records:
            origin = source / record['path']
            if sha256_file(origin) != record['sha256']:
                raise ValueError(f'Source changed during copy: {origin}. Preview again.')
            destination = staging / record['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, destination)
            if sha256_file(destination) != record['sha256']:
                raise ValueError(f'Source changed during copy: {origin}. Preview again.')
        updated = deepcopy(config)
        updated.update(cell_name=workspace.cell_name, tune=name)
        if options:
            updated['model_source']['mode'] = mode
        new_workspace = infer_workspace(target, cell_config=updated, project_name=workspace.project_name,
                                        cell_name=workspace.cell_name)
        if 'workspace_identity' in updated:
            updated['workspace_identity'] = new_workspace.identity
        if 'tune_dir' in updated:
            updated['tune_dir'] = str(target)
        _remap_paths(updated, source, target)
        for relative in sorted(paths):
            if not relative.startswith('cell_configs/') or not relative.endswith('.json'):
                continue
            destination = staging / relative
            payload = json.loads(destination.read_text())
            before = deepcopy(payload)
            if source / relative == config_path:
                payload = updated
            else:
                _remap_paths(payload, source, target)
                if destination.name == 'sim_config.json' and any(key in payload for key in
                        ('workspace_identity', 'project_name', 'tune_dir', 'cell', 'tune')):
                    apply_workspace_metadata(payload, new_workspace)
            if payload != before:
                destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        if config_path.parent == source:
            (staging / config_path.name).write_text(json.dumps(updated, indent=2) + '\n', encoding='utf-8')
        if target.exists():
            raise ValueError(f'Working-copy destination now exists: {target}')
        staging.rename(target)
    report.update(status='created', cell_config_updated=True)
    return report
