"""Preview/register native source snapshots with explicit backed-up refresh."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

from scp_pipeline.loaders.source_context import (declared_source_files, declared_source_directories,
    sha256_file, source_options, source_relative_path, within, original_source_root,
    guard_source_destinations)


def preview_model_sources(tune_dir, cell_config):
    tune = Path(tune_dir).expanduser().resolve()
    config = deepcopy(cell_config)
    options = source_options(config)
    if options is None:
        raise ValueError('External registration requires model_source with source_root and folders.')
    requested_mode = options.get('mode', 'copy')
    config['model_source']['mode'] = 'reference'
    root = original_source_root(config, tune)
    guard_source_destinations(config, tune)
    for folder in options.get('folders', []):
        if within(tune / 'model_sources', (root / folder).resolve()):
            raise ValueError('The snapshot destination must not be inside a selected source folder.')
    records = [{'source_path': p.relative_to(root).as_posix(),
                'destination_path': (Path('model_sources') / p.relative_to(root)).as_posix(),
                'sha256': sha256_file(p), 'size_bytes': p.stat().st_size}
               for p in declared_source_files(config, tune)]
    if not records:
        raise ValueError('The selected model source folders contain no native inputs.')
    return {'source_root': str(root), 'mode': requested_mode,
            'directories': declared_source_directories(config, tune),
            'file_count': len(records), 'size_bytes': sum(p['size_bytes'] for p in records), 'files': records}


def register_model_sources(tune_dir, cell_config, *, dry_run=True, refresh=False):
    """Stage declared folders without executing model code or changing upstream."""
    tune = Path(tune_dir).expanduser().resolve()
    preview = preview_model_sources(tune, cell_config)
    config = deepcopy(cell_config)
    config['model_source'].setdefault('mode', 'copy')
    config['model_source']['source_root'] = preview['source_root']
    config['paths'] = dict(config.get('paths', {}))
    for role in ('hoc_template', 'python_factory', 'manifest', 'modfiles'):
        value = config['paths'].get(role)
        if value not in (None, ''):
            config['paths'][role] = source_relative_path(config, value, tune).as_posix()
    config['model_source']['source_root'] = __import__('os').path.relpath(preview['source_root'], tune)
    snapshot = tune / 'model_sources'
    if snapshot.is_symlink() or not within(snapshot.resolve(), tune):
        raise ValueError('Refusing a snapshot destination outside the selected tune.')
    manifest = tune / 'MODEL_SOURCE_PROVENANCE.json'
    if not within(manifest.resolve(), tune):
        raise ValueError('Source provenance destination escapes the selected tune.')
    old = json.loads(manifest.read_text()) if manifest.is_file() else None
    baseline = {p['destination_path']: p['sha256'] for p in old.get('files', [])} if old else {}
    incoming = {p['destination_path']: p['sha256'] for p in preview['files']}
    unchanged = bool(old and baseline == incoming and
                     all((snapshot / directory).is_dir() for directory in preview['directories']) and
                     all((tune / p).is_file() and sha256_file(tune / p) == digest for p, digest in baseline.items()))
    if snapshot.exists() and not unchanged and not refresh:
        raise ValueError('Source snapshot differs from upstream or has local edits. It was not overwritten. '
                         'Use explicit refresh=True after reviewing the preview; refresh creates backups.')
    target_config = tune / 'cell_configs' / 'cell_config.json'
    if not within(target_config.resolve(), tune):
        raise ValueError('Recipe destination escapes the selected tune.')
    if target_config.is_file() and not refresh and json.loads(target_config.read_text()) != config:
        raise ValueError('Existing cell_config.json differs; use explicit refresh=True or set_model_source_mode. '
                         'No recipe or source files were overwritten.')
    result = {'status': 'dry_run' if dry_run else ('unchanged' if unchanged else 'registered'),
              'config': config, 'preview': preview}
    if dry_run:
        return result
    tune.mkdir(parents=True, exist_ok=True)
    if refresh and (snapshot.exists() or target_config.exists()):
        backup = tune / 'restore_backups' / ('sources_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
        if not within(backup.resolve(), tune):
            raise ValueError('Source backup destination escapes the selected tune.')
        backup.mkdir(parents=True)
        if snapshot.exists():
            shutil.copytree(snapshot, backup / 'model_sources')
        for path in (manifest, target_config):
            if path.exists():
                shutil.copy2(path, backup / path.name)
        result['backup_dir'] = str(backup)
    if not unchanged:
        root = Path(preview['source_root'])
        with tempfile.TemporaryDirectory(prefix='.source_stage_', dir=tune) as tmp:
            for folder in preview['directories']:
                (Path(tmp) / folder).mkdir(parents=True, exist_ok=True)
            for record in preview['files']:
                source = root / record['source_path']
                target = Path(tmp) / record['source_path']
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target, follow_symlinks=True)
                if sha256_file(target) != record['sha256']:
                    raise RuntimeError(f'Source changed after preview: {source}. Preview and register again.')
            if snapshot.exists():
                shutil.rmtree(snapshot)
            shutil.copytree(tmp, snapshot)
    if not unchanged:
        manifest.write_text(json.dumps({'format_version': 1, **preview}, indent=2) + '\n', encoding='utf-8')
    target_config.parent.mkdir(parents=True, exist_ok=True)
    target_config.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    return result


def set_model_source_mode(tune_dir, mode):
    if mode not in {'copy', 'reference'}:
        raise ValueError('Source mode must be copy or reference.')
    path = Path(tune_dir).expanduser().resolve() / 'cell_configs' / 'cell_config.json'
    if not within(path.resolve(), Path(tune_dir).expanduser().resolve()):
        raise ValueError('Recipe destination escapes the selected tune.')
    config = json.loads(path.read_text())
    source_options(config)
    config['model_source']['mode'] = mode
    from scp_pipeline.loaders import validate_cell_loader_config
    validate_cell_loader_config(config, base_dir=Path(tune_dir))
    path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    return {'mode': mode, 'restart_required': True}
