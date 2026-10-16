"""Project-owned navigation settings, separate from scientific tune configs."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

PROJECT_FILE = '.scp_project.json'
_FIELDS = {'format_version', 'project_name', 'workspace_root', 'selected_tune',
           'output_dir', 'analysis_settings', 'selected_project'}


def resolve_project_path(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(root) / path).resolve()


def portable_path(root, value):
    if value in (None, ''):
        return None
    path, root = resolve_project_path(root, value), Path(root).resolve()
    return os.path.relpath(path, root) if path == root or root in path.parents else str(path)


def load_project_settings(project_root):
    root = Path(project_root).expanduser().resolve()
    path = root / PROJECT_FILE
    if not path.exists():
        return {'format_version': 1, 'project_name': root.name,
                'workspace_root': '.', 'selected_tune': None,
                'output_dir': None, 'analysis_settings': None}
    if path.is_symlink():
        raise ValueError(f'Refusing symlinked project settings: {path}')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        raise ValueError(f'Cannot read project settings {path}: {exc}') from exc
    if not isinstance(value, dict) or value.get('format_version') != 1:
        raise ValueError(f'{path} must be a format_version=1 settings object.')
    if set(value) - _FIELDS:
        raise ValueError(f'Unknown project settings fields in {path}: {sorted(set(value) - _FIELDS)}')
    from .workspaces import _name
    _name(value.get('project_name'), 'project_name')
    _name(value.get('selected_project'), 'selected_project', optional=True)
    for key in ('workspace_root', 'selected_tune', 'output_dir', 'analysis_settings'):
        item = value.get(key)
        if item is not None and (not isinstance(item, str) or not item.strip()):
            raise ValueError(f'{path}: {key} must be a nonempty path or null.')
    return value


def save_project_settings(project_root, *, dry_run=False, **updates):
    """Validate navigation before writes; ``dry_run`` never creates directories."""
    root = Path(project_root).expanduser().resolve()
    value = deepcopy(load_project_settings(root))
    if set(updates) - (_FIELDS - {'format_version'}):
        raise ValueError('Unsupported project setting; runtime objects/scientific settings do not belong here.')
    from .workspaces import _name
    value.update(updates)
    _name(value['project_name'], 'project_name')
    _name(value.get('selected_project'), 'selected_project', optional=True)
    for key in ('workspace_root', 'selected_tune', 'output_dir', 'analysis_settings'):
        value[key] = portable_path(root, value.get(key))
    value['workspace_root'] = value['workspace_root'] or '.'
    for parent in (root, *root.parents):
        if parent.exists() and not parent.is_dir():
            raise ValueError(f'Project settings directory is not a directory: {parent}')
    path = root / PROJECT_FILE
    if path.is_symlink():
        raise ValueError(f'Refusing symlinked project settings: {path}')
    if dry_run:
        return value
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=root, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return value


def project_for_tune(tune_dir):
    """Find only an explicitly registered ancestor project, never sibling repos."""
    tune = Path(tune_dir).resolve()
    for parent in tune.parents:
        if (parent / PROJECT_FILE).is_file():
            return parent
    return None


def load_analysis_settings(project_root, path=None):
    """Return editable project overrides on a copy of packaged defaults."""
    from importlib.resources import files
    value = json.loads(files('scp_pipeline.analysis').joinpath('analysis_defaults.json').read_text())
    # Bundled scientific comparison choices do not belong to a new project.
    value.update(compare_list_paths=[], compare_list_dir_paths=[], compare_preset_enabled=False)
    for key, raw in list(value.items()):
        if key.endswith('_preset_path') and isinstance(raw, str) and raw.startswith('modules/analysis/'):
            value[key] = str(files('scp_pipeline.analysis').joinpath(raw.removeprefix('modules/analysis/')))
    settings = load_project_settings(project_root)
    selected = path or settings.get('analysis_settings')
    if selected:
        overrides = json.loads(resolve_project_path(project_root, selected).read_text())
        if not isinstance(overrides, dict):
            raise TypeError('Analysis settings must be a JSON object.')
        value.update(overrides)
        for key, raw in overrides.items():
            if key.endswith('_preset_path') and raw:
                value[key] = str(resolve_project_path(project_root, raw))
    return value
