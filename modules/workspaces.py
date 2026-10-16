"""Pure path discovery and portable identities for SCP tune workspaces.

This module does not import NEURON, execute model code, or write configuration.
Workspace identity is a routing namespace, not biological model equivalence.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


class WorkspaceMismatchError(ValueError):
    """Trial pooling would combine different logical workspaces."""


def _root(repo_root=None) -> Path:
    return Path(repo_root or Path(__file__).resolve().parents[1]).expanduser().resolve()


def _name(value, label, *, optional=False):
    if value is None and optional:
        return None
    if (not isinstance(value, str) or not value.strip() or value in {'.', '..'}
            or any(char in value for char in ('/', '\\', '\x00'))
            or Path(value).is_absolute()):
        raise ValueError(f'{label} must be a nonempty, safe single directory name; got {value!r}.')
    return value


def _within(path, root):
    return path == root or root in path.parents


def _scope(repo_root, project_name=None):
    root = _root(repo_root)
    project = _name(project_name, 'project_name', optional=True)
    namespace = root / 'cells' if project is None else root / 'projects' / project / 'cells'
    if not _within(namespace.resolve(), root):
        raise ValueError(f'Workspace namespace escapes the SCP root: {namespace}')
    if namespace.resolve() != namespace:
        raise ValueError(f'Workspace namespace must not alias another directory through a symlink: {namespace}')
    if project is not None and not _within(namespace.resolve(), (root / 'projects' / project).resolve()):
        raise ValueError(f'Cell namespace escapes its project: {namespace}')
    return namespace


def _structure(path, *, tunes_parent=None):
    """Match adjacent trailing components, never the first ancestor named cells."""
    parts = Path(path).parts
    parent_matches = tunes_parent is None or (len(parts) >= 2 and parts[-2] == tunes_parent)
    if len(parts) >= 6 and parts[-6] == 'projects' and parts[-4] == 'cells' and parent_matches:
        return 'project', parts[-5], parts[-3], parts[-1]
    if len(parts) >= 4 and parts[-4] == 'cells' and parent_matches:
        return 'legacy', None, parts[-3], parts[-1]
    return None


def _guard_managed_path(candidate, repo_root=None):
    """Check the lexical namespace before resolving away symlink components."""
    lexical = Path(candidate).expanduser().absolute()
    root = _root(repo_root)
    structure = _structure(lexical)
    # Recognizable relocated workspaces carry their own namespace root. This
    # also keeps direct preparation/session APIs safe outside this checkout.
    if structure is not None:
        root = lexical.parents[5 if structure[0] == 'project' else 3]
    try:
        parts = lexical.relative_to(root).parts
    except ValueError:
        return
    project = parts[1] if len(parts) >= 3 and parts[0] == 'projects' and parts[2] == 'cells' else None
    if parts and (parts[0] == 'cells' or project is not None):
        scope = _scope(root, project)
        if not _within(lexical.resolve(), scope.resolve()):
            raise ValueError(f'Explicit managed workspace escapes its namespace: {candidate}')


def _read_config(tune):
    for path in (tune / 'cell_configs/cell_config.json', tune / 'cell_config.json'):
        if path.is_file():
            value = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(value, dict):
                raise TypeError(f'Cell configuration must be an object: {path}')
            return value
    return {}


@dataclass(frozen=True)
class Workspace:
    tune_dir: Path
    project_name: str | None
    cell_name: str
    tune_name: str
    layout: str

    @property
    def identity(self):
        return {'project_name': self.project_name, 'cell_name': self.cell_name, 'tune_name': self.tune_name}

    @property
    def label(self):
        return '/'.join(str(part) for part in (self.project_name, self.cell_name, self.tune_name) if part is not None)


def infer_workspace(tune_dir, *, repo_root=None, cell_config=None, project_name=None,
                    cell_name=None, tune_name=None, tunes_parent=None) -> Workspace:
    _guard_managed_path(tune_dir, repo_root)
    tune = Path(tune_dir).expanduser().resolve()
    config = _read_config(tune) if cell_config is None else cell_config
    if not isinstance(config, Mapping):
        raise TypeError('cell_config must be a mapping.')
    declared = _name(config.get('project_name'), 'cell_config.project_name', optional=True)
    structure = _structure(tune, tunes_parent=tunes_parent)
    if structure is not None:
        layout, project, cell, name = structure
        if layout == 'legacy' and config.get('workspace_layout') == 'project_owned':
            layout, project = 'project_owned', declared
        if declared is not None and declared != project:
            raise ValueError(f'Configured project {declared!r} conflicts with the selected workspace {tune} '
                             f'(project={project!r}). Restore/rebase identity explicitly; no files were changed.')
    else:
        layout = 'project_owned' if config.get('workspace_layout') == 'project_owned' else 'external'
        project = declared if declared is not None else _name(project_name, 'project_name', optional=True)
        parent_name = tunes_parent or 'tunes'
        cell = config.get('cell_name') or cell_name or (tune.parent.parent.name if tune.parent.name == parent_name else tune.parent.name) or 'cell'
        name = config.get('tune') or tune.name or tune_name
    return Workspace(tune, _name(project, 'project_name', optional=True),
                     _name(str(cell), 'cell_name'), _name(str(name), 'tune_name'), layout)


def resolve_workspace(*, repo_root=None, project_name=None, cell_name=None, tune_name=None,
                      tune_dir_override=None, relative_to=None, tunes_parent='tunes', cell_config=None,
                      project_root=None, workspace_root=None) -> Workspace:
    root = _root(repo_root)
    if project_root is not None:
        from .projects import load_project_settings, resolve_project_path
        owner = Path(project_root).expanduser().resolve()
        settings = load_project_settings(owner)
        base = resolve_project_path(owner, workspace_root or settings.get('workspace_root') or '.')
        if tune_dir_override in (None, '') and not (cell_name and tune_name):
            tune_dir_override = settings.get('selected_tune')
        if tune_dir_override not in (None, ''):
            candidate = Path(tune_dir_override).expanduser()
            if not candidate.is_absolute():
                candidate = Path(relative_to or owner) / candidate
        else:
            candidate = base / 'cells' / _name(cell_name, 'cell_name') / _name(tunes_parent, 'tunes_parent') / _name(tune_name, 'tune_name')
        _guard_managed_path(candidate, root)
        return infer_workspace(candidate, repo_root=root, cell_config=cell_config,
                               cell_name=cell_name, tune_name=tune_name)
    if tune_dir_override not in (None, ''):
        candidate = Path(tune_dir_override).expanduser()
        if not candidate.is_absolute():
            candidate = Path(relative_to or root) / candidate
        _guard_managed_path(candidate, root)
        return infer_workspace(candidate, repo_root=root, cell_config=cell_config,
                               project_name=project_name, cell_name=cell_name, tune_name=tune_name,
                               tunes_parent=None)
    cell = _name(cell_name, 'cell_name')
    tune = _name(tune_name, 'tune_name')
    parent = _name(tunes_parent, 'tunes_parent')
    scope = _scope(root, project_name)
    candidate = scope / cell / parent / tune
    if not _within(candidate.resolve(), scope.resolve()):
        raise ValueError(f'Workspace escapes its cell namespace: {candidate}')
    return infer_workspace(candidate, repo_root=root, cell_config=cell_config,
                           project_name=project_name, cell_name=cell, tune_name=tune, tunes_parent=parent)


def _directories(base, namespace):
    if not base.is_dir():
        return []
    names = []
    for path in sorted(base.iterdir(), key=lambda item: item.name):
        if path.name.startswith('.') or not path.is_dir():
            continue
        _name(path.name, 'workspace directory')
        if not _within(path.resolve(), namespace.resolve()):
            raise ValueError(f'Discovered workspace directory escapes its namespace: {path}')
        names.append(path.name)
    return names


def discover_projects(repo_root=None):
    root = _root(repo_root)
    base = root / 'projects'
    if not _within(base.resolve(), root):
        raise ValueError('Project discovery root escapes SCP.')
    projects = _directories(base, base)
    for project in projects:
        _scope(root, project)
    return projects


def discover_cells(repo_root=None, project_name=None):
    scope = _scope(repo_root, project_name)
    return _directories(scope, scope)


def discover_tunes(repo_root, cell_name, project_name=None, tunes_parent='tunes'):
    scope = _scope(repo_root, project_name)
    base = scope / _name(cell_name, 'cell_name') / _name(tunes_parent, 'tunes_parent')
    if not _within(base.resolve(), scope.resolve()):
        raise ValueError('Tune discovery escapes its cell namespace.')
    return _directories(base, scope)


def apply_workspace_metadata(sim_config, workspace):
    """Reassert effective routing after config inheritance without editing files."""
    sim_config.update(tune_dir=str(workspace.tune_dir), cell=workspace.cell_name, tune=workspace.tune_name,
                      project_name=workspace.project_name, workspace_identity=workspace.identity)
    if workspace.layout == 'project_owned':
        sim_config['workspace_layout'] = 'project_owned'
        sim_config.pop('project_root', None)
        from .projects import project_for_tune, resolve_project_path
        config = _read_config(workspace.tune_dir)
        owner = config.get('workspace_project_root')
        root = resolve_project_path(workspace.tune_dir, owner) if owner else project_for_tune(workspace.tune_dir)
        if root:
            sim_config['project_root'] = str(root)
    else:
        sim_config.pop('workspace_layout', None)
        sim_config.pop('project_root', None)
        from .projects import project_for_tune
        owner = project_for_tune(workspace.tune_dir)
        if owner:
            sim_config['project_root'] = str(owner)
    return sim_config


def result_workspace_identity(results):
    """Read portable identity from new or historical results, without rebuilding."""
    sim = results.get('sim_cfg', results) or {}
    meta = results.get('meta', {}) or {}
    explicit = [value for value in (results.get('workspace_identity'), sim.get('workspace_identity'),
                                    meta.get('workspace_identity')) if value is not None]
    identities = []
    for value in explicit:
        if not isinstance(value, Mapping):
            raise ValueError('workspace_identity must be an object.')
        identities.append({key: _name(value.get(key), key, optional=True)
                           for key in ('project_name', 'cell_name', 'tune_name')})
    if identities and any(value != identities[0] for value in identities[1:]):
        raise WorkspaceMismatchError('Conflicting saved workspace_identity records.')
    structure = _structure(Path(str(sim['tune_dir']))) if sim.get('tune_dir') else None
    declared = sim.get('project_name', (meta.get('cell_config') or {}).get('project_name'))
    if structure and structure[0] == 'legacy' and sim.get('workspace_layout') == 'project_owned':
        structure = ('project_owned', declared, structure[2], structure[3])
    if identities:
        identity = identities[0]
        if declared is not None and declared != identity['project_name']:
            raise WorkspaceMismatchError('Saved project metadata conflicts with workspace_identity.')
        if structure and identity != {'project_name': structure[1], 'cell_name': structure[2], 'tune_name': structure[3]}:
            raise WorkspaceMismatchError('Saved tune path conflicts with workspace_identity.')
        return identity
    if structure:
        if declared is not None and declared != structure[1]:
            raise WorkspaceMismatchError('Saved tune path conflicts with declared project metadata.')
        return {'project_name': structure[1], 'cell_name': structure[2], 'tune_name': structure[3]}
    return {'project_name': _name(declared, 'project_name', optional=True),
            'cell_name': _name(sim.get('cell'), 'cell_name', optional=True),
            'tune_name': _name(sim.get('tune'), 'tune_name', optional=True)}


def assert_same_workspace(left, right, *, context='trial pooling'):
    a, b = result_workspace_identity(left), result_workspace_identity(right)
    if a['project_name'] != b['project_name']:
        raise WorkspaceMismatchError(f'{context}: refusing different workspaces: {a} versus {b}. Use comparison instead of pooling trials.')
    if a['project_name'] is not None and any(identity[key] is None for identity in (a, b) for key in ('cell_name', 'tune_name')):
        raise WorkspaceMismatchError(f'{context}: named-project results require complete workspace identity.')
    for key in ('cell_name', 'tune_name'):
        if a[key] is not None and b[key] is not None and a[key] != b[key]:
            raise WorkspaceMismatchError(f'{context}: refusing different workspaces: {a} versus {b}.')
    if any(identity[key] is None for identity in (a, b) for key in ('cell_name', 'tune_name')):
        return ['Incomplete unscoped legacy workspace identity; historical pooling retained, but compatibility could not be established.']
    return []


def saved_namespace_tune_dir(sim_config, repo_root=None):
    if sim_config.get('tune_dir'):
        return Path(str(sim_config['tune_dir'])).expanduser().resolve()
    identity = result_workspace_identity({'sim_cfg': sim_config})
    if identity['cell_name'] is None or identity['tune_name'] is None:
        return None
    return resolve_workspace(repo_root=repo_root, project_name=identity['project_name'],
                             cell_name=identity['cell_name'], tune_name=identity['tune_name'], cell_config={}).tune_dir
