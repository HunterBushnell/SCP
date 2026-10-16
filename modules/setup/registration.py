"""Non-executing model registration in SCP or project-owned workspaces."""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path

from scp_pipeline.projects import load_project_settings, resolve_project_path, save_project_settings
from .model_sources import preview_model_sources, register_model_sources


def preview_workspace_registration(*, project_root=None, repo_root=None, tune_dir, cell_name, tune_name,
                                   cell_config, conditions, project_name=None, output_dir=None,
                                   target_options=None, synapse_options=None, workspace_root=None):
    """Validate source/recipe inputs without importing model code or writing.

    Select SCP routing with ``repo_root`` (and optional ``project_name``), or
    project-owned routing with ``project_root``. Explicit tune paths take precedence.
    """
    from scp_pipeline.workspaces import _name
    from scp_pipeline.loaders import validate_cell_loader_config
    from scp_pipeline.workspaces import infer_workspace, resolve_workspace
    config = deepcopy(cell_config)
    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
        tune = resolve_workspace(project_root=root, tune_dir_override=tune_dir, cell_config={}).tune_dir
        config.update(cell_name=_name(cell_name, 'cell_name'), tune=_name(tune_name, 'tune_name'),
                      project_name=_name(project_name or load_project_settings(root)['project_name'], 'project_name'),
                      workspace_layout='project_owned', workspace_project_root=os.path.relpath(root, tune))
        workspace = infer_workspace(tune, cell_config=config)
        output_root = root
    else:
        if repo_root is None:
            raise ValueError('Registration requires repo_root for SCP workspaces or project_root for a project-owned workspace.')
        output_root = Path(repo_root).expanduser().resolve()
        workspace = resolve_workspace(repo_root=output_root, tune_dir_override=tune_dir,
                                      cell_name=cell_name, tune_name=tune_name,
                                      project_name=project_name, cell_config={})
        tune = workspace.tune_dir
        root = output_root / 'projects' / workspace.project_name if workspace.project_name else output_root
        config.pop('workspace_layout', None)
        config.pop('workspace_project_root', None)
        if workspace.project_name is None:
            config.pop('project_name', None)
        else:
            config['project_name'] = workspace.project_name
    # Explicit tune paths determine identity, even when shorthand fields differ.
    config.update(cell_name=workspace.cell_name, tune=workspace.tune_name)
    # Reject unusable navigation settings before staging any sources/configs.
    load_project_settings(root)
    if project_root is None and root != output_root:
        load_project_settings(output_root)
    resolved_output = str(resolve_project_path(output_root, output_dir)) if output_dir else None
    save_project_settings(root, dry_run=True,
                          project_name=config.get('project_name') or load_project_settings(root)['project_name'],
                          selected_tune=tune, workspace_root=workspace_root or '.', output_dir=resolved_output)
    loader = config.get('cell_loader')
    if loader not in {'allen_manifest', 'hoc_template', 'python_factory'}:
        raise ValueError(f'Unknown registration loader {loader!r}.')
    config.setdefault('color', 'k')
    if loader == 'allen_manifest':
        config.setdefault('tuning', {'soma_diam_multiplier': 1.0})
    if loader != 'allen_manifest' or conditions:
        for key in ('v_init_mV', 'celsius_C'):
            value = conditions.get(key) if isinstance(conditions, dict) else None
            if isinstance(value, bool) or value is None or not math.isfinite(float(value)):
                raise ValueError(f'Explicit finite conditions.{key} is required before registration.')
    if (tune / 'cell_configs/cell_config.json').exists() or (tune / 'cell_config.json').exists():
        raise ValueError('A model recipe already exists here. Open that workspace; registration never overwrites it.')
    inventory = None
    if config.get('model_source'):
        from scp_pipeline.loaders.source_context import guard_source_destinations
        guard_source_destinations(config, tune, [root / '.scp_project.json', output_root / '.scp_project.json',
                                               *([resolved_output] if resolved_output else [])])
        inventory = preview_model_sources(tune, config)
        check = deepcopy(config)
        check['model_source']['mode'] = 'reference'
        validate_cell_loader_config(check, base_dir=tune)
        if loader == 'allen_manifest':
            from scp_pipeline.loaders import discover_cell_source_artifacts
            discover_cell_source_artifacts(check, base_dir=tune)
    else:
        validate_cell_loader_config(config, base_dir=tune)
    mod_path = (config.get('paths') or {}).get('modfiles')
    from .optional_configs import preview_setup_outputs
    optional = preview_setup_outputs(tune, target_options=target_options,
                                     synapse_options=synapse_options, cell_config=config,
                                     include_required=True, conditions=conditions)
    return {'status': 'preview', 'project_root': str(root), 'tune_dir': str(tune),
            'workspace_identity': workspace.identity,
            'cell_config': config, 'conditions': deepcopy(conditions or {}),
            'output_dir': resolved_output or str(tune / 'output_data'),
            'sources': inventory, 'optional_outputs': optional,
            'mechanisms': 'compile/load on Prepare' if mod_path else 'built-in only',
            'creates': ['cell_configs/cell_config.json'] + (['model_sources/', 'MODEL_SOURCE_PROVENANCE.json'] if inventory else [])
                       + [str(Path(item['path']).relative_to(tune)) for item in optional['files'] if item['status'] == 'create']}


def register_workspace(*, dry_run=True, workspace_root=None, **kwargs):
    """Stage approved sources/configs only. Construction is a separate action."""
    preview = preview_workspace_registration(workspace_root=workspace_root, **kwargs)
    if dry_run:
        return preview
    root, tune = Path(preview['project_root']), Path(preview['tune_dir'])
    config = preview['cell_config']
    if config.get('model_source'):
        config = register_model_sources(tune, config, dry_run=False)['config']
    else:
        path = tune / 'cell_configs/cell_config.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    from .optional_configs import create_setup_outputs
    preview['optional_outputs'] = create_setup_outputs(
        tune, target_options=kwargs.get('target_options'),
        synapse_options=kwargs.get('synapse_options'), cell_config=config,
        include_required=True, conditions=preview['conditions'])
    navigation = {}
    if kwargs.get('project_root') is None and config.get('project_name') is None:
        navigation['selected_project'] = None
    save_project_settings(root, project_name=config.get('project_name') or load_project_settings(root)['project_name'],
                          selected_tune=tune, workspace_root=workspace_root or '.',
                          output_dir=preview['output_dir'] if kwargs.get('output_dir') else None, **navigation)
    if kwargs.get('project_root') is None and config.get('project_name') is not None:
        save_project_settings(kwargs['repo_root'], selected_project=config['project_name'])
    preview['status'] = 'registered'
    return preview
