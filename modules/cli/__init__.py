"""Installed command dispatch and project-relative routing."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {'-h', '--help'}:
        print('usage: scp-pipeline {setup,run,check} [options] [--project-root PATH] [--workspace-root PATH]')
        return 0 if arguments else 2
    command = arguments.pop(0)
    if command not in {'setup', 'run', 'check'}:
        raise SystemExit(f'Unknown SCP command: {command}')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--project-root')
    parser.add_argument('--workspace-root')
    owner, arguments = parser.parse_known_args(arguments)
    tune = None
    if owner.project_root and not any(flag in arguments for flag in ('--help', '-h')):
        from scp_pipeline.projects import load_project_settings, resolve_project_path
        root = Path(owner.project_root).expanduser().resolve()
        settings = load_project_settings(root)
        def option(name):
            if name in arguments:
                return arguments[arguments.index(name) + 1]
            return next((arg.split('=', 1)[1] for arg in arguments if arg.startswith(name + '=')), None)
        explicit = option('--tune-dir')
        if explicit:
            tune = Path(explicit).expanduser().resolve()
        elif option('--cell'):
            from scp_pipeline.workspaces import _name
            base = resolve_project_path(root, owner.workspace_root or settings.get('workspace_root') or '.')
            tune = base / 'cells' / _name(option('--cell'), 'cell_name') / 'tunes' / _name(option('--tune') or ('orig' if command == 'setup' else 'tuned'), 'tune_name')
        elif settings.get('selected_tune'):
            from scp_pipeline.workspaces import resolve_workspace
            tune = resolve_workspace(project_root=root, tune_dir_override=settings['selected_tune']).tune_dir
        elif '--help' not in arguments and '-h' not in arguments:
            raise ValueError('Select --tune-dir, --cell/--tune, or save a project selection in the notebook.')
        if tune is not None and not explicit:
            arguments += ['--tune-dir', str(tune)]
        if command == 'run' and option('--output-dir') is None and settings.get('output_dir'):
            arguments += ['--output-dir', str(resolve_project_path(root, settings['output_dir']))]
        if command == 'setup' and tune is not None and not any(flag in arguments for flag in ('--help', '-h')):
            from scp_pipeline.setup.mechanisms import load_tune_cell_config
            config = load_tune_cell_config(tune)
            if config and config.get('project_name') not in (None, settings['project_name']):
                raise ValueError('The selected tune belongs to another project; no setup operations were performed.')
    module = importlib.import_module('scp_pipeline.cli.' + command)
    previous = sys.argv
    try:
        sys.argv = ['scp-pipeline ' + command, *arguments]
        result = module.main()
    finally:
        sys.argv = previous
    if (not result and command == 'setup' and owner.project_root and tune is not None
            and not any(flag in arguments for flag in ('--help', '-h', '--preview-model-sources', '--list-models-only'))):
        from scp_pipeline.projects import save_project_settings
        import os
        path = tune / 'cell_configs/cell_config.json'
        if path.is_file():
            config = json.loads(path.read_text())
            if config.get('project_name') not in (None, settings['project_name']):
                raise ValueError('The selected tune belongs to another project; its identity was not rewritten.')
            config.update(project_name=settings['project_name'], workspace_layout='project_owned',
                          workspace_project_root=os.path.relpath(root, tune))
            path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
            save_project_settings(root, selected_tune=tune, workspace_root=owner.workspace_root or settings.get('workspace_root') or '.')
    return result or 0
