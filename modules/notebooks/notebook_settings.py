"""JSON-only notebook preferences, independent of model construction/configs."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile

from ._pipeline_ui.common import PIPELINE_UI_DEFAULTS

SETTINGS_FILE = '.scp_pipeline_settings.json'
NAVIGATION_KEYS = frozenset({'project_name', 'cell_name', 'tune_name', 'tune_dir_override'})
SETTING_KEYS = frozenset(PIPELINE_UI_DEFAULTS) - NAVIGATION_KEYS
_INTEGER_KEYS = {'n_trials', 'seed', 'act_n_cpus', 'input_preview_trial_idx',
                 'input_preview_plot_columns', 'diagnostic_trial_idx'}
_TEXT_KEYS = {'act_active_module', 'act_workspace_override', 'synapse_connection', 'output_stem'}
_LIST_KEYS = {'input_preview_groups', 'diagnostic_input_groups'}


def settings_root(repo_root, project_root=None, project_name=None):
    if project_root is not None:
        return Path(project_root).expanduser().resolve()
    from scp_pipeline.workspaces import _scope
    return _scope(repo_root, project_name).parent


def settings_path(root, path=None):
    raw = Path(path or SETTINGS_FILE).expanduser()
    # Keep the lexical final component so symlink checks cannot be resolved away.
    raw = raw if raw.is_absolute() else Path(root) / raw
    return raw.parent.resolve() / raw.name


def _json_value(value, label):
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item, label)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json_value(item, label)
        return
    raise ValueError(f'{label} must contain only finite JSON values, not runtime objects.')


def validate_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError('Notebook settings must be a JSON object.')
    unknown = set(settings) - SETTING_KEYS
    if unknown:
        raise ValueError(f'Unsupported notebook settings: {sorted(unknown)}. '
                         'Navigation and model recipes do not belong in notebook defaults.')
    _json_value(settings, 'Notebook settings')
    for key, value in settings.items():
        default = PIPELINE_UI_DEFAULTS[key]
        nullable = default is None or key == 'fi_protocol_overrides'
        if value is None and nullable:
            continue
        valid = True
        if key in _INTEGER_KEYS:
            valid = type(value) is int
            if valid:
                valid = value >= (1 if key in {'n_trials', 'act_n_cpus', 'input_preview_plot_columns'} else 0)
        elif key in _TEXT_KEYS:
            valid = isinstance(value, str)
        elif isinstance(default, bool):
            valid = isinstance(value, bool)
        elif isinstance(default, dict) or key == 'fi_protocol_overrides':
            valid = isinstance(value, dict)
        elif isinstance(default, list) or key in _LIST_KEYS:
            valid = isinstance(value, list)
            if valid and key.endswith('amps_pA'):
                valid = bool(value) and all(type(v) in (int, float) for v in value)
            elif valid:
                valid = all(isinstance(v, str) for v in value)
        elif isinstance(default, str):
            valid = isinstance(value, str)
        else:
            valid = type(value) in (int, float)
        if not valid:
            raise ValueError(f'Invalid notebook setting {key}: {value!r}.')
    # Catch values that would fail widget population before applying anything.
    choices = {
        'input_preview_plot_size': {'compact', 'standard'},
        'diagnostic_window_mode': {'stimulus', 'full', 'manual'},
        'diagnostic_raster_style': {'dot', 'line'},
        'diagnostic_figure_size': {'compact', 'standard'},
    }
    for key, allowed in choices.items():
        if key in settings and settings[key] not in allowed:
            raise ValueError(f'{key} must be one of {sorted(allowed)}.')
    for key, allowed in {
        'input_preview_plots': {'weight_distribution', 'distance_distribution', 'weight_vs_distance'},
        'diagnostic_plots': {'input_rate', 'input_raster', 'membrane_voltage', 'output_rate', 'output_raster'},
    }.items():
        if key in settings and not set(settings[key]).issubset(allowed):
            raise ValueError(f'{key} contains unsupported plot names.')
    if settings.get('input_preview_plot_columns', 1) > 3:
        raise ValueError('input_preview_plot_columns must be 1, 2, or 3.')
    for key in ('passive_protocol_overrides', 'active_protocol_overrides', 'fi_protocol_overrides', 'passive_target_overrides'):
        for field, value in (settings.get(key) or {}).items():
            if field == 'conditions' and key != 'passive_target_overrides' and isinstance(value, dict):
                values = value.values()
            else:
                values = [value]
            if any(item is not None and type(item) not in (int, float) for item in values):
                raise ValueError(f'{key}.{field} must contain numeric overrides or null.')
    simulation = settings.get('simulation_overrides') or {}
    routing = {'cell_config', 'cell_loader', 'paths', 'model_source', 'project_name', 'cell_name',
               'tune_name', 'tune_dir', 'workspace_identity', 'workspace_project_root', 'output_dir'}
    if routing & simulation.keys():
        raise ValueError('Notebook simulation overrides cannot contain construction or workspace routing.')
    for block in ('conditions', 'iclamp', 'save', 'cell_recording'):
        if block in simulation and not isinstance(simulation[block], dict):
            raise ValueError(f'simulation_overrides.{block} must be an object.')
    for block, names in (
        (simulation, ('tstart', 'tstop', 'dt', 'bins', 'jitter', 'stim_start_ms', 'stim_duration_ms',
                      'input_stats_bin_ms', 'n_traces_to_save')),
        (simulation.get('conditions', {}), ('v_init_mV', 'celsius_C')),
        (simulation.get('iclamp', {}), ('amp_nA', 'delay_ms', 'dur_ms', 'tstop_ms', 'dt_ms')),
        (simulation.get('cell_recording', {}), ('n_trials',)),
    ):
        for name in names:
            if name in block and block[name] is not None and type(block[name]) not in (int, float):
                raise ValueError(f'simulation_overrides {name} must be numeric or null.')
    recording_vars = simulation.get('cell_recording', {}).get('vars', {})
    if not isinstance(recording_vars, dict):
        raise ValueError('simulation_overrides.cell_recording.vars must be an object.')
    for section in ('target', 'simulation', 'optimizer', 'filter', 'act_cell', 'modules'):
        if section in (settings.get('act_overrides') or {}) and not isinstance(settings['act_overrides'][section], dict):
            raise ValueError(f'act_overrides.{section} must be an object.')
    return deepcopy(settings)


def profile_document(settings):
    return {'format_version': 1, 'kind': 'scp_notebook_settings',
            'settings': validate_settings(settings)}


def parse_profile(value):
    if isinstance(value, (str, bytes)):
        try:
            value = json.loads(value)
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f'Cannot read notebook defaults JSON: {exc}') from exc
    if (not isinstance(value, dict) or type(value.get('format_version')) is not int
            or value['format_version'] != 1 or value.get('kind') != 'scp_notebook_settings'
            or set(value) != {'format_version', 'kind', 'settings'}):
        raise ValueError('Expected a format_version=1 scp_notebook_settings profile, '
                         'not a model or project configuration.')
    return validate_settings(value['settings'])


def read_profile(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f'Refusing symlinked notebook defaults: {path}')
    try:
        return parse_profile(path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        raise ValueError(f'Cannot load notebook defaults {path}: {exc}') from exc


def write_profile(path, settings):
    path = Path(path)
    document = profile_document(settings)
    if path.is_symlink():
        raise ValueError(f'Refusing symlinked notebook defaults: {path}')
    if path.exists():
        read_profile(path)  # Never overwrite an unrelated or malformed JSON file.
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(document, handle, indent=2, allow_nan=False)
        handle.write('\n')
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return document


def preferences(settings):
    return {key: deepcopy(value) for key, value in settings.items() if key in SETTING_KEYS}
