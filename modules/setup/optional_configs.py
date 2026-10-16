"""Non-executing, create-only setup of optional tune configuration files."""
from copy import deepcopy
import json
import math
from pathlib import Path

from .defaults import default_target_config, default_syn_group_templates, default_syn_config_for_templates
from .target_config import infer_target_source_mode, _has_mapping_data
from scp_pipeline.loaders.source_context import within, sha256_file, guard_source_destinations


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f'{label} must contain finite numbers.')


def _merge_known(target, values, label):
    if values is None:
        return
    if not isinstance(values, dict) or set(values) - set(target):
        raise ValueError(f'{label} contains unsupported fields or is not an object.')
    target.update(deepcopy(values))


def target_payload(options, tune_dir):
    """Validate entered targets without extracting data or importing ACT."""
    allowed = {'target_source_mode', 'include_manual_with_file_source', 'manual_passive',
               'manual_fi_curve', 'passive_trace', 'active_trace', 'allen_nwb',
               'target_description', 'notes'}
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError('Unsupported target generation options.')
    for key in ('manual_passive', 'manual_fi_curve', 'passive_trace', 'active_trace', 'allen_nwb'):
        if options.get(key) is not None and not isinstance(options[key], dict):
            raise ValueError(f'{key} must be a configuration object.')
    if not isinstance(options.get('include_manual_with_file_source', False), bool):
        raise ValueError('include_manual_with_file_source must be true or false.')
    mode = infer_target_source_mode(**{key: options.get(key) for key in
        ('target_source_mode', 'passive_trace', 'active_trace', 'allen_nwb')})
    include = (bool(options.get('include_manual_with_file_source'))
               or _has_mapping_data(options.get('manual_passive'))
               or _has_mapping_data(options.get('manual_fi_curve')))
    payload = default_target_config(mode=mode, include_manual_with_file_source=include)
    if 'manual' in payload:
        for key, source in (('passive', 'manual_passive'), ('fi_curve', 'manual_fi_curve')):
            _merge_known(payload['manual'][key], options.get(source), source)
        passive = payload['manual']['passive']
        for key, value in passive.items():
            if value is not None:
                _finite(value, key)
                if key != 'v_rest_mV' and value <= 0:
                    raise ValueError(f'{key} must be positive.')
        fi = payload['manual']['fi_curve']
        currents, rates = fi['currents_pA'], fi['rates_Hz']
        if not isinstance(currents, list) or not isinstance(rates, list) or len(currents) != len(rates):
            raise ValueError('FI current and rate arrays must have the same length.')
        for value in currents + rates:
            _finite(value, 'FI arrays')
        if any(value < 0 for value in rates):
            raise ValueError('FI rates must be nonnegative.')
        if fi.get('csv') and currents:
            raise ValueError('Choose FI points or an FI CSV, not both.')
    if mode == 'traces':
        payload['traces']['passive']['format'] = payload['traces']['format']
        _merge_known(payload['traces']['passive'], options.get('passive_trace'), 'passive_trace')
        _merge_known(payload['traces']['active'], options.get('active_trace'), 'active_trace')
        if payload['traces']['passive']['format'] not in {'csv', 'npy'}:
            raise ValueError('Passive trace format must be csv or npy.')
        if payload['traces']['active']['format'] != 'npy':
            raise ValueError('Active trace targets currently require ACT-compatible npy files.')
    if mode == 'allen_nwb':
        nwb = options.get('allen_nwb') or {}
        if not isinstance(nwb, dict) or set(nwb) - set(payload['allen_nwb']):
            raise ValueError('Unsupported Allen NWB options.')
        for key, value in nwb.items():
            if key in {'passive', 'active'}:
                _merge_known(payload['allen_nwb'][key], value, f'allen_nwb.{key}')
            else:
                payload['allen_nwb'][key] = deepcopy(value)
    for key in ('notes',):
        if options.get(key) is not None:
            payload[key] = str(options[key])
    if options.get('target_description') is not None:
        payload['target_source']['description'] = str(options['target_description'])

    inputs, warnings = [], []
    numeric = {'stim_start_ms', 'stim_stop_ms', 'dt_ms', 'current_pA', 'end_margin_ms',
               'min_current_pA', 'max_current_pA', 'spike_threshold_mV', 'refractory_ms'}

    def check(block, prefix=''):
        for key, value in block.items():
            label = f'{prefix}.{key}' if prefix else key
            if isinstance(value, dict):
                check(value, label)
            elif key in numeric and value is not None:
                _finite(value, label)
                if key in {'dt_ms', 'refractory_ms'} and value <= 0:
                    raise ValueError(f'{label} must be positive.')
            elif key == 'sweep_ids' and value is not None:
                if not isinstance(value, list) or any(type(n) is not int or n < 0 for n in value):
                    raise ValueError(f'{label} must be a list of nonnegative integers.')
            elif key == 'stimulus_names' and value is not None:
                if not isinstance(value, list) or any(not isinstance(name, str) or not name.strip() for name in value):
                    raise ValueError(f'{label} must be a list of stimulus names.')
            elif key == 'reducer' and value not in (None, '', 'median', 'mean'):
                raise ValueError(f'{label} must be median or mean.')
            elif key == 'tau_field' and value not in (None, '', 'tau_avg_ms', 'tau_rest_to_trough_ms'):
                raise ValueError(f'{label} must be tau_avg_ms or tau_rest_to_trough_ms.')
            elif key in {'file', 'csv'} and value not in (None, ''):
                if not isinstance(value, str):
                    raise ValueError(f'{label} must be a file path.')
                path = Path(value).expanduser()
                path = (path if path.is_absolute() else Path(tune_dir) / path).resolve()
                if not path.is_file():
                    raise FileNotFoundError(f'Target input {label} not found: {path}')
                inputs.append({'field': label, 'path': str(path)})
        start, stop = block.get('stim_start_ms'), block.get('stim_stop_ms')
        if start is not None and stop is not None and stop <= start:
            raise ValueError(f'{prefix}: stimulus stop must be after start.')
        low, high = block.get('min_current_pA'), block.get('max_current_pA')
        if low is not None and high is not None and low > high:
            raise ValueError(f'{prefix}: minimum current cannot exceed maximum current.')

    check(payload)
    if mode == 'manual' and not any(v is not None for v in payload['manual']['passive'].values()) and not inputs and not payload['manual']['fi_curve']['currents_pA']:
        warnings.append('Incomplete: blank manual target template; enter targets later before comparing measurements.')
    file_source_supplied = (any(payload['traces'][key].get('file') for key in ('passive', 'active'))
                            if mode == 'traces' else payload['allen_nwb'].get('file') if mode == 'allen_nwb' else True)
    if not file_source_supplied:
        warnings.append('Incomplete target template: no data files supplied yet.')
    if mode == 'traces' and payload['traces']['active'].get('file'):
        warnings.append('Active NPY targets are ACT-compatible passthrough only; no new active-trace conversion is performed.')
    json.dumps(payload, allow_nan=False)
    return payload, inputs, warnings


def preview_setup_outputs(tune_dir, *, target_options=None, synapse_options=None, cell_config=None,
                          include_required=False, conditions=None, runtime_sim_config=None):
    """Preview missing optional files; existing files are never filled or overwritten."""
    tune = Path(tune_dir).expanduser().resolve()
    files, inputs, warnings = [], [], []
    if synapse_options is not None:
        if not isinstance(synapse_options, dict) or set(synapse_options) - {'template_kinds', 'weight_style'}:
            raise ValueError('Unsupported synapse generation options.')
        group_payloads = default_syn_group_templates(**synapse_options)
    else:
        group_payloads = {}

    def destination(relative):
        path = tune / relative
        if path.is_symlink() or not within(path.resolve(), tune):
            raise ValueError(f'Config destination escapes the selected tune: {path}')
        for parent in path.parents:
            if parent.exists() and not parent.is_dir():
                raise ValueError(f'Config parent is not a directory: {parent}')
            if parent == tune:
                break
        if path.exists() and not path.is_file():
            raise ValueError(f'Config destination is not a file: {path}')
        if cell_config and cell_config.get('model_source'):
            guard_source_destinations(cell_config, tune, [path])
        return path

    if include_required:
        from .defaults import default_sim_config, default_geometry_config
        from scp_pipeline.simulation.current_injection import validate_required_sim_conditions
        if not isinstance(cell_config, dict) or not cell_config:
            raise ValueError('Save/register a model recipe before generating its missing configs.')
        sim_path = destination('cell_configs/sim_config.json')
        geometry_path = destination('cell_configs/geometry.json')
        sim = (json.loads(sim_path.read_text()) if sim_path.is_file()
               else default_sim_config(cell_name=cell_config.get('cell_name', 'Cell'), specimen_id=None,
                                       model_type=cell_config.get('model_type', 'perisomatic')))
        if not isinstance(sim, dict):
            raise ValueError(f'Expected a configuration object: {sim_path}')
        if not sim_path.exists() and conditions:
            sim['conditions'] = deepcopy(conditions)
        from scp_pipeline.simulation.session import _merge_overrides
        effective_sim = _merge_overrides(deepcopy(sim), runtime_sim_config)
        validate_required_sim_conditions(cell_config, effective_sim)
        if not sim_path.exists():
            sim = effective_sim
        for path, payload in ((sim_path, sim), (geometry_path, default_geometry_config(cell_name=cell_config.get('cell_name', 'Cell')))):
            if path.exists():
                if not isinstance(json.loads(path.read_text()), dict):
                    raise ValueError(f'Expected a configuration object: {path}')
                files.append({'path': str(path), 'status': 'preserved', 'sha256': sha256_file(path)})
            else:
                files.append({'path': str(path), 'status': 'create', 'contents': payload})
        # Syntax/shape checks only: never instantiate synapses or extract targets.
        for relative in ('cell_configs/target_config.json', 'cell_configs/syn_config.json'):
            path = destination(relative)
            if path.is_file():
                payload = json.loads(path.read_text())
                if not isinstance(payload, dict):
                    raise ValueError(f'Expected a configuration object: {path}')
                if path.name == 'target_config.json':
                    source = payload.get('target_source')
                    # Scalar/missing sources are retained legacy manual configs.
                    if isinstance(source, dict):
                        infer_target_source_mode(target_source_mode=source.get('mode'))
                if path.name == 'syn_config.json':
                    groups = payload.get('group_files', [])
                    if not isinstance(groups, list) or any(not isinstance(name, str) for name in groups):
                        raise ValueError(f'{path}: group_files must be a list of file paths.')
                    for name in groups:
                        group = (path.parent / name).resolve()
                        if not group.is_file():
                            if group in {(tune / 'cell_configs/syn_groups' / filename).resolve() for filename in group_payloads}:
                                continue
                            raise FileNotFoundError(f'Synapse group config not found: {group}')
                        if not isinstance(json.loads(group.read_text()), dict):
                            raise ValueError(f'Expected a synapse-group object: {group}')

    if target_options is not None:
        target = destination('cell_configs/target_config.json')
        if target.exists():
            files.append({'path': str(target), 'status': 'preserved', 'sha256': sha256_file(target)})
        else:
            payload, inputs, notes = target_payload(target_options, tune)
            warnings.extend(notes)
            files.append({'path': str(target), 'status': 'create', 'contents': payload})
    if synapse_options is not None:
        payloads = {f'cell_configs/syn_groups/{name}': data for name, data in group_payloads.items()}
        payloads['cell_configs/syn_config.json'] = default_syn_config_for_templates(filenames=group_payloads)
        for relative, payload in payloads.items():
            path = destination(relative)
            files.append({'path': str(path), 'status': 'preserved' if path.exists() else 'create',
                          **({'sha256': sha256_file(path)} if path.exists() else {'contents': payload})})
        if (tune / 'cell_configs/syn_config.json').exists():
            warnings.append('Existing syn_config.json is preserved; new disabled templates are not automatically added to its group list.')
    return {'status': 'preview', 'tune_dir': str(tune), 'files': files,
            'resolved_inputs': inputs, 'warnings': warnings, 'constructs_cell': False}


def create_setup_outputs(tune_dir, *, target_options=None, synapse_options=None, cell_config=None,
                         include_required=False, conditions=None, runtime_sim_config=None):
    """Apply a fully validated create-only preview without modifying existing files."""
    report = preview_setup_outputs(tune_dir, target_options=target_options,
                                   synapse_options=synapse_options, cell_config=cell_config,
                                   include_required=include_required, conditions=conditions,
                                   runtime_sim_config=runtime_sim_config)
    created = []
    try:
        for record in report['files']:
            if record['status'] != 'create':
                continue
            path = Path(record['path'])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x', encoding='utf-8') as handle:
                created.append(path)
                json.dump(record['contents'], handle, indent=2, allow_nan=False)
                handle.write('\n')
            record['status'] = 'created'
    except Exception:
        for path in reversed(created):
            path.unlink()
        raise
    report['status'] = 'created' if created else 'unchanged'
    return report
