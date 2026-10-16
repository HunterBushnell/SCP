"""Initial optional-file drafts and file-only working-copy controls."""
from copy import deepcopy
import json
import math
from pathlib import Path
from html import escape

from scp_pipeline.setup.defaults import default_target_config, SYNAPSE_TEMPLATE_FILENAMES


def _parse(raw, default, label):
    if isinstance(default, bool):
        return bool(raw)
    raw = raw.strip()
    if not raw:
        return None
    if isinstance(default, list):
        value = json.loads(raw)
        if not isinstance(value, list):
            raise ValueError(f'{label} must be a JSON list.')
        return value
    if default is None or isinstance(default, (int, float)):
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f'{label} must be a finite number or blank.') from exc
        if not math.isfinite(value):
            raise ValueError(f'{label} must be a finite number or blank.')
        return value
    return raw


def build_optional_controls(ui):
    w, c = ui.widgets, ui.controls
    c.update({
        'setup_targets_enabled': w.Checkbox(description='Generate missing target config', value=False, indent=False),
        'setup_target_mode': w.Dropdown(options=['manual', 'traces', 'allen_nwb', 'none'], value='manual', description='Target mode'),
        'setup_target_manual': w.Checkbox(description='Also include manual targets', value=False, indent=False),
        'setup_target_existing': w.HTML(),
        'setup_synapses_enabled': w.Checkbox(description='Generate missing disabled synapse templates', value=False, indent=False,
                                            layout=w.Layout(width='100%')),
        'setup_synapse_kinds': w.SelectMultiple(options=list(SYNAPSE_TEMPLATE_FILENAMES), value=('input_blocks',), description='Kinds'),
        'setup_synapse_weights': w.Dropdown(options=['distributed', 'fixed'], description='Weights'),
        'setup_draft_discard': w.Checkbox(description='Discard unsaved setup draft on next navigation or load', value=False, indent=False,
                                         layout=w.Layout(display='none', width='100%')),
        'setup_copy_name': w.Text(value='tuned', description='New tune'),
        'setup_copy_mode': w.Dropdown(options=['inherit', 'copy', 'reference'], description='Use sources'),
        'setup_copy_preview': w.Button(description='Preview copy'),
        'setup_copy_create': w.Button(description='Create working copy'),
        'setup_copy_summary': w.HTML(),
    })
    # Preserve the existing schema/defaults, rendering blank optional numbers.
    schema = default_target_config(mode='traces', include_manual_with_file_source=True)
    schema['allen_nwb'] = default_target_config(mode='allen_nwb')['allen_nwb']
    fields = {}
    path_keys = {'file', 'csv'}
    text_keys = {'time_column', 'voltage_column', 'current_column', 'sweep_column'}

    def field(path, default):
        key = 'setup_target_' + '_'.join(path)
        label = path[-1]
        if isinstance(default, bool):
            widget = w.Checkbox(value=default, description=label, indent=False)
        else:
            string = ('' if default is None else json.dumps(default) if isinstance(default, list) else str(default))
            widget = w.Text(value=string, description=label, layout=w.Layout(width='100%'),
                            style={'description_width': '160px'})
        widget.tooltip = '.'.join(path) + (' — relative paths resolve from the selected tune, not the notebook.' if label in path_keys else '')
        c[key] = widget
        fields[key] = (path, default, label in path_keys | text_keys)
        return widget

    def block(path, defaults):
        children = []
        for key, value in defaults.items():
            if isinstance(value, dict):
                children += [w.HTML(f'<b>{escape(key)}</b>'), block((*path, key), value)]
            else:
                children.append(field((*path, key), value))
        return w.VBox(children)

    def advanced_box(name, title, child):
        toggle = w.ToggleButton(description=title, value=False, layout=w.Layout(width='300px'))
        child.layout.display = 'none'
        toggle.observe(lambda change: setattr(child.layout, 'display', '' if change['new'] else 'none'), names='value')
        c[name] = toggle
        return w.VBox([toggle, child])

    manual = block(('manual',), schema['manual'])
    # FI arrays remain the existing JSON arrays, with no scientific values supplied.
    trace_defaults = schema['traces']
    trace_defaults['passive']['format'] = trace_defaults['format']
    traces = w.VBox([w.HTML('<small>Passive CSV/NPY follows the existing trace schema. Active targets require ACT-compatible '
                            'NPY arrays (trial × time × columns, voltage mV/current nA); SCP does not convert arbitrary active traces. '
                            'No extraction is performed by setup.</small>')])
    traces.children += (block(('traces', 'passive'), {'file': None}), block(('traces', 'active'), {'file': None}))
    trace_advanced = advanced_box('setup_target_trace_advanced', 'Advanced trace settings', w.VBox([
        w.HTML('<b>Passive extraction settings</b>'), block(('traces', 'passive'), {k: v for k, v in trace_defaults['passive'].items() if k != 'file'}),
        w.HTML('<b>Active passthrough settings</b>'), block(('traces', 'active'), {k: v for k, v in trace_defaults['active'].items() if k != 'file'})]))
    traces.children += (trace_advanced,)
    nwb = w.VBox([block(('allen_nwb',), {'file': None})])
    nwb_advanced = advanced_box('setup_target_nwb_advanced', 'Advanced NWB extraction settings',
                               block(('allen_nwb',), {k: v for k, v in schema['allen_nwb'].items() if k != 'file'}))
    nwb.children += (nwb_advanced,)
    # File paths and nullable list/string fields need schema-specific parsing.
    for key, (path, default, is_text) in list(fields.items()):
        if path[-1] == 'sweep_ids' and default is None:
            fields[key] = (path, [], False)
    c['setup_target_form'] = w.VBox([c['setup_target_mode'], c['setup_target_manual'], manual, traces, nwb])
    c['setup_synapse_form'] = w.VBox([c['setup_synapse_kinds'], c['setup_synapse_weights'],
        w.HTML('<small>Templates are disabled. This neither creates inputs/synapses nor imports BMTool. Edit the generated configs before using synapses.</small>')])
    optional = w.VBox([w.HTML('<b>Optional configuration generation</b><p><small>Required cell, geometry, and runtime configs are created as needed. '
        'sim_config.json provides conditions for Steps 2–3; its Step 5 fields do not require running Step 5. '
        'Existing scientific files are never overwritten here. Targets may be blank/incomplete and edited later in the config.</small></p>'),
        c['setup_target_existing'], c['setup_targets_enabled'], c['setup_target_form'],
        c['setup_synapses_enabled'], c['setup_synapse_form'], c['setup_draft_discard']])
    c['setup_optional_box'] = optional
    c['setup_copy_box'] = w.VBox([w.HTML('<b>Create a working copy</b><p><small>Copy the selected saved tune, including its saved source snapshot—not today’s upstream sources. '
        'Previous results, exports, archives, compiled files, and caches are excluded. Copy mode is independent; two reference tunes share external edits. '
        'Creation selects the new tune but does not load it.</small></p>'), c['setup_copy_summary'], c['setup_copy_name'], c['setup_copy_mode']])
    ui._setup_target_fields = fields
    ui._setup_draft_keys = ['setup_targets_enabled', 'setup_target_mode', 'setup_target_manual',
                            'setup_synapses_enabled', 'setup_synapse_kinds', 'setup_synapse_weights', *fields]
    ui._setup_draft_initial = draft_values(ui)
    ui._setup_draft_baseline = draft_values(ui)
    def visibility(_change=None):
        update_optional_visibility(ui)
        enabled = c['setup_targets_enabled'].value and not getattr(ui, '_setup_has_target', False)
        mode = c['setup_target_mode'].value
        c['setup_target_form'].layout.display = '' if enabled else 'none'
        c['setup_target_manual'].layout.display = '' if mode in {'traces', 'allen_nwb'} else 'none'
        manual.layout.display = '' if mode == 'manual' or (mode in {'traces', 'allen_nwb'} and c['setup_target_manual'].value) else 'none'
        traces.layout.display = '' if mode == 'traces' else 'none'
        nwb.layout.display = '' if mode == 'allen_nwb' else 'none'
        c['setup_synapse_form'].layout.display = '' if c['setup_synapses_enabled'].value else 'none'
    ui._setup_optional_visibility = visibility
    for key in ('setup_targets_enabled', 'setup_target_mode', 'setup_target_manual', 'setup_synapses_enabled'):
        c[key].observe(visibility, names='value')
    for key in ui._setup_draft_keys:
        c[key].observe(lambda _change: setattr(c['setup_draft_discard'], 'value', False), names='value')
    return optional, c['setup_copy_box']


def update_optional_visibility(ui):
    c = ui.controls
    try:
        tune = ui._selected_tune_dir()
        target = tune / 'cell_configs/target_config.json'
        exists = target.is_file()
        c['setup_target_existing'].value = (f'<small>Existing targets (read-only here): {escape(str(target))}. Edit this config directly for later changes.</small>' if exists else '')
        ui._setup_has_target = exists
        c['setup_targets_enabled'].disabled = exists or ui._setup_attempted
        has_recipe = (tune / 'cell_configs/cell_config.json').is_file() or (tune / 'cell_config.json').is_file()
        c['setup_copy_box'].layout.display = '' if has_recipe and c['setup_action'].value == 'existing' else 'none'
        path = tune / 'cell_configs/cell_config.json'
        config = json.loads(path.read_text()) if path.is_file() else {}
        external = bool(config.get('model_source'))
        c['setup_copy_mode'].layout.display = '' if external else 'none'
        c['setup_copy_summary'].value = f'<small>Source: {escape(str(tune))}; destination is a sibling tune. Source mode: {escape((config.get("model_source") or {}).get("mode", "tune-local"))}.</small>'
    except (ValueError, OSError, TypeError):
        ui._setup_has_target = False
        c['setup_copy_box'].layout.display = 'none'


def draft_values(ui):
    return {key: deepcopy(ui.controls[key].value) for key in getattr(ui, '_setup_draft_keys', [])}


def remember_draft(ui):
    ui._setup_draft_baseline = draft_values(ui)
    ui.controls['setup_draft_discard'].value = False
    ui.controls['setup_draft_discard'].layout.display = 'none'


def reset_draft(ui):
    for key, value in ui._setup_draft_initial.items():
        ui.controls[key].value = value
    remember_draft(ui)


def guard_draft(ui):
    if draft_values(ui) == getattr(ui, '_setup_draft_baseline', {}):
        return
    confirm = ui.controls['setup_draft_discard']
    if not confirm.value:
        ui.controls['setup_manage_toggle'].value = True
        confirm.layout.display = ''
        raise ValueError('Unsaved optional-setup draft. Save cell setup first, or check Discard unsaved setup draft in Cell setup and navigate again.')
    reset_draft(ui)


def generation_options(ui):
    c = ui.controls
    target = None
    if c['setup_targets_enabled'].value and not getattr(ui, '_setup_has_target', False):
        mode = c['setup_target_mode'].value
        target = {'target_source_mode': mode, 'include_manual_with_file_source': c['setup_target_manual'].value}
        blocks = {}
        for key, (path, default, text) in ui._setup_target_fields.items():
            if path[0] == 'manual' and not (mode == 'manual' or mode in {'traces', 'allen_nwb'} and c['setup_target_manual'].value):
                continue
            if path[0] not in {'manual', mode}:
                continue
            raw = c[key].value
            value = (raw.strip() or None) if text else _parse(raw, default, '.'.join(path))
            if isinstance(default, list) and value is None and path[-1] != 'sweep_ids':
                value = []
            block = blocks
            for part in path[:-1]:
                block = block.setdefault(part, {})
            block[path[-1]] = value
        if 'manual' in blocks:
            target.update(manual_passive=blocks['manual']['passive'], manual_fi_curve=blocks['manual']['fi_curve'])
        if 'traces' in blocks:
            target.update(passive_trace=blocks['traces']['passive'], active_trace=blocks['traces']['active'])
        if 'allen_nwb' in blocks:
            target['allen_nwb'] = blocks['allen_nwb']
    syn = None
    if c['setup_synapses_enabled'].value:
        if not c['setup_synapse_kinds'].value:
            raise ValueError('Select at least one disabled synapse template kind, or disable generation.')
        syn = {'template_kinds': list(c['setup_synapse_kinds'].value), 'weight_style': c['setup_synapse_weights'].value}
    return {'target_options': target, 'synapse_options': syn}
