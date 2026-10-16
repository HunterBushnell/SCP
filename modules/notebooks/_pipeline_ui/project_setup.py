"""Project registration controls; discovery and Save never construct a cell."""
from __future__ import annotations

import json
import math
import os
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from html import escape

from scp_pipeline.projects import PROJECT_FILE, load_project_settings, resolve_project_path, save_project_settings
from scp_pipeline.setup.registration import preview_workspace_registration, register_workspace
from .setup_outputs import (build_optional_controls, generation_options, guard_draft, reset_draft)


def navigation_root(ui):
    if ui.project_root is None:
        return ui.repo_root
    return resolve_project_path(ui.project_root, ui.workspace_root or '.')


_RECIPE_FIELDS = (
    'setup_loader', 'setup_entry', 'setup_template', 'setup_callable', 'setup_args',
    'setup_kwargs', 'setup_sections', 'setup_source_root', 'setup_source_mode',
    'setup_modfiles', 'setup_folders', 'setup_files', 'setup_working_dir',
    'setup_voltage', 'setup_celsius', 'setup_specimen', 'setup_model_type',
)
_FORM_FIELDS = (
    'workspace_root', 'project_label', 'output_dir', 'setup_action',
    'cell_name', 'tune_name', 'tune_dir_override', *_RECIPE_FIELDS,
)
_OUTPUT_ACTIONS = ('setup_open', 'setup_discover', 'setup_preview', 'setup_save',
                   'setup_project_save', 'setup_copy_preview', 'setup_copy_create')


def _form_values(ui):
    return {key: ui.controls[key].value for key in _FORM_FIELDS}


def _remember_form(ui):
    ui._project_settings_baseline = _form_values(ui)
    ui.controls['setup_discard_edits'].value = False
    ui.controls['setup_discard_edits'].layout.display = 'none'


def _initial_action(ui):
    if ui.cell_override is not None:
        return 'live'
    from scp_pipeline.workspaces import discover_cells, discover_tunes
    root = navigation_root(ui)
    project = ui._selected_project()
    cells = discover_cells(root, project)
    if ui.project_root is not None or not cells:
        candidates = [root / 'cells' / cell / 'tunes' / tune
                      for cell in cells for tune in discover_tunes(root, cell, project)]
        if ui.settings.get('tune_dir_override'):
            candidates.append(resolve_project_path(ui.project_root or ui.repo_root, ui.settings['tune_dir_override']))
        if not any((tune / 'cell_configs/cell_config.json').is_file()
                   or (tune / 'cell_config.json').is_file() for tune in candidates):
            return 'register'
    return 'existing'


def _feedback(ui, action_key, message, *, kind='complete'):
    # Setup actions do not replace each other's feedback or the model-load status.
    ui._set_status(action_key, kind, message)
    if action_key == 'setup_project_save':
        ui._status_kinds['setup_remember'] = kind  # Legacy helper key; one shared action.


def build_project_controls(ui):
    w = ui.widgets
    from .text_output import TextOutput
    from ..notebook_settings import settings_root
    owner = settings_root(ui.repo_root, ui.project_root, ui.settings.get('project_name'))
    saved = load_project_settings(owner)
    controls = {
        'workspace_mode': w.Dropdown(options=[('SCP workspaces', 'scp'), ('Project-owned workspace', 'project')],
                                     value='project' if ui.project_root else 'scp', description='Location'),
        'project_root': w.Text(value=str(owner), description='Project dir', layout=w.Layout(width='650px')),
        'workspace_root': w.Text(value=str(ui.workspace_root or saved.get('workspace_root') or '.'), description='Workspaces'),
        'project_label': w.Text(value=saved['project_name'], description='Project name'),
        'output_dir': w.Text(value=str(ui.settings.get('output_dir') or ''), description='Results dir',
                             placeholder='blank = selected tune/output_data', layout=w.Layout(width='650px')),
        'setup_action': w.Dropdown(options=[('Open existing workspace', 'existing'),
                                           ('Register new model', 'register'), ('Use supplied live cell', 'live')],
                                  value=_initial_action(ui), description='Action'),
        'setup_loader': w.Dropdown(options=[('HOC template', 'hoc_template'), ('Python builder', 'python_factory'),
                                           ('Existing Allen manifest', 'allen_manifest'), ('Allen download', 'adb')], description='Model type'),
        'setup_entry': w.Text(description='Entry file', placeholder='path to the model file, not a directory', layout=w.Layout(width='650px')),
        'setup_template': w.Text(description='Template'),
        'setup_callable': w.Text(description='Callable'),
        'setup_candidates': w.Dropdown(options=[('Discover templates first', '')], description='Candidates'),
        'setup_specimen': w.Text(description='Specimen ID'),
        'setup_model_type': w.Dropdown(options=['perisomatic', 'all_active'], description='Allen type'),
        'setup_voltage': w.Text(description='V init (mV)', placeholder='required for a new generic model'),
        'setup_celsius': w.Text(description='Temp (°C)', placeholder='required for a new generic model'),
        'setup_source_root': w.Text(description='Source root', placeholder='blank = entry-file directory', layout=w.Layout(width='650px')),
        'setup_modfiles': w.Text(description='MOD folder', placeholder='blank = built-in mechanisms only'),
        'setup_folders': w.Textarea(value='[]', description='Folders JSON'),
        'setup_files': w.Textarea(value='[]', description='Files JSON'),
        'setup_working_dir': w.Text(description='Working dir', placeholder='blank = entry-file directory'),
        'setup_source_mode': w.Dropdown(options=['copy', 'reference'], description='Use sources'),
        'setup_args': w.Textarea(value='[]', description='Args JSON'),
        'setup_kwargs': w.Textarea(value='{}', description='Kwargs JSON'),
        'setup_sections': w.Textarea(value='{}', description='Sections JSON', placeholder='owner-attribute mapping; {} uses canonical attributes'),
        'setup_discover': w.Button(description='Discover HOC templates'),
        'setup_discover_help': w.HTML('<small>Enter the HOC file path above, then discover its template names. '
                                      'This is not a file browser and does not load a cell.</small>'),
        'setup_open': w.Button(description='Open / reload project', icon='refresh',
                               layout=w.Layout(width='210px')),
        'setup_discard_edits': w.Checkbox(value=False, description='Discard unsaved edits', indent=False,
                                         layout=w.Layout(display='none', width='300px')),
        'setup_preview': w.Button(description='Preview model setup', layout=w.Layout(width='200px')),
        'setup_save': w.Button(description='Save cell setup', icon='save'),
        'setup_project_save': w.Button(description='Save project settings', icon='save', layout=w.Layout(width='200px')),
        'setup_manage_toggle': w.ToggleButton(description='Cell setup', value=_initial_action(ui) != 'existing', layout=w.Layout(width='200px')),
        'setup_location_summary': w.HTML(),
        'setup_model_summary': w.HTML(),
        'setup_action_help': w.HTML(),
        'setup_help': w.HTML(value=(
            '<details><summary>ⓘ <strong style="font-size:1.1em">Field guide</strong>: '
            'locations, labels, and model sources</summary>'
            '<dl><dt><b>Location</b></dt><dd>Use SCP workspaces for bundled/project cells inside SCP. '
            'Choose Project-owned workspace for a separate directory owning your work.</dd>'
            '<dt><b>Project dir</b></dt><dd>The directory owning .scp_project.json and generated work. '
            'It can be a new subfolder such as Single-Cells/SCP; the notebook may live elsewhere.</dd>'
            '<dt><b>Open / reload project</b></dt><dd>Apply this directory and reload its saved navigation '
            '<em>before</em> editing. It does not open a file editor or confirm your entries. '
            'No saved file means nothing is reset. Replacing unsaved edits requires confirmation.</dd>'
            '<dt><b>Project name</b></dt><dd>A label stored with saved work, not a folder rename.</dd>'
            '<dt><b>Workspaces</b></dt><dd>A directory relative to Project dir (or an absolute path). '
            'A dot (.) uses Project dir itself; cells/&lt;cell&gt;/tunes/&lt;tune&gt; lives below it.</dd>'
            '<dt><b>Cell / Tune</b></dt><dd>Dropdowns show all available names without filtering. '
            'For registration or a live cell, enter names in Cell setup below. '
            'An explicit Tune path takes precedence over these shorthand choices.</dd>'
            '<dt><b>Results dir</b></dt><dd>Blank uses the selected tune\'s output_data/. '
            'Relative paths resolve from Project dir in project-owned mode, '
            'or from the SCP root in SCP workspaces. The Results summary shows the resolved path.</dd>'
            '<dt><b>Save project settings</b></dt><dd>Remember the current location, project label, '
            'workspace root, Cell/Tune selection, and Results dir. No model is constructed and '
            'scientific configs are unchanged. Saving is optional before loading; '
            'Save cell setup separately registers a new model recipe.</dd>'
            '<dt><b>Model sources</b></dt><dd>Entry file selects native code, independently of the workspace. '
            'Copy uses the staged snapshot; reference uses the originals and still saves a snapshot. '
            'Declare MOD sources with MOD folder; extra dependency files/folders '
            'belong in Advanced model settings.</dd></dl></details>'
        )),
    }
    for key in _OUTPUT_ACTIONS:
        feedback = w.HTML()
        controls[f'{key}_feedback'] = feedback
        ui.statuses[key] = feedback
        ui.outputs[key] = TextOutput(layout=w.Layout(max_height='320px', overflow='auto'))
    # Keep old code-first helpers working without a second button or save path.
    controls['setup_remember'] = controls['setup_project_save']
    controls['setup_remember_feedback'] = controls['setup_project_save_feedback']
    ui.outputs['setup_remember'] = ui.outputs['setup_project_save']
    ui.statuses['setup_remember'] = ui.statuses['setup_project_save']
    ui.controls.update(controls)
    optional_box, copy_box = build_optional_controls(ui)
    controls.update({key: value for key, value in ui.controls.items() if key.startswith('setup_')})
    ui._project_recipe_defaults = {key: controls[key].value for key in _RECIPE_FIELDS}
    tooltips = {
        'workspace_mode': 'Choose SCP-contained workspaces or a project-owned directory outside SCP.',
        'project_root': 'Directory owning project settings and generated work; it need not contain this notebook.',
        'workspace_root': 'Relative to Project dir; a dot (.) places cells/<cell>/tunes/<tune> directly inside it.',
        'project_label': 'Project identity recorded with saved work. This does not rename a directory.',
        'output_dir': 'Blank uses selected tune/output_data. Relative paths use Project dir in project-owned mode or the SCP root in SCP workspaces; see the Results summary.',
        'setup_action': 'First use: Register new model. Existing configs: Open existing workspace. Live objects: use the code hook.',
        'setup_loader': 'Choose the saved model construction format. Selecting Python does not execute it.',
        'setup_entry': 'Native HOC, Python builder, or Allen manifest. Relative paths use Project dir in project-owned mode or the SCP root in SCP workspaces.',
        'setup_template': 'Exact HOC begintemplate name. Discover declarations without executing HOC.',
        'setup_callable': 'Explicit Python builder callable; discovery never imports Python code.',
        'setup_candidates': 'Choose a declaration from the selected HOC file; selection fills Template.',
        'setup_specimen': 'Allen specimen ID to download only when Save setup is explicitly clicked.',
        'setup_model_type': 'Allen model variant requested for acquisition.',
        'setup_voltage': 'Explicit initialization voltage in mV for the new model; stored in sim_config.json.',
        'setup_celsius': 'Explicit temperature in degrees Celsius; stored in sim_config.json.',
        'setup_source_root': 'Root preserving native relative paths. Blank uses the entry-file directory.',
        'setup_modfiles': 'MOD directory relative to Source root; blank means no custom MOD sources.',
        'setup_folders': 'JSON list of dependency folders relative to Source root. Only approved sources are staged.',
        'setup_files': 'JSON list of dependency files relative to Source root. The entry file is included automatically.',
        'setup_working_dir': 'Root-relative working directory for execution. Blank uses the entry-file directory.',
        'setup_source_mode': 'Copy: work on the local snapshot. Reference: load originals; snapshot remains a record.',
        'setup_args': 'JSON list of HOC constructor arguments or Python builder positional arguments.',
        'setup_kwargs': 'JSON object of Python builder keyword arguments.',
        'setup_sections': 'Map canonical soma/dend/apic/axon/all to owner attributes. Empty object uses canonical names.',
        'setup_discover': 'First enter the HOC path in Entry file. Read template declarations from that file; no file browser or model execution.',
        'setup_open': 'Optional: reload .scp_project.json before editing. This replaces unsaved selections, not scientific configs.',
        'setup_discard_edits': 'Explicitly allow the next reload to replace unsaved navigation and model selections.',
        'setup_preview': 'Validate selections and preview sources/config changes. No files are created and no model is loaded.',
        'setup_save': 'Save project navigation and register new sources/configs. Does not construct the cell.',
        'setup_project_save': 'Remember location, Cell/Tune, and Results dir in .scp_project.json. Does not save a new model recipe or construct a cell.',
        'setup_manage_toggle': 'Show model registration and advanced setup. Location and Results dir are above Save project settings.',
        'project_name': 'SCP-contained project namespace; distinct from a project-owned workspace label.',
        'cell_name': 'Show all existing cells. Enter a new cell name below in Cell setup when registering.',
        'tune_name': 'Show all existing tunes. Enter a new tune name below in Cell setup when registering. An orig/tuned pair is not required.',
        'tune_dir_override': 'Optional explicit workspace path. Takes precedence over Cell/Tune; relative to Project dir in project mode.',
    }
    for key, tooltip in tooltips.items():
        ui.controls[key].tooltip = tooltip
    advanced_fields = w.VBox([controls[key] for key in (
        'setup_folders', 'setup_files',
        'setup_working_dir', 'setup_args', 'setup_kwargs', 'setup_sections')])
    advanced_fields.layout.display = 'none'
    advanced_toggle = w.ToggleButton(description='Advanced model settings', value=False)
    advanced_toggle.observe(lambda change: setattr(advanced_fields.layout, 'display', '' if change['new'] else 'none'), names='value')
    advanced = w.VBox([advanced_toggle, advanced_fields])

    def action_box(key, button_row=None):
        box = w.VBox([
            button_row if button_row is not None else controls[key],
            controls[f'{key}_feedback'], ui.outputs[key],
        ], layout=w.Layout(margin='4px 0 8px 0', width='100%'))
        controls[f'{key}_box'] = box
        return box

    discovery_row = w.HBox([
        controls['setup_discover'], controls['setup_candidates'],
    ])
    discovery_box = action_box('setup_discover', discovery_row)
    discovery_box.children = (controls['setup_discover_help'], *discovery_box.children)
    conditions_row = w.HBox([controls['setup_voltage'], controls['setup_celsius']])
    registration = w.VBox([w.HTML('<b>Native model and runtime conditions</b>'),
                          controls['setup_loader'], controls['setup_entry'],
                          discovery_box,
                          controls['setup_template'], controls['setup_callable'],
                          w.HBox([controls['setup_specimen'], controls['setup_model_type']]),
                          controls['setup_source_root'], controls['setup_modfiles'], controls['setup_source_mode'],
                          conditions_row, advanced])

    def visibility(_change=None):
        registering = controls['setup_action'].value == 'register'
        try:
            needs_runtime = (controls['setup_action'].value != 'live'
                             and not (ui._selected_tune_dir() / 'cell_configs/sim_config.json').is_file())
        except ValueError:
            needs_runtime = False
        ui.controls['project_name'].layout.display = 'none' if controls['workspace_mode'].value == 'project' else ''
        for key in ('cell_name', 'tune_name'):
            ui.controls[key].allow_new = controls['setup_action'].value in {'register', 'live'}
            ui.controls[key].show_name_editor = ui.controls[key].allow_new
        if 'setup_name_entry_row' in controls:
            controls['setup_name_entry_row'].layout.display = '' if controls['setup_action'].value in {'register', 'live'} else 'none'
        has_discovery = bool(ui.outputs['setup_discover'].outputs
                             or controls['setup_discover_feedback'].value)
        # Saving switches to "existing"; keep previous discovery results visible
        # even when the editable registration fields are no longer relevant.
        registration.layout.display = '' if registering or has_discovery or needs_runtime else 'none'
        for child in registration.children:
            if child is not discovery_box:
                child.layout.display = '' if registering else 'none'
        conditions_row.layout.display = '' if registering or needs_runtime else 'none'
        loader = controls['setup_loader'].value
        discovering = registering and loader == 'hoc_template'
        discovery_row.layout.display = '' if discovering else 'none'
        discovery_box.layout.display = '' if discovering or has_discovery else 'none'
        for key in ('setup_template', 'setup_discover', 'setup_candidates'):
            controls[key].layout.display = '' if discovering else 'none'
        controls['setup_callable'].layout.display = '' if registering and loader == 'python_factory' else 'none'
        controls['setup_kwargs'].layout.display = '' if registering and loader == 'python_factory' else 'none'
        for key in ('setup_args', 'setup_sections'):
            controls[key].layout.display = '' if registering and loader in {'hoc_template', 'python_factory'} else 'none'
        for key in ('setup_source_root', 'setup_modfiles', 'setup_source_mode', 'setup_folders', 'setup_files', 'setup_working_dir'):
            controls[key].layout.display = '' if registering and loader != 'adb' else 'none'
        for key in ('setup_specimen', 'setup_model_type'):
            controls[key].layout.display = '' if registering and loader == 'adb' else 'none'
        controls['setup_entry'].layout.display = '' if registering and loader != 'adb' else 'none'
        for key in ('project_root', 'workspace_root', 'project_label', 'setup_open',
                    'setup_location_box', 'setup_project_details_box'):
            if key not in controls:
                continue
            controls[key].layout.display = '' if controls['workspace_mode'].value == 'project' else 'none'
        controls['setup_action_help'].value = {
            'existing': '<p>Select an existing Cell/Tune or Tune path. Its scientific configs stay authoritative. '
                        'Use Save project settings above to remember navigation and Results dir; '
                        'scientific configs are unchanged.</p>',
            'register': '<p>First-time setup: enter the Cell/Tune names below, then select the native model. '
                        'Preview model setup → Save cell setup → Prepare and load.</p>',
            'live': '<p>Create CELL_OVERRIDE in the optional code cell first. Use explicit runtime voltage/temperature; '
                    'the object stays in this kernel and must be recreated after a restart.</p>',
        }[controls['setup_action'].value]
        ui._setup_optional_visibility()

    def run(action_key, action, success):
        def callback(_button):
            output = ui.outputs[action_key]
            result, error = None, None
            _feedback(ui, action_key, 'Working…', kind='running')
            captured = StringIO()
            try:
                # Do not run navigation/widget updates inside Output capture.
                # These actions are file-only; append their logs explicitly so
                # exceptions and completion cannot disappear into a comm parent.
                output.outputs = ()
                with redirect_stdout(captured), redirect_stderr(captured):
                    if ui._setup_attempted:
                        raise ValueError('Restart before changing a loaded model.')
                    result = action()
                text = json.dumps(result, indent=2, default=str) + '\n'
            except Exception as exc:
                error = exc
                text = f'{type(exc).__name__}: {exc}\n'
            if captured.getvalue():
                output.append_stdout(captured.getvalue())
            output.append_stdout(text)
            if error is not None:
                _feedback(ui, action_key, error, kind='error')
            else:
                _feedback(ui, action_key, (result or {}).get('message') or success,
                          kind='warning' if (result or {}).get('status') == 'not_found' else 'complete')
        return callback

    controls['setup_loader'].observe(visibility, names='value')
    controls['setup_action'].observe(visibility, names='value')
    controls['workspace_mode'].observe(visibility, names='value')
    ui.outputs['setup_discover'].observe(visibility, names='outputs')
    def location_changed(change):
        if ui._setup_attempted or getattr(ui, '_changing_location', False):
            return
        defaults = ui._notebook_defaults
        try:
            guard_draft(ui)
            defaults.guard_replace()
        except ValueError as exc:
            ui._changing_location = True
            try:
                controls['workspace_mode'].value = change['old']
            finally:
                ui._changing_location = False
            _feedback(ui, 'setup_open', exc, kind='error')
            return
        if change['new'] == 'project':
            ui.project_root = Path(controls['project_root'].value).expanduser().resolve()
            ui.workspace_root = controls['workspace_root'].value.strip() or '.'
            ui.controls['project_name'].value = ''
            controls['setup_action'].value = _initial_action(ui)
        else:
            ui.project_root = None
            ui.controls['project_name'].layout.display = ''
            controls['setup_action'].value = _initial_action(ui)
        ui._release_remembered_tune(change)
        cell = ui.controls['cell_name']
        cell.options = ui._cell_options()
        if cell.value and cell.value not in cell.options:
            cell.value = cell.options[0]
        ui._on_cell_changed({'new': cell.value})
        refresh_summary(ui)
        visibility()
        if defaults.enabled:
            defaults.restore_scope()
    controls['workspace_mode'].observe(location_changed, names='value')
    controls['setup_discover'].on_click(run('setup_discover', lambda: discover(ui), 'Discovery complete; no model code executed or files written.'))
    controls['setup_open'].on_click(run('setup_open', lambda: apply_location(ui), 'Project opened; no files written or model loaded.'))
    controls['setup_preview'].on_click(run('setup_preview', lambda: preview(ui), 'Preview complete; no files written or model loaded.'))
    controls['setup_save'].on_click(run('setup_save', lambda: save(ui), 'Setup saved; no model loaded. Next: Prepare and load.'))
    controls['setup_project_save'].on_click(run('setup_project_save', lambda: save_navigation(ui), 'Project settings saved; no model constructed.'))
    controls['setup_copy_preview'].on_click(run('setup_copy_preview', lambda: working_copy(ui, dry_run=True), 'Copy preview complete; no files written.'))
    controls['setup_copy_create'].on_click(run('setup_copy_create', lambda: working_copy(ui, dry_run=False), 'Working copy created and selected; next: Prepare and load.'))
    controls['setup_candidates'].observe(
        lambda change: setattr(controls['setup_template'], 'value', change['new']) if change['new'] else None,
        names='value')
    controls['setup_location_box'] = w.VBox([
        controls['project_root'],
        action_box('setup_open', w.HBox([controls['setup_open'], controls['setup_discard_edits']])),
        w.HTML('<small>Optional for reopening saved work. Reload <b>before</b> editing; '
               'it is not a confirmation step. New project? Continue below.</small>'),
    ])
    controls['setup_project_details_box'] = w.VBox([
        controls['setup_location_box'], controls['project_label'], controls['workspace_root'],
    ])
    controls['setup_name_entry_row'] = w.HBox(
        [ui.controls[key].editor for key in ('cell_name', 'tune_name')],
        layout=w.Layout(width='100%', flex_flow='row wrap', align_items='flex-start'))
    controls['setup_workspace_box'] = w.VBox([
        w.HTML('<b>Register a cell / model</b>'), controls['setup_action'], controls['setup_action_help'],
        controls['setup_name_entry_row'], ui.controls['tune_dir_override'], registration,
    ])
    controls['setup_review_box'] = w.VBox([
        w.HTML('<p>Preview writes nothing. Save cell setup records setup '
               'without building the cell. Prepare and load below enters NEURON state. '
               'Each action keeps its own feedback; rerunning it replaces only its own output. '
               'After editing selections, rerun Preview to review the updated setup.</p>'),
        action_box('setup_preview'), action_box('setup_save'),
    ])
    copy_box.children += (action_box('setup_copy_preview'), action_box('setup_copy_create'))
    manager = w.VBox([controls['setup_workspace_box'], optional_box, controls['setup_review_box'], copy_box],
                     layout=w.Layout(display='' if controls['setup_manage_toggle'].value else 'none'))
    controls['setup_manager_box'] = manager
    controls['setup_manage_toggle'].observe(lambda change: setattr(manager.layout, 'display', '' if change['new'] else 'none'), names='value')
    # Render stock dropdowns directly with stable widths, avoiding shrinkable
    # selector VBoxes with percentage-width children in the visible row.
    # New-name text inputs are rendered beside registration, not daily navigation.
    project = ui.controls['project_name']
    selectors = [ui.controls[key] for key in ('cell_name', 'tune_name')]
    project.layout.flex = '0 0 auto'
    for selector in selectors:
        for field in (selector.dropdown, selector.editor):
            field.layout.width = '280px'
            field.layout.flex = '0 0 auto'
    controls['setup_selection_row'] = w.HBox(
        [project, *(selector.dropdown for selector in selectors)],
        layout=w.Layout(width='100%', flex_flow='row wrap', align_items='flex-start'))
    daily = w.VBox([
        controls['setup_location_summary'],
        controls['workspace_mode'], controls['setup_project_details_box'],
        controls['setup_selection_row'],
        controls['output_dir'], action_box('setup_project_save'), controls['setup_model_summary'],
    ])
    controls['setup_remember_box'] = controls['setup_project_save_box']
    controls['setup_daily_box'] = daily
    ui.controls.update(controls)
    visibility()
    if ui.project_root is not None:
        populate_existing(ui)
    _remember_form(ui)
    # A discard confirmation applies only to the form the user confirmed.
    for key in (*_FORM_FIELDS, 'project_root', 'workspace_mode'):
        ui.controls[key].observe(
            lambda _change: setattr(controls['setup_discard_edits'], 'value', False), names='value')
    for key in ('cell_name', 'tune_name', 'tune_dir_override', 'project_name', 'setup_action', 'output_dir'):
        ui.controls[key].observe(lambda _change: refresh_summary(ui), names='value')
    for key in ('cell_name', 'tune_name', 'tune_dir_override'):
        ui.controls[key].observe(visibility, names='value')
    refresh_summary(ui)
    return w.VBox([controls['setup_help'], daily, controls['setup_manage_toggle'], manager])


def refresh_summary(ui):
    if 'setup_location_summary' not in ui.controls:
        return
    if 'setup_optional_box' in ui.controls:
        ui._setup_optional_visibility()
    from ..notebook_settings import settings_root
    root = settings_root(ui.repo_root, ui.project_root, ui.settings.get('project_name'))
    ui.controls['setup_location_summary'].value = f'<b>Workspace location:</b> {escape(str(root))}'
    try:
        tune = ui._selected_tune_dir()
        config_path = tune / 'cell_configs/cell_config.json'
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        live = ui.cell_override is not None and ui.controls['setup_action'].value == 'live'
        if live:
            config = {'cell_loader': 'runtime_cell'}
        paths = config.get('paths') or {}
        loader = config.get('cell_loader', 'not registered')
        from scp_pipeline.loaders.source_context import effective_source_root
        native_root = effective_source_root(config, tune) or tune
        role = 'manifest' if loader == 'allen_manifest' else loader
        entry = paths.get(role)
        options = config.get(loader) or {}
        source = str(native_root / entry) if entry else ('supplied live object' if ui.cell_override is not None else 'not selected')
        sim_path = tune / 'cell_configs/sim_config.json'
        conditions = (json.loads(sim_path.read_text()).get('conditions') or {}) if sim_path.is_file() else {}
        if live:
            conditions = dict(conditions, **((ui.runtime_sim_config or {}).get('conditions') or {}))
        output = ui.controls['output_dir'].value.strip()
        resolved_output = (str(resolve_project_path(ui.project_root or ui.repo_root, output))
                           if output else str(tune / 'output_data'))
        details = {'Tune workspace': str(tune), 'Model': f'{loader}: {source}',
                   'Template / callable': options.get('template_name') or options.get('callable') or '—',
                   'Sources': (config.get('model_source') or {}).get('mode', 'tune-local'),
                   'Conditions': conditions or 'not configured / loader fallback',
                   'Results': resolved_output}
        ui.controls['setup_model_summary'].value = '<small>' + '<br>'.join(
            f'<b>{escape(key)}:</b> {escape(str(value))}' for key, value in details.items()) + '</small>'
    except (ValueError, OSError, TypeError) as exc:
        ui.controls['setup_model_summary'].value = f'<small>Select or register a cell. {escape(str(exc))}</small>'


def save_navigation(ui, *, dry_run=False):
    """Project metadata can be saved before a cell/model exists."""
    adopt_locations(ui)
    from ..notebook_settings import settings_root
    root = settings_root(ui.repo_root, ui.project_root, ui.settings.get('project_name'))
    has_selection = bool(ui.controls['tune_dir_override'].value.strip() or
                         (ui.controls['cell_name'].value.strip() and ui.controls['tune_name'].value.strip()))
    selected = ui._selected_tune_dir() if has_selection else None
    navigation = {'selected_project': None} if ui.project_root is None and root == ui.repo_root else {}
    if selected is not None:
        path = selected / 'cell_configs/cell_config.json'
        if path.is_file():
            config = json.loads(path.read_text())
            if config.get('model_source'):
                from scp_pipeline.loaders.source_context import guard_source_destinations
                guard_source_destinations(config, selected, [root / PROJECT_FILE, ui.repo_root / PROJECT_FILE]
                                          if ui.project_root is None else [root / PROJECT_FILE])
    if ui.project_root is None and root != ui.repo_root:
        save_project_settings(ui.repo_root, dry_run=True, selected_project=ui.settings.get('project_name'))
    result = save_project_settings(root, workspace_root=ui.workspace_root or '.', selected_tune=selected,
                                   project_name=ui.controls['project_label'].value.strip(),
                                   output_dir=ui.settings.get('output_dir'), dry_run=dry_run, **navigation)
    if dry_run:
        return result
    if ui.project_root is None and root != ui.repo_root:
        # Explicitly remembered SCP navigation also records which namespace to
        # reopen when the optional notebook selection defaults are all None.
        save_project_settings(ui.repo_root, selected_project=ui.settings.get('project_name'))
    _remember_form(ui)
    refresh_summary(ui)
    return result


def apply_location(ui):
    if ui.controls['workspace_mode'].value != 'project':
        raise ValueError('Choose Project-owned workspace to open a project directory.')
    new_root = Path(ui.controls['project_root'].value).expanduser().resolve()
    changed = new_root != ui.project_root
    if changed:
        guard_draft(ui)
        ui._notebook_defaults.guard_replace()
    result = open_settings(ui)
    if result.get('status') == 'not_found':
        adopt_locations(ui)
        ui.controls['setup_action'].value = _initial_action(ui)
        _remember_form(ui)
    if ui._notebook_defaults.enabled and changed:
        ui._notebook_defaults.restore_scope()
    refresh_summary(ui)
    return result


def adopt_locations(ui):
    controls = ui.controls
    if controls['workspace_mode'].value == 'project':
        if not controls['project_root'].value.strip():
            raise ValueError('Choose a project directory first.')
        ui.project_root = Path(controls['project_root'].value).expanduser().resolve()
        ui.workspace_root = controls['workspace_root'].value.strip() or '.'
    else:
        ui.project_root = None
    output = controls['output_dir'].value.strip()
    ui.settings['output_dir'] = str(resolve_project_path(ui.project_root or ui.repo_root, output)) if output else None
    ui.settings['tune_dir_override'] = controls['tune_dir_override'].value.strip() or None
    ui.settings['cell_name'] = controls['cell_name'].value.strip()
    ui.settings['tune_name'] = controls['tune_name'].value.strip()


def open_settings(ui):
    """Explicitly reload navigation; never reset a draft to synthesized defaults."""
    if ui.controls['workspace_mode'].value != 'project':
        raise ValueError('Select Project-owned workspace first.')
    directory = ui.controls['project_root'].value.strip()
    if not directory:
        raise ValueError('Choose a project directory first.')
    root = Path(directory).expanduser().resolve()
    path = root / PROJECT_FILE
    if path.is_symlink():
        raise ValueError(f'Refusing symlinked project settings: {path}')
    if not path.exists():
        return {'status': 'not_found', 'settings_file': str(path),
                'message': 'No saved project settings found. Your entries are unchanged; '
                           'use Register new model for first-time setup, or select an existing workspace.',
                'files_written': False}
    saved = load_project_settings(root)
    guard_draft(ui)
    if (_form_values(ui) != ui._project_settings_baseline
            and not ui.controls['setup_discard_edits'].value):
        ui.controls['setup_discard_edits'].layout.display = ''
        raise ValueError('Unsaved edits have not been replaced. To reload saved selections, '
                         'check Discard unsaved edits beside Open / reload project and click it again. '
                         'Otherwise keep editing and use Save project settings or Save cell setup when ready.')
    adopt_locations(ui)
    ui._remembered_tune_override = None
    for key, value in ui._project_recipe_defaults.items():
        ui.controls[key].value = value
    ui.controls['project_label'].value = saved['project_name']
    ui.controls['workspace_root'].value = saved.get('workspace_root') or '.'
    ui.workspace_root = ui.controls['workspace_root'].value
    ui.controls['tune_dir_override'].value = saved.get('selected_tune') or ''
    ui.controls['output_dir'].value = saved.get('output_dir') or ''
    from scp_pipeline.workspaces import discover_cells, discover_tunes
    cells = discover_cells(navigation_root(ui))
    ui.controls['cell_name'].options = cells
    if saved.get('selected_tune'):
        from scp_pipeline.workspaces import infer_workspace
        workspace = infer_workspace(resolve_project_path(ui.project_root, saved['selected_tune']))
        ui.controls['cell_name'].value = workspace.cell_name
        ui.controls['tune_name'].options = discover_tunes(navigation_root(ui), workspace.cell_name)
        ui.controls['tune_name'].value = workspace.tune_name
    else:
        ui.controls['cell_name'].value = ''
        ui.controls['tune_name'].value = 'orig'
    ui._remembered_tune_override = ui.controls['tune_dir_override'].value or None
    ui.controls['project_name'].layout.display = 'none'
    ui.controls['setup_action'].value = _initial_action(ui)
    populate_existing(ui)
    _remember_form(ui)
    return {'status': 'reloaded', 'settings_file': str(path), 'settings': saved,
            'message': 'Saved navigation reloaded; no files written or model loaded.'}


def populate_existing(ui):
    """Display the stored recipe; do not validate by executing it or modify it."""
    try:
        tune = ui._selected_tune_dir()
    except ValueError:
        return
    path = tune / 'cell_configs/cell_config.json'
    if not path.is_file():
        return
    config = json.loads(path.read_text())
    from scp_pipeline.loaders import get_cell_loader_name
    from scp_pipeline.loaders.source_context import effective_source_root, original_source_root
    loader = get_cell_loader_name(config)
    if loader not in {'hoc_template', 'python_factory', 'allen_manifest'}:
        return
    c = ui.controls
    c['setup_loader'].value = loader
    paths = config.get('paths') or {}
    role = 'manifest' if loader == 'allen_manifest' else loader
    model_root = effective_source_root(config, tune) or tune
    c['setup_entry'].value = str(model_root / paths.get(role, ''))
    options = config.get(loader) or {}
    c['setup_template'].value = options.get('template_name', '')
    c['setup_callable'].value = options.get('callable', '')
    c['setup_args'].value = json.dumps(options.get('constructor_args', options.get('args', [])))
    c['setup_kwargs'].value = json.dumps(options.get('kwargs', {}))
    c['setup_sections'].value = json.dumps(options.get('section_map', {}))
    source = config.get('model_source')
    if source:
        c['setup_source_root'].value = str(original_source_root(config, tune))
        c['setup_source_mode'].value = source.get('mode', 'copy')
        c['setup_folders'].value = json.dumps(source.get('folders', []))
        c['setup_files'].value = json.dumps(source.get('files', []))
    c['setup_modfiles'].value = paths.get('modfiles') or ''
    sim_path = tune / 'cell_configs/sim_config.json'
    conditions = (json.loads(sim_path.read_text()).get('conditions') or {}) if sim_path.is_file() else {}
    for key, field in [('v_init_mV', 'setup_voltage'), ('celsius_C', 'setup_celsius')]:
        c[field].value = '' if conditions.get(key) is None else str(conditions[key])


def discover(ui):
    adopt_locations(ui)
    from scp_pipeline.setup import discover_hoc_templates
    raw = ui.controls['setup_entry'].value.strip()
    if not raw:
        raise ValueError('Enter the HOC file path in Entry file before clicking Discover HOC templates. '
                         'This button reads that file; it does not open a file browser.')
    entry = resolve_project_path(ui.project_root or ui.repo_root, raw)
    if entry.is_dir():
        raise ValueError(f'Entry file must name a HOC file, not a directory: {entry}')
    if not entry.is_file():
        raise FileNotFoundError(f'HOC Entry file not found: {entry}. Correct the path and click Discover HOC templates again.')
    candidates = [record['template_name'] for record in discover_hoc_templates(entry.parent)
                  if record['path'] == entry.name]
    ui.controls['setup_candidates'].options = [('Select a template', ''), *((name, name) for name in candidates)]
    ui.controls['setup_candidates'].value = candidates[0] if len(candidates) == 1 else ''
    return {'file': str(entry), 'templates': candidates, 'executed_model_code': False}


def _setup_conditions(ui):
    conditions = {}
    for key, field, label in (('v_init_mV', 'setup_voltage', 'V init (mV)'),
                              ('celsius_C', 'setup_celsius', 'Temp (°C)')):
        raw = ui.controls[field].value.strip()
        if raw:
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError(f'{label} must be a finite number.') from exc
            if not math.isfinite(value):
                raise ValueError(f'{label} must be a finite number.')
            conditions[key] = value
    return conditions


def _allen_registration_kwargs(ui):
    try:
        specimen = int(ui.controls['setup_specimen'].value.strip())
        if specimen <= 0:
            raise ValueError
    except ValueError as exc:
        raise ValueError('Enter a positive integer Allen Specimen ID before previewing or saving.') from exc
    conditions = _setup_conditions(ui)
    return {'tune_dir': _selected_setup_tune(ui), 'specimen_id': specimen,
            'model_type': ui.controls['setup_model_type'].value,
            'v_init_mV': conditions.get('v_init_mV'), 'celsius_C': conditions.get('celsius_C')}


def _allen_config(ui, settings):
    """Prospective recipe only: manifest inspection waits until acquisition."""
    from scp_pipeline.setup.defaults import default_cell_config
    from scp_pipeline.workspaces import infer_workspace
    tune = settings['tune_dir']
    workspace = infer_workspace(tune, cell_config={})
    config = default_cell_config(cell_name=workspace.cell_name, tune_name=workspace.tune_name,
        specimen_id=settings['specimen_id'], model_type=settings['model_type'], color='k',
        project_name=workspace.project_name)
    if ui.project_root is not None:
        config.update(workspace_layout='project_owned', project_name=ui.controls['project_label'].value.strip(),
                      workspace_project_root=os.path.relpath(ui.project_root, tune))
    return config


def registration_kwargs(ui):
    adopt_locations(ui)
    c = ui.controls
    def parsed(key, expected):
        value = json.loads(c[key].value)
        if not isinstance(value, expected):
            raise ValueError(f'{key} must be JSON {expected.__name__}.')
        return value
    loader = c['setup_loader'].value
    if loader == 'adb':
        raise ValueError('Allen download: Preview reports acquisition only; Save downloads and scaffolds without loading a cell.')
    tune = _selected_setup_tune(ui)
    if not c['setup_entry'].value.strip():
        raise ValueError('Choose an Entry file for the model before Preview / validate or Save setup.')
    path_root = ui.project_root or ui.repo_root
    entry = resolve_project_path(path_root, c['setup_entry'].value)
    source_root = resolve_project_path(path_root, c['setup_source_root'].value) if c['setup_source_root'].value.strip() else entry.parent
    try:
        entry_relative = entry.relative_to(source_root).as_posix()
    except ValueError as exc:
        raise ValueError(f'Entry file {entry} must be inside Source root {source_root}. '
                         'Choose their common native-source root and declare the needed dependencies.') from exc
    paths = {{'hoc_template': 'hoc_template', 'python_factory': 'python_factory', 'allen_manifest': 'manifest'}[loader]: entry_relative,
             'modfiles': None}
    folders = parsed('setup_folders', list)
    files = parsed('setup_files', list)
    # Whole Allen bundles and Python packages require explicit dependency selection.
    files = list(dict.fromkeys([entry_relative, *files]))
    if c['setup_modfiles'].value.strip():
        mod = resolve_project_path(source_root, c['setup_modfiles'].value)
        paths['modfiles'] = mod.relative_to(source_root).as_posix()
        folders = list(dict.fromkeys([*folders, paths['modfiles']]))
    config = {'cell_loader': loader, 'paths': paths,
              'model_source': {'mode': c['setup_source_mode'].value, 'source_root': str(source_root),
                               'folders': folders, 'files': files}}
    if c['setup_working_dir'].value.strip():
        config['model_source']['working_directory'] = c['setup_working_dir'].value.strip()
    section_map = parsed('setup_sections', dict)
    if loader == 'hoc_template':
        config['hoc_template'] = {'template_name': c['setup_template'].value.strip(),
                                  'constructor_args': parsed('setup_args', list)}
        if section_map:
            config['hoc_template']['section_map'] = section_map
    elif loader == 'python_factory':
        config['python_factory'] = {'callable': c['setup_callable'].value.strip(),
                                    'args': parsed('setup_args', list), 'kwargs': parsed('setup_kwargs', dict)}
        if section_map:
            config['python_factory']['section_map'] = section_map
    conditions = _setup_conditions(ui)
    return {'project_root': ui.project_root, 'repo_root': ui.repo_root, 'tune_dir': tune,
            'cell_name': c['cell_name'].value.strip(), 'tune_name': c['tune_name'].value.strip(),
            'project_name': c['project_label'].value.strip() if ui.project_root else ui._selected_project(), 'cell_config': config,
            'conditions': conditions, 'output_dir': ui.settings.get('output_dir'), **generation_options(ui)}


def _selected_setup_tune(ui):
    controls = ui.controls
    if (not controls['tune_dir_override'].value.strip()
            and (not controls['cell_name'].value.strip() or not controls['tune_name'].value.strip())):
        if controls['setup_action'].value == 'existing':
            raise ValueError('No existing workspace selected. Choose a Cell/Tune or Tune path; '
                             'for a first-time setup choose Action → Register new model.')
        raise ValueError('Enter a Cell and Tune name (or an explicit Tune path) for the new workspace.')
    return ui._selected_tune_dir()


def preview(ui):
    adopt_locations(ui)
    save_navigation(ui, dry_run=True)
    action = ui.controls['setup_action'].value
    if action == 'register':
        if ui.controls['setup_loader'].value == 'adb':
            settings = _allen_registration_kwargs(ui)
            from scp_pipeline.setup import preview_setup_outputs
            optional = preview_setup_outputs(settings['tune_dir'], cell_config=_allen_config(ui, settings),
                include_required=True, conditions=_setup_conditions(ui), **generation_options(ui))
            return {'action': 'Allen acquisition on Save', **settings,
                    'tune_dir': str(settings['tune_dir']), 'constructs_cell': False, 'optional_outputs': optional}
        return preview_workspace_registration(**registration_kwargs(ui))
    tune = _selected_setup_tune(ui)
    if action == 'live':
        if ui.cell_override is None:
            raise ValueError('Create CELL_OVERRIDE in this kernel before selecting a live cell.')
        from scp_pipeline.model import validate_cell, wrap_cell
        from scp_pipeline.model.runtime_cell import prepare_runtime_workspace
        cell = validate_cell(wrap_cell(ui.cell_override))
        prepare_runtime_workspace(tune, cell, ui.runtime_sim_config, dry_run=True)
        if ui.project_root is not None:
            from scp_pipeline.workspaces import _name
            _name(ui.controls['project_label'].value.strip(), 'project_name')
        from scp_pipeline.setup import preview_setup_outputs
        path = tune / 'cell_configs/cell_config.json'
        config = json.loads(path.read_text()) if path.is_file() else {'cell_loader': 'runtime_cell', 'cell_name': ui.controls['cell_name'].value}
        optional = preview_setup_outputs(tune, cell_config=config, include_required=True,
                                         runtime_sim_config=ui.runtime_sim_config, **generation_options(ui))
        return {'tune_dir': str(tune), 'construction': 'supplied live object', 'reconstructible': False,
                'optional_outputs': optional}
    from scp_pipeline.tuning.notebook_setup import validate_step1_outputs
    from scp_pipeline.setup import preview_setup_outputs
    config = json.loads((tune / 'cell_configs/cell_config.json').read_text())
    optional = preview_setup_outputs(tune, cell_config=config, include_required=True,
                                     conditions=_setup_conditions(ui), **generation_options(ui))
    report = validate_step1_outputs(tune, require_compiled_modfiles=False,
                                   require_sim_config=False, require_geometry_config=False)
    from scp_pipeline.simulation.current_injection import validate_required_sim_conditions
    # Missing runtime/geometry configs were validated prospectively above.
    return {'tune_dir': str(tune), 'valid': report.ok, 'compilation': 'checked/loaded on Prepare',
            'existing_configs': 'read-only', 'optional_outputs': optional,
            'output_dir': ui.settings.get('output_dir') or str(tune / 'output_data')}


def save(ui):
    adopt_locations(ui)
    save_navigation(ui, dry_run=True)
    if ui.controls['setup_action'].value == 'register':
        if ui.controls['setup_loader'].value == 'adb':
            result = save_allen_download(ui)
        else:
            result = register_workspace(dry_run=False, workspace_root=ui.workspace_root if ui.project_root else None,
                                        **registration_kwargs(ui))
            _select_registered_tune(ui, result['tune_dir'])
        reset_draft(ui)
        _remember_form(ui)
        refresh_summary(ui)
        return result
    preview(ui)
    from scp_pipeline.setup import create_setup_outputs
    tune = _selected_setup_tune(ui)
    path = tune / 'cell_configs/cell_config.json'
    if ui.controls['setup_action'].value == 'live':
        from scp_pipeline.model.runtime_cell import prepare_runtime_workspace, adopt_cell
        had_recipe = path.is_file()
        prepare_runtime_workspace(tune, adopt_cell(ui.cell_override), ui.runtime_sim_config, include_target_config=False)
        if ui.project_root is not None and not had_recipe:
            config = json.loads(path.read_text())
            config.update(project_name=ui.controls['project_label'].value.strip(), workspace_layout='project_owned',
                          workspace_project_root=os.path.relpath(ui.project_root, tune))
            path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    config = json.loads(path.read_text()) if path.is_file() else {}
    optional = create_setup_outputs(tune, cell_config=config, include_required=True,
                                    conditions=_setup_conditions(ui),
                                    runtime_sim_config=ui.runtime_sim_config if ui.controls['setup_action'].value == 'live' else None,
                                    **generation_options(ui))
    result = save_navigation(ui)
    result['optional_outputs'] = optional
    reset_draft(ui)
    _remember_form(ui)
    return result


def _select_registered_tune(ui, tune_dir):
    """Refresh the existing-name selectors after saving, without constructing."""
    from scp_pipeline.workspaces import infer_workspace
    workspace = infer_workspace(tune_dir)
    ui._syncing_model_selection = True
    try:
        if ui.project_root is None:
            from ..notebook_settings import settings_root
            project = ui.controls['project_name']
            name = workspace.project_name or ''
            if name not in [value for _label, value in project.options]:
                project.options = [*project.options, (name, name)]
            project.value = name
            ui.settings['project_name'] = workspace.project_name
            owner = settings_root(ui.repo_root, project_name=workspace.project_name)
            ui.controls['project_root'].value = str(owner)
            ui.controls['project_label'].value = load_project_settings(owner)['project_name']
            ui.controls['workspace_root'].value = '.'
            ui.workspace_root = '.'
        cell, tune = ui.controls['cell_name'], ui.controls['tune_name']
        cell.options = ui._cell_options(workspace.cell_name)
        cell.value = workspace.cell_name
        tune.options = ui._tune_options(cell.value, workspace.tune_name)
        tune.value = workspace.tune_name
        ui.controls['tune_dir_override'].value = str(workspace.tune_dir)
        ui._remembered_tune_override = str(workspace.tune_dir)
        ui.settings.update(cell_name=workspace.cell_name, tune_name=workspace.tune_name,
                           tune_dir_override=str(workspace.tune_dir))
    finally:
        ui._syncing_model_selection = False
    ui.controls['setup_action'].value = 'existing'


def save_allen_download(ui):
    from scp_pipeline.setup import prepare_tune
    settings = _allen_registration_kwargs(ui)
    save_navigation(ui, dry_run=True)
    tune = settings['tune_dir']
    from scp_pipeline.setup import preview_setup_outputs, create_setup_outputs
    optional = generation_options(ui)
    config = _allen_config(ui, settings)
    preview_setup_outputs(tune, cell_config=config, include_required=True,
                          conditions=_setup_conditions(ui), **optional)
    if (tune / 'cell_configs/cell_config.json').exists() or (tune / 'cell_config.json').exists():
        raise ValueError('A recipe already exists; open it instead of downloading over it.')
    from scp_pipeline.workspaces import infer_workspace
    workspace = infer_workspace(tune, cell_config={})
    result = prepare_tune(**settings, cell_name=workspace.cell_name,
                          tune_name=workspace.tune_name, source_type='adb',
                          do_compile_modfiles=False, load_compiled_dll=False, do_validate=False,
                          do_base_configs=False, do_target_config=False, do_synapse_configs=False)
    path = tune / 'cell_configs/cell_config.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(config, indent=2, allow_nan=False) + '\n')
    result['optional_outputs'] = create_setup_outputs(tune, cell_config=config, include_required=True,
        conditions=_setup_conditions(ui), **optional)
    _select_registered_tune(ui, tune)
    save_navigation(ui)
    return result


def working_copy(ui, *, dry_run=True):
    if ui.controls['setup_action'].value != 'existing':
        raise ValueError('Select a saved existing tune before creating a working copy.')
    from scp_pipeline.setup import create_working_copy
    save_navigation(ui, dry_run=True)
    if not dry_run:
        guard_draft(ui)
    result = create_working_copy(source_tune_dir=_selected_setup_tune(ui),
        target_tune_name=ui.controls['setup_copy_name'].value.strip(),
        content_policy='model', dry_run=dry_run, source_mode=ui.controls['setup_copy_mode'].value)
    if not dry_run:
        try:
            _select_registered_tune(ui, result['target'])
            save_navigation(ui)
            refresh_summary(ui)
        except Exception as exc:
            raise RuntimeError(
                f"Working copy was created at {result['target']}, but updating "
                f"the notebook selection/settings failed: {exc}. Do not create "
                "it again; select this existing tune using Cell/Tune or Tune path."
            ) from exc
    return result
