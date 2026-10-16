"""Optional remembered defaults and their compact notebook controls."""
from __future__ import annotations

import base64
from copy import deepcopy
from html import escape
import json

from ..notebook_settings import (
    SETTING_KEYS, SETTINGS_FILE, parse_profile, preferences, profile_document,
    read_profile, settings_path, settings_root, validate_settings, write_profile,
)
from .common import parse_float_list, parse_optional_float, parse_optional_int


class NotebookDefaultsUI:
    def __init__(self, ui, *, enabled=False, path=None):
        self.ui = ui
        self.enabled = enabled
        self.path_override = path
        self.initial = preferences(ui.settings)
        self.saved = deepcopy(self.initial)
        self.edited = set()
        self.observed = set()
        self.suspended = False
        self.dirty = False
        self.error = None
        self.active_path = self.path()
        self.loaded = False

    def path(self, path=None):
        u = self.ui
        root = settings_root(u.repo_root, u.project_root, u.settings.get('project_name'))
        raw = path if path is not None else self.path_override
        return settings_path(root, raw)

    def initialize(self):
        if self.enabled:
            self.restore_scope()

    def restore_scope(self):
        self.active_path = self.path()
        values = deepcopy(self.initial)
        self.loaded = False
        self.error = None
        try:
            if self.active_path.exists() or self.active_path.is_symlink():
                values.update(read_profile(self.active_path))
                self.loaded = True
        except ValueError as exc:
            self.error = str(exc)
        try:
            self._apply(values)
        except (ValueError, TypeError) as exc:
            self.error = f'Cannot restore notebook defaults: {exc}'
            self.loaded = False
        self.saved = preferences(self.ui.settings)
        self.dirty = False
        self.edited.clear()
        self.refresh()

    def _apply(self, values):
        values = validate_settings(values)
        u = self.ui
        candidate = {key: deepcopy(value) for key, value in u.settings.items() if key not in SETTING_KEYS}
        candidate.update(deepcopy(self.initial))
        candidate.update(values)
        # A profile never changes construction options after NEURON has loaded.
        if u._setup_attempted:
            for key in ('quiet_step1_output', 'recompile_modfiles'):
                candidate[key] = u.settings[key]
        before = deepcopy(u.settings)
        self.suspended = True
        try:
            u.apply_settings(candidate)
        except Exception:
            u.apply_settings(before)
            raise
        finally:
            self.suspended = False

    def watch(self):
        for key, control in self.ui.controls.items():
            if (id(control) in self.observed or not hasattr(control, 'value')
                    or key.startswith(('setup_', 'notebook_'))
                    or key in {'workspace_mode', 'project_root', 'workspace_root', 'project_label',
                               'output_dir', 'project_name', 'cell_name', 'tune_name', 'tune_dir_override'}
                    or isinstance(control, self.ui.widgets.HTML)):
                continue
            self.observed.add(id(control))
            control.observe(lambda change, name=key: self.changed(name), names='value')
        for index, row in enumerate(self.ui._act_conductance_controls):
            for field, control in row.items():
                if id(control) not in self.observed:
                    self.observed.add(id(control))
                    control.observe(lambda change, name=f'act_conductance_{index}_{field}': self.changed(name), names='value')

    def changed(self, key=None):
        u = self.ui
        if self.suspended or any(getattr(u, name, False) for name in (
            '_syncing_passive_targets', '_syncing_active_current_amp',
            '_syncing_input_preview_options', '_syncing_simulation_options',
            '_syncing_diagnostic_options', '_act_syncing')):
            return
        # Expanding a panel is not a changed experiment preference.
        if key and (key.endswith(('_toggle', '_box')) or key == 'diagnostic_manual_window_box'):
            return
        if key:
            self.edited.add(key)
        self.dirty = True
        self.refresh()

    def collect(self):
        """Validate current inputs, not the last-valid values observers retained."""
        u = self.ui
        values = preferences(u.settings)
        for key in SETTING_KEYS:
            control = u.controls.get(key)
            if control is None or key not in self.edited or key.startswith(('passive_target_', 'act_')):
                continue
            raw = control.value
            original = values.get(key)
            if key.endswith('amps_pA'):
                raw = parse_float_list(raw, name=key)
            elif key in {'seed', 'n_trials', 'input_preview_trial_idx', 'diagnostic_trial_idx', 'input_preview_plot_columns'}:
                raw = parse_optional_int(raw, name=key, positive=key in {'n_trials', 'input_preview_plot_columns'})
            elif isinstance(original, bool):
                raw = bool(raw)
            elif isinstance(original, (int, float)) or key in {
                'active_current_display_amp_pA', 'input_preview_weight_bin', 'diagnostic_window_start_ms',
                'diagnostic_window_stop_ms', 'diagnostic_rate_bin_ms', 'diagnostic_smoothing_ms'}:
                raw = parse_optional_float(raw, name=key)
            elif isinstance(raw, tuple):
                raw = list(raw)
            elif raw == '' and original is None:
                raw = None
            values[key] = raw
        # Existing validators omit default timing values and preserve inheritance.
        before = deepcopy(u.settings)
        try:
            if 'passive_stim_delay' in u.controls:
                values['passive_protocol_overrides'] = u._passive_protocol_overrides_from_controls()
                targets = u._passive_targets_from_controls()
                values['passive_target_overrides'] = {
                    key: targets[key] for key in u._passive_target_dirty if key in targets}
            for scope in ('active', 'fi'):
                if f'{scope}_stim_delay' in u.controls:
                    values[f'{scope}_protocol_overrides'] = u._step3_protocol_overrides_from_controls(
                        scope=scope, setting_key=f'{scope}_protocol_overrides', label=scope.title())
            if 'sim_n_inputs_to_save' in u.controls:
                values['simulation_overrides'] = u._validated_simulation_overrides()
            for key in self.edited:
                control = u.controls.get(key)
                if control is None:
                    continue
                if key in {'act_target_currents', 'act_target_frequencies', 'act_filter_window'}:
                    if control.value.strip():
                        parse_float_list(control.value, name=key)
                elif key in {'act_opt_random_state', 'act_opt_n_estimators', 'act_opt_max_depth', 'act_opt_max_n_spikes'}:
                    parse_optional_int(control.value, name=key)
                elif (key.startswith(('act_sim_', 'act_opt_', 'act_nwb_')) and key not in {
                        'act_opt_train_features', 'act_nwb_stimuli', 'act_nwb_include_negative', 'act_nwb_average'}
                        or key == 'act_saturation_threshold'):
                    parse_optional_float(control.value, name=key)
            for row in u._act_conductance_controls:
                if not row['blocked'].value:
                    for key in ('low', 'high', 'bounds_variation'):
                        parse_optional_float(row[key].value, name=f'ACT conductance {key}')
            # Inspection may choose an ACT module for display; that alone isn't an override.
            if 'act_active_module' not in self.edited:
                values['act_active_module'] = self.saved.get('act_active_module')
        finally:
            u.settings.clear()
            u.settings.update(before)
        return validate_settings(values)

    def guard_replace(self, *, discard=False):
        if not self.dirty or discard:
            return
        choice = self.ui.controls.get('notebook_unsaved_action')
        action = choice.value if choice is not None else 'cancel'
        if action == 'save':
            self.save()
        elif action != 'discard':
            raise ValueError('Unsaved notebook defaults: choose Save or Discard in Notebook defaults '
                             '(Cancel keeps the current session), then retry.')
        if choice is not None:
            choice.value = 'cancel'

    def save(self, path=None):
        values = self.collect()
        destination = self.path(path)
        document = write_profile(destination, values)
        self.active_path = destination
        self.ui.settings.update(deepcopy(values))
        self.saved = deepcopy(values)
        self.dirty = False
        self.loaded = True
        self.error = None
        self.edited.clear()
        self.refresh()
        return {'path': str(destination), **document, 'model_configs_changed': False}

    def load(self, path=None, *, discard=False):
        values = read_profile(self.path(path))  # Validate before changing controls or dirty state.
        self.guard_replace(discard=discard)
        self._apply(values)
        self.active_path = self.path(path)
        self.saved = preferences(self.ui.settings)
        self.dirty = False
        self.loaded = True
        self.error = None
        self.edited.clear()
        self.refresh()
        return {'path': str(self.active_path), 'status': 'loaded', 'model_configs_changed': False,
                'locked_loading_options_preserved': self.ui._setup_attempted}

    def reset(self, *, discard=False):
        self.guard_replace(discard=discard)
        self._apply(self.initial)
        self.edited.clear()
        self.dirty = preferences(self.ui.settings) != self.saved
        self.refresh()
        return {'status': 'session reset', 'files_written': False}

    def refresh(self):
        control = self.ui.controls.get('notebook_defaults_status')
        if control is None:
            return
        state = 'Unsaved changes' if self.dirty else ('Restored / saved' if self.loaded else 'Using initial defaults (not saved)')
        control.value = f'<small><b>{state}</b> — {escape(str(self.active_path))}</small>'
        if self.error:
            control.value += f'<p role="alert">{escape(self.error)} No saved choices were applied.</p>'

    def build_panel(self):
        u, w = self.ui, self.ui.widgets
        from .text_output import TextOutput
        if 'notebook_defaults_box' in u.controls:
            return u.controls['notebook_defaults_box']
        c = {
            'notebook_defaults_status': w.HTML(),
            'notebook_save': w.Button(description='Save notebook defaults', icon='save', layout=w.Layout(width='220px')),
            'notebook_options_toggle': w.ToggleButton(description='Notebook defaults options', value=False, layout=w.Layout(width='230px')),
            'notebook_settings_path': w.Text(value=str(self.path_override or SETTINGS_FILE), description='Defaults file', layout=w.Layout(width='650px')),
            'notebook_reload': w.Button(description='Reload saved defaults'),
            'notebook_reset': w.Button(description='Reset session defaults'),
            'notebook_export': w.Button(description='Export defaults JSON'),
            'notebook_import': w.FileUpload(accept='.json', multiple=False, description='Import JSON'),
            'notebook_unsaved_action': w.Dropdown(description='Unsaved edits', options=[('Cancel / keep editing', 'cancel'), ('Save first', 'save'), ('Discard', 'discard')]),
        }
        u.controls.update(c)
        advanced = w.VBox([
            w.HTML('<small>One defaults file applies across cells in this project. No scientific configs are rewritten. '
                   'Reload/reset requires Save or Discard when edits are unsaved. File paths are project-relative.</small>'),
            c['notebook_settings_path'], c['notebook_unsaved_action'],
        ], layout=w.Layout(display='none'))
        c['notebook_options_toggle'].observe(
            lambda change: setattr(advanced.layout, 'display', '' if change['new'] else 'none'), names='value')
        c['notebook_settings_path'].observe(lambda change: setattr(self, 'path_override', change['new'].strip() or None), names='value')

        def action(key, callback):
            status, output = w.HTML(), TextOutput(layout=w.Layout(max_height='280px', overflow='auto'))
            u.statuses[key], u.outputs[key] = status, output
            def clicked(_button):
                output.clear_output(wait=True)
                output.outputs = ()
                try:
                    result = callback()
                    output.append_stdout(json.dumps(result, indent=2, default=str) + '\n')
                    u._set_status(key, 'complete', 'Complete. Model files/configs unchanged.')
                    c['notebook_unsaved_action'].value = 'cancel'
                except Exception as exc:
                    output.append_stdout(f'{type(exc).__name__}: {exc}\n')
                    u._set_status(key, 'error', str(exc))
                    advanced.layout.display = ''
                    c['notebook_options_toggle'].value = True
            if key != 'notebook_import':
                c[key].on_click(clicked)
            return w.VBox([c[key], status, output]), clicked

        save_box, _ = action('notebook_save', self.save)
        for key, callback in [('notebook_reload', self.load), ('notebook_reset', self.reset)]:
            box, _ = action(key, callback)
            advanced.children = (*advanced.children, box)
        export_output = w.HTML()
        def export():
            raw = json.dumps(profile_document(self.collect()), indent=2, allow_nan=False).encode()
            export_output.value = '<a download="scp_pipeline_settings.json" href="data:application/json;base64,' + base64.b64encode(raw).decode() + '">Download notebook defaults</a>'
            return {'status': 'download ready', 'files_written': False}
        box, _ = action('notebook_export', export)
        advanced.children = (*advanced.children, box, export_output)
        def imported():
            uploaded = c['notebook_import'].value
            file = next(iter(uploaded.values())) if isinstance(uploaded, dict) else uploaded[0]
            values = parse_profile(bytes(file['content']))
            self.guard_replace()
            self._apply(values)
            self.dirty = True
            self.refresh()
            return {'status': 'imported into session; Save notebook defaults to persist', 'files_written': False}
        box, callback = action('notebook_import', imported)
        def upload_changed(change):
            if change['new']:
                callback(None)
                c['notebook_import'].value = {} if isinstance(change['new'], dict) else ()
        c['notebook_import'].observe(upload_changed, names='value')
        advanced.children = (*advanced.children, box)
        panel = w.VBox([c['notebook_defaults_status'], w.HBox([save_box, c['notebook_options_toggle']]), advanced])
        u.controls['notebook_defaults_box'] = panel
        self.refresh()
        return panel
