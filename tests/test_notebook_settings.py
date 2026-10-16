"""Remembered compact-notebook defaults use temporary, synthetic workspaces."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from datetime import datetime, timezone

from scp_pipeline.notebooks import PipelineNotebookUI, PIPELINE_UI_DEFAULTS
from scp_pipeline.notebooks.notebook_settings import (
    SETTINGS_FILE, parse_profile, preferences, profile_document, read_profile,
    settings_path, settings_root, validate_settings, write_profile,
)


class NotebookSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.profile = self.root / SETTINGS_FILE

    def ui(self, *, render=True, **kwargs):
        ui = PipelineNotebookUI(project_root=self.root, **kwargs)
        if render:
            for panel in (ui.step1_panel, ui.step2_panel, ui.step3_panel, ui.step4_panel, ui.step5_panel):
                panel()
        return ui

    def click(self, ui, key):
        with redirect_stdout(io.StringIO()):
            ui.controls[key].click()

    def test_no_file_or_model_construction_on_render_and_missing_auto_restore(self):
        with mock.patch('scp_pipeline.model.load_cell') as load:
            ui = self.ui(restore_notebook_settings=True)
        load.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertFalse(ui._notebook_defaults.loaded)
        self.assertEqual(ui.controls['setup_manager_box'].layout.display, '')
        self.assertFalse(ui.controls['loading_options_toggle'].value)

    def test_empty_cell_directories_still_open_first_time_management(self):
        (self.root / 'cells/Cell/tunes/orig').mkdir(parents=True)
        ui = self.ui()
        self.assertEqual(ui.controls['setup_action'].value, 'register')
        self.assertTrue(ui.controls['setup_manage_toggle'].value)

    def test_initial_defaults_restore_and_two_notebook_copies(self):
        write_profile(self.profile, {'active_amps_pA': [200], 'seed': 17, 'quiet_step1_output': False})
        ui = self.ui(restore_notebook_settings=True, settings={'active_amps_pA': [1]})
        self.assertEqual(ui.settings['active_amps_pA'], [200])
        self.assertEqual(ui.controls['active_amps_pA'].value, '200')
        other = self.ui(restore_notebook_settings=True)
        self.assertEqual(other.settings['seed'], 17)
        legacy = self.ui(restore_notebook_settings=False)
        self.assertEqual(legacy.settings['active_amps_pA'], PIPELINE_UI_DEFAULTS['active_amps_pA'])

    def test_save_current_widgets_without_running_and_no_navigation_or_recipe(self):
        ui = self.ui()
        ui.controls['active_amps_pA'].value = '25, 50'
        ui.controls['passive_stim_dur'].value = '500'
        ui.controls['seed'].value = '42'
        ui.controls['recompile_modfiles'].value = True
        self.assertTrue(ui._notebook_defaults.dirty)
        ui.save_notebook_settings()
        saved = read_profile(self.profile)
        self.assertEqual(saved['active_amps_pA'], [25, 50])
        self.assertEqual(saved['passive_protocol_overrides']['stim_dur'], 500)
        self.assertEqual(saved['seed'], 42)
        self.assertTrue(saved['recompile_modfiles'])
        self.assertEqual(saved['simulation_overrides'], {})
        self.assertEqual(saved['passive_target_overrides'], {})
        self.assertFalse({'cell_name', 'tune_name', 'project_name', 'cell_loader'} & saved.keys())
        self.assertEqual(list(self.root.iterdir()), [self.profile])
        self.assertFalse(ui._notebook_defaults.dirty)

    def test_invalid_widgets_never_save_last_valid_values(self):
        for key in ('passive_amps_pA', 'active_amps_pA', 'seed', 'passive_stim_dur',
                    'active_stim_dur', 'fi_stim_dur', 'sim_celsius_C', 'active_spike_threshold_mV'):
            with self.subTest(key=key):
                ui = self.ui()
                write_profile(self.profile, {'seed': 12})
                before = self.profile.read_bytes()
                ui.controls[key].value = 'invalid'
                with self.assertRaises(ValueError):
                    ui.save_notebook_settings()
                self.assertEqual(self.profile.read_bytes(), before)

    def test_reload_discard_reset_and_no_implicit_writes(self):
        write_profile(self.profile, {'seed': 17})
        ui = self.ui(restore_notebook_settings=True)
        ui.controls['seed'].value = '21'
        before = self.profile.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Unsaved'):
            ui.load_notebook_settings()
        self.assertEqual(ui.controls['seed'].value, '21')
        ui.load_notebook_settings(discard=True)
        self.assertEqual(ui.controls['seed'].value, '17')
        ui._notebook_defaults.reset()
        self.assertIsNone(ui.settings['seed'])
        self.assertEqual(self.profile.read_bytes(), before)

    def test_malformed_profile_rejected_without_partial_application(self):
        ui = self.ui()
        for document in (
            {'format_version': 2, 'kind': 'scp_notebook_settings', 'settings': {}},
            profile_document({'seed': 1}) | {'extra': True},
            {'format_version': 1, 'kind': 'scp_notebook_settings', 'settings': {'seed': 'bad'}},
            {'format_version': 1, 'kind': 'scp_notebook_settings', 'settings': {'active_amps_pA': [1], 'diagnostic_raster_style': 'bad'}},
        ):
            self.profile.write_text(json.dumps(document))
            before = deepcopy(ui.settings)
            with self.assertRaises(ValueError):
                ui.load_notebook_settings(discard=True)
            self.assertEqual(ui.settings, before)
        ui = self.ui(restore_notebook_settings=True)
        self.assertIsNotNone(ui._notebook_defaults.error)
        self.assertIn('No saved choices', ui.controls['notebook_defaults_status'].value)

    def test_refuse_unrelated_json_runtime_values_and_symlinks(self):
        self.profile.write_text('{"conditions":{"celsius_C":34}}')
        before = self.profile.read_bytes()
        with self.assertRaises(ValueError):
            write_profile(self.profile, {'seed': 1})
        self.assertEqual(self.profile.read_bytes(), before)
        for values in ({'cell_name': 'A'}, {'simulation_overrides': {'cell': object()}},
                       {'active_amps_pA': [float('nan')]}, {'n_trials': True}):
            with self.assertRaises(ValueError):
                validate_settings(values)
        link = self.root / 'linked.json'
        link.symlink_to(self.profile)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            write_profile(settings_path(self.root, link), {})

    def test_scope_and_explicit_relative_path(self):
        ui = self.ui(notebook_settings_path='preferences/notebook.json')
        ui.save_notebook_settings()
        self.assertTrue((self.root / 'preferences/notebook.json').is_file())
        self.assertFalse(self.profile.exists())
        self.assertEqual(settings_root(self.root, project_name='alpha'), self.root / 'projects/alpha')
        self.assertEqual(settings_root(self.root), self.root)

    def test_cell_switch_retains_notebook_choices(self):
        ui = self.ui()
        ui.controls['active_amps_pA'].value = '200'
        ui.controls['cell_name'].value = 'Another'
        ui.controls['tune_name'].value = 'working'
        self.assertEqual(ui.settings['active_amps_pA'], [200])

    def test_project_settings_saved_before_any_cell(self):
        ui = self.ui()
        ui.controls['project_label'].value = 'Example_project'
        self.click(ui, 'setup_project_save')
        nav = json.loads((self.root / '.scp_project.json').read_text())
        self.assertEqual(nav['project_name'], 'Example_project')
        self.assertIsNone(nav['selected_tune'])
        self.assertFalse((self.root / 'cells').exists())
        self.assertFalse(ui._setup_attempted)

    def test_project_switch_guard_and_automatic_restore(self):
        first = self.root
        second = self.root / 'second'
        second.mkdir()
        write_profile(second / SETTINGS_FILE, {'active_amps_pA': [99]})
        ui = self.ui(restore_notebook_settings=True)
        ui.controls['active_amps_pA'].value = '200'
        ui.controls['project_root'].value = str(second)
        self.click(ui, 'setup_open')
        self.assertEqual(ui.project_root, first)
        self.assertEqual(ui._status_kinds['setup_open'], 'error')
        ui.controls['notebook_unsaved_action'].value = 'discard'
        self.click(ui, 'setup_open')
        self.assertEqual(ui.project_root, second)
        self.assertEqual(ui.settings['active_amps_pA'], [99])
        self.assertEqual(ui.controls['active_amps_pA'].value, '99')

    def test_loaded_model_not_replaced_and_loading_options_stay_locked(self):
        ui = self.ui()
        write_profile(self.profile, {'active_amps_pA': [75], 'recompile_modfiles': True})
        supplied = object()
        ui.pipeline_state = supplied
        ui._setup_attempted = True
        with mock.patch('scp_pipeline.notebooks.pipeline_ui.PipelineNotebookUI._sync_controls_from_settings'):
            ui.load_notebook_settings()
        self.assertIs(ui.pipeline_state, supplied)
        self.assertFalse(ui.settings['recompile_modfiles'])
        self.assertEqual(ui.settings['active_amps_pA'], [75])

    def test_import_export_are_session_only_until_explicit_save(self):
        ui = self.ui()
        document = json.dumps(profile_document({'active_amps_pA': [87]})).encode()
        ui.controls['notebook_import'].value = ({'name': 'defaults.json', 'type': 'application/json',
            'size': len(document), 'content': memoryview(document),
            'last_modified': datetime.now(timezone.utc)},)
        self.assertEqual(ui.settings['active_amps_pA'], [87])
        self.assertTrue(ui._notebook_defaults.dirty)
        self.assertFalse(self.profile.exists())
        self.click(ui, 'notebook_export')
        self.assertEqual(ui._status_kinds['notebook_export'], 'complete')
        self.assertFalse(self.profile.exists())
        self.click(ui, 'notebook_save')
        self.assertEqual(read_profile(self.profile)['active_amps_pA'], [87])

    def test_optional_act_invalid_text_does_not_save_old_valid_value(self):
        ui = self.ui()
        for key in ('act_target_currents', 'act_nwb_min', 'act_opt_n_estimators'):
            ui.controls[key].value = 'invalid'
            with self.assertRaises(ValueError):
                ui.save_notebook_settings()
            ui.controls[key].value = ''
        ui.save_notebook_settings()  # Optional ACT need not be configured/installed.

    def test_dirty_project_change_save_first_saves_the_old_scope(self):
        second = self.root / 'second'
        second.mkdir()
        write_profile(second / SETTINGS_FILE, {'active_amps_pA': [99]})
        ui = self.ui(restore_notebook_settings=True)
        ui.controls['active_amps_pA'].value = '201'
        ui.controls['notebook_unsaved_action'].value = 'save'
        ui.controls['project_root'].value = str(second)
        self.click(ui, 'setup_open')
        self.assertEqual(read_profile(self.profile)['active_amps_pA'], [201])
        self.assertEqual(ui.settings['active_amps_pA'], [99])
        self.assertEqual(ui.controls['notebook_unsaved_action'].value, 'cancel')

    def test_scp_namespace_change_respects_dirty_guard_and_profile_scope(self):
        (self.root / 'cells/PV/tunes/tuned').mkdir(parents=True)
        namespace = self.root / 'projects/alpha'
        (namespace / 'cells/A/tunes/orig').mkdir(parents=True)
        write_profile(namespace / SETTINGS_FILE, {'active_amps_pA': [51]})
        ui = PipelineNotebookUI(self.root, restore_notebook_settings=True)
        ui.step1_panel(); ui.step3_panel()
        ui.controls['active_amps_pA'].value = '200'
        self.click(ui, 'notebook_save')
        ui.controls['active_amps_pA'].value = '250'
        with redirect_stdout(io.StringIO()):
            ui.controls['project_name'].value = 'alpha'
        self.assertEqual(ui.controls['project_name'].value, '')
        self.assertIsNone(ui.settings['project_name'])
        ui.controls['notebook_unsaved_action'].value = 'discard'
        ui.controls['project_name'].value = 'alpha'
        self.assertEqual(ui.settings['active_amps_pA'], [51])
        self.assertEqual(ui.controls['active_amps_pA'].value, '51')

    def test_remembered_namespace_switching_updates_lists_paths_and_feedback(self):
        from scp_pipeline.projects import save_project_settings
        from scp_pipeline.workspaces import discover_cells
        for project, cells in ((None, ('PV',)), ('bV1', ('PV', 'SST')),
                               ('A1', ('ET', 'IT', 'PV', 'SST'))):
            base = self.root if project is None else self.root / 'projects' / project
            for cell in cells:
                (base / 'cells' / cell / 'tunes/tuned').mkdir(parents=True)
        a1 = self.root / 'projects/A1'
        save_project_settings(a1, selected_tune=a1 / 'cells/ET/tunes/tuned')
        ui = PipelineNotebookUI(self.root, {'project_name': 'A1'}, restore_notebook_settings=True)
        for panel in (ui.step1_panel, ui.step2_panel, ui.step3_panel, ui.step4_panel, ui.step5_panel):
            panel()
        with mock.patch('scp_pipeline.model.load_cell') as load:
            for project, cell in (('bV1', 'PV'), ('', 'PV'), ('A1', 'ET'), ('bV1', 'PV'), ('', 'PV')):
                with self.subTest(project=project):
                    ui.controls['project_name'].value = project
                    base = self.root / 'projects' / project if project else self.root
                    self.assertEqual(ui.settings['project_name'], project or None)
                    self.assertEqual(list(ui.controls['cell_name'].options),
                                     discover_cells(self.root, project_name=project or None))
                    self.assertEqual(ui.controls['cell_name'].value, cell)
                    self.assertEqual(ui._selected_tune_dir(), base / 'cells' / cell / 'tunes/tuned')
                    self.click(ui, 'setup_remember')
                    self.assertEqual(ui._status_kinds['setup_remember'], 'complete')
                    output = json.loads(''.join(record.get('text', '')
                                               for record in ui.outputs['setup_remember'].outputs))
                    self.assertEqual(output['project_name'], base.name)
                    self.assertEqual(output['selected_tune'], f'cells/{cell}/tunes/tuned')
        load.assert_not_called()

    def test_remembered_paths_do_not_pin_later_cell_or_tune_selection(self):
        from scp_pipeline.projects import save_project_settings
        for cell in ('First', 'Second'):
            for tune in ('orig', 'tuned'):
                (self.root / 'cells' / cell / 'tunes' / tune).mkdir(parents=True)
        first = self.root / 'cells/First/tunes/orig'
        save_project_settings(self.root, selected_tune=first)
        ui = self.ui(restore_notebook_settings=True)
        ui.controls['cell_name'].value = 'Second'
        ui.controls['tune_name'].value = 'tuned'
        self.assertEqual(ui._selected_tune_dir(), self.root / 'cells/Second/tunes/tuned')
        self.assertFalse(ui.controls['tune_dir_override'].value)
        # A deliberately entered explicit path still has precedence.
        ui.controls['tune_dir_override'].value = str(first)
        ui.controls['cell_name'].value = 'First'
        self.assertEqual(ui._selected_tune_dir(), first)

    def test_namespace_reopening_restores_its_own_output_and_navigation(self):
        from scp_pipeline.projects import save_project_settings
        for project, cell in (('first', 'A'), ('second', 'B')):
            base = self.root / 'projects' / project
            tune = base / 'cells' / cell / 'tunes/orig'
            tune.mkdir(parents=True)
            save_project_settings(base, selected_tune=tune, output_dir=base / 'recordings')
        ui = PipelineNotebookUI(self.root, {'project_name': 'first'}, restore_notebook_settings=True)
        ui.step1_panel()
        ui.controls['project_name'].value = 'second'
        self.assertEqual(ui._selected_tune_dir(), self.root / 'projects/second/cells/B/tunes/orig')
        self.assertEqual(ui.settings['output_dir'], str(self.root / 'projects/second/recordings'))
        self.assertEqual(ui.controls['project_root'].value, str(self.root / 'projects/second'))

    def test_location_mode_switch_keeps_external_fields_and_refreshes_cells(self):
        (self.root / 'projects/research/cells/Old/tunes/tuned').mkdir(parents=True)
        external = self.root / 'external'
        (external / 'cells/New/tunes/orig').mkdir(parents=True)
        ui = PipelineNotebookUI(self.root, {'project_name': 'research', 'cell_name': 'Old'}, restore_notebook_settings=True)
        ui.step1_panel()
        ui.controls['project_root'].value = str(external)
        ui.controls['workspace_mode'].value = 'project'
        self.assertEqual(ui.project_root, external)
        self.assertEqual(ui.controls['project_root'].value, str(external))
        self.assertEqual(ui.controls['cell_name'].options, ('New',))
        self.assertEqual(ui.controls['cell_name'].value, 'New')

    def test_code_settings_sync_does_not_reopen_project_or_discard_explicit_paths(self):
        for project in ('first', 'second'):
            (self.root / 'projects' / project / 'cells/Cell/tunes/orig').mkdir(parents=True)
        ui = PipelineNotebookUI(self.root, {'project_name': 'first'}, restore_notebook_settings=True)
        ui.step1_panel()
        # Refreshing dropdown options must not trigger a transient switch to Bundled cells.
        (self.root / 'projects/new_namespace').mkdir()
        selected = self.root / 'arbitrary/tune'
        with mock.patch.object(ui._notebook_defaults, 'restore_scope') as restore:
            ui.apply_settings(dict(ui.settings, project_name='second', cell_name='Cell',
                                   tune_name='orig', tune_dir_override=str(selected)))
        restore.assert_not_called()
        self.assertEqual(ui.controls['project_name'].value, 'second')
        self.assertEqual(ui._selected_tune_dir(), selected)

    def test_defaults_outputs_do_not_erase_project_or_model_logs(self):
        ui = self.ui()
        ui.outputs['step1'].append_stdout('Prepared model log')
        self.click(ui, 'setup_open')
        project_feedback = ui.outputs['setup_open'].outputs
        self.click(ui, 'notebook_save')
        saved_feedback = ui.outputs['notebook_save'].outputs
        self.click(ui, 'notebook_reload')
        self.assertEqual(ui.outputs['notebook_save'].outputs, saved_feedback)
        self.assertEqual(ui.outputs['setup_open'].outputs, project_feedback)
        self.assertIn('Prepared model log', str(ui.outputs['step1'].outputs))

    def test_model_config_bytes_are_not_preset_overrides(self):
        configs = self.root / 'cells/Cell/tunes/orig/cell_configs'
        configs.mkdir(parents=True)
        (configs / 'sim_config.json').write_text(json.dumps({'conditions': {'v_init_mV': -72, 'celsius_C': 35}}))
        (configs / 'target_config.json').write_text(json.dumps({'target_source': {'mode': 'manual'},
            'manual': {'passive': {'v_rest_mV': -70, 'rin_MOhm': 10, 'tau_ms': 5}}}))
        before = {path: path.read_bytes() for path in configs.iterdir()}
        ui = self.ui(settings={'cell_name': 'Cell', 'tune_name': 'orig'})
        ui.save_notebook_settings()
        self.assertEqual(read_profile(self.profile)['simulation_overrides'], {})
        self.assertEqual(read_profile(self.profile)['passive_target_overrides'], {})
        self.assertTrue(all(path.read_bytes() == raw for path, raw in before.items()))


if __name__ == '__main__':
    unittest.main()
