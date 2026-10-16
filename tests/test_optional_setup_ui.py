"""Compact optional setup drafts are separate from notebook/session defaults."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest import mock

from tests import test_project_setup_ui as fixtures
from tests.pipeline_ui_test_support import fake_state, make_repo
from scp_pipeline.notebooks._pipeline_ui.setup_outputs import generation_options
from scp_pipeline.notebooks._pipeline_ui import project_setup


class OptionalSetupUITests(fixtures.ProjectSetupUIFixture, unittest.TestCase):
    def test_initial_targets_saved_once_and_separate_from_runtime_settings(self):
        ui = self.ui(restore_notebook_settings=True)
        self.configure_registration(ui)
        c = ui.controls
        before_settings = dict(ui.settings)
        c['setup_targets_enabled'].value = True
        c['setup_target_manual_passive_v_rest_mV'].value = '-68'
        c['setup_target_manual_fi_curve_currents_pA'].value = '[0, 100]'
        c['setup_target_manual_fi_curve_rates_Hz'].value = '[0, 15]'
        c['setup_synapses_enabled'].value = True
        before = self.files()
        self.click(ui, 'setup_preview')
        self.assertEqual(ui._status_kinds['setup_preview'], 'complete', self.output_text(ui, 'setup_preview'))
        self.assertEqual(self.files(), before)
        with mock.patch('scp_pipeline.model.load_cell') as load:
            self.click(ui, 'setup_save')
        load.assert_not_called()
        self.assertEqual(ui._status_kinds['setup_save'], 'complete', self.output_text(ui, 'setup_save'))
        target = self.tune / 'cell_configs/target_config.json'
        contents = target.read_bytes()
        self.assertEqual(json.loads(contents)['manual']['passive']['v_rest_mV'], -68)
        self.assertTrue(c['setup_targets_enabled'].disabled)
        self.assertIn(str(target), c['setup_target_existing'].value)
        self.assertEqual(c['setup_target_form'].layout.display, 'none')
        self.assertFalse(c['setup_synapses_enabled'].value)
        self.assertEqual(ui.settings['passive_target_overrides'], before_settings['passive_target_overrides'])
        self.assertFalse(any(key.startswith('setup_target') for key in ui.settings))
        self.click(ui, 'setup_save')
        self.assertEqual(target.read_bytes(), contents)

    def test_mode_visibility_invalid_values_and_retry(self):
        ui = self.ui()
        self.configure_registration(ui)
        c = ui.controls
        c['setup_targets_enabled'].value = True
        self.assertEqual(c['setup_target_mode'].value, 'manual')
        c['setup_target_manual_passive_tau_ms'].value = 'not a number'
        before = self.files()
        self.click(ui, 'setup_save')
        self.assertEqual(ui._status_kinds['setup_save'], 'error')
        self.assertEqual(self.files(), before)
        c['setup_target_manual_passive_tau_ms'].value = ''
        for mode in ('traces', 'allen_nwb', 'none', 'manual'):
            c['setup_target_mode'].value = mode
            expected = '' if mode in {'traces', 'allen_nwb'} else 'none'
            self.assertEqual(c['setup_target_manual'].layout.display, expected)
            self.click(ui, 'setup_preview')
            self.assertEqual(ui._status_kinds['setup_preview'], 'complete', self.output_text(ui, 'setup_preview'))
        self.click(ui, 'setup_save')
        self.assertEqual(ui._status_kinds['setup_save'], 'complete', self.output_text(ui, 'setup_save'))

    def test_existing_tune_missing_optional_and_required_configs(self):
        self.register()
        (self.tune / 'cell_configs/geometry.json').unlink()
        ui = self.ui()
        c = ui.controls
        c['setup_synapses_enabled'].value = True
        c['setup_targets_enabled'].value = True
        c['setup_target_mode'].value = 'none'
        recipe = (self.tune / 'cell_configs/cell_config.json').read_bytes()
        self.click(ui, 'setup_save')
        self.assertEqual(ui._status_kinds['setup_save'], 'complete', self.output_text(ui, 'setup_save'))
        self.assertTrue((self.tune / 'cell_configs/geometry.json').is_file())
        self.assertTrue((self.tune / 'cell_configs/syn_config.json').is_file())
        self.assertEqual((self.tune / 'cell_configs/cell_config.json').read_bytes(), recipe)

    def test_copy_preview_create_selects_without_loading_and_feedback_persists(self):
        self.register()
        ui = self.ui()
        c = ui.controls
        before = self.files()
        with mock.patch('scp_pipeline.model.load_cell') as load:
            self.click(ui, 'setup_copy_preview')
            self.assertEqual(ui._status_kinds['setup_copy_preview'], 'complete', self.output_text(ui, 'setup_copy_preview'))
            self.assertEqual(self.files(), before)
            preview = self.action_snapshot(ui, 'setup_copy_preview')
            self.click(ui, 'setup_copy_create')
        load.assert_not_called()
        self.assertEqual(ui._status_kinds['setup_copy_create'], 'complete', self.output_text(ui, 'setup_copy_create'))
        self.assertEqual(ui._selected_tune_dir(), self.tune.with_name('tuned'))
        self.assertEqual(c['tune_name'].value, 'tuned')
        self.assertFalse(ui._setup_attempted)
        self.assertEqual(self.action_snapshot(ui, 'setup_copy_preview'), preview)
        self.click(ui, 'setup_copy_create')
        self.assertEqual(ui._status_kinds['setup_copy_create'], 'error')

    def test_copy_errors_do_not_capture_widget_updates_or_block_loading(self):
        self.register()
        ui = self.ui()
        c = ui.controls
        # File-only callbacks must not enter Output at all: it redirects live
        # comm feedback and can suppress exceptions raised by widget observers.
        with mock.patch.object(ui.widgets.Output, '__enter__',
                               side_effect=AssertionError('Do not capture widget callbacks')):
            c['setup_copy_name'].value = '../unsafe'
            self.click(ui, 'setup_copy_preview')
            self.assertEqual(ui._status_kinds['setup_copy_preview'], 'error')
            self.assertFalse(ui._setup_attempted)
            c['setup_copy_name'].value = 'tuned'
            for _ in range(2):
                self.click(ui, 'setup_copy_preview')
                self.assertEqual(ui._status_kinds['setup_copy_preview'], 'complete',
                                 self.output_text(ui, 'setup_copy_preview'))
            self.click(ui, 'setup_copy_create')
            self.assertEqual(ui._status_kinds['setup_copy_create'], 'complete',
                             self.output_text(ui, 'setup_copy_create'))
            self.click(ui, 'setup_copy_create')
            self.assertEqual(ui._status_kinds['setup_copy_create'], 'error')
            self.assertIn('already exists', self.output_text(ui, 'setup_copy_create'))
            self.assertFalse(c['step1_run'].disabled)
            self.assertFalse(ui._setup_attempted)
        state = fake_state(self.root, cell='Cell', tune='tuned')
        with mock.patch('scp_pipeline.notebooks._pipeline_ui.step1.prepare_pipeline_notebook',
                        return_value=state) as prepare:
            self.click(ui, 'step1_run')
        prepare.assert_called_once()
        self.assertIs(ui.pipeline_state, state)
        self.assertEqual(ui._status_kinds['step1'], 'complete')

    def test_created_copy_survives_selection_error_and_can_be_opened(self):
        self.register()
        ui = self.ui()
        destination = self.tune.with_name('tuned')
        with mock.patch.object(project_setup, '_select_registered_tune',
                               side_effect=ValueError('selection refresh failed')):
            self.click(ui, 'setup_copy_create')
        self.assertTrue((destination / 'cell_configs/cell_config.json').is_file())
        self.assertEqual(ui._status_kinds['setup_copy_create'], 'error')
        self.assertIn('Working copy was created at', self.output_text(ui, 'setup_copy_create'))
        self.assertIn(str(destination), self.output_text(ui, 'setup_copy_create'))
        self.assertIn('Do not create it again', self.output_text(ui, 'setup_copy_create'))
        self.assertFalse(ui._setup_attempted)
        ui.controls['tune_dir_override'].value = str(destination)
        state = fake_state(self.root, cell='Cell', tune='tuned')
        with mock.patch('scp_pipeline.notebooks._pipeline_ui.step1.prepare_pipeline_notebook',
                        return_value=state) as prepare:
            self.click(ui, 'step1_run')
        self.assertEqual(prepare.call_args.kwargs['tune_dir_override'], str(destination))
        self.assertEqual(ui._status_kinds['step1'], 'complete')

    def test_file_validation_can_retry_before_locking_in_both_workspace_modes(self):
        self.register()
        project_ui = self.ui()
        from scp_pipeline.notebooks import PipelineNotebookUI
        repo = make_repo(self.root)
        hosted_ui = PipelineNotebookUI(repo, {'cell_name': 'A', 'tune_name': 'tuned'})
        hosted_ui.step1_panel()
        for ui in (project_ui, hosted_ui):
            with self.subTest(project_owned=ui.project_root is not None):
                tune = ui._selected_tune_dir()
                path = tune / 'cell_configs/sim_config.json'
                original = path.read_bytes()
                config = json.loads(original)
                config['conditions']['celsius_C'] = 'invalid temperature'
                path.write_text(json.dumps(config))
                with mock.patch('scp_pipeline.notebooks._pipeline_ui.step1.prepare_pipeline_notebook') as prepare:
                    self.click(ui, 'step1_run')
                prepare.assert_not_called()
                self.assertFalse(ui._setup_attempted)
                self.assertFalse(ui.controls['setup_copy_preview'].disabled)
                self.assertFalse(ui.controls['step1_run'].disabled)
                self.assertIn('celsius_C', ui.statuses['step1'].value)
                path.write_bytes(original)
                with mock.patch('scp_pipeline.notebooks._pipeline_ui.step1.prepare_pipeline_notebook',
                                return_value=fake_state(tune.parents[3], cell=tune.parents[1].name,
                                                        tune=tune.name)) as prepare:
                    self.click(ui, 'step1_run')
                prepare.assert_called_once()
                self.assertEqual(ui._status_kinds['step1'], 'complete')

    def test_load_error_status_escapes_live_capture_but_keeps_restart_lock(self):
        self.register()
        ui = self.ui()
        shell = mock.Mock()
        shell.kernel.get_parent.return_value = {'header': {'msg_id': 'load-error'}}
        statuses = []
        original = ui._set_status
        def status(step, kind, message):
            if step == 'step1':
                statuses.append((kind, ui.outputs['step1'].msg_id))
            original(step, kind, message)
        with mock.patch('ipywidgets.widgets.widget_output.get_ipython', return_value=shell), \
                mock.patch.object(ui, '_set_status', side_effect=status), \
                mock.patch('scp_pipeline.notebooks._pipeline_ui.step1.prepare_pipeline_notebook',
                           side_effect=RuntimeError('native construction failed')):
            self.click(ui, 'step1_run')
        self.assertEqual(ui._status_kinds['step1'], 'error')
        self.assertIn('native construction failed', ui.statuses['step1'].value)
        self.assertEqual(statuses[-1], ('error', ''))
        self.assertTrue(ui.controls['step1_run'].disabled)
        self.assertTrue(ui.controls['setup_copy_create'].disabled)
        shell.showtraceback.assert_not_called()

    def test_postload_widget_error_finishes_status_and_preserves_loaded_cell(self):
        self.register()
        ui = self.ui()
        ui.step2_panel()
        state = fake_state(self.root, cell='Cell', tune='orig')
        with mock.patch('scp_pipeline.notebooks._pipeline_ui.step1.prepare_pipeline_notebook',
                        return_value=state), \
                mock.patch.object(ui, '_refresh_connections', side_effect=ValueError('bad notebook controls')):
            self.click(ui, 'step1_run')
        self.assertIs(ui.pipeline_state, state)
        self.assertTrue(ui._restart_required)
        self.assertEqual(ui._status_kinds['step1'], 'error')
        self.assertIn('Cell loaded', ui.statuses['step1'].value)
        self.assertIn('bad notebook controls', ui.step1_load_log)
        self.assertTrue(ui.controls['step2_run'].disabled)

    def test_load_summary_names_the_effective_native_source_and_mode(self):
        self.register()
        config_path = self.tune / 'cell_configs/cell_config.json'
        config = json.loads(config_path.read_text())
        for mode in ('copy', 'reference'):
            with self.subTest(mode=mode):
                config['model_source']['mode'] = mode
                config_path.write_text(json.dumps(config))
                ui = self.ui()
                ui.pipeline_state = fake_state(self.root, cell='Cell', tune='orig')
                ui.pipeline_state.context.cell_config = config
                captured = io.StringIO()
                with redirect_stdout(captured):
                    ui._print_step1_load_summary(quiet_output=True)
                expected = self.tune / 'model_sources/Cell.hoc' if mode == 'copy' else self.entry
                self.assertIn(str(expected), captured.getvalue())
                self.assertIn('Source mode: ' + mode, captured.getvalue())
                self.assertIn('SetupUICell', captured.getvalue())
                self.assertIn('built-in mechanisms only', captured.getvalue())

    def test_dirty_draft_blocks_navigation_then_confirm_discards(self):
        self.register()
        from scp_pipeline.setup import create_working_copy
        create_working_copy(source_tune_dir=self.tune, target_tune_name='tuned', content_policy='model')
        ui = self.ui()
        c = ui.controls
        c['setup_targets_enabled'].value = True
        c['setup_target_manual_passive_tau_ms'].value = '12'
        remembered_path = c['tune_dir_override'].value
        c['tune_name'].value = 'tuned'
        self.assertEqual(c['tune_name'].value, 'orig')
        self.assertEqual(ui.settings['tune_name'], 'orig')
        self.assertEqual(c['tune_dir_override'].value, remembered_path)
        self.assertNotEqual(c['setup_draft_discard'].layout.display, 'none')
        self.assertEqual(c['setup_target_manual_passive_tau_ms'].value, '12')
        c['setup_draft_discard'].value = True
        c['tune_name'].value = 'tuned'
        self.assertEqual(c['tune_name'].value, 'tuned')
        self.assertFalse(c['setup_targets_enabled'].value)
        self.assertEqual(c['setup_target_manual_passive_tau_ms'].value, '')

    def test_live_cell_save_creates_only_bookkeeping_when_optional_generation_disabled(self):
        from neuron import h
        from scp_pipeline.notebooks import PipelineNotebookUI
        cell = h.Section(name='optional_live_soma')
        cell.insert('pas')
        ui = PipelineNotebookUI(project_root=self.root,
            settings={'cell_name': 'Live', 'tune_name': 'orig'}, cell_override=cell,
            runtime_sim_config={'conditions': {'v_init_mV': -65, 'celsius_C': 30}})
        ui.step1_panel()
        self.assertEqual(ui.controls['setup_action'].value, 'live')
        self.click(ui, 'setup_save')
        self.assertEqual(ui._status_kinds['setup_save'], 'complete', self.output_text(ui, 'setup_save'))
        tune = ui._selected_tune_dir()
        config = json.loads((tune / 'cell_configs/cell_config.json').read_text())
        self.assertEqual(config['cell_loader'], 'runtime_cell')
        self.assertEqual(config['workspace_layout'], 'project_owned')
        self.assertEqual(config['project_name'], ui.controls['project_label'].value)
        self.assertFalse((tune / 'cell_configs/target_config.json').exists())
        self.assertFalse((tune / 'cell_configs/syn_groups').exists())
        self.assertFalse(ui._setup_attempted)
        self.assertEqual(cell.L, 100)

    def test_final_status_published_after_live_output_capture_even_errors(self):
        ui = self.ui()
        shell = mock.Mock()
        shell.kernel.get_parent.return_value = {'header': {'msg_id': 'optional-feedback'}}
        states = []
        original = project_setup._feedback
        def feedback(owner, key, message, **kwargs):
            states.append((kwargs.get('kind', 'complete'), owner.outputs[key].msg_id))
            return original(owner, key, message, **kwargs)
        with mock.patch('ipywidgets.widgets.widget_output.get_ipython', return_value=shell), \
                mock.patch.object(project_setup, '_feedback', side_effect=feedback):
            self.click(ui, 'setup_discover')
            ui.controls['setup_entry'].value = str(self.entry)
            self.click(ui, 'setup_discover')
            self.click(ui, 'setup_discover')
        self.assertTrue(any(kind == 'error' for kind, _ in states))
        self.assertTrue(all(not msg_id for kind, msg_id in states if kind != 'running'))
        self.assertEqual(ui._status_kinds['setup_discover'], 'complete')
        shell.showtraceback.assert_not_called()

    def test_upper_menus_have_existing_names_only_and_new_editors_below(self):
        ui = self.ui()
        c = ui.controls
        for key in ('cell_name', 'tune_name'):
            self.assertFalse(any(label == 'Enter new name…' for label, _value in c[key].dropdown.options))
            c[key].editor.value = 'New' + key
            self.assertFalse(any(label == 'Enter new name…' for label, _value in c[key].dropdown.options))
            self.assertNotIn('New' + key, [value for _label, value in c[key].dropdown.options])


    def test_unsaved_generation_draft_blocks_native_loading_before_locking(self):
        self.register()
        ui = self.ui()
        ui.controls['setup_targets_enabled'].value = True
        with mock.patch('scp_pipeline.notebooks._pipeline_ui.step1.prepare_pipeline_notebook') as prepare:
            self.click(ui, 'step1_run')
        prepare.assert_not_called()
        self.assertFalse(ui._setup_attempted)
        self.assertFalse(ui.controls['setup_save'].disabled)
        self.assertIn('Unsaved optional-setup draft', ui.statuses['step1'].value)
        self.assertFalse((self.tune / 'cell_configs/target_config.json').exists())

    def test_invalid_project_label_blocks_generation_before_writes(self):
        self.register()
        ui = self.ui()
        ui.controls['setup_targets_enabled'].value = True
        ui.controls['project_label'].value = '../unsafe'
        before = self.files()
        self.click(ui, 'setup_save')
        self.assertEqual(ui._status_kinds['setup_save'], 'error')
        self.assertEqual(self.files(), before)

    def test_allen_acquisition_preserves_preexisting_scientific_files(self):
        ui = self.ui()
        self.configure_registration(ui)
        c = ui.controls
        c['setup_loader'].value = 'adb'
        c['setup_specimen'].value = '12345'
        self.tune.mkdir(parents=True)
        configs = self.tune / 'cell_configs'
        configs.mkdir()
        sim = configs / 'sim_config.json'
        sim.write_text('{"conditions":{"v_init_mV":-77,"celsius_C":22},"tstop":123}\n')
        geometry = configs / 'geometry.json'
        geometry.write_text('{"custom":"preserve"}\n')
        before = {path: path.read_bytes() for path in (sim, geometry)}
        def acquire(**kwargs):
            self.assertFalse(kwargs['do_base_configs'])
            (kwargs['tune_dir'] / 'manifest.json').write_text('{}')
            return {'status': 'acquired'}
        with mock.patch('scp_pipeline.setup.prepare_tune', side_effect=acquire) as download:
            self.click(ui, 'setup_preview')
            download.assert_not_called()
            self.click(ui, 'setup_save')
        self.assertEqual(ui._status_kinds['setup_save'], 'complete', self.output_text(ui, 'setup_save'))
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertEqual(json.loads((configs / 'cell_config.json').read_text())['color'], 'k')

    def test_live_preflight_does_not_leave_partial_bookkeeping_on_invalid_geometry(self):
        from neuron import h
        from scp_pipeline.notebooks import PipelineNotebookUI
        cell = h.Section(name='optional_invalid_geometry')
        geometry = self.tune / 'cell_configs/geometry.json'
        geometry.parent.mkdir(parents=True)
        geometry.write_text('[]')
        ui = PipelineNotebookUI(project_root=self.root,
            settings={'cell_name': 'Cell', 'tune_name': 'orig'}, cell_override=cell,
            runtime_sim_config={'conditions': {'v_init_mV': -65, 'celsius_C': 30}})
        ui.step1_panel()
        before = self.files()
        for action in ('setup_preview', 'setup_save'):
            self.click(ui, action)
            self.assertEqual(ui._status_kinds[action], 'error', self.output_text(ui, action))
            self.assertIn('configuration object', self.output_text(ui, action))
            self.assertEqual(self.files(), before)

    def test_live_optional_save_preserves_stored_runtime_settings_with_session_override(self):
        from neuron import h
        from scp_pipeline.notebooks import PipelineNotebookUI
        self.register()
        sim = self.tune / 'cell_configs/sim_config.json'
        sim.write_text('{"conditions":{"v_init_mV":null,"celsius_C":null},"dt":0.025}\n')
        cell = h.Section(name='optional_overridden_conditions')
        ui = PipelineNotebookUI(project_root=self.root,
            settings={'cell_name': 'Cell', 'tune_name': 'orig'}, cell_override=cell,
            runtime_sim_config={'conditions': {'v_init_mV': -65, 'celsius_C': 30}})
        ui.step1_panel()
        ui.controls['setup_targets_enabled'].value = True
        ui.controls['setup_target_mode'].value = 'none'
        contents = sim.read_bytes()
        recipe = (sim.parent / 'cell_config.json').read_bytes()
        self.click(ui, 'setup_save')
        self.assertEqual(ui._status_kinds['setup_save'], 'complete', self.output_text(ui, 'setup_save'))
        self.assertEqual(sim.read_bytes(), contents)
        self.assertEqual((sim.parent / 'cell_config.json').read_bytes(), recipe)
        self.assertEqual(json.loads((sim.parent / 'target_config.json').read_text())['target_source']['mode'], 'none')


if __name__ == '__main__':
    unittest.main()
