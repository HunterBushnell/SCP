"""Explicitly unset notebook selections never choose an implicit model."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scp_pipeline.notebooks import PipelineNotebookUI
from scp_pipeline.notebooks.notebook_settings import SETTINGS_FILE, write_profile
from scp_pipeline.projects import PROJECT_FILE, save_project_settings


UNSELECTED = {'project_name': None, 'cell_name': None, 'tune_name': None}


class OptionalNotebookSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for cell in ('PV', 'SST'):
            for tune in ('orig', 'tuned'):
                (self.root / 'cells' / cell / 'tunes' / tune).mkdir(parents=True)

    def ui(self, **kwargs):
        return PipelineNotebookUI(self.root, dict(UNSELECTED), restore_notebook_settings=True, **kwargs)

    def test_all_panels_render_without_selection_or_implicit_model_reads(self):
        before = set(self.root.rglob('*'))
        with mock.patch('scp_pipeline.model.load_cell') as load:
            ui = self.ui()
            for panel in (ui.step1_panel, ui.step2_panel, ui.step3_panel, ui.step4_panel, ui.step5_panel):
                panel()
            self.assertEqual(ui.controls['cell_name'].value, '')
            self.assertEqual(ui.controls['tune_name'].value, '')
            self.assertEqual(ui.controls['cell_name'].dropdown.value, '')
            self.assertIn(('Select cell…', ''), ui.controls['cell_name'].dropdown.options)
            self.assertIn('Select a Cell and Tune', ui.controls['setup_model_summary'].value)
            with self.assertRaisesRegex(ValueError, 'Select a Cell and Tune'):
                ui._selected_tune_dir()
            with redirect_stdout(io.StringIO()):
                ui.controls['step1_run'].click()
            self.assertFalse(ui._setup_attempted)
            self.assertEqual(ui._status_kinds['step1'], 'error')
        load.assert_not_called()
        self.assertEqual(set(self.root.rglob('*')), before)

    def test_selecting_a_cell_still_requires_a_tune_choice(self):
        ui = self.ui()
        ui.step1_panel()
        ui.controls['cell_name'].dropdown.value = 'PV'
        self.assertEqual(ui.controls['tune_name'].value, '')
        self.assertIn(('orig', 'orig'), ui.controls['tune_name'].dropdown.options)
        ui.controls['tune_name'].dropdown.value = 'orig'
        self.assertEqual(ui._selected_tune_dir(), self.root / 'cells/PV/tunes/orig')

    def test_saved_bundled_navigation_overrides_unset_initial_values_read_only(self):
        selected = self.root / 'cells/SST/tunes/orig'
        save_project_settings(self.root, selected_tune=selected)
        before = (self.root / PROJECT_FILE).read_bytes()
        ui = self.ui()
        ui.step1_panel()
        self.assertEqual(ui.controls['cell_name'].value, 'SST')
        self.assertEqual(ui.controls['tune_name'].value, 'orig')
        self.assertEqual(ui._selected_tune_dir(), selected)
        self.assertEqual((self.root / PROJECT_FILE).read_bytes(), before)

    def test_external_project_navigation_and_no_checkout_navigation_write(self):
        project = self.root / 'external'
        selected = project / 'cells/Custom/tunes/working'
        selected.mkdir(parents=True)
        save_project_settings(project, selected_tune=selected)
        ui = self.ui(project_root=project)
        ui.step1_panel()
        self.assertEqual(ui.controls['cell_name'].value, 'Custom')
        self.assertEqual(ui.controls['tune_name'].value, 'working')
        self.assertEqual(ui._selected_tune_dir(), selected)
        with redirect_stdout(io.StringIO()):
            ui.controls['setup_remember'].click()
        self.assertEqual(ui._status_kinds['setup_remember'], 'complete')
        self.assertFalse((self.root / PROJECT_FILE).exists())

    def test_remembered_named_project_reopens_without_notebook_hardcoding(self):
        project = self.root / 'projects/research'
        selected = project / 'cells/Custom/tunes/working'
        selected.mkdir(parents=True)
        write_profile(project / SETTINGS_FILE, {'seed': 72})
        ui = self.ui()
        ui.step1_panel()
        ui.controls['project_name'].value = 'research'
        self.assertEqual(ui.controls['cell_name'].value, '')
        self.assertEqual(ui.controls['tune_name'].value, '')
        ui.controls['cell_name'].dropdown.value = 'Custom'
        ui.controls['tune_name'].dropdown.value = 'working'
        with redirect_stdout(io.StringIO()):
            ui.controls['setup_remember'].click()
        self.assertEqual(ui._status_kinds['setup_remember'], 'complete')
        saved = json.loads((self.root / PROJECT_FILE).read_text())
        self.assertEqual(saved['selected_project'], 'research')
        self.assertIsNone(saved['selected_tune'])
        with mock.patch('scp_pipeline.model.load_cell') as load:
            reopened = self.ui()
            reopened.step1_panel()
        load.assert_not_called()
        self.assertEqual(reopened.controls['project_name'].value, 'research')
        self.assertEqual(reopened.controls['cell_name'].value, 'Custom')
        self.assertEqual(reopened.controls['tune_name'].value, 'working')
        self.assertEqual(reopened.settings['seed'], 72)
        self.assertEqual(reopened._selected_tune_dir(), selected)

    def test_remembering_a_project_preserves_bundled_navigation(self):
        selected = self.root / 'cells/SST/tunes/orig'
        save_project_settings(self.root, selected_tune=selected)
        research = self.root / 'projects/research'
        (research / 'cells/Custom/tunes/working').mkdir(parents=True)
        ui = self.ui()
        ui.step1_panel()
        ui.controls['project_name'].value = 'research'
        ui.controls['cell_name'].dropdown.value = 'Custom'
        ui.controls['tune_name'].dropdown.value = 'working'
        with redirect_stdout(io.StringIO()):
            ui.controls['setup_remember'].click()
        self.assertEqual(json.loads((self.root / PROJECT_FILE).read_text())['selected_tune'],
                         'cells/SST/tunes/orig')
        ui.controls['project_name'].value = ''
        self.assertEqual(ui._selected_tune_dir(), selected)
        with redirect_stdout(io.StringIO()):
            ui.controls['setup_remember'].click()
        reopened = self.ui()
        self.assertIsNone(reopened.settings['project_name'])
        self.assertEqual(reopened._selected_tune_dir(), selected)

    def test_invalid_remembered_project_names_are_rejected_without_writes(self):
        with self.assertRaisesRegex(ValueError, 'selected_project'):
            save_project_settings(self.root, selected_project='../outside')
        self.assertFalse((self.root / PROJECT_FILE).exists())

    def test_explicit_tune_path_still_works_with_unset_cell_and_tune(self):
        tune = self.root / 'cells/PV/tunes/orig'
        ui = self.ui()
        ui.step1_panel()
        ui.controls['tune_dir_override'].value = str(tune)
        self.assertEqual(ui._selected_tune_dir(), tune)

    def test_location_mode_change_does_not_select_a_first_cell_for_blank_inputs(self):
        project = self.root / 'external'
        (project / 'cells/Custom/tunes/orig').mkdir(parents=True)
        ui = self.ui()
        ui.step1_panel()
        ui.controls['project_root'].value = str(project)
        ui.controls['workspace_mode'].value = 'project'
        self.assertEqual(ui.controls['cell_name'].value, '')
        self.assertEqual(ui.controls['tune_name'].value, '')
        self.assertIn(('Custom', 'Custom'), ui.controls['cell_name'].dropdown.options)

    def test_legacy_callers_without_explicit_none_retain_pv_defaults(self):
        ui = PipelineNotebookUI(self.root)
        ui.step1_panel()
        self.assertEqual(ui.controls['cell_name'].value, 'PV')
        self.assertEqual(ui.controls['tune_name'].value, 'tuned')
