"""Workspace menus stay unfiltered; new names require explicit entry."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scp_pipeline.notebooks import PipelineNotebookUI
from scp_pipeline.notebooks._pipeline_ui.name_selector import WorkspaceNameSelector
from tests.pipeline_ui_test_support import all_descendants


class WorkspaceNameSelectorTests(unittest.TestCase):
    def selector(self, **kwargs):
        control = WorkspaceNameSelector(**kwargs)
        self.addCleanup(control.close)
        self.addCleanup(control.dropdown.close)
        self.addCleanup(control.editor.close)
        return control

    def test_selected_value_does_not_filter_existing_choices(self):
        selector = self.selector(description='Tune', options=['orig', 'tuned', 'tuned_adb'], value='tuned')
        self.assertEqual(selector.dropdown.options,
                         (('orig', 'orig'), ('tuned', 'tuned'), ('tuned_adb', 'tuned_adb')))
        self.assertEqual(selector.editor.layout.display, 'none')
        observed = []
        selector.observe(lambda change: observed.append(change['new']), names='value')
        selector.dropdown.value = 'orig'
        self.assertEqual(selector.value, 'orig')
        self.assertEqual(observed, ['orig'])
        self.assertEqual(len(selector.dropdown.options), 3)

    def test_new_name_entry_is_explicit_and_keeps_existing_choices(self):
        selector = self.selector(description='Cell', options=['PV', 'SST'], value='PV', allow_new=True)
        self.assertEqual(selector.editor.layout.display, 'none')
        selector.dropdown.value = None
        self.assertEqual(selector.value, '')
        self.assertEqual(selector.editor.layout.display, '')
        selector.editor.value = 'NewCell'
        self.assertEqual(selector.value, 'NewCell')
        self.assertEqual(selector.options, ('PV', 'SST'))
        self.assertEqual(selector.dropdown.options,
                         (('PV', 'PV'), ('SST', 'SST'), ('Enter new name…', None)))
        selector.dropdown.value = 'SST'
        self.assertEqual(selector.value, 'SST')
        self.assertEqual(selector.editor.layout.display, 'none')

    def test_code_first_values_option_refresh_and_disabled_state(self):
        selector = self.selector(description='Cell', options=['A', 'B'], value='A')
        selector.value = 'CodeCell'
        self.assertEqual(selector.dropdown.value, 'CodeCell')
        selector.options = ['CodeCell', 'Other']
        self.assertEqual(selector.dropdown.options, (('CodeCell', 'CodeCell'), ('Other', 'Other')))
        selector.disabled = True
        self.assertTrue(selector.dropdown.disabled)
        self.assertTrue(selector.editor.disabled)
        selector.disabled = False
        self.assertFalse(selector.dropdown.disabled)

    def test_empty_registration_exposes_name_entry_without_fabricating_cells(self):
        selector = self.selector(description='Cell', options=[''], value='', allow_new=True)
        self.assertIsNone(selector.dropdown.value)
        self.assertEqual(selector.editor.layout.display, '')
        selector.editor.value = 'Custom'
        self.assertEqual(selector.value, 'Custom')
        selector.allow_new = False
        self.assertEqual(selector.editor.layout.display, 'none')
        self.assertEqual(selector.dropdown.value, 'Custom')

    def test_project_cell_and_tune_choices_are_complete_and_update_immediately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for project, cells in (('A1', ('ET', 'IT', 'PV', 'SST')), ('bV1', ('PV', 'SST'))):
                for cell in cells:
                    for tune in ('orig', 'tuned'):
                        (root / 'projects' / project / 'cells' / cell / 'tunes' / tune).mkdir(parents=True)
            ui = PipelineNotebookUI(root, {'project_name': 'A1', 'cell_name': 'ET', 'tune_name': 'tuned'},
                                    restore_notebook_settings=True)
            panel = ui.step1_panel()
            self.assertFalse(any(isinstance(widget, ui.widgets.Combobox) for widget in all_descendants(panel)))
            cell, tune = ui.controls['cell_name'], ui.controls['tune_name']
            self.assertEqual(cell.dropdown.options, (('ET', 'ET'), ('IT', 'IT'), ('PV', 'PV'), ('SST', 'SST')))
            self.assertEqual(tune.dropdown.options, (('orig', 'orig'), ('tuned', 'tuned')))
            with mock.patch('scp_pipeline.model.load_cell') as load:
                cell.dropdown.value = 'IT'
                tune.dropdown.value = 'orig'
                self.assertEqual(ui._selected_tune_dir(), root / 'projects/A1/cells/IT/tunes/orig')
                ui.controls['project_name'].value = 'bV1'
                self.assertEqual(cell.dropdown.options, (('PV', 'PV'), ('SST', 'SST')))
                self.assertEqual(tune.dropdown.options, (('orig', 'orig'), ('tuned', 'tuned')))
                with redirect_stdout(io.StringIO()):
                    ui.controls['setup_remember'].click()
                self.assertEqual(ui._status_kinds['setup_remember'], 'complete')
            load.assert_not_called()

    def test_ui_shows_new_name_option_only_for_registration_or_live_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ui = PipelineNotebookUI(project_root=root)
            ui.step1_panel()
            cell, tune = ui.controls['cell_name'], ui.controls['tune_name']
            self.assertTrue(cell.allow_new)
            cell.editor.value = 'NewCell'
            tune.dropdown.value = None
            tune.editor.value = 'working'
            self.assertEqual(ui.settings['cell_name'], 'NewCell')
            self.assertEqual(ui.settings['tune_name'], 'working')
            self.assertEqual(ui._selected_tune_dir(), root / 'cells/NewCell/tunes/working')
            ui.controls['setup_action'].value = 'existing'
            self.assertFalse(cell.allow_new)
            self.assertFalse(tune.allow_new)
            self.assertEqual(cell.editor.layout.display, 'none')
            self.assertEqual(tune.editor.layout.display, 'none')
            ui.controls['setup_action'].value = 'live'
            self.assertTrue(cell.allow_new)

    def test_visible_selection_row_contains_plain_noncollapsing_dropdowns(self):
        from ipywidgets.embed import dependency_state
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'cells/PV/tunes/tuned').mkdir(parents=True)
            ui = PipelineNotebookUI(root)
            panel = ui.step1_panel()
            row = ui.controls['setup_selection_row']
            expected = (ui.controls['project_name'], ui.controls['cell_name'].dropdown,
                        ui.controls['tune_name'].dropdown)
            self.assertEqual(row.children, expected)
            self.assertTrue(all(type(control) is ui.widgets.Dropdown for control in row.children))
            self.assertEqual(row.layout.flex_flow, 'row wrap')
            for field in row.children:
                self.assertEqual(field.layout.width, '280px')
                self.assertEqual(field.layout.flex, '0 0 auto')
                self.assertNotEqual(field.layout.display, 'none')
            # Check the serialized view graph, not only the Python control map.
            state = dependency_state([panel], drop_defaults=False)
            serialized_row = state[row.model_id]['state']['children']
            self.assertEqual(serialized_row, ['IPY_MODEL_' + field.model_id for field in expected])
            for field in expected:
                self.assertEqual(state[field.model_id]['model_name'], 'DropdownModel')

    def test_new_name_editors_are_rendered_directly_and_only_shown_on_request(self):
        with tempfile.TemporaryDirectory() as directory:
            ui = PipelineNotebookUI(project_root=Path(directory))
            panel = ui.step1_panel()
            cell, tune = ui.controls['cell_name'], ui.controls['tune_name']
            row = ui.controls['setup_name_entry_row']
            self.assertEqual(row.children, (cell.editor, tune.editor))
            descendants = list(all_descendants(panel))
            self.assertNotIn(cell, descendants)
            self.assertIn(cell.dropdown, descendants)
            self.assertIn(cell.editor, descendants)
            cell.editor.value = 'Cell'
            tune.dropdown.value = None
            tune.editor.value = 'working'
            self.assertEqual(ui.settings['cell_name'], 'Cell')
            self.assertEqual(ui.settings['tune_name'], 'working')
            self.assertEqual(tune.editor.layout.display, '')
            ui.controls['setup_action'].value = 'existing'
            self.assertEqual(cell.editor.layout.display, 'none')
            self.assertEqual(tune.editor.layout.display, 'none')
