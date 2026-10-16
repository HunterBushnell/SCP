"""First-use project setup checks use disposable sources and never load NEURON."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scp_pipeline.notebooks import PipelineNotebookUI
from scp_pipeline.projects import PROJECT_FILE
from scp_pipeline.setup import register_workspace
from tests.pipeline_ui_test_support import all_descendants


class ProjectSetupUIFixture:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "project"
        self.source = self.root / "native"
        self.source.mkdir(parents=True)
        self.entry = self.source / "Cell.hoc"
        self.entry.write_text(
            "begintemplate SetupUICell\n"
            "public soma\ncreate soma\n"
            "proc init() { soma { L=20 diam=20 insert pas } }\n"
            "endtemplate SetupUICell\n",
            encoding="utf-8",
        )
        self.tune = self.root / "cells" / "Cell" / "tunes" / "orig"

    def ui(self, **kwargs):
        ui = PipelineNotebookUI(project_root=self.root, **kwargs)
        ui.step1_panel()
        return ui

    def configure_registration(self, ui):
        controls = ui.controls
        controls["setup_action"].value = "register"
        controls["project_label"].value = "Synthetic_project"
        controls["cell_name"].value = "Cell"
        controls["tune_name"].value = "orig"
        controls["setup_entry"].value = str(self.entry)
        controls["setup_template"].value = "SetupUICell"
        controls["setup_voltage"].value = "-65"
        controls["setup_celsius"].value = "30"

    def register(self):
        with redirect_stdout(io.StringIO()):
            register_workspace(
                dry_run=False,
                project_root=self.root,
                tune_dir=self.tune,
                cell_name="Cell",
                tune_name="orig",
                project_name="Synthetic_project",
                cell_config={
                    "cell_loader": "hoc_template",
                    "paths": {"hoc_template": "Cell.hoc", "modfiles": None},
                    "hoc_template": {
                        "template_name": "SetupUICell",
                        "constructor_args": [],
                    },
                    "model_source": {
                        "mode": "copy",
                        "source_root": str(self.source),
                        "files": ["Cell.hoc"],
                    },
                },
                conditions={"v_init_mV": -65, "celsius_C": 30},
            )

    def files(self):
        return {
            path.relative_to(self.root).as_posix(): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }

    def click(self, ui, name):
        captured = io.StringIO()
        with redirect_stdout(captured):
            ui.controls[name].click()
        return captured.getvalue()

    def output_text(self, ui, name):
        return "".join(record.get("text", "") for record in ui.outputs[name].outputs)

    def action_snapshot(self, ui, name):
        return (ui.controls[f"{name}_feedback"].value, ui.outputs[name].outputs)


class ProjectSetupUITests(ProjectSetupUIFixture, unittest.TestCase):
    def test_empty_project_starts_with_registration_not_a_missing_workspace(self):
        ui = self.ui()
        self.assertEqual(ui.controls["setup_action"].value, "register")
        self.assertEqual(ui.controls["cell_name"].value, "")
        self.assertEqual(ui.controls["tune_name"].value, "orig")
        self.assertFalse((self.root / PROJECT_FILE).exists())

    def test_existing_workspaces_and_live_objects_keep_their_appropriate_actions(self):
        self.register()
        ui = self.ui()
        self.assertEqual(ui.controls["setup_action"].value, "existing")
        self.assertEqual(ui.controls["setup_manage_toggle"].description, "Cell setup")
        self.assertFalse(ui.controls["setup_manage_toggle"].value)
        self.assertEqual(ui.controls["cell_name"].value, "Cell")
        self.assertEqual(ui.controls["project_label"].value, "Synthetic_project")
        self.assertEqual(ui._selected_tune_dir(), self.tune)
        live = self.ui(cell_override=object())
        self.assertEqual(live.controls["setup_action"].value, "live")

    def test_existing_cells_without_navigation_settings_are_not_treated_as_new(self):
        self.register()
        (self.root / PROJECT_FILE).unlink()
        ui = self.ui(settings={"cell_name": "Cell", "tune_name": "orig"})
        self.assertEqual(ui.controls["setup_action"].value, "existing")

    def test_reloading_absent_settings_preserves_the_students_unsaved_fields(self):
        ui = self.ui()
        self.configure_registration(ui)
        controls = ui.controls
        controls["project_root"].value = str(self.root / "SCP")
        controls["workspace_root"].value = "work"
        controls["project_label"].value = "A1_single_cells"
        controls["output_dir"].value = "recordings"
        keys = (
            "project_root", "workspace_root", "project_label", "cell_name",
            "tune_name", "tune_dir_override", "output_dir", "setup_action",
            "setup_entry", "setup_template", "setup_voltage", "setup_celsius",
        )
        before = {key: controls[key].value for key in keys}
        files = self.files()
        self.click(ui, "setup_open")
        self.assertEqual({key: controls[key].value for key in keys}, before)
        self.assertEqual(self.files(), files)
        self.assertFalse((self.root / "SCP").exists())
        self.assertIn("no saved", controls["setup_open_feedback"].value.lower())
        self.assertFalse(ui._setup_attempted)

    def test_reloading_saved_settings_requires_confirmation_before_discarding_edits(self):
        self.register()
        ui = self.ui()
        controls = ui.controls
        files = self.files()
        controls["project_label"].value = "Unsaved_name"
        controls["output_dir"].value = "unsaved_results"
        self.click(ui, "setup_open")
        self.assertEqual(controls["project_label"].value, "Unsaved_name")
        self.assertEqual(controls["output_dir"].value, "unsaved_results")
        self.assertRegex(controls["setup_open_feedback"].value.lower(), "unsaved|discard")
        self.assertEqual(self.files(), files)
        controls["setup_discard_edits"].value = True
        self.click(ui, "setup_open")
        self.assertEqual(controls["project_label"].value, "Synthetic_project")
        self.assertEqual(controls["output_dir"].value, "")
        self.assertEqual(ui._selected_tune_dir(), self.tune)
        self.assertFalse(controls["setup_discard_edits"].value)
        self.assertRegex(controls["setup_open_feedback"].value.lower(), "reload|loaded")
        self.assertEqual(self.files(), files)

    def test_clean_saved_settings_can_be_reloaded_without_confirmation(self):
        self.register()
        ui = self.ui()
        files = self.files()
        self.click(ui, "setup_open")
        self.assertEqual(ui._status_kinds["setup_open"], "complete")
        self.assertEqual(ui._selected_tune_dir(), self.tune)
        self.assertEqual(self.files(), files)

    def test_discard_confirmation_expires_if_the_user_makes_another_edit(self):
        self.register()
        ui = self.ui()
        controls = ui.controls
        controls["project_label"].value = "First_draft"
        self.click(ui, "setup_open")
        controls["setup_discard_edits"].value = True
        controls["output_dir"].value = "newer_draft_results"
        self.assertFalse(controls["setup_discard_edits"].value)
        self.click(ui, "setup_open")
        self.assertEqual(controls["project_label"].value, "First_draft")
        self.assertEqual(controls["output_dir"].value, "newer_draft_results")
        self.assertRegex(controls["setup_open_feedback"].value.lower(), "unsaved|discard")

    def test_preview_is_nonexecuting_nonmutating_and_has_action_specific_feedback(self):
        ui = self.ui()
        self.configure_registration(ui)
        before = self.files()
        preparation_status = ui.statuses["step1"].value
        with mock.patch("scp_pipeline.model.load_cell") as load:
            self.click(ui, "setup_preview")
        load.assert_not_called()
        self.assertEqual(self.files(), before)
        self.assertFalse(self.tune.exists())
        self.assertFalse(ui._setup_attempted)
        self.assertEqual(ui._status_kinds["setup_preview"], "complete")
        self.assertEqual(ui.statuses["step1"].value, preparation_status)
        feedback = ui.controls["setup_preview_feedback"].value.lower()
        self.assertIn("preview", feedback)
        result = json.loads(self.output_text(ui, "setup_preview"))
        self.assertIn("tune_dir", result)

    def test_save_persists_navigation_without_loading_and_reopen_preserves_its_label(self):
        ui = self.ui()
        self.configure_registration(ui)
        with mock.patch("scp_pipeline.model.load_cell") as load:
            self.click(ui, "setup_save")
        load.assert_not_called()
        self.assertTrue((self.root / PROJECT_FILE).is_file())
        self.assertTrue((self.tune / "cell_configs/cell_config.json").is_file())
        self.assertFalse(ui._setup_attempted)
        self.assertEqual(ui.controls["setup_action"].value, "existing")
        self.assertIn("saved", ui.controls["setup_save_feedback"].value.lower())
        self.click(ui, "setup_open")
        self.assertEqual(ui._status_kinds["setup_open"], "complete")
        self.assertEqual(ui.controls["project_label"].value, "Synthetic_project")

    def test_newly_registered_tune_does_not_pin_later_tune_selection(self):
        ui = self.ui()
        self.configure_registration(ui)
        self.click(ui, 'setup_save')
        self.assertEqual(ui._status_kinds['setup_save'], 'complete')
        self.assertEqual(ui._selected_tune_dir(), self.tune)
        ui.controls['tune_name'].value = 'tuned'
        self.assertEqual(ui._selected_tune_dir(), self.tune.parent / 'tuned')
        self.assertFalse(ui.controls['tune_dir_override'].value)

    def test_new_names_entered_from_dropdown_register_without_loading(self):
        ui = self.ui()
        self.configure_registration(ui)
        for key, name in (('cell_name', 'NewCell'), ('tune_name', 'baseline')):
            selector = ui.controls[key]
            selector.dropdown.value = None
            selector.editor.value = name
        before = self.entry.read_bytes()
        with mock.patch('scp_pipeline.model.load_cell') as load:
            self.click(ui, 'setup_preview')
            self.assertEqual(ui._status_kinds['setup_preview'], 'complete')
            self.click(ui, 'setup_save')
        load.assert_not_called()
        self.assertEqual(ui._status_kinds['setup_save'], 'complete')
        tune = self.root / 'cells/NewCell/tunes/baseline'
        self.assertEqual(ui._selected_tune_dir(), tune)
        config = json.loads((tune / 'cell_configs/cell_config.json').read_text())
        self.assertEqual(config['cell_name'], 'NewCell')
        self.assertEqual(self.entry.read_bytes(), before)
        self.assertFalse(ui.controls['cell_name'].allow_new)
        self.assertEqual(ui.controls['cell_name'].editor.layout.display, 'none')

    def test_incomplete_first_preview_reports_a_recoverable_setup_error(self):
        ui = self.ui()
        before = self.files()
        self.click(ui, "setup_preview")
        self.assertEqual(ui._status_kinds["setup_preview"], "error")
        self.assertRegex(ui.controls["setup_preview_feedback"].value.lower(), "cell|entry|model")
        self.assertFalse(ui._setup_attempted)
        self.assertEqual(self.files(), before)
        self.configure_registration(ui)
        self.click(ui, "setup_preview")
        self.assertEqual(ui._status_kinds["setup_preview"], "complete")

    def test_restart_guard_does_not_allow_setup_actions_after_model_loading(self):
        ui = self.ui()
        self.configure_registration(ui)
        ui._setup_attempted = True
        before = self.files()
        self.click(ui, "setup_preview")
        self.assertIn("restart", ui.controls["setup_preview_feedback"].value.lower())
        self.assertEqual(self.files(), before)

    def test_empty_existing_selection_explains_how_to_register_a_model(self):
        ui = self.ui()
        ui.controls["setup_action"].value = "existing"
        before = self.files()
        self.click(ui, "setup_preview")
        self.assertEqual(ui._status_kinds["setup_preview"], "error")
        feedback = ui.controls["setup_preview_feedback"].value.lower()
        self.assertIn("workspace", feedback)
        self.assertIn("register", feedback)
        self.assertFalse(ui._setup_attempted)
        self.assertEqual(self.files(), before)

    def test_fields_follow_the_setup_sequence_and_explain_their_meanings(self):
        ui = self.ui()
        nodes = list(all_descendants(ui.step1_panel()))
        controls = ui.controls
        visible = dict(controls, cell_name=controls['cell_name'].dropdown,
                       tune_name=controls['tune_name'].dropdown)
        for first, second in (
            ("project_root", "setup_open"),
            ("setup_open", "project_label"),
            ("cell_name", "setup_entry"),
            ("output_dir", "setup_entry"),
            ("setup_entry", "setup_preview"),
            ("setup_preview", "setup_save"),
            ("setup_save", "step1_run"),
        ):
            self.assertLess(nodes.index(visible[first]), nodes.index(visible[second]))
        for name in ("project_root", "workspace_root", "project_label", "cell_name",
                     "tune_name", "tune_dir_override", "output_dir", "setup_open",
                     "setup_preview", "setup_save"):
            self.assertTrue(visible[name].tooltip, name)
        html = "\n".join(str(getattr(widget, "value", "")) for widget in nodes
                         if isinstance(widget, ui.widgets.HTML))
        self.assertIn("<details", html.lower())
        self.assertIn("<summary", html.lower())
        self.assertIn(".scp_project.json", html)
        self.assertNotIn("setup_feedback", controls)

    def test_field_guide_is_first_and_only_its_title_is_emphasized(self):
        ui = self.ui()
        controls = ui.controls
        setup_panel = ui.step1_panel().children[0]
        self.assertIs(setup_panel.children[0], controls['setup_help'])
        nodes = list(all_descendants(setup_panel))
        self.assertLess(nodes.index(controls['setup_help']), nodes.index(controls['setup_location_summary']))
        html = controls['setup_help'].value
        self.assertTrue(html.startswith('<details><summary>'))
        self.assertRegex(html, r'<strong[^>]*font-size:1\.1em[^>]*>Field guide</strong>')
        self.assertIn('</strong>: locations, labels, and model sources</summary>', html)

    def test_main_view_contains_all_navigation_fields_and_one_save_button(self):
        ui = self.ui()
        controls = ui.controls
        controls['setup_manage_toggle'].value = False
        daily = list(all_descendants(controls['setup_daily_box']))
        manager = list(all_descendants(controls['setup_manager_box']))
        for name in ('workspace_mode', 'project_root', 'workspace_root', 'project_label',
                     'output_dir', 'setup_project_save', 'setup_open'):
            self.assertIn(controls[name], daily)
            self.assertNotIn(controls[name], manager)
            self.assertNotEqual(controls[name].layout.display, 'none')
        self.assertNotEqual(controls['setup_project_details_box'].layout.display, 'none')
        self.assertEqual(controls['project_name'].layout.display, 'none')
        self.assertLess(daily.index(controls['workspace_mode']), daily.index(controls['cell_name'].dropdown))
        self.assertLess(daily.index(controls['tune_name'].dropdown), daily.index(controls['output_dir']))
        self.assertLess(daily.index(controls['output_dir']), daily.index(controls['setup_project_save']))
        nodes = list(all_descendants(ui.step1_panel()))
        saves = [node for node in nodes if isinstance(node, ui.widgets.Button)
                 and node.description in ('Save project settings', 'Remember selection')]
        self.assertEqual(saves, [controls['setup_project_save']])
        self.assertIs(controls['setup_remember'], controls['setup_project_save'])
        self.assertIn(controls['setup_entry'], manager)
        self.assertIn(controls['setup_save'], manager)

    def test_location_mode_reveals_project_details_without_opening_management(self):
        self.register()
        ui = PipelineNotebookUI(self.root)
        ui.step1_panel()
        controls = ui.controls
        self.assertFalse(controls['setup_manage_toggle'].value)
        self.assertEqual(controls['setup_project_details_box'].layout.display, 'none')
        self.assertNotEqual(controls['project_name'].layout.display, 'none')
        controls['workspace_mode'].value = 'project'
        self.assertNotEqual(controls['setup_project_details_box'].layout.display, 'none')
        self.assertEqual(controls['project_name'].layout.display, 'none')
        self.assertFalse(controls['setup_manage_toggle'].value)
        controls['workspace_mode'].value = 'scp'
        self.assertEqual(controls['setup_project_details_box'].layout.display, 'none')
        self.assertNotEqual(controls['project_name'].layout.display, 'none')
        self.assertNotEqual(controls['output_dir'].layout.display, 'none')

    def test_main_save_persists_navigation_once_before_any_model_exists(self):
        from scp_pipeline.notebooks._pipeline_ui.project_setup import save_navigation
        ui = self.ui()
        controls = ui.controls
        controls['setup_manage_toggle'].value = False
        controls['project_label'].value = 'Example_project'
        controls['workspace_root'].value = 'work'
        controls['cell_name'].value = 'NewCell'
        controls['tune_name'].value = 'orig'
        controls['output_dir'].value = 'recordings'
        source = self.entry.read_bytes()
        with mock.patch('scp_pipeline.model.load_cell') as load, mock.patch(
                'scp_pipeline.notebooks._pipeline_ui.project_setup.save_navigation', wraps=save_navigation) as save:
            self.click(ui, 'setup_project_save')
        save.assert_called_once()
        load.assert_not_called()
        self.assertEqual(ui._status_kinds['setup_project_save'], 'complete')
        saved = json.loads((self.root / PROJECT_FILE).read_text())
        self.assertEqual(saved['project_name'], 'Example_project')
        self.assertEqual(saved['workspace_root'], 'work')
        self.assertEqual(saved['selected_tune'], 'work/cells/NewCell/tunes/orig')
        self.assertEqual(saved['output_dir'], 'recordings')
        self.assertEqual(self.entry.read_bytes(), source)
        self.assertFalse((self.root / 'work').exists())
        self.assertFalse(ui._setup_attempted)

    def test_results_update_preserves_scientific_files_and_other_navigation(self):
        self.register()
        before = self.files()
        ui = self.ui()
        ui.controls['setup_manage_toggle'].value = False
        ui.controls['output_dir'].value = 'recordings'
        self.assertIn(str(self.root / 'recordings'), ui.controls['setup_model_summary'].value)
        self.assertEqual(self.files(), before, 'Editing the Results field must not write files.')
        self.click(ui, 'setup_project_save')
        self.assertEqual(ui._status_kinds['setup_project_save'], 'complete')
        after = self.files()
        self.assertEqual(set(after), set(before))
        self.assertEqual([name for name in before if before[name] != after[name]], [PROJECT_FILE])
        saved = json.loads((self.root / PROJECT_FILE).read_text())
        self.assertEqual(saved['selected_tune'], 'cells/Cell/tunes/orig')
        self.assertEqual(saved['output_dir'], 'recordings')
        self.assertEqual(saved['project_name'], 'Synthetic_project')

    def test_project_save_feedback_stays_outside_collapsed_model_management(self):
        ui = self.ui()
        self.click(ui, 'setup_project_save')
        saved = self.action_snapshot(ui, 'setup_project_save')
        ui.controls['setup_manage_toggle'].value = True
        self.click(ui, 'setup_preview')  # Incomplete model setup has its own error.
        ui.controls['setup_manage_toggle'].value = False
        self.assertEqual(self.action_snapshot(ui, 'setup_project_save'), saved)
        daily = list(all_descendants(ui.controls['setup_daily_box']))
        self.assertIn(ui.outputs['setup_project_save'], daily)
        self.assertIn(ui.statuses['setup_project_save'], daily)
        self.assertNotEqual(ui.controls['setup_daily_box'].layout.display, 'none')

    def test_each_action_owns_feedback_and_output_beside_its_button(self):
        ui = self.ui()
        controls = ui.controls
        nodes = list(all_descendants(ui.step1_panel()))
        outputs = []
        for name in ("setup_open", "setup_discover", "setup_preview", "setup_save"):
            box = controls[f"{name}_box"]
            local = list(all_descendants(box))
            feedback = controls[f"{name}_feedback"]
            output = ui.outputs[name]
            self.assertIs(feedback, ui.statuses[name])
            self.assertLess(local.index(controls[name]), local.index(feedback))
            self.assertLess(local.index(feedback), local.index(output))
            self.assertNotIn(ui.outputs["step1"], local)
            self.assertIsNot(output, ui.outputs["step1"])
            outputs.append(output)
        self.assertEqual(len({id(output) for output in outputs}), 4)
        self.assertLess(nodes.index(controls["project_root"]), nodes.index(controls["setup_open_box"]))
        self.assertLess(nodes.index(controls["setup_entry"]), nodes.index(controls["setup_discover_box"]))
        self.assertLess(nodes.index(controls["setup_open_box"]), nodes.index(controls["project_label"]))
        self.assertLess(nodes.index(ui.outputs["setup_discover"]), nodes.index(controls["setup_template"]))
        self.assertLess(nodes.index(ui.outputs["setup_preview"]), nodes.index(controls["setup_save_box"]))
        self.assertLess(nodes.index(ui.outputs["setup_save"]), nodes.index(controls["step1_run"]))

    def test_other_actions_preserve_previous_success_feedback_and_json(self):
        ui = self.ui()
        self.configure_registration(ui)
        self.click(ui, "setup_open")
        opened = self.action_snapshot(ui, "setup_open")
        self.assertIn("not_found", self.output_text(ui, "setup_open"))
        self.click(ui, "setup_discover")
        discovered = self.action_snapshot(ui, "setup_discover")
        self.assertIn("SetupUICell", self.output_text(ui, "setup_discover"))
        self.click(ui, "setup_preview")
        previewed = self.action_snapshot(ui, "setup_preview")
        self.assertIn("tune_dir", self.output_text(ui, "setup_preview"))
        self.click(ui, "setup_save")
        self.assertEqual(self.action_snapshot(ui, "setup_open"), opened)
        self.assertEqual(self.action_snapshot(ui, "setup_discover"), discovered)
        self.assertEqual(self.action_snapshot(ui, "setup_preview"), previewed)
        self.assertIn("tune_dir", self.output_text(ui, "setup_save"))
        self.assertEqual(ui._status_kinds["step1"], "waiting")

    def test_repeating_action_replaces_only_its_previous_output(self):
        ui = self.ui()
        self.configure_registration(ui)
        self.click(ui, "setup_discover")
        discovered = self.action_snapshot(ui, "setup_discover")
        self.click(ui, "setup_preview")
        old_output = self.output_text(ui, "setup_preview")
        self.assertIn("orig", old_output)
        ui.outputs["setup_preview"].append_stdout("stale feedback marker\n")
        ui.controls["tune_name"].value = "working"
        self.click(ui, "setup_preview")
        self.assertIn("working", self.output_text(ui, "setup_preview"))
        self.assertNotIn("stale feedback marker", self.output_text(ui, "setup_preview"))
        self.assertEqual(self.action_snapshot(ui, "setup_discover"), discovered)

    def test_discovery_recovers_from_missing_entry_in_a_live_output_context(self):
        ui = self.ui()
        before = self.files()
        shell = mock.Mock()
        shell.kernel.get_parent.return_value = {'header': {'msg_id': 'discovery-regression'}}
        with mock.patch('ipywidgets.widgets.widget_output.get_ipython', return_value=shell), \
                mock.patch('scp_pipeline.model.load_cell') as load:
            for value, expected in (('', 'Entry file'), (str(self.root), 'file'),
                                    (str(self.root / 'missing.hoc'), 'missing.hoc')):
                ui.controls['setup_entry'].value = value
                self.click(ui, 'setup_discover')
                self.assertEqual(ui._status_kinds['setup_discover'], 'error')
                output = self.output_text(ui, 'setup_discover')
                self.assertIn(expected, output)
                self.assertNotIn('UnboundLocalError', output)
                self.assertFalse(ui._setup_attempted)
            ui.controls['setup_entry'].value = str(self.entry)
            for _ in range(2):
                self.click(ui, 'setup_discover')
                self.assertEqual(ui._status_kinds['setup_discover'], 'complete')
                result = json.loads(self.output_text(ui, 'setup_discover'))
                self.assertEqual(result['templates'], ['SetupUICell'])
                self.assertFalse(result['executed_model_code'])
                self.assertEqual(ui.controls['setup_template'].value, 'SetupUICell')
                self.assertFalse(ui.controls['setup_discover'].disabled)
        shell.showtraceback.assert_not_called()
        load.assert_not_called()
        self.assertEqual(self.files(), before)

    def test_every_setup_action_handles_errors_before_ipython_can_suppress_them(self):
        ui = self.ui()
        shell = mock.Mock()
        shell.kernel.get_parent.return_value = {'header': {'msg_id': 'setup-action-regression'}}
        actions = {'setup_open': 'apply_location', 'setup_discover': 'discover',
                   'setup_preview': 'preview', 'setup_save': 'save',
                   'setup_project_save': 'save_navigation'}
        before = self.files()
        with mock.patch('ipywidgets.widgets.widget_output.get_ipython', return_value=shell):
            for key, function in actions.items():
                with self.subTest(action=key):
                    with mock.patch('scp_pipeline.notebooks._pipeline_ui.project_setup.' + function,
                                    side_effect=ValueError('Correct this input and retry')):
                        self.click(ui, key)
                    self.assertEqual(ui._status_kinds[key], 'error')
                    self.assertIn('ValueError: Correct this input and retry', self.output_text(ui, key))
                    self.assertNotIn('UnboundLocalError', self.output_text(ui, key))
                    with mock.patch('scp_pipeline.notebooks._pipeline_ui.project_setup.' + function,
                                    return_value={'status': 'ok', 'message': 'Retry succeeded'}):
                        self.click(ui, key)
                    self.assertEqual(ui._status_kinds[key], 'complete')
                    self.assertEqual(json.loads(self.output_text(ui, key))['status'], 'ok')
                    self.assertFalse(ui._setup_attempted)
        shell.showtraceback.assert_not_called()
        self.assertEqual(self.files(), before)

    def test_completed_discovery_stays_visible_after_save_and_loader_selection_changes(self):
        ui = self.ui()
        self.configure_registration(ui)
        panel = ui.step1_panel()

        def visible_path(target):
            def find_path(widget):
                if widget is target:
                    return [widget]
                for child in getattr(widget, "children", ()):
                    path = find_path(child)
                    if path is not None:
                        return [widget, *path]
                return None

            path = find_path(panel)
            self.assertIsNotNone(path, "The action result must remain in the displayed panel")
            for ancestor in path:
                self.assertNotEqual(
                    ancestor.layout.display, "none",
                    f"Discovery result hidden by {type(ancestor).__name__}",
                )

        self.click(ui, "setup_discover")
        discovered = self.action_snapshot(ui, "setup_discover")
        self.click(ui, "setup_save")
        self.assertEqual(ui.controls["setup_action"].value, "existing")
        for widget in (ui.controls["setup_discover_feedback"], ui.outputs["setup_discover"]):
            visible_path(widget)
        self.assertEqual(self.action_snapshot(ui, "setup_discover"), discovered)
        ui.controls["setup_action"].value = "register"
        ui.controls["setup_loader"].value = "python_factory"
        for widget in (ui.controls["setup_discover_feedback"], ui.outputs["setup_discover"]):
            visible_path(widget)
        self.assertEqual(self.action_snapshot(ui, "setup_discover"), discovered)

    def test_error_and_recovery_are_local_and_do_not_erase_other_action_results(self):
        ui = self.ui()
        self.configure_registration(ui)
        self.click(ui, "setup_discover")
        discovered = self.action_snapshot(ui, "setup_discover")
        ui.controls["setup_voltage"].value = "not-a-voltage"
        self.click(ui, "setup_preview")
        self.assertEqual(ui._status_kinds["setup_preview"], "error")
        self.assertIn("ValueError", self.output_text(ui, "setup_preview"))
        failed_preview = self.action_snapshot(ui, "setup_preview")
        self.click(ui, "setup_open")
        self.assertEqual(self.action_snapshot(ui, "setup_preview"), failed_preview)
        self.assertEqual(self.action_snapshot(ui, "setup_discover"), discovered)
        ui.controls["setup_voltage"].value = "-65"
        self.click(ui, "setup_preview")
        self.assertEqual(ui._status_kinds["setup_preview"], "complete")
        self.assertNotIn("ValueError", self.output_text(ui, "setup_preview"))
        self.assertEqual(self.action_snapshot(ui, "setup_discover"), discovered)

    def test_setup_success_errors_and_restart_guard_preserve_preparation_output(self):
        ui = self.ui()
        self.configure_registration(ui)
        ui.outputs["step1"].append_stdout("Existing model preparation log\n")
        ui.step1_load_log = "Detailed preparation notes"
        prepared = (ui.outputs["step1"].outputs, ui.statuses["step1"].value,
                    ui._status_kinds["step1"], ui.step1_load_log)
        self.click(ui, "setup_preview")
        previewed = self.action_snapshot(ui, "setup_preview")
        ui.controls["setup_voltage"].value = "invalid"
        self.click(ui, "setup_save")
        self.assertEqual(ui._status_kinds["setup_save"], "error")
        self.assertEqual(self.action_snapshot(ui, "setup_preview"), previewed)
        ui._setup_attempted = True
        for name in ("setup_open", "setup_discover", "setup_save"):
            self.click(ui, name)
            self.assertEqual(ui._status_kinds[name], "error")
            self.assertIn("restart", self.output_text(ui, name).lower())
            self.assertEqual(self.action_snapshot(ui, "setup_preview"), previewed)
        self.assertEqual(
            (ui.outputs["step1"].outputs, ui.statuses["step1"].value,
             ui._status_kinds["step1"], ui.step1_load_log), prepared)


if __name__ == "__main__":
    unittest.main()
