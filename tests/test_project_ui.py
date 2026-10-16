"""Namespace selection tests use disposable workspaces, not production cells."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from modules.analysis import analysis
from modules.analysis.ui import _engine, extra, selection, state
from modules.notebooks.pipeline_ui import PipelineNotebookUI
from tests.pipeline_ui_test_support import write_test_tune


def make_workspaces(root):
    (root / "run_pipeline.py").write_text("", encoding="utf-8")
    locations = {}
    for project, cells in ((None, {"PV": ("orig", "tuned")}),
                           ("alpha", {"PV": ("tuned",), "OnlyAlpha": ("orig",)}),
                           ("beta", {"PV": ("orig",), "OnlyBeta": ("custom",)})):
        base = root / "cells" if project is None else root / "projects" / project / "cells"
        for cell, tunes in cells.items():
            for tune in tunes:
                path = base / cell / "tunes" / tune
                run = path / "output_data" / "same_run"
                run.mkdir(parents=True)
                (run / "run_manifest.json").write_text("{}", encoding="utf-8")
                locations[(project, cell, tune)] = path
    return locations


def analysis_globals(root, project=None):
    return {"BASE_DIR": root, "CELLS_DIR": root / "cells", "project_name": project,
            "cell_name": "PV", "tunes_dir": "tunes", "model_dir": "tuned",
            "run_single_stem": "latest", "use_widgets": False}


class ProjectUITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.locations = make_workspaces(self.root)

    def test_compact_defaults_and_scope_isolation(self):
        settings = {"cell_name": "PV", "tune_name": "tuned"}
        ui = PipelineNotebookUI(self.root, settings)
        ui.step1_panel()
        self.assertIsNone(settings["project_name"])
        self.assertEqual(ui.controls["project_name"].value, "")
        self.assertEqual(tuple(ui.controls["cell_name"].options), ("PV",))
        self.assertEqual(ui._selected_tune_dir(), self.locations[(None, "PV", "tuned")])
        ui.controls["project_name"].value = "alpha"
        self.assertEqual(settings["project_name"], "alpha")
        self.assertEqual(set(ui.controls["cell_name"].options), {"PV", "OnlyAlpha"})
        self.assertEqual(ui._selected_tune_dir(), self.locations[("alpha", "PV", "tuned")])
        ui.controls["project_name"].value = "beta"
        self.assertEqual(set(ui.controls["cell_name"].options), {"PV", "OnlyBeta"})
        self.assertEqual(tuple(ui.controls["tune_name"].options), ("orig",))
        self.assertEqual(ui._selected_tune_dir(), self.locations[("beta", "PV", "orig")])

    def test_compact_prepare_forwards_and_locks_project(self):
        write_test_tune(self.locations[("alpha", "PV", "tuned")], cell='PV', tune='tuned')
        settings = {"project_name": "alpha", "cell_name": "PV", "tune_name": "tuned"}
        ui = PipelineNotebookUI(self.root, settings)
        ui.step1_panel()
        context = SimpleNamespace(project_name="alpha", cell_name="PV", tune_name="tuned",
                                  cell_config={}, sim_config={})
        prepared = SimpleNamespace(tune_dir=self.locations[("alpha", "PV", "tuned")],
                                   context=context, cell=object())
        with mock.patch("modules.notebooks._pipeline_ui.step1.prepare_pipeline_notebook", return_value=prepared) as prepare, redirect_stdout(StringIO()):
            ui.controls["step1_run"].click()
        self.assertEqual(prepare.call_args.kwargs["project_name"], "alpha")
        self.assertTrue(ui.controls["project_name"].disabled)
        changed = dict(settings, project_name="beta")
        ui.apply_settings(changed)
        self.assertEqual(changed["project_name"], "alpha")

    def test_compact_explicit_workspace_wins(self):
        tune = self.locations[("beta", "PV", "orig")]
        ui = PipelineNotebookUI(self.root, {"project_name": "alpha", "cell_name": "PV",
                                           "tune_name": "tuned", "tune_dir_override": str(tune)})
        self.assertEqual(ui._selected_tune_dir(), tune)

    def test_compact_code_edited_project_cell_and_tune_remain_authoritative(self):
        ui = PipelineNotebookUI(self.root, {"cell_name": "PV", "tune_name": "tuned"})
        ui.step1_panel()
        edited = {"project_name": "beta", "cell_name": "OnlyBeta", "tune_name": "custom"}
        ui.apply_settings(edited)
        self.assertEqual(ui.controls["project_name"].value, "beta")
        self.assertEqual(ui.controls["cell_name"].value, "OnlyBeta")
        self.assertEqual(ui.controls["tune_name"].value, "custom")
        self.assertEqual(ui._selected_tune_dir(), self.locations[("beta", "OnlyBeta", "custom")])

    def test_analysis_discovery_does_not_flatten_scopes(self):
        self.assertEqual(analysis.list_projects(self.root), ["alpha", "beta"])
        self.assertEqual(analysis.list_cells(self.root), ["PV"])
        self.assertEqual(analysis.list_cells(self.root, "alpha"), ["OnlyAlpha", "PV"])
        self.assertEqual(analysis.list_tunes(self.root, "PV", "beta"), ["tunes"])
        self.assertEqual(analysis.list_models(self.root, "PV", "tunes", "alpha"), ["tuned"])
        self.assertEqual(analysis.list_models(self.root, "PV", "tunes", "beta"), ["orig"])

    def test_analysis_manual_and_override_paths(self):
        g = analysis_globals(self.root, "alpha")
        selected = state.get_selection_from_globals(g)
        self.assertEqual(selected["tune_dir"], self.locations[("alpha", "PV", "tuned")])
        self.assertEqual(selected["project_name"], "alpha")
        g["tune_dir_override"] = "projects/beta/cells/PV/tunes/orig"
        selected = state.get_selection_from_globals(g)
        self.assertEqual(selected["tune_dir"], self.locations[("beta", "PV", "orig")])
        self.assertEqual(selected["project_name"], "beta")
        self.assertEqual(selected["model"], "orig")

    def test_analysis_widget_changes_scope_without_changing_run_tokens(self):
        g = dict(analysis_globals(self.root), use_widgets=True, _HAVE_WIDGETS=True)
        with mock.patch("modules.analysis.ui.selection._maybe_import_display", return_value=(lambda *args: None, None)):
            selection.build_selection_ui(g)
        self.assertEqual(g["project_dd"].value, "")
        g["project_dd"].value = "alpha"
        self.assertEqual(set(g["cell_dd"].options), {"OnlyAlpha", "PV"})
        self.assertEqual(g["cell_dd"].value, "PV")
        self.assertEqual(state.get_selection_from_globals(g)["project_name"], "alpha")
        self.assertEqual(g["compare_list_sel"].options, (("same_run", "same_run"),))
        g["project_dd"].value = "beta"
        self.assertEqual(set(g["cell_dd"].options), {"OnlyBeta", "PV"})
        self.assertEqual(g["model_dd"].value, "orig")
        self.assertEqual(state.get_selection_from_globals(g)["tune_dir"], self.locations[("beta", "PV", "orig")])

    def test_analysis_empty_project_is_browseable(self):
        (self.root / "projects" / "empty").mkdir()
        g = dict(analysis_globals(self.root), use_widgets=True, _HAVE_WIDGETS=True)
        with mock.patch("modules.analysis.ui.selection._maybe_import_display", return_value=(lambda *args: None, None)):
            selection.build_selection_ui(g)
        g["project_dd"].value = "empty"
        self.assertEqual(g["compare_list_sel"].options, ())

    def test_project_comparison_paths_are_repository_relative(self):
        base = self.locations[("alpha", "PV", "tuned")] / "output_data"
        token = "projects/beta/cells/PV/tunes/orig/output_data/same_run"
        self.assertEqual(_engine._coerce_run_path(token, base), self.root / token)

    def test_project_display_labels_preserve_legacy_tokens(self):
        legacy = self.locations[(None, "PV", "tuned")] / "output_data" / "same_run"
        alpha = self.locations[("alpha", "PV", "tuned")] / "output_data" / "same_run"
        beta = self.locations[("beta", "PV", "orig")] / "output_data" / "same_run"
        for project, tune, run in (("alpha", "tuned", alpha), ("beta", "orig", beta)):
            (run / "sim_cfg.json").write_text(json.dumps({
                "tune_dir": str(self.locations[(project, "PV", tune)])
            }), encoding="utf-8")
        self.assertEqual(analysis.run_label(alpha), "same_run")
        self.assertEqual(analysis.comparison_run_label(legacy), "same_run")
        self.assertEqual(analysis.comparison_run_label(alpha), "alpha__PV__tuned__same_run")
        self.assertNotEqual(analysis.comparison_run_label(alpha), analysis.comparison_run_label(beta))
        self.assertNotIn("/", analysis.plot_dir_for_compare(legacy.parent, alpha, beta).parent.name)

    def test_exported_run_metadata_preserves_project_label(self):
        run = self.root / "exported" / "same_run"
        run.mkdir(parents=True)
        (run / "run_manifest.json").write_text("{}", encoding="utf-8")
        (run / "sim_cfg.json").write_text(json.dumps({"project_name": "alpha", "cell": "PV", "tune": "tuned"}), encoding="utf-8")
        self.assertEqual(analysis.comparison_run_label(run), "alpha__PV__tuned__same_run")

    def test_legacy_run_copied_into_project_keeps_saved_identity(self):
        run = self.locations[("alpha", "PV", "tuned")] / "output_data" / "same_run"
        (run / "sim_cfg.json").write_text(json.dumps({"cell": "PV", "tune": "tuned", "project_name": None}), encoding="utf-8")
        self.assertEqual(analysis.comparison_run_label(run), "same_run")

    def test_metadata_poor_legacy_run_is_not_assigned_current_project(self):
        run = self.locations[("alpha", "PV", "tuned")] / "output_data" / "same_run"
        # Even a current cell config cannot establish this historical run's identity.
        config_dir = run.parent.parent / "cell_configs"
        config_dir.mkdir()
        (config_dir / "cell_config.json").write_text('{"project_name": "alpha"}', encoding="utf-8")
        self.assertEqual(analysis.comparison_run_label(run), "same_run")

    def test_exported_manifest_identity_survives_without_sidecar(self):
        run = self.root / "exported" / "same_run"
        run.mkdir(parents=True)
        (run / "run_manifest.json").write_text(json.dumps({"workspace_identity": {
            "project_name": "alpha", "cell_name": "PV", "tune_name": "tuned"
        }}), encoding="utf-8")
        self.assertEqual(analysis.comparison_run_label(run), "alpha__PV__tuned__same_run")

    def test_manifest_recorded_path_not_current_location_defines_label(self):
        run = self.locations[("beta", "PV", "orig")] / "output_data" / "same_run"
        (run / "run_manifest.json").write_text(json.dumps({"sim_cfg": {
            "tune_dir": "/old/checkout/projects/alpha/cells/PV/tunes/tuned"
        }}), encoding="utf-8")
        self.assertEqual(analysis.comparison_run_label(run), "alpha__PV__tuned__same_run")

    def test_saved_meta_identity_is_used_without_construction_or_path_inference(self):
        run = self.root / "exported" / "same_run"
        run.mkdir(parents=True)
        (run / "run_manifest.json").write_text("{}", encoding="utf-8")
        (run / "meta.json").write_text(json.dumps({"workspace_identity": {
            "project_name": "alpha", "cell_name": "PV", "tune_name": "tuned"
        }}), encoding="utf-8")
        self.assertEqual(analysis.comparison_run_label(run), "alpha__PV__tuned__same_run")

    def test_conflicting_manifest_and_sidecar_identity_is_not_hidden(self):
        run = self.locations[("alpha", "PV", "tuned")] / "output_data" / "same_run"
        (run / "run_manifest.json").write_text(json.dumps({"workspace_identity": {
            "project_name": "alpha", "cell_name": "PV", "tune_name": "tuned"
        }}), encoding="utf-8")
        (run / "sim_cfg.json").write_text(json.dumps({
            "project_name": "beta", "cell": "PV", "tune": "tuned"
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            analysis.comparison_run_label(run)

    def test_input_sampling_uses_effective_selected_workspace(self):
        g = analysis_globals(self.root, "alpha")
        g.update(input_sample_groups=["synthetic"], input_sample_show_ref=False)
        with mock.patch("modules.analysis.ui.extra.input_sampling.sample_group_rates_from_path", side_effect=RuntimeError("synthetic stop")) as sample, mock.patch("matplotlib.pyplot.show"), redirect_stdout(StringIO()):
            extra.run_input_sampling_from_globals(g)
        self.assertEqual(sample.call_args.args[0], self.locations[("alpha", "PV", "tuned")])

    def test_sampling_widget_initial_and_refreshed_paths_follow_project(self):
        g = dict(analysis_globals(self.root), use_widgets=True, _HAVE_WIDGETS=True,
                 extra_mode="input_sampling", input_sample_source="selection")
        displayed = []
        with mock.patch("modules.analysis.ui.selection._maybe_import_display", return_value=(lambda *args: None, None)):
            selection.build_selection_ui(g)
        with mock.patch("modules.analysis.ui.extra._maybe_import_display", return_value=(displayed.append, None)), mock.patch("modules.analysis.ui.extra._list_groups_for_sampling", return_value=["synthetic"]) as groups, mock.patch("modules.analysis.ui.extra.run_input_sampling_from_globals"):
            extra.build_extra_ui(g)
            self.assertEqual(groups.call_args.args[0], self.locations[(None, "PV", "tuned")])
            g["project_dd"].value = "alpha"
            self.assertEqual(groups.call_args.args[0], self.locations[("alpha", "PV", "tuned")])
            def descendants(widget):
                yield widget
                for child in getattr(widget, "children", ()):
                    yield from descendants(child)
            button = next(widget for panel in displayed for widget in descendants(panel)
                          if getattr(widget, "description", "") == "Run extra analysis")
            with redirect_stdout(StringIO()):
                button.click()
            self.assertEqual(groups.call_args.args[0], self.locations[("alpha", "PV", "tuned")])

    def test_model_inspection_keeps_selected_workspace_and_override(self):
        g = analysis_globals(self.root, "alpha")
        g.update(load_cell_for_analysis=True, extra_cell_tables=True, CELL_OVERRIDE=object())
        selected = state.get_selection_from_globals(g)
        run = selected["base"] / "same_run"
        with mock.patch("modules.analysis.ui.extra.resolve_single_from_globals", return_value=(selected, run, {})), mock.patch("modules.analysis.ui.extra.analysis.load_cell_and_geometry", return_value=(None, {}, {})) as load:
            extra.run_extra_tables_from_globals(g)
        load.assert_called_once_with(selected["tune_dir"], cell_override=g["CELL_OVERRIDE"])


if __name__ == "__main__":
    unittest.main()
