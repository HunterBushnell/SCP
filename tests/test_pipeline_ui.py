from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from modules.notebooks.pipeline_ui import (
    PipelineNotebookUI,
    _parse_float_list,
    _parse_optional_int,
)
from modules.notebooks.pipeline_workflow import RestartKernelRequired


def _make_repo(parent: Path) -> Path:
    root = parent / "repo"
    for cell, tunes in {"A": ("orig", "tuned"), "B": ("custom",)}.items():
        for tune in tunes:
            (root / "cells" / cell / "tunes" / tune).mkdir(parents=True)
    return root


def _fake_state(root: Path, *, cell: str = "A", tune: str = "tuned") -> SimpleNamespace:
    tune_dir = root / "cells" / cell / "tunes" / tune
    return SimpleNamespace(
        repo_root=root,
        tune_dir=tune_dir,
        context=SimpleNamespace(cell_name=cell, tune_name=tune),
        cell=object(),
    )


def _all_descendants(widget):
    yield widget
    for child in getattr(widget, "children", ()):
        yield from _all_descendants(child)


class PipelineNotebookUITests(unittest.TestCase):
    def test_parsers_accept_lists_and_report_invalid_values(self) -> None:
        self.assertEqual(_parse_float_list("-50, -100", name="amps"), [-50.0, -100.0])
        self.assertEqual(_parse_float_list("[0, 50]", name="amps"), [0.0, 50.0])
        self.assertIsNone(_parse_optional_int("", name="seed"))
        self.assertEqual(_parse_optional_int("17", name="seed"), 17)
        with self.assertRaisesRegex(ValueError, "invalid number"):
            _parse_float_list("0, nope", name="amps")
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            _parse_optional_int("0", name="ADB ID", positive=True)

    def test_panels_are_independent_and_refresh_tunes_without_tabs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_repo(Path(tmp))
            settings = {"cell_name": "A", "tune_name": "orig"}
            ui = PipelineNotebookUI(root, settings)
            panels = [
                ui.step1_panel(),
                ui.step2_panel(),
                ui.step3_panel(),
                ui.step4_panel(),
                ui.step5_panel(),
            ]

            self.assertEqual(len({id(panel) for panel in panels}), 5)
            self.assertTrue(all(type(panel).__name__ == "VBox" for panel in panels))
            forbidden = (ui.widgets.Accordion, ui.widgets.Tab)
            self.assertFalse(
                any(
                    isinstance(widget, forbidden)
                    for panel in panels
                    for widget in _all_descendants(panel)
                )
            )
            self.assertIn("A", ui.controls["cell_name"].options)
            ui.controls["cell_name"].value = "B"
            self.assertEqual(tuple(ui.controls["tune_name"].options), ("custom",))
            self.assertEqual(ui.controls["tune_name"].value, "custom")
            self.assertTrue(ui.controls["step2_run"].disabled)
            self.assertTrue(ui.controls["step5_run"].disabled)

    def test_two_way_settings_sync_and_model_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_repo(Path(tmp))
            settings = {"cell_name": "A", "tune_name": "tuned"}
            ui = PipelineNotebookUI(root, settings)
            ui.step1_panel()
            ui.step2_panel()
            ui.step3_panel()
            state = _fake_state(root)

            ui.controls["passive_amps_pA"].value = "-25, -75"
            self.assertEqual(settings["passive_amps_pA"], [-25.0, -75.0])

            with (
                mock.patch(
                    "modules.notebooks.pipeline_ui.prepare_pipeline_notebook",
                    return_value=state,
                ) as prepare,
                mock.patch("modules.notebooks.pipeline_ui._clear_output"),
            ):
                ui.controls["step1_run"].click()
                ui.controls["step1_run"].click()

            self.assertEqual(prepare.call_count, 1)
            self.assertTrue(ui.controls["cell_name"].disabled)
            self.assertFalse(ui.controls["step2_run"].disabled)
            refreshed = {
                **settings,
                "cell_name": "B",
                "tune_name": "custom",
                "passive_amps_pA": [-10],
                "active_amps_pA": [125],
            }
            ui.apply_settings(refreshed)
            self.assertEqual(refreshed["cell_name"], "A")
            self.assertEqual(refreshed["tune_name"], "tuned")
            self.assertEqual(ui.controls["passive_amps_pA"].value, "-10")
            self.assertEqual(ui.controls["active_amps_pA"].value, "125")
            self.assertIn("Model selection is locked", ui.statuses["step1"].value)

    def test_setup_validation_can_retry_but_attempt_failure_requires_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_repo(Path(tmp))
            ui = PipelineNotebookUI(root, {"cell_name": "A", "tune_name": "tuned"})
            ui.step1_panel()
            ui.controls["adb_specimen_id"].value = "invalid"
            with (
                mock.patch(
                    "modules.notebooks.pipeline_ui.prepare_pipeline_notebook"
                ) as prepare,
                mock.patch("modules.notebooks.pipeline_ui._clear_output"),
                mock.patch("builtins.print"),
            ):
                ui.controls["step1_run"].click()
            prepare.assert_not_called()
            self.assertFalse(ui.controls["step1_run"].disabled)

            ui.controls["adb_specimen_id"].value = ""
            with (
                mock.patch(
                    "modules.notebooks.pipeline_ui.prepare_pipeline_notebook",
                    side_effect=RuntimeError("model build failed"),
                ) as prepare,
                mock.patch("modules.notebooks.pipeline_ui._clear_output"),
                mock.patch("modules.notebooks.pipeline_ui.traceback.print_exc"),
                mock.patch("builtins.print"),
            ):
                ui.controls["step1_run"].click()
                ui.controls["step1_run"].click()
            self.assertEqual(prepare.call_count, 1)
            self.assertTrue(ui.controls["step1_run"].disabled)
            self.assertIn("restart the kernel", ui.statuses["step1"].value.lower())

    def test_passive_and_active_callbacks_reuse_step1_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_repo(Path(tmp))
            ui = PipelineNotebookUI(
                root,
                {
                    "cell_name": "A",
                    "tune_name": "tuned",
                    "passive_protocol_overrides": {"h_dt": 0.1},
                    "active_protocol_overrides": {"stim_dur": 750.0},
                },
            )
            ui.step1_panel()
            ui.step2_panel()
            ui.step3_panel()
            state = _fake_state(root)
            passive_result = object()
            active_result = object()
            with (
                mock.patch(
                    "modules.notebooks.pipeline_ui.prepare_pipeline_notebook",
                    return_value=state,
                ),
                mock.patch(
                    "modules.notebooks.pipeline_ui.run_passive_stage",
                    return_value=passive_result,
                ) as passive,
                mock.patch(
                    "modules.notebooks.pipeline_ui.run_active_stage",
                    return_value=active_result,
                ) as active,
                mock.patch("modules.notebooks.pipeline_ui._clear_output"),
            ):
                ui.controls["step1_run"].click()
                ui.controls["step2_run"].click()
                ui.controls["step3_run"].click()

            self.assertIs(passive.call_args.args[0], state)
            self.assertIs(active.call_args.args[0], state)
            self.assertEqual(
                passive.call_args.kwargs["protocol_overrides"], {"h_dt": 0.1}
            )
            self.assertEqual(
                active.call_args.kwargs["protocol_overrides"], {"stim_dur": 750.0}
            )
            self.assertIs(ui.passive_result, passive_result)
            self.assertIs(ui.active_result, active_result)

    def test_restart_required_disables_stage_buttons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_repo(Path(tmp))
            ui = PipelineNotebookUI(root, {"cell_name": "A", "tune_name": "tuned"})
            ui.step1_panel()
            ui.step2_panel()
            ui.step3_panel()
            ui.step4_panel()
            ui.step5_panel()
            with (
                mock.patch(
                    "modules.notebooks.pipeline_ui.prepare_pipeline_notebook",
                    return_value=_fake_state(root),
                ),
                mock.patch(
                    "modules.notebooks.pipeline_ui.run_passive_stage",
                    side_effect=RestartKernelRequired("source changed"),
                ),
                mock.patch("modules.notebooks.pipeline_ui._clear_output"),
                mock.patch("modules.notebooks.pipeline_ui.traceback.print_exc"),
            ):
                ui.controls["step1_run"].click()
                ui.controls["step2_run"].click()

            for key in ("step2_run", "step3_run", "step4_initialize", "step5_run"):
                self.assertTrue(ui.controls[key].disabled, key)
            self.assertIn("Kernel restart required", ui.statuses["step2"].value)
            self.assertIn("Kernel restart required", ui.statuses["step4"].value)

    def test_bmtool_buttons_follow_initialization_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_repo(Path(tmp))
            ui = PipelineNotebookUI(root, {"cell_name": "A", "tune_name": "tuned"})
            ui.step1_panel()
            ui.step4_panel()
            tuner = mock.Mock()
            with (
                mock.patch(
                    "modules.notebooks.pipeline_ui.prepare_pipeline_notebook",
                    return_value=_fake_state(root),
                ),
                mock.patch(
                    "modules.notebooks.pipeline_ui.prepare_interactive_synapse_tuner",
                    return_value=tuner,
                ) as initialize,
                mock.patch("modules.notebooks.pipeline_ui._clear_output"),
            ):
                ui.controls["step1_run"].click()
                self.assertEqual(
                    ui.controls["synapse_connection"].value,
                    "excitatory_facilitating",
                )
                self.assertTrue(ui.controls["step4_initialize"].disabled)
                ui.controls["enable_synapse_tuning"].value = True
                self.assertFalse(ui.controls["step4_initialize"].disabled)
                ui.controls["step4_initialize"].click()
                ui.controls["step4_single_event"].click()
                ui.controls["step4_interactive"].click()

            self.assertIs(initialize.call_args.args[0], ui.pipeline_state)
            tuner.SingleEvent.assert_called_once_with()
            tuner.InteractiveTuner.assert_called_once_with()
            self.assertIs(ui.tuner, tuner)
            self.assertTrue(ui.controls["synapse_connection"].disabled)

    def test_step5_passes_widget_values_and_supports_repeated_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_repo(Path(tmp))
            ui = PipelineNotebookUI(root, {"cell_name": "A", "tune_name": "tuned"})
            ui.step1_panel()
            ui.step5_panel()
            state = _fake_state(root)
            result = SimpleNamespace(
                results={"mode": "iclamp"},
                manifest_path=Path(tmp) / "run_manifest.json",
            )
            with (
                mock.patch(
                    "modules.notebooks.pipeline_ui.prepare_pipeline_notebook",
                    return_value=state,
                ),
                mock.patch(
                    "modules.notebooks.pipeline_ui.run_fresh_simulation",
                    return_value=result,
                ) as fresh,
                mock.patch(
                    "modules.notebooks.pipeline_ui.show_run_diagnostics",
                    return_value={"ok": True},
                ) as diagnostics,
                mock.patch("modules.notebooks.pipeline_ui._clear_output"),
                mock.patch("builtins.print"),
            ):
                ui.controls["step1_run"].click()
                ui.controls["n_trials"].value = 3
                ui.controls["seed"].value = "42"
                ui.controls["run_mode"].value = "iclamp"
                ui.controls["output_stem"].value = "widget_run"
                ui.controls["diagnostic_plot"].value = "summary"
                ui.controls["step5_run"].click()
                ui.controls["step5_run"].click()

            self.assertEqual(fresh.call_count, 2)
            kwargs = fresh.call_args.kwargs
            self.assertEqual(kwargs["n_trials"], 3)
            self.assertEqual(kwargs["seed"], 42)
            self.assertTrue(kwargs["iclamp"])
            self.assertEqual(kwargs["output_stem"], "widget_run")
            self.assertFalse(diagnostics.call_args.kwargs["include_inputs"])
            self.assertEqual(diagnostics.call_args.kwargs["diagnostic_plot"], "summary")
            self.assertIs(ui.simulation_result, result)
            self.assertEqual(ui.diagnostics, {"ok": True})


if __name__ == "__main__":
    unittest.main()
