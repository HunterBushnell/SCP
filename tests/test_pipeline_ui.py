from __future__ import annotations

import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from modules.notebooks.pipeline_ui import _parse_float_list, _parse_optional_int
from modules.notebooks.pipeline_workflow import RestartKernelRequired
from tests.pipeline_ui_test_support import (
    PipelineNotebookUI,
    all_descendants as _all_descendants,
    fake_state as _fake_state,
    make_repo as _make_repo,
)


class PipelineNotebookUICoreTests(unittest.TestCase):
    def test_parsers_accept_lists_and_report_invalid_values(self) -> None:
        self.assertEqual(_parse_float_list("-50, -100", name="amps"), [-50.0, -100.0])
        self.assertEqual(_parse_float_list("[0, 50]", name="amps"), [0.0, 50.0])
        self.assertIsNone(_parse_optional_int("", name="seed"))
        self.assertEqual(_parse_optional_int("17", name="seed"), 17)
        with self.assertRaisesRegex(ValueError, "invalid number"):
            _parse_float_list("0, nope", name="amps")
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            _parse_optional_int("0", name="trial count", positive=True)

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
            self.assertEqual(
                {
                    name: type(component).__module__
                    for name, component in ui._components.items()
                },
                {
                    "step1": "modules.notebooks._pipeline_ui.step1",
                    "step2": "modules.notebooks._pipeline_ui.step2",
                    "step3": "modules.notebooks._pipeline_ui.step3",
                    "step3_act": "modules.notebooks._pipeline_ui.step3_act",
                    "step4": "modules.notebooks._pipeline_ui.step4",
                    "step5": "modules.notebooks._pipeline_ui.step5",
                },
            )
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
            self.assertTrue(ui.controls["quiet_step1_output"].value)
            self.assertNotIn("adb_specimen_id", ui.controls)
            self.assertNotIn("adb_model_type", ui.controls)
            self.assertIn("step2_proposal", ui.controls)
            self.assertNotIn("compute_act_passive_proposal", ui.controls)
            self.assertIsInstance(ui.controls["step2_proposal"], ui.widgets.Button)
            self.assertEqual(
                ui.controls["active_current_display_amp_pA"].value,
                300.0,
            )
            self.assertIn(
                "Experimental",
                ui.controls["act_experimental_note"].value,
            )
            self.assertIsNot(
                ui.outputs["step2_proposal"], ui.outputs["step2_passive"]
            )
            self.assertTrue(ui.controls["step2_proposal"].disabled)
            self.assertTrue(ui.controls["step2_run"].disabled)
            self.assertIsInstance(ui.controls["step3_active"], ui.widgets.Button)
            self.assertIsInstance(ui.controls["step3_fi"], ui.widgets.Button)
            self.assertIsNot(
                ui.outputs["step4_initialize"],
                ui.outputs["step4_single_event"],
            )
            self.assertIsNot(
                ui.outputs["step4_single_event"],
                ui.outputs["step4_interactive"],
            )
            self.assertIsNot(
                ui.outputs["step5_check_inputs"],
                ui.outputs["step5_run"],
            )
            self.assertIsNot(
                ui.outputs["step5_run"],
                ui.outputs["step5_plot"],
            )
            self.assertTrue(ui.controls["quiet_input_preview_output"].value)
            self.assertTrue(ui.controls["quiet_simulation_output"].value)
            self.assertEqual(
                tuple(ui.controls["input_preview_plots"].value),
                (
                    "weight_distribution",
                    "distance_distribution",
                    "weight_vs_distance",
                ),
            )
            self.assertEqual(
                ui.controls["input_preview_options_box"].layout.display,
                "none",
            )
            ui.controls["input_preview_options_toggle"].value = True
            self.assertEqual(
                ui.controls["input_preview_options_box"].layout.display,
                "",
            )
            self.assertEqual(
                ui.controls["input_preview_options_toggle"].description,
                "Hide advanced options",
            )
            self.assertEqual(
                ui.controls["simulation_options_box"].layout.display,
                "none",
            )
            ui.controls["simulation_options_toggle"].value = True
            self.assertEqual(
                ui.controls["simulation_options_box"].layout.display,
                "",
            )
            self.assertEqual(
                ui.controls["simulation_options_toggle"].description,
                "Hide advanced options",
            )
            self.assertEqual(ui.controls["sim_iclamp_box"].layout.display, "none")
            self.assertEqual(
                tuple(ui.controls["diagnostic_plots"].value),
                (
                    "input_rate",
                    "membrane_voltage",
                    "output_rate",
                    "output_raster",
                ),
            )
            self.assertEqual(
                ui.controls["diagnostic_options_box"].layout.display,
                "none",
            )
            ui.controls["diagnostic_options_toggle"].value = True
            self.assertEqual(
                ui.controls["diagnostic_options_box"].layout.display,
                "",
            )
            self.assertIsNot(
                ui.outputs["step3_active"],
                ui.outputs["step3_fi"],
            )
            self.assertIsNot(
                ui.outputs["step3_fi"],
                ui.outputs["step3_act"],
            )
            self.assertEqual(ui.controls["active_timing_box"].layout.display, "none")
            self.assertEqual(ui.controls["fi_timing_box"].layout.display, "none")
            self.assertTrue(ui.controls["step3_active"].disabled)
            self.assertTrue(ui.controls["step3_fi"].disabled)
            self.assertTrue(ui.controls["step3_act_prepare"].disabled)
            self.assertTrue(ui.controls["step3_act_run"].disabled)
            self.assertTrue(ui.controls["step3_act_cancel"].disabled)
            self.assertTrue(ui.controls["step3_act_review"].disabled)
            self.assertTrue(ui.controls["step3_act_evaluate"].disabled)
            self.assertTrue(ui.controls["step3_act_review_evaluation"].disabled)
            self.assertEqual(ui.controls["act_options_box"].layout.display, "none")
            ui.controls["act_options_toggle"].value = True
            self.assertEqual(ui.controls["act_options_box"].layout.display, "")
            ui.controls["act_target_mode"].value = "fi_arrays"
            ui.controls["act_target_currents"].value = "0, 75, 150"
            ui.controls["act_target_frequencies"].value = "0, 4, 20"
            ui.controls["act_active_module"].value = "spiking"
            self.assertEqual(settings["act_active_module"], "spiking")
            self.assertEqual(
                settings["act_overrides"]["target"]["fi_currents_pA"],
                [0.0, 75.0, 150.0],
            )
            self.assertEqual(
                settings["act_overrides"]["target"]["fi_frequencies_hz"],
                [0.0, 4.0, 20.0],
            )
            self.assertTrue(ui.controls["step5_check_inputs"].disabled)
            self.assertTrue(ui.controls["step5_run"].disabled)
            self.assertTrue(ui.controls["step5_plot"].disabled)

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
            ui.controls["passive_target_rin_mohm"].value = "120.5"
            self.assertEqual(
                settings["passive_target_overrides"]["target_rin_mohm"],
                120.5,
            )
            ui.controls["quiet_step1_output"].value = False
            self.assertFalse(settings["quiet_step1_output"])
            ui.controls["quiet_step1_output"].value = True

            with (
                mock.patch(
                    "modules.notebooks._pipeline_ui.step1.prepare_pipeline_notebook",
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
                "passive_target_overrides": {
                    "target_rin_mohm": 115.0,
                    "target_tau_ms": 7.0,
                    "target_v_rest_mv": -69.0,
                },
                "active_amps_pA": [125],
            }
            ui.apply_settings(refreshed)
            self.assertEqual(refreshed["cell_name"], "A")
            self.assertEqual(refreshed["tune_name"], "tuned")
            self.assertEqual(ui.controls["passive_amps_pA"].value, "-10")
            self.assertEqual(ui.controls["passive_target_rin_mohm"].value, "115.0")
            self.assertEqual(ui.controls["passive_target_tau_ms"].value, "7.0")
            self.assertEqual(ui.controls["passive_target_v_rest_mv"].value, "-69.0")
            self.assertEqual(ui.controls["active_amps_pA"].value, "125")
            self.assertEqual(
                ui.controls["active_current_display_amp_pA"].value,
                125.0,
            )
            self.assertIn("Model selection is locked", ui.statuses["step1"].value)

    def test_setup_validation_can_retry_but_attempt_failure_requires_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_repo(Path(tmp))
            ui = PipelineNotebookUI(root, {"cell_name": "A", "tune_name": "tuned"})
            ui.step1_panel()
            ui.controls["cell_name"].value = ""
            ui.controls["tune_name"].value = ""
            with (
                mock.patch(
                    "modules.notebooks._pipeline_ui.step1.prepare_pipeline_notebook"
                ) as prepare,
                mock.patch("modules.notebooks.pipeline_ui._clear_output"),
                mock.patch("builtins.print"),
            ):
                ui.controls["step1_run"].click()
            prepare.assert_not_called()
            self.assertFalse(ui.controls["step1_run"].disabled)

            ui.controls["cell_name"].value = "A"
            ui.controls["tune_name"].value = "tuned"
            with (
                mock.patch(
                    "modules.notebooks._pipeline_ui.step1.prepare_pipeline_notebook",
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
            self.assertIn("model build failed", ui.step1_load_log)

    def test_quiet_load_captures_chatter_and_keeps_curated_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_repo(Path(tmp))
            settings = {"cell_name": "A", "tune_name": "tuned"}
            ui = PipelineNotebookUI(root, settings)
            ui.step1_panel()
            state = _fake_state(root)
            previous_neuron_options = os.environ.get("NEURON_MODULE_OPTIONS")

            def noisy_prepare(**_kwargs):
                self.assertIn(
                    "-nogui",
                    os.environ.get("NEURON_MODULE_OPTIONS", "").split(),
                )
                sys.stdout.write("third-party setup chatter\n")
                sys.stderr.write("third-party stderr\n")
                warnings.warn("third-party warning", RuntimeWarning)
                return state

            with (
                mock.patch(
                    "modules.notebooks._pipeline_ui.step1.prepare_pipeline_notebook",
                    side_effect=noisy_prepare,
                ),
                mock.patch("modules.notebooks.pipeline_ui._clear_output"),
                mock.patch("builtins.print") as visible_print,
            ):
                ui.controls["step1_run"].click()

            self.assertIn("third-party setup chatter", ui.step1_load_log)
            self.assertIn("third-party stderr", ui.step1_load_log)
            self.assertIn("third-party warning", ui.step1_load_log)
            visible_print.assert_any_call("Pipeline tune loaded")
            self.assertTrue(ui.controls["quiet_step1_output"].disabled)
            self.assertEqual(
                os.environ.get("NEURON_MODULE_OPTIONS"),
                previous_neuron_options,
            )

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
                    "modules.notebooks._pipeline_ui.step1.prepare_pipeline_notebook",
                    return_value=_fake_state(root),
                ),
                mock.patch(
                    "modules.notebooks._pipeline_ui.step2.run_passive_stage",
                    side_effect=RestartKernelRequired("source changed"),
                ),
                mock.patch("modules.notebooks.pipeline_ui._clear_output"),
                mock.patch("modules.notebooks.pipeline_ui.traceback.print_exc"),
            ):
                ui.controls["step1_run"].click()
                ui.controls["step2_run"].click()

            for key in (
                "step2_proposal",
                "step2_run",
                "step3_active",
                "step3_fi",
                "step4_initialize",
                "step5_check_inputs",
                "step5_run",
                "step5_plot",
            ):
                self.assertTrue(ui.controls[key].disabled, key)
            self.assertIn("Kernel restart required", ui.statuses["step2"].value)
            self.assertIn("Kernel restart required", ui.statuses["step4"].value)


if __name__ == "__main__":
    unittest.main()
