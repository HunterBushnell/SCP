from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.pipeline_ui_test_support import (
    PipelineNotebookUI,
    REPO_ROOT,
    fake_state as _fake_state,
    make_repo as _make_repo,
)


class PipelineNotebookUIStep5Tests(unittest.TestCase):
    def test_input_preview_groups_are_loaded_from_expanded_synapse_config(self) -> None:
        settings = {"cell_name": "PV", "tune_name": "tuned"}
        ui = PipelineNotebookUI(REPO_ROOT, settings)
        ui.step5_panel()

        options = [
            value for _label, value in ui.controls["input_preview_groups"].options
        ]
        self.assertEqual(options, ["pn_exc", "bg_exc", "bg_inh"])
        self.assertEqual(
            tuple(ui.controls["input_preview_groups"].value),
            ("pn_exc", "bg_exc", "bg_inh"),
        )
        ui.controls["input_preview_groups"].value = ("pn_exc", "bg_inh")
        self.assertEqual(settings["input_preview_groups"], ["pn_exc", "bg_inh"])

    def test_simulation_options_load_config_and_sync_session_overrides(self) -> None:
        settings = {"cell_name": "PV", "tune_name": "tuned"}
        ui = PipelineNotebookUI(REPO_ROOT, settings)
        ui.step5_panel()

        sim_path = (
            REPO_ROOT
            / "cells"
            / "PV"
            / "tunes"
            / "tuned"
            / "cell_configs"
            / "sim_config.json"
        )
        configured = json.loads(sim_path.read_text(encoding="utf-8"))
        self.assertEqual(ui.controls["sim_tstop_ms"].value, configured["tstop"])
        self.assertEqual(
            ui.controls["sim_iclamp_amp_nA"].value,
            configured["iclamp"]["amp_nA"],
        )

        ui.controls["sim_dt_ms"].value = 0.05
        ui.controls["sim_plots_profile"].value = "inputs"
        self.assertEqual(settings["simulation_overrides"]["dt"], 0.05)
        self.assertTrue(settings["simulation_overrides"]["save_plots"])
        self.assertTrue(settings["simulation_overrides"]["save_plots_inputs"])
        self.assertFalse(settings["simulation_overrides"]["save_plots_synapses"])

        settings["simulation_overrides"] = {
            "tstop": 900.0,
            "iclamp": {"amp_nA": 0.3},
        }
        ui.apply_settings(settings)
        self.assertEqual(ui.controls["sim_tstop_ms"].value, 900.0)
        self.assertEqual(ui.controls["sim_iclamp_amp_nA"].value, 0.3)

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
                output_stem="widget_run",
                command=("python", "run_pipeline.py"),
                stdout="routine simulation chatter\n",
            )
            preview_result = SimpleNamespace(
                syn_state={"records": {"exc": []}, "preview_only": True},
                trial_idx=2,
                summary={"total_n_syn": 3},
                command=("python", "-m", "pipeline_preview_worker"),
                stdout="routine preview chatter\n",
            )
            with (
                mock.patch(
                    "modules.notebooks._pipeline_ui.step1.prepare_pipeline_notebook",
                    return_value=state,
                ),
                mock.patch(
                    "modules.notebooks._pipeline_ui.step5.preview_pipeline_inputs",
                    return_value=preview_result,
                ) as preview,
                mock.patch(
                    "modules.notebooks._pipeline_ui.step5.show_synapse_preview",
                ) as show_preview,
                mock.patch(
                    "modules.notebooks._pipeline_ui.step5.run_fresh_simulation",
                    return_value=result,
                ) as fresh,
                mock.patch(
                    "modules.notebooks._pipeline_ui.step5.show_run_diagnostics",
                    return_value={"ok": True},
                ) as diagnostics,
                mock.patch("modules.notebooks.pipeline_ui._clear_output"),
                mock.patch("builtins.print"),
            ):
                ui.controls["step1_run"].click()
                ui.controls["n_trials"].value = 3
                ui.controls["seed"].value = "42"
                ui.controls["run_mode"].value = "iclamp"
                self.assertEqual(ui.controls["sim_iclamp_box"].layout.display, "")
                ui.controls["sim_tstop_ms"].value = 900.0
                ui.controls["sim_iclamp_amp_nA"].value = 0.35
                ui.controls["sim_plots_profile"].value = "inputs"
                ui.controls["sim_cell_recording_enabled"].value = True
                ui.controls["sim_record_ion_currents"].value = True
                ui.controls["output_stem"].value = "widget_run"
                ui.controls["input_preview_plots"].value = (
                    "weight_vs_distance",
                )
                ui.controls["input_preview_trial_idx"].value = 2
                ui.controls["input_preview_show_table"].value = False
                ui.controls["input_preview_histogram_mode"].value = "count"
                ui.controls["input_preview_distance_bin_um"].value = 50.0
                ui.controls["input_preview_weight_bin"].value = "0.2"
                ui.controls["input_preview_plot_columns"].value = 1
                ui.controls["input_preview_plot_size"].value = "standard"
                ui.controls["step5_check_inputs"].click()
                ui.controls["step5_run"].click()
                ui.controls["step5_run"].click()
                self.assertFalse(ui.controls["step5_plot"].disabled)
                self.assertEqual(
                    tuple(ui.controls["diagnostic_plots"].options),
                    (("Membrane voltage", "membrane_voltage"),),
                )
                self.assertTrue(ui.controls["diagnostic_rate_bin_ms"].disabled)
                self.assertTrue(ui.controls["diagnostic_raster_style"].disabled)
                ui.controls["diagnostic_window_mode"].value = "manual"
                ui.controls["diagnostic_window_start_ms"].value = "10"
                ui.controls["diagnostic_window_stop_ms"].value = "50"
                ui.controls["diagnostic_figure_size"].value = "standard"
                ui.controls["step5_plot"].click()

            self.assertEqual(preview.call_args.kwargs["seed"], 42)
            self.assertEqual(preview.call_args.kwargs["trial_idx"], 2)
            self.assertFalse(preview.call_args.kwargs["stream_output"])
            self.assertIs(show_preview.call_args.args[0], preview_result.syn_state)
            self.assertEqual(
                show_preview.call_args.kwargs["plot_kinds"],
                ["weight_vs_distance"],
            )
            self.assertFalse(show_preview.call_args.kwargs["show_table"])
            self.assertFalse(show_preview.call_args.kwargs["histogram_density"])
            self.assertEqual(show_preview.call_args.kwargs["distance_bin_um"], 50.0)
            self.assertEqual(show_preview.call_args.kwargs["weight_bin"], 0.2)
            self.assertEqual(show_preview.call_args.kwargs["plot_columns"], 1)
            self.assertEqual(show_preview.call_args.kwargs["figsize"], (4.4, 3.4))
            self.assertEqual(fresh.call_count, 2)
            kwargs = fresh.call_args.kwargs
            self.assertEqual(kwargs["n_trials"], 3)
            self.assertEqual(kwargs["seed"], 42)
            self.assertTrue(kwargs["iclamp"])
            self.assertEqual(kwargs["output_stem"], "widget_run")
            self.assertFalse(kwargs["stream_output"])
            self.assertEqual(kwargs["sim_overrides"]["tstop"], 900.0)
            self.assertEqual(kwargs["sim_overrides"]["iclamp"]["amp_nA"], 0.35)
            self.assertEqual(kwargs["sim_overrides"]["plots_profile"], "inputs")
            self.assertTrue(kwargs["sim_overrides"]["save_plots_inputs"])
            self.assertTrue(
                kwargs["sim_overrides"]["cell_recording"]["enabled"]
            )
            self.assertTrue(
                kwargs["sim_overrides"]["cell_recording"]["vars"]["ion_currents"]
            )
            self.assertEqual(diagnostics.call_count, 1)
            self.assertFalse(diagnostics.call_args.kwargs["include_inputs"])
            self.assertEqual(diagnostics.call_args.kwargs["diagnostic_plot"], "custom")
            self.assertEqual(
                diagnostics.call_args.kwargs["diagnostic_plots"],
                ["membrane_voltage"],
            )
            self.assertEqual(
                diagnostics.call_args.kwargs["plot_options"]["plot_window"],
                (10.0, 50.0),
            )
            self.assertEqual(
                diagnostics.call_args.kwargs["plot_options"]["figsize"],
                (8.0, 4.0),
            )
            self.assertIs(ui.input_preview_result, preview_result)
            self.assertIs(ui.simulation_result, result)
            self.assertEqual(ui.diagnostics, {"ok": True})
            self.assertIn("routine preview chatter", ui.input_preview_log)
            self.assertIn("routine simulation chatter", ui.simulation_log)

    def test_diagnostic_controls_refresh_trials_and_saved_input_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_repo(Path(tmp))
            settings = {"cell_name": "A", "tune_name": "tuned"}
            ui = PipelineNotebookUI(root, settings)
            ui.step5_panel()
            ui.simulation_result = SimpleNamespace(
                results={
                    "mode": "multi",
                    "sim_cfg": {
                        "bins": 10.0,
                        "plots_input_smooth_ms": 20.0,
                    },
                    "traces": {"V": [[-65.0], [-64.0]]},
                    "spikes": [[], [], []],
                    "inputs_by_trial": [
                        {"inputs": {"exc": {}, "inh": {}}},
                        {"inputs": {"exc": {}, "inh": {}}},
                    ],
                    "meta": {},
                }
            )

            ui._refresh_diagnostic_controls()

            self.assertEqual(
                tuple(ui.controls["diagnostic_trial_idx"].options),
                (("Trial 0", 0), ("Trial 1", 1), ("Trial 2", 2)),
            )
            self.assertFalse(ui.controls["diagnostic_trial_idx"].disabled)
            self.assertEqual(
                tuple(ui.controls["diagnostic_input_groups"].value),
                ("exc", "inh"),
            )
            self.assertEqual(ui.controls["diagnostic_rate_bin_ms"].value, 10.0)
            self.assertEqual(ui.controls["diagnostic_smoothing_ms"].value, 20.0)
            ui.controls["diagnostic_plots"].value = (
                "input_raster",
                "membrane_voltage",
            )
            ui.controls["diagnostic_trial_idx"].value = 2
            ui.controls["diagnostic_input_groups"].value = ("exc",)
            self.assertEqual(
                settings["diagnostic_plots"],
                ["input_raster", "membrane_voltage"],
            )
            self.assertEqual(settings["diagnostic_trial_idx"], 2)
            self.assertEqual(settings["diagnostic_input_groups"], ["exc"])


if __name__ == "__main__":
    unittest.main()
