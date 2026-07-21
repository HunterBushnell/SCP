from __future__ import annotations

import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.pipeline_ui_test_support import (
    PipelineNotebookUI,
    fake_state as _fake_state,
    make_repo as _make_repo,
)


class PipelineNotebookUIACTTests(unittest.TestCase):
    def test_act_buttons_prepare_run_and_evaluate_in_background(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_repo(Path(tmp))
            ui = PipelineNotebookUI(root, {"cell_name": "A", "tune_name": "tuned"})
            ui.step1_panel()
            ui.step3_panel()
            state = _fake_state(root)
            module_spec = {
                "lto": {
                    "enabled": True,
                    "name": "seg_lto",
                    "conductances": [
                        {
                            "variable_name": "gbar_test",
                            "low": 0.0,
                            "high": 1.0,
                            "n_slices": 2,
                        }
                    ],
                }
            }
            config = {
                "target": {
                    "mode": "fi_arrays",
                    "fi_currents_pA": [0, 100],
                    "fi_frequencies_hz": [0, 10],
                },
                "act_cell": {"passive": ["g_pas"], "active_channels": ["gbar_test"]},
                "simulation": {
                    "h_v_init": -70,
                    "h_tstop": 1500,
                    "h_dt": 0.025,
                    "h_celsius": 37,
                    "ci_delay_ms": 200,
                    "ci_dur_ms": 1000,
                },
                "optimizer": {
                    "n_cpus": 1,
                    "random_state": 42,
                    "n_estimators": 10,
                    "max_depth": None,
                    "train_features": ["spike_frequency", "mean_i"],
                    "spike_threshold": -20,
                    "max_n_spikes": 20,
                },
                "filter": {
                    "filtered_out_features": None,
                    "window_of_inspection": [200, 1200],
                    "saturation_threshold": -55,
                },
                "modules": module_spec,
            }
            inspection = SimpleNamespace(
                workspace=Path(tmp) / "act_workspace",
                config_path=Path(tmp) / "act_workspace" / "act_active_config.json",
                builder_path=Path(tmp) / "act_workspace" / "cell_builder.py",
                target_path=Path(tmp) / "act_workspace" / "target_sf.csv",
                target_mode="fi_arrays",
                target_point_count=2,
                loader_name="allen_manifest",
                loader_support="supported",
                config_source="existing",
                enabled_modules=["lto"],
                workload={
                    "target_points": 2,
                    "training_traces": 4,
                    "evaluation_traces": 2,
                    "total_traces": 6,
                },
                output_status={"lto": {"status": "missing"}},
                config_fingerprint="config",
                target_config_fingerprint="target",
                resolved_config=config,
                act_available=False,
                act_message="not probed",
            )
            prepared = SimpleNamespace(**inspection.__dict__)
            prepared.act_available = True
            prepared.act_message = "ready"
            run_result = SimpleNamespace(
                predictions={"gbar_test": 0.25},
                metrics={"lto": [{"metric": "Train MAE (g)", "value": "0.1"}]},
                output_status={"lto": {"status": "current"}},
            )
            evaluation = SimpleNamespace(manifest_path=Path(tmp) / "evaluation_manifest.json")
            with (
                mock.patch(
                    "modules.notebooks._pipeline_ui.step1.prepare_pipeline_notebook",
                    return_value=state,
                ),
                mock.patch(
                    "modules.notebooks._pipeline_ui.step3_act.inspect_act_active_stage",
                    return_value=inspection,
                ),
                mock.patch(
                    "modules.notebooks._pipeline_ui.step3_act.prepare_act_active_stage",
                    return_value=prepared,
                ) as prepare_act,
                mock.patch(
                    "modules.notebooks._pipeline_ui.step3_act.run_fresh_act_active",
                    return_value=run_result,
                ) as run_act,
                mock.patch(
                    "modules.notebooks._pipeline_ui.step3_act.evaluate_fresh_act_predictions",
                    return_value=evaluation,
                ) as evaluate_act,
                mock.patch(
                    "modules.notebooks._pipeline_ui.step3_act.Step3ACTUI._print_act_review"
                ),
                mock.patch("modules.notebooks.pipeline_ui._clear_output"),
            ):
                ui.controls["step1_run"].click()
                ui.controls["step3_act_prepare"].click()
                ui._act_thread.join(timeout=2)
                self.assertIs(ui.act_workspace_result, prepared)
                ui.controls["step3_act_run"].click()
                ui._act_thread.join(timeout=2)
                self.assertIs(ui.act_run_result, run_result)
                self.assertEqual(ui.act_predictions, {"gbar_test": 0.25})
                ui.controls["step3_act_evaluate"].click()
                ui._act_thread.join(timeout=2)

            self.assertIs(prepare_act.call_args.args[0], state)
            self.assertIs(run_act.call_args.args[0], state)
            self.assertEqual(run_act.call_args.kwargs["modules"], "lto")
            self.assertIs(evaluate_act.call_args.args[0], state)
            self.assertIs(ui.act_evaluation_result, evaluation)

    def test_act_cancel_terminates_the_worker_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ui = PipelineNotebookUI(
                _make_repo(Path(tmp)), {"cell_name": "A", "tune_name": "tuned"}
            )
            ui.step3_panel()
            process = mock.Mock(pid=4321)
            process.poll.return_value = None
            ui.act_job = process
            ui._act_thread = mock.Mock()
            ui._act_thread.is_alive.return_value = True
            force_thread = mock.Mock()
            with (
                mock.patch(
                    "modules.notebooks._pipeline_ui.step3_act.os.getpgid",
                    return_value=4321,
                ),
                mock.patch("modules.notebooks._pipeline_ui.step3_act.os.killpg") as killpg,
                mock.patch(
                    "modules.notebooks._pipeline_ui.step3_act.threading.Thread",
                    return_value=force_thread,
                ),
            ):
                ui._on_act_cancel(None)
            killpg.assert_called_once_with(4321, signal.SIGTERM)
            force_thread.start.assert_called_once_with()
            self.assertTrue(ui._act_cancel_requested)

    def test_act_settings_noop_preserves_preparation_and_changes_invalidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = {"cell_name": "A", "tune_name": "tuned"}
            ui = PipelineNotebookUI(_make_repo(Path(tmp)), settings)
            ui.step3_panel()
            ui.pipeline_state = _fake_state(ui.repo_root)
            prepared = SimpleNamespace(output_status={}, act_available=True)
            ui.act_workspace_result = prepared

            with mock.patch(
                "modules.notebooks._pipeline_ui.step3_act.Step3ACTUI._refresh_act_inspection"
            ) as refresh:
                ui.apply_settings(settings)
                self.assertIs(ui.act_workspace_result, prepared)
                refresh.assert_not_called()

                settings["act_overrides"] = {
                    "optimizer": {"n_estimators": 25}
                }
                ui.apply_settings(settings)

            self.assertIsNone(ui.act_workspace_result)
            refresh.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
