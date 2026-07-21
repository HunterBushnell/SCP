from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.pipeline_ui_test_support import (
    PipelineNotebookUI,
    fake_state as _fake_state,
    make_repo as _make_repo,
    write_manual_passive_targets as _write_manual_passive_targets,
)


class PipelineNotebookUITuningTests(unittest.TestCase):
    def test_passive_and_active_callbacks_reuse_step1_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_repo(Path(tmp))
            _write_manual_passive_targets(root)
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
            resolution = SimpleNamespace(
                target_source_mode="manual",
                passive_targets={
                    "target_rin_mohm": 110.0,
                    "target_tau_ms": 6.25,
                    "target_v_rest_mv": -70.5,
                },
            )
            proposal_result = SimpleNamespace(resolution=resolution)
            passive_result = SimpleNamespace(resolution=resolution)
            active_result = object()
            fi_result = object()
            self.assertEqual(ui.controls["passive_target_rin_mohm"].value, "101.5")
            self.assertEqual(ui.controls["passive_target_tau_ms"].value, "6.25")
            self.assertEqual(ui.controls["passive_target_v_rest_mv"].value, "-70.5")
            self.assertEqual(ui.controls["passive_timing_box"].layout.display, "none")
            self.assertEqual(
                ui.controls["passive_timing_toggle"].description,
                "Show advanced options",
            )
            ui.controls["passive_timing_toggle"].value = True
            self.assertEqual(ui.controls["passive_timing_box"].layout.display, "")
            self.assertEqual(
                ui.controls["passive_timing_toggle"].description,
                "Hide advanced options",
            )
            ui.controls["passive_target_rin_mohm"].value = "110"
            ui.controls["active_timing_toggle"].value = True
            ui.controls["fi_timing_toggle"].value = True
            self.assertEqual(ui.controls["active_timing_box"].layout.display, "")
            self.assertEqual(ui.controls["fi_timing_box"].layout.display, "")
            ui.controls["fi_stim_dur"].value = "500"
            ui.controls["active_spike_threshold_mV"].value = "-15"
            ui.controls["fi_spike_threshold_mV"].value = "-10"
            ui.controls["active_include_currents"].value = False
            ui.controls["active_current_display_amp_pA"].value = 150.0
            with (
                mock.patch(
                    "modules.notebooks._pipeline_ui.step1.prepare_pipeline_notebook",
                    return_value=state,
                ),
                mock.patch(
                    "modules.notebooks._pipeline_ui.step2.compute_passive_proposal",
                    return_value=proposal_result,
                ) as proposal,
                mock.patch(
                    "modules.notebooks._pipeline_ui.step2.run_passive_stage",
                    return_value=passive_result,
                ) as passive,
                mock.patch(
                    "modules.notebooks._pipeline_ui.step3.run_active_protocol_stage",
                    return_value=active_result,
                ) as active,
                mock.patch(
                    "modules.notebooks._pipeline_ui.step3.run_fi_curve_stage",
                    return_value=fi_result,
                ) as fi,
                mock.patch("modules.notebooks.pipeline_ui._clear_output"),
            ):
                ui.controls["step1_run"].click()
                ui.controls["step2_proposal"].click()
                ui.controls["step2_run"].click()
                ui.controls["step3_active"].click()
                ui.controls["step3_fi"].click()

            self.assertIs(proposal.call_args.args[0], state)
            self.assertIs(passive.call_args.args[0], state)
            self.assertIs(active.call_args.args[0], state)
            self.assertIs(fi.call_args.args[0], state)
            self.assertFalse(passive.call_args.kwargs["compute_act_proposal"])
            expected_targets = {
                "target_rin_mohm": 110.0,
                "target_tau_ms": 6.25,
                "target_v_rest_mv": -70.5,
            }
            self.assertEqual(
                proposal.call_args.kwargs["manual_passive_targets"],
                expected_targets,
            )
            self.assertEqual(
                passive.call_args.kwargs["manual_passive_targets"],
                expected_targets,
            )
            self.assertEqual(
                passive.call_args.kwargs["protocol_overrides"], {"h_dt": 0.1}
            )
            self.assertEqual(
                active.call_args.kwargs["protocol_overrides"], {"stim_dur": 750.0}
            )
            self.assertEqual(active.call_args.kwargs["spike_threshold_mV"], -15.0)
            self.assertFalse(active.call_args.kwargs["include_currents"])
            self.assertEqual(active.call_args.kwargs["current_amp_pA"], 150.0)
            self.assertEqual(
                active.call_args.kwargs["active_amps_pA"], [150.0, 300.0]
            )
            self.assertEqual(
                fi.call_args.kwargs["fi_amps_pA"],
                [0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0],
            )
            self.assertEqual(
                fi.call_args.kwargs["protocol_overrides"], {"stim_dur": 500.0}
            )
            self.assertEqual(fi.call_args.kwargs["spike_threshold_mV"], -10.0)
            self.assertIs(ui.passive_proposal_result, proposal_result)
            self.assertIs(ui.passive_result, passive_result)
            self.assertIs(ui.active_result, active_result)
            self.assertIs(ui.fi_result, fi_result)


if __name__ == "__main__":
    unittest.main()
