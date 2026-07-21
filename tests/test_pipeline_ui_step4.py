from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.pipeline_ui_test_support import (
    PipelineNotebookUI,
    fake_state as _fake_state,
    make_repo as _make_repo,
)


class PipelineNotebookUIStep4Tests(unittest.TestCase):
    def test_bmtool_buttons_follow_initialization_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_repo(Path(tmp))
            ui = PipelineNotebookUI(root, {"cell_name": "A", "tune_name": "tuned"})
            ui.step1_panel()
            ui.step4_panel()
            tuner = mock.Mock()
            with (
                mock.patch(
                    "modules.notebooks._pipeline_ui.step1.prepare_pipeline_notebook",
                    return_value=_fake_state(root),
                ),
                mock.patch(
                    "modules.notebooks._pipeline_ui.step4.prepare_interactive_synapse_tuner",
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


if __name__ == "__main__":
    unittest.main()
