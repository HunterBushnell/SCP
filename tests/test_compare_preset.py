from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from modules.analysis.ui import _engine, outputs
from modules.analysis.ui.state import active_compare_preset_path, get_selection_from_globals


class PaperComparePresetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.defaults_path = cls.repo_root / "modules" / "analysis" / "analysis_defaults.json"

    def test_bundled_preset_is_configured_but_off_by_default(self) -> None:
        defaults = json.loads(self.defaults_path.read_text(encoding="utf-8"))
        self.assertFalse(defaults["compare_preset_enabled"])

        preset_path = self.repo_root / defaults["compare_preset_path"]
        self.assertTrue(preset_path.is_file())
        preset = json.loads(preset_path.read_text(encoding="utf-8"))
        self.assertEqual(preset["title"], "paper_compare")
        self.assertTrue(preset["defaults"])
        self.assertTrue(preset["entries"])

        loaded = _engine._load_compare_preset(defaults["compare_preset_path"], self.repo_root)
        self.assertEqual(
            [entry["label"] for entry in loaded],
            ["PN bio", "PV base", "PV tuned", "SST base", "SST tuned"],
        )

    def test_enabled_flag_separates_available_path_from_active_mode(self) -> None:
        path = "modules/analysis/analysis_presets/paper_compare.json"
        self.assertIsNone(active_compare_preset_path({
            "compare_preset_path": path,
            "compare_preset_enabled": False,
        }))
        self.assertEqual(active_compare_preset_path({
            "compare_preset_path": path,
            "compare_preset_enabled": True,
        }), path)
        self.assertEqual(active_compare_preset_path({
            "compare_preset_path": path,
        }), path)

    def test_output_ui_exposes_an_unchecked_but_usable_preset_control(self) -> None:
        if outputs._maybe_import_widgets() is None:
            self.skipTest("ipywidgets is unavailable")
        state = json.loads(self.defaults_path.read_text(encoding="utf-8"))
        output_defaults_path = (
            self.repo_root / "modules" / "analysis" / "analysis_presets" / "output_plotting.json"
        )
        state.update(json.loads(output_defaults_path.read_text(encoding="utf-8"))["defaults"])
        state.update({
            "use_widgets": True,
            "_HAVE_WIDGETS": True,
            "auto_run_outputs": False,
        })

        with mock.patch.object(
            outputs,
            "_maybe_import_display",
            return_value=(lambda *_args, **_kwargs: None, None),
        ):
            outputs.build_outputs_ui(state)

        self.assertEqual(
            state["compare_preset_path_txt"].value,
            state["compare_preset_path"],
        )
        self.assertFalse(state["compare_preset_cb"].value)
        self.assertFalse(state["compare_preset_cb"].disabled)

    def test_selection_only_passes_an_enabled_preset_to_the_engine(self) -> None:
        base = {
            "use_widgets": False,
            "_HAVE_WIDGETS": False,
            "cell_name": "SST",
            "tunes_dir": "tunes",
            "model_dir": "tuned_ampa",
            "run_single_stem": "latest",
            "run_compare_a": "latest",
            "run_compare_b": "none",
            "compare_list": [],
            "compare_list_paths": [],
            "compare_list_paths_enabled": True,
            "CELLS_DIR": self.repo_root / "cells",
            "compare_preset_path": "modules/analysis/analysis_presets/paper_compare.json",
        }

        inactive = get_selection_from_globals(dict(base, compare_preset_enabled=False))
        active = get_selection_from_globals(dict(base, compare_preset_enabled=True))
        self.assertIsNone(inactive["compare_preset_path"])
        self.assertEqual(active["compare_preset_path"], base["compare_preset_path"])

    def test_enabled_preset_settings_override_general_ui_values(self) -> None:
        merged = _engine._merge_preset_defaults(
            {
                "plot_window": [0, 1000],
                "output_curve_mode": "raw",
                "multi_shade_mode": "sem",
            },
            {
                "plot_window": [200, 900],
                "output_curve_mode": "normalized",
                "multi_shade_mode": None,
            },
        )

        self.assertEqual(merged["plot_window"], [200, 900])
        self.assertEqual(merged["output_curve_mode"], "normalized")
        self.assertIsNone(merged["multi_shade_mode"])


if __name__ == "__main__":
    unittest.main()
