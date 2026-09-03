from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

from modules.analysis import auto_plots, single_plot_panel


class AutoPlotOutputLayoutTests(unittest.TestCase):
    def _save_mock_single_plot(self, run_dir: Path, expected_plot_dir: Path) -> None:
        preset_path = run_dir.parent / "single_plot.json"
        preset_path.write_text("{}")
        panel_result = {
            "fig": None,
            "warnings": [],
            "exported_paths": [],
            "requested_export_paths": [expected_plot_dir / "single_plot.svg"],
        }

        with patch.object(
            single_plot_panel,
            "plot_single_plot_panel_from_results",
            return_value=panel_result,
        ):
            auto_plots.save_default_plots(
                {},
                run_dir,
                plot_mode="single_plot",
                single_plot_preset=preset_path,
            )

    def test_wrapped_run_creates_only_batch_plot_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            batch_dir = Path(tmp) / "slurm_123"
            run_dir = batch_dir / "results"
            run_dir.mkdir(parents=True)

            self._save_mock_single_plot(run_dir, batch_dir / "plots")

            self.assertTrue((batch_dir / "plots").is_dir())
            self.assertFalse((run_dir / "plots").exists())

    def test_ordinary_run_keeps_plot_directory_inside_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "output_data" / "notebook_run"
            run_dir.mkdir(parents=True)

            self._save_mock_single_plot(run_dir, run_dir / "plots")

            self.assertTrue((run_dir / "plots").is_dir())


if __name__ == "__main__":
    unittest.main()
