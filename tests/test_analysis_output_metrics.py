from __future__ import annotations

import json
import unittest
from pathlib import Path

from modules.analysis import analysis


class OutputRiseMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.curve = {
            "t_ms": [0.0, 50.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 200.0],
            "rate_hz": [20.0, 20.0, 20.0, 25.0, 40.0, 65.0, 90.0, 120.0, 20.0],
            "units": "Hz",
        }
        self.sim_cfg = {
            "tstart": 0.0,
            "tstop": 200.0,
            "stim_start_ms": 100.0,
            "stim_stop_ms": 200.0,
        }

    def _metrics(self, curve=None, **kwargs):
        return analysis.compute_output_metrics(
            curve or self.curve,
            self.sim_cfg,
            baseline_mode="point",
            baseline_center_ms=50.0,
            peak_window_ms=100.0,
            **kwargs,
        )

    def test_default_rise_uses_baseline_to_peak_and_interpolated_crossings(self) -> None:
        metrics = self._metrics()

        self.assertAlmostEqual(metrics["rise_start_pct"], 10.0)
        self.assertAlmostEqual(metrics["rise_stop_pct"], 90.0)
        self.assertAlmostEqual(metrics["rise_start_rate_hz"], 30.0)
        self.assertAlmostEqual(metrics["rise_stop_rate_hz"], 110.0)
        self.assertAlmostEqual(metrics["rise_delta_rate_hz"], 80.0)
        self.assertAlmostEqual(metrics["rise_start_time_ms"], 113.3333333333)
        self.assertAlmostEqual(metrics["rise_stop_time_ms"], 146.6666666667)
        self.assertAlmostEqual(metrics["rise_start_latency_ms"], 13.3333333333)
        self.assertAlmostEqual(metrics["rise_stop_latency_ms"], 46.6666666667)
        self.assertAlmostEqual(metrics["rise_time_ms"], 33.3333333333)

    def test_adjustable_percent_range_changes_endpoints_and_deltas(self) -> None:
        metrics = self._metrics(rise_percent_range=[20.0, 80.0])

        self.assertAlmostEqual(metrics["rise_start_rate_hz"], 40.0)
        self.assertAlmostEqual(metrics["rise_stop_rate_hz"], 100.0)
        self.assertAlmostEqual(metrics["rise_delta_rate_hz"], 60.0)
        self.assertAlmostEqual(metrics["rise_start_time_ms"], 120.0)
        self.assertAlmostEqual(metrics["rise_stop_time_ms"], 143.3333333333)
        self.assertAlmostEqual(metrics["rise_time_ms"], 23.3333333333)

    def test_normalized_curve_still_reports_raw_hz_and_plot_values(self) -> None:
        normalized = analysis.normalize_output_curve(
            self.curve,
            self.sim_cfg,
            mode="normalized",
            norm_mode="peak",
            baseline_ms=100.0,
            baseline_mode="point",
            baseline_center_ms=50.0,
            norm_window="stim",
        )
        metrics = self._metrics(normalized)

        self.assertAlmostEqual(metrics["rise_start_rate_hz"], 30.0)
        self.assertAlmostEqual(metrics["rise_stop_rate_hz"], 110.0)
        self.assertAlmostEqual(metrics["rise_start_value"], 0.1)
        self.assertAlmostEqual(metrics["rise_stop_value"], 0.9)
        self.assertAlmostEqual(metrics["rise_time_ms"], 33.3333333333)

    def test_first_crossing_uses_stimulus_onset_when_already_above_threshold(self) -> None:
        curve = dict(self.curve)
        curve["rate_hz"] = [20.0, 20.0, 35.0, 25.0, 40.0, 65.0, 90.0, 120.0, 20.0]
        metrics = self._metrics(curve)

        self.assertAlmostEqual(metrics["rise_start_time_ms"], 100.0)
        self.assertAlmostEqual(metrics["rise_start_latency_ms"], 0.0)

    def test_first_upward_crossing_is_kept_when_curve_recrosses_threshold(self) -> None:
        curve = dict(self.curve)
        curve["rate_hz"] = [20.0, 20.0, 20.0, 40.0, 25.0, 50.0, 90.0, 120.0, 20.0]
        metrics = self._metrics(curve)

        self.assertAlmostEqual(metrics["rise_start_time_ms"], 105.0)

    def test_disabled_rise_metric_leaves_results_empty(self) -> None:
        metrics = self._metrics(rise_metric_enabled=False)

        self.assertFalse(metrics["rise_metric_enabled"])
        self.assertIsNone(metrics["rise_start_time_ms"])
        self.assertIsNone(metrics["rise_stop_time_ms"])
        self.assertIsNone(metrics["rise_time_ms"])
        self.assertIsNone(metrics["rise_delta_rate_hz"])

    def test_invalid_percent_ranges_are_rejected(self) -> None:
        invalid_ranges = ([90.0, 10.0], [-1.0, 90.0], [10.0, 101.0], [10.0])
        for percent_range in invalid_ranges:
            with self.subTest(percent_range=percent_range):
                with self.assertRaises(ValueError):
                    self._metrics(rise_percent_range=percent_range)

    def test_step6_metric_preset_enables_and_reports_rise_outputs(self) -> None:
        preset_path = (
            Path(__file__).resolve().parents[1]
            / "modules"
            / "analysis"
            / "analysis_presets"
            / "output_metrics.json"
        )
        defaults = json.loads(preset_path.read_text())["defaults"]

        self.assertTrue(defaults["output_rise_metric_enabled"])
        self.assertEqual(defaults["output_rise_percent_range"], [10.0, 90.0])
        self.assertTrue(defaults["output_show_rise_points"])
        self.assertIn("rise_start_time_ms", defaults["output_metrics_plot_keys"])
        self.assertIn("rise_stop_time_ms", defaults["output_metrics_plot_keys"])
        self.assertIn("rise_time_ms", defaults["output_metrics_plot_keys"])
        self.assertIn("rise_delta_rate_hz", defaults["output_metrics_plot_keys"])
        self.assertIn("auc_raw_hz_s", defaults["output_metrics_plot_keys"])
        self.assertIn("auc_normalized_s", defaults["output_metrics_plot_keys"])


if __name__ == "__main__":
    unittest.main()
