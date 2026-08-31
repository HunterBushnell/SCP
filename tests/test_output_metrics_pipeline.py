from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from modules.analysis import output_metrics
from modules.simulation.result_saving import save_results


class OutputMetricsPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = {
            "mode": "multi",
            "spikes": [
                np.asarray([90.0, 110.0]),
                np.asarray([100.0, 120.0, 140.0]),
                np.asarray([]),
                np.asarray([130.0, 135.0, 160.0]),
            ],
            "sim_cfg": {
                "tstart": 0.0,
                "tstop": 250.0,
                "stim_start_ms": 100.0,
                "stim_stop_ms": 200.0,
                "stim_duration_ms": 100.0,
                "bins": 5.0,
            },
        }
        self.settings = {
            "output_metrics_bin_ms": 5.0,
            "output_metrics_smooth_ms": 0.0,
            "output_metrics_smooth_mode": "center",
            "output_metrics_curve_mode": "raw",
            "output_metrics_norm_mode": "peak",
            "output_metrics_norm_window": "stim",
            "output_metric_mode": "window",
            "output_metric_window_ms": 50.0,
            "output_peak_window_ms": 100.0,
            "output_drop_window_ms": 50.0,
            "output_rebound_window_ms": 100.0,
            "output_auc_window": "stim",
            "output_t50_mode": "relative",
            "output_rise_metric_enabled": True,
            "output_rise_percent_range": [10.0, 90.0],
            "output_stim_spike_metrics_enabled": True,
            "output_first_spike_metric_enabled": True,
            "output_isi_metrics_enabled": True,
            "output_metrics_std_mode": "std",
        }

    def test_raw_spike_metrics_are_aggregated_per_trial(self) -> None:
        metrics = output_metrics.compute_output_metrics_from_results(
            self.results,
            self.settings,
        )

        self.assertEqual(metrics["output_metrics_n_trials"], 4)
        self.assertAlmostEqual(metrics["stim_spike_count"], 1.75)
        self.assertAlmostEqual(metrics["stim_spike_count_median"], 2.0)
        self.assertAlmostEqual(metrics["stim_mean_rate_hz"], 17.5)
        self.assertEqual(metrics["stim_response_trials"], 3)
        self.assertAlmostEqual(metrics["stim_response_fraction"], 0.75)
        self.assertEqual(metrics["stim_repetitive_trials"], 2)
        self.assertAlmostEqual(metrics["stim_repetitive_fraction"], 0.5)

        self.assertEqual(metrics["first_spike_contributing_trials"], 3)
        self.assertAlmostEqual(metrics["first_spike_latency_ms"], 40.0 / 3.0)
        self.assertAlmostEqual(metrics["first_spike_latency_median_ms"], 10.0)

        self.assertEqual(metrics["initial_isi_contributing_trials"], 2)
        self.assertAlmostEqual(metrics["first_to_second_isi_ms"], 12.5)
        self.assertAlmostEqual(metrics["initial_pair_rate_hz"], 125.0)
        self.assertAlmostEqual(metrics["mean_isi_ms"], 17.5)
        self.assertAlmostEqual(metrics["min_isi_ms"], 12.5)
        self.assertAlmostEqual(metrics["peak_within_trial_rate_hz"], 125.0)

    def test_synchronized_single_spikes_raise_psth_without_within_trial_rate(self) -> None:
        results = {
            "mode": "multi",
            "spikes": [np.asarray([110.0]) for _ in range(5)],
            "sim_cfg": dict(self.results["sim_cfg"]),
        }
        metrics = output_metrics.compute_output_metrics_from_results(results, self.settings)

        self.assertAlmostEqual(metrics["peak_rate_hz_raw"], 200.0)
        self.assertAlmostEqual(metrics["stim_spike_count"], 1.0)
        self.assertAlmostEqual(metrics["stim_response_fraction"], 1.0)
        self.assertAlmostEqual(metrics["stim_repetitive_fraction"], 0.0)
        self.assertIsNone(metrics["first_to_second_isi_ms"])
        self.assertIsNone(metrics["peak_within_trial_rate_hz"])

    def test_raw_and_normalized_auc_are_always_calculated(self) -> None:
        raw_metrics = output_metrics.compute_output_metrics_from_results(
            self.results,
            self.settings,
        )
        normalized_settings = dict(
            self.settings,
            output_metrics_curve_mode="normalized",
        )
        normalized_metrics = output_metrics.compute_output_metrics_from_results(
            self.results,
            normalized_settings,
        )

        self.assertIsNotNone(raw_metrics["auc_raw_hz_s"])
        self.assertIsNotNone(raw_metrics["auc_normalized_s"])
        self.assertAlmostEqual(
            raw_metrics["auc_raw_hz_s"],
            normalized_metrics["auc_raw_hz_s"],
        )
        self.assertAlmostEqual(
            raw_metrics["auc_normalized_s"],
            normalized_metrics["auc_normalized_s"],
        )
        self.assertAlmostEqual(raw_metrics["auc"], raw_metrics["auc_raw_hz_s"])
        self.assertAlmostEqual(
            normalized_metrics["auc"],
            normalized_metrics["auc_normalized_s"],
        )
        self.assertEqual(raw_metrics["auc_raw_units"], "Hz*s")
        self.assertEqual(raw_metrics["auc_normalized_units"], "normalized*s")
        self.assertEqual(raw_metrics["auc_normalization_mode"], "peak")
        self.assertEqual(raw_metrics["auc_normalization_window"], "stim")
        self.assertIsNotNone(raw_metrics["auc_normalization_scale_hz"])

    def test_full_data_and_important_view_are_saved_separately(self) -> None:
        metrics = output_metrics.compute_output_metrics_from_results(self.results, self.settings)
        settings = dict(self.settings)
        settings.update({
            "output_metrics_save_formats": ["json", "csv", "md"],
            "output_metrics_important_keys": ["stim_spike_count", "first_spike_latency_ms"],
            "output_metrics_show_params": False,
        })
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            saved = output_metrics.save_output_metrics_artifacts(metrics, run_dir, settings)

            self.assertEqual(set(saved), {
                "output_metrics_json",
                "output_metrics_csv",
                "output_metrics_important_md",
            })
            payload = json.loads(saved["output_metrics_json"].read_text())
            self.assertIn("peak_within_trial_rate_hz", payload)
            self.assertIn("auc_raw_hz_s", payload)
            self.assertIn("auc_normalized_s", payload)
            markdown = saved["output_metrics_important_md"].read_text()
            self.assertIn("Mean stimulus spikes per trial", markdown)
            self.assertIn("First-spike latency", markdown)
            self.assertNotIn("Peak within-trial instantaneous rate", markdown)

    def test_saved_metrics_without_explicit_auc_variants_are_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            analysis_dir = run_dir / "analysis"
            analysis_dir.mkdir(parents=True)
            (analysis_dir / "output_metrics.json").write_text(
                json.dumps({
                    "peak_latency_ms": 10.0,
                    "output_metrics_n_trials": 4,
                    "stim_spike_count": 1.0,
                    "stim_response_fraction": 1.0,
                    "stim_repetitive_fraction": 0.0,
                    "first_spike_latency_ms": 10.0,
                    "first_spike_contributing_trials": 4,
                    "first_to_second_isi_ms": None,
                    "initial_isi_contributing_trials": 0,
                    "auc": 1.0,
                }),
                encoding="utf-8",
            )

            report = output_metrics.load_or_compute_output_metrics(
                run_dir,
                results=self.results,
                settings=self.settings,
                prefer_saved=True,
                save=False,
            )

            self.assertFalse(report["used_saved"])
            self.assertTrue(report["warnings"])
            self.assertIn("auc_raw_hz_s", report["metrics"])
            self.assertIn("auc_normalized_s", report["metrics"])

    def test_result_saving_runs_metrics_independently_of_plot_saving(self) -> None:
        results = dict(self.results)
        results["sim_cfg"] = dict(self.results["sim_cfg"])
        results["sim_cfg"].update({
            "output": "metrics_run",
            "output_stem": "metrics_run",
            "save_output": True,
            "output_format": "pkl",
            "save_full_results": False,
            "save_sidecars": True,
            "save_model_artifacts": False,
            "save_fit_json_sidecar": False,
            "save_plots": False,
            "save_output_metrics": True,
            "save_output_metrics_formats": ["json", "csv", "md"],
            "save_output_metrics_overwrite": True,
            "cell": "test_cell",
            "tune": "test_tune",
        })
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = save_results(results, base_dir=tmp)
            self.assertIsNotNone(manifest_path)
            manifest = json.loads(Path(manifest_path).read_text())
            run_dir = Path(manifest_path).parent
            self.assertNotIn("plots", manifest["files"])
            self.assertIn("output_metrics", manifest["files"])
            self.assertTrue((run_dir / "analysis" / "output_metrics.json").is_file())
            self.assertTrue((run_dir / "analysis" / "output_metrics_important.md").is_file())


if __name__ == "__main__":
    unittest.main()
