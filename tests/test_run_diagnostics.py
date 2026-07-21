from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO

import matplotlib
import numpy as np

matplotlib.use("Agg")

from modules.notebooks.run_diagnostics import show_run_diagnostics


class RunDiagnosticsTests(unittest.TestCase):
    def test_single_plot_without_inputs_omits_both_input_panels(self) -> None:
        time_ms = np.arange(0.0, 20.0, 0.1)
        results = {
            "mode": "multi",
            "sim_cfg": {"tstart": 0.0, "tstop": 20.0, "dt": 0.1},
            "traces": {"T": time_ms, "V": [np.full(time_ms.size, -65.0)]},
            "spikes": [np.array([10.0])],
            "meta": {},
        }

        payload = show_run_diagnostics(
            results,
            diagnostic_plot="single_plot",
            include_inputs=False,
            plot_options={
                "output_recompute_bin_ms": 2.0,
                "output_recompute_smooth_ms": 0.0,
            },
        )

        plotted = payload["single_plot"]
        self.assertFalse(plotted["included_panels"]["input_rate"])
        self.assertFalse(plotted["included_panels"]["input_raster"])

        import matplotlib.pyplot as plt

        plt.close(plotted["fig"])

    def test_large_trial_summary_is_compact(self) -> None:
        results = {
            "mode": "multi",
            "spikes": [np.arange(index % 4) for index in range(50)],
        }
        output = StringIO()

        with redirect_stdout(output):
            show_run_diagnostics(results, diagnostic_plot=None)

        text = output.getvalue()
        self.assertIn("Trials: 50", text)
        self.assertIn("Spike-count range:", text)
        self.assertNotIn("Spike counts: [", text)

    def test_custom_input_panels_use_saved_groups(self) -> None:
        time_ms = np.arange(0.0, 20.0, 1.0)
        group_stats = {
            "n_syn": 2,
            "rate_hz_by_bin_per_syn": [0.0] * 10 + [5.0] * 10,
        }
        results = {
            "mode": "single",
            "sim_cfg": {
                "tstart": 0.0,
                "tstop": 20.0,
                "stim_start_ms": 10.0,
                "stim_duration_ms": 5.0,
            },
            "traces": {"T": time_ms, "V": np.full(time_ms.size, -65.0)},
            "spikes": np.array([]),
            "inputs": {
                "exc": {"spike_trains": [np.array([11.0]), np.array([12.0])]}
            },
            "meta": {
                "input_stats": {
                    "bin_ms": 1.0,
                    "t_ms": time_ms.tolist(),
                    "tstart_ms": 0.0,
                    "tstop_ms": 20.0,
                    "trials": [{"trial_idx": 0, "groups": {"exc": group_stats}}],
                    "group_means": {"exc": {}},
                }
            },
        }

        payload = show_run_diagnostics(
            results,
            diagnostic_plot="custom",
            diagnostic_plots=["input_rate", "input_raster"],
            plot_options={
                "top_input_groups": ["exc"],
                "raster_input_groups": ["exc"],
                "input_smooth_ms": 0.0,
                "figsize": (6.0, 4.0),
            },
        )

        plotted = payload["custom_plot"]
        self.assertEqual(len(plotted["axes"]), 2)
        self.assertEqual(plotted["used_groups_top"], ["exc"])
        self.assertEqual(plotted["used_groups_raster"], ["exc"])

        import matplotlib.pyplot as plt

        plt.close(plotted["fig"])

    def test_custom_diagnostic_renders_only_selected_panels(self) -> None:
        time_ms = np.arange(0.0, 100.0, 0.1)
        results = {
            "mode": "multi",
            "sim_cfg": {
                "tstart": 0.0,
                "tstop": 100.0,
                "dt": 0.1,
                "stim_start_ms": 20.0,
                "stim_duration_ms": 40.0,
                "color": "C0",
            },
            "traces": {"T": time_ms, "V": [np.full(time_ms.size, -65.0)]},
            "spikes": [np.array([30.0, 50.0]), np.array([40.0])],
            "meta": {},
        }

        payload = show_run_diagnostics(
            results,
            diagnostic_plot="custom",
            diagnostic_plots=[
                "membrane_voltage",
                "output_rate",
                "output_raster",
            ],
            plot_options={
                "trial_idx": 0,
                "output_recompute_bin_ms": 5.0,
                "output_recompute_smooth_ms": 10.0,
                "plot_window": (10.0, 70.0),
                "auto_plot_window_from_stim": False,
                "figsize": (6.0, 5.0),
            },
        )

        plotted = payload["custom_plot"]
        self.assertEqual(len(plotted["axes"]), 3)
        self.assertEqual(
            plotted["included_panels"],
            {
                "input_rate": False,
                "input_raster": False,
                "membrane_voltage": True,
                "output_rate": True,
                "output_raster": True,
            },
        )
        self.assertEqual(tuple(plotted["axes"][0].get_xlim()), (10.0, 70.0))

        import matplotlib.pyplot as plt

        plt.close(plotted["fig"])

    def test_custom_iclamp_limits_compact_output_to_voltage(self) -> None:
        time_ms = np.arange(0.0, 20.0, 0.1)
        results = {
            "mode": "iclamp",
            "sim_cfg": {},
            "traces": {"T": time_ms, "V": np.full(time_ms.size, -65.0)},
            "meta": {"frequency_hz": 0.0},
        }

        payload = show_run_diagnostics(
            results,
            diagnostic_plot="custom",
            diagnostic_plots=["membrane_voltage", "output_rate"],
            plot_options={"trial_idx": 0, "plot_window": (2.0, 12.0)},
        )

        plotted = payload["custom_plot"]
        self.assertIsNotNone(plotted["fig"])
        self.assertTrue(plotted["warnings"])
        self.assertEqual(tuple(plotted["fig"].axes[0].get_xlim()), (2.0, 12.0))

        import matplotlib.pyplot as plt

        plt.close(plotted["fig"])


if __name__ == "__main__":
    unittest.main()
