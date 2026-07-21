from __future__ import annotations

import unittest
from copy import deepcopy
from unittest import mock

from modules.tuning import (
    active_amplitude_colors,
    compact_display_data,
    display_active_analysis,
    display_fi_analysis,
    display_passive_analysis,
    display_tuning_rows,
    format_tuning_value,
    passive_amplitude_colors,
    plot_active_trace_check,
    plot_passive_trace_check,
)


class TuningDisplayTests(unittest.TestCase):
    def test_compact_display_data_uses_four_significant_figures_only_in_copy(self) -> None:
        source = {
            "v_rest_mV": -71.23456789,
            "rin_MOhm": 98.9876543,
            "tau_ms": 5.9876543,
            "g_bar_leak": 0.000123456789,
            "spike_count": 3,
            "nested": [1.23456789],
        }
        original = deepcopy(source)

        compact = compact_display_data(source)

        self.assertEqual(compact["v_rest_mV"], -71.23)
        self.assertEqual(compact["rin_MOhm"], 98.99)
        self.assertEqual(compact["tau_ms"], 5.988)
        self.assertEqual(compact["g_bar_leak"], 0.0001235)
        self.assertEqual(compact["spike_count"], 3)
        self.assertEqual(compact["nested"], [1.235])
        self.assertEqual(source, original)

    def test_value_and_table_display_are_compact_without_changing_rows(self) -> None:
        rows = [{"amp_pA": -50.0, "measured_value": 98.9876543}]
        original = deepcopy(rows)

        self.assertEqual(format_tuning_value(-71.23456789), "-71.23")
        self.assertEqual(format_tuning_value(0.000123456789), "0.0001235")
        with mock.patch("IPython.display.display") as display:
            displayed_rows = display_tuning_rows("Metrics", rows)

        self.assertEqual(displayed_rows[0]["measured_value"], 98.99)
        self.assertEqual(display.call_args.args[0].iloc[0]["measured_value"], 98.99)
        self.assertEqual(rows, original)

    def test_significant_digits_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            compact_display_data(1.0, significant_digits=0)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            format_tuning_value(1.0, significant_digits=0)

    def test_passive_analysis_groups_metrics_and_uses_trace_colors(self) -> None:
        colors = passive_amplitude_colors([-50.0, -100.0])
        metrics = [
            {
                "amp_pA": amp,
                "spike_frequency_hz": 0.0,
                "V_rest": -70.0 + index,
                "R_in_rest_to_final": 100.0 + index,
                "tau_avg": 10.0 + index,
                "tau_rest_to_trough": 9.5 + index,
                "sag_ratio": 0.1 + index * 0.05,
            }
            for index, amp in enumerate((-50.0, -100.0))
        ]
        comparison = [
            {
                "amp_pA": amp,
                "metric": metric,
                "unit": unit,
                "target_value": target,
                "measured_value": measured,
                "delta": measured - target,
                "abs_delta": abs(measured - target),
                "pct_error": (measured - target) / abs(target) * 100.0,
                "status": "ok",
            }
            for amp, measured_values in (
                (-50.0, (-70.0, 100.0, 10.0)),
                (-100.0, (-69.0, 101.0, 11.0)),
            )
            for metric, unit, target, measured in zip(
                ("v_rest_mV", "rin_MOhm", "tau_ms"),
                ("mV", "MOhm", "ms"),
                (-70.0, 100.0, 10.0),
                measured_values,
            )
        ]

        with (
            mock.patch("IPython.display.display") as display,
            mock.patch("builtins.print") as printed,
        ):
            summary = display_passive_analysis(
                metrics,
                comparison,
                amplitude_colors=colors,
            )

        self.assertEqual(display.call_count, 2)
        grouped = summary["target_comparison"]
        self.assertEqual(
            [row["Metric"] for row in grouped],
            [
                "Resting voltage (mV)",
                "Resting voltage (mV)",
                "Input resistance (MΩ)",
                "Input resistance (MΩ)",
                "Membrane tau (ms)",
                "Membrane tau (ms)",
            ],
        )
        self.assertEqual(
            [row["Current (pA)"] for row in grouped],
            [-50.0, -100.0, -50.0, -100.0, -50.0, -100.0],
        )
        rendered = display.call_args_list[1].args[0].to_html()
        self.assertIn('rowspan="2"', rendered)
        self.assertIn(colors[-50.0], rendered)
        self.assertIn(colors[-100.0], rendered)
        self.assertNotIn("spike_frequency", rendered)
        self.assertEqual(summary["unexpected_spikes"], [])
        self.assertFalse(
            any("unexpected spiking" in str(call).lower() for call in printed.call_args_list)
        )

    def test_passive_analysis_warns_only_for_nonzero_spike_frequency(self) -> None:
        metrics = [
            {
                "amp_pA": -50.0,
                "spike_frequency_hz": 2.5,
                "V_rest": -70.0,
                "R_in_rest_to_final": 100.0,
                "tau_avg": 10.0,
                "tau_rest_to_trough": 9.5,
                "sag_ratio": 0.1,
            }
        ]
        with (
            mock.patch("IPython.display.display"),
            mock.patch("builtins.print") as printed,
        ):
            summary = display_passive_analysis(metrics, [])

        self.assertEqual(summary["unexpected_spikes"][0]["spike_frequency_hz"], 2.5)
        self.assertTrue(
            any("unexpected spiking" in str(call).lower() for call in printed.call_args_list)
        )

    def test_passive_plot_uses_the_same_amplitude_colors_as_tables(self) -> None:
        from matplotlib import pyplot as plt
        from matplotlib.colors import to_hex

        amplitudes = [-50.0, -100.0]
        colors = passive_amplitude_colors(amplitudes)
        records = {
            "T": {amp: [0.0, 1.0, 2.0] for amp in amplitudes},
            "V": {
                -50.0: [-70.0, -71.0, -70.5],
                -100.0: [-70.0, -72.0, -71.0],
            },
        }
        figure = plot_passive_trace_check(
            looped_records=records,
            sim_params={"stim_delay": 0.5, "stim_dur": 1.0},
            sim_amps=amplitudes,
            cell_name="A",
            tune_name="tuned",
            amplitude_colors=colors,
        )
        try:
            plotted_colors = [to_hex(line.get_color()) for line in figure.axes[0].lines]
            self.assertEqual(
                plotted_colors,
                [colors[-50.0], colors[-100.0]],
            )
        finally:
            plt.close(figure)

    def test_active_analysis_and_plot_share_current_colors(self) -> None:
        from matplotlib import pyplot as plt
        from matplotlib.colors import to_hex

        amplitudes = [150.0, 300.0]
        colors = active_amplitude_colors(amplitudes)
        metrics = [
            {
                "amp_pA": amp,
                "spike_count": index + 1,
                "spike_frequency_hz": 5.0 * (index + 1),
                "rest_voltage_mv": -70.0,
                "peak_voltage_mv": 30.0,
                "min_voltage_mv": -75.0,
                "first_spike_latency_ms": 12.34567,
                "mean_isi_ms": 25.6789,
                "min_isi_ms": 24.5,
                "isi_cv": 0.123456,
                "adaptation_ratio": 1.23456,
            }
            for index, amp in enumerate(amplitudes)
        ]
        records = {
            "T": {amp: [0.0, 1.0, 2.0] for amp in amplitudes},
            "V": {
                150.0: [-70.0, 20.0, -60.0],
                300.0: [-70.0, 30.0, -55.0],
            },
            "I": {},
        }
        figure = plot_active_trace_check(
            looped_records=records,
            sim_params={"stim_delay": 0.5, "stim_dur": 1.0},
            sim_amps=amplitudes,
            cell_name="A",
            tune_name="tuned",
            include_currents=False,
            amplitude_colors=colors,
        )
        try:
            self.assertEqual(
                [to_hex(line.get_color()) for line in figure.axes[0].lines],
                [colors[150.0], colors[300.0]],
            )
            with mock.patch("IPython.display.display") as display:
                rows = display_active_analysis(metrics, amplitude_colors=colors)
            self.assertEqual(display.call_count, 2)
            rendered = "\n".join(
                call.args[0].to_html() for call in display.call_args_list
            )
            self.assertIn(colors[150.0], rendered)
            self.assertIn(colors[300.0], rendered)
            self.assertEqual(
                rows["firing_summary"][0]["First-spike latency (ms)"],
                12.34567,
            )
        finally:
            plt.close(figure)

    def test_active_plot_uses_the_selected_sweep_for_ionic_currents(self) -> None:
        from matplotlib import pyplot as plt

        amplitudes = [150.0, 300.0]
        records = {
            "T": {amp: [0.0, 1.0, 2.0] for amp in amplitudes},
            "V": {amp: [-70.0, -60.0, -65.0] for amp in amplitudes},
            "I": {
                150.0: {"ina": [0.0, -1.0, 0.0]},
                300.0: {"ik": [0.0, 2.0, 0.0]},
            },
        }

        figure = plot_active_trace_check(
            looped_records=records,
            sim_params={"stim_delay": 0.5, "stim_dur": 1.0},
            sim_amps=amplitudes,
            cell_name="A",
            tune_name="tuned",
            current_amp=150.0,
        )
        try:
            self.assertEqual(len(figure.axes), 2)
            self.assertEqual(figure.axes[1].get_title(), "Recorded currents @ 150 pA")
            self.assertEqual(
                [line.get_label() for line in figure.axes[1].lines],
                ["ina"],
            )
        finally:
            plt.close(figure)

    def test_fi_analysis_uses_one_compact_comparison_table(self) -> None:
        fi_rows = [
            {"amp_pA": 0.0, "spike_frequency_hz": 0.0},
            {"amp_pA": 100.0, "spike_frequency_hz": 12.34567},
        ]
        comparison = [
            {
                "amp_pA": 0.0,
                "target_lookup": "exact",
                "target_frequency_hz": 0.0,
                "measured_frequency_hz": 0.0,
                "delta": 0.0,
                "pct_error": None,
            },
            {
                "amp_pA": 100.0,
                "target_lookup": "interpolated",
                "target_frequency_hz": 10.0,
                "measured_frequency_hz": 12.34567,
                "delta": 2.34567,
                "pct_error": 23.4567,
            },
        ]
        with (
            mock.patch("IPython.display.display") as display,
            mock.patch("builtins.print") as printed,
        ):
            rows = display_fi_analysis(
                fi_rows,
                comparison,
                model_color="#1f77b4",
                reference_color="#000000",
            )

        self.assertEqual(display.call_count, 1)
        self.assertEqual(rows[1]["Target lookup"], "Interpolated")
        rendered = display.call_args.args[0].to_html()
        self.assertIn("Model frequency (Hz)", rendered)
        self.assertIn("Target frequency (Hz)", rendered)
        self.assertFalse(
            any("no configured" in str(call).lower() for call in printed.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
