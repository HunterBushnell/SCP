from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

from modules.tuning.manual_synapse_probe import calculate_manual_synapse_metrics


class ManualSynapseMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spikes = np.asarray(
            [100.0 + 20.0 * index for index in range(8)]
            + [490.0 + 20.0 * index for index in range(4)]
        )
        self.delay = 1.3

        single_t = np.arange(0.0, 220.0 + 0.05, 0.1)
        single_i = np.zeros_like(single_t)
        elapsed = single_t - (self.spikes[0] + self.delay)
        active = elapsed >= 0.0
        single_i[active] = -0.020 * (
            1.0 - np.exp(-elapsed[active] / 0.25)
        ) * np.exp(-elapsed[active] / 3.0)
        self.single = {
            "T": single_t,
            "V": np.full_like(single_t, -65.0),
            "I": single_i,
        }

        self.train, self.amplitudes_pA = self._train_trace(len(self.spikes))

    def _train_trace(self, spike_count: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
        spikes = self.spikes[:spike_count]
        train_t = np.arange(0.0, float(spikes[-1]) + 150.0 + 0.05, 0.1)
        train_v = np.full_like(train_t, -65.0)
        train_i = np.zeros_like(train_t)
        amplitudes_pA = np.asarray(
            [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 15.0, 17.0, 19.0, 21.0]
        )[:spike_count]
        for spike, amplitude in zip(spikes, amplitudes_pA):
            index = int(np.searchsorted(train_t, spike + self.delay))
            train_i[index] = -amplitude / 1000.0
        first_response = int(np.searchsorted(train_t, spikes[0] + self.delay))
        train_v[first_response] = -63.0
        return {"T": train_t, "V": train_v, "I": train_i}, amplitudes_pA

    def test_reports_voltage_single_event_and_bmtool_stp_metrics(self) -> None:
        metrics = calculate_manual_synapse_metrics(
            single_event_traces=self.single,
            train_traces=self.train,
            spike_times_ms=self.spikes,
            netcon_delay_ms=self.delay,
        )

        np.testing.assert_allclose(
            metrics["pulse_amplitudes_pA"], self.amplitudes_pA, atol=1e-12
        )
        percentile_90 = float(np.percentile(self.amplitudes_pA, 90))
        self.assertAlmostEqual(metrics["simple_ppr"], 12.0 / 10.0)
        self.assertAlmostEqual(metrics["ppr"], (12.0 - 10.0) / percentile_90)
        self.assertAlmostEqual(
            metrics["induction"],
            (float(np.mean(self.amplitudes_pA[5:8])) - 10.0) / percentile_90,
        )
        self.assertAlmostEqual(
            metrics["recovery"],
            (
                float(np.mean(self.amplitudes_pA[8:12]))
                - float(np.mean(self.amplitudes_pA[0:4]))
            )
            / percentile_90,
        )
        self.assertTrue(metrics["stp_protocol_complete"])
        self.assertAlmostEqual(metrics["rest_voltage_mV"], -65.0)
        self.assertAlmostEqual(metrics["peak_voltage_mV"], -63.0)
        self.assertAlmostEqual(metrics["psp_amplitude_mV"], 2.0)
        self.assertGreater(metrics["single_event_peak_amplitude_pA"], 0.0)
        self.assertGreater(metrics["latency_ms"], self.delay)
        self.assertTrue(math.isfinite(metrics["rise_time_ms"]))
        self.assertTrue(math.isfinite(metrics["decay_time_ms"]))
        self.assertTrue(math.isfinite(metrics["half_width_ms"]))

    def test_six_pulse_train_reports_ppr_only(self) -> None:
        train, amplitudes = self._train_trace(6)
        metrics = calculate_manual_synapse_metrics(
            single_event_traces=self.single,
            train_traces=train,
            spike_times_ms=self.spikes[:6],
            netcon_delay_ms=self.delay,
        )

        self.assertFalse(metrics["stp_protocol_complete"])
        percentile_90 = float(np.percentile(amplitudes, 90))
        self.assertAlmostEqual(metrics["simple_ppr"], 12.0 / 10.0)
        self.assertAlmostEqual(metrics["ppr"], (12.0 - 10.0) / percentile_90)
        self.assertTrue(math.isnan(metrics["induction"]))
        self.assertTrue(math.isnan(metrics["recovery"]))

    def test_eight_pulse_train_adds_induction_but_not_recovery(self) -> None:
        train, amplitudes = self._train_trace(8)
        metrics = calculate_manual_synapse_metrics(
            single_event_traces=self.single,
            train_traces=train,
            spike_times_ms=self.spikes[:8],
            netcon_delay_ms=self.delay,
        )

        percentile_90 = float(np.percentile(amplitudes, 90))
        self.assertFalse(metrics["stp_protocol_complete"])
        self.assertTrue(math.isfinite(metrics["simple_ppr"]))
        self.assertTrue(math.isfinite(metrics["ppr"]))
        self.assertAlmostEqual(
            metrics["induction"],
            (float(np.mean(amplitudes[5:8])) - 10.0) / percentile_90,
        )
        self.assertTrue(math.isnan(metrics["recovery"]))

    def test_single_pulse_does_not_report_train_comparison_metrics(self) -> None:
        train, _ = self._train_trace(1)
        metrics = calculate_manual_synapse_metrics(
            single_event_traces=self.single,
            train_traces=train,
            spike_times_ms=self.spikes[:1],
            netcon_delay_ms=self.delay,
        )

        for name in ("simple_ppr", "ppr", "induction", "recovery"):
            self.assertTrue(math.isnan(metrics[name]))

    def test_rejects_unsorted_spike_times(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            calculate_manual_synapse_metrics(
                single_event_traces=self.single,
                train_traces=self.train,
                spike_times_ms=[100.0, 90.0],
                netcon_delay_ms=self.delay,
            )


class ManualSynapseNotebookTests(unittest.TestCase):
    def test_manual_probe_precedes_bmtool_initialization(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "4_synapses.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [
            (index, cell, "".join(cell.get("source", [])))
            for index, cell in enumerate(notebook.get("cells", []))
            if cell.get("cell_type") == "code"
        ]
        manual_index, _, manual_source = next(
            item for item in code_cells if "run_manual_synapse_probe(" in item[2]
        )
        bmtool_index = next(
            index for index, _, source in code_cells if "create_scp_synapse_tuner(" in source
        )

        self.assertLess(manual_index, bmtool_index)
        for control in (
            "MANUAL_SECTION_GROUP",
            "MANUAL_SPIKE_TIMES_MS",
            "MANUAL_PARAM_OVERRIDES",
            "MANUAL_SAVE_PATH",
        ):
            self.assertIn(control, manual_source)


if __name__ == "__main__":
    unittest.main()
