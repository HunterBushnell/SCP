from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from neuron import h

from modules.model.load_cell import load_cell
from modules.simulation.cell_runtime import _build_cell_recorders_for_site, run_cell
from modules.simulation.current_injection import (
    get_rec_vars_for_i_in_sec,
    run_current_injection,
    run_iclamp_test,
)
from modules.simulation.recording import (
    _record_mechanism_current,
    _resolve_mechanism_current_reference,
)
from modules.simulation.result_loading import load_results
from modules.simulation.session import SimulationOptions, SimulationSession
from modules.tuning.active import active_metric_rows, run_active_protocol
from modules.tuning.passive import run_passive_protocol


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "extracellular"
CONDITIONS = {"v_init_mV": -70.0, "celsius_C": 32.0}
CURRENT_KEYS = {"pas.i", "extracellular.i_membrane"}


def cell_config() -> dict:
    return {
        "cell_name": "synthetic_extracellular",
        "cell_loader": "hoc_template",
        "paths": {
            "hoc_template": str(FIXTURE / "model" / "ExtracellularCells.hoc"),
            "modfiles": None,
        },
        "hoc_template": {
            "template_name": "ScpExtracellularCell",
            "constructor_args": [],
            "section_map": {
                "soma": "somatic", "dend": "basal", "apic": "apical",
                "axon": "axonal", "all": "all",
            },
        },
    }


def protocol() -> dict:
    return {
        "stim_amp": 0.025, "stim_delay": 5.0, "stim_dur": 10.0,
        "h_tstop": 20.0, "h_dt": 0.025, "conditions": dict(CONDITIONS),
    }


def fake_segment(**attributes) -> SimpleNamespace:
    return SimpleNamespace(
        sec=SimpleNamespace(name=lambda: "synthetic_soma"), x=0.5, **attributes,
    )


class CurrentReferenceTests(unittest.TestCase):
    def test_mechanism_reference_takes_precedence(self) -> None:
        reference = object()
        segment = fake_segment(
            test=SimpleNamespace(_ref_i=reference), _ref_i_test=object(), _ref_i=object(),
        )
        self.assertIs(_resolve_mechanism_current_reference(segment, "test", "i"), reference)

    def test_qualified_reference_is_used_without_bare_fallback(self) -> None:
        reference = object()
        segment = fake_segment(test=SimpleNamespace(), _ref_i_test=reference, _ref_i=object())
        self.assertIs(_resolve_mechanism_current_reference(segment, "test", "i"), reference)

        del segment._ref_i_test
        with self.assertRaisesRegex(RuntimeError, r"test\.i.*synthetic_soma") as caught:
            _resolve_mechanism_current_reference(segment, "test", "i")
        self.assertIsInstance(caught.exception.__cause__, AttributeError)

    def test_missing_mechanism_still_allows_qualified_reference(self) -> None:
        reference = object()
        self.assertIs(
            _resolve_mechanism_current_reference(fake_segment(_ref_i_test=reference), "test", "i"),
            reference,
        )

    def test_builtin_membrane_current_resolves_on_segment(self) -> None:
        cell = load_cell(cell_config(), base_dir=FIXTURE)
        segment = cell.soma[0](0.5)
        self.assertFalse(hasattr(segment.extracellular, "_ref_i_membrane"))
        reference = _resolve_mechanism_current_reference(segment, "extracellular", "i_membrane")
        self.assertEqual(reference[0], segment.i_membrane)

    def test_native_pointer_reference_and_failure_are_preserved(self) -> None:
        reference = object()
        segment = fake_segment(pointer=SimpleNamespace(_ref_im=reference))
        vector = mock.Mock()
        with mock.patch("modules.simulation.recording.h", SimpleNamespace(Vector=lambda: vector)):
            self.assertIs(_record_mechanism_current(segment, "pointer", "im"), vector.record.return_value)
        vector.record.assert_called_once_with(reference)

        failure = ValueError("native POINTER is not initialized")

        class InvalidPointer:
            @property
            def _ref_im(self):
                raise failure

        segment.pointer = InvalidPointer()
        segment._ref_im_pointer = object()
        with self.assertRaisesRegex(RuntimeError, r"pointer\.im.*synthetic_soma") as caught:
            _resolve_mechanism_current_reference(segment, "pointer", "im")
        self.assertIs(caught.exception.__cause__, failure)

    def test_recorder_failure_keeps_original_exception(self) -> None:
        failure = RuntimeError("invalid native data handle")
        vector = mock.Mock()
        vector.record.side_effect = failure
        with mock.patch("modules.simulation.recording.h", SimpleNamespace(Vector=lambda: vector)):
            with self.assertRaisesRegex(RuntimeError, r"test\.i.*synthetic_soma") as caught:
                _record_mechanism_current(fake_segment(test=SimpleNamespace(_ref_i=object())), "test", "i")
        self.assertIs(caught.exception.__cause__, failure)

    def test_both_recording_paths_report_unresolved_requested_currents(self) -> None:
        segment = fake_segment(test=SimpleNamespace(), _ref_i=object())
        section = mock.Mock()
        section.return_value = segment
        section.psection.return_value = {"density_mechs": {"test": {"i": [0.0]}}}
        segment.sec = section
        section.name.return_value = "unresolved_soma"
        for operation in (
            lambda: get_rec_vars_for_i_in_sec(section, 0.5),
            lambda: _build_cell_recorders_for_site(segment, {"v": False, "mech_currents": True}),
        ):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(RuntimeError, r"test\.i.*unresolved_soma"):
                    operation()


class ExtracellularWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cell = load_cell(cell_config(), base_dir=FIXTURE)

    def assert_trace(self, times, voltage, currents) -> None:
        self.assertGreater(len(times), 2)
        self.assertEqual(len(times), len(voltage))
        self.assertTrue(np.isfinite(times).all())
        self.assertTrue(np.isfinite(voltage).all())
        self.assertAlmostEqual(float(voltage[0]), CONDITIONS["v_init_mV"])
        self.assertEqual(set(currents), CURRENT_KEYS)
        for current in currents.values():
            self.assertEqual(len(current), len(times))
            self.assertTrue(np.isfinite(current).all())
        self.assertAlmostEqual(float(h.celsius), CONDITIONS["celsius_C"])

    def test_passive_active_and_fi_sweeps_record_builtin_currents(self) -> None:
        for runner, amps in (
            (run_passive_protocol, [-25.0, -50.0]),
            (run_active_protocol, [25.0, 50.0]),
            (run_active_protocol, [0.0, 25.0, 50.0]),
        ):
            h.celsius = 10.0
            records = runner(cell=self.cell, sim_params=protocol(), sim_amps=amps)
            for amp in amps:
                self.assert_trace(records["T"][amp], records["V"][amp], records["I"][amp])
                self.assertTrue(np.isfinite(records["F"][amp]))
            rows = active_metric_rows(looped_records=records, sim_params=protocol(), sim_amps=amps)
            self.assertEqual(len(rows), len(amps))

    def test_current_injection_records_the_actual_reference_values(self) -> None:
        segment = self.cell.soma[0](0.5)
        direct_membrane = h.Vector().record(segment._ref_i_membrane)
        direct_pas = h.Vector().record(segment.pas._ref_i)
        result = run_current_injection(self.cell, protocol())
        self.assert_trace(result["T"], result["V"], result["I"])
        np.testing.assert_array_equal(result["I"]["extracellular.i_membrane"], np.asarray(direct_membrane))
        np.testing.assert_array_equal(result["I"]["pas.i"], np.asarray(direct_pas))

    def test_disabled_currents_bypass_discovery_and_preserve_voltage(self) -> None:
        with_currents = run_current_injection(self.cell, protocol())
        with mock.patch(
            "modules.simulation.current_injection.get_rec_vars_for_i_in_sec",
            side_effect=AssertionError("current discovery must be bypassed"),
        ) as discovery:
            without_currents = run_current_injection(self.cell, protocol(), record_currents=False)
        discovery.assert_not_called()
        self.assertEqual(without_currents["I"], {})
        np.testing.assert_array_equal(without_currents["T"], with_currents["T"])
        np.testing.assert_array_equal(without_currents["V"], with_currents["V"])
        self.assertEqual(without_currents["F"], with_currents["F"])

    def test_step5_iclamp_forwards_toggle_before_discovery(self) -> None:
        cfg = {
            "dt": 0.025, "tstop": 20.0, "conditions": dict(CONDITIONS),
            "iclamp": {"amp_nA": 0.025, "delay_ms": 5.0, "dur_ms": 10.0},
        }
        with mock.patch(
            "modules.simulation.current_injection.get_rec_vars_for_i_in_sec",
            side_effect=AssertionError("current discovery must be bypassed"),
        ) as discovery:
            result = run_iclamp_test(self.cell, cfg)
        discovery.assert_not_called()
        self.assertNotIn("I", result["iclamp"])
        self.assertFalse(result["meta"]["record_currents"])

        enabled = run_iclamp_test(self.cell, cfg, {"record_currents": True})
        self.assert_trace(enabled["traces"]["T"], enabled["traces"]["V"], enabled["iclamp"]["I"])

    def test_step5_cell_recording_uses_same_current_references(self) -> None:
        segment = self.cell.soma[0](0.5)
        direct_membrane = h.Vector().record(segment._ref_i_membrane)
        direct_pas = h.Vector().record(segment.pas._ref_i)
        trace = run_cell(self.cell, {
            "dt": 0.025, "tstop": 20.0, "conditions": dict(CONDITIONS),
            "cell_recording": {
                "enabled": True,
                "sites": [{"sec": "soma", "idx": 0, "x": 0.5, "label": "soma"}],
                "vars": {"v": False, "mech_currents": True},
            },
        })
        self.assert_trace(trace["T"], trace["V"], trace["cell_recordings"]["soma"])
        np.testing.assert_array_equal(
            trace["cell_recordings"]["soma"]["extracellular.i_membrane"],
            np.asarray(direct_membrane),
        )
        np.testing.assert_array_equal(trace["cell_recordings"]["soma"]["pas.i"], np.asarray(direct_pas))

    def test_saved_step5_results_keep_existing_current_toggle_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp) / "tune"
            configs = tune / "cell_configs"
            configs.mkdir(parents=True)
            (configs / "cell_config.json").write_text(json.dumps(cell_config()), encoding="utf-8")
            (configs / "geometry.json").write_text("{}", encoding="utf-8")
            (configs / "sim_config.json").write_text(json.dumps({
                "dt": 0.025, "tstart": 0.0, "tstop": 20.0, "conditions": dict(CONDITIONS),
                "save": {"enabled": True, "format": "pkl", "full_results": True},
                "plots_profile": "off",
                "iclamp": {"amp_nA": 0.025, "delay_ms": 5.0, "dur_ms": 10.0},
            }), encoding="utf-8")
            for enabled in (False, True):
                session = SimulationSession.from_tune(tune, options=SimulationOptions(
                    iclamp=True, output_stem=f"current_recording_{enabled}",
                    sim_overrides={"iclamp": {"record_currents": enabled}},
                ))
                result = session.run()
                saved = session.save()
                self.assertIsNotNone(saved)
                loaded = load_results(saved)
                self.assertEqual(loaded["meta"]["record_currents"], enabled)
                if enabled:
                    self.assert_trace(loaded["traces"]["T"], loaded["traces"]["V"], loaded["iclamp"]["I"])
                else:
                    self.assertNotIn("I", result["iclamp"])
                    self.assertNotIn("I", loaded["iclamp"])
                self.assertFalse((configs / "syn_config.json").exists())
                self.assertFalse((configs / "target_config.json").exists())

    def test_optional_groups_are_empty_and_multiple_templates_stay_scoped(self) -> None:
        cfg = cell_config()
        cfg["hoc_template"] = {"template_name": "ScpExtracellularAlternative"}
        alternative = load_cell(cfg, base_dir=FIXTURE)
        self.assertEqual(self.cell.apic, ())
        self.assertEqual(self.cell.axon, ())
        self.assertEqual(len(self.cell.all), 2)
        self.assertEqual(len(alternative.all), 1)
        self.assertTrue(
            {sec.name() for sec in self.cell.all}.isdisjoint(sec.name() for sec in alternative.all)
        )


if __name__ == "__main__":
    unittest.main()
