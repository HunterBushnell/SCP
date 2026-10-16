"""Opt-in synapse recordings use only built-ins and original synthetic fixtures."""
from __future__ import annotations

import contextlib
import copy
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import neuron
from neuron import h
import numpy as np

from modules.input_generation.recording_config import (
    SYNAPTIC_RECORDING_VARS,
    normalize_syn_recording_config,
    resolve_n_syn_recording_trials,
)
from modules.input_generation.sim_config import _normalize_sim_config
from modules.model.synapses import add_synapses
from modules.simulation.cell_runtime import run_cell
from modules.simulation.multi_run import run_multi
from modules.simulation.single_run import run_single
from modules.simulation.synaptic_recording import (
    finish_synaptic_recording,
    prepare_synaptic_recording,
    synaptic_recording_nbytes,
)


def config(**overrides):
    result = {
        "dt": 0.05, "tstart": 0.0, "tstop": 12.0, "bins": 1.0,
        "conditions": {"v_init_mV": -65.0, "celsius_C": 32.0},
        "n_trials": 3, "n_traces_to_save": 0, "n_inputs_to_save": 0,
        "save_input_stats": False, "log_input_summary": False,
        "regen_inputs_each_trial": False,
        "syn_recording": {
            "enabled": True, "n_trials": 1, "default_mode": "both",
            "default_sample_per_group": 2,
        },
    }
    result.update(overrides)
    return result


def new_cell():
    section = h.Section(name="synthetic_recording_soma")
    section.L = section.diam = 20.0
    section.insert("pas")
    return SimpleNamespace(h=h, soma=[section], synapses=[], netcons=[], stims=[], vecs=[])


def attach_builtins(cell, *args, **kwargs):
    group = []
    records = []
    for index, constructor in enumerate((h.ExpSyn, h.Exp2Syn, h.ExpSyn)):
        synapse = constructor(cell.soma[0](0.5))
        synapse.e = float(-5 * index)
        stimulus = h.NetStim()
        stimulus.start = 2.0
        stimulus.number = 2
        stimulus.interval = 3.0
        connection = h.NetCon(stimulus, synapse)
        connection.delay = 0.25
        connection.weight[0] = 0.003 * (index + 1)
        cell.synapses.append(synapse)
        cell.stims.append(stimulus)
        cell.netcons.append(connection)
        group.append(synapse)
        # Deliberately unlike native weights: recording must not multiply by it.
        records.append({"syn_id": 10 + index, "type": "builtin", "weight": 99 + index})
    return {"synapses": {"exc": group}, "records": {"exc": records}}


class FakeVector:
    def record(self, reference, *args):
        self.values = np.asarray(reference, dtype=float)
        return self

    def as_numpy(self):
        return self.values

    def __array__(self, dtype=None, copy=None):
        return np.array(self.values, dtype=dtype, copy=True)


class SynapticConfigTests(unittest.TestCase):
    def test_defaults_and_independent_trial_caps(self):
        self.assertFalse(normalize_syn_recording_config({})["enabled"])
        self.assertEqual(resolve_n_syn_recording_trials({}), 0)
        cfg = config(syn_recording={"enabled": "true"}, n_traces_to_save=4,
                     cell_recording={"n_trials": 99})
        self.assertEqual(resolve_n_syn_recording_trials(cfg), 4)
        self.assertEqual(resolve_n_syn_recording_trials(cfg, 2), 2)
        cfg["syn_recording"]["n_trials"] = 0
        self.assertEqual(resolve_n_syn_recording_trials(cfg), 0)
        self.assertEqual(normalize_syn_recording_config(cfg)["sample_seed"], 0)

    def test_group_overrides_merge_variables_without_mutation(self):
        cfg = config(syn_recording={
            "enabled": True, "default_mode": "sample", "default_sample_per_group": 5,
            "default_vars": {"g": False, "record_Pr": True},
            "groups": {"exc": {"mode": "both", "sample_per_group": 2, "vars": {"g": True}},
                       "off": False},
        })
        before = copy.deepcopy(cfg)
        normalized = normalize_syn_recording_config(cfg)
        self.assertEqual(cfg, before)
        self.assertEqual(normalized["groups"]["exc"]["mode"], "both")
        self.assertTrue(normalized["groups"]["exc"]["vars"]["record_Pr"])
        self.assertTrue(normalized["groups"]["exc"]["vars"]["g"])
        self.assertFalse(normalized["groups"]["off"]["enabled"])

    def test_normalized_pipeline_carries_the_full_config(self):
        normalized = _normalize_sim_config(config())
        self.assertEqual(normalized["syn_recording"]["n_trials"], 1)
        self.assertEqual(normalized["syn_recording"]["max_memory_mb"], 256)
        self.assertEqual(normalized["syn_recording"]["default_mode"], "both")

    def test_rejects_invalid_modes_vars_counts_intervals_and_budgets(self):
        for override in (
            {"default_mode": "all"}, {"default_vars": {"wrong": True}},
            {"n_trials": -1}, {"n_trials": 1.5}, {"default_sample_per_group": -2},
            {"record_dt_ms": float("nan")}, {"record_dt_ms": 0.01},
            {"max_memory_mb": 0}, {"max_memory_mb": float("inf")},
            {"groups": {"x": {"mode": "wrong"}}},
        ):
            with self.subTest(override=override), self.assertRaises((TypeError, ValueError)):
                normalize_syn_recording_config(config(syn_recording={"enabled": True, **override}))


class SynapticReferenceTests(unittest.TestCase):
    def fake_state(self, synapses):
        return {"synapses": {"test": synapses},
                "records": {"test": [{"syn_id": index + 20, "type": "synthetic"}
                                      for index in range(len(synapses))]}}

    def test_all_eight_variables_and_state_means(self):
        synapses = [SimpleNamespace(**{
            f"_ref_{name}": np.array([1.0, 2.0, 3.0]) * (index + 1)
            for name in SYNAPTIC_RECORDING_VARS
        }) for index in range(3)]
        cfg = config(syn_recording={"enabled": True, "n_trials": 1,
                     "default_mode": "both", "default_sample_per_group": 3,
                     "default_vars": {name: True for name in SYNAPTIC_RECORDING_VARS}})
        fake_h = SimpleNamespace(Vector=FakeVector, _ref_t=np.array([0.0, 1.0, 2.0]))
        with mock.patch("modules.simulation.synaptic_recording.h", fake_h):
            payload = finish_synaptic_recording(prepare_synaptic_recording(
                SimpleNamespace(), cfg, syn_state=self.fake_state(synapses)))
        group = payload["groups"]["test"]
        for variable in SYNAPTIC_RECORDING_VARS:
            samples = group["samples"][variable]
            expected = samples.mean(axis=0) if variable in {"record_use", "record_Pr"} else samples.sum(axis=0)
            np.testing.assert_array_equal(group["aggregate"][variable], expected)
            self.assertEqual(group["coverage"][variable]["available_count"], 3)
        self.assertEqual(group["aggregation"]["i"], "sum")
        self.assertEqual(group["aggregation"]["record_Pr"], "mean")
        self.assertEqual(payload["units"]["i"], "nA")
        self.assertEqual(payload["units"]["g"], "uS")
        self.assertEqual(payload["units"]["record_use"], "1")

    def test_missing_variables_are_reported_not_zero_filled(self):
        cfg = config(syn_recording={"enabled": True, "n_trials": 1,
                     "default_mode": "both", "default_sample_per_group": 2})
        state = self.fake_state([
            SimpleNamespace(_ref_i=np.array([1.0, 2.0]), _ref_g=np.array([3.0, 4.0])),
            SimpleNamespace(_ref_i=np.array([5.0, 6.0])),
        ])
        fake_h = SimpleNamespace(Vector=FakeVector, _ref_t=np.array([0.0, 1.0]))
        with mock.patch("modules.simulation.synaptic_recording.h", fake_h):
            payload = finish_synaptic_recording(prepare_synaptic_recording(SimpleNamespace(), cfg, syn_state=state))
        group = payload["groups"]["test"]
        self.assertNotIn("g", group["aggregate"])
        np.testing.assert_array_equal(group["aggregate"]["i"], [6.0, 8.0])
        self.assertEqual(group["coverage"]["g"]["missing_synapse_ids"], [21])
        self.assertEqual(group["coverage"]["g"]["aggregate_count"], 0)
        self.assertTrue(np.isnan(group["samples"]["g"][1]).all())

    def test_invalid_native_pointer_is_not_an_absent_optional_variable(self):
        failure = ValueError("native pointer is not initialized")

        class InvalidPointer:
            @property
            def _ref_i(self):
                raise failure

        with self.assertRaisesRegex(RuntimeError, r"'i'.*'test'.*syn_id=20.*synthetic") as caught:
            prepare_synaptic_recording(SimpleNamespace(), config(), syn_state=self.fake_state([InvalidPointer()]))
        self.assertIs(caught.exception.__cause__, failure)

        class ExposedWithoutReference:
            i = 1.0

        with self.assertRaisesRegex(RuntimeError, "exposed variable") as caught:
            prepare_synaptic_recording(SimpleNamespace(), config(), syn_state=self.fake_state([ExposedWithoutReference()]))
        self.assertIsInstance(caught.exception.__cause__, AttributeError)

    def test_record_failure_retains_native_error_and_context(self):
        failure = RuntimeError("invalid data handle")
        count = 0

        def make_vector():
            nonlocal count
            count += 1
            if count == 1:
                return FakeVector()
            result = mock.Mock()
            result.record.side_effect = failure
            return result

        state = self.fake_state([SimpleNamespace(_ref_i=np.array([1.0]))])
        fake_h = SimpleNamespace(Vector=make_vector, _ref_t=np.array([0.0]))
        with mock.patch("modules.simulation.synaptic_recording.h", fake_h):
            with self.assertRaisesRegex(RuntimeError, r"Cannot record.*syn_id=20") as caught:
                prepare_synaptic_recording(SimpleNamespace(), config(), syn_state=state)
        self.assertIs(caught.exception.__cause__, failure)

    def test_disabled_and_exhausted_capture_bypass_discovery_and_allocation(self):
        class NeverInspect:
            @property
            def synapses(self):
                raise AssertionError("disabled recording inspected synapses")

        with mock.patch("modules.simulation.synaptic_recording.h") as fake_h:
            for cfg, trial_idx in (
                (config(syn_recording={"enabled": False}), 0),
                (config(syn_recording={"enabled": True, "n_trials": 0}), 0),
                (config(), 1),
            ):
                self.assertIsNone(prepare_synaptic_recording(NeverInspect(), cfg, trial_idx=trial_idx))
            fake_h.Vector.assert_not_called()

    def test_memory_preflight_includes_retained_arrays_before_vector_allocation(self):
        state = self.fake_state([SimpleNamespace(_ref_i=np.array([1.0]))])
        cfg = config(syn_recording={"enabled": True, "n_trials": 1, "max_memory_mb": 0.001})
        with mock.patch("modules.simulation.synaptic_recording.h") as fake_h:
            with self.assertRaisesRegex(MemoryError, "Exact group aggregates"):
                prepare_synaptic_recording(SimpleNamespace(), cfg, syn_state=state)
            fake_h.Vector.assert_not_called()
        cfg["syn_recording"]["max_memory_mb"] = 1.0
        with mock.patch("modules.simulation.synaptic_recording.h") as fake_h:
            with self.assertRaisesRegex(MemoryError, "record_dt_ms"):
                prepare_synaptic_recording(SimpleNamespace(), cfg, syn_state=state, retained_bytes=1024**2)
            fake_h.Vector.assert_not_called()
        array = np.zeros(12)
        self.assertEqual(synaptic_recording_nbytes({"a": array, "b": [array]}), array.nbytes)


class NativeSynapticRecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        h.load_file("stdrun.hoc")

    def setUp(self):
        self.cell = new_cell()

    def test_exact_aggregates_and_samples_match_independent_native_vectors(self):
        state = attach_builtins(self.cell)
        independent = {
            variable: [h.Vector().record(getattr(synapse, f"_ref_{variable}"))
                       for synapse in self.cell.synapses]
            for variable in ("i", "g")
        }
        result = run_cell(self.cell, config(), syn_state=state)
        group = result["syn_recordings"]["groups"]["exc"]
        self.assertNotIn("I", result)
        self.assertNotIn("G", result)
        for variable, vectors in independent.items():
            values = np.array([np.array(vector) for vector in vectors])
            np.testing.assert_allclose(group["aggregate"][variable], values.sum(axis=0), atol=1e-15)
            self.assertGreater(np.max(np.abs(group["aggregate"][variable] - values[0])), 1e-5)
            np.testing.assert_array_equal(group["samples"][variable], values[group["sample_indices"]])
        self.assertEqual(group["synapse_ids"], [10 + index for index in group["sample_indices"]])
        np.testing.assert_array_equal(result["syn_recordings"]["T"], result["T"])
        np.testing.assert_allclose(np.diff(result["T"]), 0.05)

    def test_recording_does_not_change_voltage_or_global_rng(self):
        state = attach_builtins(self.cell)
        np.random.seed(1234)
        rng_state = np.random.get_state()
        recorded = run_cell(self.cell, config(), syn_state=state)
        after = np.random.get_state()
        np.testing.assert_array_equal(rng_state[1], after[1])
        self.assertEqual(rng_state[2:], after[2:])
        disabled = run_cell(self.cell, config(syn_recording={"enabled": False}), syn_state=state)
        np.testing.assert_array_equal(recorded["V"], disabled["V"])
        self.assertNotIn("syn_recordings", disabled)

    def test_sampling_reproducible_group_local_and_bounded(self):
        state = attach_builtins(self.cell)
        first = run_cell(self.cell, config(), syn_state=state)["syn_recordings"]
        second = run_cell(self.cell, config(), syn_state=state)["syn_recordings"]
        self.assertEqual(first["groups"]["exc"]["sample_indices"], second["groups"]["exc"]["sample_indices"])
        self.assertEqual(first["groups"]["exc"]["samples"]["i"].shape[0], 2)
        cfg = config(syn_recording={"enabled": True, "n_trials": 1, "default_mode": "sample",
                     "default_sample_per_group": 20})
        sampled = run_cell(self.cell, cfg, syn_state=state)["syn_recordings"]["groups"]["exc"]
        self.assertEqual(sampled["sample_indices"], [0, 1, 2])
        self.assertEqual(sampled["aggregate"], {})

    def test_separate_time_axis_optional_variables_and_unmapped_fallback(self):
        attach_builtins(self.cell)
        cfg = config(syn_recording={"enabled": True, "n_trials": 1, "record_dt_ms": 0.2,
                     "default_vars": {"i_AMPA": True}})
        result = run_cell(self.cell, cfg)
        payload = result["syn_recordings"]
        self.assertIn("unmapped", payload["groups"])
        self.assertIn("positional", payload["identity_source"])
        self.assertLess(len(payload["T"]), len(result["T"]))
        np.testing.assert_allclose(np.diff(payload["T"]), 0.2)
        group = payload["groups"]["unmapped"]
        self.assertNotIn("i_AMPA", group["aggregate"])
        self.assertEqual(group["coverage"]["i_AMPA"]["missing_synapse_ids"], [0, 1, 2])

    def test_group_override_can_disable_population(self):
        state = attach_builtins(self.cell)
        cfg = config(syn_recording={"enabled": True, "n_trials": 1, "groups": {"exc": False}})
        result = run_cell(self.cell, cfg, syn_state=state)
        self.assertNotIn("syn_recordings", result)

    def test_single_and_multi_capture_without_saved_voltage_and_respect_caps(self):
        cfg = config()
        with mock.patch("modules.simulation.single_run.synapses.add_synapses", side_effect=attach_builtins):
            single = run_single(self.cell, {}, cfg, {}, {})
        self.assertEqual(single["traces"], {})
        self.assertIsNotNone(single["syn_recordings"])
        self.assertIn("exc", single["syn_records"])
        cfg["syn_recording"]["n_trials"] = 2
        cfg["trial_offset"] = 17
        with mock.patch("modules.simulation.multi_run.synapses.add_synapses", side_effect=attach_builtins), contextlib.redirect_stdout(io.StringIO()):
            multi = run_multi(self.cell, {}, cfg, {}, {}, mode_registry={})
        self.assertEqual(multi["traces"], {})
        self.assertEqual([entry["trial_idx"] for entry in multi["syn_recordings_by_trial"]], [0, 1])
        self.assertEqual(multi["meta"]["n_syn_recording_trials"], 2)
        first, second = [entry["recordings"] for entry in multi["syn_recordings_by_trial"]]
        self.assertEqual(first["memory"]["retained_bytes"], 0)
        self.assertEqual(second["memory"]["retained_bytes"], synaptic_recording_nbytes(first))

    def test_snapshot_never_enables_recording_or_overrides_explicit_synaptic_cap(self):
        cfg = config(syn_recording={"enabled": False}, snapshot={"enabled": True, "force_deterministic": False})
        with mock.patch("modules.simulation.single_run.synapses.add_synapses", side_effect=attach_builtins):
            result = run_single(self.cell, {}, cfg, {}, {})
        self.assertIsNone(result["syn_recordings"])
        cfg = config(snapshot={"enabled": True, "force_deterministic": False})
        with mock.patch("modules.simulation.multi_run.synapses.add_synapses", side_effect=attach_builtins), contextlib.redirect_stdout(io.StringIO()):
            result = run_multi(self.cell, {}, cfg, {}, {}, mode_registry={})
        self.assertEqual(len(result["syn_recordings_by_trial"]), 1)


class NativeSCPAttachmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        h.load_file("stdrun.hoc")
        compiler = Path(sys.executable).with_name("nrnivmodl")
        if not compiler.exists() or shutil.which("c++") is None:
            raise unittest.SkipTest("synthetic MOD compilation requires nrnivmodl and c++")
        cls.build = tempfile.TemporaryDirectory(prefix="scp_recording_fixture_")
        cls.addClassCleanup(cls.build.cleanup)
        fixture = Path(__file__).parent / "fixtures" / "synaptic_recording"
        environment = dict(os.environ)
        environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
        compiled = subprocess.run([str(compiler), str(fixture)], cwd=cls.build.name,
                                  env=environment, capture_output=True, text=True, timeout=90)
        if compiled.returncode:
            raise RuntimeError(f"Original synthetic recording fixture did not compile:\n{compiled.stdout}\n{compiled.stderr}")
        libraries = list(Path(cls.build.name).glob("*/libnrnmech.so")) + list(Path(cls.build.name).glob("*/.libs/libnrnmech.so"))
        if not libraries:
            raise RuntimeError("Synthetic fixture compiler did not produce a mechanism library")
        h.nrn_load_dll(str(libraries[0]))

    def _fixture_workflow(self):
        class RegularVectorStim(neuron.hclass(h.NetStim)):
            """Test-only VecStim adapter for regular trains, using native events."""
            def play(self, vector):
                times = np.array(vector)
                if len(times) > 2:
                    np.testing.assert_allclose(np.diff(times), times[1] - times[0])
                self.start = float(times[0])
                self.number = len(times)
                self.interval = float(times[1] - times[0]) if len(times) > 1 else 1.0
                self.noise = 0.0

        class FixtureHoc:
            def __getattr__(self, name):
                return RegularVectorStim if name == "VecStim" else getattr(h, name)

            def __setattr__(self, name, value):
                setattr(h, name, value)

        cell = new_cell()
        cell.h = FixtureHoc()
        h.distance(0, cell.soma[0](0.5))
        geometry = {"groups": {"soma": [SimpleNamespace(sec=cell.soma[0], x=0.5)]}}
        groups = {"exc": {"state": True, "syns": {
            "type": "ScpRecordingSyn", "N_syn_resolved": 3, "segs": "soma",
            "params": {"initW": 0.004, "wt_std": 0.0},
        }}}
        inputs = {"exc": SimpleNamespace(mode="synthetic", spike_trains=[np.array([2.0, 4.0, 6.0])], meta={})}
        cfg = config(syn_recording={"enabled": True, "n_trials": 1,
                     "default_mode": "both", "default_sample_per_group": 3,
                     "default_vars": {variable: True for variable in SYNAPTIC_RECORDING_VARS}})
        return cell, geometry, cfg, groups, inputs

    def test_unmodified_scp_attachment_records_native_all_variables_and_weights(self):
        cell, geometry, cfg, groups, inputs = self._fixture_workflow()
        state = add_synapses(cell, geometry, cfg, groups, inputs)
        native = {variable: [h.Vector().record(getattr(syn, f"_ref_{variable}"))
                             for syn in cell.synapses] for variable in SYNAPTIC_RECORDING_VARS}
        result = run_cell(cell, cfg, syn_state=state)
        recorded = result["syn_recordings"]["groups"]["exc"]
        self.assertEqual(len(state["records"]["exc"]), 3)
        self.assertTrue(all(record.weight == 0.004 for record in state["records"]["exc"]))
        for variable, vectors in native.items():
            values = np.array([np.array(vector) for vector in vectors])
            np.testing.assert_array_equal(recorded["samples"][variable], values)
            expected = values.mean(axis=0) if variable in {"record_use", "record_Pr"} else values.sum(axis=0)
            np.testing.assert_allclose(recorded["aggregate"][variable], expected, atol=1e-15)
        self.assertGreater(np.max(recorded["aggregate"]["g"]), 0.01)
        self.assertGreater(np.max(np.abs(recorded["aggregate"]["i"])), 0.01)

    def test_full_native_single_and_multi_workflows_without_mocked_attachment(self):
        cell, geometry, cfg, groups, inputs = self._fixture_workflow()
        single = run_single(cell, geometry, cfg, groups, inputs)
        self.assertEqual(single["traces"], {})
        self.assertEqual(single["syn_records"]["exc"][0].weight, 0.004)
        self.assertGreater(np.max(single["syn_recordings"]["groups"]["exc"]["aggregate"]["g"]), 0.01)
        cfg["syn_recording"]["n_trials"] = 2
        with contextlib.redirect_stdout(io.StringIO()):
            multi = run_multi(cell, geometry, cfg, groups, inputs, mode_registry={})
        self.assertEqual(len(multi["spikes"]), 3)
        self.assertEqual(multi["traces"], {})
        self.assertEqual([entry["trial_idx"] for entry in multi["syn_recordings_by_trial"]], [0, 1])
        for entry in multi["syn_recordings_by_trial"]:
            group = entry["recordings"]["groups"]["exc"]
            self.assertEqual(set(group["aggregate"]), set(SYNAPTIC_RECORDING_VARS))
            np.testing.assert_allclose(group["aggregate"]["g"], group["samples"]["g"].sum(axis=0))


if __name__ == "__main__":
    unittest.main()
