"""Result-only contracts for measured synaptic recordings and legacy payloads."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from modules.simulation.result_appending import (
    _append_results_to_path,
    _ensure_multi_results,
    append_multi_results,
)
from modules.simulation.result_loading import load_results
from modules.simulation.result_saving import _write_results_file, save_results
from modules.workspaces import WorkspaceMismatchError
from scripts import merge_array_results, merge_two_runs


def _recording(value=1.0, *, times=None):
    times = np.array([0.0, 0.5, 1.0]) if times is None else np.asarray(times)
    return {
        "format_version": 1,
        "T": times,
        "units": {"i": "nA", "g": "uS"},
        "groups": {"exc": {
            "mode": "both", "n_synapses": 3,
            "synapse_ids": ["exc:1"], "sample_indices": [1],
            "aggregate": {"i": np.full(times.size, value)},
            "samples": {"i": np.full((1, times.size), value / 3)},
            "coverage": {"i": {"available_count": 3, "missing_synapse_ids": [],
                                   "aggregate_count": 3, "sample_count": 1}},
            "aggregation": {"i": "sum"},
        }},
    }


def _result(*, mode="multi", stem="run", n_trials=1, cap=3, voltage=False):
    result = {
        "mode": mode,
        "sim_cfg": {
            "cell": "PV", "tune": "tuned", "n_trials": n_trials,
            "tstart": 0.0, "tstop": 10.0, "dt": 0.1, "bins": 1.0,
            "output": stem, "save_output": True, "save_model_artifacts": False,
            "save_fit_json_sidecar": False, "save_sidecars": True,
            "save_full_results": True, "output_format": "pickle",
            "n_traces_to_save": 0, "n_inputs_to_save": 0,
            "syn_recording": {"enabled": True, "n_trials": cap},
        },
        "traces": {}, "meta": {"cell_config": {}, "geometry_config": {}, "syn_config": {}},
        "syn_records": {"exc": [{"synapse_id": "exc:1", "section": "dend[0]"}]},
    }
    if mode == "single":
        result["spikes"] = np.array([2.0])
        result["syn_recordings"] = _recording()
    else:
        result["spikes"] = [np.array([2.0])] * n_trials
        result["syn_recordings_by_trial"] = [
            {"trial_idx": idx, "recordings": _recording(idx + 1)}
            for idx in range(n_trials)
        ]
        result["syn_records_by_trial"] = [
            {"trial_idx": idx, "records": copy.deepcopy(result["syn_records"])}
            for idx in range(n_trials)
        ]
    if voltage:
        traces = {"T": np.array([0.0, 1.0]), "V": np.array([-65., -64.]),
                  "I": np.array([0.1, 0.2]), "G": np.array([0.2, 0.3])}
        result["traces"] = traces if mode == "single" else {
            key: value if key == "T" else [value] for key, value in traces.items()
        }
    return result


class SynapticRecordingResultTests(unittest.TestCase):
    def assertRecording(self, actual, expected):
        self.assertEqual(actual["format_version"], 1)
        self.assertEqual(actual["units"], expected["units"])
        np.testing.assert_array_equal(actual["T"], expected["T"])
        group = actual["groups"]["exc"]
        source = expected["groups"]["exc"]
        for key in ("mode", "n_synapses", "synapse_ids", "sample_indices", "coverage", "aggregation"):
            self.assertEqual(group[key], source[key])
        for key in ("aggregate", "samples"):
            np.testing.assert_array_equal(group[key]["i"], source[key]["i"])

    def test_roundtrip_single_multi_full_pickle_npz_and_manifest_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ("single", "multi"):
                for fmt, full, sidecars in (("pickle", True, False), ("npz", True, False),
                                             ("pickle", False, True)):
                    with self.subTest(mode=mode, fmt=fmt, full=full):
                        source = _result(mode=mode, stem=f"{mode}_{fmt}_{full}", voltage=True)
                        source["sim_cfg"].update(output_format=fmt, save_full_results=full, save_sidecars=sidecars)
                        # Saved payloads must survive conversion even if a new
                        # working capture limit is smaller than their contents.
                        source["sim_cfg"]["syn_recording"]["n_trials"] = 0
                        saved = Path(save_results(source, base_dir=tmp))
                        for path in (saved, saved.parent / "run_manifest.json"):
                            loaded = load_results(path)
                            key = "syn_recordings" if mode == "single" else "syn_recordings_by_trial"
                            actual = loaded[key] if mode == "single" else loaded[key][0]["recordings"]
                            expected = source[key] if mode == "single" else source[key][0]["recordings"]
                            self.assertRecording(actual, expected)
                            self.assertEqual(loaded["syn_records"], source["syn_records"])
                            for variable in ("I", "G"):
                                np.testing.assert_array_equal(loaded["traces"][variable], source["traces"][variable])
                        if not full:
                            files = json.loads(saved.read_text())["files"]
                            self.assertIn(key, files)
                            self.assertIn("syn_records", files)
                            self.assertNotEqual(files[key], files["syn_records"])

    def test_synaptic_time_axis_roundtrips_without_voltage(self):
        with tempfile.TemporaryDirectory() as tmp:
            for fmt, full in (("pickle", True), ("npz", True), ("pickle", False)):
                source = _result(mode="single", stem=f"novoltage_{fmt}_{full}")
                source["sim_cfg"].update(output_format=fmt, save_full_results=full)
                loaded = load_results(save_results(source, base_dir=tmp))
                self.assertEqual(loaded["traces"], {})
                self.assertRecording(loaded["syn_recordings"], source["syn_recordings"])

    def test_lowlevel_npz_writer_preserves_new_and_placement_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _result(n_trials=2)
            target = Path(tmp) / "standalone.npz"
            _write_results_file(source, target, "npz")
            loaded = load_results(target)
            self.assertEqual([entry["trial_idx"] for entry in loaded["syn_recordings_by_trial"]], [0, 1])
            self.assertEqual(loaded["syn_records_by_trial"], source["syn_records_by_trial"])
            self.assertRecording(loaded["syn_recordings_by_trial"][1]["recordings"], _recording(2))

    def test_legacy_npz_and_sidecars_without_new_key_still_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "legacy.npz"
            np.savez(target, mode=np.array("single"), sim_cfg_json=np.array("{}"),
                     meta_json=np.array("{}"), T=np.array([0, 1]),
                     I=np.array([3, 4]), G=np.array([5, 6]))
            loaded = load_results(target)
            self.assertNotIn("syn_recordings", loaded)
            np.testing.assert_array_equal(loaded["traces"]["I"], [3, 4])
            source = _result(stem="legacy")
            source.pop("syn_recordings_by_trial")
            source["sim_cfg"]["save_full_results"] = False
            self.assertNotIn("syn_recordings_by_trial", load_results(save_results(source, base_dir=root)))

    def test_all_single_coercions_preserve_and_copy_recordings(self):
        source = _result(mode="single", voltage=True)
        for coerce in (_ensure_multi_results, merge_array_results._ensure_multi, merge_two_runs._ensure_multi):
            with self.subTest(coerce=coerce.__module__):
                coerced = coerce(source)
                entry = coerced["syn_recordings_by_trial"][0]
                self.assertEqual(entry["trial_idx"], 0)
                self.assertRecording(entry["recordings"], source["syn_recordings"])
                entry["recordings"]["T"][0] = 99
                self.assertEqual(source["syn_recordings"]["T"][0], 0)
                self.assertEqual(coerced["syn_records_by_trial"][0]["records"], source["syn_records"])
                np.testing.assert_array_equal(coerced["traces"]["I"][0], source["traces"]["I"])

    def test_append_uses_trial_offsets_independent_cap_and_copies(self):
        base, incoming = _result(n_trials=2, cap=3), _result(n_trials=2, cap=3)
        merged = append_multi_results(base, incoming)
        self.assertEqual(len(merged["spikes"]), 4)
        self.assertEqual(merged["traces"], {})
        self.assertEqual([entry["trial_idx"] for entry in merged["syn_recordings_by_trial"]], [0, 1, 2])
        self.assertEqual([entry["trial_idx"] for entry in merged["syn_records_by_trial"]], [0, 1, 2, 3])
        merged["syn_recordings_by_trial"][2]["recordings"]["groups"]["exc"]["aggregate"]["i"][0] = 99
        self.assertEqual(incoming["syn_recordings_by_trial"][0]["recordings"]["groups"]["exc"]["aggregate"]["i"][0], 1)
        self.assertEqual(len(base["syn_recordings_by_trial"]), 2)

    def test_append_null_cap_falls_back_to_top_level_not_cell_recording(self):
        base, incoming = _result(cap=None), _result()
        base["sim_cfg"].update(n_traces_to_save=2, cell_recording={"n_trials": 0})
        merged = append_multi_results(base, incoming)
        self.assertEqual(len(merged["syn_recordings_by_trial"]), 2)
        self.assertEqual(merged["traces"], {})

    def test_append_incoming_only_and_zero_cap(self):
        base, incoming = _result(n_trials=2), _result()
        base.pop("syn_recordings_by_trial")
        merged = append_multi_results(base, incoming)
        self.assertEqual(merged["syn_recordings_by_trial"][0]["trial_idx"], 2)
        base["sim_cfg"]["syn_recording"]["n_trials"] = 0
        self.assertIsNone(append_multi_results(base, incoming)["syn_recordings_by_trial"])

    def test_mismatched_recording_axes_are_rejected_without_mutation(self):
        for times in ([0, 1], [0, 0.6, 1.2]):
            base, incoming = _result(), _result()
            incoming["syn_recordings_by_trial"][0]["recordings"] = _recording(times=times)
            with self.assertRaisesRegex(ValueError, "syn_recording_time_mismatch"):
                append_multi_results(base, incoming)
            self.assertEqual(len(base["spikes"]), 1)

    def test_workspace_guard_still_rejects_foreign_recordings(self):
        base, incoming = _result(), _result()
        incoming["sim_cfg"]["cell"] = "SST"
        with self.assertRaises(WorkspaceMismatchError):
            append_multi_results(base, incoming)

    def test_append_to_npz_file_and_manifest_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            for target_kind in ("file", "directory"):
                source = _result(mode="single", stem=target_kind)
                source["sim_cfg"]["output_format"] = "npz"
                saved = Path(save_results(source, base_dir=tmp))
                target = saved if target_kind == "file" else saved.parent
                _append_results_to_path(_result(mode="single"), target)
                for path in (saved, saved.parent / "run_manifest.json"):
                    loaded = load_results(path)
                    self.assertEqual([entry["trial_idx"] for entry in loaded["syn_recordings_by_trial"]], [0, 1])

    def test_array_merge_offsets_and_independent_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            parts, output = Path(tmp) / "parts", Path(tmp) / "merged"
            for idx, mode in enumerate(("multi", "single")):
                save_results(_result(mode=mode, stem=f"slurm_123_{idx}", n_trials=2, cap=3), base_dir=parts)
            argv = ["merge_array_results.py", "--input-dir", str(parts), "--output-dir", str(output), "--job-id", "123"]
            with mock.patch.object(sys, "argv", argv):
                merge_array_results.main()
            merged = load_results(output / "slurm_merged_123")
            self.assertEqual(len(merged["spikes"]), 3)
            self.assertEqual([entry["trial_idx"] for entry in merged["syn_recordings_by_trial"]], [0, 1, 2])
            self.assertEqual([entry["trial_idx"] for entry in merged["syn_records_by_trial"]], [0, 1, 2])
            self.assertEqual(merged["traces"], {})

    def test_array_merge_rejects_mismatched_recording_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            parts = Path(tmp) / "parts"
            for idx in range(2):
                result = _result(stem=f"slurm_123_{idx}")
                if idx:
                    result["syn_recordings_by_trial"][0]["recordings"] = _recording(times=[0, 1])
                save_results(result, base_dir=parts)
            argv = ["merge_array_results.py", "--input-dir", str(parts), "--output-dir", str(Path(tmp) / "merged"), "--job-id", "123"]
            with mock.patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "syn_recording_time_mismatch"):
                merge_array_results.main()

    def test_two_run_merge_preserves_recordings_and_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = save_results(_result(mode="single", stem="a", cap=2), base_dir=root / "parts")
            b = save_results(_result(stem="b", n_trials=2, cap=2), base_dir=root / "parts")
            report = merge_two_runs.merge_two_runs(run_a=a, run_b=b, output_dir=root / "merged",
                                                   dry_run=False, keep_logs=False)
            self.assertFalse(report.errors, report.errors)
            merged = load_results(report.saved_path)
            self.assertEqual(len(merged["spikes"]), 3)
            self.assertEqual([entry["trial_idx"] for entry in merged["syn_recordings_by_trial"]], [0, 1])
            self.assertEqual([entry["trial_idx"] for entry in merged["syn_records_by_trial"]], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
