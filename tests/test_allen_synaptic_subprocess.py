"""Fresh-process acceptance of the real, file-backed public Step 5 baselines."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

from tests import test_allen_step5_subprocess as allen_step5


RESULT_SENTINEL = "SCP_SYNAPTIC_ACCEPTANCE_JSON="
REPO_ROOT = allen_step5.REPO_ROOT
SUBPROCESS_TIMEOUT_SECONDS = allen_step5.SUBPROCESS_TIMEOUT_SECONDS
_SUBPROCESS_PROGRAM = textwrap.dedent(
    f"""
    import csv
    import json
    import sys
    from pathlib import Path
    import numpy as np

    repo_root = Path(sys.argv[1]).resolve()
    tune_dir = Path(sys.argv[2]).resolve()
    sys.path.insert(0, str(repo_root))
    from modules.simulation import SimulationOptions, SimulationSession

    options = SimulationOptions(
        mode="single", n_trials=1, seed=20261014, load_mechanisms=True,
        sim_overrides={{
            "dt": 0.1, "jitter": 0, "randomness_mode": "fixed",
            "save": {{"enabled": False}}, "save_output": False,
            "save_plots": False, "plots_profile": "off",
            "snapshot": {{"enabled": False}},
            "append": {{"enabled": False, "path": None}},
            "append_enabled": False, "append_to": None,
            "n_traces_to_save": 1, "n_inputs_to_save": 1,
            "syn_recording": {{
                "enabled": True, "n_trials": 1, "default_mode": "sample",
                "default_sample_per_group": 3, "sample_seed": 0,
                "record_dt_ms": 0.5, "max_memory_mb": 256,
                "default_vars": {{"i": True, "g": True}}, "groups": {{}},
            }},
        }},
    )
    session = SimulationSession.from_tune(tune_dir, options=options)
    result = session.run()
    traces = result["traces"]
    times, voltages = np.asarray(traces["T"]), np.asarray(traces["V"])
    recordings = result["syn_recordings"]
    syn_times = np.asarray(recordings["T"])
    groups = recordings["groups"]
    pn_cfg = session.groups_cfg["pn_exc"]
    csv_blocks = [block for block in pn_cfg["input_blocks"]
                  if block["mode"] == "inhomogeneous_poisson"]
    assert len(csv_blocks) == 1, "Public pn_exc must retain its CSV stimulus block"
    csv_block = csv_blocks[0]
    source = csv_block["source"]
    csv_path = Path(source["path"])
    assert csv_path.is_file(), f"Missing real input CSV: {{csv_path}}"
    with csv_path.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    assert source_rows, "Real input CSV must not be empty"
    assert any(float(row[source["rate_col"]]) > 0 for row in source_rows)
    input_trains = session.inputs_by_group["pn_exc"].spike_trains
    csv_spike_count = sum(int(np.count_nonzero(
        (np.asarray(train) >= csv_block["start_ms"]) &
        (np.asarray(train) < csv_block["stop_ms"])
    )) for train in input_trains)
    group_summary = {{}}
    for name, group in groups.items():
        arrays = {{key: np.asarray(values) for key, values in group["samples"].items()}}
        group_summary[name] = {{
            "mode": group["mode"], "n_synapses": group["n_synapses"],
            "sample_count": len(group["synapse_ids"]),
            "variables": sorted(arrays),
            "matching_time_axis": all(values.ndim == 2 and values.shape ==
                                      (len(group["synapse_ids"]), syn_times.size)
                                      for values in arrays.values()),
            "finite": all(np.isfinite(values).all() for values in arrays.values()),
        }}
    payload = {{
        "mode": result["mode"], "tstop": result["sim_cfg"]["tstop"],
        "sample_count": times.size, "last_time_ms": float(times[-1]),
        "matching_voltage_lengths": times.shape == voltages.shape,
        "finite_voltage": bool(times.size and voltages.size and
                                np.isfinite(times).all() and np.isfinite(voltages).all()),
        "syn_sample_count": syn_times.size,
        "syn_finite_time": bool(syn_times.ndim == 1 and syn_times.size and np.isfinite(syn_times).all()),
        "syn_last_time_ms": float(syn_times[-1]),
        "syn_time_step_ms": float(np.diff(syn_times).mean()),
        "groups": group_summary,
        "disabled_groups": sorted(name for name, cfg in session.groups_cfg.items() if not cfg["state"]),
        "csv_path": str(csv_path), "csv_rows": len(source_rows),
        "csv_input_train_count": len(input_trains), "csv_spike_count": csv_spike_count,
        "csv_window": [csv_block["start_ms"], csv_block["stop_ms"]],
        "saved_path": session.saved_path, "save_output": result["sim_cfg"]["save_output"],
        "snapshot_enabled": session.snapshot_enabled,
        "append_enabled": result["sim_cfg"].get("append_enabled", False),
    }}
    print({RESULT_SENTINEL!r} + json.dumps(payload, sort_keys=True))
    """
)


def _source_hashes(tune_dir: Path) -> dict[str, str]:
    """Include native assets/libraries/configs, but not previous run outputs."""
    return {
        path.relative_to(tune_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tune_dir.rglob("*")
        if path.is_file() and not {"output_data", "__pycache__"}.intersection(path.relative_to(tune_dir).parts)
    }


class AllenSynapticSubprocessTests(unittest.TestCase):
    _require_optional_runtime = allen_step5.AllenStep5SubprocessTests._require_optional_runtime

    def _absolute_temporary_csv_sources(self, copied_tune: Path) -> None:
        from modules.data_paths import resolve_data_alias

        # A temporary tune is outside the repo, so its relative cells/... paths
        # need real absolute input paths. Do not rewrite native/scientific data.
        for path in (copied_tune / "cell_configs" / "syn_groups").glob("*.json"):
            config = json.loads(path.read_text(encoding="utf-8"))
            changed = False
            for group in config.values():
                if not group.get("state", True):
                    continue  # Disabled templates are not runtime dependencies.
                for block in group.get("input_blocks", []):
                    source = block.get("source", {})
                    raw = source.get("path")
                    if not raw or Path(raw).suffix.lower() != ".csv":
                        continue
                    actual = resolve_data_alias(raw, REPO_ROOT)
                    self.assertTrue(actual.is_file(), f"Public baseline input CSV missing: {actual}")
                    self.assertGreater(actual.stat().st_size, 0, f"Public baseline input CSV empty: {actual}")
                    source["path"] = str(actual.resolve())
                    changed = True
            if changed:
                path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    def _run_baseline(self, cell_name: str) -> dict:
        from modules.setup.mechanisms import (
            find_compiled_mechanism_dll, find_nrnivmodl, resolve_modfiles_dir,
        )

        original = REPO_ROOT / "cells" / cell_name / "tunes" / "tuned"
        self._require_optional_runtime(original)
        before = _source_hashes(original)
        try:
            with tempfile.TemporaryDirectory(prefix=f"scp-{cell_name.lower()}-synaptic-") as tmp:
                copied_tune = Path(tmp) / "cells" / cell_name / "tunes" / "tuned"
                shutil.copytree(original, copied_tune,
                                ignore=shutil.ignore_patterns("output_data", "__pycache__"))
                self._absolute_temporary_csv_sources(copied_tune)
                if find_compiled_mechanism_dll(copied_tune) is None:
                    mod_dir = resolve_modfiles_dir(copied_tune)
                    if mod_dir is None or not mod_dir.is_dir() or not any(mod_dir.glob("*.mod")):
                        self.skipTest(f"optional Allen mechanism sources unavailable for {cell_name}")
                    self.assertTrue(mod_dir.is_relative_to(copied_tune))
                    compiler = find_nrnivmodl()
                    if compiler is None or not (shutil.which("c++") or shutil.which("g++")):
                        self.skipTest(f"nrnivmodl/C++ build tools unavailable for {cell_name}")
                    try:
                        built = subprocess.run([compiler], cwd=mod_dir, capture_output=True,
                                               text=True, check=False,
                                               timeout=SUBPROCESS_TIMEOUT_SECONDS)
                    except subprocess.TimeoutExpired as exc:
                        self.fail(f"{cell_name} temporary mechanism build timed out: {exc}")
                    self.assertEqual(built.returncode, 0,
                                     f"{cell_name} build failed.\n{built.stdout}\n{built.stderr}")
                    self.assertIsNotNone(find_compiled_mechanism_dll(copied_tune))
                env = dict(os.environ, SCP_ROOT=str(REPO_ROOT), MPLBACKEND="Agg",
                           PYTHONDONTWRITEBYTECODE="1")
                try:
                    completed = subprocess.run(
                        [sys.executable, "-c", _SUBPROCESS_PROGRAM, str(REPO_ROOT), str(copied_tune)],
                        cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                        check=False, timeout=SUBPROCESS_TIMEOUT_SECONDS,
                    )
                except subprocess.TimeoutExpired as exc:
                    self.fail(f"{cell_name} public synaptic run timed out: {exc}")
                self.assertEqual(completed.returncode, 0,
                                 f"{cell_name} subprocess failed.\n{completed.stdout}\n{completed.stderr}")
                sentinels = [line[len(RESULT_SENTINEL):] for line in completed.stdout.splitlines()
                             if line.startswith(RESULT_SENTINEL)]
                self.assertEqual(len(sentinels), 1,
                                 f"Missing/duplicate result sentinel.\n{completed.stdout}\n{completed.stderr}")
                return json.loads(sentinels[0])
        finally:
            self.assertEqual(_source_hashes(original), before,
                             "Public native assets/configs must remain byte-identical")

    def _assert_baseline(self, cell_name: str, *, disabled: list[str], csv_window: list[float]) -> None:
        payload = self._run_baseline(cell_name)
        self.assertEqual(payload["mode"], "single")
        self.assertEqual(payload["tstop"], 1000.0)
        self.assertAlmostEqual(payload["last_time_ms"], 1000.0, places=6)
        self.assertGreater(payload["sample_count"], 2)
        self.assertTrue(payload["matching_voltage_lengths"])
        self.assertTrue(payload["finite_voltage"])
        self.assertGreater(payload["syn_sample_count"], 2)
        self.assertTrue(payload["syn_finite_time"])
        # Fixed-step floating-point scheduling can stop one recording interval
        # before tstop; the independently sampled synaptic axis is authoritative.
        self.assertGreaterEqual(payload["syn_last_time_ms"], 999.5 - 1e-6)
        self.assertLessEqual(payload["syn_last_time_ms"], 1000.0 + 1e-6)
        self.assertAlmostEqual(payload["syn_time_step_ms"], 0.5, places=6)
        self.assertNotEqual(payload["sample_count"], payload["syn_sample_count"])
        self.assertGreater(len(payload["groups"]), 0)
        for name, group in payload["groups"].items():
            with self.subTest(group=name):
                self.assertEqual(group["mode"], "sample")
                self.assertGreater(group["n_synapses"], 0)
                self.assertEqual(group["sample_count"], min(3, group["n_synapses"]))
                self.assertEqual(group["variables"], ["g", "i"])
                self.assertTrue(group["matching_time_axis"])
                self.assertTrue(group["finite"])
        self.assertEqual(payload["disabled_groups"], sorted(disabled))
        self.assertTrue(set(disabled).isdisjoint(payload["groups"]))
        self.assertEqual(payload["csv_window"], csv_window)
        self.assertTrue(Path(payload["csv_path"]).is_relative_to(REPO_ROOT))
        self.assertGreater(payload["csv_rows"], 0)
        self.assertGreater(payload["csv_input_train_count"], 0)
        self.assertGreater(payload["csv_spike_count"], 0)
        self.assertIsNone(payload["saved_path"])
        self.assertIs(payload["save_output"], False)
        self.assertIs(payload["snapshot_enabled"], False)
        self.assertIs(payload["append_enabled"], False)

    def test_pv_public_file_driven_step5_in_fresh_subprocess(self) -> None:
        self._assert_baseline("PV", disabled=["pv_inh", "sst_inh"], csv_window=[290.0, 840.0])

    def test_sst_public_file_driven_step5_in_fresh_subprocess(self) -> None:
        self._assert_baseline("SST", disabled=["vip_inh"], csv_window=[290.0, 840.0])


if __name__ == "__main__":
    unittest.main()
