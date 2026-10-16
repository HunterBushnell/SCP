from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_SENTINEL = "SCP_STEP5_ACCEPTANCE_JSON="
SUBPROCESS_TIMEOUT_SECONDS = 120


_SUBPROCESS_PROGRAM = textwrap.dedent(
    f"""
    import json
    import math
    import sys
    from pathlib import Path

    repo_root = Path(sys.argv[1]).resolve()
    tune_dir = Path(sys.argv[2]).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from modules.simulation import SimulationOptions, SimulationSession
    from modules.simulation.current_injection import run_current_injection

    options = SimulationOptions(
        iclamp=True,
        load_mechanisms=True,
        sim_overrides={{
            "save_output": False,
            "tstop": 20.0,
            "dt": 0.025,
            "stim_start_ms": 5.0,
            "stim_duration_ms": 10.0,
            "iclamp": {{
                "amp_nA": 0.05,
                "delay_ms": 5.0,
                "dur_ms": 10.0,
                "tstop_ms": 20.0,
                "dt_ms": 0.025,
                "record_currents": False,
            }},
        }},
    )
    session = SimulationSession.from_tune(tune_dir, options=options)
    result = session.run()
    times = list(result["traces"]["T"])
    voltages = list(result["traces"]["V"])
    currents = run_current_injection(session.cell, {{
        "stim_amp": 0.05, "stim_delay": 5.0, "stim_dur": 10.0,
        "h_tstop": 20.0, "h_dt": 0.025,
        "conditions": result["meta"]["conditions"],
    }})
    payload = {{
        "mode": result.get("mode"),
        "section_count": len(session.cell.all),
        "soma_count": len(session.cell.soma),
        "sample_count": len(times),
        "matching_trace_lengths": len(times) == len(voltages),
        "finite_trace": bool(times) and bool(voltages) and all(
            math.isfinite(float(value)) for value in (*times, *voltages)
        ),
        "last_time_ms": float(times[-1]),
        "conditions": result.get("meta", {{}}).get("conditions", {{}}),
        "groups_empty": session.groups_cfg == {{}},
        "saved_path": None if session.saved_path is None else str(session.saved_path),
        "save_output": result.get("sim_cfg", {{}}).get("save_output"),
        "iclamp_currents_omitted": "I" not in result["iclamp"],
        "current_trace_count": len(currents["I"]),
        "finite_current_traces": all(
            len(values) == len(currents["T"]) and all(math.isfinite(float(v)) for v in values)
            for values in currents["I"].values()
        ),
    }}
    print({RESULT_SENTINEL!r} + json.dumps(payload, sort_keys=True))
    """
)


class AllenStep5SubprocessTests(unittest.TestCase):
    def _require_optional_runtime(self, tune_dir: Path) -> None:
        missing_dependencies = [
            name
            for name in ("neuron", "allensdk")
            if importlib.util.find_spec(name) is None
        ]
        if missing_dependencies:
            self.skipTest(
                "missing optional dependencies: " + ", ".join(missing_dependencies)
            )

        required_assets = [
            tune_dir / "manifest.json",
            tune_dir / "cell_configs" / "cell_config.json",
            tune_dir / "cell_configs" / "geometry.json",
            tune_dir / "cell_configs" / "sim_config.json",
        ]
        manifest_path = tune_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            def declared_model_files(value: object) -> list[Path]:
                if isinstance(value, dict):
                    return [
                        path
                        for nested in value.values()
                        for path in declared_model_files(nested)
                    ]
                if isinstance(value, list):
                    return [
                        path
                        for nested in value
                        for path in declared_model_files(nested)
                    ]
                if isinstance(value, str) and Path(value).suffix.lower() in {
                    ".json",
                    ".swc",
                }:
                    declared = Path(value)
                    return [declared if declared.is_absolute() else tune_dir / declared]
                return []

            required_assets.extend(declared_model_files(manifest))
        missing_assets = [path for path in required_assets if not path.is_file()]
        if missing_assets:
            self.skipTest(
                "missing optional Allen test assets: "
                + ", ".join(str(path.relative_to(REPO_ROOT)) for path in missing_assets)
            )

    def _run_iclamp(self, cell_name: str) -> dict:
        from modules.setup.mechanisms import (
            find_compiled_mechanism_dll,
            find_nrnivmodl,
            resolve_modfiles_dir,
        )

        tune_dir = REPO_ROOT / "cells" / cell_name / "tunes" / "tuned"
        self._require_optional_runtime(tune_dir)
        if find_compiled_mechanism_dll(tune_dir) is not None:
            return self._run_iclamp_subprocess(cell_name, tune_dir)

        mod_dir = resolve_modfiles_dir(tune_dir)
        if mod_dir is None or not mod_dir.is_dir() or not any(mod_dir.glob("*.mod")):
            self.skipTest(f"optional Allen mechanism sources are unavailable for {cell_name}")
        compiler = find_nrnivmodl()
        if compiler is None or not (shutil.which("c++") or shutil.which("g++")):
            self.skipTest(
                "compiled optional Allen library is unavailable and nrnivmodl/C++ "
                f"build tools are missing for {cell_name}"
            )

        # Build only a temporary source copy; keep production cells read-only.
        with tempfile.TemporaryDirectory(prefix=f"scp-{cell_name.lower()}-regression-") as tmp:
            copied_tune = Path(tmp) / "tuned"
            shutil.copytree(
                tune_dir,
                copied_tune,
                ignore=shutil.ignore_patterns("output_data", "x86_64", "__pycache__"),
            )
            copied_mod_dir = resolve_modfiles_dir(copied_tune)
            self.assertIsNotNone(copied_mod_dir)
            self.assertTrue(copied_mod_dir.is_relative_to(copied_tune))
            try:
                built = subprocess.run(
                    [compiler], cwd=str(copied_mod_dir), capture_output=True,
                    text=True, check=False, timeout=SUBPROCESS_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as exc:
                self.fail(f"{cell_name} temporary mechanism build timed out: {exc}")
            self.assertEqual(
                built.returncode, 0,
                f"{cell_name} mechanism build failed.\nstdout:\n{built.stdout}\nstderr:\n{built.stderr}",
            )
            self.assertIsNotNone(find_compiled_mechanism_dll(copied_tune))
            return self._run_iclamp_subprocess(cell_name, copied_tune)

    def _run_iclamp_subprocess(self, cell_name: str, tune_dir: Path) -> dict:
        try:
            completed = subprocess.run(
                [sys.executable, "-c", _SUBPROCESS_PROGRAM, str(REPO_ROOT), str(tune_dir)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            self.fail(
                f"{cell_name} Step 5 IClamp subprocess exceeded "
                f"{SUBPROCESS_TIMEOUT_SECONDS}s. stdout={exc.stdout!r} stderr={exc.stderr!r}"
            )

        self.assertEqual(
            completed.returncode,
            0,
            f"{cell_name} subprocess failed.\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}",
        )
        sentinel_lines = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith(RESULT_SENTINEL)
        ]
        self.assertEqual(
            len(sentinel_lines),
            1,
            f"Expected exactly one result sentinel.\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}",
        )
        try:
            return json.loads(sentinel_lines[0][len(RESULT_SENTINEL) :])
        except json.JSONDecodeError as exc:
            self.fail(f"Invalid result sentinel JSON for {cell_name}: {exc}")

    def _assert_regression(
        self,
        payload: dict,
        *,
        expected_sections: int,
        expected_v_init_mv: float,
    ) -> None:
        self.assertEqual(payload["mode"], "iclamp")
        self.assertEqual(payload["section_count"], expected_sections)
        self.assertEqual(payload["soma_count"], 1)
        self.assertGreater(payload["sample_count"], 2)
        self.assertTrue(payload["matching_trace_lengths"])
        self.assertTrue(payload["finite_trace"])
        self.assertAlmostEqual(payload["last_time_ms"], 20.0, places=6)
        self.assertTrue(payload["groups_empty"])
        self.assertIsNone(payload["saved_path"])
        self.assertIs(payload["save_output"], False)
        self.assertTrue(payload["iclamp_currents_omitted"])
        self.assertGreater(payload["current_trace_count"], 0)
        self.assertTrue(payload["finite_current_traces"])
        self.assertAlmostEqual(
            float(payload["conditions"]["v_init_mV"]),
            expected_v_init_mv,
        )
        self.assertAlmostEqual(float(payload["conditions"]["celsius_C"]), 34.0)

    def test_pv_tuned_step5_iclamp_in_fresh_subprocess(self) -> None:
        self._assert_regression(
            self._run_iclamp("PV"),
            expected_sections=44,
            expected_v_init_mv=-71.0,
        )

    def test_sst_tuned_step5_iclamp_in_fresh_subprocess(self) -> None:
        self._assert_regression(
            self._run_iclamp("SST"),
            expected_sections=48,
            expected_v_init_mv=-65.0,
        )


if __name__ == "__main__":
    unittest.main()
