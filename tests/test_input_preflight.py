from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.check_setup import Reporter, _check_input_dependencies, _check_tune_bundle


ROOT = Path(__file__).resolve().parents[1]


class InputPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tune = self.root / "cells" / "Synthetic" / "tunes" / "orig"
        self.configs = self.tune / "cell_configs"
        self.configs.mkdir(parents=True)
        # A metadata-only Python factory validates paths without importing or
        # executing model source. Deliberately invalid code proves that contract.
        (self.tune / "model.py").write_text("this model must never execute\n", encoding="utf-8")
        self.write_json("cell_config.json", {
            "cell_name": "Synthetic", "tune": "orig", "cell_loader": "python_factory",
            "paths": {"python_factory": "model.py", "modfiles": None},
            "python_factory": {"callable": "build_cell"},
        })
        self.write_json("geometry.json", {})
        self.write_json("target_config.json", {"target_source": {"mode": "none"}})
        self.write_json("sim_config.json", {
            "iclamp": {"enabled": False},
            "conditions": {"v_init_mV": -65.0, "celsius_C": 34.0},
        })
        (self.root / "run_pipeline.py").write_text("", encoding="utf-8")

    def write_json(self, name: str, data) -> None:
        path = self.configs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def check_bundle(self, *, steps=None):
        reporter = Reporter()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _check_tune_bundle(
                reporter, repo_root=self.root, cell="Synthetic", tune="orig",
                compile_modfiles=False, steps={"5"} if steps is None else steps,
            )
        return reporter, output.getvalue()

    @staticmethod
    def group(path="missing/spikes.npz", *, state=True, n_syn=1):
        return {
            "state": state, "mode": "precomputed", "syns": {"N_syn": n_syn},
            "source": {"path": path},
        }

    def test_enabled_missing_group_source_fails_actionably(self) -> None:
        self.write_json("syn_config.json", {"drive": self.group()})
        reporter, output = self.check_bundle()
        self.assertEqual(reporter.errors, 1, output)
        for expected in ("Missing Step 5 input", "group 'drive'", "missing/spikes.npz", "Supply", "disable"):
            self.assertIn(expected, output)
        self.assertIn(str(self.tune / "missing/spikes.npz"), output)

    def test_missing_input_block_reports_group_block_and_config_index(self) -> None:
        group = self.group()
        group["input_blocks"] = [{
            "name": "stimulus", "mode": "inhomogeneous_poisson",
            "source": {"path": "missing/rates.csv"},
        }]
        self.write_json("syn_config.json", {"drive": group})
        reporter, output = self.check_bundle()
        self.assertEqual(reporter.errors, 1, output)
        self.assertIn("group 'drive' block 'stimulus' (input_blocks[0])", output)
        self.assertIn("missing/rates.csv", output)
        self.assertNotIn("missing/spikes.npz", output)  # Unused group-level source.

    def test_disabled_groups_blocks_and_explicit_zero_counts_skip_sources(self) -> None:
        active = self.group()
        active["input_blocks"] = [{
            "state": False, "name": "disabled", "mode": "precomputed",
            "source": {"path": "missing/block.npz"},
        }]
        self.write_json("syn_config.json", {
            "disabled": self.group(state=False),
            "zero": self.group(n_syn=0),
            "string_zero": self.group(n_syn="0"),
            "active_with_disabled_block": active,
        })
        reporter, output = self.check_bundle()
        self.assertEqual(reporter.errors, 0, output)
        self.assertNotIn("Missing Step 5 input", output)

    def test_density_derived_count_does_not_hide_missing_dependencies(self) -> None:
        self.write_json("syn_config.json", {"derived": self.group(n_syn=None)})
        reporter, output = self.check_bundle()
        self.assertEqual(reporter.errors, 1, output)
        self.assertIn("group 'derived'", output)

    def test_inline_precomputed_trains_take_precedence_including_empty_pool(self) -> None:
        groups = {}
        for name, trains in (("inline", [[1.0]]), ("empty", [])):
            groups[name] = self.group()
            groups[name]["source"]["trains"] = trains
        groups["blocks"] = self.group()
        groups["blocks"]["input_blocks"] = [{
            "mode": "precomputed", "source": {"trains": [], "path": "missing/block.npz"},
        }]
        self.write_json("syn_config.json", groups)
        reporter, output = self.check_bundle()
        self.assertEqual(reporter.errors, 0, output)

    def test_inline_trains_do_not_override_rate_curve_files(self) -> None:
        group = self.group("missing/rates.csv")
        group["mode"] = "inhomogeneous_poisson"
        group["source"]["trains"] = []
        self.write_json("syn_config.json", {"rates": group})
        reporter, output = self.check_bundle()
        self.assertEqual(reporter.errors, 1, output)
        self.assertIn("missing/rates.csv", output)

    def test_existing_relative_sources_and_run_directory_pass_read_only(self) -> None:
        rates = self.tune / "data" / "rates.csv"
        rates.parent.mkdir()
        rates.write_text("Time,AvgFiringRate\n0,1\n", encoding="utf-8")
        run = self.tune / "output_data" / "saved_run"
        run.mkdir(parents=True)
        group = self.group("data/rates.csv")
        group["mode"] = "inhomogeneous_poisson"
        self.write_json("syn_config.json", {"rates": group, "replay": self.group("output_data/saved_run")})
        before = {p: p.read_bytes() for p in self.tune.rglob("*") if p.is_file()}
        with patch("modules.input_generation.generate_inputs", side_effect=AssertionError("must not generate inputs")):
            reporter, output = self.check_bundle()
        self.assertEqual(reporter.errors, 0, output)
        self.assertIn("Input dependency for group 'rates'", output)
        self.assertIn("Input dependency for group 'replay'", output)
        self.assertEqual(before, {p: p.read_bytes() for p in self.tune.rglob("*") if p.is_file()})

    def test_root_relative_cell_project_and_legacy_alias_sources_pass(self) -> None:
        sources = {
            "cell": "cells/Synthetic/data/rates.csv",
            "project": "projects/study/cells/Synthetic/data/rates.csv",
        }
        for relative in sources.values():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"Time,AvgFiringRate\n0,2\n")
        aliases = {"external_data/rates.csv": {
            "path": sources["cell"],
            "sha256": hashlib.sha256((self.root / sources["cell"]).read_bytes()).hexdigest(),
        }}
        (self.root / "cells" / "DATA_PATH_ALIASES.json").write_text(json.dumps({"aliases": aliases}), encoding="utf-8")
        sources["legacy"] = "external_data/rates.csv"
        self.write_json("syn_config.json", {name: self.group(path) for name, path in sources.items()})
        reporter, output = self.check_bundle()
        self.assertEqual(reporter.errors, 0, output)
        self.assertNotIn("Missing Step 5 input", output)

    def test_expanded_group_files_are_inspected(self) -> None:
        self.write_json("syn_config.json", {"group_files": ["syn_groups/drive.json"]})
        self.write_json("syn_groups/drive.json", {"drive": self.group()})
        reporter, output = self.check_bundle()
        self.assertEqual(reporter.errors, 1, output)
        self.assertIn("group 'drive'", output)

    def test_malformed_and_missing_includes_are_actionable(self) -> None:
        for payload, expected in (
            ({"group_files": "wrong.json"}, "must be a list"),
            ({"group_files": ["absent.json"]}, "absent.json"),
            ({"__includes__": []}, "use 'group_files'"),
        ):
            with self.subTest(payload=payload):
                self.write_json("syn_config.json", payload)
                reporter, output = self.check_bundle()
                self.assertEqual(reporter.errors, 1, output)
                self.assertIn("Cannot inspect Step 5 input dependencies", output)
                self.assertIn(expected, output)
                self.assertIn("Fix syn_config.json", output)

    def test_malformed_group_block_and_source_report_without_crashing(self) -> None:
        groups = {
            "bad_group": [],
            "bad_syns": {"syns": "wrong"},
            "bad_blocks": {"input_blocks": "wrong"},
            "bad_block": {"input_blocks": [1]},
            "bad_source": {"source": "wrong"},
            "bad_path": {"source": {"path": 1}},
        }
        self.write_json("syn_config.json", groups)
        reporter, output = self.check_bundle()
        self.assertEqual(reporter.errors, len(groups), output)
        for name in groups:
            self.assertIn(name, output)

    def test_iclamp_skips_missing_inputs_and_malformed_includes(self) -> None:
        self.write_json("sim_config.json", {
            "iclamp": {"enabled": True},
            "conditions": {"v_init_mV": -65.0, "celsius_C": 34.0},
        })
        self.write_json("syn_config.json", {"group_files": ["absent.json"]})
        reporter, output = self.check_bundle()
        self.assertEqual(reporter.errors, 0, output)
        self.assertNotIn("input dependencies", output)
        (self.configs / "syn_config.json").unlink()
        reporter, output = self.check_bundle()
        self.assertEqual(reporter.errors, 0, output)

    def test_other_steps_do_not_require_input_files(self) -> None:
        self.write_json("syn_config.json", {"drive": self.group()})
        reporter, output = self.check_bundle(steps={"1", "2", "3"})
        self.assertEqual(reporter.errors, 0, output)
        self.assertNotIn("Missing Step 5 input", output)

    def test_public_saved_baseline_dependencies_pass_without_sdk_or_model_load(self) -> None:
        for cell in ("PV", "SST"):
            with self.subTest(cell=cell):
                tune = ROOT / "cells" / cell / "tunes" / "tuned"
                reporter = Reporter()
                output = io.StringIO()
                with contextlib.redirect_stdout(output), patch(
                    "modules.input_generation.generate_inputs", side_effect=AssertionError("must not generate inputs")
                ):
                    _check_input_dependencies(
                        reporter, tune, tune / "cell_configs" / "syn_config.json",
                        json.loads((tune / "cell_configs" / "sim_config.json").read_text()),
                    )
                self.assertEqual(reporter.errors, 0, output.getvalue())
                self.assertIn("Input dependency", output.getvalue())


if __name__ == "__main__":
    unittest.main()
