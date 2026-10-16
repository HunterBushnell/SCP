from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from modules.notebooks.bootstrap import (
    DEFAULT_STEP5_EXTERNAL_INPUTS,
    check_required_external_inputs,
)
from modules.setup.defaults import default_syn_group_templates


ROOT = Path(__file__).resolve().parents[1]
CSV_HASHES = {
    "pyrFiringRateAvg.csv": "fe8f98657a38fe810dc84ddef980b1c8a218e6680a5c3e9ff243967361961d1d",
    "PVFiringRateAvg.csv": "090d3d23e7a7f497543a87313428017f16d2c6182a5f14747c1465b56c116fa0",
    "SSTFiringRateAvg.csv": "6dfcd646feeead94ce75b4ee5e923ccf2bf6de7e6db476fc85ebf9c15b828206",
}
PUBLIC_DATA = {
    "PN": ("pyrFiringRateAvg.csv",),
    "PV": ("pyrFiringRateAvg.csv", "PVFiringRateAvg.csv"),
    "SST": ("pyrFiringRateAvg.csv", "SSTFiringRateAvg.csv"),
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def project_saved(source, template):
    """Ignore generated snapshot fields, never active scientific fields."""
    if isinstance(template, dict):
        return {key: project_saved(source[key], value) for key, value in template.items()}
    if isinstance(template, list):
        if len(source) != len(template):
            raise AssertionError("Saved input-block count differs from the active config")
        return [project_saved(item, shape) for item, shape in zip(source, template)]
    return source


class PublicBaselineTests(unittest.TestCase):
    def test_public_synaptic_settings_match_saved_baselines(self) -> None:
        for cell in ("PV", "SST"):
            tune = ROOT / "cells" / cell / "tunes" / "tuned"
            saved = read_json(tune / "output_data" / f"{cell}5_base" / "results" / "syn_config.json")
            config_dir = tune / "cell_configs"
            for relative in read_json(config_dir / "syn_config.json")["group_files"]:
                for name, actual in read_json(config_dir / relative).items():
                    with self.subTest(cell=cell, group=name):
                        expected = copy.deepcopy(saved[name])
                        for block in expected.get("input_blocks", []):
                            source = block.get("source", {})
                            if source.get("path") == "external_data/pyrFiringRateAvg.csv":
                                source["path"] = f"cells/{cell}/data/pyrFiringRateAvg.csv"
                        if cell == "PV" and name in ("pv_inh", "sst_inh"):
                            # Inactive, inherited dependency paths are not replaced by
                            # the baseline snapshot's illustrative placeholders.
                            expected["input_blocks"][0]["source"]["path"] = {
                                "pv_inh": "cells/PV/tunes/tuned/output_data/PV5_base/results/spikes.npz",
                                "sst_inh": "cells/SST/tunes/tuned/output_data/SST5.3_tuned/results/spikes.npz",
                            }[name]
                        self.assertEqual(actual, project_saved(expected, actual))

    def test_tuned_intrinsic_models_and_geometry_are_preserved(self) -> None:
        for cell, multiplier in (("PV", 6.0), ("SST", 10.0)):
            tune = ROOT / "cells" / cell / "tunes" / "tuned"
            results = tune / "output_data" / f"{cell}5_base" / "results"
            with self.subTest(cell=cell):
                actual_cell = read_json(tune / "cell_configs" / "cell_config.json")
                self.assertEqual(actual_cell, read_json(results / "cell_config.json"))
                self.assertEqual(actual_cell["tuning"]["soma_diam_multiplier"], multiplier)
                self.assertEqual(
                    read_json(tune / "cell_configs" / "geometry.json"),
                    read_json(results / "geometry_config.json"),
                )
                # Compare actual archived bytes, not a potentially stale historical
                # manifest hash. This includes fits, morphologies, and mechanisms.
                archive = results / "model_artifacts"
                for artifact in read_json(archive / "manifest.json")["artifacts"]:
                    with self.subTest(cell=cell, artifact=artifact["target_relative_path"]):
                        self.assertEqual(
                            (tune / artifact["target_relative_path"]).read_bytes(),
                            (archive / artifact["archive_path"]).read_bytes(),
                        )

    def test_baseline_does_not_restore_production_run_options(self) -> None:
        for cell in ("PV", "SST"):
            config = read_json(ROOT / "cells" / cell / "tunes" / "tuned" / "cell_configs" / "sim_config.json")
            with self.subTest(cell=cell):
                self.assertEqual(config["n_trials"], 1)
                self.assertEqual(config["n_traces_to_save"], 1)
                self.assertEqual(config["n_inputs_to_save"], 1)
                self.assertFalse(config["save"]["enabled"])
                self.assertIsNone(config["save"]["stem"])
                self.assertFalse(config["load"]["enabled"])
                self.assertFalse(config["append"]["enabled"])


class PublicDataTests(unittest.TestCase):
    def test_cell_scoped_csv_bytes_are_unchanged(self) -> None:
        for cell, names in PUBLIC_DATA.items():
            for name in names:
                with self.subTest(cell=cell, csv=name):
                    path = ROOT / "cells" / cell / "data" / name
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), CSV_HASHES[name])
        for name in CSV_HASHES:
            self.assertFalse((ROOT / "external_data" / name).exists())

    def test_public_configs_use_their_own_data(self) -> None:
        for cell in PUBLIC_DATA:
            config_paths = sorted((ROOT / "cells" / cell / "tunes").glob("*/cell_configs/syn_groups/*.json"))
            found_rate_input = False
            for path in config_paths:
                text = path.read_text(encoding="utf-8")
                with self.subTest(config=path.relative_to(ROOT)):
                    self.assertNotIn("external_data/", text)
                    self.assertNotIn("projects/", text)
                    if "pyrFiringRateAvg.csv" in text:
                        found_rate_input = True
                        self.assertIn(f"cells/{cell}/data/pyrFiringRateAvg.csv", text)
            self.assertTrue(found_rate_input, cell)

    def test_generic_disabled_templates_do_not_select_a_public_cell(self) -> None:
        for payload in default_syn_group_templates().values():
            for group in payload.values():
                self.assertFalse(group["state"])
                for block in group["input_blocks"]:
                    source = block.get("source", {})
                    if "path" in source:
                        self.assertEqual(source["path"], "data/input_rates.csv")

    def test_bootstrap_and_analysis_paths_are_public_and_present(self) -> None:
        self.assertEqual(check_required_external_inputs(ROOT, warn=False), [])
        for relative in DEFAULT_STEP5_EXTERNAL_INPUTS:
            self.assertTrue(str(relative).startswith("cells/"))
        defaults = read_json(ROOT / "modules" / "analysis" / "analysis_defaults.json")
        preset = read_json(ROOT / "modules" / "analysis" / "analysis_presets" / "paper_compare.json")
        for entry in defaults["compare_list_paths"] + preset["entries"]:
            relative = entry["path"].split("@", 1)[0].split(";", 1)[0]
            self.assertTrue((ROOT / relative).exists(), relative)
            self.assertNotIn("projects/", relative)
        for relative in defaults["compare_list_dir_paths"]:
            self.assertTrue((ROOT / relative).is_dir(), relative)

    @unittest.skipUnless(shutil.which("git"), "Git is required to verify data visibility")
    def test_public_csvs_are_git_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
            shutil.copy2(ROOT / ".gitignore", root / ".gitignore")
            paths = [f"cells/{cell}/data/{name}" for cell, names in PUBLIC_DATA.items() for name in names]
            result = subprocess.run(
                ["git", "-C", str(root), "check-ignore", "--no-index", "--stdin"],
                input="\n".join(paths) + "\n", text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 1, result.stderr or result.stdout)
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
