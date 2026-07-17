from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from neuron import h

from modules.loaders import (
    available_cell_loaders,
    discover_cell_source_artifacts,
    validate_cell_loader_config,
)
from modules.model.geometry import define_geometry
from modules.model.load_cell import load_cell


FIXTURE_TUNE = Path(__file__).resolve().parent / "fixtures" / "hoc_template"


def fixture_config() -> dict:
    return json.loads(
        (FIXTURE_TUNE / "cell_configs" / "cell_config.json").read_text(encoding="utf-8")
    )


class HocTemplateLoaderTests(unittest.TestCase):
    def test_registry_and_artifact_discovery(self) -> None:
        self.assertEqual(available_cell_loaders(), ("allen_manifest", "hoc_template"))
        config = fixture_config()
        validated = validate_cell_loader_config(config, base_dir=FIXTURE_TUNE)
        discovered = discover_cell_source_artifacts(config, base_dir=FIXTURE_TUNE)
        expected = (FIXTURE_TUNE / "model" / "ScopedCell.hoc").resolve()
        self.assertEqual(validated["hoc_template"], expected)
        self.assertEqual(discovered, validated)

        rooted = fixture_config()
        rooted["paths"]["root"] = "model"
        rooted["paths"]["hoc_template"] = "ScopedCell.hoc"
        rooted_path = validate_cell_loader_config(rooted, base_dir=FIXTURE_TUNE)
        self.assertEqual(rooted_path["hoc_template"], expected)

    def test_object_owned_instances_are_scoped(self) -> None:
        unrelated = h.Section(name="scp_unrelated_global")
        try:
            first = load_cell(fixture_config(), base_dir=FIXTURE_TUNE)
            second_config = fixture_config()
            second_config["hoc_template"]["constructor_args"] = [1.5]
            second = load_cell(second_config, base_dir=FIXTURE_TUNE)

            self.assertIsNot(first.model, second.model)
            self.assertEqual(len(first.soma), 1)
            self.assertEqual(len(first.dend), 1)
            self.assertEqual(len(first.apic), 0)
            self.assertEqual(len(first.axon), 0)
            self.assertEqual(len(first.all), 2)
            self.assertEqual(len(second.all), 2)
            self.assertNotIn(unrelated.name(), {section.name() for section in first.all})
            self.assertTrue(
                {section.name() for section in first.all}.isdisjoint(
                    {section.name() for section in second.all}
                )
            )
            self.assertAlmostEqual(float(first.soma[0].L), 20.0)
            self.assertAlmostEqual(float(second.soma[0].L), 30.0)

            geometry = define_geometry(first, {})
            self.assertEqual(geometry["meta"]["counts"]["all_secs"], 2)
            self.assertEqual(geometry["meta"]["counts"]["soma_secs"], 1)
            self.assertEqual(geometry["meta"]["counts"]["dend_secs"], 1)
        finally:
            h.delete_section(sec=unrelated)

    def test_invalid_configs_fail_clearly(self) -> None:
        unknown = fixture_config()
        unknown["cell_loader"] = "not_a_loader"
        with self.assertRaisesRegex(ValueError, "Unsupported cell_loader"):
            load_cell(unknown, base_dir=FIXTURE_TUNE)

        missing_path = fixture_config()
        missing_path["paths"]["hoc_template"] = "model/missing.hoc"
        with self.assertRaisesRegex(FileNotFoundError, "hoc_template"):
            load_cell(missing_path, base_dir=FIXTURE_TUNE)

        missing_template = fixture_config()
        missing_template["hoc_template"]["template_name"] = "NoSuchTemplate"
        with self.assertRaisesRegex(AttributeError, "NoSuchTemplate"):
            load_cell(missing_template, base_dir=FIXTURE_TUNE)

        empty_soma = copy.deepcopy(fixture_config())
        empty_soma["hoc_template"]["section_map"]["soma"] = "apical"
        with self.assertRaisesRegex(ValueError, "no canonical soma"):
            load_cell(empty_soma, base_dir=FIXTURE_TUNE)

        with tempfile.TemporaryDirectory() as tmp:
            alternate = Path(tmp) / "ScopedCell.hoc"
            alternate.write_text(
                (FIXTURE_TUNE / "model" / "ScopedCell.hoc").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            conflicting = fixture_config()
            conflicting["paths"]["hoc_template"] = str(alternate)
            with self.assertRaisesRegex(RuntimeError, "Restart"):
                load_cell(conflicting, base_dir=FIXTURE_TUNE)

    def test_common_names_need_no_explicit_section_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            hoc = tune / "CommonOnly.hoc"
            hoc.write_text(
                "begintemplate CommonOnly\n"
                "public soma\n"
                "create soma[1]\n"
                "proc init() { soma[0] { L=10 diam=10 insert pas } }\n"
                "endtemplate CommonOnly\n",
                encoding="utf-8",
            )
            config = {
                "cell_loader": "hoc_template",
                "paths": {"hoc_template": "CommonOnly.hoc", "modfiles": None},
                "hoc_template": {
                    "template_name": "CommonOnly",
                    "constructor_args": [],
                    "section_map": {},
                },
            }
            cell = load_cell(config, base_dir=tune)
            self.assertEqual(len(cell.soma), 1)
            self.assertEqual(len(cell.all), 1)
            self.assertEqual(cell.dend, ())

    def test_one_hoc_source_is_loaded_once_for_multiple_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            hoc = tune / "MultipleTemplates.hoc"
            hoc.write_text(
                "begintemplate ScpMultiOne\n"
                "public soma\n"
                "create soma[1]\n"
                "proc init() { soma[0] { L=11 diam=10 insert pas } }\n"
                "endtemplate ScpMultiOne\n"
                "begintemplate ScpMultiTwo\n"
                "public soma\n"
                "create soma[1]\n"
                "proc init() { soma[0] { L=22 diam=10 insert pas } }\n"
                "endtemplate ScpMultiTwo\n",
                encoding="utf-8",
            )
            config = {
                "cell_loader": "hoc_template",
                "paths": {"hoc_template": hoc.name, "modfiles": None},
                "hoc_template": {
                    "template_name": "ScpMultiOne",
                    "constructor_args": [],
                    "section_map": {},
                },
            }
            first = load_cell(config, base_dir=tune)
            self.assertAlmostEqual(float(first.soma[0].L), 11.0)

            # The second template was defined by the first load. Replacing the
            # source with invalid HOC proves SCP does not execute that absolute
            # path a second time in the same process.
            hoc.write_text("this is intentionally invalid HOC\n", encoding="utf-8")
            second_config = copy.deepcopy(config)
            second_config["hoc_template"]["template_name"] = "ScpMultiTwo"
            second = load_cell(second_config, base_dir=tune)
            self.assertAlmostEqual(float(second.soma[0].L), 22.0)


if __name__ == "__main__":
    unittest.main()
