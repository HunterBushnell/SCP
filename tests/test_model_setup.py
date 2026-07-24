from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from modules.setup.model_setup import resolve_step1_model_setup
from modules.setup.step1_prepare import prepare_base_configs


HOC_SOURCE = """\
begintemplate AutoCell
public soma, somatic, all
create soma[1]
objref somatic, all
proc init() {
    somatic = new SectionList()
    all = new SectionList()
    soma[0] { somatic.append() all.append() }
}
endtemplate AutoCell
"""


class Step1ModelSetupTests(unittest.TestCase):
    def test_fresh_staged_hoc_model_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            model = tune / "model"
            model.mkdir()
            (model / "AutoCell.hoc").write_text(HOC_SOURCE, encoding="utf-8")
            (tune / "modfiles").mkdir()
            archived = tune / "output_data" / "run_1" / "model_artifacts"
            archived.mkdir(parents=True)
            (archived / "OldCell.hoc").write_text(
                HOC_SOURCE.replace("AutoCell", "OldCell"), encoding="utf-8"
            )

            setup = resolve_step1_model_setup(tune, cell_name="Auto")

        self.assertEqual(setup["discovery"], "staged_hoc_template")
        self.assertEqual(setup["source_type"], "existing")
        self.assertEqual(setup["cell_loader"], "hoc_template")
        self.assertEqual(setup["loader_paths"]["hoc_template"], "model/AutoCell.hoc")
        self.assertEqual(setup["loader_paths"]["modfiles"], "modfiles")
        self.assertEqual(
            setup["loader_config"]["hoc_template"]["template_name"],
            "AutoCell",
        )
        self.assertNotIn(
            "section_map", setup["loader_config"]["hoc_template"]
        )

    def test_nested_manifest_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            manifest = tune / "model" / "allen" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}", encoding="utf-8")

            setup = resolve_step1_model_setup(tune, cell_name="A")

        self.assertEqual(setup["discovery"], "staged_manifest")
        self.assertEqual(setup["cell_loader"], "allen_manifest")
        self.assertEqual(
            setup["loader_paths"]["manifest"], "model/allen/manifest.json"
        )

    def test_discovered_hoc_setup_scaffolds_config_owned_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            model = tune / "model"
            model.mkdir()
            (model / "AutoCell.HOC").write_text(HOC_SOURCE, encoding="utf-8")
            setup = resolve_step1_model_setup(tune, cell_name="Auto")
            prepare_base_configs(
                tune_dir=tune,
                cell_name="Auto",
                tune_name="orig",
                specimen_id=setup["specimen_id"],
                model_type=setup["model_type"],
                cell_loader=setup["cell_loader"],
                loader_paths=setup["loader_paths"],
                loader_config=setup["loader_config"],
            )
            cell_config = json.loads(
                (tune / "cell_configs" / "cell_config.json").read_text(
                    encoding="utf-8"
                )
            )
            sim_config = json.loads(
                (tune / "cell_configs" / "sim_config.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(cell_config["cell_loader"], "hoc_template")
        self.assertEqual(cell_config["color"], "k")
        self.assertNotIn("tuning", cell_config)
        self.assertEqual(
            cell_config["paths"]["hoc_template"], "model/AutoCell.HOC"
        )
        self.assertEqual(
            cell_config["hoc_template"], {"template_name": "AutoCell"}
        )
        self.assertEqual(
            sim_config["conditions"], {"v_init_mV": None, "celsius_C": None}
        )

    def test_existing_cell_config_is_authoritative_on_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            configs = tune / "cell_configs"
            configs.mkdir()
            (configs / "cell_config.json").write_text(
                json.dumps(
                    {
                        "cell_loader": "hoc_template",
                        "paths": {
                            "hoc_template": "custom/Chosen.hoc",
                            "modfiles": None,
                        },
                        "hoc_template": {
                            "template_name": "Chosen",
                            "constructor_args": [2.0],
                            "section_map": {"soma": "body"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            setup = resolve_step1_model_setup(tune, cell_name="A")

        self.assertEqual(setup["discovery"], "cell_config")
        self.assertIsNone(setup["loader_paths"]["modfiles"])
        self.assertEqual(
            setup["loader_config"]["hoc_template"]["constructor_args"], [2.0]
        )

    def test_ambiguous_staged_templates_require_an_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            model = tune / "model"
            model.mkdir()
            (model / "One.hoc").write_text(
                HOC_SOURCE.replace("AutoCell", "One"), encoding="utf-8"
            )
            (model / "Two.hoc").write_text(
                HOC_SOURCE.replace("AutoCell", "Two"), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "multiple HOC template"):
                resolve_step1_model_setup(tune, cell_name="A")

    def test_adb_download_request_is_resolved_from_one_override_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            setup = resolve_step1_model_setup(
                Path(tmp),
                cell_name="Custom",
                overrides={
                    "source_type": "adb",
                    "cell_loader": "allen_manifest",
                    "specimen_id": 123,
                    "model_type": "all active",
                },
            )

        self.assertEqual(setup["discovery"], "overrides")
        self.assertEqual(setup["specimen_id"], 123)
        self.assertEqual(setup["model_type"], "all active")
        self.assertEqual(setup["loader_paths"]["manifest"], "manifest.json")

    def test_explicit_hoc_override_supports_custom_construction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            setup = resolve_step1_model_setup(
                Path(tmp),
                cell_name="Custom",
                overrides={
                    "cell_loader": "hoc_template",
                    "paths": {"hoc_template": "native/Cell.hoc", "modfiles": None},
                    "hoc_template": {
                        "template_name": "Cell",
                        "constructor_args": [4.0],
                        "section_map": {"soma": "body"},
                    },
                },
            )

        self.assertEqual(setup["discovery"], "overrides")
        self.assertIsNone(setup["loader_paths"]["modfiles"])
        self.assertEqual(
            setup["loader_config"]["hoc_template"]["constructor_args"], [4.0]
        )

    def test_missing_source_has_an_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                FileNotFoundError, "Stage one manifest.json or one HOC template"
            ):
                resolve_step1_model_setup(Path(tmp), cell_name="A")

    def test_fill_preserves_config_owned_metadata_and_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            prepare_base_configs(
                tune_dir=tune,
                cell_name="A",
                tune_name="orig",
                specimen_id=None,
                cell_loader="allen_manifest",
                loader_paths={"manifest": "manifest.json"},
                color="cyan",
                soma_diam_multiplier=3.75,
                v_init_mV=-80.0,
                celsius_C=31.0,
            )
            prepare_base_configs(
                tune_dir=tune,
                cell_name="A",
                tune_name="orig",
                specimen_id=None,
                cell_loader="allen_manifest",
                loader_paths={"manifest": "manifest.json"},
                config_mode="fill",
            )
            cell_config = json.loads(
                (tune / "cell_configs" / "cell_config.json").read_text(
                    encoding="utf-8"
                )
            )
            sim_config = json.loads(
                (tune / "cell_configs" / "sim_config.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(cell_config["color"], "cyan")
        self.assertEqual(cell_config["tuning"]["soma_diam_multiplier"], 3.75)
        self.assertEqual(
            sim_config["conditions"], {"v_init_mV": -80.0, "celsius_C": 31.0}
        )


if __name__ == "__main__":
    unittest.main()
