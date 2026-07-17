from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from modules.setup.step1_prepare import prepare_tune
from modules.setup.validation import validate_tune
from modules.tuning.notebook_setup import validate_step1_outputs
from scripts.check_setup import (
    Reporter,
    _check_external_dependencies,
    _check_tune_bundle,
)


FIXTURE_TUNE = Path(__file__).resolve().parent / "fixtures" / "hoc_template"


class GenericSetupTests(unittest.TestCase):
    def test_full_generic_setup_check_keeps_act_bmtool_and_synapses_optional(self) -> None:
        reporter = Reporter()
        steps = {str(index) for index in range(1, 8)}
        with contextlib.redirect_stdout(io.StringIO()):
            _check_external_dependencies(
                reporter,
                repo_root=FIXTURE_TUNE.parents[2],
                steps=steps,
                check_act=False,
                check_bmtool=False,
            )
            _check_tune_bundle(
                reporter,
                repo_root=FIXTURE_TUNE.parents[2],
                cell="unused",
                tune="unused",
                compile_modfiles=False,
                steps=steps,
                tune_dir_override=FIXTURE_TUNE,
            )
        self.assertEqual(reporter.errors, 0)
        self.assertEqual(reporter.warnings, 0)

    def test_tuning_setup_preserves_actionable_loader_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp) / "tune"
            config_dir = tune / "cell_configs"
            config_dir.mkdir(parents=True)
            (config_dir / "cell_config.json").write_text(
                json.dumps({"cell_loader": "not_registered"}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unsupported cell_loader"):
                validate_step1_outputs(
                    tune,
                    require_sim_config=False,
                    require_geometry_config=False,
                    require_compiled_modfiles=False,
                )

    def test_existing_hoc_rerun_preserves_loader_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp) / "tune"
            shutil.copytree(FIXTURE_TUNE, tune)
            config_path = tune / "cell_configs" / "cell_config.json"
            before = json.loads(config_path.read_text(encoding="utf-8"))

            prepare_tune(
                tune_dir=tune,
                cell_name=before["cell_name"],
                tune_name=before["tune"],
                source_type="existing",
                cell_loader="hoc_template",
                loader_paths={},
                loader_config={},
                do_download=False,
                do_compile_modfiles=False,
                do_base_configs=True,
                do_target_config=False,
                do_synapse_configs=False,
                config_mode="fill",
                do_validate=False,
            )

            after = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(after["paths"], before["paths"])
            self.assertEqual(after["hoc_template"], before["hoc_template"])

            # A partial explicit update changes only that field; sibling loader
            # metadata still survives the fill-mode rerun.
            prepare_tune(
                tune_dir=tune,
                cell_name=before["cell_name"],
                tune_name=before["tune"],
                source_type="existing",
                cell_loader="hoc_template",
                loader_paths={},
                loader_config={"hoc_template": {"template_name": "UpdatedCell"}},
                do_download=False,
                do_compile_modfiles=False,
                do_base_configs=True,
                do_target_config=False,
                do_synapse_configs=False,
                config_mode="fill",
                do_validate=False,
            )
            partially_updated = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(partially_updated["hoc_template"]["template_name"], "UpdatedCell")
            self.assertEqual(
                partially_updated["hoc_template"]["constructor_args"],
                before["hoc_template"]["constructor_args"],
            )
            self.assertEqual(
                partially_updated["hoc_template"]["section_map"],
                before["hoc_template"]["section_map"],
            )

    def test_existing_manifest_free_fixture_validates(self) -> None:
        checks = validate_tune(
            tune_dir=FIXTURE_TUNE,
            cell_name="synthetic_scoped_cell",
            validate_modfiles=True,
            validate_load_cell=True,
            validate_inputs=False,
            validate_synapses=False,
            allow_missing_modfiles=True,
        )
        self.assertEqual(checks["cell_loader"], "hoc_template")
        self.assertEqual(checks["mechanisms"]["status"], "skipped")
        self.assertEqual(checks["conditions"]["v_init_mV"], -68.0)

    def test_step1_scaffolds_generic_hoc_tune_without_targets_or_modfiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp) / "cells" / "synthetic" / "tunes" / "orig"
            model_dir = tune / "model"
            model_dir.mkdir(parents=True)
            template_text = (FIXTURE_TUNE / "model" / "ScopedCell.hoc").read_text(
                encoding="utf-8"
            )
            (model_dir / "ScopedCell.hoc").write_text(
                template_text.replace("ScopedCell", "SetupCell"),
                encoding="utf-8",
            )

            summary = prepare_tune(
                tune_dir=tune,
                cell_name="synthetic",
                tune_name="orig",
                source_type="existing",
                cell_loader="hoc_template",
                loader_paths={"hoc_template": "model/ScopedCell.hoc", "modfiles": None},
                loader_config={
                    "hoc_template": {
                        "template_name": "SetupCell",
                        "constructor_args": [1.0],
                        "section_map": {
                            "soma": "somatic",
                            "dend": "basal",
                            "apic": "apical",
                            "axon": "axonal",
                            "all": "all",
                        },
                    }
                },
                v_init_mV=-68.0,
                celsius_C=32.0,
                do_compile_modfiles=True,
                allow_missing_modfiles=True,
                do_target_config=True,
                target_source_mode="none",
                do_synapse_configs=False,
                validate_inputs_cfg=False,
            )
            self.assertEqual(summary["actions"]["mechanisms"]["compile_modfiles"]["status"], "skipped")
            cell_config = json.loads(
                (tune / "cell_configs" / "cell_config.json").read_text(encoding="utf-8")
            )
            sim_config = json.loads(
                (tune / "cell_configs" / "sim_config.json").read_text(encoding="utf-8")
            )
            target_config = json.loads(
                (tune / "cell_configs" / "target_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(cell_config["cell_loader"], "hoc_template")
            self.assertNotIn("manifest", cell_config["paths"])
            self.assertNotIn("tuning", cell_config)
            self.assertEqual(sim_config["conditions"], {"v_init_mV": -68.0, "celsius_C": 32.0})
            self.assertEqual(target_config["target_source"]["mode"], "none")
            self.assertFalse((tune / "cell_configs" / "syn_config.json").exists())

    def test_hoc_validation_requires_explicit_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp) / "tune"
            shutil.copytree(FIXTURE_TUNE, tune)
            sim_path = tune / "cell_configs" / "sim_config.json"
            sim_config = json.loads(sim_path.read_text(encoding="utf-8"))
            sim_config["conditions"]["celsius_C"] = None
            sim_path.write_text(json.dumps(sim_config, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "conditions.celsius_C"):
                validate_tune(
                    tune_dir=tune,
                    cell_name="synthetic_scoped_cell",
                    validate_modfiles=False,
                    validate_load_cell=False,
                    validate_inputs=False,
                    validate_synapses=False,
                    allow_missing_modfiles=True,
                )


if __name__ == "__main__":
    unittest.main()
