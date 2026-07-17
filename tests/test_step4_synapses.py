from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from modules.tuning.synapse_tuning_config import (
    default_synapse_tuning_config,
    ensure_synapse_tuning_config,
)


FIXTURE_TUNE = Path(__file__).resolve().parent / "fixtures" / "hoc_template"
NEUTRAL_CONNECTIONS = {
    "excitatory_facilitating",
    "excitatory_depressing",
    "inhibitory_static",
    "inhibitory_stp",
}


class Step4SynapseConfigTests(unittest.TestCase):
    def test_default_catalog_is_loader_neutral(self) -> None:
        config = default_synapse_tuning_config()
        self.assertEqual(set(config["connections"]), NEUTRAL_CONNECTIONS)
        self.assertEqual(config["default_connection"], "excitatory_facilitating")
        serialized = json.dumps(config)
        self.assertNotIn("PV", serialized)
        self.assertNotIn("SST", serialized)

    def test_generation_uses_tune_runtime_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp) / "tune"
            shutil.copytree(FIXTURE_TUNE, tune)

            path, config, status = ensure_synapse_tuning_config(tune)

            self.assertEqual(status, "created")
            self.assertTrue(path.is_file())
            self.assertEqual(set(config["connections"]), NEUTRAL_CONNECTIONS)
            self.assertEqual(config["general_settings"]["celsius"], 32.0)
            for connection in config["connections"].values():
                self.assertEqual(
                    connection["spec_settings"]["vclamp_amp"],
                    -68.0,
                )

    def test_existing_config_is_not_rewritten_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp) / "tune"
            config_dir = tune / "cell_configs"
            config_dir.mkdir(parents=True)
            path = config_dir / "synapse_tuning_config.json"
            original = (
                '{\n  "connections": {"custom": {"spec_settings": '
                '{"level_of_detail": "ExpSyn", "sec_id": 0}}}\n}\n'
            )
            path.write_text(original, encoding="utf-8")

            returned_path, config, status = ensure_synapse_tuning_config(tune)

            self.assertEqual(returned_path, path.resolve())
            self.assertEqual(status, "existing")
            self.assertEqual(config["default_connection"], "custom")
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_overwrite_replaces_existing_config_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp) / "tune"
            shutil.copytree(FIXTURE_TUNE, tune)
            path = tune / "cell_configs" / "synapse_tuning_config.json"
            path.write_text('{"connections": {"custom": {}}}\n', encoding="utf-8")

            returned_path, config, status = ensure_synapse_tuning_config(
                tune,
                overwrite=True,
            )

            self.assertEqual(returned_path, path.resolve())
            self.assertEqual(status, "overwritten")
            self.assertEqual(set(config["connections"]), NEUTRAL_CONNECTIONS)


if __name__ == "__main__":
    unittest.main()
