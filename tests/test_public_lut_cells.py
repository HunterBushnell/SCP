from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CELLS_ROOT = REPO_ROOT / "cells"
LUT_COMMIT = "f9739d8aa7f94eac67fcaa67e8e04e26787dee0f"
MODFILES = {
    "Im.mod",
    "Nap.mod",
    "cadyn.mod",
    "capool.mod",
    "h.mod",
    "kaprox.mod",
    "kdrca1.mod",
    "leak.mod",
    "na3.mod",
    "sahp.mod",
}
SECTION_MAP = {
    "soma": "somatic",
    "dend": "basal",
    "apic": "apical",
    "axon": "axonal",
    "all": "all",
}
EXPECTED = {
    "EUSmn": {
        "conditions": {
            "orig": {"v_init_mV": -70.0, "celsius_C": 34.0},
            "tuned": {"v_init_mV": -55.0, "celsius_C": 34.0},
        },
        "passive": {"v_rest_mV": -55.0, "rin_MOhm": 11.1, "tau_ms": 4.9},
        "upstream_template": "PUD_TEMPLATE.hoc",
    },
    "HYPO": {
        "conditions": {
            "orig": {"v_init_mV": -62.0, "celsius_C": 31.0},
            "tuned": {"v_init_mV": -62.0, "celsius_C": 31.0},
        },
        "passive": {"v_rest_mV": -62.0, "rin_MOhm": 410.0, "tau_ms": 104.0},
        "upstream_template": "HYPO_TEMPLATE.hoc",
    },
    "PGN": {
        "conditions": {
            "orig": {"v_init_mV": -57.8, "celsius_C": 31.0},
            "tuned": {"v_init_mV": -57.8, "celsius_C": 31.0},
        },
        "passive": {"v_rest_mV": -57.8, "rin_MOhm": 483.6, "tau_ms": 40.0},
        "upstream_template": "PGN_TEMPLATE.hoc",
    },
}


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PublicLutCellBundleTests(unittest.TestCase):
    def test_configs_preserve_tune_values_and_use_explicit_hoc_contract(self) -> None:
        for cell_name, expected in EXPECTED.items():
            for tune in ("orig", "tuned"):
                with self.subTest(cell=cell_name, tune=tune):
                    tune_dir = CELLS_ROOT / cell_name / "tunes" / tune
                    config_dir = tune_dir / "cell_configs"
                    cell_config = _read_json(config_dir / "cell_config.json")
                    sim_config = _read_json(config_dir / "sim_config.json")
                    target_config = _read_json(config_dir / "target_config.json")

                    self.assertEqual(cell_config["cell_name"], cell_name)
                    self.assertEqual(cell_config["tune"], tune)
                    self.assertEqual(cell_config["cell_loader"], "hoc_template")
                    self.assertEqual(
                        cell_config["hoc_template"]["constructor_args"], []
                    )
                    self.assertEqual(
                        cell_config["hoc_template"]["section_map"], SECTION_MAP
                    )
                    self.assertEqual(
                        sim_config["conditions"], expected["conditions"][tune]
                    )
                    self.assertEqual(
                        target_config["manual"]["passive"], expected["passive"]
                    )

                    self.assertFalse((config_dir / "syn_config.json").exists())
                    self.assertFalse(
                        any((config_dir / "syn_groups").glob("*"))
                        if (config_dir / "syn_groups").exists()
                        else False
                    )
                    self.assertEqual(
                        {path.name for path in (tune_dir / "modfiles").glob("*.mod")},
                        MODFILES,
                    )

    def test_public_provenance_covers_every_hoc_and_mod_source(self) -> None:
        for cell_name, expected in EXPECTED.items():
            for tune in ("orig", "tuned"):
                with self.subTest(cell=cell_name, tune=tune):
                    tune_dir = CELLS_ROOT / cell_name / "tunes" / tune
                    provenance_path = tune_dir / "SOURCE_PROVENANCE.json"
                    provenance = _read_json(provenance_path)

                    self.assertFalse(
                        (tune_dir / "SOURCE_PROVENANCE.local.json").exists()
                    )
                    self.assertEqual(provenance["cell_name"], cell_name)
                    self.assertEqual(provenance["tune"], tune)
                    self.assertEqual(
                        provenance["source_repository"]["commit"], LUT_COMMIT
                    )
                    self.assertEqual(
                        provenance["rights"],
                        {
                            "redistribution_status": "redistributed_with_permission",
                            "scp_mit_scope": "excluded_from_scp_mit",
                            "upstream_license": "no upstream license asserted",
                        },
                    )
                    self.assertFalse(provenance["compiled_outputs_included"])

                    expected_files = {
                        path.relative_to(tune_dir).as_posix()
                        for path in (
                            *sorted((tune_dir / "model").glob("*.hoc")),
                            *sorted((tune_dir / "modfiles").glob("*.mod")),
                        )
                    }
                    entries = provenance["files"]
                    self.assertEqual(
                        {entry["destination_path"] for entry in entries},
                        expected_files,
                    )

                    derivatives = []
                    for entry in entries:
                        destination = tune_dir / entry["destination_path"]
                        self.assertNotIn("x86_64", destination.parts)
                        self.assertEqual(
                            entry["current_sha256"], _sha256(destination)
                        )
                        if entry["relationship"] == "exact_copy":
                            self.assertEqual(
                                entry["source_sha256"], entry["current_sha256"]
                            )
                        else:
                            self.assertEqual(
                                entry["relationship"], "manual_derivative"
                            )
                            self.assertNotEqual(
                                entry["source_sha256"], entry["current_sha256"]
                            )
                            derivatives.append(entry)

                    if tune == "orig":
                        self.assertEqual(provenance["model_relationship"], "exact_copy")
                        self.assertEqual(derivatives, [])
                    else:
                        self.assertEqual(
                            provenance["model_relationship"], "manual_derivative"
                        )
                        self.assertEqual(len(derivatives), 1)
                        self.assertEqual(
                            Path(derivatives[0]["source_path"]).name,
                            expected["upstream_template"],
                        )

    def test_each_public_cell_explains_scope_and_rights(self) -> None:
        for cell_name in EXPECTED:
            with self.subTest(cell=cell_name):
                readme = (CELLS_ROOT / cell_name / "README.md").read_text(
                    encoding="utf-8"
                )
                compact_readme = " ".join(readme.split())
                self.assertIn("hoc_template", readme)
                self.assertIn(
                    "not as an independent biological-validation claim",
                    compact_readme,
                )
                self.assertIn("excluded from SCP's MIT license", compact_readme)


if __name__ == "__main__":
    unittest.main()
