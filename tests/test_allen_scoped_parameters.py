from __future__ import annotations

import unittest

from neuron import h


class _Description:
    def __init__(self, data: dict) -> None:
        self.data = data


class _Utils:
    def __init__(self, data: dict) -> None:
        self.h = h
        self.description = _Description(data)


class AllenScopedParameterTests(unittest.TestCase):
    def test_legacy_e_na_is_applied_after_declared_mechanisms(self) -> None:
        try:
            from modules.loaders.allen_manifest import _apply_genome_based_parameters
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("allensdk"):
                self.skipTest("AllenSDK is not installed")
            raise

        section = h.Section(name="scp_allen_compat")
        try:
            utils = _Utils(
                {
                    "passive": [{"ra": 100.0, "e_pas": -65.0, "cm": []}],
                    "genome": [
                        {
                            "section": "scp_allen_compat",
                            "name": "e_na",
                            "value": 53.0,
                            "mechanism": "",
                        },
                        {
                            "section": "scp_allen_compat",
                            "name": "gnabar_hh",
                            "value": 0.12,
                            "mechanism": "hh",
                        },
                    ],
                    "conditions": [{"erev": []}],
                }
            )
            _apply_genome_based_parameters(utils, (section,))
            self.assertEqual(h.ismembrane("hh", sec=section), 1)
            self.assertAlmostEqual(float(section.ena), 53.0)
            self.assertAlmostEqual(float(section.gnabar_hh), 0.12)
        finally:
            h.delete_section(sec=section)


if __name__ == "__main__":
    unittest.main()
