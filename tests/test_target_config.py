from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from modules.setup.defaults import default_target_config
from modules.setup.target_config import infer_target_source_mode, prepare_target_config
from modules.tuning.active_targets import resolve_active_tuning_targets
from modules.tuning.passive_targets import resolve_passive_tuning_inputs
from modules.tuning.targets import update_passive_targets_in_config


class TargetConfigScaffoldingTests(unittest.TestCase):
    def test_templates_are_sparse_and_mode_specific(self) -> None:
        expected_blocks = {
            "none": set(),
            "manual": {"manual"},
            "traces": {"traces"},
            "allen_nwb": {"allen_nwb"},
        }
        source_blocks = {"manual", "traces", "allen_nwb"}
        for mode, expected in expected_blocks.items():
            with self.subTest(mode=mode):
                config = default_target_config(mode=mode)
                self.assertEqual(config["target_source"]["mode"], mode)
                self.assertEqual(set(config) & source_blocks, expected)

        mixed = default_target_config(
            mode="traces", include_manual_with_file_source=True
        )
        self.assertEqual(set(mixed) & source_blocks, {"manual", "traces"})
        targetless = default_target_config(
            mode="none", include_manual_with_file_source=True
        )
        self.assertEqual(set(targetless) & source_blocks, set())

    def test_mode_inference_uses_supplied_file_paths(self) -> None:
        self.assertEqual(infer_target_source_mode(), "manual")
        self.assertEqual(
            infer_target_source_mode(passive_trace={"file": "passive.csv"}),
            "traces",
        )
        self.assertEqual(
            infer_target_source_mode(active_trace={"file": "active.npy"}),
            "traces",
        )
        self.assertEqual(
            infer_target_source_mode(allen_nwb={"file": "cell.nwb"}),
            "allen_nwb",
        )
        with self.assertRaisesRegex(ValueError, "Both user-trace and Allen NWB"):
            infer_target_source_mode(
                passive_trace={"file": "passive.csv"},
                allen_nwb={"file": "cell.nwb"},
            )
        with self.assertRaisesRegex(ValueError, "user-trace target file"):
            infer_target_source_mode(
                target_source_mode="manual",
                passive_trace={"file": "passive.csv"},
            )

    def test_new_config_defaults_to_blank_manual_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            summary = prepare_target_config(tune_dir=tune, target_source_mode=None)
            config = self._read_config(tune)

        self.assertEqual(summary["target_source_mode"], "manual")
        self.assertEqual(
            set(config), {"schema_version", "target_source", "manual", "notes"}
        )
        self.assertIsNone(config["manual"]["passive"]["v_rest_mV"])

    def test_file_path_inference_writes_only_matching_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            prepare_target_config(
                tune_dir=tune,
                passive_trace={"file": "targets/passive.csv"},
            )
            config = self._read_config(tune)

        self.assertEqual(config["target_source"]["mode"], "traces")
        self.assertEqual(
            set(config), {"schema_version", "target_source", "traces", "notes"}
        )
        self.assertEqual(config["traces"]["passive"]["file"], "targets/passive.csv")

    def test_explicit_file_mode_can_include_blank_manual_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            prepare_target_config(
                tune_dir=tune,
                target_source_mode="allen_nwb",
                include_manual_with_file_source=True,
            )
            config = self._read_config(tune)

        self.assertIn("allen_nwb", config)
        self.assertIn("manual", config)
        self.assertNotIn("traces", config)

    def test_fill_preserves_existing_blocks_and_overwrite_prunes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            path = tune / "cell_configs" / "target_config.json"
            path.parent.mkdir(parents=True)
            legacy = {
                "schema_version": 1,
                "target_source": {"mode": "none", "description": "keep"},
                "manual": {"passive": {"v_rest_mV": -65.0}},
                "traces": {"passive": {"file": "old.csv"}},
                "allen_nwb": {"file": "old.nwb"},
                "notes": "preserve me",
            }
            path.write_text(json.dumps(legacy), encoding="utf-8")

            prepare_target_config(
                tune_dir=tune,
                config_mode="fill",
                target_source_mode="manual",
            )
            filled = self._read_config(tune)
            self.assertEqual(filled["manual"]["passive"]["v_rest_mV"], -65.0)
            self.assertIn("traces", filled)
            self.assertIn("allen_nwb", filled)
            self.assertEqual(filled["notes"], "preserve me")

            prepare_target_config(
                tune_dir=tune,
                config_mode="overwrite",
                target_source_mode="manual",
            )
            overwritten = self._read_config(tune)

        self.assertIn("manual", overwritten)
        self.assertNotIn("traces", overwritten)
        self.assertNotIn("allen_nwb", overwritten)

    def test_passive_update_does_not_expand_unrelated_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            prepare_target_config(
                tune_dir=tune,
                target_source_mode="manual",
            )
            updated = update_passive_targets_in_config(
                tune,
                {"v_rest_mV": -67.0, "rin_MOhm": None, "tau_ms": None},
            )

        self.assertEqual(updated["manual"]["passive"]["v_rest_mV"], -67.0)
        self.assertNotIn("traces", updated)
        self.assertNotIn("allen_nwb", updated)

    def test_passive_update_creates_a_valid_sparse_manual_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            updated = update_passive_targets_in_config(
                Path(tmp),
                {"v_rest_mV": -67.0, "rin_MOhm": 50.0, "tau_ms": 5.0},
            )

        self.assertEqual(updated["target_source"]["mode"], "manual")
        self.assertEqual(
            set(updated), {"schema_version", "target_source", "manual", "notes"}
        )

    @staticmethod
    def _read_config(tune: Path) -> dict:
        return json.loads(
            (tune / "cell_configs" / "target_config.json").read_text(
                encoding="utf-8"
            )
        )


class PartialTargetResolutionTests(unittest.TestCase):
    def test_partial_manual_passive_targets_are_optional_without_act(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            config_path = tune / "cell_configs" / "target_config.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                json.dumps(
                    {
                        "target_source": {"mode": "manual"},
                        "manual": {"passive": {"v_rest_mV": -65.0}},
                    }
                ),
                encoding="utf-8",
            )
            context = SimpleNamespace(
                tune_dir=tune,
                repo_root=tune,
                cell_config={"cell_loader": "hoc_template"},
            )
            with mock.patch(
                "modules.tuning.passive_targets.passive_area_summary",
                return_value={"passive_area_cm2": 1.0},
            ), mock.patch(
                "modules.tuning.passive_targets.import_act_passive_module"
            ) as import_act:
                resolution = resolve_passive_tuning_inputs(
                    context=context,
                    cell=object(),
                    compute_act_proposal=False,
                )

        import_act.assert_not_called()
        self.assertEqual(resolution.passive_targets["target_v_rest_mv"], -65.0)
        self.assertIsNone(resolution.passive_targets["target_rin_mohm"])
        self.assertIsNone(resolution.passive_targets["target_tau_ms"])

    def test_blank_trace_template_is_targetless_without_act(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            prepare_target_config(tune_dir=tune, target_source_mode="traces")
            context = SimpleNamespace(
                tune_dir=tune,
                repo_root=tune,
                cell_config={"cell_loader": "hoc_template"},
            )
            with mock.patch(
                "modules.tuning.passive_targets.passive_area_summary",
                return_value={"passive_area_cm2": 1.0},
            ), mock.patch(
                "modules.tuning.passive_targets.import_act_passive_module"
            ) as import_act:
                resolution = resolve_passive_tuning_inputs(
                    context=context,
                    cell=object(),
                    compute_act_proposal=False,
                )

        import_act.assert_not_called()
        self.assertIsNone(resolution.passive_targets["target_v_rest_mv"])
        self.assertEqual(resolution.target_source_mode, "traces")

    def test_blank_nwb_template_is_targetless_without_act(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            prepare_target_config(tune_dir=tune, target_source_mode="allen_nwb")
            context = SimpleNamespace(
                tune_dir=tune,
                repo_root=tune,
                cell_config={"cell_loader": "hoc_template"},
            )
            with mock.patch(
                "modules.tuning.passive_targets.passive_area_summary",
                return_value={"passive_area_cm2": 1.0},
            ), mock.patch(
                "modules.tuning.passive_targets.import_act_passive_module"
            ) as import_act:
                resolution = resolve_passive_tuning_inputs(
                    context=context,
                    cell=object(),
                    compute_act_proposal=False,
                )

        import_act.assert_not_called()
        self.assertIsNone(resolution.passive_targets["target_v_rest_mv"])
        self.assertEqual(resolution.target_source_mode, "allen_nwb")

    def test_configured_missing_trace_path_fails_before_act_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            prepare_target_config(
                tune_dir=tune,
                target_source_mode="traces",
                passive_trace={"file": "missing.csv"},
            )
            context = SimpleNamespace(
                tune_dir=tune,
                repo_root=tune,
                cell_config={"cell_loader": "hoc_template"},
            )
            with mock.patch(
                "modules.tuning.passive_targets.import_act_passive_module"
            ) as import_act, self.assertRaisesRegex(
                FileNotFoundError, "Configured passive trace file does not exist"
            ):
                resolve_passive_tuning_inputs(context=context, cell=object())

        import_act.assert_not_called()

    def test_blank_active_file_template_is_valid_until_act_is_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            prepare_target_config(
                tune_dir=tune,
                target_source_mode="traces",
                include_manual_with_file_source=True,
            )
            context = SimpleNamespace(tune_dir=tune)
            resolution = resolve_active_tuning_targets(
                context=context,
                require_target=False,
            )

        self.assertEqual(resolution.target_source_mode, "traces")
        self.assertEqual(resolution.target_mode, "trace_npy")
        self.assertIsNone(resolution.trace_npy_path)
        self.assertEqual(resolution.fi_reference_points, [])

    def test_manual_fi_can_fill_a_missing_side_of_trace_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            prepare_target_config(
                tune_dir=tune,
                target_source_mode="traces",
                include_manual_with_file_source=True,
            )
            config_path = tune / "cell_configs" / "target_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["manual"]["fi_curve"]["currents_pA"] = [0.0, 100.0]
            config["manual"]["fi_curve"]["rates_Hz"] = [0.0, 20.0]
            config_path.write_text(json.dumps(config), encoding="utf-8")

            resolution = resolve_active_tuning_targets(
                context=SimpleNamespace(tune_dir=tune),
                require_target=False,
            )

        self.assertEqual(resolution.target_source_mode, "traces")
        self.assertEqual(resolution.target_mode, "fi_arrays")
        self.assertEqual(resolution.fi_reference_points, [(0.0, 0.0), (100.0, 20.0)])


if __name__ == "__main__":
    unittest.main()
