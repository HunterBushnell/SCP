from __future__ import annotations

import contextlib
import inspect
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modules.notebooks.bootstrap import (
    ensure_modfiles,
    finish_step5_notebook_setup,
)
from modules.setup.fit_json import coerce_fit_genome_values_to_numeric, find_fit_json
from modules.setup.mechanisms import resolve_modfiles_dir
from modules.setup.step1_prepare import prepare_mechanisms
from modules.simulation.result_paths import _copy_fit_json_sidecar, _find_fit_json_path
from modules.simulation.snapshots import _collect_mechanism_info
from modules.tuning.act_active import _build_act_cell
from scripts import step1_prepare as step1_cli
from scripts.restore_run_state import _find_fit_json_in_tune


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ConfiguredModPathTests(unittest.TestCase):
    def test_step1_cli_omits_unspecified_hoc_overrides_on_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "step1_prepare.py",
                "--tune-dir",
                tmp,
                "--cell-loader",
                "hoc_template",
                "--source-type",
                "existing",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(step1_cli, "prepare_tune", return_value={}) as prepare,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                step1_cli.main()

            kwargs = prepare.call_args.kwargs
            self.assertEqual(kwargs["loader_paths"], {})
            self.assertEqual(kwargs["loader_config"], {})

    def test_stored_custom_disabled_and_legacy_default_mod_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            config_path = tune / "cell_configs" / "cell_config.json"

            _write_json(config_path, {"paths": {"modfiles": "native/mechanisms"}})
            self.assertEqual(
                resolve_modfiles_dir(tune),
                (tune / "native" / "mechanisms").resolve(),
            )
            self.assertEqual(
                resolve_modfiles_dir(tune, {"paths": {}}),
                (tune / "native" / "mechanisms").resolve(),
            )

            _write_json(config_path, {"paths": {"modfiles": None}})
            self.assertIsNone(resolve_modfiles_dir(tune))

            _write_json(config_path, {"paths": {}})
            self.assertEqual(resolve_modfiles_dir(tune), (tune / "modfiles").resolve())

            config_path.unlink()
            self.assertEqual(resolve_modfiles_dir(tune), (tune / "modfiles").resolve())

    def test_step1_existing_tune_rerun_uses_stored_mod_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            configured = tune / "native" / "mechanisms"
            configured.mkdir(parents=True)
            _write_json(
                tune / "cell_configs" / "cell_config.json",
                {"cell_loader": "hoc_template", "paths": {"modfiles": "native/mechanisms"}},
            )

            result = prepare_mechanisms(
                tune_dir=tune,
                load_compiled_dll=False,
                allow_missing_modfiles=True,
                cell_config={"cell_loader": "hoc_template", "paths": {}},
            )
            compile_result = result["compile_modfiles"]
            self.assertEqual(compile_result["status"], "skipped")
            self.assertEqual(compile_result["modfiles_dir"], str(configured.resolve()))

    def test_step5_bootstrap_compiles_in_configured_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            configured = tune / "native" / "mechanisms"
            configured.mkdir(parents=True)
            (configured / "Synthetic.mod").write_text("NEURON {}\n", encoding="utf-8")
            _write_json(
                tune / "cell_configs" / "cell_config.json",
                {"paths": {"modfiles": "native/mechanisms"}},
            )

            with (
                mock.patch(
                    "modules.setup.mechanisms.find_nrnivmodl",
                    return_value="/synthetic/nrnivmodl",
                ),
                mock.patch("modules.notebooks.bootstrap.subprocess.check_call") as check_call,
            ):
                ensure_modfiles(tune, compile_modfiles=True)

            check_call.assert_called_once_with(
                ["/synthetic/nrnivmodl"],
                cwd=str(configured.resolve()),
            )

    def test_step5_bootstrap_accepts_disabled_or_absent_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            config_path = tune / "cell_configs" / "cell_config.json"
            _write_json(config_path, {"paths": {"modfiles": None}})
            with mock.patch("modules.notebooks.bootstrap.subprocess.check_call") as check_call:
                with contextlib.redirect_stdout(io.StringIO()):
                    ensure_modfiles(tune, compile_modfiles=True)
            check_call.assert_not_called()

            _write_json(config_path, {"paths": {}})
            with mock.patch("modules.notebooks.bootstrap.subprocess.check_call") as check_call:
                with contextlib.redirect_stdout(io.StringIO()):
                    ensure_modfiles(tune, compile_modfiles=True)
            check_call.assert_not_called()

    def test_step5_external_input_checks_are_opt_in(self) -> None:
        signature = inspect.signature(finish_step5_notebook_setup)
        self.assertIs(signature.parameters["check_external_inputs"].default, False)
        with mock.patch(
            "modules.notebooks.bootstrap.check_required_external_inputs",
            side_effect=AssertionError("external inputs should not be checked"),
        ):
            result = finish_step5_notebook_setup(
                Path(__file__).resolve().parents[1],
                install_deps=False,
                print_status=False,
            )
        self.assertEqual(result["missing_external"], [])


class ConfiguredConsumerPathTests(unittest.TestCase):
    def test_act_cell_receives_configured_or_disabled_mod_path(self) -> None:
        class FakeACTCell:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.builder = None

            def set_custom_cell_builder(self, builder):
                self.builder = builder

        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            config_path = tune / "cell_configs" / "cell_config.json"
            cfg = {"tune_dir": str(tune), "act_cell": {}}
            builder = object()

            _write_json(config_path, {"paths": {"modfiles": "native/mod"}})
            cell = _build_act_cell(
                cfg,
                act_api={"ACTCellModel": FakeACTCell},
                builder=builder,
                active_channels=[],
            )
            self.assertEqual(
                cell.kwargs["path_to_mod_files"],
                str((tune / "native" / "mod").resolve()),
            )
            self.assertIs(cell.builder, builder)

            _write_json(config_path, {"paths": {"modfiles": None}})
            cell = _build_act_cell(
                cfg,
                act_api={"ACTCellModel": FakeACTCell},
                builder=builder,
                active_channels=[],
            )
            self.assertIsNone(cell.kwargs["path_to_mod_files"])

    def test_snapshot_uses_configured_mod_sources_and_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            mod_dir = tune / "native" / "mod"
            dll = mod_dir / "x86_64" / ".libs" / "libnrnmech.so"
            dll.parent.mkdir(parents=True)
            dll.write_bytes(b"synthetic-library")
            (mod_dir / "Synthetic.mod").write_text("NEURON {}\n", encoding="utf-8")
            _write_json(
                tune / "cell_configs" / "cell_config.json",
                {"paths": {"modfiles": "native/mod"}},
            )

            info = _collect_mechanism_info({"tune_dir": str(tune)})
            self.assertEqual(info["modfiles_dir"], str(mod_dir.resolve()))
            self.assertEqual(info["modfiles"], ["Synthetic.mod"])
            self.assertEqual(info["dll_path"], str(dll.resolve()))


class ConfiguredAllenFitPathTests(unittest.TestCase):
    def _make_nested_allen_tune(self, tune: Path) -> Path:
        fit_path = tune / "native" / "model" / "synthetic_fit.json"
        _write_json(fit_path, {"genome": [{"section": "soma", "value": "1.25"}]})
        _write_json(
            tune / "native" / "model" / "manifest.json",
            {"biophys": [{"model_file": ["synthetic_fit.json"]}]},
        )
        _write_json(
            tune / "cell_configs" / "cell_config.json",
            {
                "cell_loader": "allen_manifest",
                "paths": {"manifest": "native/model/manifest.json"},
            },
        )
        return fit_path.resolve()

    def test_configured_manifest_drives_setup_save_and_restore_fit_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp) / "tune"
            tune.mkdir()
            fit_path = self._make_nested_allen_tune(tune)

            self.assertEqual(find_fit_json(tune), fit_path)
            self.assertEqual(
                find_fit_json(
                    tune,
                    cell_config={"cell_loader": "allen_manifest", "paths": {}},
                ),
                fit_path,
            )
            self.assertEqual(_find_fit_json_path({"tune_dir": str(tune)}), fit_path)
            self.assertEqual(_find_fit_json_in_tune(tune), fit_path)

            result = coerce_fit_genome_values_to_numeric(tune)
            self.assertEqual(result["status"], "updated")
            self.assertEqual(json.loads(fit_path.read_text())["genome"][0]["value"], 1.25)

            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            sidecar = _copy_fit_json_sidecar({"tune_dir": str(tune)}, run_dir)
            self.assertIsNotNone(sidecar)
            self.assertTrue((run_dir / "synthetic_fit.json").is_file())

    def test_legacy_root_fit_fallback_remains_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            fit_path = tune / "legacy_fit.json"
            _write_json(fit_path, {"genome": []})
            self.assertEqual(find_fit_json(tune), fit_path.resolve())

    def test_non_allen_tunes_ignore_fit_json_compatibility_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            _write_json(tune / "orphan_fit.json", {"genome": []})
            _write_json(
                tune / "cell_configs" / "cell_config.json",
                {"cell_loader": "hoc_template", "paths": {"hoc_template": "Cell.hoc"}},
            )
            self.assertIsNone(find_fit_json(tune))
            self.assertIsNone(_find_fit_json_path({"tune_dir": str(tune)}))
            self.assertIsNone(_find_fit_json_in_tune(tune))


if __name__ == "__main__":
    unittest.main()
