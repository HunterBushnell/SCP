"""ACT installation and API checks do not depend on a sibling checkout."""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scp_pipeline.tuning import act_integration as dependency


ROOT = Path(__file__).resolve().parents[1]
PASSIVE = '''
class ACTPassiveModule:
    @staticmethod
    def compute_spp(R_in, soma_area, tau, V_rest):
        raise AssertionError("API validation must not calculate properties")
    @staticmethod
    def compute_gpp(passive_V, dt, I_t_start, I_t_end, I_amp):
        raise AssertionError("API validation must not calculate properties")
'''
CELL = '''
class ACTCellModel:
    def __init__(self, **kwargs):
        raise AssertionError("API validation must not build a cell")
    def set_custom_cell_builder(self, builder, cell_builder_kwargs=None): pass
'''
MODULE = '''
class ACTModule:
    def __init__(self, **kwargs):
        raise AssertionError("API validation must not run optimization")
    def run(self): pass
'''
SIMULATOR = '''
class ACTSimulator:
    def __init__(self, **kwargs):
        raise AssertionError("API validation must not construct a simulator")
    def submit_job(self, cell, parameters=None): pass
    def run_jobs(self, n_cpu=None): pass
'''
TYPES = '''
class _Parameters:
    def __init__(self, **kwargs):
        raise AssertionError("API validation must not construct parameters")
ConductanceOptions = ConstantCurrentInjection = FilterParameters = _Parameters
GaussianCurrentInjection = OptimizationParameters = _Parameters
RampCurrentInjection = SimulationParameters = _Parameters
'''


class ACTDependencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.path = list(sys.path)
        self.modules = {k: v for k, v in sys.modules.items() if k == "act" or k.startswith("act.")}
        self._clear_modules()
        self.addCleanup(self._restore_imports)
        self.env = mock.patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        for key in (*dependency.ACT_ENV_VARS, "SCP_AUTO_CLONE_ACT"):
            os.environ.pop(key, None)
        missing = mock.patch.object(dependency, "resolve_external_repo_checkout", side_effect=FileNotFoundError("missing"))
        self.checkout = missing.start()
        self.addCleanup(missing.stop)
        clone = mock.patch.object(dependency, "_clone_external_repo", side_effect=AssertionError("must not download"))
        self.clone = clone.start()
        self.addCleanup(clone.stop)

    def _clear_modules(self):
        for name in list(sys.modules):
            if name == "act" or name.startswith("act."):
                del sys.modules[name]

    def _restore_imports(self):
        self._clear_modules()
        sys.modules.update(self.modules)
        sys.path[:] = self.path
        importlib.invalidate_caches()

    def package(self, name, **overrides):
        root = self.base / name
        package = root / "act"
        package.mkdir(parents=True)
        sources = {
            "__init__": "", "passive": PASSIVE, "cell_model": CELL,
            "module": MODULE, "simulator": SIMULATOR, "types": TYPES,
            "data_processing": "def get_summary_features(**kwargs): pass\n",
        }
        sources.update(overrides)
        for stem, source in sources.items():
            (package / f"{stem}.py").write_text(source, encoding="utf-8")
        importlib.invalidate_caches()
        return root

    def installed_fixture(self, name="site-packages", **overrides):
        root = self.package(name, **overrides)
        sys.path.insert(0, str(root))
        return root

    def test_environment_package_needs_no_checkout_or_process_mutation(self):
        installed = self.installed_fixture()
        path, env = list(sys.path), dict(os.environ)
        self.assertEqual(dependency.resolve_act_repo(repo_root=self.base), installed)
        self.assertNotIn("act", sys.modules)
        self.assertEqual(dependency.ensure_act_on_syspath(repo_root=self.base), installed)
        passive = dependency.import_act_passive_module(repo_root=self.base)
        self.assertEqual(passive.__name__, "ACTPassiveModule")
        api = dependency.import_act_active_api(repo_root=self.base)
        self.assertIn("get_summary_features", api)
        self.assertEqual(sys.path, path)
        self.assertEqual(dict(os.environ), env)
        self.checkout.assert_not_called()
        self.clone.assert_not_called()

    def test_installed_package_wins_over_automatic_checkout(self):
        installed = self.installed_fixture()
        self.checkout.side_effect = None
        self.checkout.return_value = self.package("legacy")
        self.assertEqual(dependency.ensure_act_on_syspath(repo_root=self.base), installed)
        self.checkout.assert_not_called()

    def test_passive_import_does_not_import_optimization_modules(self):
        self.installed_fixture(module="raise AssertionError('optimizer must stay unloaded')\n")
        dependency.import_act_passive_module(repo_root=self.base)
        self.assertNotIn("act.module", sys.modules)
        self.assertNotIn("act.simulator", sys.modules)

    def test_all_explicit_environment_aliases_override_package(self):
        self.installed_fixture()
        explicit = self.package("explicit")
        for key in dependency.ACT_ENV_VARS:
            with self.subTest(variable=key), mock.patch.dict(os.environ, {key: str(explicit)}):
                self.assertEqual(dependency.resolve_act_repo(repo_root=self.base), explicit)
        self.assertEqual(dependency.resolve_act_repo(extra_candidates=[explicit]), explicit)

    def test_explicit_checkout_imports_selected_copy(self):
        self.installed_fixture()
        explicit = self.package("explicit")
        os.environ["SCP_ACT_PATH"] = str(explicit)
        dependency.import_act_active_api(repo_root=self.base)
        self.assertEqual(Path(sys.modules["act.module"].__file__).parent.parent, explicit)
        self.clone.assert_not_called()

    def test_invalid_explicit_checkout_does_not_use_installed_copy(self):
        self.installed_fixture()
        os.environ["SCP_ACT_PATH"] = str(self.base / "missing")
        with self.assertRaisesRegex(FileNotFoundError, "Correct/unset"):
            dependency.import_act_active_api(repo_root=self.base)
        self.checkout.assert_not_called()

    def test_loaded_package_and_submodule_switches_require_restart(self):
        self.installed_fixture()
        dependency.import_act_passive_module(repo_root=self.base)
        os.environ["SCP_ACT_PATH"] = str(self.package("other"))
        path = list(sys.path)
        with self.assertRaisesRegex(RuntimeError, "Restart the kernel"):
            dependency.import_act_passive_module(repo_root=self.base)
        self.assertEqual(sys.path, path)
        del sys.modules["act"]
        with self.assertRaisesRegex(RuntimeError, "Restart the kernel"):
            dependency.import_act_passive_module(repo_root=self.base)

    def test_nonprepending_checkout_cannot_silently_select_another_package(self):
        self.installed_fixture()
        os.environ["SCP_ACT_PATH"] = str(self.package("explicit"))
        with self.assertRaisesRegex(ImportError, "shadowed.*prepend=True"):
            dependency.ensure_act_on_syspath(repo_root=self.base, prepend=False)
        self.assertNotIn("act", sys.modules)

    def test_missing_package_is_actionable_and_never_downloads_by_default(self):
        with mock.patch.object(dependency.importlib.util, "find_spec", return_value=None):
            with self.assertRaisesRegex(ModuleNotFoundError, "requirements-act.txt.*No checkout was downloaded"):
                dependency.ensure_act_on_syspath(repo_root=self.base)
        self.clone.assert_not_called()

    def test_existing_legacy_checkout_still_works(self):
        legacy = self.package("legacy")
        self.checkout.side_effect = None
        self.checkout.return_value = legacy
        with mock.patch.object(dependency.importlib.util, "find_spec", return_value=None):
            self.assertEqual(dependency.ensure_act_on_syspath(repo_root=self.base), legacy)
        dependency.import_act_active_api(repo_root=self.base)
        self.assertNotIn("SCP_ACT_PATH", os.environ)

    def test_explicit_clone_opt_in_and_destination_are_honored(self):
        cloned = self.package("cloned")
        self.clone.side_effect = None
        self.clone.return_value = cloned
        destination = self.base / "new-checkout"
        os.environ["SCP_ACT_DIR"] = str(destination)
        os.environ["SCP_AUTO_CLONE_ACT"] = "1"
        self.assertEqual(dependency.ensure_act_on_syspath(repo_root=self.base), cloned)
        self.assertEqual(self.clone.call_args.kwargs["target_dir"], destination)
        self.checkout.assert_not_called()

    def test_nonempty_invalid_override_is_never_overwritten(self):
        directory = self.base / "keep"
        directory.mkdir()
        sentinel = directory / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        os.environ["SCP_ACT_DIR"] = str(directory)
        with self.assertRaises(FileNotFoundError):
            dependency.ensure_act_on_syspath(repo_root=self.base, auto_clone=True)
        self.assertEqual(sentinel.read_text(), "keep")
        self.clone.assert_not_called()

    def test_unrelated_act_name_does_not_trigger_import_or_fallback(self):
        root = self.base / "unrelated"
        root.mkdir()
        (root / "act.py").write_text("raise AssertionError('do not execute')\n")
        sys.path.insert(0, str(root))
        with self.assertRaisesRegex(ImportError, "unrelated distribution.*pip install act"):
            dependency.import_act_active_api(repo_root=self.base)
        self.assertNotIn("act", sys.modules)
        self.checkout.assert_not_called()

    def test_import_failure_preserves_original_cause_without_fallback(self):
        self.installed_fixture(module="raise ModuleNotFoundError('missing sklearn', name='sklearn')\n")
        with self.assertRaisesRegex(ImportError, "missing sklearn.*requirements-act.txt") as raised:
            dependency.import_act_active_api(repo_root=self.base)
        self.assertIsInstance(raised.exception.__cause__, ModuleNotFoundError)
        self.checkout.assert_not_called()
        self.clone.assert_not_called()

    def test_missing_passive_methods_are_actionable(self):
        self.installed_fixture(passive="class ACTPassiveModule: pass\n")
        with self.assertRaisesRegex(ImportError, "Incompatible ACT passive API.*compute_spp"):
            dependency.import_act_passive_module(repo_root=self.base)

    def test_incompatible_active_classes_and_methods_are_actionable(self):
        cases = (
            {"module": "class ACTModule: pass\n"},
            {"cell_model": "class ACTCellModel:\n def __init__(self, unsupported): pass\n"},
            {"types": TYPES + "\nclass ConductanceOptions:\n def __init__(self): pass\n"},
        )
        for index, overrides in enumerate(cases):
            with self.subTest(case=index):
                self._clear_modules()
                self.installed_fixture(f"bad-{index}", **overrides)
                with self.assertRaisesRegex(ImportError, "Incompatible ACT active API.*requirements-act.txt"):
                    dependency.import_act_active_api(repo_root=self.base)
        self.clone.assert_not_called()

    def test_active_backend_uses_common_api_even_with_unavailable_old_checkout(self):
        self.installed_fixture()
        from scp_pipeline.tuning.act_active import _import_act_api
        api = _import_act_api({"repo_root": str(self.base / "deleted-checkout")})
        self.assertIn("ACTSimulator", api)
        self.checkout.assert_not_called()

    def test_worker_config_loading_never_searches_host_or_sibling_checkouts(self):
        from scp_pipeline.tuning.act_active import load_act_active_config
        config = self.base / "act_active_config.json"
        for stored in (None, str(self.base / "old-checkout")):
            with self.subTest(recorded_code_location=stored):
                payload = {"workspace": "."}
                if stored is not None:
                    payload["repo_root"] = stored
                config.write_text(json.dumps(payload), encoding="utf-8")
                before = config.read_bytes()
                with contextlib.chdir(self.base), mock.patch(
                    "scp_pipeline.tuning.act_active.resolve_repo_root",
                    side_effect=AssertionError("must not discover a sibling checkout"),
                ):
                    loaded = load_act_active_config(config)
                self.assertEqual(loaded["workspace"], str(self.base))
                if stored is not None:
                    self.assertEqual(loaded["repo_root"], stored)
                self.assertEqual(config.read_bytes(), before)

    def test_setup_check_uses_stage_specific_installed_apis_without_cloning(self):
        self.installed_fixture()
        from scp_pipeline.cli.check import Reporter, _check_external_dependencies
        reporter, output = Reporter(), io.StringIO()
        os.environ["SCP_AUTO_CLONE_ACT"] = "1"
        with contextlib.redirect_stdout(output):
            _check_external_dependencies(reporter, self.base, {"2", "3"}, check_act=True)
        self.assertEqual(reporter.errors, 0, output.getvalue())
        self.assertIn("ACT passive API available", output.getvalue())
        self.assertIn("ACT active API available", output.getvalue())
        self.clone.assert_not_called()

    def test_setup_check_never_clones_missing_act_even_when_opted_in(self):
        from scp_pipeline.cli.check import Reporter, _check_external_dependencies
        reporter, output = Reporter(), io.StringIO()
        os.environ["SCP_AUTO_CLONE_ACT"] = "1"
        with contextlib.redirect_stdout(output), mock.patch.object(dependency.importlib.util, "find_spec", return_value=None):
            _check_external_dependencies(reporter, self.base, {"2"}, check_act=True)
        self.assertEqual(reporter.errors, 1)
        self.clone.assert_not_called()

    def test_unrequested_checks_stay_lazy(self):
        from scp_pipeline.cli.check import Reporter, _check_external_dependencies
        with mock.patch.object(dependency, "import_act_active_api") as active, mock.patch.object(dependency, "import_act_passive_module") as passive:
            _check_external_dependencies(Reporter(), self.base, {"2", "3"})
            _check_external_dependencies(Reporter(), self.base, {"4"}, check_act=True)
        active.assert_not_called()
        passive.assert_not_called()
        self.assertNotIn("act", sys.modules)

    def test_source_pin_and_packaging_extra_match(self):
        with mock.patch("setuptools.setup") as setup:
            exec(compile((ROOT / "setup.py").read_text(), str(ROOT / "setup.py"), "exec"), {"__name__": "__main__"})
        self.assertEqual(setup.call_args.kwargs["extras_require"]["act"], [dependency.ACT_REQUIREMENT])
        requirements = (ROOT / "requirements-act.txt").read_text()
        self.assertIn("-c requirements.txt", requirements)
        self.assertIn(dependency.ACT_REQUIREMENT, requirements)
        self.assertIn("requirements-act.txt", (ROOT / "environment.yml").read_text())

    def test_normal_import_stays_lazy_in_fresh_process(self):
        result = subprocess.run([
            sys.executable, "-c", "import sys; import scp_pipeline.tuning; "
            "assert not any(n == 'act' or n.startswith('act.') for n in sys.modules)",
        ], cwd=self.base, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
