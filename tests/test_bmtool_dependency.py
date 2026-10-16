"""Environment-installed BMTool is optional and independent of checkout paths."""
from __future__ import annotations

import contextlib
import importlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scp_pipeline.tuning import bmtool_dependency as dependency
from scp_pipeline.tuning.bmtool_synapse_adapter import ensure_bmtool_on_syspath


ROOT = Path(__file__).resolve().parents[1]
API = '''
class SynapseTuner:
    def __init__(self, *, hoc_cell=None, **kwargs):
        raise AssertionError("Dependency checks must never construct a tuner")

class SynapseOptimizer:
    def __init__(self, tuner):
        raise AssertionError("Dependency checks must never construct an optimizer")
'''


class BMToolDependencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.path = list(sys.path)
        self.modules = {k: v for k, v in sys.modules.items() if k == "bmtool" or k.startswith("bmtool.")}
        self._clear_modules()
        self.addCleanup(self._restore_imports)
        self.env = mock.patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        for key in (*dependency.BMTOOL_ENV_VARS, "SCP_AUTO_CLONE_BMTOOL"):
            os.environ.pop(key, None)
        missing = mock.patch.object(dependency, "resolve_external_repo_checkout", side_effect=FileNotFoundError("missing"))
        self.checkout = missing.start()
        self.addCleanup(missing.stop)
        clone = mock.patch.object(dependency, "_clone_external_repo", side_effect=AssertionError("must not download"))
        self.clone = clone.start()
        self.addCleanup(clone.stop)

    def _clear_modules(self):
        for name in list(sys.modules):
            if name == "bmtool" or name.startswith("bmtool."):
                del sys.modules[name]

    def _restore_imports(self):
        self._clear_modules()
        sys.modules.update(self.modules)
        sys.path[:] = self.path
        importlib.invalidate_caches()

    def package(self, name, api=API):
        root = self.base / name
        folder = root / "bmtool"
        folder.mkdir(parents=True)
        (folder / "__init__.py").write_text("", encoding="utf-8")
        (folder / "synapses.py").write_text(api, encoding="utf-8")
        importlib.invalidate_caches()
        return root

    def install_fixture_on_path(self, name="site-packages", api=API):
        root = self.package(name, api)
        sys.path.insert(0, str(root))
        return root

    def test_available_package_needs_no_checkout_or_path_mutation(self):
        installed = self.install_fixture_on_path()
        old_path = list(sys.path)
        old_env = dict(os.environ)
        self.assertEqual(ensure_bmtool_on_syspath(repo_root=self.base), installed)
        tuner, optimizer = dependency.import_bmtool_synapse_api(repo_root=self.base)
        self.assertEqual(tuner.__name__, "SynapseTuner")
        self.assertEqual(optimizer.__name__, "SynapseOptimizer")
        self.assertEqual(sys.path, old_path)
        self.assertEqual(dict(os.environ), old_env)
        self.checkout.assert_not_called()
        self.clone.assert_not_called()

    def test_installed_package_wins_over_automatically_discovered_checkout(self):
        installed = self.install_fixture_on_path()
        self.checkout.side_effect = None
        self.checkout.return_value = self.package("legacy")
        self.assertEqual(dependency.ensure_bmtool_available(repo_root=self.base), installed)
        self.checkout.assert_not_called()

    def test_explicit_checkout_overrides_unimported_installed_package(self):
        self.install_fixture_on_path()
        checkout = self.package("explicit")
        os.environ["SCP_BMTOOL_PATH"] = str(checkout)
        dependency.import_bmtool_synapse_api(repo_root=self.base)
        self.assertEqual(Path(sys.modules["bmtool.synapses"].__file__).parent.parent, checkout)
        self.clone.assert_not_called()

    def test_invalid_explicit_override_does_not_silently_use_installed_package(self):
        self.install_fixture_on_path()
        os.environ["SCP_BMTOOL_PATH"] = str(self.base / "invalid")
        with self.assertRaisesRegex(FileNotFoundError, "Correct/unset"):
            dependency.import_bmtool_synapse_api(repo_root=self.base)
        self.checkout.assert_not_called()

    def test_nonprepending_checkout_cannot_silently_import_shadowing_package(self):
        self.install_fixture_on_path()
        os.environ["SCP_BMTOOL_PATH"] = str(self.package("explicit"))
        with self.assertRaisesRegex(ImportError, "shadowed.*prepend=True"):
            dependency.ensure_bmtool_available(repo_root=self.base, prepend=False)
        self.assertNotIn("bmtool", sys.modules)

    def test_namespace_collision_is_reported_without_fallback(self):
        spec = importlib.machinery.ModuleSpec("bmtool", loader=None, is_package=True)
        with mock.patch.object(dependency.importlib.util, "find_spec", return_value=spec), self.assertRaisesRegex(ImportError, "namespace package.*name collision"):
            dependency.import_bmtool_synapse_api(repo_root=self.base)
        self.checkout.assert_not_called()

    def test_switching_imported_packages_requires_restart(self):
        self.install_fixture_on_path()
        dependency.import_bmtool_synapse_api(repo_root=self.base)
        os.environ["SCP_BMTOOL_PATH"] = str(self.package("other"))
        old_path = list(sys.path)
        with self.assertRaisesRegex(RuntimeError, "Restart the kernel"):
            dependency.import_bmtool_synapse_api(repo_root=self.base)
        self.assertEqual(sys.path, old_path)

    def test_stale_submodule_also_requires_restart(self):
        self.install_fixture_on_path()
        dependency.import_bmtool_synapse_api(repo_root=self.base)
        del sys.modules["bmtool"]
        os.environ["SCP_BMTOOL_PATH"] = str(self.package("other"))
        with self.assertRaisesRegex(RuntimeError, "Restart the kernel"):
            dependency.import_bmtool_synapse_api(repo_root=self.base)

    def test_import_failure_preserves_cause_without_falling_back(self):
        self.install_fixture_on_path(api="raise ModuleNotFoundError('missing transitive dependency', name='_missing_dependency')\n")
        with self.assertRaisesRegex(ImportError, "missing transitive dependency") as raised:
            dependency.import_bmtool_synapse_api(repo_root=self.base)
        self.assertIsInstance(raised.exception.__cause__, ModuleNotFoundError)
        self.checkout.assert_not_called()
        self.clone.assert_not_called()

    def test_missing_or_incompatible_apis_are_actionable(self):
        cases = (
            "class SynapseTuner: pass\n",
            "class SynapseTuner:\n def __init__(self, templates_dir): pass\nclass SynapseOptimizer:\n def __init__(self, tuner): pass\n",
            "class SynapseTuner:\n def __init__(self, **kwargs): pass\nclass SynapseOptimizer:\n def __init__(self): pass\n",
        )
        for index, source in enumerate(cases):
            with self.subTest(case=index):
                self._clear_modules()
                self.install_fixture_on_path(f"incompatible-{index}", source)
                with self.assertRaisesRegex(ImportError, "Incompatible BMTool.*0.8.6"):
                    dependency.import_bmtool_synapse_api(repo_root=self.base)
        self.clone.assert_not_called()

    def test_missing_dependency_never_downloads_by_default(self):
        with mock.patch.object(dependency.importlib.util, "find_spec", return_value=None):
            with self.assertRaisesRegex(ModuleNotFoundError, "requirements-bmtool.txt.*No checkout was downloaded"):
                dependency.ensure_bmtool_available(repo_root=self.base)
        self.clone.assert_not_called()

    def test_existing_legacy_checkout_is_still_usable(self):
        checkout = self.package("legacy")
        self.checkout.side_effect = None
        self.checkout.return_value = checkout
        with mock.patch.object(dependency.importlib.util, "find_spec", return_value=None):
            self.assertEqual(dependency.ensure_bmtool_available(repo_root=self.base), checkout)
        dependency.import_bmtool_synapse_api(repo_root=self.base)
        self.assertNotIn("SCP_BMTOOL_PATH", os.environ)

    def test_automatic_clone_requires_explicit_opt_in(self):
        checkout = self.package("cloned")
        self.clone.side_effect = None
        self.clone.return_value = checkout
        os.environ["SCP_AUTO_CLONE_BMTOOL"] = "1"
        with mock.patch.object(dependency.importlib.util, "find_spec", return_value=None):
            self.assertEqual(dependency.ensure_bmtool_available(repo_root=self.base), checkout)
        self.clone.assert_called_once()

    def test_explicit_clone_destination_does_not_select_unrelated_checkout(self):
        requested = self.base / "destination"
        self.clone.side_effect = None
        self.clone.return_value = self.package("cloned")
        dependency.ensure_bmtool_available(repo_root=self.base, target_dir=requested, auto_clone=True)
        self.assertEqual(self.clone.call_args.kwargs["target_dir"], requested)
        self.checkout.assert_not_called()

    def test_setup_check_accepts_installed_api_and_never_clones(self):
        self.install_fixture_on_path()
        from scp_pipeline.cli.check import Reporter, _check_external_dependencies
        reporter = Reporter()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _check_external_dependencies(reporter, self.base, {"4"}, check_bmtool=True)
        self.assertEqual(reporter.errors, 0, output.getvalue())
        self.assertIn("BMTool synapse API available", output.getvalue())
        self.clone.assert_not_called()

    def test_setup_check_cannot_clone_even_with_clone_environment_flag(self):
        from scp_pipeline.cli.check import Reporter, _check_external_dependencies
        reporter = Reporter()
        output = io.StringIO()
        os.environ["SCP_AUTO_CLONE_BMTOOL"] = "1"
        with contextlib.redirect_stdout(output), mock.patch.object(dependency.importlib.util, "find_spec", return_value=None):
            _check_external_dependencies(reporter, self.base, {"4"}, check_bmtool=True)
        self.assertEqual(reporter.errors, 1)
        self.assertIn("Install bmtool==0.8.6", output.getvalue())
        self.clone.assert_not_called()

    def test_optional_checks_do_not_import_bmtool(self):
        from scp_pipeline.cli.check import Reporter, _check_external_dependencies
        with mock.patch.object(dependency, "import_bmtool_synapse_api") as api:
            _check_external_dependencies(Reporter(), self.base, {"4"})
            _check_external_dependencies(Reporter(), self.base, {"2"}, check_bmtool=True)
        api.assert_not_called()
        self.assertNotIn("bmtool", sys.modules)

    def test_packaging_extra_and_optional_requirements_share_pin(self):
        with mock.patch("setuptools.setup") as setup:
            exec(compile((ROOT / "setup.py").read_text(), str(ROOT / "setup.py"), "exec"), {"__name__": "__main__"})
        self.assertEqual(setup.call_args.kwargs["extras_require"]["bmtool"], [dependency.BMTOOL_REQUIREMENT])
        requirements = (ROOT / "requirements-bmtool.txt").read_text()
        self.assertIn("-c requirements.txt", requirements)
        self.assertIn(dependency.BMTOOL_REQUIREMENT, requirements)
        self.assertNotIn("bmtool==", (ROOT / "requirements.txt").read_text())
        self.assertIn("requirements-bmtool.txt", (ROOT / "environment.yml").read_text())

    def test_normal_project_import_is_lazy_in_fresh_process(self):
        result = subprocess.run([
            sys.executable, "-c", "import sys; import scp_pipeline; import scp_pipeline.tuning; "
            "assert not any(n == 'bmtool' or n.startswith('bmtool.') for n in sys.modules)",
        ], cwd=self.base, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
