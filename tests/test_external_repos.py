from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from modules.tuning.external_repos import (
    ExternalRepoSpec,
    ensure_external_repo_checkout,
    resolve_external_repo_checkout,
)
from modules.tuning.passive import normalize_act_passive_metrics


def _spec() -> ExternalRepoSpec:
    return ExternalRepoSpec(
        display_name="ExampleTool",
        directory_name="example-tool",
        package_name="_scp_example_external_tool",
        marker_rel=Path("example_tool") / "__init__.py",
        repo_url="unused",
        path_env_vars=("SCP_TEST_TOOL_PATH", "SCP_TEST_TOOL_DIR"),
        target_dir_env="SCP_TEST_TOOL_DIR",
        repo_url_env="SCP_TEST_TOOL_REPO_URL",
        repo_branch_env="SCP_TEST_TOOL_REPO_BRANCH",
        auto_clone_env="SCP_AUTO_CLONE_TEST_TOOL",
        canonical_path_env="SCP_TEST_TOOL_PATH",
    )


def _create_git_source(base: Path) -> Path:
    source = base / "source"
    marker = source / "example_tool" / "__init__.py"
    marker.parent.mkdir(parents=True)
    marker.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=SCP Tests",
            "-c",
            "user.email=scp-tests@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return source


class ExternalRepoTests(unittest.TestCase):
    def test_resolve_existing_checkout_does_not_clone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "SCP"
            checkout = base / "custom"
            (checkout / "example_tool").mkdir(parents=True)
            (checkout / "example_tool" / "__init__.py").write_text(
                "", encoding="utf-8"
            )
            with mock.patch.dict(
                os.environ,
                {"SCP_TEST_TOOL_PATH": str(checkout)},
                clear=False,
            ):
                resolved = resolve_external_repo_checkout(
                    _spec(),
                    repo_root=root,
                )
            self.assertEqual(resolved, checkout.resolve())

    def test_explicit_action_clones_to_empty_canonical_location(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "SCP"
            root.mkdir()
            source = _create_git_source(base)
            target = base / "mods" / "example-tool"
            original_syspath = list(sys.path)
            with mock.patch.dict(
                os.environ,
                {
                    "SCP_TEST_TOOL_DIR": str(target),
                    "SCP_TEST_TOOL_REPO_URL": str(source),
                },
                clear=False,
            ):
                try:
                    checkout = ensure_external_repo_checkout(
                        _spec(),
                        repo_root=root,
                    )
                    self.assertEqual(checkout, target.resolve())
                    self.assertTrue(
                        (checkout / "example_tool" / "__init__.py").is_file()
                    )
                    self.assertEqual(
                        os.environ["SCP_TEST_TOOL_PATH"],
                        str(target.resolve()),
                    )
                    status = subprocess.run(
                        ["git", "-C", str(checkout), "status", "--short"],
                        check=True,
                        capture_output=True,
                        text=True,
                        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
                    )
                    self.assertEqual(status.stdout, "")
                finally:
                    sys.path[:] = original_syspath

    def test_opt_out_reports_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "SCP"
            root.mkdir()
            with mock.patch.dict(
                os.environ,
                {
                    "SCP_TEST_TOOL_DIR": str(Path(tmp) / "missing"),
                    "SCP_AUTO_CLONE_TEST_TOOL": "0",
                },
                clear=False,
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "SCP_AUTO_CLONE_TEST_TOOL=0",
                ):
                    ensure_external_repo_checkout(_spec(), repo_root=root)

    def test_invalid_nonempty_target_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "SCP"
            root.mkdir()
            target = base / "mods" / "example-tool"
            target.mkdir(parents=True)
            sentinel = target / "keep.txt"
            sentinel.write_text("keep me", encoding="utf-8")
            with mock.patch.dict(
                os.environ,
                {"SCP_TEST_TOOL_DIR": str(target)},
                clear=False,
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "Move it aside",
                ):
                    ensure_external_repo_checkout(_spec(), repo_root=root)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep me")

    def test_failed_clone_leaves_no_partial_install(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "SCP"
            root.mkdir()
            target = base / "mods" / "example-tool"
            with mock.patch.dict(
                os.environ,
                {
                    "SCP_TEST_TOOL_DIR": str(target),
                    "SCP_TEST_TOOL_REPO_URL": str(base / "does-not-exist"),
                },
                clear=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "Could not clone"):
                    ensure_external_repo_checkout(_spec(), repo_root=root)
            self.assertFalse(target.exists())
            leftovers = list(target.parent.glob(".example-tool-clone-*"))
            self.assertEqual(leftovers, [])

    def test_imported_copy_requires_restart_before_switching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "SCP"
            checkout = base / "checkout"
            marker = checkout / "example_tool" / "__init__.py"
            marker.parent.mkdir(parents=True)
            marker.write_text("", encoding="utf-8")
            fake_module = SimpleNamespace(
                __file__=str(base / "other" / "__init__.py")
            )
            original_syspath = list(sys.path)
            with mock.patch.dict(
                os.environ,
                {"SCP_TEST_TOOL_PATH": str(checkout)},
                clear=False,
            ), mock.patch.dict(
                sys.modules,
                {"_scp_example_external_tool": fake_module},
            ):
                try:
                    with self.assertRaisesRegex(RuntimeError, "Restart the kernel"):
                        ensure_external_repo_checkout(_spec(), repo_root=root)
                finally:
                    sys.path[:] = original_syspath


class PristineACTCompatibilityTests(unittest.TestCase):
    def test_original_act_passive_fields_receive_scp_fallbacks(self) -> None:
        pristine_result = SimpleNamespace(
            R_in=120.0,
            tau1=8.5,
            tau2=2.0,
            sag_ratio=0.2,
            V_rest=-70.0,
        )
        fallback = {
            "R_in_rest_to_final": 100.0,
            "tau_rest_to_trough": 8.0,
            "tau_avg": 7.0,
            "sag_ratio": 0.25,
            "V_rest": -69.0,
        }
        normalized = normalize_act_passive_metrics(
            pristine_result,
            fallback=fallback,
        )
        self.assertEqual(normalized["R_in_rest_to_final"], 100.0)
        self.assertEqual(normalized["tau_rest_to_trough"], 8.5)
        self.assertEqual(normalized["tau_avg"], 7.0)
        self.assertEqual(normalized["sag_ratio"], 0.2)
        self.assertEqual(normalized["V_rest"], -70.0)

    def test_current_act_passive_fields_are_used_directly(self) -> None:
        current_result = SimpleNamespace(
            R_in_rest_to_final=np.float64(105.0),
            tau_rest_to_trough=np.float64(9.0),
            tau_avg=np.float64(8.0),
            sag_ratio=np.float64(0.1),
            V_rest=np.float64(-71.0),
        )
        normalized = normalize_act_passive_metrics(
            current_result,
            fallback={},
        )
        self.assertEqual(
            normalized,
            {
                "R_in_rest_to_final": 105.0,
                "tau_rest_to_trough": 9.0,
                "tau_avg": 8.0,
                "sag_ratio": 0.1,
                "V_rest": -71.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
