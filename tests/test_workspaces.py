from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from modules.workspaces import (
    WorkspaceMismatchError,
    apply_workspace_metadata,
    assert_same_workspace,
    discover_cells,
    discover_projects,
    discover_tunes,
    infer_workspace,
    resolve_workspace,
    result_workspace_identity,
    saved_namespace_tune_dir,
)


ROOT = Path(__file__).resolve().parents[1]


class WorkspaceResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "SCP"
        self.root.mkdir()

    def workspace(self, project=None, cell="ExampleCell", tune="tuned") -> Path:
        base = self.root / "cells" if project is None else self.root / "projects" / project / "cells"
        path = base / cell / "tunes" / tune
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_legacy_and_project_shorthand_are_distinct_without_creating_paths(self) -> None:
        before = list(self.root.rglob("*"))
        legacy = resolve_workspace(repo_root=self.root, cell_name="PV", tune_name="tuned")
        project = resolve_workspace(repo_root=self.root, project_name="study_a", cell_name="PV", tune_name="tuned")
        self.assertEqual(legacy.tune_dir, self.root / "cells/PV/tunes/tuned")
        self.assertEqual(project.tune_dir, self.root / "projects/study_a/cells/PV/tunes/tuned")
        self.assertEqual(legacy.layout, "legacy")
        self.assertEqual(project.layout, "project")
        self.assertEqual(project.identity, {"project_name": "study_a", "cell_name": "PV", "tune_name": "tuned"})
        self.assertEqual(project.label, "study_a/PV/tuned")
        self.assertEqual(list(self.root.rglob("*")), before)

    def test_explicit_path_wins_over_conflicting_project_cell_and_tune(self) -> None:
        tune = self.workspace("study_b", "Chosen", "orig")
        resolved = resolve_workspace(repo_root=self.root, project_name="study_a", cell_name="PV", tune_name="tuned",
                                     tune_dir_override=tune.relative_to(self.root))
        self.assertEqual(resolved.tune_dir, tune)
        self.assertEqual(resolved.identity, {"project_name": "study_b", "cell_name": "Chosen", "tune_name": "orig"})

    def test_explicit_path_parent_and_custom_shorthand_are_inferred_consistently(self) -> None:
        normal = self.workspace("study_b", "Chosen", "orig")
        explicit = resolve_workspace(repo_root=self.root, tune_dir_override=normal,
                                     project_name="wrong", cell_name="PV", tune_name="wrong", tunes_parent="variants")
        self.assertEqual(explicit.identity, {"project_name": "study_b", "cell_name": "Chosen", "tune_name": "orig"})
        custom = resolve_workspace(repo_root=self.root, project_name="study_a", cell_name="Chosen",
                                   tune_name="tuned", tunes_parent="variants", cell_config={})
        self.assertEqual(custom.tune_dir, self.root / "projects/study_a/cells/Chosen/variants/tuned")
        self.assertEqual(infer_workspace(custom.tune_dir, cell_config={}).identity, custom.identity)

    def test_explicit_cli_relative_base_and_external_workspaces_are_supported(self) -> None:
        tune = self.workspace("study_a")
        resolved = resolve_workspace(repo_root=self.root, relative_to=self.root.parent,
                                     tune_dir_override=tune.relative_to(self.root.parent))
        self.assertEqual(resolved.tune_dir, tune)
        external = self.root.parent / "outside" / "Custom" / "tunes" / "orig"
        resolved = resolve_workspace(repo_root=self.root, tune_dir_override=external,
                                     cell_config={"cell_name": "Custom", "tune": "orig"})
        self.assertEqual(resolved.layout, "external")
        self.assertEqual(resolved.cell_name, "Custom")
        self.assertIsNone(resolved.project_name)
        inferred = resolve_workspace(repo_root=self.root, tune_dir_override=external,
                                     cell_config={})
        self.assertEqual(inferred.identity, {"project_name": None, "cell_name": "Custom", "tune_name": "orig"})

    def test_adjacent_structure_not_first_cells_ancestor_controls_inference(self) -> None:
        tune = self.root.parent / "cells" / "misleading" / "projects" / "study_a" / "cells" / "Correct" / "tunes" / "tuned"
        resolved = infer_workspace(tune, cell_config={"cell_name": "PV", "tune": "orig"})
        self.assertEqual(resolved.identity, {"project_name": "study_a", "cell_name": "Correct", "tune_name": "tuned"})

    def test_optional_stored_namespace_is_read_without_rewriting_config(self) -> None:
        tune = self.workspace("study_a")
        config = tune / "cell_configs/cell_config.json"
        config.parent.mkdir()
        original = b'{\r\n  "cell_name": "ExampleCell", "tune": "tuned"\r\n}\r\n'
        config.write_bytes(original)
        resolved = infer_workspace(tune)
        self.assertEqual(resolved.project_name, "study_a")
        self.assertEqual(config.read_bytes(), original)
        config.write_text(json.dumps({"project_name": "wrong", "cell_name": "ExampleCell"}))
        before = config.read_bytes()
        with self.assertRaisesRegex(ValueError, "conflicts"):
            infer_workspace(tune)
        self.assertEqual(config.read_bytes(), before)

    def test_project_metadata_cannot_relabel_legacy_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicts"):
            infer_workspace(self.workspace(), cell_config={"project_name": "study_a"})

    def test_unsafe_shorthand_components_are_rejected(self) -> None:
        for field in ("project_name", "cell_name", "tune_name", "tunes_parent"):
            for value in ("", " ", ".", "..", "../other", "/absolute", "a/b", "a\\b", "bad\x00name"):
                kwargs = dict(repo_root=self.root, project_name="study_a", cell_name="ExampleCell", tune_name="tuned")
                kwargs[field] = value
                with self.subTest(field=field, value=repr(value)), self.assertRaises(ValueError):
                    resolve_workspace(**kwargs)
        valid = resolve_workspace(repo_root=self.root, project_name="study-a", cell_name="Cell_2-", tune_name="tuned.v2")
        self.assertEqual(valid.cell_name, "Cell_2-")

    def test_deterministic_discovery_handles_empty_roots_and_never_executes_models(self) -> None:
        self.assertEqual(discover_projects(self.root), [])
        self.assertEqual(discover_cells(self.root), [])
        self.assertEqual(discover_tunes(self.root, "Missing", "missing"), [])
        for project in ("z_project", "a_project"):
            for cell in ("ZCell", "ACell"):
                for tune_name in ("tuned", "orig"):
                    tune = self.workspace(project, cell, tune_name)
                    (tune / "network_driver.py").write_text("raise AssertionError('discovery must not execute Python')\n")
        (self.root / "projects/.hidden").mkdir()
        self.workspace(None, "PV")
        self.assertEqual(discover_projects(self.root), ["a_project", "z_project"])
        self.assertEqual(discover_cells(self.root), ["PV"])
        self.assertEqual(discover_cells(self.root, "a_project"), ["ACell", "ZCell"])
        self.assertEqual(discover_tunes(self.root, "ACell", "a_project"), ["orig", "tuned"])

    @unittest.skipUnless(hasattr(os, "symlink"), "Symlink checks require symlink support")
    def test_namespace_and_cell_symlinks_cannot_escape(self) -> None:
        outside = self.root.parent / "outside"
        outside.mkdir()
        (self.root / "projects").mkdir()
        (self.root / "projects/escaped").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            resolve_workspace(repo_root=self.root, project_name="escaped", cell_name="Cell", tune_name="orig")
        with self.assertRaisesRegex(ValueError, "escapes"):
            discover_projects(self.root)
        (self.root / "projects/escaped").unlink()
        first = self.workspace("study_a")
        second = self.workspace("study_b")
        escaped_cell = first.parent.parent.parent / "ForeignCell"
        escaped_cell.symlink_to(second.parent.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            resolve_workspace(repo_root=self.root, project_name="study_a", cell_name="ForeignCell", tune_name="tuned")
        with self.assertRaisesRegex(ValueError, "escapes"):
            discover_cells(self.root, "study_a")
        with self.assertRaisesRegex(ValueError, "escapes"):
            resolve_workspace(repo_root=self.root, tune_dir_override=escaped_cell / "tunes/tuned")

    @unittest.skipUnless(hasattr(os, "symlink"), "Symlink checks require symlink support")
    def test_project_alias_cannot_silently_select_another_project(self) -> None:
        self.workspace("study_b")
        (self.root / "projects/study_a").symlink_to(self.root / "projects/study_b", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "alias|escapes"):
            resolve_workspace(repo_root=self.root, project_name="study_a", cell_name="ExampleCell", tune_name="tuned")
        with self.assertRaisesRegex(ValueError, "alias|escapes"):
            discover_projects(self.root)

    def test_effective_routing_replaces_stale_values_in_memory_only(self) -> None:
        workspace = infer_workspace(self.workspace("study_a"), cell_config={})
        raw = {"cell": "PV", "tune": "orig", "tune_dir": "/old/workspace", "conditions": {"celsius_C": 34}}
        original = deepcopy(raw)
        updated = apply_workspace_metadata(deepcopy(raw), workspace)
        self.assertEqual(updated["tune_dir"], str(workspace.tune_dir))
        self.assertEqual(updated["workspace_identity"], workspace.identity)
        self.assertEqual(updated["conditions"], raw["conditions"])
        self.assertEqual(raw, original)

    def test_module_import_and_discovery_do_not_import_neuron(self) -> None:
        program = "import sys; from modules.workspaces import discover_projects; discover_projects(sys.argv[1]); assert 'neuron' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", program, str(self.root)], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class SavedWorkspaceIdentityTests(unittest.TestCase):
    def test_new_identity_is_portable_and_agrees_across_saved_locations(self) -> None:
        identity = {"project_name": "study_a", "cell_name": "ExampleCell", "tune_name": "tuned"}
        for prefix in ("/old/server/SCP", "/new/checkout/SCP"):
            result = {"sim_cfg": {"workspace_identity": identity, "tune_dir": prefix + "/projects/study_a/cells/ExampleCell/tunes/tuned"},
                      "meta": {"workspace_identity": deepcopy(identity)}}
            self.assertEqual(result_workspace_identity(result), identity)

    def test_historical_identity_can_be_inferred_without_existing_sources(self) -> None:
        self.assertEqual(result_workspace_identity({"sim_cfg": {"tune_dir": "/missing/SCP/cells/PV/tunes/tuned"}}),
                         {"project_name": None, "cell_name": "PV", "tune_name": "tuned"})
        self.assertEqual(result_workspace_identity({"sim_cfg": {"tune_dir": "/missing/SCP/projects/study_a/cells/PV/tunes/tuned"}}),
                         {"project_name": "study_a", "cell_name": "PV", "tune_name": "tuned"})
        self.assertEqual(result_workspace_identity({"sim_cfg": {"cell": "PV", "tune": "tuned"}}),
                         {"project_name": None, "cell_name": "PV", "tune_name": "tuned"})

    def test_conflicting_saved_identity_fields_fail(self) -> None:
        identity = {"project_name": "study_a", "cell_name": "PV", "tune_name": "tuned"}
        for result in (
            {"sim_cfg": {"workspace_identity": identity}, "meta": {"workspace_identity": {**identity, "project_name": "study_b"}}},
            {"sim_cfg": {"workspace_identity": identity, "project_name": "study_b"}},
            {"sim_cfg": {"workspace_identity": identity, "tune_dir": "/SCP/projects/study_b/cells/PV/tunes/tuned"}},
        ):
            with self.subTest(result=result), self.assertRaises(WorkspaceMismatchError):
                result_workspace_identity(result)
        with self.assertRaisesRegex(ValueError, "object"):
            result_workspace_identity({"workspace_identity": "not an object"})

    def test_pooling_guards_every_identity_component_and_keeps_legacy_warning(self) -> None:
        base = {"project_name": "study_a", "cell": "PV", "tune": "tuned"}
        self.assertEqual(assert_same_workspace(base, deepcopy(base)), [])
        for key, value in (("project_name", "study_b"), ("cell", "SST"), ("tune", "orig"), ("project_name", None)):
            with self.subTest(key=key), self.assertRaises(WorkspaceMismatchError):
                assert_same_workspace(base, {**base, key: value})
        with self.assertRaisesRegex(WorkspaceMismatchError, "complete"):
            assert_same_workspace(base, {"project_name": "study_a", "cell": "PV"})
        warnings = assert_same_workspace({"cell": "PV"}, {"cell": "PV", "tune": "tuned"})
        self.assertEqual(len(warnings), 1)
        self.assertIn("Incomplete", warnings[0])

    def test_saved_namespace_fallback_routes_project_and_legacy_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = saved_namespace_tune_dir({"project_name": "study_a", "cell": "PV", "tune": "tuned"}, root)
            legacy = saved_namespace_tune_dir({"cell": "PV", "tune": "tuned"}, root)
            self.assertEqual(project, root / "projects/study_a/cells/PV/tunes/tuned")
            self.assertEqual(legacy, root / "cells/PV/tunes/tuned")
            self.assertIsNone(saved_namespace_tune_dir({"cell": "PV"}, root))


if __name__ == "__main__":
    unittest.main()
