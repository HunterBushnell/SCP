from __future__ import annotations

import ast
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "guides" / "project_workspaces.md"


class ProjectDocumentationTests(unittest.TestCase):
    def test_guide_is_linked_and_explains_additive_identity(self) -> None:
        for relative in ("README.md", "docs/README.md", "projects/README.md"):
            with self.subTest(document=relative):
                self.assertIn("project_workspaces.md", (ROOT / relative).read_text())

        text = GUIDE.read_text(encoding="utf-8")
        for required in (
            "projects/<project>/cells/<cell>/tunes/<tune>/",
            "project_name = None",
            "--project",
            "PROJECT=study_a",
            "workspace_identity",
            "tune_dir_override",
            "biological",
            "not isolate",
            "local exclusions",
            "no automatic migration",
        ):
            with self.subTest(required=required):
                self.assertIn(required, " ".join(text.split()))

    def test_notebook_selection_snippets_are_valid_python(self) -> None:
        snippets = re.findall(r"```python\n(.*?)\n```", GUIDE.read_text(), re.S)
        self.assertGreaterEqual(len(snippets), 3)
        for index, snippet in enumerate(snippets):
            with self.subTest(snippet=index):
                ast.parse(snippet)
        selection = {}
        exec(compile(snippets[1], str(GUIDE), "exec"), {}, selection)
        self.assertEqual(selection["project_name"], "study_a")
        self.assertEqual(selection["cell_name"], "ExampleCell")
        self.assertEqual(selection["tune_name"], "tuned")
        self.assertIsNone(selection["tune_dir_override"])

    def test_project_readme_links_resolve(self) -> None:
        document = ROOT / "projects" / "README.md"
        for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", document.read_text()):
            path = unquote(target.split("#", 1)[0])
            with self.subTest(link=target):
                self.assertTrue((document.parent / path).resolve().exists())


@unittest.skipUnless(shutil.which("git"), "Git is required to verify ignore rules")
class ProjectGitVisibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        subprocess.run(
            ["git", "init", "-q", str(self.root)], check=True, capture_output=True
        )
        shutil.copy2(ROOT / ".gitignore", self.root / ".gitignore")

    def ignored(self, paths: list[str]) -> set[str]:
        result = subprocess.run(
            ["git", "-C", str(self.root), "check-ignore", "--no-index", "--stdin"],
            input="\n".join(paths) + "\n",
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertIn(result.returncode, (0, 1), result.stderr)
        return set(result.stdout.splitlines())

    def test_project_sources_and_configs_remain_git_eligible(self) -> None:
        tune = "projects/study_a/cells/ExampleCell/tunes/orig"
        paths = [
            "projects/README.md",
            f"{tune}/model/Cell.hoc",
            f"{tune}/model/cell.py",
            f"{tune}/modfiles/channel.mod",
            f"{tune}/cell_configs/cell_config.json",
            f"{tune}/cell_configs/sim_config.json",
            f"{tune}/model_sources/native/cell.py",
            f"{tune}/MODEL_SOURCE_PROVENANCE.json",
        ]
        self.assertEqual(self.ignored(paths), set())

    def test_project_generated_products_are_ignored(self) -> None:
        tune = "projects/study_a/cells/ExampleCell/tunes/tuned"
        paths = [
            f"{tune}/output_data/example_run/sim_cfg.json",
            f"{tune}/output_data/example_run/results/traces.npz",
            f"{tune}/notebook_exports/diagnostic.csv",
            f"{tune}/tuning_exports/proposal.json",
            f"{tune}/restore_backups/example/cell_config.json",
            f"{tune}/act_workspace/cell_builder.py",
            f"{tune}/targets_ephys.nwb",
            f"{tune}/modfiles/x86_64/libnrnmech.so",
            f"{tune}/mechanisms/aarch64/libnrnmech.so",
            f"{tune}/.scp_build/hash/sources/mechanisms/channel.mod",
            f"{tune}/model/__pycache__/cell.cpython-311.pyc",
        ]
        self.assertEqual(self.ignored(paths), set(paths))

    def test_existing_curated_public_output_exceptions_remain(self) -> None:
        paths = [
            "cells/PV/tunes/tuned/output_data/PV5_base/results/sim_cfg.json",
            "cells/PV/tunes/tuned/output_data/PV5_tuned/results/sim_cfg.json",
            "cells/SST/tunes/tuned/output_data/SST5_base/results/sim_cfg.json",
            "cells/SST/tunes/tuned/output_data/SST5_tuned/results/sim_cfg.json",
        ]
        self.assertEqual(self.ignored(paths), set())


if __name__ == "__main__":
    unittest.main()
