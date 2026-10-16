"""Reader-facing contracts for the compact notebook when copied into a project.

These checks intentionally avoid scientific defaults and executable notebook
settings: a user's local cell selection is independent of portable guidance.
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
# Matching the closing label boundary also catches a linked image's outer
# target, such as the Colab badge, without assuming labels are unnested.
LINK_PATTERN = re.compile(r"\]\(([^)]+)\)")


def _plain(text: str) -> str:
    """Normalize presentation without requiring an exact Markdown sentence."""
    return " ".join(re.sub(r"[*`]", "", text).lower().split())


class PortableNotebookGuidanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        notebook = json.loads((REPO_ROOT / "0_pipeline.ipynb").read_text(encoding="utf-8"))
        cls.markdown_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        ]
        cls.markdown = "\n\n".join(cls.markdown_cells)
        cls.text = _plain(cls.markdown)

    def _step(self, number: int) -> str:
        matching = [
            cell for cell in self.markdown_cells
            if re.search(rf"(?m)^#+\s+Step\s+{number}\s*:", cell)
        ]
        self.assertEqual(len(matching), 1, f"Expected one guidance section for Step {number}")
        return _plain(matching[0])

    def test_markdown_links_survive_copying_the_notebook(self) -> None:
        targets = LINK_PATTERN.findall(self.markdown)
        self.assertTrue(targets, "The portable notebook needs links to full guidance")
        local_document_links = 0
        for raw_target in targets:
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            with self.subTest(target=target):
                parsed = urlsplit(target)
                self.assertTrue(
                    target.startswith("#") or parsed.scheme in {"http", "https"},
                    "Relative checkout links break when the notebook is copied into a project",
                )
                if parsed.scheme:
                    self.assertTrue(parsed.netloc, "Web links must include a host")
                prefix = "/HunterBushnell/SCP/blob/main/"
                if parsed.netloc.lower() != "github.com" or not parsed.path.startswith(prefix):
                    continue
                relative = unquote(parsed.path[len(prefix):])
                if not (relative.startswith("docs/") or relative.endswith(".ipynb")):
                    continue
                local_document_links += 1
                path = (REPO_ROOT / relative).resolve()
                self.assertTrue(path.is_relative_to(REPO_ROOT))
                self.assertTrue(path.is_file(), f"Missing SCP documentation target: {relative}")
        self.assertGreater(local_document_links, 0)

    def test_project_navigation_and_explicit_persistence_are_explained(self) -> None:
        for term in (
            "project-owned workspace", "project dir", "kernel", "workspaces",
            "cells/<cell>/tunes/<tune>", "tune path", "results dir",
            "output_data", ".scp_project.json", "open / reload project",
        ):
            with self.subTest(term=term):
                self.assertIn(term, self.text)
        self.assertRegex(self.text, r"(?:session-only|session only)")
        self.assertIn("scientific", self.text)
        self.assertIn("save project settings", self.text)
        self.assertIn("save notebook defaults", self.text)
        self.assertNotIn("notebook-relative paths resolve from the scp root", self.text)

    def test_new_registration_is_available_without_the_detailed_setup_notebook(self) -> None:
        step = self._step(1)
        for term in (
            "open existing workspace", "register new model", "entry file",
            "template", "callable", "preview model setup", "save cell setup",
            "prepare and load",
        ):
            with self.subTest(term=term):
                self.assertIn(term, step)
        for term in ("source root", "mod folder", "files json", "folders json"):
            with self.subTest(term=term):
                self.assertIn(term, self.text)
        self.assertIn("temperature", self.text)
        self.assertRegex(self.text, r"(?:initialization voltage|initial voltage|voltage/temperature)")
        self.assertNotIn("select an already prepared cell/tune", step)

    def test_manual_protocols_and_current_clamp_do_not_require_optional_tools(self) -> None:
        passive = self._step(2)
        active = self._step(3)
        simulation = self._step(5)
        self.assertIn("run passive", passive)
        self.assertIn("optional", passive)
        self.assertIn("act", passive)
        self.assertIn("active protocol", active)
        self.assertIn("fi curve", active)
        self.assertRegex(active, r"(?:independent|separate)")
        for term in ("current clamp", "run simulation", "plot results", "check inputs", "synap"):
            with self.subTest(term=term):
                self.assertIn(term, simulation)
        self.assertRegex(simulation, r"(?:skip|only for|synapse inputs only|synaptic inputs only)")
        self.assertNotIn("use check inputs, run simulation, then plot results", simulation)

    def test_optional_tools_and_live_object_process_boundaries_are_visible(self) -> None:
        synapses = self._step(4)
        self.assertIn("bmtool", synapses)
        self.assertRegex(synapses, r"(?:optional|skip)")
        self.assertIn("cell_override", self.text)
        self.assertIn("runtime_sim_config", self.text)
        self.assertRegex(self.text, r"(?:recreat|rebuild).*(?:kernel|restart)")
        self.assertIn("python builder", self.text)
        self.assertIn("experimental", self._step(3))


if __name__ == "__main__":
    unittest.main()
