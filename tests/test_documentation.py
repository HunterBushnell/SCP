from __future__ import annotations

import re
import unittest
from pathlib import Path
from urllib.parse import unquote


REPO_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_FILES = [
    REPO_ROOT / "README.md",
    *sorted((REPO_ROOT / "docs").rglob("*.md")),
]
LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


class DocumentationTests(unittest.TestCase):
    def test_relative_markdown_links_resolve(self) -> None:
        failures: list[str] = []
        for document in MARKDOWN_FILES:
            text = document.read_text(encoding="utf-8")
            for target in LINK_PATTERN.findall(text):
                target = target.strip().split(maxsplit=1)[0].strip("<>")
                if not target or target.startswith(
                    ("#", "http://", "https://", "mailto:")
                ):
                    continue
                path_text = unquote(target.split("#", 1)[0])
                if not path_text:
                    continue
                if "/" not in path_text and Path(path_text).suffix.lower() not in {
                    ".md",
                    ".ipynb",
                    ".py",
                    ".sh",
                }:
                    continue
                resolved = (document.parent / path_text).resolve()
                if not resolved.exists():
                    failures.append(
                        f"{document.relative_to(REPO_ROOT)} -> {target}"
                    )
        self.assertEqual(failures, [], "Broken local Markdown links:\n" + "\n".join(failures))

    def test_compact_pipeline_release_language_is_consistent(self) -> None:
        required = {
            "README.md": (
                "recommended compact Steps 1–5 front door",
                "experimental, review-only, and not release-blocking",
            ),
            "docs/quickstart.md": (
                "Run All",
                "session-only",
                "fresh process",
            ),
            "docs/pipeline_overview.md": (
                "exactly one shared tuning cell",
                "Fresh-Process Boundary",
                "not release-blocking",
            ),
        }
        for relative, phrases in required.items():
            text = (REPO_ROOT / relative).read_text(encoding="utf-8")
            normalized = " ".join(text.split())
            for phrase in phrases:
                with self.subTest(document=relative, phrase=phrase):
                    self.assertIn(phrase, normalized)


if __name__ == "__main__":
    unittest.main()
