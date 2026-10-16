"""Transfer contracts, using isolated notebook ASTs and no real Git operations."""
from __future__ import annotations

import ast
import builtins
import hashlib
import json
import re
import types
import unittest
from pathlib import Path
from unittest.mock import Mock


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_REPO = "https://github.com/cyneuro/SCP"
LAB = "Neural Engineering Laboratory at the University of Missouri"
CONTACT_EMAIL = "hrbncv@missouri.edu"
UPDATE_NOTEBOOKS = (
    "1_setup.ipynb",
    "2_passive.ipynb",
    "3_active.ipynb",
    "4_synapses.ipynb",
    "5_simulate.ipynb",
    "6_analysis.ipynb",
    "7_tools.ipynb",
    "extra_notebooks/act_segmentation.ipynb",
)
PUBLIC_NOTEBOOKS = ("0_pipeline.ipynb", *UPDATE_NOTEBOOKS)
HELPER_NAMES = {"_scp_repo_slug", "_scp_repo_matches"}
# Custody/contact metadata does not change the license or third-party credits.
PRESERVED_CREDIT_HASHES = {
    "LICENSE": "89b37287afc58cdf90b79bc8e9b7c3e2b596b17505c12cc3faa370c8b5b4e65e",
    "THIRD_PARTY_NOTICES.md": "5d22dbd001e077273e53709853173c81570e7cca66ffbbd54530f12e6aa88343",
}


def _notebook(name: str) -> dict:
    return json.loads((REPO_ROOT / name).read_text(encoding="utf-8"))


def _code_trees(name: str) -> list[ast.Module]:
    return [
        ast.parse("".join(cell.get("source", [])))
        for cell in _notebook(name)["cells"]
        if cell["cell_type"] == "code"
    ]


def _bootstrap_tree(name: str) -> ast.Module:
    for tree in _code_trees(name):
        if any(
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "SCP_REPO_URL"
                    for target in node.targets)
            for node in ast.walk(tree)
        ):
            return tree
    raise AssertionError(f"No SCP bootstrap in {name}")


def _helper_nodes(name: str) -> list[ast.FunctionDef]:
    nodes = [
        node for node in _bootstrap_tree(name).body
        if isinstance(node, ast.FunctionDef) and node.name in HELPER_NAMES
    ]
    if {node.name for node in nodes} != HELPER_NAMES:
        raise AssertionError(f"Expected both inline repository helpers in {name}")
    return nodes


def _compile_nodes(nodes: list[ast.stmt], filename: str):
    return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                   filename, "exec")


def _helpers(name: str) -> dict:
    namespace: dict = {}
    exec(_compile_nodes(_helper_nodes(name), name), namespace)
    return namespace


def _update_nodes(name: str) -> list[ast.stmt]:
    """Extract only helpers, path/enable guards, and the guarded update block."""
    body = _bootstrap_tree(name).body
    start = next(
        index for index, node in enumerate(body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "repo_root_text"
                for target in node.targets)
    )
    stop = next(
        index for index in range(start, len(body))
        if isinstance(body[index], ast.If)
        and any(isinstance(node, ast.Name) and node.id == "auto_update_repo"
                for node in ast.walk(body[index].test))
    )
    return [*_helper_nodes(name), *body[start:stop + 1]]


class _MockRepoPath:
    """Only the path operations used by the isolated update guard are allowed."""

    def __init__(self, path: str, git_dir: bool):
        self.path = path
        self.git_dir = git_dir

    def __str__(self) -> str:
        return self.path

    def __truediv__(self, child: str):
        if child != ".git":
            raise AssertionError(f"Unexpected filesystem access: {child}")
        return types.SimpleNamespace(is_dir=lambda: self.git_dir)


def _run_update(
    name: str,
    *,
    origin: str = "https://github.com/HunterBushnell/SCP.git",
    configured: str = CANONICAL_REPO + ".git",
    path: str = "/content/SCP",
    in_colab: bool = True,
    enabled: str = "1",
    git_dir: bool = True,
    changes: str = "",
    current_branch: str = "main",
    configured_branch: str | None = None,
    output_error: Exception | None = None,
    output_error_command: str = "status",
    call_error: Exception | None = None,
    call_error_command: str = "fetch",
) -> tuple[Mock, Mock]:
    outputs = {
        ("git", "status", "--porcelain", "--untracked-files=no"): changes,
        ("git", "rev-parse", "--abbrev-ref", "HEAD"): current_branch,
        ("git", "remote", "get-url", "origin"): origin,
    }

    def check_output(command, **kwargs):
        if output_error is not None and command[1] == output_error_command:
            raise output_error
        if kwargs != {"cwd": path, "text": True}:
            raise AssertionError(f"Unexpected read options: {kwargs}")
        return outputs[tuple(command)]

    def check_call(command, **kwargs):
        if call_error is not None and command[1] == call_error_command:
            raise call_error

    subprocess = Mock()
    subprocess.check_output.side_effect = check_output
    subprocess.check_call.side_effect = check_call
    printed = Mock()
    namespace = {
        "os": types.SimpleNamespace(environ={"SCP_AUTO_UPDATE": enabled}),
        "subprocess": subprocess,
        "repo_root": _MockRepoPath(path, git_dir),
        "IN_COLAB": in_colab,
        "SCP_REPO_URL": configured,
        "SCP_REPO_BRANCH": configured_branch,
        "print": printed,
    }
    exec(_compile_nodes(_update_nodes(name), name), namespace)
    return subprocess, printed


class RepositoryIdentityTests(unittest.TestCase):
    def test_all_eight_embedded_helpers_and_update_guards_stay_in_sync(self) -> None:
        expected = ast.dump(ast.Module(body=_update_nodes(UPDATE_NOTEBOOKS[0]),
                                       type_ignores=[]), include_attributes=False)
        for name in UPDATE_NOTEBOOKS:
            with self.subTest(notebook=name):
                actual = ast.dump(ast.Module(body=_update_nodes(name), type_ignores=[]),
                                  include_attributes=False)
                self.assertEqual(actual, expected)

    def test_old_and_new_github_owners_are_aliases_in_both_directions(self) -> None:
        variants = (
            "https://github.com/{owner}/SCP.git",
            "http://github.com/{owner}/scp/",
            " HTTPS://GITHUB.COM/{owner}/SCP.GIT/ ",
            "git@github.com:{owner}/SCP.git",
            "git@GITHUB.COM:{owner}/SCP.GIT/",
            "ssh://git@github.com/{owner}/SCP.git/",
            "SSH://git@GITHUB.COM/{owner}/scp/",
        )
        for name in UPDATE_NOTEBOOKS:
            helpers = _helpers(name)
            matches = helpers["_scp_repo_matches"]
            for old_owner, new_owner in (("HunterBushnell", "cyneuro"),
                                         ("cyneuro", "HunterBushnell")):
                for template in variants:
                    origin = template.format(owner=old_owner)
                    configured = f"https://github.com/{new_owner}/SCP.git"
                    with self.subTest(notebook=name, origin=origin, configured=configured):
                        self.assertTrue(matches(origin, configured))
            self.assertNotEqual(
                helpers["_scp_repo_slug"]("https://github.com/HunterBushnell/SCP"),
                helpers["_scp_repo_slug"](CANONICAL_REPO),
                "Owner aliases belong in the explicit matcher, not general URL identity",
            )

    def test_custom_forks_only_match_the_explicitly_configured_repository(self) -> None:
        for name in UPDATE_NOTEBOOKS:
            matches = _helpers(name)["_scp_repo_matches"]
            for origin, configured, expected in (
                ("git@github.com:example/SCP.git", "https://github.com/EXAMPLE/scp/", True),
                ("https://github.com/example/SCP", CANONICAL_REPO, False),
                (CANONICAL_REPO, "https://github.com/example/SCP", False),
                ("https://github.com/HunterBushnell/another", CANONICAL_REPO, False),
                ("https://github.com/cyneuro/SCP-extra", CANONICAL_REPO, False),
                ("https://git.example.org/team/SCP.git", "https://git.example.org/team/SCP.git", True),
                ("/local/checkouts/SCP", "/local/checkouts/SCP", True),
            ):
                with self.subTest(notebook=name, origin=origin, configured=configured):
                    self.assertEqual(matches(origin, configured), expected)

    def test_deceptive_or_non_github_origins_cannot_impersonate_github(self) -> None:
        invalid_origins = (
            "https://github.com.evil.example/cyneuro/SCP.git",
            "https://evil.example/github.com/cyneuro/SCP.git",
            "https://evil.example/path/github.com/HunterBushnell/SCP.git",
            "https://github.com@evil.example/cyneuro/SCP.git",
            "git@evil.example:github.com/cyneuro/SCP.git",
            "ssh://git@github.com.evil.example/cyneuro/SCP.git",
            "https://github.com:8443/cyneuro/SCP.git",
            "https://github.com:not-a-port/cyneuro/SCP.git",
            "https://github.com/cyneuro/SCP.git?next=another",
            "https://github.com/cyneuro/SCP.git#another",
            "https://github.com/path/cyneuro/SCP.git",
            "https://github.com//cyneuro/SCP.git",
            "https://github.com/cyneuro/SCP.git//",
            "https://github.com/cyneuro//SCP.git",
            "ftp://github.com/cyneuro/SCP.git",
            "github.com/cyneuro/SCP",
            "cyneuro/scp",
            "hunterbushnell/scp",
        )
        for name in UPDATE_NOTEBOOKS:
            matches = _helpers(name)["_scp_repo_matches"]
            for origin in invalid_origins:
                with self.subTest(notebook=name, origin=origin):
                    self.assertFalse(matches(origin, CANONICAL_REPO))
                    self.assertFalse(matches(CANONICAL_REPO, origin))


class RepositoryUpdateGuardTests(unittest.TestCase):
    def _assert_updated(self, subprocess: Mock, branch: str = "main") -> None:
        self.assertEqual(
            [call.args[0] for call in subprocess.check_call.call_args_list],
            [["git", "fetch", "--depth", "1", "origin", branch],
             ["git", "reset", "--hard", f"origin/{branch}"]],
        )
        for call in subprocess.check_call.call_args_list:
            self.assertEqual(call.kwargs, {"cwd": "/content/SCP"})

    def test_aliases_refresh_clean_ephemeral_colab_clones_in_both_directions(self) -> None:
        for name in UPDATE_NOTEBOOKS:
            for origin, configured in (
                ("git@github.com:HunterBushnell/SCP.git", CANONICAL_REPO + ".git"),
                (CANONICAL_REPO + ".git", "https://github.com/HunterBushnell/SCP.git"),
            ):
                with self.subTest(notebook=name, origin=origin):
                    subprocess, _ = _run_update(name, origin=origin, configured=configured)
                    self._assert_updated(subprocess)
                    self.assertEqual(subprocess.check_output.call_count, 3)

    def test_explicit_custom_fork_and_branch_or_detached_head_are_honored(self) -> None:
        for name in UPDATE_NOTEBOOKS:
            for branch, current in ((None, "HEAD"), ("release/test", "release/test"),
                                    ("release/test", "HEAD")):
                with self.subTest(notebook=name, branch=branch, current=current):
                    subprocess, _ = _run_update(
                        name, origin="git@github.com:example/SCP.git",
                        configured="https://github.com/example/SCP.git",
                        configured_branch=branch, current_branch=current,
                    )
                    self._assert_updated(subprocess, branch or "main")

    def test_outer_guards_prevent_even_git_inspection(self) -> None:
        cases = (
            {"in_colab": False},
            {"enabled": "0"},
            {"enabled": "false"},
            {"enabled": "False"},
            {"path": "/home/user/SCP"},
            {"path": "/content/drive/MyDrive/SCP"},
            {"git_dir": False},
        )
        for name in UPDATE_NOTEBOOKS:
            for options in cases:
                with self.subTest(notebook=name, options=options):
                    subprocess, _ = _run_update(name, **options)
                    subprocess.check_output.assert_not_called()
                    subprocess.check_call.assert_not_called()

    def test_inner_guards_never_fetch_or_reset(self) -> None:
        cases = (
            {"changes": " M README.md\n"},
            {"origin": "https://github.com/example/SCP.git"},
            {"origin": "https://github.com.evil.example/cyneuro/SCP.git"},
            {"origin": "https://evil.example/github.com/cyneuro/SCP.git"},
            {"current_branch": "feature/local-work"},
            {"configured_branch": "release/test", "current_branch": "main"},
        )
        for name in UPDATE_NOTEBOOKS:
            for options in cases:
                with self.subTest(notebook=name, options=options):
                    subprocess, printed = _run_update(name, **options)
                    subprocess.check_call.assert_not_called()
                    self.assertTrue(printed.called, "Skipped updates should explain why")

    def test_git_inspection_and_fetch_exceptions_continue_with_local_copy(self) -> None:
        for name in UPDATE_NOTEBOOKS:
            for command in ("status", "rev-parse", "remote"):
                with self.subTest(notebook=name, failure=command):
                    subprocess, printed = _run_update(
                        name, output_error=RuntimeError("offline"),
                        output_error_command=command,
                    )
                    subprocess.check_call.assert_not_called()
                    self.assertIn("continuing with local copy", printed.call_args.args[0])
            for command, call_count in (("fetch", 1), ("reset", 2)):
                with self.subTest(notebook=name, failure=command):
                    subprocess, printed = _run_update(
                        name, call_error=RuntimeError("offline"), call_error_command=command,
                    )
                    self.assertEqual(subprocess.check_call.call_count, call_count)
                    self.assertEqual(subprocess.check_call.call_args.args[0][1], command)
                    self.assertIn("continuing with local copy", printed.call_args.args[0])


class TransferMetadataTests(unittest.TestCase):
    def test_all_public_notebooks_default_to_canonical_clone_and_colab_links(self) -> None:
        for name in PUBLIC_NOTEBOOKS:
            with self.subTest(notebook=name):
                assignments = [
                    node.value for node in ast.walk(_bootstrap_tree(name))
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "SCP_REPO_URL"
                            for target in node.targets)
                ]
                self.assertEqual(len(assignments), 1)
                assignment = assignments[0]
                self.assertIsInstance(assignment, ast.Call)
                self.assertEqual(ast.unparse(assignment.func), "os.environ.get")
                self.assertEqual([ast.literal_eval(arg) for arg in assignment.args],
                                 ["SCP_REPO_URL", CANONICAL_REPO + ".git"])
                markdown = "\n".join(
                    "".join(cell.get("source", [])) for cell in _notebook(name)["cells"]
                    if cell["cell_type"] == "markdown"
                )
                self.assertIn(
                    f"https://colab.research.google.com/github/cyneuro/SCP/blob/main/{name}",
                    markdown,
                )
                self.assertNotIn("github.com/HunterBushnell/SCP", markdown)

    def test_compact_notebook_keeps_import_first_local_behavior_and_no_auto_update(self) -> None:
        tree = _bootstrap_tree("0_pipeline.ipynb")
        self.assertFalse(any(isinstance(node, ast.FunctionDef) and node.name in HELPER_NAMES
                             for node in ast.walk(tree)))
        self.assertFalse(any(isinstance(node, ast.Constant) and node.value == "SCP_AUTO_UPDATE"
                             for node in ast.walk(tree)))
        import_try = next(
            node for node in tree.body
            if isinstance(node, ast.Try) and isinstance(node.body[0], ast.Import)
            and any(alias.name == "scp_pipeline" for alias in node.body[0].names)
        )
        for missing in (None, "scp_pipeline", "neuron"):
            with self.subTest(missing=missing):
                subprocess = Mock()

                def import_package(name, *args, **kwargs):
                    self.assertEqual(name, "scp_pipeline")
                    if missing:
                        raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)
                    return types.SimpleNamespace()

                namespace = {
                    "__builtins__": dict(vars(builtins), __import__=import_package),
                    "IN_COLAB": False,
                    "subprocess": subprocess,
                }
                code = _compile_nodes([import_try], "compact-local-import")
                if missing:
                    with self.assertRaises(ImportError):
                        exec(code, namespace)
                else:
                    exec(code, namespace)
                subprocess.check_call.assert_not_called()
                subprocess.check_output.assert_not_called()

    def test_readme_identifies_lab_contact_and_original_creator(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        text = " ".join(readme.split())
        self.assertIn(LAB, text)
        self.assertIn(CONTACT_EMAIL, readme)
        self.assertIn(CANONICAL_REPO, readme)
        self.assertRegex(text, re.compile(
            r"(?:creat(?:ed|or)|original).{0,120}Hunter Bushnell"
            r"|Hunter Bushnell.{0,120}(?:creat(?:ed|or)|original)", re.IGNORECASE))

    def test_citation_preserves_author_and_affiliation_and_adds_contact(self) -> None:
        citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertIn(f'repository-code: "{CANONICAL_REPO}"', citation)
        authors = re.search(r"(?ms)^authors:\s*\n(.*?)(?=^\S|\Z)", citation)
        self.assertIsNotNone(authors)
        author_text = authors.group(1)
        self.assertEqual(author_text.count("family-names:"), 1)
        self.assertIn("family-names: Bushnell", author_text)
        self.assertIn("given-names: Hunter", author_text)
        self.assertIn(f'affiliation: "{LAB}"', author_text)
        contact = re.search(r"(?ms)^contact:\s*\n(.*?)(?=^\S|\Z)", citation)
        self.assertIsNotNone(contact)
        self.assertIn(CONTACT_EMAIL, contact.group(1))

    def test_packaging_uses_lab_maintainer_and_canonical_project_urls(self) -> None:
        tree = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))
        setup_call = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "setup"
        )
        keywords = {keyword.arg: keyword.value for keyword in setup_call.keywords}
        for key, expected in {
            "url": CANONICAL_REPO,
            "maintainer": LAB,
            "maintainer_email": CONTACT_EMAIL,
            "project_urls": {
                "Source": CANONICAL_REPO,
                "Documentation": CANONICAL_REPO + "/blob/main/docs/README.md",
                "Issues": CANONICAL_REPO + "/issues",
            },
        }.items():
            with self.subTest(field=key):
                self.assertIn(key, keywords)
                self.assertEqual(ast.literal_eval(keywords[key]), expected)
        self.assertNotIn("author", keywords, "Maintainer changes must not reassign authorship")
        self.assertNotIn("author_email", keywords)

    def test_license_and_third_party_notices_remain_byte_for_byte_unchanged(self) -> None:
        for relative, expected in PRESERVED_CREDIT_HASHES.items():
            with self.subTest(file=relative):
                self.assertEqual(hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest(),
                                 expected)


if __name__ == "__main__":
    unittest.main()
