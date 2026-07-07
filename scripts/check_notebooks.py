#!/usr/bin/env python3
"""Lightweight notebook lint checks for portability and config safety.

Checks:
- Code-cell Python syntax (`ast.parse`)
- Duplicate literal keys in dict literals (silent Python override risk)
- Hardcoded user-specific absolute paths in source cells

Usage:
  python scripts/check_notebooks.py
  python scripts/check_notebooks.py --notebooks 3_active.ipynb 4_synapses.ipynb
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_NOTEBOOKS = [
    "1_download.ipynb",
    "colab_notebooks/2_colab.ipynb",
    "2_passive.ipynb",
    "colab_notebooks/3_colab.ipynb",
    "3_active.ipynb",
    "4_synapses.ipynb",
    "5_simulate.ipynb",
    "6_analysis.ipynb",
]

FORBIDDEN_SOURCE_PATTERNS = [
    "/home/",
    "/Users/",
    "C:\\Users\\",
]


def _is_repo_root(path: Path) -> bool:
    return (path / "run_pipeline.py").is_file() and (path / "modules").is_dir()


def _find_repo_root(start: Path) -> Path:
    env_root = os.environ.get("SCP_ROOT")
    if env_root:
        cand = Path(env_root).expanduser().resolve()
        if _is_repo_root(cand):
            return cand
        raise FileNotFoundError(
            f"SCP_ROOT is set but is not an SCP repo root: {cand} "
            "(expected run_pipeline.py and modules/)"
        )

    start = start.resolve()
    for cand in (start, *start.parents):
        if _is_repo_root(cand):
            return cand

    for base in (start, start.parent):
        try:
            for child in base.iterdir():
                if child.is_dir() and _is_repo_root(child):
                    return child.resolve()
        except Exception:
            pass

    raise FileNotFoundError(
        f"Could not locate SCP repo root from {start}. "
        "Run from inside the repo or set SCP_ROOT=/path/to/SCP."
    )


def _literal_key(node: ast.AST) -> Optional[object]:
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (str, int, float, bool, type(None))
    ):
        return node.value

    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        val = node.operand.value
        return -val if isinstance(node.op, ast.USub) else +val

    return None


def _find_duplicate_literal_keys(tree: ast.AST) -> list[tuple[object, int, int]]:
    duplicates: list[tuple[object, int, int]] = []

    class _Visitor(ast.NodeVisitor):
        def visit_Dict(self, node: ast.Dict) -> None:  # type: ignore[override]
            seen: dict[object, tuple[int, int]] = {}
            for key_node in node.keys:
                if key_node is None:
                    continue
                key = _literal_key(key_node)
                if key is None:
                    continue
                line = int(getattr(key_node, "lineno", 0))
                col = int(getattr(key_node, "col_offset", 0))
                if key in seen:
                    duplicates.append((key, line, col))
                else:
                    seen[key] = (line, col)
            self.generic_visit(node)

    _Visitor().visit(tree)
    return duplicates


def _iter_code_cells(nb: dict) -> Iterable[tuple[int, str]]:
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        yield idx, source


def _sanitize_ipython_source(source: str) -> str:
    lines = source.splitlines()

    first_nonempty = ""
    for line in lines:
        if line.strip():
            first_nonempty = line.lstrip()
            break

    # Cell magics (e.g., %%bash) are not Python; skip full-cell parsing.
    if first_nonempty.startswith("%%"):
        return "pass\n"

    sanitized: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        # Line magics/shell/help commands are valid in IPython but not ast.parse.
        if stripped.startswith(("!", "%", "?")):
            sanitized.append("pass")
        else:
            sanitized.append(line)

    return "\n".join(sanitized) + "\n"


def _check_notebook(path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not path.is_file():
        errors.append(f"missing notebook: {path}")
        return errors, warnings

    try:
        nb = json.loads(path.read_text())
    except Exception as exc:
        errors.append(f"{path}: invalid JSON ({exc})")
        return errors, warnings

    for cell_idx, source in _iter_code_cells(nb):
        cell_label = f"{path}:cell{cell_idx}"

        for pattern in FORBIDDEN_SOURCE_PATTERNS:
            if pattern in source:
                errors.append(
                    f"{cell_label}: hardcoded user path detected: {pattern!r}"
                )

        try:
            tree = ast.parse(source)
        except SyntaxError:
            sanitized = _sanitize_ipython_source(source)
            try:
                tree = ast.parse(sanitized)
            except SyntaxError as exc:
                errors.append(
                    f"{cell_label}: syntax error at line {exc.lineno}, col {exc.offset}: {exc.msg}"
                )
                continue

        for key, line, col in _find_duplicate_literal_keys(tree):
            errors.append(
                f"{cell_label}: duplicate dict key {key!r} at line {line}, col {col}"
            )

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebooks",
        nargs="*",
        default=DEFAULT_NOTEBOOKS,
        help="Notebook paths relative to repo root.",
    )
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd())
    print(f"Repo root: {repo_root}")

    all_errors: list[str] = []
    all_warnings: list[str] = []

    for rel in args.notebooks:
        nb_path = (repo_root / rel).resolve()
        errors, warnings = _check_notebook(nb_path)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

    if all_warnings:
        print("\nWarnings:")
        for w in all_warnings:
            print(f"  - {w}")

    if all_errors:
        print("\nErrors:")
        for e in all_errors:
            print(f"  - {e}")
        print(f"\nFAILED: {len(all_errors)} error(s), {len(all_warnings)} warning(s).")
        return 1

    print(
        f"Notebook checks passed: {len(args.notebooks)} notebook(s), "
        f"{len(all_warnings)} warning(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
