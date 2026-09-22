#!/usr/bin/env python3
"""Normalize public notebook defaults, kernels, and saved execution state."""

from __future__ import annotations

import json
from pathlib import Path


PUBLIC_NOTEBOOKS = (
    "0_pipeline.ipynb",
    "1_setup.ipynb",
    "2_passive.ipynb",
    "3_active.ipynb",
    "4_synapses.ipynb",
    "5_simulate.ipynb",
    "6_analysis.ipynb",
    "7_tools.ipynb",
    "extra_notebooks/act_segmentation.ipynb",
)

SOURCE_REPLACEMENTS = {
    "2_passive.ipynb": {
        'cell_name = "SST"': 'cell_name = "PV"',
    },
    "6_analysis.ipynb": {
        'cell_name = "SST"': 'cell_name = "PV"',
    },
}


def _source_text(source: str | list[str]) -> str:
    return "".join(source) if isinstance(source, list) else source


def _restore_source_type(original: str | list[str], updated: str) -> str | list[str]:
    return updated.splitlines(keepends=True) if isinstance(original, list) else updated


def normalize_notebook(path: Path, replacements: dict[str, str]) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    original = json.dumps(notebook, ensure_ascii=False, sort_keys=True)

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        cell["execution_count"] = None
        cell["outputs"] = []
        source = cell.get("source", [])
        text = _source_text(source)
        for old, new in replacements.items():
            text = text.replace(old, new)
        cell["source"] = _restore_source_type(source, text)

    metadata = notebook.setdefault("metadata", {})
    metadata["kernelspec"] = {
        "display_name": "scp-py311",
        "language": "python",
        "name": "scp-py311",
    }

    changed = json.dumps(notebook, ensure_ascii=False, sort_keys=True) != original
    if changed:
        path.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return changed


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    for relative in PUBLIC_NOTEBOOKS:
        changed = normalize_notebook(
            repo_root / relative,
            SOURCE_REPLACEMENTS.get(relative, {}),
        )
        print(f"{'updated' if changed else 'clean':7} {relative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
