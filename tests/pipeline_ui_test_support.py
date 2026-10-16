"""Shared fixtures for compact pipeline UI tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from modules.notebooks.pipeline_ui import PipelineNotebookUI


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_repo(parent: Path) -> Path:
    root = parent / "repo"
    for cell, tunes in {"A": ("orig", "tuned"), "B": ("custom",)}.items():
        for tune in tunes:
            directory = root / "cells" / cell / "tunes" / tune
            write_test_tune(directory, cell=cell, tune=tune)
    return root


def write_test_tune(directory: Path, *, cell: str, tune: str) -> None:
    """A file-valid synthetic workspace, without constructing its model."""
    configs = directory / 'cell_configs'
    configs.mkdir(parents=True)
    (directory / 'Cell.hoc').write_text(
        'begintemplate SCPUITestCell\npublic soma\ncreate soma\n'
        'proc init() { soma { insert pas } }\nendtemplate SCPUITestCell\n')
    (configs / 'cell_config.json').write_text(json.dumps({
        'cell_name': cell, 'tune': tune, 'cell_loader': 'hoc_template',
        'paths': {'hoc_template': 'Cell.hoc', 'modfiles': None},
        'hoc_template': {'template_name': 'SCPUITestCell', 'constructor_args': []},
    }))
    (configs / 'sim_config.json').write_text(json.dumps({
        'conditions': {'v_init_mV': -65, 'celsius_C': 30},
    }))
    (configs / 'geometry.json').write_text('{}\n')


def fake_state(
    root: Path,
    *,
    cell: str = "A",
    tune: str = "tuned",
) -> SimpleNamespace:
    tune_dir = root / "cells" / cell / "tunes" / tune
    return SimpleNamespace(
        repo_root=root,
        tune_dir=tune_dir,
        context=SimpleNamespace(cell_name=cell, tune_name=tune),
        cell=object(),
    )


def write_manual_passive_targets(root: Path) -> None:
    path = root / "cells" / "A" / "tunes" / "tuned" / "cell_configs"
    path.mkdir(parents=True, exist_ok=True)
    (path / "target_config.json").write_text(
        json.dumps(
            {
                "target_source": {"mode": "manual", "description": "test"},
                "manual": {
                    "passive": {
                        "rin_MOhm": 101.5,
                        "tau_ms": 6.25,
                        "v_rest_mV": -70.5,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def all_descendants(widget):
    yield widget
    for child in getattr(widget, "children", ()):
        yield from all_descendants(child)


__all__ = [
    "PipelineNotebookUI",
    "REPO_ROOT",
    "all_descendants",
    "fake_state",
    "make_repo",
    "write_manual_passive_targets",
    "write_test_tune",
]
