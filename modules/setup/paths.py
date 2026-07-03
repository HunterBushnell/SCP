"""Canonical path resolution for Step 1 tune preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class Step1Paths:
    tune_dir: Path
    config_dir: Path
    syn_groups_dir: Path
    manifest: Path
    modfiles_dir: Path


def resolve_step1_paths(tune_dir: Path) -> Step1Paths:
    tune_dir = Path(tune_dir).expanduser().resolve()
    return Step1Paths(
        tune_dir=tune_dir,
        config_dir=tune_dir / "cell_configs",
        syn_groups_dir=tune_dir / "cell_configs" / "syn_groups",
        manifest=tune_dir / "manifest.json",
        modfiles_dir=tune_dir / "modfiles",
    )
