"""Canonical path resolution for Step 1 tune preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Step1Paths:
    tune_dir: Path
    config_dir: Path
    syn_groups_dir: Path
    cell_config: Path
    sim_config: Path
    geometry_config: Path
    syn_config: Path
    manifest: Path
    modfiles_dir: Path


def resolve_step1_paths(tune_dir: Path) -> Step1Paths:
    tune_dir = Path(tune_dir).expanduser().resolve()
    return Step1Paths(
        tune_dir=tune_dir,
        config_dir=tune_dir / "cell_configs",
        syn_groups_dir=tune_dir / "cell_configs" / "syn_groups",
        cell_config=tune_dir / "cell_configs" / "cell_config.json",
        sim_config=tune_dir / "cell_configs" / "sim_config.json",
        geometry_config=tune_dir / "cell_configs" / "geometry.json",
        syn_config=tune_dir / "cell_configs" / "syn_config.json",
        manifest=tune_dir / "manifest.json",
        modfiles_dir=tune_dir / "modfiles",
    )
