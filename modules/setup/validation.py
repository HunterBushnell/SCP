"""Lightweight Step 1 validation checks."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import os

from modules.loaders import get_cell_loader_name, loader_requires_manifest

from .mechanisms import find_compiled_mechanism_dll
from .paths import resolve_step1_paths
from .json_utils import _read_json


@contextmanager
def _pushd(path: Path):
    """
    Temporarily change the process working directory.

    AllenSDK manifest loading resolves some resources relative to cwd, so Step-1
    validation uses this context for load_cell smoke tests.
    """
    old = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(old))


def validate_tune(
    *,
    tune_dir: Path,
    cell_name: str,
    soma_diam_multiplier: float,
    validate_modfiles: bool = True,
    validate_load_cell: bool = True,
    validate_inputs: bool = True,
    validate_synapses: bool = True,
) -> Dict[str, Any]:
    """Run lightweight validation checks for Step-1 output layout."""
    tune_dir = Path(tune_dir).expanduser().resolve()
    paths = resolve_step1_paths(tune_dir)

    checks: Dict[str, Any] = {
        "files": {
            "tune_dir": tune_dir.is_dir(),
            "cell_config": paths.cell_config.is_file(),
            "sim_config": paths.sim_config.is_file(),
            "geometry": paths.geometry_config.is_file(),
            "syn_config": paths.syn_config.is_file(),
        },
    }

    if not paths.cell_config.is_file() and validate_load_cell:
        raise FileNotFoundError(f"Missing cell_config.json at {paths.cell_config}")

    cell_config = _read_json(paths.cell_config) if paths.cell_config.is_file() else {}
    cell_config.setdefault("cell_name", cell_name)
    tuning = cell_config.setdefault("tuning", {})
    if not isinstance(tuning, dict) or "soma_diam_multiplier" not in tuning:
        raise KeyError(
            "cell_config.json must define tuning.soma_diam_multiplier; "
            "this value is no longer read from sim_config.json."
        )
    tuning["soma_diam_multiplier"] = float(tuning["soma_diam_multiplier"])

    loader_name = get_cell_loader_name(cell_config)
    checks["cell_loader"] = loader_name
    if loader_requires_manifest(loader_name):
        manifest_path = Path(str(cell_config.get("paths", {}).get("manifest", "manifest.json")))
        if not manifest_path.is_absolute():
            manifest_path = tune_dir / manifest_path
        checks["files"]["manifest"] = manifest_path.is_file()
        if not checks["files"]["manifest"]:
            raise FileNotFoundError(f"Missing manifest.json at {manifest_path}")

    if validate_modfiles:
        dll = find_compiled_mechanism_dll(tune_dir)
        checks["compiled_dll"] = str(dll) if dll else None
        if dll is None:
            raise FileNotFoundError(
                "Compiled mechanisms not found. Run compile_modfiles first."
            )

    if validate_load_cell:
        from modules.model.load_cell import load_cell

        with _pushd(tune_dir):
            cell = load_cell(cell_config)
        checks["load_cell"] = {
            "ok": True,
            "Vinit": getattr(cell, "Vinit", None),
        }

    if validate_inputs and validate_synapses:
        from modules.input_generation import inputs

        if not paths.syn_config.is_file():
            raise FileNotFoundError(f"Missing syn_config.json at {paths.syn_config}")
        sim_cfg, groups_cfg = inputs.check_inputs(path=tune_dir, verbose=False)
        checks["inputs_check"] = {
            "ok": True,
            "n_groups": int(len(groups_cfg)),
            "n_active_groups": int(
                sum(
                    1
                    for gcfg in groups_cfg.values()
                    if bool(gcfg.get("state", True)) and bool(gcfg.get("mode"))
                )
            ),
            "tstop": float(sim_cfg.get("tstop", 0.0)),
            "dt": float(sim_cfg.get("dt", 0.0)),
        }
    elif validate_inputs:
        checks["inputs_check"] = {"status": "skipped", "reason": "synapse configs disabled"}

    return checks
