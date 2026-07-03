"""Lightweight Step 1 validation checks."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import os

from modules.input_generation import inputs
from modules.model.load_cell import load_cell

from .mechanisms import find_compiled_mechanism_dll
from .paths import resolve_step1_paths


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
) -> Dict[str, Any]:
    """Run lightweight validation checks for Step-1 output layout."""
    tune_dir = Path(tune_dir).expanduser().resolve()
    paths = resolve_step1_paths(tune_dir)

    checks: Dict[str, Any] = {
        "manifest_exists": paths.manifest.is_file(),
    }
    if not checks["manifest_exists"]:
        raise FileNotFoundError(f"Missing manifest.json at {paths.manifest}")

    if validate_modfiles:
        dll = find_compiled_mechanism_dll(tune_dir)
        checks["compiled_dll"] = str(dll) if dll else None
        if dll is None:
            raise FileNotFoundError(
                "Compiled mechanisms not found. Run compile_modfiles first."
            )

    if validate_load_cell:
        with _pushd(tune_dir):
            cell = load_cell(
                {
                    "cell_name": cell_name,
                    "cell_loader": "allen_manifest",
                    "paths": {"manifest": "manifest.json"},
                    "tuning": {"soma_diam_multiplier": float(soma_diam_multiplier)},
                }
            )
        checks["load_cell"] = {
            "ok": True,
            "Vinit": cell.Vinit,
        }

    if validate_inputs:
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

    return checks
