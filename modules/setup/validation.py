"""Lightweight Step 1 validation checks."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import math
import os

from modules import loaders as cell_loaders

from .mechanisms import find_compiled_mechanism_dll, resolve_modfiles_dir
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
    soma_diam_multiplier: Optional[float] = None,
    validate_modfiles: bool = True,
    validate_load_cell: bool = True,
    validate_inputs: bool = True,
    validate_synapses: bool = True,
    allow_missing_modfiles: bool = False,
) -> Dict[str, Any]:
    """Run lightweight validation checks for Step-1 output layout."""
    # Retained as an optional argument for callers using the pre-registry API.
    # Loader validation reads any Allen-only multiplier from cell_config itself.
    _ = soma_diam_multiplier
    tune_dir = Path(tune_dir).expanduser().resolve()
    paths = resolve_step1_paths(tune_dir)

    checks: Dict[str, Any] = {
        "files": {
            "tune_dir": tune_dir.is_dir(),
            "cell_config": paths.cell_config.is_file(),
            "sim_config": paths.sim_config.is_file(),
            "target_config": paths.target_config.is_file(),
            "geometry": paths.geometry_config.is_file(),
            "syn_config": paths.syn_config.is_file(),
        },
    }

    if not paths.cell_config.is_file() and validate_load_cell:
        raise FileNotFoundError(f"Missing cell_config.json at {paths.cell_config}")

    cell_config = _read_json(paths.cell_config) if paths.cell_config.is_file() else {}
    cell_config.setdefault("cell_name", cell_name)
    tuning = cell_config.get("tuning")
    if tuning is not None and not isinstance(tuning, dict):
        raise TypeError("cell_config.json tuning must be an object when provided.")
    if isinstance(tuning, dict) and "soma_diam_multiplier" in tuning:
        tuning["soma_diam_multiplier"] = float(tuning["soma_diam_multiplier"])

    loader_name = cell_loaders.get_cell_loader_name(cell_config)
    checks["cell_loader"] = loader_name
    if loader_name == "hoc_template":
        if not paths.sim_config.is_file():
            raise FileNotFoundError(
                "hoc_template setup requires cell_configs/sim_config.json with explicit "
                "conditions.v_init_mV and conditions.celsius_C values."
            )
        sim_config = _read_json(paths.sim_config)
        conditions = sim_config.get("conditions")
        if not isinstance(conditions, dict):
            raise KeyError(
                "hoc_template sim_config.json must define a conditions object with "
                "numeric v_init_mV and celsius_C values."
            )
        from modules.simulation.current_injection import validate_required_sim_conditions

        validate_required_sim_conditions(cell_config, sim_config)
        checks["conditions"] = {
            "v_init_mV": float(conditions["v_init_mV"]),
            "celsius_C": float(conditions["celsius_C"]),
        }
    loader_validator = getattr(cell_loaders, "validate_cell_loader_config", None)
    if callable(loader_validator):
        resolved_loader_paths = loader_validator(cell_config, base_dir=tune_dir)
        checks["loader_paths"] = {
            str(key): str(value) for key, value in resolved_loader_paths.items()
        }
    elif cell_loaders.loader_requires_manifest(loader_name):
        manifest_path = Path(str(cell_config.get("paths", {}).get("manifest", "manifest.json")))
        if not manifest_path.is_absolute():
            manifest_path = tune_dir / manifest_path
        checks["files"]["manifest"] = manifest_path.is_file()
        if not checks["files"]["manifest"]:
            raise FileNotFoundError(f"Missing manifest.json at {manifest_path}")

    if validate_modfiles:
        dll = find_compiled_mechanism_dll(tune_dir, cell_config=cell_config)
        checks["compiled_dll"] = str(dll) if dll else None
        if dll is None:
            mod_dir = resolve_modfiles_dir(tune_dir, cell_config)
            has_mod_sources = bool(
                mod_dir is not None and mod_dir.is_dir() and any(mod_dir.glob("*.mod"))
            )
            if allow_missing_modfiles and not has_mod_sources:
                checks["mechanisms"] = {
                    "status": "skipped",
                    "reason": "model has no configured .mod sources",
                }
            else:
                raise FileNotFoundError(
                    "Compiled mechanisms not found. Run compile_modfiles first."
                )

    if validate_load_cell:
        from modules.model.load_cell import load_cell

        with _pushd(tune_dir):
            cell = load_cell(cell_config, base_dir=tune_dir)
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
