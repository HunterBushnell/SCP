"""Public entry points for Step 5 synaptic input generation.

The implementation is split across focused modules:
- config.py: JSON discovery and normalization
- timing.py: concrete time-window resolution
- density.py: distance-density synapse counts
- processing.py: per-group orchestration and mode execution
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import json

import numpy as np

from modules.core import randomness

from .config import (
    _expand_group_includes,
    _inject_path_metadata,
    _normalize_group_configs,
    _normalize_sim_config,
    _resolve_config_root,
)
from .density import _compile_density_from_spec, _resolve_n_syn
from .processing import (
    _build_default_mode_registry,
    _finalize_inputs,
    _init_rng,
    _make_group_signature,
    _process_all_groups,
)
from .types import GroupInputs


def generate_inputs(
    path: Optional[Union[Path, str]] = None,
    geometry: Optional[Any] = None,
    mode_registry: Optional[Mapping[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
    seed_override: Optional[int] = None,
    trial_rng: Optional[randomness.TrialRandomness] = None,
    sim_cfg_override: Optional[Dict[str, Any]] = None,
    # cache: Optional[Dict[str, "GroupInputs"]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, "GroupInputs"]]:
    """
    Step 5.2.3 main entry point: load, normalize, and materialize synaptic inputs.

    This function assumes the split-config layout:
      - sim_config.json: simulation-level settings (dt, tstart, tstop, seed, etc.)
      - syn_config.json: per-group input/synapse configs (one top-level key per group)
    Configs may live in the tune directory or in tune_dir/cell_configs/.

    Path resolution:
      - If `path` is None:
          Use the current working directory as the tune directory.
          Prefer `cell_configs/` when it contains config files.
      - If `path` is a directory:
          Treat it as the tune directory. Prefer `cell_configs/` when present.
      - If `path` is a file:
          Treat its parent directory as the config root (no auto-redirect).
          Expect both `sim_config.json` and `syn_config.json` there.

    Parameters
    ----------
    path
        Optional directory or file path controlling where configs are loaded from.
        See rules above. Typical notebook usage is `path=None` with the tune
        directory as the working directory.
    sim_cfg_override
        Optional simulation config dict. When provided, it replaces the contents
        of sim_config.json while syn_config.json is still loaded from disk.
    geometry
        Geometry / segment-group information from Step 5.2.2 (may be None if not used).
    rng
        Optional NumPy random number generator. If None, a new generator is created
        based on the seed in `sim_config.json`.
    mode_registry
        Optional mapping from mode names (str) to handler callables. If None,
        a default registry for core modes is constructed.
    cache
        Optional dictionary for caching `GroupInputs` objects keyed by a stable
        signature of (sim_cfg, group_cfg). If provided, per-group inputs may be
        reused instead of regenerated.

    Returns
    -------
    sim_cfg : dict
        Normalized simulation-level config.
    groups_cfg : dict[str, dict]
        Normalized per-group configs (one entry per group name).
    inputs_by_group : dict[str, GroupInputs]
        Materialized per-group input objects, one entry per active synapse group.
    """
    # ------------------------------------------------------------------
    # Resolve the tune directory and config file paths
    # ------------------------------------------------------------------
    config_root = _resolve_config_root(path)
    syn_path = config_root / "syn_config.json"
    sim_path = config_root / "sim_config.json"

    if not syn_path.is_file():
        raise FileNotFoundError(f"Missing syn_config.json in {config_root}")
    if not sim_path.is_file():
        raise FileNotFoundError(f"Missing sim_config.json in {config_root}")

    # ------------------------------------------------------------------
    # Load raw JSON configs (or use override)
    # ------------------------------------------------------------------
    if sim_cfg_override is None:
        with sim_path.open("r") as f:
            sim_cfg_raw = json.load(f)
    else:
        sim_cfg_raw = dict(sim_cfg_override)
    with syn_path.open("r") as f:
        groups_cfg_raw = json.load(f)

    # ------------------------------------------------------------------
    # 5.2.3.2 – Normalize configs
    # ------------------------------------------------------------------
    sim_cfg = _normalize_sim_config(sim_cfg_raw)
    _inject_path_metadata(sim_cfg, config_root)
    groups_cfg_expanded = _expand_group_includes(groups_cfg_raw, config_root)
    groups_cfg = _normalize_group_configs(groups_cfg_expanded)

    # ------------------------------------------------------------------
    # 5.2.3.3 – Shared resources: RNG and mode registry
    # ------------------------------------------------------------------
    if seed_override is not None:
        sim_cfg["seed"] = int(seed_override)

    base_rng = None if trial_rng is not None else _init_rng(rng, sim_cfg)
    if mode_registry is None:
        mode_registry = _build_default_mode_registry()

    # ------------------------------------------------------------------
    # 2.3.4 – Per-group processing
    # ------------------------------------------------------------------
    inputs_by_group = _process_all_groups(
        sim_cfg=sim_cfg,
        groups_cfg=groups_cfg,
        geometry=geometry,
        mode_registry=mode_registry,
        rng=base_rng,
        trial_rng=trial_rng,
        # cache=cache,
    )

    # ------------------------------------------------------------------
    # 2.3.5 – Final sanity checks
    # ------------------------------------------------------------------
    _finalize_inputs(sim_cfg, groups_cfg, inputs_by_group)

    return sim_cfg, groups_cfg, inputs_by_group


def check_inputs(
    path: Union[str, Path, None] = None,
    *,
    verbose: bool = True,
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Lightweight pre-2.3 sanity check.

    - Loads config using the same rules as generate_inputs
      (tune root or cell_configs/)
    - Runs the same normalization as generate_inputs
    - Prints a concise summary of sim + per-group configs
    If path is None: assume current working directory and look there.
    If path is a file: treat its parent as the config root (no auto-redirect).
    If path is a directory: prefer cell_configs/ when present.
    """

    config_root = _resolve_config_root(path)
    syn_path = config_root / "syn_config.json"
    sim_path = config_root / "sim_config.json"

    if not syn_path.is_file():
        raise FileNotFoundError(f"Missing syn_config.json in {config_root}")
    if not sim_path.is_file():
        raise FileNotFoundError(f"Missing sim_config.json in {config_root}")

    with sim_path.open("r") as f:
        sim_raw = json.load(f)
    with syn_path.open("r") as f:
        groups_raw = json.load(f)

    sim_cfg = _normalize_sim_config(sim_raw)
    _inject_path_metadata(sim_cfg, config_root)
    groups_cfg_expanded = _expand_group_includes(groups_raw, config_root)
    groups_cfg = _normalize_group_configs(groups_cfg_expanded)


    if verbose:
        print("=== check_inputs: synapse config summary ===")
        print("Sim cfg:", sim_cfg)

        print("\nSynapse groups:")
        for gname, gcfg in groups_cfg.items():
            state = gcfg.get("state", True)
            mode = gcfg.get("mode")
            syns = gcfg.get("syns", {}) or {}
            n_syn = syns.get("N_syn")
            blocks = gcfg.get("input_blocks", []) or []
            source_summary = _summarize_input_block_sources(blocks, gcfg.get("source", {}) or {})

            print(
                f"  - {gname:<12}  state={state!r:5}  mode={mode!r:<18}  "
                f"sources={source_summary:<30}  "
                f"N_syn={n_syn}  input_blocks={len(blocks)}"
            )

    return sim_cfg, groups_cfg


def _summarize_input_block_sources(
    blocks: list[Dict[str, Any]],
    fallback_source: Dict[str, Any],
) -> str:
    """Return a compact source-path summary for check_inputs output."""
    paths: list[str] = []
    for block in blocks:
        source = (block or {}).get("source", {}) or {}
        path = source.get("path")
        if path is not None and str(path) not in paths:
            paths.append(str(path))

    if not paths and fallback_source.get("path") is not None:
        paths.append(str(fallback_source["path"]))

    if not paths:
        return "None"
    if len(paths) == 1:
        return repr(paths[0])
    return f"{len(paths)} paths"
