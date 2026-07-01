"""
Shared helpers for SCP teaching notebooks.

These helpers keep notebook setup cells concise while preserving the same
runtime behavior used in local and Colab workflows.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

from modules.load_cell import load_cell


def is_scp_repo_root(path: Path) -> bool:
    """Return True when `path` looks like the SCP repository root."""
    return (path / "modules").is_dir() and (path / "run_pipeline.py").is_file()


def find_scp_repo_root(start: Path | None = None) -> Path:
    """
    Locate the SCP repository root from current context.

    Resolution order:
    1. `SCP_ROOT` environment variable (if valid)
    2. Current working directory and its parents
    3. Direct child directories of cwd and cwd.parent (for parent-launched kernels)
    """
    env_root = os.environ.get("SCP_ROOT")
    if env_root:
        env_path = Path(env_root).expanduser().resolve()
        if is_scp_repo_root(env_path):
            return env_path
        raise FileNotFoundError(
            f"SCP_ROOT does not point to an SCP repo root: {env_path}. "
            "Expected modules/ and run_pipeline.py."
        )

    start_path = (start or Path.cwd()).resolve()
    for cand in (start_path, *start_path.parents):
        if is_scp_repo_root(cand):
            return cand

    for base in (start_path, start_path.parent):
        try:
            for child in base.iterdir():
                if child.is_dir() and is_scp_repo_root(child):
                    return child.resolve()
        except Exception:
            pass

    raise FileNotFoundError(
        f"Could not locate SCP repo root from {start_path}. "
        "Set SCP_ROOT or launch Jupyter from inside the repo."
    )


def ensure_scp_repo_on_syspath(start: Path | None = None) -> Path:
    """Find SCP repo root and prepend it to `sys.path` if missing."""
    repo_root = find_scp_repo_root(start=start)
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo_root


def resolve_external_repo(
    repo_name: str,
    marker_rel: Path,
    env_vars: Sequence[str],
    repo_root: Path,
) -> Path:
    """
    Resolve a sibling external repository used by SCP notebooks (e.g., ACT/bmtool).
    """
    candidates: list[Path] = []
    for var in env_vars:
        raw = os.environ.get(var)
        if raw:
            candidates.append(Path(raw).expanduser())

    cwd = Path.cwd()
    candidates.extend(
        [
            repo_root.parent / "mods" / repo_name,
            repo_root / "mods" / repo_name,
            Path.home() / "mods" / repo_name,
            (cwd / ".." / "mods" / repo_name).resolve(),
            (cwd / "mods" / repo_name).resolve(),
        ]
    )

    seen: set[Path] = set()
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except Exception:
            resolved = cand
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / marker_rel).is_file():
            return resolved

    raise FileNotFoundError(
        f"{repo_name} repo not found. Set {', '.join(env_vars)} "
        f"or place it at ../mods/{repo_name} relative to SCP."
    )


def ensure_external_repo_on_syspath(
    repo_name: str,
    marker_rel: Path,
    env_vars: Sequence[str],
    repo_root: Path | None = None,
    prepend: bool = False,
) -> Path:
    """
    Resolve an external repo and add it to `sys.path`.

    Set `prepend=True` to prioritize it over installed packages.
    """
    root = repo_root if repo_root is not None else find_scp_repo_root()
    external_path = resolve_external_repo(
        repo_name=repo_name,
        marker_rel=marker_rel,
        env_vars=env_vars,
        repo_root=root,
    )
    ext_str = str(external_path)
    if ext_str not in sys.path:
        if prepend:
            sys.path.insert(0, ext_str)
        else:
            sys.path.append(ext_str)
    return external_path


def resolve_cell_config_for_notebook(cell_name: str, tune_dir: Path | None = None) -> Dict[str, Any]:
    """
    Load and normalize `cell_config.json` for notebook build steps.

    `cell_configs/cell_config.json` is treated as the canonical source. The
    caller can provide `tune_dir` or rely on the current working directory.
    """
    base = Path(tune_dir) if tune_dir is not None else Path(".")
    cell_cfg_path = base / "cell_configs" / "cell_config.json"

    if cell_cfg_path.is_file():
        cell_config: Dict[str, Any] = json.loads(cell_cfg_path.read_text())
        if not isinstance(cell_config, dict):
            cell_config = {}
    else:
        cell_config = {}

    cell_config.setdefault("cell_name", cell_name)
    cell_config.setdefault("cell_loader", "allen_manifest")
    paths = cell_config.setdefault("paths", {})
    if not isinstance(paths, dict):
        paths = {}
        cell_config["paths"] = paths
    paths.setdefault("manifest", "manifest.json")

    tuning = cell_config.get("tuning")
    if not isinstance(tuning, dict) or "soma_diam_multiplier" not in tuning:
        raise KeyError(
            "cell_configs/cell_config.json must define tuning.soma_diam_multiplier. "
            "Run Step 0 scaffold or set it manually before Steps 2-5."
        )
    tuning["soma_diam_multiplier"] = float(tuning["soma_diam_multiplier"])

    return cell_config


def build_cell_for_notebook(cell_config: Dict[str, Any]):
    """
    Build a cell via `modules.load_cell` and expose common section attrs.

    The returned object is compatible with existing notebook calls that access
    `cell.soma`, `cell.dend`, `cell.apic`, and `cell.axon`.
    """
    loaded = load_cell(cell_config)

    if not hasattr(loaded, "soma"):
        loaded.soma = loaded.h.soma
    if not hasattr(loaded, "dend"):
        loaded.dend = list(loaded.h.dend) if hasattr(loaded.h, "dend") else []
    if not hasattr(loaded, "apic"):
        loaded.apic = list(loaded.h.apic) if hasattr(loaded.h, "apic") else []
    if not hasattr(loaded, "axon"):
        loaded.axon = list(loaded.h.axon) if hasattr(loaded.h, "axon") else []

    return loaded


def _ensure_synapse_test_state(cell: Any) -> None:
    """
    Ensure notebook-level synapse state containers exist on the cell.

    These lists keep NEURON objects alive and mirror the state shape expected by
    notebook experiments and run_sim helpers.
    """
    for attr in ("synapses", "netcons", "stims", "vecs", "syn_locs"):
        if not hasattr(cell, attr):
            setattr(cell, attr, [])


def build_synapse_test_cell(cell_config: Dict[str, Any]):
    """
    Build a notebook-compatible cell for manual synapse tests/tuning.

    This wraps `build_cell_for_notebook` and guarantees:
      - `cell.all` is present for bmtool section lookup,
      - synapse state containers exist (`synapses/netcons/stims/vecs/syn_locs`).
    """
    loaded = build_cell_for_notebook(cell_config)

    if not hasattr(loaded, "all"):
        loaded.all = loaded.h.SectionList()
        for sec in loaded.h.allsec():
            loaded.all.append(sec)

    _ensure_synapse_test_state(loaded)
    return loaded


def add_single_synapse_for_notebook(
    cell: Any,
    syn_loc: Any,
    syn_params: Dict[str, Any],
    spike_train: Sequence[float],
    *,
    netcon_weight: float = 1.0,
) -> Dict[str, Any]:
    """
    Attach one synapse + VecStim/NetCon pair to a built cell.

    Expected `syn_params` shape matches Step 4 notebook entries:
      {
        "spec_settings": {"level_of_detail": "<mechanism_name>", ...},
        "spec_syn_param": {... mechanism attributes ...},
      }
    """
    from neuron import h

    spec_settings = syn_params.get("spec_settings", {})
    syn_cfg = syn_params.get("spec_syn_param", {})
    mech_name = spec_settings.get("level_of_detail")
    if not mech_name:
        raise KeyError("syn_params['spec_settings']['level_of_detail'] is required")

    syn = getattr(h, mech_name)(syn_loc)
    for param, val in syn_cfg.items():
        if hasattr(syn, param):
            setattr(syn, param, val)

    spike_times = [float(t) for t in spike_train]
    vec = h.Vector(spike_times)
    stim = h.VecStim()
    stim.play(vec)
    nc = h.NetCon(stim, syn)
    nc.weight[0] = float(netcon_weight)

    _ensure_synapse_test_state(cell)
    cell.synapses.append(syn)
    cell.vecs.append(vec)
    cell.stims.append(stim)
    cell.netcons.append(nc)

    return {
        "syn": syn,
        "vec": vec,
        "stim": stim,
        "netcon": nc,
    }
