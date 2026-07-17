from __future__ import annotations

import hashlib
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from neuron import h

from .result_paths import _resolve_tune_path, _sha256_file


def _snapshot_cfg(sim_cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    snap = sim_cfg.get("snapshot", None)
    if isinstance(snap, dict):
        return bool(snap.get("enabled", False)), snap
    if snap is True:
        return True, {"enabled": True}
    if isinstance(snap, str) and snap.strip().lower() in ("true", "1", "yes", "on"):
        return True, {"enabled": True}
    return False, {}


def _apply_snapshot_deterministic(sim_cfg: Dict[str, Any], snapshot_cfg: Dict[str, Any]) -> None:
    """
    Best-effort deterministic settings for snapshot comparisons.
    Only applies if snapshot_cfg.force_deterministic is True (default).
    """
    if not snapshot_cfg.get("force_deterministic", True):
        snapshot_cfg["deterministic_applied"] = False
        return

    seed = snapshot_cfg.get("deterministic_seed")
    if seed is None:
        seed = sim_cfg.get("random_seed", sim_cfg.get("seed", 0))
    try:
        seed = int(seed)
    except Exception:
        seed = 0

    try:
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass

    try:
        if hasattr(h, "cvode"):
            h.cvode.active(0)
    except Exception:
        pass
    try:
        if hasattr(h, "nthread"):
            h.nthread(1)
    except Exception:
        pass
    try:
        if hasattr(h, "Random123_globalindex"):
            h.Random123_globalindex(seed)
    except Exception:
        pass

    snapshot_cfg["deterministic_applied"] = True
    snapshot_cfg["deterministic_seed"] = seed


def _collect_versions() -> Dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "python_exe": sys.executable,
        "numpy": np.__version__,
    }
    try:
        versions["neuron"] = str(getattr(h, "nrnversion", lambda: "unknown")())
    except Exception:
        versions["neuron"] = "unknown"
    return versions


def _collect_env_snapshot() -> Dict[str, Any]:
    keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NEURONHOME",
        "NRNHOME",
        "NRN_NMODL_PATH",
        "NRNMECH_DLL",
    )
    snap: Dict[str, Any] = {}
    for key in keys:
        val = os.environ.get(key)
        if val not in (None, ""):
            snap[key] = val
    return snap


def _collect_neuron_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    for key in ("dt", "tstop", "t", "celsius", "secondorder", "v_init", "steps_per_ms"):
        try:
            state[key] = float(getattr(h, key))
        except Exception:
            pass
    try:
        if hasattr(h, "cvode"):
            cvode = h.cvode
            state["cvode_active"] = int(cvode.active())
            for name in ("atol", "rtol", "minstep", "maxstep"):
                try:
                    state[f"cvode_{name}"] = float(getattr(cvode, name)())
                except Exception:
                    pass
    except Exception:
        pass
    try:
        if hasattr(h, "secondorder"):
            state["secondorder"] = int(h.secondorder)
    except Exception:
        pass
    try:
        if hasattr(h, "nthread"):
            state["nthread"] = int(h.nthread())
    except Exception:
        pass
    return state


def _collect_mechanism_info(sim_cfg: Dict[str, Any]) -> Dict[str, Any]:
    from modules.setup.mechanisms import (
        find_compiled_mechanism_dll,
        resolve_modfiles_dir,
    )

    info: Dict[str, Any] = {}
    tune_path = _resolve_tune_path(sim_cfg)
    if tune_path is None:
        return info

    info["tune_dir"] = str(tune_path)
    mod_dir = resolve_modfiles_dir(tune_path)
    info["modfiles_dir"] = None if mod_dir is None else str(mod_dir)
    if mod_dir is None:
        return info
    mod_files = sorted(p for p in mod_dir.glob("*.mod")) if mod_dir.is_dir() else []
    if mod_files:
        info["modfiles_count"] = len(mod_files)
        info["modfiles"] = [p.name for p in mod_files]
        hsh = hashlib.sha256()
        for p in mod_files:
            try:
                hsh.update(p.name.encode("ascii", errors="ignore"))
                hsh.update(_sha256_file(p).encode("ascii"))
            except Exception:
                continue
        info["modfiles_sha256"] = hsh.hexdigest()

    dll = find_compiled_mechanism_dll(tune_path)
    if dll is not None:
        info["dll_path"] = str(dll)
        try:
            stat = dll.stat()
            info["dll_size"] = int(stat.st_size)
            info["dll_mtime"] = float(stat.st_mtime)
        except Exception:
            pass
        try:
            info["dll_sha256"] = _sha256_file(dll)
        except Exception:
            pass

    return info


def _snapshot_netcon_state(syn_state: Dict[str, Any]) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    netcons = syn_state.get("netcons", {}) or {}
    for group, ncs in netcons.items():
        weights: List[Optional[float]] = []
        delays: List[Optional[float]] = []
        for nc in ncs or []:
            try:
                weights.append(float(nc.weight[0]))
            except Exception:
                weights.append(None)
            try:
                delays.append(float(nc.delay))
            except Exception:
                delays.append(None)
        snapshot[group] = {"n": len(ncs), "weights": weights, "delays": delays}
    return snapshot


def _snapshot_synapse_params(
    syn_state: Dict[str, Any],
    groups_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    syn_by_group = syn_state.get("synapses", {}) or {}
    for group, syn_list in syn_by_group.items():
        if not syn_list:
            continue
        syn = syn_list[0]
        gcfg = groups_cfg.get(group, {}) or {}
        syn_cfg = gcfg.get("syns", {}) or {}
        params_cfg = syn_cfg.get("params", {}) or {}
        present = {}
        missing = []
        for key in params_cfg:
            if hasattr(syn, key):
                try:
                    present[key] = float(getattr(syn, key))
                except Exception:
                    present[key] = getattr(syn, key)
            else:
                missing.append(key)
        snapshot[group] = {
            "type": syn_cfg.get("type"),
            "params_present": present,
            "params_missing": missing,
        }
    return snapshot
