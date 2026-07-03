"""
Core Step 5 simulation dispatch functions.

Run-mode implementations live in `single_run.py` and `multi_run.py`; this
module keeps mode inference, parametric placeholder, and unified dispatch.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

from .. import randomness
from .multi_run import run_multi
from .result_loading import load_results
from .result_saving import save_results
from .single_run import run_single


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _infer_mode(sim_cfg: Dict[str, Any]) -> str:
    """
    Infer mode from sim_cfg:
      - 'param' if param_study has non-empty param_vals
      - 'multi' if n_trials > 1
      - 'single' otherwise
    """
    param = sim_cfg.get("param_study") or {}
    param_vals = param.get("param_vals") or []
    if len(param_vals) > 0:
        return "param"

    n_trials = int(sim_cfg.get("n_trials", 1))
    if n_trials > 1:
        return "multi"

    return "single"


# ---------------------------------------------------------------------
# core run functions
# ---------------------------------------------------------------------

def run_param(
    cell: Any,
    geom: Any,
    sim_cfg: Dict[str, Any],
    groups_cfg: Dict[str, Any],
    inputs_by_group: Dict[str, Any],
    *,
    rm: Optional[randomness.RandomnessManager] = None,
) -> Dict[str, Any]:
    """
    Placeholder for parametric study mode.

    Intended final shape:
      {
        "mode": "param",
        "sim_cfg": { ... },
        "param_study": { ... },
        "spikes": { param_val: [np.ndarray, ...], ... },
        "traces": { ... },
        "meta": { ... }
      }

    Not implemented yet.
    """
    raise NotImplementedError("Parametric mode is not implemented yet.")


# ---------------------------------------------------------------------
# unified entrypoint
# ---------------------------------------------------------------------

def run_sim(
    cell: Any,
    geom: Any,
    sim_cfg: Dict[str, Any],
    groups_cfg: Dict[str, Any],
    inputs_by_group: Dict[str, Any],
    mode_registry: Optional[Dict[str, Any]] = None,
    trial_callback: Optional[Any] = None,
    meta_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Unified entrypoint: infers mode from sim_cfg and dispatches to the
    appropriate run_* function.

    Mode inference:
      - 'param' if param_study.param_vals is non-empty
      - 'multi' if n_trials > 1
      - 'single' otherwise
    """
    
    sim_cfg_local = copy.deepcopy(sim_cfg)

    # If sim_cfg["load"] is a filename/path, load instead of running NEURON
    load_target = sim_cfg_local.get("load")
    load_enabled = sim_cfg_local.get("load_enabled", True)
    if load_enabled and load_target:
        p = Path(load_target)
        if not p.is_absolute():
            p = Path("output_data") / p  # interpret as relative to output_data/
        result = load_results(p)
        meta = result.get("meta", {})
        meta["loaded_from"] = str(p)
        result["meta"] = meta
        return result
    

    rm = randomness.RandomnessManager(sim_cfg_local)
    mode = _infer_mode(sim_cfg_local)

    if mode == "single":
        result = run_single(cell, geom, sim_cfg_local, groups_cfg, inputs_by_group, rm=rm)
    elif mode == "multi":
        result = run_multi(
            cell,
            geom,
            sim_cfg_local,
            groups_cfg,
            inputs_by_group,
            rm=rm,
            mode_registry=mode_registry,
            trial_callback=trial_callback,
        )
    elif mode == "param":
        result = run_param(cell, geom, sim_cfg_local, groups_cfg, inputs_by_group, rm=rm)
    else:
        raise ValueError(f"run_sim: unrecognized mode '{mode}'")

    # Record randomness metadata (e.g., auto-generated base seeds)
    meta = result.setdefault("meta", {})
    meta["randomness"] = rm.meta().as_dict()
    if meta_overrides:
        for key, value in meta_overrides.items():
            meta[key] = copy.deepcopy(value)

    # auto-save if sim_cfg['output'] is set
    save_results(result)  # no-op if output is None/empty
    return result

def summarize_results(results):
    mode = results["mode"]
    print(f"mode={mode}, n_traces_to_save={results['sim_cfg'].get('n_traces_to_save')}")

    if mode == "single":
        T = results["traces"].get("T", [])
        V = results["traces"].get("V", [])
        print(f"  single: len(T)={len(T)}, len(V)={len(V)}, n_spikes={len(results['spikes'])}")
    elif mode == "multi":
        spikes = results["spikes"]
        print(f"  multi: n_trials={len(spikes)}, spike_counts={[len(s) for s in spikes]}")
        if results["traces"]:
            print(f"  multi: traces V stored for {len(results['traces']['V'])} trial(s)")
