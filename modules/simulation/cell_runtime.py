from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import numpy as np
from neuron import h

from scp_pipeline.model.geometry import cell_sections, cell_soma_segment

from .current_injection import apply_sim_conditions
from scp_pipeline.loaders.source_context import cell_execution
from .recording import _record_mechanism_current
from .synaptic_recording import prepare_synaptic_recording, finish_synaptic_recording


_SITE_SPEC_RE = re.compile(
    r"^(?P<sec>[A-Za-z_]\w*)(?:\[(?P<idx>\d+)\])?(?:\((?P<x>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\))?$"
)


def _get_soma_segment(cell):
    """
    Return the canonical soma(0.5) segment for a loaded cell.
    """
    return cell_soma_segment(cell)


def _parse_bool_like(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "on"}:
            return True
        if v in {"false", "0", "no", "off", ""}:
            return False
    return bool(value)


def _normalize_runtime_recording_site(site_raw: Any) -> Dict[str, Any]:
    if isinstance(site_raw, str):
        m = _SITE_SPEC_RE.match(site_raw.strip())
        if not m:
            raise ValueError(
                f"run_cell: invalid site spec {site_raw!r}; expected 'sec', 'sec[idx]' or 'sec[idx](x)'"
            )
        return {
            "sec": m.group("sec"),
            "idx": int(m.group("idx") or 0),
            "x": float(m.group("x") or 0.5),
        }
    if isinstance(site_raw, (list, tuple)):
        if not site_raw:
            raise ValueError("run_cell: empty site list/tuple is invalid")
        site_raw = {
            "sec": site_raw[0],
            "idx": site_raw[1] if len(site_raw) > 1 else 0,
            "x": site_raw[2] if len(site_raw) > 2 else 0.5,
        }
    if not isinstance(site_raw, dict):
        raise TypeError(
            f"run_cell: site must be string/dict/[sec,idx,x] (got {type(site_raw)!r})"
        )

    sec = site_raw.get("sec", site_raw.get("section", site_raw.get("name")))
    if sec in (None, ""):
        raise ValueError("run_cell: site is missing 'sec' (or 'section'/'name')")
    idx_raw = site_raw.get("idx", site_raw.get("index", 0))
    x_raw = site_raw.get("x", 0.5)
    label = site_raw.get("label")

    idx = int(idx_raw)
    x = float(x_raw)
    if idx < 0:
        raise ValueError(f"run_cell: site idx must be >= 0 (got {idx})")
    if x < 0.0 or x > 1.0:
        raise ValueError(f"run_cell: site x must be in [0, 1] (got {x})")

    out = {"sec": str(sec), "idx": idx, "x": x}
    if label not in (None, ""):
        out["label"] = str(label)
    return out


def _get_cell_recording_cfg(sim_cfg: Dict[str, Any]) -> Dict[str, Any]:
    default_vars = {
        "v": True,
        "i_cap": False,
        "ion_currents": False,
        "mech_currents": False,
        "ion_concentrations": False,
        "ion_reversals": False,
        "mech_conductances": False,
        "mech_states": False,
    }
    cfg_raw = sim_cfg.get("cell_recording", {})
    if cfg_raw is None:
        cfg_raw = {}
    if isinstance(cfg_raw, (bool, str)):
        cfg_raw = {"enabled": cfg_raw}
    if not isinstance(cfg_raw, dict):
        raise TypeError("run_cell: sim_cfg['cell_recording'] must be a dict/bool/string")

    enabled = _parse_bool_like(cfg_raw.get("enabled", False), default=False)
    vars_raw = cfg_raw.get("vars", {}) or {}
    if not isinstance(vars_raw, dict):
        raise TypeError("run_cell: sim_cfg['cell_recording']['vars'] must be a dict")

    vars_cfg = dict(default_vars)
    for key, val in vars_raw.items():
        if key not in vars_cfg:
            allowed = ", ".join(sorted(vars_cfg.keys()))
            raise ValueError(
                f"run_cell: unknown cell_recording vars key {key!r}; allowed: {allowed}"
            )
        vars_cfg[key] = _parse_bool_like(val, default=vars_cfg[key])

    sites_raw = cfg_raw.get("sites", [{"sec": "soma", "idx": 0, "x": 0.5}])
    if isinstance(sites_raw, (str, dict, tuple)):
        sites_raw = [sites_raw]
    if not isinstance(sites_raw, list):
        raise TypeError("run_cell: sim_cfg['cell_recording']['sites'] must be a list")
    if not sites_raw:
        sites_raw = [{"sec": "soma", "idx": 0, "x": 0.5}]

    sites = [_normalize_runtime_recording_site(site) for site in sites_raw]
    n_trials_raw = cfg_raw.get("n_trials", cfg_raw.get("n_traces_to_save", None))
    if n_trials_raw is None:
        n_trials_raw = sim_cfg.get("n_traces_to_save", 1)
    try:
        n_trials = int(n_trials_raw)
    except Exception:
        n_trials = int(sim_cfg.get("n_traces_to_save", 1))
    if n_trials < 0:
        n_trials = 0
    return {"enabled": bool(enabled), "n_trials": int(n_trials), "vars": vars_cfg, "sites": sites}


def _resolve_recording_site(cell: Any, site: Dict[str, Any]) -> Tuple[Any, str]:
    sec_name = str(site["sec"])
    sec_list = cell_sections(cell, sec_name)
    if not sec_list:
        raise ValueError(f"run_cell: section list '{sec_name}' not found on cell")
    idx = int(site["idx"])
    if idx < 0 or idx >= len(sec_list):
        raise ValueError(
            f"run_cell: section index out of range for '{sec_name}' (idx={idx}, n={len(sec_list)})"
        )
    x = float(site["x"])
    seg = sec_list[idx](x)
    default_label = f"{sec_name}[{idx}]({x:.3f})"
    return seg, str(site.get("label", default_label))


def _build_cell_recorders_for_site(seg: Any, vars_cfg: Dict[str, bool]) -> Dict[str, Any]:
    recorders: Dict[str, Any] = {}

    if vars_cfg.get("v", True) and hasattr(seg, "_ref_v"):
        recorders["v"] = h.Vector().record(seg._ref_v)
    if vars_cfg.get("i_cap", False) and hasattr(seg, "_ref_i_cap"):
        recorders["i_cap"] = h.Vector().record(seg._ref_i_cap)

    if vars_cfg.get("ion_currents", False):
        for name in ("ina", "ik", "ica", "ih"):
            ref_name = f"_ref_{name}"
            if hasattr(seg, ref_name):
                recorders[name] = h.Vector().record(getattr(seg, ref_name))

    if vars_cfg.get("ion_concentrations", False):
        for name in ("nai", "ki", "cai", "nao", "ko", "cao"):
            ref_name = f"_ref_{name}"
            if hasattr(seg, ref_name):
                recorders[name] = h.Vector().record(getattr(seg, ref_name))

    if vars_cfg.get("ion_reversals", False):
        for name in ("ena", "ek", "eca"):
            ref_name = f"_ref_{name}"
            if hasattr(seg, ref_name):
                recorders[name] = h.Vector().record(getattr(seg, ref_name))

    if vars_cfg.get("mech_currents", False) or vars_cfg.get("mech_conductances", False) or vars_cfg.get("mech_states", False):
        density_mechs = seg.sec.psection().get("density_mechs", {})
        for mech in sorted(density_mechs.keys()):
            for param in sorted(density_mechs[mech].keys()):
                want = False
                if vars_cfg.get("mech_currents", False) and param.startswith("i"):
                    recorders[f"{mech}.{param}"] = _record_mechanism_current(seg, mech, param)
                    continue
                elif vars_cfg.get("mech_conductances", False) and param.startswith("g"):
                    want = True
                elif vars_cfg.get("mech_states", False) and not param.startswith("i") and not param.startswith("g"):
                    want = True
                if not want:
                    continue
                attr = getattr(seg, mech, None)
                if attr is None:
                    continue
                ref_name = f"_ref_{param}"
                if hasattr(attr, ref_name):
                    recorders[f"{mech}.{param}"] = h.Vector().record(getattr(attr, ref_name))

    return recorders


@cell_execution
def run_cell(
    cell,
    sim_cfg,
    *,
    syn_state=None,
    trial_idx=0,
    syn_recording_retained_bytes=0,
):
    sim_traces = {}

    # Opt-in recorder discovery and memory preflight happen before any Vector
    # allocation. Disabled/exhausted trial capture does not inspect synapses.
    syn_recorders = prepare_synaptic_recording(
        cell, sim_cfg, syn_state=syn_state, trial_idx=trial_idx,
        retained_bytes=syn_recording_retained_bytes,
    )

    # Recorders
    tvec = h.Vector().record(h._ref_t)
    vseg = _get_soma_segment(cell)
    vvec = h.Vector().record(vseg._ref_v)  # somatic Vm

    cell_recording_cfg = _get_cell_recording_cfg(sim_cfg)
    cell_recorders: Dict[str, Dict[str, Any]] = {}
    if cell_recording_cfg.get("enabled", False):
        for site in cell_recording_cfg.get("sites", []):
            seg, label = _resolve_recording_site(cell, site)
            label_base = label
            dupe_idx = 2
            while label in cell_recorders:
                label = f"{label_base}#{dupe_idx}"
                dupe_idx += 1
            site_recorders = _build_cell_recorders_for_site(seg, cell_recording_cfg.get("vars", {}))
            if site_recorders:
                cell_recorders[label] = site_recorders

    # Simulation parameters (ms)
    dt     = float(sim_cfg.get("dt", 0.025))
    tstart = float(sim_cfg.get("tstart", 0.0))
    tstop  = float(sim_cfg["tstop"])

    h.dt = dt
    if hasattr(h, "steps_per_ms"):
        # stdrun.hoc's run() otherwise silently recomputes dt from this value.
        h.steps_per_ms = 1.0 / dt
    h.tstop = tstop
    conditions = apply_sim_conditions(cell, sim_cfg)
    h.finitialize(conditions["v_init_mV"])
    h.t = tstart

    h.run()

    sim_traces["T"] = np.array(tvec)
    sim_traces["V"] = np.array(vvec)
    syn_recordings = finish_synaptic_recording(syn_recorders)
    if syn_recordings is not None:
        sim_traces["syn_recordings"] = syn_recordings
    if cell_recorders:
        sim_traces["cell_recordings"] = {
            site: {name: np.array(vec) for name, vec in recs.items()}
            for site, recs in cell_recorders.items()
        }

    return sim_traces
