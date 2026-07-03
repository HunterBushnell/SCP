"""Cell-recording option normalization for simulation configs."""

from __future__ import annotations

from typing import Any, Dict, List

import re



# 2.3.2 – Configure all groups
# ====================================================================

_SITE_SPEC_RE = re.compile(
    r"^(?P<sec>[A-Za-z_]\w*)(?:\[(?P<idx>\d+)\])?(?:\((?P<x>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\))?$"
)


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


def _normalize_recording_site_spec(site_raw: Any, *, key: str) -> Dict[str, Any]:
    if isinstance(site_raw, str):
        match = _SITE_SPEC_RE.match(site_raw.strip())
        if not match:
            raise ValueError(
                f"{key}: invalid site spec {site_raw!r}; expected 'sec', 'sec[idx]' or 'sec[idx](x)'"
            )
        sec = match.group("sec")
        idx = int(match.group("idx") or 0)
        x = float(match.group("x") or 0.5)
        return {"sec": sec, "idx": idx, "x": x}

    if isinstance(site_raw, (list, tuple)):
        if not site_raw:
            raise ValueError(f"{key}: empty site list/tuple is not valid")
        site_raw = {
            "sec": site_raw[0],
            "idx": site_raw[1] if len(site_raw) > 1 else 0,
            "x": site_raw[2] if len(site_raw) > 2 else 0.5,
        }

    if not isinstance(site_raw, dict):
        raise TypeError(
            f"{key}: each site must be a string, dict, or [sec, idx, x] tuple (got {type(site_raw)!r})"
        )

    sec = site_raw.get("sec", site_raw.get("section", site_raw.get("name")))
    if sec in (None, ""):
        raise ValueError(f"{key}: site dict is missing required 'sec' (or 'section'/'name')")
    sec = str(sec)

    idx_raw = site_raw.get("idx", site_raw.get("index", 0))
    x_raw = site_raw.get("x", 0.5)
    try:
        idx = int(idx_raw)
    except Exception as exc:
        raise ValueError(f"{key}: site idx must be integer-like (got {idx_raw!r})") from exc
    if idx < 0:
        raise ValueError(f"{key}: site idx must be >= 0 (got {idx})")

    try:
        x = float(x_raw)
    except Exception as exc:
        raise ValueError(f"{key}: site x must be numeric (got {x_raw!r})") from exc
    if x < 0.0 or x > 1.0:
        raise ValueError(f"{key}: site x must be within [0, 1] (got {x})")

    out = {"sec": sec, "idx": idx, "x": x}
    label = site_raw.get("label")
    if label not in (None, ""):
        out["label"] = str(label)
    return out


def _normalize_cell_recording_config(
    sim_cfg_raw: Dict[str, Any],
    sim_cfg: Dict[str, Any],
) -> None:
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
    default_sites = [{"sec": "soma", "idx": 0, "x": 0.5}]

    cell_rec_raw = sim_cfg.get("cell_recording", None)

    rec_sec_list = sim_cfg_raw.get("rec_sec_list", sim_cfg.get("rec_sec_list", None))
    rec_var_toggles = sim_cfg_raw.get("rec_var_toggles", sim_cfg.get("rec_var_toggles", None))
    if rec_var_toggles is None:
        rec_var_toggles = sim_cfg_raw.get("rec_vars", sim_cfg.get("rec_vars", None))

    if cell_rec_raw is None and (rec_sec_list is not None or rec_var_toggles is not None):
        cell_rec_raw = {
            "enabled": True,
            "sites": rec_sec_list,
            "vars": rec_var_toggles,
        }

    if cell_rec_raw is None:
        cell_rec_cfg = {}
    elif isinstance(cell_rec_raw, dict):
        cell_rec_cfg = dict(cell_rec_raw)
    elif isinstance(cell_rec_raw, (bool, str)):
        cell_rec_cfg = {"enabled": cell_rec_raw}
    else:
        raise TypeError(
            "sim['cell_recording'] must be a dict/bool/string or null"
        )

    enabled = _parse_bool_like(cell_rec_cfg.get("enabled", False), default=False)

    sites_raw = cell_rec_cfg.get("sites", None)
    if sites_raw is None:
        sites_raw = default_sites
    if isinstance(sites_raw, (str, dict, tuple)):
        sites_raw = [sites_raw]
    if not isinstance(sites_raw, list):
        raise TypeError("sim['cell_recording']['sites'] must be a list (or single string/dict)")
    if not sites_raw:
        sites_raw = default_sites

    sites: List[Dict[str, Any]] = []
    for i, item in enumerate(sites_raw):
        sites.append(
            _normalize_recording_site_spec(
                item,
                key=f"sim['cell_recording']['sites'][{i}]",
            )
        )

    vars_raw = cell_rec_cfg.get("vars", {})
    if vars_raw is None:
        vars_raw = {}
    if not isinstance(vars_raw, dict):
        raise TypeError("sim['cell_recording']['vars'] must be a dict")

    vars_norm = dict(default_vars)
    for k, v in vars_raw.items():
        if k not in default_vars:
            allowed = ", ".join(sorted(default_vars.keys()))
            raise ValueError(
                f"sim['cell_recording']['vars'] has unknown key {k!r}; allowed keys: {allowed}"
            )
        vars_norm[k] = _parse_bool_like(v, default=default_vars[k])

    n_trials_raw = cell_rec_cfg.get("n_trials", cell_rec_cfg.get("n_traces_to_save", None))
    if n_trials_raw in (None, ""):
        n_trials_norm = None
    else:
        try:
            n_trials_norm = int(n_trials_raw)
        except Exception as exc:
            raise ValueError(
                "sim['cell_recording']['n_trials'] must be integer-like or null"
            ) from exc
        if n_trials_norm < 0:
            raise ValueError("sim['cell_recording']['n_trials'] must be >= 0")

    sim_cfg["cell_recording"] = {
        "enabled": bool(enabled),
        "n_trials": n_trials_norm,
        "sites": sites,
        "vars": vars_norm,
    }


def _sync_trace_limit_with_cell_recording(
    sim_cfg: Dict[str, Any],
    *,
    prefer_cell: bool,
) -> None:
    """
    Keep top-level `n_traces_to_save` and `cell_recording.n_trials` aligned.

    - prefer_cell=True: `cell_recording.n_trials` (if set) overrides top-level.
    - prefer_cell=False: top-level trace limit wins (used after snapshot overrides).
    """
    cell_cfg = sim_cfg.get("cell_recording")
    if not isinstance(cell_cfg, dict):
        return

    trace_raw = sim_cfg.get("n_traces_to_save", 1)
    try:
        trace_n = int(trace_raw)
    except Exception as exc:
        raise ValueError(
            f"sim['n_traces_to_save'] must be integer-like (got {trace_raw!r})"
        ) from exc
    if trace_n < 0:
        raise ValueError("sim['n_traces_to_save'] must be >= 0")

    rec_raw = cell_cfg.get("n_trials", None)
    rec_n = None
    if rec_raw is not None:
        try:
            rec_n = int(rec_raw)
        except Exception as exc:
            raise ValueError(
                "sim['cell_recording']['n_trials'] must be integer-like or null"
            ) from exc
        if rec_n < 0:
            raise ValueError("sim['cell_recording']['n_trials'] must be >= 0")

    if prefer_cell and rec_n is not None:
        merged = rec_n
    else:
        merged = trace_n

    sim_cfg["n_traces_to_save"] = int(merged)
    cell_cfg["n_trials"] = int(merged)
    sim_cfg["cell_recording"] = cell_cfg
