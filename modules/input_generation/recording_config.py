"""Cell-recording option normalization for simulation configs."""

from __future__ import annotations

from typing import Any, Dict, List

import math
import re



# 2.3.2 – Configure all groups
# ====================================================================

_SITE_SPEC_RE = re.compile(
    r"^(?P<sec>[A-Za-z_]\w*)(?:\[(?P<idx>\d+)\])?(?:\((?P<x>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\))?$"
)


SYNAPTIC_RECORDING_VARS = {
    "i": True,
    "g": True,
    "i_AMPA": False,
    "i_NMDA": False,
    "g_AMPA": False,
    "g_NMDA": False,
    "record_use": False,
    "record_Pr": False,
}


def _recording_integer(value: Any, *, key: str, minimum: int | None = 0) -> int:
    try:
        number = int(value)
        if isinstance(value, bool) or float(value) != number:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if minimum is not None and number < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return number


def _synaptic_vars(value: Any, *, key: str, defaults: Dict[str, bool]) -> Dict[str, bool]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a dict of variable toggles")
    out = dict(defaults)
    for name, enabled in value.items():
        if name not in SYNAPTIC_RECORDING_VARS:
            raise ValueError(f"{key} has unknown variable {name!r}")
        out[name] = _parse_bool_like(enabled)
    return out


def normalize_syn_recording_config(sim_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the opt-in synapse recorder without importing NEURON.

    Group mode captures exact population aggregates, sample mode captures a
    bounded deterministic subset, and both combines them. The sample limit
    never reduces the population used for group aggregates.
    """
    key = "sim['syn_recording']"
    raw = sim_cfg.get("syn_recording")
    if raw is None:
        raw = {}
    elif isinstance(raw, (bool, str)):
        raw = {"enabled": raw}
    if not isinstance(raw, dict):
        raise TypeError(f"{key} must be a dict, bool, string, or null")

    def mode(value: Any, path: str) -> str:
        result = str(value).strip().lower()
        if result not in {"group", "sample", "both"}:
            raise ValueError(f"{path} must be 'group', 'sample', or 'both'")
        return result

    default_mode = mode(raw.get("default_mode", "group"), f"{key}['default_mode']")
    sample = _recording_integer(
        raw.get("default_sample_per_group", 10), key=f"{key}['default_sample_per_group']"
    )
    variables = _synaptic_vars(
        raw.get("default_vars"), key=f"{key}['default_vars']", defaults=SYNAPTIC_RECORDING_VARS
    )
    n_trials = raw.get("n_trials")
    if n_trials is not None:
        n_trials = _recording_integer(n_trials, key=f"{key}['n_trials']")
    seed = _recording_integer(raw.get("sample_seed", 0), key=f"{key}['sample_seed']", minimum=None)

    record_dt = raw.get("record_dt_ms")
    if record_dt is not None:
        try:
            record_dt = float(record_dt)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key}['record_dt_ms'] must be finite and positive or null") from exc
        if not math.isfinite(record_dt) or record_dt <= 0:
            raise ValueError(f"{key}['record_dt_ms'] must be finite and positive or null")
        if record_dt < float(sim_cfg.get("dt", 0.025)):
            raise ValueError(f"{key}['record_dt_ms'] must be >= sim['dt']")
    try:
        memory = float(raw.get("max_memory_mb", 256))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key}['max_memory_mb'] must be finite and > 0") from exc
    if not math.isfinite(memory) or memory <= 0:
        raise ValueError(f"{key}['max_memory_mb'] must be finite and > 0")

    groups_raw = raw.get("groups")
    if groups_raw is None:
        groups_raw = {}
    if not isinstance(groups_raw, dict):
        raise TypeError(f"{key}['groups'] must be a dict")
    groups: Dict[str, Any] = {}
    for name, override in groups_raw.items():
        path = f"{key}['groups'][{name!r}]"
        if not isinstance(name, str) or not name:
            raise ValueError(f"{key}['groups'] keys must be nonempty strings")
        if isinstance(override, bool):
            override = {"enabled": override}
        if not isinstance(override, dict):
            raise TypeError(f"{path} must be a dict or bool")
        groups[name] = {
            "enabled": _parse_bool_like(override.get("enabled", True)),
            "mode": mode(override.get("mode", default_mode), f"{path}['mode']"),
            "sample_per_group": _recording_integer(
                override.get("sample_per_group", sample), key=f"{path}['sample_per_group']"
            ),
            "vars": _synaptic_vars(override.get("vars"), key=f"{path}['vars']", defaults=variables),
        }
    return {
        "enabled": _parse_bool_like(raw.get("enabled", False)),
        "default_mode": default_mode,
        "default_sample_per_group": sample,
        "default_vars": variables,
        "groups": groups,
        "n_trials": n_trials,
        "sample_seed": seed,
        "record_dt_ms": record_dt,
        "max_memory_mb": memory,
    }


def resolve_n_syn_recording_trials(sim_cfg: Dict[str, Any], n_trials: int | None = None) -> int:
    """Return the independent synaptic capture cap (null inherits Vm cap).

    ``n_trials=None`` leaves the requested cap unclamped for result merging.
    This dependency-light helper is also used by persistence-only code.
    """
    cfg = normalize_syn_recording_config(sim_cfg)
    if not cfg["enabled"]:
        return 0
    requested = cfg["n_trials"]
    if requested is None:
        requested = _recording_integer(sim_cfg.get("n_traces_to_save", 1), key="sim['n_traces_to_save']")
    return requested if n_trials is None else min(requested, max(0, int(n_trials)))


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
