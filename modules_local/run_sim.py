from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Mapping, Tuple
import json, copy, os, pickle, math, random, time, sys, hashlib
import shutil
import re
import numpy as np
import matplotlib.pyplot as plt
from neuron import h, gui  # gui not needed in headless scripts
############################################################




######## Simple Current Injection simulation #########

def get_rec_vars_for_i_in_sec(sec,seg):#create recording variables for every current in a section
    current_recording_vars = {}

    for mech in sec.psection()['density_mechs']:
        for param in sec.psection()['density_mechs'][mech]:
            if param[0] == 'i':#assumes only current names start with i
                attr = getattr(sec(seg),mech)
                ref = getattr(attr, f"_ref_{param}")
                current_recording_vars[f"{mech}.{param}"] = h.Vector().record(ref)
    # print(current_recording_vars)
    # ivecs = {}
    # for name,vec in current_recording_vars.items():
    #     # print(name)
    #     ivecs[name] = vec.as_numpy().copy

    return current_recording_vars #ivecs


#function to calculate the frequency of a voltage trace
def get_frequency(v,sim_params):

    start_idx = int(sim_params['stim_delay']/sim_params['h_dt'])
    end_idx = int((sim_params['stim_delay']+sim_params['stim_dur'])/sim_params['h_dt'])
    
    v_range = v.as_numpy()[start_idx:end_idx]
    # Calculate the slope of the voltage
    slope = np.diff(v_range)
    above_threshold_indices = np.where(v_range[:-1] > -20)[0]
    positive_to_negative_indices = np.where((slope[:-1] > 0) & (slope[1:] < 0))[0]
    event_indices = np.intersect1d(above_threshold_indices, positive_to_negative_indices)
    spikes = len(event_indices)

    if spikes> 0:
        duration_sec = sim_params['stim_dur'] / 1000.0
        freq = spikes / duration_sec
        return freq
    else:
        return 0
    

def _get_hoc(cell):
    return getattr(cell, "h", cell)


def run_current_injection(cell,sim_params,):
    
    # from neuron import h
    # h.load_file("stdrun.hoc")

    hoc = _get_hoc(cell)
    stim = h.IClamp(hoc.soma[0](0.5))
    stim.amp = sim_params['stim_amp']
    stim.delay = sim_params['stim_delay']
    stim.dur = sim_params['stim_dur']
    h.tstop = sim_params['h_tstop']
    h.dt = sim_params['h_dt']
    # h.steps_per_ms = 1 / h.dt
    # return h, stim

    recorded_data = {}

    # Attach recorders
    vvec = h.Vector().record(hoc.soma[0](0.5)._ref_v)
    tvec = h.Vector().record(h._ref_t)

    # Attach *all* the current recorders in that section
    current_recording_vars = get_rec_vars_for_i_in_sec(hoc.soma[0], 0.5)

    # Run
    h.finitialize()
    h.run()

    # Stash numpy arrays
    recorded_data['T'] = np.array(tvec)                # Time (same for all sims)
    recorded_data['V'] = np.array(vvec)                   # Voltage
    recorded_data['F'] = get_frequency(vvec, sim_params)
    recorded_data['I'] = {name: vec.as_numpy().copy() for name, vec in current_recording_vars.items()} #ivecs

    return recorded_data


def run_iclamp_test(cell, sim_cfg, iclamp_cfg=None):
    """
    Run a simple somatic current injection (IClamp) test.

    Uses sim_cfg plus optional iclamp_cfg overrides to build parameters.
    Returns a results dict compatible with save_results (traces + meta).
    """
    sim_cfg = dict(sim_cfg or {})
    iclamp = dict(sim_cfg.get("iclamp", {}) or {})
    if iclamp_cfg:
        iclamp.update(iclamp_cfg)

    def _get_float(key, fallback):
        val = iclamp.get(key, None)
        if val in (None, "", False):
            val = fallback
        return float(val)

    amp_nA = _get_float("amp_nA", 0.2)
    delay_ms = _get_float("delay_ms", sim_cfg.get("tstart", 0.0))
    dur_ms = _get_float("dur_ms", sim_cfg.get("stim_duration_ms", 500.0))
    dt_ms = _get_float("dt_ms", sim_cfg.get("dt", 0.025))

    tstop_raw = iclamp.get("tstop_ms", None)
    if tstop_raw in (None, "", False):
        tstop_raw = sim_cfg.get("tstop", delay_ms + dur_ms)
    tstop_ms = float(tstop_raw)

    sim_params = {
        "stim_amp": amp_nA,
        "stim_delay": delay_ms,
        "stim_dur": dur_ms,
        "h_tstop": tstop_ms,
        "h_dt": dt_ms,
    }

    record_currents = bool(iclamp.get("record_currents", False))
    recorded = run_current_injection(cell, sim_params)
    if not record_currents:
        recorded.pop("I", None)

    meta = {
        "experiment": "iclamp",
        "amp_nA": amp_nA,
        "delay_ms": delay_ms,
        "dur_ms": dur_ms,
        "tstop_ms": tstop_ms,
        "dt_ms": dt_ms,
        "record_currents": record_currents,
        "frequency_hz": recorded.get("F"),
    }

    results = {
        "mode": "iclamp",
        "sim_cfg": sim_cfg,
        "meta": meta,
        "traces": {"T": recorded.get("T"), "V": recorded.get("V")},
        "iclamp": recorded,
    }
    return results

def looped_current_injection(cell,sim_params,sim_amps,):

    looped_records = { # Could automate from items in recorded_data later?
        'T': {},
        'V': {},
        'F': {},
        'I': {},
    }

    for idx, amp_val in enumerate(sim_amps):
        # 1) set the new amp
        sim_params['stim_amp'] = amp_val / 1000.0   # convert to nA

        # 3) run_sim
        recorded_data = run_current_injection(cell,sim_params)
        T,V,F,I = recorded_data['T'],recorded_data['V'],recorded_data['F'],recorded_data['I']
        for rec_var in recorded_data:
            looped_records[rec_var][amp_val] = recorded_data[rec_var]
    
    return looped_records

def plot_looped_currents(cell_name,trial_amp,currents,looped_records,window):

    plt.figure(figsize = (6,4))
    if currents:
        for cur in currents:
            plt.plot(looped_records['T'][trial_amp], looped_records['I'][trial_amp][cur], label=cur)
    else:
        for cur_name, cur_trace in looped_records['I'][trial_amp].items():
            plt.plot(looped_records['T'][trial_amp], cur_trace, label=cur_name) 

    plt.xlabel("Time (ms)")
    plt.xlim(window[0],window[1])
    plt.ylabel("Current (A/cm²)")
    # plt.ylim(-0.01,0.01)
    plt.title(f"{cell_name} currents @ {trial_amp} pA")
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

        
def run_FI(cell,sim_params,amps,):

    looped_records = looped_current_injection(cell,sim_params,amps)
    freq_records = looped_records['F']
    freq_list = [freq_records[amp] for amp in freq_records]

    plt.plot(amps, freq_list, marker='o')
    plt.title(f"FI CURVE")
    plt.xlabel("Stimulus Amplitude (nA)")
    plt.ylabel("Frequency (Hz)")
    plt.grid(),plt.show()
    
    return freq_records


#############################################
#############################################

import numpy as np
from neuron import h, gui  # gui not needed in headless scripts


def _get_soma_segment(cell):
    """
    Return a NEURON soma(0.5) segment for both the new LoadedCell
    (which exposes `cell.h.soma`) and older cell wrappers that have
    `cell.soma` directly.
    """
    # Prefer the NEURON hoc object inside LoadedCell.
    h_obj = getattr(cell, "h", None)
    if h_obj is not None and hasattr(h_obj, "soma") and len(h_obj.soma) > 0:
        return h_obj.soma[0](0.5)

    # Fallback to older pattern where the cell itself had `soma`.
    if hasattr(cell, "soma") and len(cell.soma) > 0:
        return cell.soma[0](0.5)

    raise AttributeError("run_sim: could not find soma on cell or cell.h")


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
    hoc = _get_hoc(cell)
    sec_name = str(site["sec"])
    if not hasattr(hoc, sec_name):
        raise ValueError(f"run_cell: section list '{sec_name}' not found on cell")
    sec_list = getattr(hoc, sec_name)
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
            attr = getattr(seg, mech, None)
            if attr is None:
                continue
            for param in sorted(density_mechs[mech].keys()):
                want = False
                if vars_cfg.get("mech_currents", False) and param.startswith("i"):
                    want = True
                elif vars_cfg.get("mech_conductances", False) and param.startswith("g"):
                    want = True
                elif vars_cfg.get("mech_states", False) and not param.startswith("i") and not param.startswith("g"):
                    want = True
                if not want:
                    continue
                ref_name = f"_ref_{param}"
                if hasattr(attr, ref_name):
                    recorders[f"{mech}.{param}"] = h.Vector().record(getattr(attr, ref_name))

    return recorders


def run_cell(cell, sim_cfg):
    sim_traces = {}

    # Recorders
    tvec = h.Vector().record(h._ref_t)
    vseg = _get_soma_segment(cell)
    vvec = h.Vector().record(vseg._ref_v)  # somatic Vm

    isynvec = None
    gsynvec = None
    if hasattr(cell, "synapses") and len(cell.synapses) > 0:
        isynvec = h.Vector().record(cell.synapses[0]._ref_i)
        gsynvec = h.Vector().record(cell.synapses[0]._ref_g)

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

    h.t = tstart
    v_init = float(getattr(cell, "Vinit", -65.0))
    h.finitialize(v_init)
    h.dt = dt
    h.tstop = tstop

    h.run()

    sim_traces["T"] = np.array(tvec)
    sim_traces["V"] = np.array(vvec)
    if isynvec is not None:
        sim_traces["I"] = np.array(isynvec)
        sim_traces["G"] = np.array(gsynvec)
    if cell_recorders:
        sim_traces["cell_recordings"] = {
            site: {name: np.array(vec) for name, vec in recs.items()}
            for site, recs in cell_recorders.items()
        }

    return sim_traces

#############################################
#############################################

"""
run_sim.py

Core simulation entrypoints for single / multi / parametric runs,
built on top of:
  - inputs.generate_inputs(...)
  - synapses.add_synapses(...)
  - run_cell(cell, sim_cfg)
"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np

from . import synapses  # assumes run_sim.py lives next to synapses.py
from . import randomness
from . import inputs as inputs_mod


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


def _clear_cell_state(cell: Any) -> None:
    """
    Best-effort clearing of NEURON-related state on the cell between trials.
    Assumes the cell exposes lists/containers with these attribute names
    (missing ones are ignored).
    """
    for attr in ("syn_locs", "vecs", "stims", "synapses", "netcons"):
        if hasattr(cell, attr):
            lst = getattr(cell, attr)
            try:
                lst.clear()
            except AttributeError:
                # older code may use h.List or similar; fall back to manual deletion
                try:
                    while len(lst) > 0:
                        lst.remove(lst[0])
                except Exception:
                    pass


def _warn_preexisting_synapses(cell: Any, *, context: str = "") -> None:
    counts = []
    for attr in ("synapses", "netcons", "stims", "vecs"):
        if hasattr(cell, attr):
            try:
                n = len(getattr(cell, attr))
            except Exception:
                n = None
            if n:
                counts.append(f"{attr}={n}")
    if counts:
        label = f" ({context})" if context else ""
        msg = "WARNING: pre-attached synapse objects detected"
        print(f"{msg}{label}: " + ", ".join(counts))
        print("         This can change results; attach synapses inside run_sim only.")


def _detect_spikes(T: np.ndarray, V: np.ndarray, v_thresh: float = 0.0) -> np.ndarray:
    """
    Simple spike detector: returns times where V crosses v_thresh from below.
    This is intentionally minimal and can be replaced later with a better detector.
    """
    above = V > v_thresh
    crossings = np.where(above[1:] & ~above[:-1])[0] + 1
    return T[crossings]


def _as_bool(val: Any, default: bool = True) -> bool:
    if val is None:
        return default
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("false", "0", "no", "off", ""):
            return False
        if v in ("true", "1", "yes", "on"):
            return True
    return bool(val)


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


def _sha256_file(path: Path) -> str:
    hsh = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hsh.update(chunk)
    return hsh.hexdigest()


def _resolve_tune_path(sim_cfg: Dict[str, Any]) -> Optional[Path]:
    tune_dir = sim_cfg.get("tune_dir")
    if tune_dir:
        try:
            return Path(str(tune_dir)).expanduser().resolve()
        except Exception:
            return None

    cell = sim_cfg.get("cell")
    tune = sim_cfg.get("tune")
    if cell and tune:
        base = Path(__file__).resolve().parent.parent
        return base / "cells" / str(cell) / "tunes" / str(tune)
    return None


def _find_fit_json_path(sim_cfg: Dict[str, Any]) -> Optional[Path]:
    tune_path = _resolve_tune_path(sim_cfg)
    if tune_path is None or not tune_path.is_dir():
        return None

    manifest_path = tune_path / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest_data = json.loads(manifest_path.read_text())
            biophys = manifest_data.get("biophys", [])
            if isinstance(biophys, list):
                for entry in biophys:
                    if not isinstance(entry, dict):
                        continue
                    model_file = entry.get("model_file")
                    model_file_items = model_file if isinstance(model_file, list) else [model_file]
                    for item in model_file_items:
                        if not isinstance(item, str):
                            continue
                        cand = Path(item)
                        if not cand.name.endswith("_fit.json"):
                            continue
                        cand = (tune_path / cand).resolve() if not cand.is_absolute() else cand.resolve()
                        if cand.is_file():
                            return cand
        except Exception:
            pass

    fit_candidates = sorted(tune_path.glob("*_fit.json"))
    if fit_candidates:
        return fit_candidates[0].resolve()
    return None


def _copy_fit_json_sidecar(sim_cfg: Dict[str, Any], run_dir: Path) -> Optional[Dict[str, str]]:
    if sim_cfg.get("save_fit_json_sidecar", True) is False:
        return None

    fit_path = _find_fit_json_path(sim_cfg)
    if fit_path is None:
        return None

    target = run_dir / fit_path.name
    try:
        shutil.copy2(fit_path, target)
    except Exception:
        return None

    fit_info: Dict[str, str] = {
        "filename": target.name,
        "source_path": str(fit_path),
    }
    try:
        fit_info["sha256"] = _sha256_file(target)
    except Exception:
        pass
    return fit_info


def _collect_mechanism_info(sim_cfg: Dict[str, Any]) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    tune_path = _resolve_tune_path(sim_cfg)
    if tune_path is None:
        return info

    info["tune_dir"] = str(tune_path)
    mod_dir = tune_path / "modfiles"
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

    dll_candidates = [
        mod_dir / "x86_64" / ".libs" / "libnrnmech.so",
        mod_dir / "x86_64" / "libnrnmech.so",
    ]
    for dll in dll_candidates:
        if dll.is_file():
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
            break

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


def _resolve_trace_trials_to_save(sim_cfg: Dict[str, Any], fallback: int) -> int:
    raw = None
    cell_rec = sim_cfg.get("cell_recording")
    if isinstance(cell_rec, dict):
        raw = cell_rec.get("n_trials", None)
    if raw is None:
        raw = sim_cfg.get("n_traces_to_save", fallback)
    try:
        n = int(raw)
    except Exception:
        return max(0, int(fallback))
    if n < 0:
        return max(0, int(fallback))
    return max(0, n)


def _set_trace_trials_to_save(sim_cfg: Dict[str, Any], n_traces: int) -> None:
    n = max(0, int(n_traces))
    sim_cfg["n_traces_to_save"] = n
    cell_rec = sim_cfg.get("cell_recording")
    if isinstance(cell_rec, dict):
        cell_rec = dict(cell_rec)
        cell_rec["n_trials"] = n
        sim_cfg["cell_recording"] = cell_rec


def _resolve_inputs_to_save(sim_cfg: Dict[str, Any], n_trials: int, fallback: int) -> int:
    raw = sim_cfg.get("n_inputs_to_save", None)
    if raw is None:
        raw = _resolve_trace_trials_to_save(sim_cfg, fallback=fallback)
    if isinstance(raw, str):
        if raw.strip().lower() in ("all",):
            return max(0, int(n_trials))
        try:
            raw = int(raw)
        except Exception:
            return max(0, int(fallback))
    try:
        raw = int(raw)
    except Exception:
        return max(0, int(fallback))
    if raw < 0:
        return max(0, int(n_trials))
    return max(0, raw)


def _coerce_bin_width(val: Any, default: float) -> float:
    try:
        bw = float(val)
    except Exception:
        bw = float(default)
    if bw <= 0:
        bw = float(default)
    return bw


def _smooth_rate_curve(
    centers: np.ndarray,
    rates: np.ndarray,
    bin_ms: float,
    smooth_ms: Optional[float],
    *,
    mode: str = "center",
) -> Tuple[np.ndarray, np.ndarray]:
    if smooth_ms is None:
        return centers, rates
    try:
        smooth_ms = float(smooth_ms)
    except Exception:
        return centers, rates
    if smooth_ms <= 0 or bin_ms <= 0:
        return centers, rates

    k = int(round(smooth_ms / bin_ms))
    if k <= 1 or rates.size < k:
        return centers, rates
    if mode == "center" and k % 2 == 0:
        k += 1

    kernel = np.ones(k, dtype=float) / float(k)
    if mode == "center":
        y = np.convolve(rates, kernel, mode="valid")
        drop = (len(centers) - len(y)) // 2
        if drop < 0:
            return centers, rates
        return centers[drop : drop + len(y)], y
    if mode == "causal":
        pad = (k - 1, 0)
        y = np.convolve(np.pad(rates, pad), kernel, mode="valid")
        return centers[: len(y)], y
    return centers, rates


def _prepare_input_stats_bins(
    tstart: float,
    tstop: float,
    bin_width: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    bw = _coerce_bin_width(bin_width, 25.0)
    t0 = float(tstart)
    t1 = float(tstop)
    if t1 < t0:
        t1 = t0
    bins = np.arange(t0, t1 + bw, bw, dtype=float)
    if bins.size < 2:
        bins = np.array([t0, t0 + bw], dtype=float)
    centers = bins[:-1] + 0.5 * bw
    return bw, bins, centers


def _compute_input_stats_for_trial(
    inputs_by_group: Dict[str, Any],
    bins: np.ndarray,
    bin_width: float,
    tstart: float,
    tstop: float,
) -> Dict[str, Any]:
    bw_s = bin_width / 1000.0
    dur_s = max(1e-9, (float(tstop) - float(tstart)) / 1000.0)
    groups: Dict[str, Any] = {}

    for g, gi in inputs_by_group.items():
        trains = [np.asarray(tr, dtype=float) for tr in (gi.spike_trains or [])]
        n_syn = len(trains)
        if n_syn:
            all_spikes = np.concatenate(trains)
        else:
            all_spikes = np.array([], dtype=float)

        counts, _ = np.histogram(all_spikes, bins=bins)
        total_spikes = int(all_spikes.size)
        rate_hz_total = total_spikes / dur_s
        rate_hz_per_syn = rate_hz_total / n_syn if n_syn > 0 else 0.0

        rate_hz_by_bin_total = counts / bw_s
        if n_syn > 0:
            rate_hz_by_bin_per_syn = rate_hz_by_bin_total / n_syn
        else:
            rate_hz_by_bin_per_syn = np.zeros_like(rate_hz_by_bin_total, dtype=float)

        groups[g] = {
            "n_syn": int(n_syn),
            "total_spikes": total_spikes,
            "rate_hz_total": float(rate_hz_total),
            "rate_hz_per_syn": float(rate_hz_per_syn),
            "counts_by_bin": counts.tolist(),
            "rate_hz_by_bin_total": rate_hz_by_bin_total.tolist(),
            "rate_hz_by_bin_per_syn": rate_hz_by_bin_per_syn.tolist(),
        }

    return groups


def _aggregate_input_stats(trial_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trial_stats:
        return {}

    group_names = set(trial_stats[0].get("groups", {}).keys())
    group_means: Dict[str, Any] = {}

    for g in group_names:
        n_syns: List[int] = []
        totals: List[int] = []
        rate_totals: List[float] = []
        rate_per_syns: List[float] = []
        counts_stack: List[List[float]] = []
        rate_bin_total_stack: List[List[float]] = []
        rate_bin_per_syn_stack: List[List[float]] = []

        for trial in trial_stats:
            gstats = trial.get("groups", {}).get(g)
            if not gstats:
                continue
            n_syns.append(int(gstats.get("n_syn", 0)))
            totals.append(int(gstats.get("total_spikes", 0)))
            rate_totals.append(float(gstats.get("rate_hz_total", 0.0)))
            rate_per_syns.append(float(gstats.get("rate_hz_per_syn", 0.0)))
            counts_stack.append(list(gstats.get("counts_by_bin", [])))
            rate_bin_total_stack.append(list(gstats.get("rate_hz_by_bin_total", [])))
            rate_bin_per_syn_stack.append(list(gstats.get("rate_hz_by_bin_per_syn", [])))

        if not totals:
            continue

        mean_counts = np.mean(np.asarray(counts_stack, dtype=float), axis=0).tolist() if counts_stack else []
        mean_rate_bin_total = (
            np.mean(np.asarray(rate_bin_total_stack, dtype=float), axis=0).tolist()
            if rate_bin_total_stack
            else []
        )
        mean_rate_bin_per_syn = (
            np.mean(np.asarray(rate_bin_per_syn_stack, dtype=float), axis=0).tolist()
            if rate_bin_per_syn_stack
            else []
        )

        group_means[g] = {
            "n_syn": int(n_syns[0]) if n_syns else 0,
            "mean_total_spikes": float(np.mean(totals)),
            "mean_rate_hz_total": float(np.mean(rate_totals)),
            "mean_rate_hz_per_syn": float(np.mean(rate_per_syns)),
            "mean_counts_by_bin": mean_counts,
            "mean_rate_hz_by_bin_total": mean_rate_bin_total,
            "mean_rate_hz_by_bin_per_syn": mean_rate_bin_per_syn,
        }

    return group_means


# ---------------------------------------------------------------------
# core run functions
# ---------------------------------------------------------------------

def run_single(
    cell: Any,
    geom: Any,
    sim_cfg: Dict[str, Any],
    groups_cfg: Dict[str, Any],
    inputs_by_group: Dict[str, Any],
    *,
    rm: Optional[randomness.RandomnessManager] = None,
) -> Dict[str, Any]:
    """
    Run a single simulation (one trial) with the given config and inputs.

    Returns a standardized results dict:
      {
        "mode": "single",
        "sim_cfg": { ... },
        "spikes": 1D np.ndarray of spike times,
        "traces": {
           "T": 1D np.ndarray (time),
           "V": 1D np.ndarray (soma voltage)
        } or {},
        "meta": { ... }
      }
    """
    sim_cfg_local = copy.deepcopy(sim_cfg)
    snapshot_enabled, snapshot_cfg = _snapshot_cfg(sim_cfg_local)
    if snapshot_enabled:
        _apply_snapshot_deterministic(sim_cfg_local, snapshot_cfg)
    trial_rng = rm.trial(0) if rm is not None else None
    n_traces_to_save = _resolve_trace_trials_to_save(sim_cfg_local, fallback=1)
    if snapshot_enabled and snapshot_cfg.get("save_all_traces", True):
        n_traces_to_save = max(n_traces_to_save, 1)
        _set_trace_trials_to_save(sim_cfg_local, n_traces_to_save)
    else:
        _set_trace_trials_to_save(sim_cfg_local, n_traces_to_save)
    n_inputs_to_save = _resolve_inputs_to_save(sim_cfg_local, 1, n_traces_to_save)
    tstart = float(sim_cfg_local.get("tstart", 0.0))
    tstop = float(sim_cfg_local.get("tstop", tstart))

    input_stats = None
    if _as_bool(sim_cfg_local.get("save_input_stats", True), default=True):
        bin_width = sim_cfg_local.get("input_stats_bin_ms", sim_cfg_local.get("bins", 25.0))
        bin_width, bins, centers = _prepare_input_stats_bins(tstart, tstop, bin_width)
        trial_groups = _compute_input_stats_for_trial(
            inputs_by_group, bins, bin_width, tstart, tstop
        )
        trial_stats = [{"trial_idx": 0, "groups": trial_groups}]
        input_stats = {
            "bin_ms": bin_width,
            "t_ms": centers.tolist(),
            "tstart_ms": tstart,
            "tstop_ms": tstop,
            "trials": trial_stats,
            "group_means": _aggregate_input_stats(trial_stats),
        }

    # reset cell state and attach synapses
    _warn_preexisting_synapses(cell, context="run_single")
    _clear_cell_state(cell)
    syn_state = synapses.add_synapses(
        cell, geom, sim_cfg_local, groups_cfg, inputs_by_group, trial_rng=trial_rng
    )
    syn_records = syn_state.get("records", {})
    syn_records_by_trial: Optional[List[Dict[str, Any]]] = None
    if _as_bool(sim_cfg_local.get("save_syn_records_by_trial", False), default=False):
        syn_records_by_trial = [{"trial_idx": 0, "records": syn_records}]
    syn_param_snapshot: Optional[Dict[str, Any]] = None
    netcon_snapshot: Optional[Dict[str, Any]] = None
    if snapshot_enabled:
        syn_param_snapshot = _snapshot_synapse_params(syn_state, groups_cfg)
        netcon_snapshot = _snapshot_netcon_state(syn_state)

    # run the actual simulation (existing 3.1 primitive)
    sim_traces = run_cell(cell, sim_cfg_local)  # assumes this is defined below / in this module

    T = np.asarray(sim_traces.get("T", []), dtype=float)
    V = np.asarray(sim_traces.get("V", []), dtype=float)
    spikes = _detect_spikes(T, V) if T.size and V.size else np.array([], dtype=float)
    cell_recordings = sim_traces.get("cell_recordings")

    traces_out: Dict[str, Any] = {}
    if n_traces_to_save > 0 and T.size and V.size:
        traces_out = {"T": T, "V": V}

    inputs_out: Optional[Dict[str, Any]] = None
    if n_inputs_to_save > 0:
        inputs_out = {}
        for g, gi in inputs_by_group.items():
            inputs_out[g] = {
                "mode": gi.mode,
                "spike_trains": [np.asarray(tr).copy() for tr in gi.spike_trains],
                "meta": gi.meta,
            }

    result = {
        "mode": "single",
        "sim_cfg": sim_cfg_local,
        "spikes": spikes,
        "traces": traces_out,
        "cell_recordings": cell_recordings,
        "syn_records": syn_records,
        "syn_records_by_trial": syn_records_by_trial,
        "inputs": inputs_out,
        "meta": {
            "cell": sim_cfg_local.get("cell"),
            "tune": sim_cfg_local.get("tune"),
            "n_trials": 1,
            "syn_config": copy.deepcopy(groups_cfg),
        },
    }
    if rm is not None:
        result["meta"]["randomness"] = rm.meta().as_dict()
    if snapshot_enabled:
        result["meta"]["snapshot"] = copy.deepcopy(snapshot_cfg)
        result["meta"]["versions"] = _collect_versions()
        result["meta"]["neuron_state"] = _collect_neuron_state()
        result["meta"]["env"] = _collect_env_snapshot()
        result["meta"]["mechanisms"] = _collect_mechanism_info(sim_cfg_local)
        if netcon_snapshot is not None:
            result["meta"]["netcon_snapshot"] = netcon_snapshot
        if syn_param_snapshot is not None:
            result["meta"]["synapse_param_snapshot"] = syn_param_snapshot
    if input_stats is not None:
        result["meta"]["input_stats"] = input_stats
    return result


def run_multi(
    cell: Any,
    geom: Any,
    sim_cfg: Dict[str, Any],
    groups_cfg: Dict[str, Any],
    inputs_by_group: Dict[str, Any],
    *,
    rm: Optional[randomness.RandomnessManager] = None,
    mode_registry: Optional[Dict[str, Any]] = None,
    trial_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run multiple trials with the same config. Currently:
      - reuses inputs_by_group for all trials,
      - re-attaches synapses fresh for each trial.

    Returns:
      {
        "mode": "multi",
        "sim_cfg": { ... },
        "spikes": [np.ndarray, ...],      # one per trial
        "traces": {
           "T": 1D np.ndarray,
           "V": [np.ndarray, ...]         # up to cell_recording.n_trials (legacy: n_traces_to_save)
        } or {},
        "meta": {
           "n_trials": int,
           "trial_ids": [0, 1, ...]
        }
      }
    """
    sim_cfg_local = copy.deepcopy(sim_cfg)
    snapshot_enabled, snapshot_cfg = _snapshot_cfg(sim_cfg_local)
    if snapshot_enabled:
        _apply_snapshot_deterministic(sim_cfg_local, snapshot_cfg)
    n_trials = int(sim_cfg_local.get("n_trials", 1))
    n_traces_to_save = _resolve_trace_trials_to_save(sim_cfg_local, fallback=1)
    if snapshot_enabled and snapshot_cfg.get("save_all_traces", True):
        n_traces_to_save = max(n_traces_to_save, n_trials)
        _set_trace_trials_to_save(sim_cfg_local, n_traces_to_save)
    else:
        _set_trace_trials_to_save(sim_cfg_local, n_traces_to_save)
    trial_offset = int(sim_cfg_local.get("trial_offset", 0) or 0)

    regen_inputs = _as_bool(sim_cfg_local.get("regen_inputs_each_trial", True), default=True)

    # Prebuild mode registry once for per-trial input regeneration
    if mode_registry is None:
        mode_registry = inputs_mod._build_default_mode_registry()
        try:
            from modules_local import input_modes_user

            user_reg = input_modes_user.get_user_mode_registry()
            # user registry wins on name collisions
            mode_registry = {**mode_registry, **user_reg}
        except Exception:
            pass

    spikes_by_trial: List[np.ndarray] = []
    trace_V_store: List[np.ndarray] = []
    cell_recordings_store: List[Dict[str, Any]] = []
    T_ref: Optional[np.ndarray] = None
    input_summaries: List[Dict[str, Any]] = []
    inputs_store: List[Dict[str, Any]] = []
    tstart = float(sim_cfg_local.get("tstart", 0.0))
    tstop = float(sim_cfg_local.get("tstop", 0.0))
    sim_dur_s = max(1e-9, (tstop - tstart) / 1000.0)
    inputs_to_save = _resolve_inputs_to_save(sim_cfg_local, n_trials, n_traces_to_save)
    input_stats_enabled = _as_bool(sim_cfg_local.get("save_input_stats", True), default=True)
    save_syn_records_by_trial = _as_bool(
        sim_cfg_local.get("save_syn_records_by_trial", False), default=False
    )
    input_stats_trials: List[Dict[str, Any]] = []
    syn_records_by_trial: List[Dict[str, Any]] = []
    syn_param_snapshot: Optional[Dict[str, Any]] = None
    netcon_snapshot: Optional[Dict[str, Any]] = None
    input_bin_width = sim_cfg_local.get("input_stats_bin_ms", sim_cfg_local.get("bins", 25.0))
    input_bin_width, input_bins, input_centers = _prepare_input_stats_bins(
        tstart, tstop, input_bin_width
    )

    _warn_preexisting_synapses(cell, context="run_multi")
    for trial_idx in range(n_trials):
        trial_start = time.perf_counter()
        trial_rng_idx = trial_idx + trial_offset
        trial_rng = rm.trial(trial_rng_idx) if rm is not None else None

        # Optionally regenerate inputs per trial (fresh randomness)
        if regen_inputs:
            gcfg_trial = copy.deepcopy(groups_cfg)
            inputs_trial = inputs_mod._process_all_groups(
                sim_cfg=sim_cfg_local,
                groups_cfg=gcfg_trial,
                geometry=geom,
                mode_registry=mode_registry,
                rng=None,
                trial_rng=trial_rng,
            )
            groups_cfg_for_trial = gcfg_trial
        else:
            inputs_trial = inputs_by_group
            groups_cfg_for_trial = groups_cfg

        if inputs_to_save > 0 and len(inputs_store) < inputs_to_save:
            trial_inputs: Dict[str, Any] = {}
            for g, gi in inputs_trial.items():
                trial_inputs[g] = {
                    "mode": gi.mode,
                    "spike_trains": [np.asarray(tr).copy() for tr in gi.spike_trains],
                    "meta": gi.meta,
            }
            inputs_store.append({"trial_idx": trial_idx, "inputs": trial_inputs})

        if input_stats_enabled:
            groups_stats = _compute_input_stats_for_trial(
                inputs_trial, input_bins, input_bin_width, tstart, tstop
            )
            input_stats_trials.append({"trial_idx": trial_idx, "groups": groups_stats})

        # Optional per-trial input summary (helps detect identical inputs)
        log_input_summary = bool(sim_cfg_local.get("log_input_summary", True))
        summary: Dict[str, Any] = {}
        for g, gi in inputs_trial.items():
            trains = gi.spike_trains or []
            total_spikes = int(sum(len(tr) for tr in trains))
            sum_spike_times = float(sum(float(np.sum(tr)) for tr in trains)) if trains else 0.0
            summary[g] = {
                "n_syn": int(len(trains)),
                "total_spikes": total_spikes,
                "sum_spike_times": sum_spike_times,
            }
        input_summaries.append({"trial_idx": trial_idx, "groups": summary})
        if log_input_summary:
            parts = [f"{g}={summary[g]['total_spikes']}" for g in summary]
            print(f"[trial {trial_idx+1}/{n_trials}] input_spikes: " + " ".join(parts))

        _clear_cell_state(cell)
        syn_state = synapses.add_synapses(
            cell, geom, sim_cfg_local, groups_cfg_for_trial, inputs_trial, trial_rng=trial_rng
        )
        if save_syn_records_by_trial:
            syn_records_by_trial.append(
                {"trial_idx": trial_idx, "records": syn_state.get("records", {})}
            )
        if snapshot_enabled and trial_idx == 0:
            syn_param_snapshot = _snapshot_synapse_params(syn_state, groups_cfg_for_trial)
            netcon_snapshot = _snapshot_netcon_state(syn_state)

        sim_traces = run_cell(cell, sim_cfg_local)

        T = np.asarray(sim_traces.get("T", []), dtype=float)
        V = np.asarray(sim_traces.get("V", []), dtype=float)
        spikes = _detect_spikes(T, V) if T.size and V.size else np.array([], dtype=float)

        spikes_by_trial.append(spikes)

        # save traces for a subset of trials
        if n_traces_to_save > 0 and len(trace_V_store) < n_traces_to_save and T.size and V.size:
            if T_ref is None:
                T_ref = T
            trace_V_store.append(V)
            if sim_traces.get("cell_recordings") is not None:
                cell_recordings_store.append(
                    {
                        "trial_idx": trial_idx,
                        "recordings": sim_traces.get("cell_recordings"),
                    }
                )

        if trial_callback is not None:
            try:
                trial_callback(
                    {
                        "trial_idx": trial_idx,
                        "spikes": spikes,
                        "traces": sim_traces,
                        "sim_cfg": sim_cfg_local,
                        "syn_records": syn_state.get("records", {}),
                    }
                )
            except Exception:
                # Do not fail the simulation because a callback failed
                pass

        # Progress log to stdout (captured by SLURM)
        spike_count = len(spikes)
        rate_hz = spike_count / sim_dur_s if sim_dur_s > 0 else 0.0
        elapsed = time.perf_counter() - trial_start
        print(f"[trial {trial_idx+1}/{n_trials}] spikes={spike_count}  rate={rate_hz:.2f} Hz  time={elapsed:.2f}s")

    input_stats = None
    if input_stats_enabled:
        input_stats = {
            "bin_ms": input_bin_width,
            "t_ms": input_centers.tolist(),
            "tstart_ms": tstart,
            "tstop_ms": tstop,
            "trials": input_stats_trials,
            "group_means": _aggregate_input_stats(input_stats_trials),
        }

    traces_out: Dict[str, Any] = {}
    if T_ref is not None and trace_V_store:
        traces_out = {
            "T": T_ref,
            "V": trace_V_store,
        }

    # Compute and store average firing-rate curve (raw, unsmoothed)
    bin_width = float(sim_cfg_local.get("bins", 25.0))
    bins = np.arange(0, tstop + bin_width, bin_width)
    centers = bins[:-1] + 0.5 * bin_width
    bw_s = bin_width / 1000.0
    if spikes_by_trial:
        per_trial_rates = []
        for tr in spikes_by_trial:
            tr = np.asarray(tr)
            counts, _ = np.histogram(tr, bins=bins)
            per_trial_rates.append(counts / bw_s)
        mean_rate = np.mean(per_trial_rates, axis=0)
    else:
        mean_rate = np.array([], dtype=float)

    smooth_ms = sim_cfg_local.get("avg_rate_curve_smooth_ms", 25.0)
    smooth_mode = sim_cfg_local.get("avg_rate_curve_smooth_mode", "center") or "center"
    centers, mean_rate = _smooth_rate_curve(
        centers,
        mean_rate,
        bin_width,
        smooth_ms,
        mode=str(smooth_mode),
    )
    try:
        smooth_ms_val = float(smooth_ms) if smooth_ms is not None else 0.0
    except Exception:
        smooth_ms_val = 0.0

    avg_rate_curve = {
        "bin_ms": bin_width,
        "smooth_ms": smooth_ms_val,
        "smooth_mode": str(smooth_mode),
        "t_ms": centers.tolist(),
        "rate_hz": mean_rate.tolist(),
    }

    result = {
        "mode": "multi",
        "sim_cfg": sim_cfg_local,
        "spikes": spikes_by_trial,
        "traces": traces_out,
        "cell_recordings_by_trial": cell_recordings_store if cell_recordings_store else None,
        "inputs_by_trial": inputs_store if inputs_store else None,
        "syn_records_by_trial": syn_records_by_trial if syn_records_by_trial else None,
        "meta": {
            "cell": sim_cfg_local.get("cell"),
            "tune": sim_cfg_local.get("tune"),
            "n_trials": n_trials,
            "trial_ids": list(range(n_trials)),
            "avg_rate_curve": avg_rate_curve,
            "input_summaries": input_summaries,
            "syn_config": copy.deepcopy(groups_cfg),
        },
    }
    if rm is not None:
        result["meta"]["randomness"] = rm.meta().as_dict()
    if snapshot_enabled:
        result["meta"]["snapshot"] = copy.deepcopy(snapshot_cfg)
        result["meta"]["versions"] = _collect_versions()
        result["meta"]["neuron_state"] = _collect_neuron_state()
        result["meta"]["env"] = _collect_env_snapshot()
        result["meta"]["mechanisms"] = _collect_mechanism_info(sim_cfg_local)
        if netcon_snapshot is not None:
            result["meta"]["netcon_snapshot"] = netcon_snapshot
        if syn_param_snapshot is not None:
            result["meta"]["synapse_param_snapshot"] = syn_param_snapshot
    if input_stats is not None:
        result["meta"]["input_stats"] = input_stats
    return result


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


# ---------------------------------------------------------------------
# saving / loading
# ---------------------------------------------------------------------
from pathlib import Path
import pickle

def _build_output_path(
    sim_cfg: Dict[str, Any],
    base_dir: Union[str, Path] = "output_data",
) -> Optional[Path]:    
    '''
    Build a unique output path based on sim_cfg and base_dir.
    Returns None if sim_cfg['output'] is None/empty.
    1) base_dir / {output_stem}/
    2) filename: {cell}_{tune}_{output_stem}.{suffix}
    3) If run folder exists, append _1, _2, ... to output_stem.
    4) suffix based on sim_cfg['output_format']: 'pickle' -> .pkl
    '''    

    if sim_cfg.get("save_output") is False:
        return None

    output_stem = sim_cfg.get("output_stem")
    if output_stem not in (None, ""):
        sim_cfg["output"] = output_stem
    output_stem = sim_cfg.get("output")
    if not output_stem:
        return None  # don't save if output is None/empty

    cell = sim_cfg.get("cell", "cell")
    tune = sim_cfg.get("tune", "tune")
    fmt  = sim_cfg.get("output_format", "pickle")

    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    if fmt == "npz":
        suffix = ".npz"
    else:
        suffix = ".pkl"

    run_stem = str(output_stem)
    run_dir = base / run_stem
    idx = 1
    while run_dir.exists():
        run_stem = f"{output_stem}_{idx}"
        run_dir = base / run_stem
        idx += 1

    if run_stem != output_stem:
        sim_cfg["output"] = run_stem

    run_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{cell}_{tune}_{run_stem}"
    path = run_dir / (stem + suffix)
    return path


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _save_sidecars(results: Dict[str, Any], run_dir: Path) -> Dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    mode = results.get("mode", "")
    sim_cfg = results.get("sim_cfg", {}) or {}
    meta = copy.deepcopy(results.get("meta", {}) or {})

    files: Dict[str, str] = {}

    _write_json(run_dir / "sim_cfg.json", sim_cfg)
    files["sim_cfg"] = "sim_cfg.json"

    syn_config = meta.pop("syn_config", None)
    cell_config = meta.pop("cell_config", None)
    geometry_config = meta.pop("geometry_config", None)
    avg_rate_curve = meta.pop("avg_rate_curve", None)
    input_stats = meta.pop("input_stats", None)
    input_summaries = meta.pop("input_summaries", None)
    fit_sidecar = _copy_fit_json_sidecar(sim_cfg, run_dir)
    if fit_sidecar is not None:
        meta["fit_json"] = fit_sidecar

    _write_json(run_dir / "meta.json", meta)
    files["meta"] = "meta.json"

    if syn_config is not None:
        _write_json(run_dir / "syn_config.json", syn_config)
        files["syn_config"] = "syn_config.json"
    if cell_config is not None:
        _write_json(run_dir / "cell_config.json", cell_config)
        files["cell_config"] = "cell_config.json"
    if geometry_config is not None:
        _write_json(run_dir / "geometry_config.json", geometry_config)
        files["geometry_config"] = "geometry_config.json"
    if avg_rate_curve is not None:
        _write_json(run_dir / "avg_rate_curve.json", avg_rate_curve)
        files["avg_rate_curve"] = "avg_rate_curve.json"
    if input_stats is not None:
        _write_json(run_dir / "input_stats.json", input_stats)
        files["input_stats"] = "input_stats.json"
    if input_summaries is not None:
        _write_json(run_dir / "input_summaries.json", input_summaries)
        files["input_summaries"] = "input_summaries.json"
    if fit_sidecar is not None:
        files["fit_json"] = fit_sidecar["filename"]

    spikes = results.get("spikes", None)
    if spikes is not None:
        spike_path = run_dir / "spikes.npz"
        if mode == "multi":
            np.savez(spike_path, spikes=np.array(spikes, dtype=object))
        else:
            np.savez(spike_path, spikes=np.asarray(spikes))
        files["spikes"] = "spikes.npz"

    traces = results.get("traces", {}) or {}
    if "T" in traces:
        trace_path = run_dir / "traces.npz"
        if mode == "multi":
            np.savez(
                trace_path,
                T=np.asarray(traces.get("T")),
                V_trials=np.array(traces.get("V", []), dtype=object),
            )
        else:
            np.savez(
                trace_path,
                T=np.asarray(traces.get("T")),
                V=np.asarray(traces.get("V")),
            )
        files["traces"] = "traces.npz"

    cell_recordings = results.get("cell_recordings")
    if cell_recordings is not None:
        cell_rec_path = run_dir / "cell_recordings.pkl"
        with cell_rec_path.open("wb") as f:
            pickle.dump(cell_recordings, f)
        files["cell_recordings"] = "cell_recordings.pkl"

    cell_recordings_by_trial = results.get("cell_recordings_by_trial")
    if cell_recordings_by_trial:
        cell_rec_trial_path = run_dir / "cell_recordings_by_trial.pkl"
        with cell_rec_trial_path.open("wb") as f:
            pickle.dump(cell_recordings_by_trial, f)
        files["cell_recordings_by_trial"] = "cell_recordings_by_trial.pkl"

    inputs_payload = {}
    if results.get("inputs") is not None:
        inputs_payload["inputs"] = results.get("inputs")
    if results.get("inputs_by_trial") is not None:
        inputs_payload["inputs_by_trial"] = results.get("inputs_by_trial")
    if inputs_payload:
        inputs_path = run_dir / "inputs_sample.pkl"
        with inputs_path.open("wb") as f:
            pickle.dump(inputs_payload, f)
        files["inputs_sample"] = "inputs_sample.pkl"

    if sim_cfg.get("save_syn_records_sidecar", True):
        syn_records = results.get("syn_records")
        if syn_records:
            syn_path = run_dir / "syn_records.pkl"
            with syn_path.open("wb") as f:
                pickle.dump(syn_records, f)
            files["syn_records"] = "syn_records.pkl"

    syn_records_by_trial = results.get("syn_records_by_trial")
    if syn_records_by_trial:
        syn_by_trial_path = run_dir / "syn_records_by_trial.pkl"
        with syn_by_trial_path.open("wb") as f:
            pickle.dump(syn_records_by_trial, f)
        files["syn_records_by_trial"] = "syn_records_by_trial.pkl"

    return files



def save_results(
    results: Dict[str, Any],
    base_dir: Union[str, Path] = "output_data",
) -> Optional[Path]:    

    sim_cfg = results.get("sim_cfg", {})
    out_path = _build_output_path(sim_cfg, base_dir=base_dir)
    if out_path is None:
        append_to = sim_cfg.get("append_to")
        if append_to:
            try:
                _append_results_to_path(results, append_to, base_dir=base_dir)
            except Exception as exc:
                print(f"append_to failed: {exc}")
        return None

    fmt = sim_cfg.get("output_format", "pickle")

    run_dir = out_path.parent
    manifest = {
        "format_version": 1,
        "mode": results.get("mode", ""),
        "output_stem": sim_cfg.get("output"),
        "files": {},
    }

    save_sidecars = sim_cfg.get("save_sidecars", True)
    save_full_results = sim_cfg.get("save_full_results", False)
    if not save_sidecars and not save_full_results:
        # ensure at least one artifact is written
        save_sidecars = True
    if save_sidecars:
        manifest["files"].update(_save_sidecars(results, run_dir))

    if save_full_results and fmt == "npz":
        # compact, interoperable: arrays + JSON metadata
        mode = results.get("mode", "")
        meta = results.get("meta", {})
        traces = results.get("traces", {}) or {}
        spikes = results.get("spikes", None)

        payload = {
            "mode": np.array(mode),
            "sim_cfg_json": np.array(json.dumps(sim_cfg, default=_json_default)),
            "meta_json": np.array(json.dumps(meta, default=_json_default)),
        }

        if "T" in traces:
            payload["T"] = traces["T"]

        if mode == "single":
            if "V" in traces:
                payload["V"] = traces["V"]
            if spikes is not None:
                payload["spikes"] = spikes
            if results.get("cell_recordings") is not None:
                payload["cell_recordings"] = np.array(results.get("cell_recordings"), dtype=object)
        elif mode == "multi":
            if spikes is not None:
                payload["spikes"] = np.array(spikes, dtype=object)
            if "V" in traces:
                payload["V_trials"] = np.array(traces["V"], dtype=object)
            if results.get("cell_recordings_by_trial") is not None:
                payload["cell_recordings_by_trial"] = np.array(
                    results.get("cell_recordings_by_trial"), dtype=object
                )

        np.savez(out_path, **payload)
        manifest["files"]["results_npz"] = out_path.name
    elif save_full_results:
        # full Python dict with everything
        with out_path.open("wb") as f:
            pickle.dump(results, f)
        manifest["files"]["results_pkl"] = out_path.name

    _write_json(run_dir / "run_manifest.json", manifest)

    # Optional: auto-save plots into run_dir/plots
    if sim_cfg.get("save_plots", False):
        try:
            from modules_local.analysis import analysis as analysis_mod

            analysis_mod.save_default_plots(
                results,
                run_dir,
                save_inputs=bool(sim_cfg.get("save_plots_inputs", True)),
                save_synapses=bool(sim_cfg.get("save_plots_synapses", False)),
                win_size=float(sim_cfg.get("plots_win_size", 25.0)),
                input_bin_ms=sim_cfg.get("plots_input_bin_ms", None),
                input_smooth_ms=sim_cfg.get("plots_input_smooth_ms", 25.0),
                raster_style=str(sim_cfg.get("plots_raster_style", "dot")),
                plot_mode=str(sim_cfg.get("save_plots_mode", "default")),
                single_plot_preset=sim_cfg.get("save_plots_single_plot_preset", None),
            )
        except Exception as exc:
            print(f"save_plots failed: {exc}")

    append_to = sim_cfg.get("append_to")
    if sim_cfg.get("append_enabled", True) and append_to:
        try:
            _append_results_to_path(results, append_to, base_dir=run_dir.parent)
        except Exception as exc:
            print(f"append_to failed: {exc}")

    return out_path if save_full_results else (run_dir / "run_manifest.json")

def save_results_with_name(
    results: Dict[str, Any],
    output_stem: str,
    base_dir: Union[str, Path] = "output_data",
) -> Optional[Path]:
    """
    Manually save an existing results dict under a given output name,
    regardless of what was set in sim_cfg['output'] originally.

    Example:
        results["sim_cfg"]["color"] = "m"
        save_results_with_name(results, "sst2_seg_tuned_batch1")
    """
    sim_cfg = results.setdefault("sim_cfg", {})
    sim_cfg["output"] = str(output_stem)
    return save_results(results, base_dir=base_dir)


def _ensure_multi_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce a single-trial results dict into multi-trial format for appending.
    """
    mode = results.get("mode", "single")
    if mode == "multi":
        return results
    if mode != "single":
        raise ValueError(f"append_to: unsupported results mode {mode!r}")

    sim_cfg = copy.deepcopy(results.get("sim_cfg", {}))
    traces = results.get("traces", {}) or {}
    inputs = results.get("inputs", None)
    cell_recordings = results.get("cell_recordings", None)

    spikes = results.get("spikes", None)
    spikes_list = [spikes] if spikes is not None else []

    multi = {
        "mode": "multi",
        "sim_cfg": sim_cfg,
        "spikes": spikes_list,
        "traces": {},
        "cell_recordings_by_trial": None,
        "inputs_by_trial": None,
        "meta": copy.deepcopy(results.get("meta", {})),
    }

    if "T" in traces and "V" in traces:
        multi["traces"] = {"T": traces["T"], "V": [traces["V"]]}

    if inputs:
        multi["inputs_by_trial"] = [{"trial_idx": 0, "inputs": inputs}]
    if cell_recordings is not None:
        multi["cell_recordings_by_trial"] = [
            {"trial_idx": 0, "recordings": cell_recordings}
        ]

    meta = multi.setdefault("meta", {})
    meta["n_trials"] = len(spikes_list)
    meta["trial_ids"] = list(range(len(spikes_list)))
    return multi


def _write_results_file(results: Dict[str, Any], out_path: Path, fmt: str) -> None:
    fmt = str(fmt).lower()
    if fmt == "npz":
        mode = results.get("mode", "")
        sim_cfg = results.get("sim_cfg", {})
        meta = results.get("meta", {})
        traces = results.get("traces", {}) or {}
        spikes = results.get("spikes", None)

        payload = {
            "mode": np.array(mode),
            "sim_cfg_json": np.array(json.dumps(sim_cfg, default=_json_default)),
            "meta_json": np.array(json.dumps(meta, default=_json_default)),
        }

        if "T" in traces:
            payload["T"] = traces["T"]

        if mode == "single":
            if "V" in traces:
                payload["V"] = traces["V"]
            if spikes is not None:
                payload["spikes"] = spikes
            if results.get("cell_recordings") is not None:
                payload["cell_recordings"] = np.array(results.get("cell_recordings"), dtype=object)
        elif mode == "multi":
            if spikes is not None:
                payload["spikes"] = np.array(spikes, dtype=object)
            if "V" in traces:
                payload["V_trials"] = np.array(traces["V"], dtype=object)
            if results.get("cell_recordings_by_trial") is not None:
                payload["cell_recordings_by_trial"] = np.array(
                    results.get("cell_recordings_by_trial"), dtype=object
                )

        np.savez(out_path, **payload)
    else:
        with out_path.open("wb") as f:
            pickle.dump(results, f)


def _write_results_to_run_dir(
    results: Dict[str, Any],
    run_dir: Path,
    *,
    fmt: Optional[str] = None,
    results_name: Optional[str] = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    sim_cfg = results.get("sim_cfg", {}) or {}

    if fmt is None:
        fmt = sim_cfg.get("output_format", "pickle")
    fmt = str(fmt).lower()

    if results_name is None:
        cell = sim_cfg.get("cell", "cell")
        tune = sim_cfg.get("tune", "tune")
        stem = sim_cfg.get("output") or "results"
        suffix = ".npz" if fmt == "npz" else ".pkl"
        results_name = f"{cell}_{tune}_{stem}{suffix}"

    out_path = run_dir / results_name

    manifest = {
        "format_version": 1,
        "mode": results.get("mode", ""),
        "output_stem": sim_cfg.get("output"),
        "files": {},
    }

    save_sidecars = sim_cfg.get("save_sidecars", True)
    if save_sidecars:
        manifest["files"].update(_save_sidecars(results, run_dir))

    if fmt == "npz":
        _write_results_file(results, out_path, "npz")
        manifest["files"]["results_npz"] = out_path.name
    else:
        _write_results_file(results, out_path, "pickle")
        manifest["files"]["results_pkl"] = out_path.name

    _write_json(run_dir / "run_manifest.json", manifest)
    return out_path


def _append_results_to_path(
    results: Dict[str, Any],
    append_to: Union[str, Path],
    *,
    base_dir: Union[str, Path, None] = None,
) -> Optional[Path]:
    """
    Append results to an existing file/folder (or create a new one).
    """
    append_path = Path(append_to)
    if not append_path.is_absolute() and base_dir is not None:
        base_dir = Path(base_dir)
        if append_path.parts and append_path.parts[0] == base_dir.name:
            append_path = base_dir / Path(*append_path.parts[1:])
        else:
            append_path = base_dir / append_path

    # Normalize run_manifest.json to its parent folder
    if append_path.name == "run_manifest.json":
        append_path = append_path.parent

    if append_path.exists():
        base_results = load_results(append_path)
        merged = append_multi_results(
            _ensure_multi_results(base_results),
            _ensure_multi_results(results),
        )
    else:
        merged = _ensure_multi_results(results)

    # Decide where/how to write
    if append_path.suffix.lower() in (".pkl", ".npz"):
        append_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = "npz" if append_path.suffix.lower() == ".npz" else "pickle"
        _write_results_file(merged, append_path, fmt)
        return append_path

    # Directory target (existing or new)
    fmt = None
    results_name = None
    manifest = append_path / "run_manifest.json"
    if manifest.is_file():
        try:
            old_manifest = json.loads(manifest.read_text())
            files = old_manifest.get("files", {}) or {}
            if files.get("results_npz"):
                fmt = "npz"
                results_name = files.get("results_npz")
            elif files.get("results_pkl"):
                fmt = "pickle"
                results_name = files.get("results_pkl")
        except Exception:
            fmt = None
            results_name = None

    return _write_results_to_run_dir(merged, append_path, fmt=fmt, results_name=results_name)


def append_multi_results(
    base_results: Dict[str, Any],
    new_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge two multi-trial results dicts in memory.

    - Requires both mode == 'multi'.
    - Concatenates spike trains.
    - Updates sim_cfg['n_trials'].
    - If both have V-traces with matching T, concatenates those too.

    Returns a NEW dict; does not modify inputs in-place.
    """
    if base_results.get("mode") != "multi":
        raise ValueError("append_multi_results: base_results.mode must be 'multi'")
    if new_results.get("mode") != "multi":
        raise ValueError("append_multi_results: new_results.mode must be 'multi'")

    merged = copy.deepcopy(base_results)
    meta = merged.setdefault("meta", {})
    warnings = meta.setdefault("append_warnings", [])

    base_spikes = list(merged.get("spikes", []) or [])
    new_spikes  = list(new_results.get("spikes", []) or [])
    base_trial_count = len(base_spikes)
    merged_spikes = base_spikes + new_spikes
    merged["spikes"] = merged_spikes

    # update n_trials
    sim_cfg = merged.setdefault("sim_cfg", {})
    n_trials = len(merged_spikes)
    sim_cfg["n_trials"] = n_trials
    meta["n_trials"] = n_trials
    meta["trial_ids"] = list(range(n_trials))

    # try to merge stored Vm traces if time axes match
    base_traces = merged.get("traces", {}) or {}
    new_traces  = new_results.get("traces", {}) or {}
    T_base = base_traces.get("T")
    T_new  = new_traces.get("T")
    max_traces = _resolve_trace_trials_to_save(sim_cfg, fallback=1)
    max_inputs = _resolve_inputs_to_save(sim_cfg, n_trials, max_traces)

    if max_traces <= 0:
        merged["traces"] = {}
    elif T_base is not None and T_new is not None:
        try:
            if np.allclose(np.asarray(T_base), np.asarray(T_new)):
                V_base = list(base_traces.get("V", []) or [])
                V_new  = list(new_traces.get("V", []) or [])
                base_traces["V"] = (V_base + V_new)[:max_traces]
                merged["traces"] = base_traces
            else:
                warnings.append("trace_time_mismatch: skipped merging traces")
        except Exception:
            warnings.append("trace_merge_failed: skipped merging traces")

    def _offset_trial_idx(entry: Dict[str, Any], offset: int, fallback: int) -> None:
        if "trial_idx" in entry:
            try:
                entry["trial_idx"] = offset + int(entry["trial_idx"])
                return
            except Exception:
                pass
        entry["trial_idx"] = offset + fallback

    def _merge_inputs_by_trial() -> None:
        if max_inputs <= 0:
            merged["inputs_by_trial"] = None
            return
        dst_inputs = merged.get("inputs_by_trial") or []
        src_inputs = new_results.get("inputs_by_trial") or []
        if not src_inputs:
            merged["inputs_by_trial"] = dst_inputs if dst_inputs else None
            return
        for idx, entry in enumerate(src_inputs):
            if len(dst_inputs) >= max_inputs:
                break
            new_entry = copy.deepcopy(entry)
            if isinstance(new_entry, dict):
                _offset_trial_idx(new_entry, base_trial_count, idx)
            dst_inputs.append(new_entry)
        merged["inputs_by_trial"] = dst_inputs if dst_inputs else None

    def _merge_cell_recordings_by_trial() -> None:
        if max_traces <= 0:
            merged["cell_recordings_by_trial"] = None
            return
        dst_recs = list(merged.get("cell_recordings_by_trial") or [])[:max_traces]
        src_recs = new_results.get("cell_recordings_by_trial") or []
        if not src_recs:
            merged["cell_recordings_by_trial"] = dst_recs if dst_recs else None
            return
        for idx, entry in enumerate(src_recs):
            if len(dst_recs) >= max_traces:
                break
            new_entry = copy.deepcopy(entry)
            if isinstance(new_entry, dict):
                _offset_trial_idx(new_entry, base_trial_count, idx)
            dst_recs.append(new_entry)
        merged["cell_recordings_by_trial"] = dst_recs if dst_recs else None

    def _merge_input_summaries() -> None:
        src_meta = new_results.get("meta", {}) or {}
        src_summaries = src_meta.get("input_summaries") or []
        if not src_summaries:
            return
        dst_summaries = meta.get("input_summaries") or []
        for idx, entry in enumerate(src_summaries):
            new_entry = copy.deepcopy(entry)
            if isinstance(new_entry, dict):
                _offset_trial_idx(new_entry, base_trial_count, idx)
            dst_summaries.append(new_entry)
        meta["input_summaries"] = dst_summaries

    def _stats_compatible(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        keys = ("bin_ms", "tstart_ms", "tstop_ms")
        for key in keys:
            if a.get(key) != b.get(key):
                return False
        a_t = a.get("t_ms") or []
        b_t = b.get("t_ms") or []
        try:
            return np.allclose(np.asarray(a_t, dtype=float), np.asarray(b_t, dtype=float))
        except Exception:
            return False

    def _merge_input_stats() -> None:
        dst_stats = meta.get("input_stats")
        src_stats = (new_results.get("meta", {}) or {}).get("input_stats")
        if not src_stats:
            return
        if dst_stats is None:
            dst_stats = copy.deepcopy(src_stats)
            for idx, entry in enumerate(dst_stats.get("trials", []) or []):
                if isinstance(entry, dict):
                    _offset_trial_idx(entry, base_trial_count, idx)
            dst_stats["group_means"] = _aggregate_input_stats(dst_stats.get("trials", []))
            meta["input_stats"] = dst_stats
            return
        if not _stats_compatible(dst_stats, src_stats):
            warnings.append("input_stats_mismatch: skipped merging input_stats")
            return
        for idx, entry in enumerate(src_stats.get("trials", []) or []):
            new_entry = copy.deepcopy(entry)
            if isinstance(new_entry, dict):
                _offset_trial_idx(new_entry, base_trial_count, idx)
            dst_stats["trials"].append(new_entry)
        dst_stats["group_means"] = _aggregate_input_stats(dst_stats.get("trials", []))
        meta["input_stats"] = dst_stats

    def _recompute_avg_rate() -> None:
        tstop = float(sim_cfg.get("tstop", 0.0))
        bin_width = float(sim_cfg.get("bins", 25.0) or 25.0)
        if tstop <= 0 or bin_width <= 0:
            warnings.append("avg_rate_skipped: invalid tstop/bin width")
            return
        bins = np.arange(0, tstop + bin_width, bin_width)
        centers = bins[:-1] + 0.5 * bin_width
        bw_s = bin_width / 1000.0
        if merged_spikes:
            per_trial_rates = []
            for tr in merged_spikes:
                tr = np.asarray(tr)
                counts, _ = np.histogram(tr, bins=bins)
                per_trial_rates.append(counts / bw_s)
            mean_rate = np.mean(per_trial_rates, axis=0)
        else:
            mean_rate = np.array([], dtype=float)
        smooth_ms = sim_cfg.get("avg_rate_curve_smooth_ms", 25.0)
        smooth_mode = sim_cfg.get("avg_rate_curve_smooth_mode", "center") or "center"
        centers, mean_rate = _smooth_rate_curve(
            centers,
            mean_rate,
            bin_width,
            smooth_ms,
            mode=str(smooth_mode),
        )
        try:
            smooth_ms_val = float(smooth_ms) if smooth_ms is not None else 0.0
        except Exception:
            smooth_ms_val = 0.0
        meta["avg_rate_curve"] = {
            "bin_ms": bin_width,
            "smooth_ms": smooth_ms_val,
            "smooth_mode": str(smooth_mode),
            "t_ms": centers.tolist(),
            "rate_hz": mean_rate.tolist(),
        }

    # sanity checks on sim_cfg compatibility
    for key in ("tstart", "tstop", "dt", "bins"):
        if key in new_results.get("sim_cfg", {}) and new_results["sim_cfg"].get(key) != sim_cfg.get(key):
            warnings.append(f"sim_cfg_mismatch:{key}")

    if meta.get("syn_config") and (new_results.get("meta", {}) or {}).get("syn_config"):
        if meta.get("syn_config") != new_results["meta"].get("syn_config"):
            warnings.append("syn_config_mismatch")

    # annotate meta
    appended = len(new_spikes)
    meta["appended_trials"] = meta.get("appended_trials", 0) + appended
    _merge_inputs_by_trial()
    _merge_cell_recordings_by_trial()
    _merge_input_summaries()
    _merge_input_stats()
    _recompute_avg_rate()

    return merged



def _load_from_manifest(manifest_path: Path) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    run_dir = manifest_path.parent
    files = manifest.get("files", {}) or {}

    pkl_name = files.get("results_pkl")
    if pkl_name:
        pkl_path = run_dir / pkl_name
        if pkl_path.is_file():
            return load_results(pkl_path)

    npz_name = files.get("results_npz")
    if npz_name:
        npz_path = run_dir / npz_name
        if npz_path.is_file():
            return load_results(npz_path)

    mode = manifest.get("mode", "single")

    sim_cfg = {}
    if files.get("sim_cfg"):
        sim_cfg = json.loads((run_dir / files["sim_cfg"]).read_text())

    meta = {}
    if files.get("meta"):
        meta = json.loads((run_dir / files["meta"]).read_text())

    if files.get("syn_config"):
        meta["syn_config"] = json.loads((run_dir / files["syn_config"]).read_text())
    if files.get("cell_config"):
        meta["cell_config"] = json.loads((run_dir / files["cell_config"]).read_text())
    if files.get("geometry_config"):
        meta["geometry_config"] = json.loads((run_dir / files["geometry_config"]).read_text())
    if files.get("avg_rate_curve"):
        meta["avg_rate_curve"] = json.loads((run_dir / files["avg_rate_curve"]).read_text())
    if files.get("input_stats"):
        meta["input_stats"] = json.loads((run_dir / files["input_stats"]).read_text())
    if files.get("input_summaries"):
        meta["input_summaries"] = json.loads((run_dir / files["input_summaries"]).read_text())

    spikes = None
    if files.get("spikes"):
        sp = np.load(run_dir / files["spikes"], allow_pickle=True)
        if "spikes" in sp.files:
            spikes = sp["spikes"]
            if mode == "multi":
                spikes = list(spikes)

    traces: Dict[str, Any] = {}
    if files.get("traces"):
        tr = np.load(run_dir / files["traces"], allow_pickle=True)
        if "T" in tr.files:
            traces["T"] = tr["T"]
        if mode == "multi":
            if "V_trials" in tr.files:
                traces["V"] = list(tr["V_trials"])
        else:
            if "V" in tr.files:
                traces["V"] = tr["V"]

    cell_recordings = None
    if files.get("cell_recordings"):
        with (run_dir / files["cell_recordings"]).open("rb") as f:
            cell_recordings = pickle.load(f)
    cell_recordings_by_trial = None
    if files.get("cell_recordings_by_trial"):
        with (run_dir / files["cell_recordings_by_trial"]).open("rb") as f:
            cell_recordings_by_trial = pickle.load(f)

    inputs_payload = None
    if files.get("inputs_sample"):
        with (run_dir / files["inputs_sample"]).open("rb") as f:
            inputs_payload = pickle.load(f)

    syn_records = None
    if files.get("syn_records"):
        with (run_dir / files["syn_records"]).open("rb") as f:
            syn_records = pickle.load(f)
    syn_records_by_trial = None
    if files.get("syn_records_by_trial"):
        with (run_dir / files["syn_records_by_trial"]).open("rb") as f:
            syn_records_by_trial = pickle.load(f)

    results = {
        "mode": mode,
        "sim_cfg": sim_cfg,
        "traces": traces,
        "spikes": spikes,
        "meta": meta,
    }
    if inputs_payload:
        if "inputs" in inputs_payload:
            results["inputs"] = inputs_payload["inputs"]
        if "inputs_by_trial" in inputs_payload:
            results["inputs_by_trial"] = inputs_payload["inputs_by_trial"]
    if syn_records is not None:
        results["syn_records"] = syn_records
    if syn_records_by_trial is not None:
        results["syn_records_by_trial"] = syn_records_by_trial
    if cell_recordings is not None:
        results["cell_recordings"] = cell_recordings
    if cell_recordings_by_trial is not None:
        results["cell_recordings_by_trial"] = cell_recordings_by_trial

    return results


def load_results(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        manifest = p / "run_manifest.json"
        if manifest.is_file():
            return _load_from_manifest(manifest)
        results_dir = p / "results"
        if results_dir.is_dir():
            return load_results(results_dir)
        candidates = list(p.glob("*.pkl")) + list(p.glob("*.npz"))
        if len(candidates) == 1:
            return load_results(candidates[0])
        raise FileNotFoundError(f"No run_manifest.json or single results file in {p}")

    if p.name == "run_manifest.json":
        return _load_from_manifest(p)

    suffix = p.suffix.lower()

    if suffix == ".npz":
        data = np.load(p, allow_pickle=True)

        mode = str(data["mode"])
        sim_cfg = json.loads(str(data["sim_cfg_json"]))
        meta = json.loads(str(data["meta_json"]))

        traces: Dict[str, Any] = {}
        spikes = None
        cell_recordings = None
        cell_recordings_by_trial = None

        if "T" in data.files:
            traces["T"] = data["T"]

        if mode == "single":
            if "V" in data.files:
                traces["V"] = data["V"]
            if "spikes" in data.files:
                spikes = data["spikes"]
            if "cell_recordings" in data.files:
                try:
                    cell_recordings = data["cell_recordings"].item()
                except Exception:
                    cell_recordings = data["cell_recordings"]
        elif mode == "multi":
            if "V_trials" in data.files:
                traces["V"] = list(data["V_trials"])
            if "spikes" in data.files:
                spikes = list(data["spikes"])
            if "cell_recordings_by_trial" in data.files:
                cell_recordings_by_trial = list(data["cell_recordings_by_trial"])

        out = {
            "mode": mode,
            "sim_cfg": sim_cfg,
            "traces": traces,
            "spikes": spikes,
            "meta": meta,
        }
        if cell_recordings is not None:
            out["cell_recordings"] = cell_recordings
        if cell_recordings_by_trial is not None:
            out["cell_recordings_by_trial"] = cell_recordings_by_trial
        return out

    with p.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "mode" in payload and "sim_cfg" in payload:
        return payload
    try:
        return load_old_multi_results(p)
    except Exception:
        return payload
        

# ---------------------------------------------------------------------
# Compatibility loader for old multi-trial pickle outputs
# ---------------------------------------------------------------------
from pathlib import Path
import pickle
from typing import Union

def load_old_multi_results(
    path: Union[str, Path],
    *,
    label: str = None,
    color: str = None,
    tstop: float = 1200.0,
    bins: float = 5.0,
    delay: float = 0.0,
) -> Dict[str, Any]:
    """
    Load an old multi-trial pickle (e.g. tune1_1000tr1200ms.pkl) and
    wrap it into a new-style `results` dict that `plot_results` can use.

    Parameters
    ----------
    path : str or Path
        Path to the old .pkl file.
    label : str, optional
        Which key in all_param_data to use (e.g. 'base tune').
        If None, use 'base tune' if present, otherwise the first key.
    color : str, optional
        Optional plotting color; stored into sim_cfg['color'].
    tstop, bins, delay : float
        Timing parameters in ms, used to build sim_cfg for plotting.

    Returns
    -------
    results : dict
        New-style results dict with:
          - mode='multi'
          - sim_cfg: contains tstop, bins, delay, n_trials, color, ...
          - spikes: list of spike-time arrays (one per trial)
          - traces: {}
          - meta: includes 'source' and 'label'
    """
    p = Path(path)
    with p.open("rb") as f:
        payload = pickle.load(f)

    def _load_json_cfg(cfg_path: Path) -> Optional[Dict[str, Any]]:
        if not cfg_path.is_file():
            return None
        try:
            return json.loads(cfg_path.read_text())
        except Exception:
            return None

    def _find_tune_cfg(pkl_path: Path) -> Optional[Dict[str, Any]]:
        for parent in pkl_path.parents:
            if parent.name == "output_data":
                tune_dir = parent.parent
                for cfg_path in (
                    tune_dir / "cell_configs" / "sim_config.json",
                    tune_dir / "sim_config.json",
                    tune_dir / "cell_config.json",
                ):
                    cfg = _load_json_cfg(cfg_path)
                    if cfg:
                        return cfg
                break
        return None

    if color is None:
        for cfg_path in (
            p.parent / "cell_config.json",
            p.parent / "sim_cfg.json",
            p.parent / "sim_config.json",
            p.parent.parent / "sim_cfg.json",
            p.parent.parent / "sim_config.json",
            p.parent.parent / "cell_config.json",
        ):
            if not cfg_path.is_file():
                continue
            try:
                cfg = json.loads(cfg_path.read_text())
            except Exception:
                continue
            cfg_color = (cfg or {}).get("color")
            if cfg_color:
                color = cfg_color
                break

    # Try to detect shape:
    #  Case 1: {'all_param_data': {...}, 'param_study': ..., 'sim_params': ...}
    if isinstance(payload, dict) and "all_param_data" in payload:
        all_param_data = payload["all_param_data"]
        param_study_old = payload.get("param_study", {})
        sim_params_old = payload.get("sim_params", {})
    else:
        # Case 2: assume payload itself is the all_param_data dict
        all_param_data = payload
        param_study_old = {}
        sim_params_old = {}

    # Choose which group to use (e.g. 'base tune')
    keys = list(all_param_data.keys())
    if not keys:
        raise ValueError(f"Old results file {p} has no parameter groups.")

    if label is None:
        if "base tune" in all_param_data:
            label = "base tune"
        else:
            label = keys[0]

    if label not in all_param_data:
        raise KeyError(f"Label {label!r} not found in old all_param_data; "
                       f"available keys = {keys}")

    spikes_by_trial = all_param_data[label]
    n_trials = len(spikes_by_trial)

    # Build a minimal sim_cfg suitable for plotting
    sim_cfg = {
        "dt":      float(sim_params_old.get("dt", 0.025)),
        "tstart":  float(sim_params_old.get("tstart", 0.0)),
        "tstop":   float(sim_params_old.get("tstop", tstop)),
        "bins":    float(sim_params_old.get("bins", bins)),
        "delay":   float(sim_params_old.get("delay", delay)),
        "n_trials": n_trials,
        "n_traces_to_save": 0,
        "color":   color,
        "param_study": {
            "input_type": None,
            "param_type": label,
            "param_vals": [None],
            "n_trials": n_trials,
        },
        "output": None,
        "output_format": "pickle",
        "cell": None,
        "tune": None,
        "jitter": None,
        "seed": None,
        "trial_randomness": "synapses",
    }

    tune_cfg = _find_tune_cfg(p)
    if tune_cfg:
        for key in ("stim_start_ms", "stim_duration_ms", "stim_stop_ms", "delay"):
            if sim_cfg.get(key) is None and tune_cfg.get(key) is not None:
                sim_cfg[key] = tune_cfg.get(key)

    results = {
        "mode": "multi",
        "sim_cfg": sim_cfg,
        "spikes": spikes_by_trial,
        "traces": {},  # no Vm available in old file
        "meta": {
            "source": "old_pipeline",
            "label": label,
            "path": str(p),
        },
    }
    return results
