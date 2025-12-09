from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Mapping, Tuple
import json, copy, os, pickle, math, random
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
    

def run_current_injection(cell,sim_params,):
    
    # from neuron import h
    # h.load_file("stdrun.hoc")

    stim = h.IClamp(cell.soma[0](0.5))
    stim.amp = sim_params['stim_amp']
    stim.delay = sim_params['stim_delay']
    stim.dur = sim_params['stim_dur']
    h.tstop = sim_params['h_tstop']
    h.dt = sim_params['h_dt']
    # h.steps_per_ms = 1 / h.dt
    # return h, stim

    recorded_data = {}

    # Attach recorders
    vvec = h.Vector().record(cell.soma[0](0.5)._ref_v)
    tvec = h.Vector().record(h._ref_t)

    # Attach *all* the current recorders in that section
    current_recording_vars = get_rec_vars_for_i_in_sec(cell.soma[0], 0.5)

    # Run
    h.finitialize()
    h.run()

    # Stash numpy arrays
    recorded_data['T'] = np.array(tvec)                # Time (same for all sims)
    recorded_data['V'] = np.array(vvec)                   # Voltage
    recorded_data['F'] = get_frequency(vvec, sim_params)
    recorded_data['I'] = {name: vec.as_numpy().copy() for name, vec in current_recording_vars.items()} #ivecs

    return recorded_data

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
    plt.ylim(-0.01,0.01)
    plt.title(f"{cell_name} currents @ {trial_amp} pA")
    plt.legend() #loc='upper right')
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


def _detect_spikes(T: np.ndarray, V: np.ndarray, v_thresh: float = 0.0) -> np.ndarray:
    """
    Simple spike detector: returns times where V crosses v_thresh from below.
    This is intentionally minimal and can be replaced later with a better detector.
    """
    above = V > v_thresh
    crossings = np.where(above[1:] & ~above[:-1])[0] + 1
    return T[crossings]


# ---------------------------------------------------------------------
# core run functions
# ---------------------------------------------------------------------

def run_single(
    cell: Any,
    geom: Any,
    sim_cfg: Dict[str, Any],
    groups_cfg: Dict[str, Any],
    inputs_by_group: Dict[str, Any],
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
    n_traces_to_save = int(sim_cfg_local.get("n_traces_to_save", 1))

    # reset cell state and attach synapses
    _clear_cell_state(cell)
    syn_state = synapses.add_synapses(cell, geom, sim_cfg_local, groups_cfg, inputs_by_group)
    syn_records = syn_state.get("records", {})

    # run the actual simulation (existing 3.1 primitive)
    sim_traces = run_cell(cell, sim_cfg_local)  # assumes this is defined below / in this module

    T = np.asarray(sim_traces.get("T", []), dtype=float)
    V = np.asarray(sim_traces.get("V", []), dtype=float)
    spikes = _detect_spikes(T, V) if T.size and V.size else np.array([], dtype=float)

    traces_out: Dict[str, Any] = {}
    if n_traces_to_save > 0 and T.size and V.size:
        traces_out = {"T": T, "V": V}

    result = {
        "mode": "single",
        "sim_cfg": sim_cfg_local,
        "spikes": spikes,
        "traces": traces_out,
        "syn_records": syn_records,
        "meta": {
            "cell": sim_cfg_local.get("cell"),
            "tune": sim_cfg_local.get("tune"),
            "n_trials": 1,
        },
    }
    return result


def run_multi(
    cell: Any,
    geom: Any,
    sim_cfg: Dict[str, Any],
    groups_cfg: Dict[str, Any],
    inputs_by_group: Dict[str, Any],
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
           "V": [np.ndarray, ...]         # up to n_traces_to_save entries
        } or {},
        "meta": {
           "n_trials": int,
           "trial_ids": [0, 1, ...]
        }
      }
    """
    sim_cfg_local = copy.deepcopy(sim_cfg)
    n_trials = int(sim_cfg_local.get("n_trials", 1))
    n_traces_to_save = int(sim_cfg_local.get("n_traces_to_save", 1))

    spikes_by_trial: List[np.ndarray] = []
    trace_V_store: List[np.ndarray] = []
    T_ref: Optional[np.ndarray] = None

    for trial_idx in range(n_trials):
        _clear_cell_state(cell)
        syn_state = synapses.add_synapses(cell, geom, sim_cfg_local, groups_cfg, inputs_by_group)

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

    traces_out: Dict[str, Any] = {}
    if T_ref is not None and trace_V_store:
        traces_out = {
            "T": T_ref,
            "V": trace_V_store,
        }

    result = {
        "mode": "multi",
        "sim_cfg": sim_cfg_local,
        "spikes": spikes_by_trial,
        "traces": traces_out,
        "meta": {
            "cell": sim_cfg_local.get("cell"),
            "tune": sim_cfg_local.get("tune"),
            "n_trials": n_trials,
            "trial_ids": list(range(n_trials)),
        },
    }
    return result


def run_param(
    cell: Any,
    geom: Any,
    sim_cfg: Dict[str, Any],
    groups_cfg: Dict[str, Any],
    inputs_by_group: Dict[str, Any],
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
    if load_target:
        p = Path(load_target)
        if not p.is_absolute():
            p = Path("output_data") / p  # interpret as relative to output_data/
        result = load_results(p)
        meta = result.get("meta", {})
        meta["loaded_from"] = str(p)
        result["meta"] = meta
        return result
    

    mode = _infer_mode(sim_cfg_local)

    if mode == "single":
        result = run_single(cell, geom, sim_cfg_local, groups_cfg, inputs_by_group)
    elif mode == "multi":
        result = run_multi(cell, geom, sim_cfg_local, groups_cfg, inputs_by_group)
    elif mode == "param":
        result = run_param(cell, geom, sim_cfg_local, groups_cfg, inputs_by_group)
    else:
        raise ValueError(f"run_sim: unrecognized mode '{mode}'")

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
    1) base_dir / {cell}_{tune}_{output_stem}.{suffix}
    2) If file exists, append _1, _2, ... until unique.
    3) suffix based on sim_cfg['output_format']: 'pickle' -> .p
    '''    

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

    stem = f"{cell}_{tune}_{output_stem}"
    path = base / (stem + suffix)
    idx = 1
    while path.exists():
        path = base / f"{stem}_{idx}{suffix}"
        idx += 1

    return path



def save_results(
    results: Dict[str, Any],
    base_dir: Union[str, Path] = "output_data",
) -> Optional[Path]:    

    sim_cfg = results.get("sim_cfg", {})
    out_path = _build_output_path(sim_cfg, base_dir=base_dir)
    if out_path is None:
        return None

    fmt = sim_cfg.get("output_format", "pickle")

    if fmt == "npz":
        # compact, interoperable: arrays + JSON metadata
        mode = results.get("mode", "")
        meta = results.get("meta", {})
        traces = results.get("traces", {}) or {}
        spikes = results.get("spikes", None)

        payload = {
            "mode": np.array(mode),
            "sim_cfg_json": np.array(json.dumps(sim_cfg)),
            "meta_json": np.array(json.dumps(meta)),
        }

        if "T" in traces:
            payload["T"] = traces["T"]

        if mode == "single":
            if "V" in traces:
                payload["V"] = traces["V"]
            if spikes is not None:
                payload["spikes"] = spikes
        elif mode == "multi":
            if spikes is not None:
                payload["spikes"] = np.array(spikes, dtype=object)
            if "V" in traces:
                payload["V_trials"] = np.array(traces["V"], dtype=object)

        np.savez(out_path, **payload)

    else:
        # full Python dict with everything
        with out_path.open("wb") as f:
            pickle.dump(results, f)

    return out_path

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

    base_spikes = list(merged.get("spikes", []) or [])
    new_spikes  = list(new_results.get("spikes", []) or [])
    merged_spikes = base_spikes + new_spikes
    merged["spikes"] = merged_spikes

    # update n_trials
    sim_cfg = merged.setdefault("sim_cfg", {})
    sim_cfg["n_trials"] = len(merged_spikes)

    # try to merge stored Vm traces if time axes match
    base_traces = merged.get("traces", {}) or {}
    new_traces  = new_results.get("traces", {}) or {}
    T_base = base_traces.get("T")
    T_new  = new_traces.get("T")

    if T_base is not None and T_new is not None:
        try:
            if np.allclose(np.asarray(T_base), np.asarray(T_new)):
                V_base = list(base_traces.get("V", []) or [])
                V_new  = list(new_traces.get("V", []) or [])
                base_traces["V"] = V_base + V_new
                merged["traces"] = base_traces
        except Exception:
            # if anything is weird, just leave traces as base_results'
            pass

    # annotate meta
    meta = merged.setdefault("meta", {})
    appended = len(new_spikes)
    meta["appended_trials"] = meta.get("appended_trials", 0) + appended

    return merged



def load_results(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".npz":
        data = np.load(p, allow_pickle=True)

        mode = str(data["mode"])
        sim_cfg = json.loads(str(data["sim_cfg_json"]))
        meta = json.loads(str(data["meta_json"]))

        traces: Dict[str, Any] = {}
        spikes = None

        if "T" in data.files:
            traces["T"] = data["T"]

        if mode == "single":
            if "V" in data.files:
                traces["V"] = data["V"]
            if "spikes" in data.files:
                spikes = data["spikes"]
        elif mode == "multi":
            if "V_trials" in data.files:
                traces["V"] = list(data["V_trials"])
            if "spikes" in data.files:
                spikes = list(data["spikes"])

        return {
            "mode": mode,
            "sim_cfg": sim_cfg,
            "traces": traces,
            "spikes": spikes,
            "meta": meta,
        }

    else:
        with p.open("rb") as f:
            return pickle.load(f)
        

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
    bins: float = 25.0,
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
