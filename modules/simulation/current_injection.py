from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from neuron import h

from modules.model.geometry import cell_soma_segment, cell_sections


def validate_required_sim_conditions(cell_or_config, sim_cfg=None):
    """Validate loader-required runtime conditions without changing NEURON state."""
    sim_cfg = sim_cfg or {}
    config = (
        cell_or_config
        if isinstance(cell_or_config, dict)
        else (getattr(cell_or_config, "config", {}) or {})
    )
    from modules.loaders import get_cell_loader_name

    if get_cell_loader_name(config) != "hoc_template":
        return
    conditions = sim_cfg.get("conditions")
    if not isinstance(conditions, dict):
        raise KeyError(
            "hoc_template runs require sim_config.conditions with explicit "
            "v_init_mV and celsius_C values."
        )
    for field in ("v_init_mV", "celsius_C"):
        value = conditions.get(field)
        if isinstance(value, bool) or value in (None, ""):
            raise ValueError(
                f"hoc_template runs require explicit numeric conditions.{field}."
            )
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"hoc_template sim_config.conditions.{field} must be numeric, got {value!r}."
            ) from exc
        if not math.isfinite(numeric):
            raise ValueError(
                f"hoc_template sim_config.conditions.{field} must be finite, got {value!r}."
            )


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


def resolve_sim_conditions(cell, sim_cfg=None):
    """Resolve loader-neutral initialization conditions for one simulation."""
    sim_cfg = sim_cfg or {}
    validate_required_sim_conditions(cell, sim_cfg)
    conditions = sim_cfg.get("conditions", {}) or {}
    if not isinstance(conditions, dict):
        raise TypeError("sim_config.conditions must be an object/dict.")

    hoc = _get_hoc(cell)
    previous = getattr(cell, "runtime_conditions", {}) or {}
    if not isinstance(previous, dict):
        previous = {}
    v_init = conditions.get("v_init_mV")
    if v_init is None:
        v_init = previous.get("v_init_mV")
    if v_init is None:
        v_init = getattr(cell, "Vinit", None)
    if v_init is None:
        v_init = getattr(hoc, "v_init", -65.0)

    celsius = conditions.get("celsius_C")
    if celsius is None:
        celsius = previous.get("celsius_C")
    if celsius is None:
        celsius = getattr(hoc, "celsius", None)

    resolved = {
        "v_init_mV": float(v_init),
        "celsius_C": None if celsius is None else float(celsius),
    }
    for key, value in resolved.items():
        if value is not None and not math.isfinite(value):
            raise ValueError(f"sim_config.conditions.{key} must be finite, got {value!r}.")
    return resolved


def apply_sim_conditions(cell, sim_cfg=None):
    """Apply temperature and initialization voltage after cell construction."""
    hoc = _get_hoc(cell)
    resolved = resolve_sim_conditions(cell, sim_cfg)
    if resolved["celsius_C"] is not None:
        hoc.celsius = resolved["celsius_C"]
    if hasattr(hoc, "v_init"):
        hoc.v_init = resolved["v_init_mV"]
    try:
        cell.runtime_conditions = dict(resolved)
    except Exception:
        pass
    return resolved


def run_current_injection(cell,sim_params,):
    
    # from neuron import h
    # h.load_file("stdrun.hoc")

    soma_seg = cell_soma_segment(cell)
    stim = h.IClamp(soma_seg)
    stim.amp = sim_params['stim_amp']
    stim.delay = sim_params['stim_delay']
    stim.dur = sim_params['stim_dur']
    h.tstop = sim_params['h_tstop']
    h.dt = sim_params['h_dt']
    # h.steps_per_ms = 1 / h.dt
    # return h, stim

    recorded_data = {}

    # Attach recorders
    vvec = h.Vector().record(soma_seg._ref_v)
    tvec = h.Vector().record(h._ref_t)

    # Attach *all* the current recorders in that section
    current_recording_vars = get_rec_vars_for_i_in_sec(cell_sections(cell, "soma")[0], 0.5)

    # Run
    resolved_conditions = apply_sim_conditions(cell, sim_params)
    h.finitialize(resolved_conditions["v_init_mV"])
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
    delay_ms = _get_float(
        "delay_ms",
        sim_cfg.get("stim_start_ms", sim_cfg.get("tstart", 0.0)),
    )
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
        "conditions": sim_cfg.get("conditions", {}) or {},
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
        "conditions": resolve_sim_conditions(cell, sim_cfg),
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
