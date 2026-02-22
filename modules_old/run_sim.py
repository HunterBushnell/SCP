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
    plt.legend(loc='left', bbox_to_anchor=(1, 1))
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

def run_cell(cell,sim_params):
    
    sim_traces = {}
    # Set recording vectors
    tvec = h.Vector().record(h._ref_t)
    vvec = h.Vector().record(cell.soma[0](0.5)._ref_v)  # somatic Vm
    isynvec = h.Vector().record(cell.synapses[0]._ref_i)
    gsynvec = h.Vector().record(cell.synapses[0]._ref_g)

    # Set initial stim
    # if sim_params['init_stim']:
    #     stim = h.IClamp(cell.soma[0](0.5))
    #     stim.amp = sim_params['init_stim']['amp']
    #     stim.delay = sim_params['init_stim']['delay']
    #     stim.dur = sim_params['init_stim']['dur']

    # Set up simulation parameters
    h.finitialize(cell.Vinit)
    h.dt = sim_params['dt']
    h.tstop = sim_params['tstop']  # ms

    h.run()

    sim_traces['T'] = np.array(tvec)
    sim_traces['V'] = np.array(vvec)
    sim_traces['I'] = np.array(isynvec)
    sim_traces['G'] = np.array(gsynvec)
    
    return sim_traces

#############################################
#############################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from neuron import h
import copy


def run_param_analysis(
        cell,
        sim_params,
        syn_params,
        param_study, 
    ):



    input_type = param_study['input_type']
    param_type = param_study['param_type']
    param_vals = param_study['param_vals']
    n_trials = param_study['n_trials']
    stim_duration   = (sim_params['tstop'] - sim_params['tstart']) / 1000

    all_param_data = {}
    all_syn_records = {}

    for param_val in param_vals:
        
        ############################### Looped Block ###############################
        # all_spike_times_flat = []   # <- pooled across *all* runs
        trial_spike_arrays = []
        per_trial_spikes = []
        per_trial_rates       = []   # <- to print individual firing rates
        all_syn_records[param_val] = []

        syn_params_copy = copy.deepcopy(syn_params)


        # Update params
        if isinstance(param_val,str): # May need more refining/specifying for single run case
            syn_params_copy = copy.deepcopy(syn_params) 

        else:
            if 'all' in input_type:
                for syn_group in syn_params_copy:
                    syn_params_copy[syn_group][param_type] = param_val
            else:
                for syn_group in input_type:
                    syn_params_copy[syn_group][param_type] = param_val
            # if (input_type == 'stim' or input_type == 'all'):
            #     syn_params_copy['stim'][param_type] = param_val
            # if (input_type == 'bg' or input_type == 'all'):
            #     syn_params_copy['bg'][param_type] = param_val

        for trial_idx in range(n_trials):

            # ------------------------------------------------------------
            # 1) (Re)build synapses on the *same* AllenCell instance
            #    (If you want different cells, instantiate inside the loop.)
            # ------------------------------------------------------------
            
            # clear any synapses left over from a previous run
            cell.syn_locs  = []
            cell.vecs      = []
            cell.stims     = []
            cell.synapses  = []
            cell.netcons   = []

            # ---------- generate all synapses ----------
            syn_records = cell.gen_syns(
                syn_params = syn_params_copy,
                sim_params = sim_params)
            all_syn_records[param_val].append(syn_records)

            # ------------------------------------------------------------
            # 2) run simulation
            # ------------------------------------------------------------
            sim_traces = run_cell(cell,sim_params)  
            T = sim_traces['T']
            V = sim_traces['V']

            # ------------------------------------------------------------
            # 3) detect spikes (convert threshold to mV)
            # ------------------------------------------------------------
            peaks, _      = find_peaks(V, height=-20, distance=int(2/h.dt))
            spike_times   = T[peaks]
            # all_spike_times_flat.extend(spike_times)
            trial_spike_arrays.append(spike_times)     # <-- store *per-trial*

            # per-trial stats
            trial_spikes = len(spike_times)
            rate_hz = trial_spikes / (stim_duration)

            per_trial_spikes.append(trial_spikes)
            per_trial_rates.append(round(rate_hz,1))
            
            print(f"param: {param_val} | trial: {trial_idx+1} | {trial_spikes} spikes  ⇒  {rate_hz:.2f} Hz")

        # ----------------------------------------------------------------
        # 4) Save in dictionary for analysis
        # ----------------------------------------------------------------
        # all_spike_times_flat = np.array(all_spike_times_flat)
        # all_param_data[param_val] = all_spike_times_flat
        all_param_data[param_val] = trial_spike_arrays
        print(f'param: {param_val} |'
              f'spikes/avg freq per trial: {per_trial_spikes}/{per_trial_rates} |' 
              f'avg freq: {round(np.average(per_trial_rates), 2)}\n')

        ############################### Looped Block ###############################
    return all_param_data, all_syn_records
