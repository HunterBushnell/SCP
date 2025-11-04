import sys, os
from functools import wraps

import os, sys, csv, json, h5py, random, math, pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

def suppress_output(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        devnull = open(os.devnull, 'w')
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            return fn(*args, **kwargs)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            devnull.close()
    return wrapper

@suppress_output
def simple_build_cell(soma_diam_multiplier = 1,using_hoc=False,hoc_filename=None,cell_name=None):

    from allensdk.model.biophys_sim.config import Config
    from allensdk.model.biophysical.utils import Utils

    if using_hoc:
      from neuron import h
      h.load_file(hoc_filename)
      return getattr(h, cell_name)()
    # Create the h object
    description = Config().load('manifest.json')
    utils = Utils(description)
    h = utils.h
    # Convert all "value" attributes to floats
    for dict in utils.description.data['genome']:
        for key, value in dict.items():
            if key == 'value': dict[key] = float(value)
    # Configure morphology
    morphology_path = description.manifest.get_path('MORPHOLOGY')
    utils.generate_morphology(morphology_path.encode('ascii', 'ignore'))
    utils.load_cell_parameters()
    # To match PP
    h.soma[0].diam = h.soma[0].diam * soma_diam_multiplier

    return h

def add_synapse(cell,syn_loc,syn_params,spike_train,):
    
    from neuron import h
    # print(syn_params['spec_settings']['level_of_detail'])
    ### Generate synaptic mechanisms ###
    syn = getattr(h, syn_params['spec_settings']['level_of_detail'])(syn_loc)
    for param, val in syn_params['spec_syn_param'].items():
        if hasattr(syn, param):
            setattr(syn, param, val)
            print(f'{param}: {val}')
    cell.synapses.append(syn)
    print(syn.initW)

    ### Generate synaptic input (spike trains) ###
    spike_times = spike_train

    ### Generate NetCon & Stimulation ###
    vec = h.Vector(spike_times)
    stim = h.VecStim()
    stim.play(vec)
    nc = h.NetCon(stim, syn)
    nc.weight[0] = 1 

    cell.vecs.append(vec)
    cell.stims.append(stim)
    cell.netcons.append(nc)

################ stim_synapses functions ################
class AllenCell:
    
    ############### Cell Generation Functions ###############
    def __init__(self, gid, soma_diam_multiplier=1.0):
        from allensdk.model.biophys_sim.config import Config
        from allensdk.model.biophysical.utils import Utils
        from neuron import h
        
        self._gid = gid
        self.synapses = []  # Keep track of all synapses
        self.netcons = []   # Keep track of NetCons
        self.stims = []     # Keep VecStims so they don't get garbage-collected
        self.vecs = []
        self.syn_locs = [] 
        
        description = Config().load('manifest.json')

        self.utils = Utils(description)
        self.h = self.utils.h
        self.Vinit = self.utils.description.data['conditions'][0]['v_init']
        # Cast all genome values to float
        for d in self.utils.description.data['genome']:
            if 'value' in d:
                d['value'] = float(d['value'])

        # Load morphology and parameters
        morphology_path = description.manifest.get_path('MORPHOLOGY')
        self.utils.generate_morphology(morphology_path.encode('ascii', 'ignore'))
        self.utils.load_cell_parameters()
        self.setup_morphology(soma_diam_multiplier)
        self._build_section_list()
    
    def setup_morphology(self,soma_diam_multiplier):

        self.soma = self.h.soma
        self.h.soma[0].diam *= soma_diam_multiplier
        self.dend = list(self.h.dend) if hasattr(self.h, 'dend') else []
        self.apic = list(self.h.apic) if hasattr(self.h, 'apic') else []
        self.axon = list(self.h.axon) if hasattr(self.h, 'axon') else []
        from neuron import h

        # Step 1: Set distance origin once (assuming soma(0.5) is your origin)
        h.distance(0, self.soma[0](0.5))

        # Step 2: Classify dendrites into proximal/distal
        self.proximal_dend_segs = []
        self.distal_dend_segs = []

        for sec in self.dend:
            for seg in sec:
                if 20 < h.distance(seg) < 100:
                    self.proximal_dend_segs.append(seg)  
                elif  h.distance(seg) >= 100:
                    self.distal_dend_segs.append(seg)

    def _build_section_list(self):
        from neuron import h
        self.all = h.SectionList()
        for sec in h.allsec():
            self.all.append(sec)


############### Synapse Generation Functions ###############


    def compute_corrected_weights(self,distances, target_dist_func, bandwidth=10):
        """
        Compute sampling weights that correct for anatomical bias and follow a target distribution.

        Parameters:
        - seg_list: list of (sec, loc, dist)
        - target_dist_func: function mapping distance → desired probability (e.g. scipy poisson.pmf)
        - bandwidth: KDE smoothing bandwidth (in µm)

        Returns:
        - normalized weights (np.array)
        """

        # Step 1: Estimate anatomical density using KDE
        kde = gaussian_kde(distances, bw_method=bandwidth / np.std(distances))
        anatomical_pdf = kde(distances)

        # Step 2: Evaluate target distribution at each distance
        target_pdf = target_dist_func(distances)

        # Step 3: Divide target by anatomical density to correct for bias
        raw_weights = target_pdf / (anatomical_pdf + 1e-12)

        # Step 4: Normalize
        weights = raw_weights / raw_weights.sum()

        return weights
    

    # def _lognormal_mu_sigma(self, mean, std):
    #     """Return μ and σ for np.random.lognormal given arithmetic mean & std."""
    #     mu  = math.log(mean**2 / math.sqrt(std**2 + mean**2))
    #     sig = math.sqrt(math.log(1 + (std**2 / mean**2)))
    #     return mu, sig
    def draw_syn_wt(self, syn_params):
        """Generate a synaptic weight from a lognormal distribution."""


        wt_mean = syn_params.get('wt_mean', syn_params.get('initW', 0.001))
        wt_std  = syn_params.get('wt_std', 0.0) * wt_mean   #scaled off mean, std = 0  → fixed

        if wt_std > 0:
            # mu_logn, sig_logn = self._lognormal_mu_sigma(wt_mean, wt_std)
            mu  = math.log(wt_mean**2 / math.sqrt(wt_std**2 + wt_mean**2))
            sig = math.sqrt(math.log(1 + (wt_std**2 / wt_mean**2)))
            draw_wt = lambda: np.random.lognormal(mu, sig)
        else:
            draw_wt = lambda: wt_mean

        syn_wt = draw_wt()

        return syn_wt
    


    def gen_syn_locs(
            self,
            n_syns,
            dens_eq,        # =lamda d: -0.015*d + 4.25, = lamda: 2.0 
            seg_list,
            ):
        
        """ Generate synapse locations by density distribution dens_eq  """
        
        import math, re, random
        from neuron import h

        all_syn_locs = []   # Or could use self.syn_locs.append(seg), instead of later when generating syns

        h.distance(0, self.soma[0](0.5))

        ### Generate Synaptic Locations ###
        ## If based on number and probability distrubution
        if n_syns is not None:
            distances = [h.distance(seg) for seg in seg_list]
            weights = self.compute_corrected_weights(distances,dens_eq)

            for syn in range(n_syns):
                sec_seg = random.choices(seg_list, weights=weights, k=1)[0]
                self.syn_locs.append(sec_seg)
                all_syn_locs.append(sec_seg)

        ## If based on density distribution
        else:
            for seg in seg_list:
                seg_dist = h.distance(seg)                  # µm
                seg_len  = seg.sec.L / seg.sec.nseg         # µm

                syn_dens = dens_eq(seg_dist)                # Synapses per µm
                if syn_dens <= 0:                           # skip negative densities
                    print('NEGATIVE/0 SYN_DENS, NO SYN CREATED!!!')
                    continue

                n_syns = math.floor(syn_dens * seg_len)     #+ rng.random())
                if n_syns == 0:
                    print(f'no syns created for seg {seg}: dist {seg_dist}, len {seg_len}, dens {syn_dens}')

                for syn in range(n_syns):
                    self.syn_locs.append(seg)
                    all_syn_locs.append(seg)

                # for _ in range(n_syn):
                #     # random position within the segment’s bounds
                #     #      segment runs from seg.x-Δx to seg.x+Δx
                #     dx = (rng.random() - 0.5) * (1/seg.sec.nseg)
                #     loc = seg.x + dx
                #     syn_locs.append(SynLoc(seg.sec, loc, seg_dist + dx*seg.sec.L))

        return all_syn_locs



    def homogeneous_poisson_timestamps(self, rate_hz, t_start, duration_ms):
        """
        Generate spike times for a homogeneous Poisson process.

        Parameters:
            rate_hz (float): desired firing rate in Hz.
            duration_ms (float): total time to generate over, in milliseconds.

        Returns:
            spike_times (list of float): spike timestamps in ms.
        """
        spike_times = []
        t = t_start         # Start of stimulation
        while t < duration_ms:
            isi = np.random.exponential(1000.0 / rate_hz)  # ISI in ms
            t += isi
            if t < duration_ms:
                spike_times.append(t)
        return spike_times


    def inhomogeneous_poisson_through_num_points(self,lambdas, win_length):
        t = np.zeros(len(lambdas) * win_length)
        lambdas = np.divide(lambdas,1000/win_length)
        
        for i, lambd in enumerate(lambdas):

            num_points = np.random.poisson(lambd)

            if num_points >= win_length:
                t[i * win_length : (i + 1) * win_length] = 1
                continue

            random_inds = np.random.choice(a = np.arange(win_length), size = num_points, replace = False)
            spikes = np.zeros(win_length)
            spikes[random_inds] = 1
            t[i * win_length : (i + 1) * win_length] = spikes

        return t
    

    def gen_spike_times(
            self,
            sim_params,
            syn_params,
            # bio_stim_input,
    ):
        """ Generate spike trains (list of timestamps) for NetCon input. """

        # If constant firing rate, specified in syn_params
        if isinstance(syn_params['freq'], int) or isinstance(syn_params['freq'], float):
            spike_times = self.homogeneous_poisson_timestamps(
                            rate_hz=syn_params['freq'], 
                            t_start=sim_params['tstart'], # 0 if delayed after gen (2 lines down)
                            duration_ms=sim_params['tstop'])
            # spike_times = [spike_time + sim_params['tstart'] for spike_time in spike_times] # Add delay
        
        # If inhomogeneous based on bio data
        elif isinstance(syn_params['freq'], str):

            PFR = pd.read_csv(os.path.join(syn_params['freq']),delimiter=",")
            bio_stim_input = np.array(PFR['AvgFiringRate'][PFR['Time'] >0])

            spike_times = []

            # Generate baseline input until csv data stim
            baseline_factor = bio_stim_input[0]
            baseline_spike_times = self.homogeneous_poisson_timestamps(
                rate_hz=baseline_factor,
                t_start=sim_params['tstart'],
                duration_ms=sim_params['delay']
            )
            for spk in baseline_spike_times:
                spike_times.append(spk)

            # Generate stim input from csv
            # bio_stim_input = [avg_fq - baseline_factor for avg_fq in bio_stim_input]
            stim_spikes = self.inhomogeneous_poisson_through_num_points(
                            bio_stim_input,
                            int(sim_params['bins']))
            time = np.arange(len(stim_spikes))
            stim_spike_times = time[stim_spikes==1]
            stim_spike_times = [spk + sim_params['delay'] for spk in stim_spike_times] # Add delay

            for spk in stim_spike_times:
                spike_times.append(spk)
            
            # Generate baseline input after csv data stim
            # baseline_spike_times = self.homogeneous_poisson_timestamps(
            #     rate_hz=baseline_factor,
            #     t_start=sim_params['tstart'],
            #     duration_ms=sim_params['delay']
            # )
            # for spk in baseline_spike_times:
            #     spike_times.append(spk)


        else:
            print('Please use valid spike train input!')

        return spike_times


    def gen_syn_mechs(
            self,
            syn_loc,
            syn_params,
            ):
        """ Generate syn_name synapse with parameters syn_params. """

        from neuron import h
        syn_type = syn_params['type']
        print(f'syn_type: {type(syn_type)} | syn_loc: {syn_loc}')
        syn = getattr(h, syn_params['type'])(syn_loc)
        for param, val in syn_params.items():
            # if param in ('wt_mean', 'wt_std'): # Skip certain keys
            #     continue
            if hasattr(syn, param):
                setattr(syn, param, val)
        syn_wt = self.draw_syn_wt(syn_params)
        syn.initW = syn_wt
        # print(syn.initW)

        return syn



    def gen_syns(
            self,
            syn_params,
            # bio_stim_input,
            sim_params,
            ):
        """ Generate syn_name synapses with parameters syn_params for segments in seg_list. """

        from neuron import h

        syn_records = {}    # list of dicts we will optionally return
        syn_id = 0

        ### Generate synapses for each syn group in syn_params ###
        for syn_group in syn_params:
            if syn_params[syn_group]['N_syn'] is not None:
                if syn_params[syn_group]['N_syn'] < 1:
                    continue
                
            # Create record lsit for group
            syn_records[syn_group] = []

            # Get syn_params for group
            syn_params_group = syn_params[syn_group]

            # Determine which dendrite sections for group syns
            syn_segs = syn_params_group['segs']
            if syn_segs == 'all':
                seg_list = self.proximal_dend_segs + self.distal_dend_segs
            elif syn_segs == 'proximal':
                seg_list = self.proximal_dend_segs
            elif syn_segs == 'distal':
                seg_list = self.distal_dend_segs
            elif syn_segs == 'soma':
                seg_list = self.soma
            else:
                print('NO DENDRITE SECTIONS SELECTED!!!')


            ### Generate synaptic locations ###
            all_syn_locs = self.gen_syn_locs(
                                n_syns = syn_params_group['N_syn'], 
                                dens_eq = syn_params_group['dist_func'], 
                                seg_list = seg_list,
                                )
            

            for syn_loc in all_syn_locs:
                ### Generate synaptic mechanisms ###
                syn = self.gen_syn_mechs(syn_loc, syn_params_group)
                self.synapses.append(syn)

                ### Generate synaptic input (spike trains) ###
                spike_times = self.gen_spike_times(
                                    sim_params, 
                                    syn_params_group,
                                    )

                ### Generate NetCon & Stimulation ###
                vec = h.Vector(spike_times)
                stim = h.VecStim()
                stim.play(vec)
                nc = h.NetCon(stim, syn)
                nc.weight[0] = 1 
                
                # self.synapses.append(syn)
                self.vecs.append(vec)
                self.stims.append(stim)
                self.netcons.append(nc)


                ### Generate synaptic record ###
                syn_records[syn_group].append({    # Could add more, or auto from syn_params_group/h.
                    "syn_id": syn_id,
                    "group": syn_group,
                    "type": syn_params_group['type'],
                    "weight": syn.initW,
                    "distance": h.distance(syn_loc),
                    "section": syn_loc.sec.name(),
                    "x": syn_loc.x,
                    "spike_times": spike_times,
                    # "NMDA_ratio": nmda_wt, # EXAMPLE, FOR FUTURE
                })
                # print(f'synapse {syn_id} generated: {syn_records[syn_group][-1]}')
                syn_id = syn_id + 1
                
            print(f'{syn_group} synapses generated: {len(all_syn_locs)}')

        return syn_records



    def __str__(self):
        return f"AllenCell(soma={self.soma}, dendrites={len(self.dend)})"



############ Other cell building functions #############
