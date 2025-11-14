
# Generate spike times for NetCon input
def gen_spike_times(
            self,
            sim_params,
            syn_params,
            jitter_tstart,
    ):
        """ Generate spike trains (list of timestamps) for NetCon input. """


        # If constant firing rate, specified in syn_params
        if isinstance(syn_params['freq'], int) or isinstance(syn_params['freq'], float):

            if syn_params['delay'] is not None:
                jitter_tstart = syn_params['delay']
            spike_times = self.homogeneous_poisson_timestamps(
                            rate_hz=syn_params['freq'], 
                            # t_start=sim_params['tstart'],
                            t_start=jitter_tstart,
                            duration_ms=sim_params['tstop'])
        
        # If inhomogeneous based on bio data
        elif isinstance(syn_params['freq'], np.ndarray) or isinstance(syn_params['freq'], list):

            # inhom_input_filepath = syn_params['freq']
            # PFR = pd.read_csv(inhom_input_filepath,delimiter=",")
            # bio_stim_input = np.array(PFR['AvgFiringRate'][PFR['Time'] >0])
            bio_stim_input = syn_params['freq']
            spike_times = []

            # Generate baseline input until csv data stim
            baseline_factor = bio_stim_input[0]
            baseline_spike_times = self.homogeneous_poisson_timestamps(
                rate_hz=baseline_factor,
                # t_start=sim_params['tstart'],
                t_start=jitter_tstart,
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
            # print(sim_params['delay']+len(time))
            stim_spike_times = time[stim_spikes==1]
            stim_spike_times = [spk + sim_params['delay'] for spk in stim_spike_times] # Add delay

            for spk in stim_spike_times:
                spike_times.append(spk)
            
            # Generate baseline input after csv data stim
            baseline_spike_times = self.homogeneous_poisson_timestamps(
                rate_hz=baseline_factor,
                t_start=sim_params['delay']+len(time),
                duration_ms=sim_params['tstop']
            )
            for spk in baseline_spike_times:
                spike_times.append(spk)

        elif isinstance(syn_params['freq'], dict):
            random_spktrn = random.randint(0,len(syn_params['freq']['base tune'])-1)
            input_spikes = syn_params['freq']['base tune'][random_spktrn]
            # print(f'spike train {random_spktrn} - spikes {spike_times}')

            spike_times = []
            for spk in input_spikes:
                spike_time = spk + 200 + 50 #diff in delays #sim_params['delay']
                spike_times.append(spike_time)


        else:
            print('Please use valid spike train input!')

        return spike_times