# ────────────────────────────────────────────────────────────────────────────
#  Moving-average helper
# ────────────────────────────────────────────────────────────────────────────
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from matplotlib.lines import Line2D
from itertools import cycle, chain
from collections import defaultdict

from neuron import h

# ---------------------------------------------------------------------------
def moving_average(arr, 
                #    *,
                   win_size: float,
                   bin_width: float,
                   mode: str = 'center'):
    """Centred or causal boxcar moving–average."""
    if win_size is None or win_size <= bin_width or arr.size < win_size/bin_width:
        return np.asarray(arr, dtype=float)

    k = int(round(win_size / bin_width))
    if k % 2 == 0 and mode == 'center':
        k += 1
    k = max(k, 1)

    kernel = np.ones(k) / k
    arr    = np.asarray(arr, dtype=float)

    if mode == 'center':
        y_full = np.convolve(arr, kernel, mode='valid')
        return y_full

    elif mode == 'causal':
        pad = (k - 1, 0)
        arr_pad = np.convolve(np.pad(arr, pad), kernel, mode='valid')
        return arr_pad

    else:
        raise ValueError("mode must be 'center' or 'causal'")
    
def _align_centers(centers, y_smooth, mode='center'):
    """Trim/shift centers to match y_smooth length based on smoothing mode."""
    if len(y_smooth) == len(centers):
        return centers
    if mode == 'center':  # symmetric 'valid' convolution
        drop = (len(centers) - len(y_smooth)) // 2
        return centers[drop: drop + len(y_smooth)]
    elif mode == 'causal':  # left-aligned boxcar
        return centers[:len(y_smooth)]
    else:
        return centers[:len(y_smooth)]

# ────────────────────────────────────────────────────────────────────────────
#  Normalization helper
# ────────────────────────────────────────────────────────────────────────────
def normalize(data,
              norm_offset,
              ):
    
    norm_data = []
    norm_max = max(data) + norm_offset

    for val in data:
        val = val + norm_offset # Normalize base data to 0
        val = val / norm_max # Normalize peak data value to 1
        norm_data.append(val)

    return norm_data

# ────────────────────────────────────────────────────────────────────────────
#  Synapse-property plots (distance / weight / distance-density)
# ────────────────────────────────────────────────────────────────────────────
# def plot_syn_records(
#         cell,
#         syn_records,
#         plotted_groups, # e.g. ['bg', 'stim']
#         plotted_props=('distance',), # 'distance', 'distance_density', 'weight', or ('weight','distance')
#         # labels=None,
#         plot_type='both', # 'hist' | 'line' | 'both'
#         bins=10.0, # bin width (µm or weight units)
#         win_size=25,
#         fig_sizes=(6, 4)):
#     """
#     syn_records    : dict -> {'bg_exc':[dict,…], 'bg_inh':[dict,…], 'stim':[dict,…], …}
#     plotted_groups : iterable of keys from syn_records
#     plotted_props  : see below
#     """

#     if plotted_groups == ['all']:
#         record_sets = [list(chain.from_iterable(syn_records.values()))]
#     else:
#         record_sets = [syn_records[g] for g in plotted_groups]
#     labels      = plotted_groups  # legend names follow plotted_groups


#     COLOR_CYCLER = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
#     alpha=0.6
#     # props = tuple(plotted_props)

#     # normalize variants like "weight probability" / "weight_probability"
#     def _norm(s):
#         return s.strip().lower().replace(' ', '_')

#     props = tuple(_norm(p) for p in plotted_props)
#     # Accept some aliases
#     if props == ('density_count',):
#         props = ('distance_count',)

#     def _auto_edges(data, bw):
#         lo, hi = data.min(), data.max()
#         return np.arange(lo, hi + bw, bw)

#     if plot_type not in ('hist', 'line', 'both'):
#         raise ValueError("plot_type must be 'hist', 'line', or 'both'")

#     # =============================== Distance / density =====================
#     if props in (('distance',), ('distance_count',), ('distance_probability',),
#                  ('distance_density',), ('density',)):

#         # "distance_density"/"density" keep your per-branch normalisation path
#         use_density_mode = props[0] in ('distance_density', 'density')
#         # when not using per-branch density, choose count vs probability
#         use_prob_mode    = props[0] == 'distance_probability'

#         plt.figure(figsize=fig_sizes)

#         for recs, lab in zip(record_sets, labels):
#             dist  = np.asarray([r['distance'] for r in recs])
#             sects = np.asarray([r['section']  for r in recs])
#             if not dist.size:
#                 continue

#             col    = next(COLOR_CYCLER)
#             # edges  = _auto_edges(dist, bins)
#             # centers = edges[:-1] + bins / 2
#             # counts, _ = np.histogram(dist, bins=edges)

#             if use_density_mode:
#                 # -------- per-branch density ----------
#                 edges = _auto_edges(dist, bins)
#                 len_exposure = np.zeros(len(edges) - 1)
#                 branches_in_bin = [set() for _ in range(len(edges) - 1)]

#                 h.distance(0, cell.soma[0](0.5))  # set origin for distance measurements
#                 for sec in cell.dend:  # or whatever section list you use
#                     for seg in sec:
#                         seg_dist = h.distance(seg)
#                         seg_len  = sec.L / sec.nseg
#                         i = np.digitize(seg_dist, edges) - 1
#                         if 0 <= i < len(len_exposure):
#                             len_exposure[i] += seg_len
#                             branches_in_bin[i].add(sec.name())

#                 # count synapses in bins
#                 counts, _ = np.histogram(dist, bins=edges)

#                 # density per µm of dendrite
#                 with np.errstate(divide='ignore', invalid='ignore'):
#                     density = np.divide(counts, len_exposure, where=len_exposure > 0)
#                 # optional: per branch average
#                 # branch_counts = np.array([len(s) for s in branches_in_bin])
#                 # density = np.divide(density, branch_counts, where=branch_counts > 0)

#                 ylab = f"Synapses / µm (bin={bins}µm)"

#                 if plot_type in ('hist', 'both'):
#                     plt.bar(centers, density, width=bins, alpha=alpha, color=col)

#                 if plot_type in ('line', 'both'):
#                     y_line = density.copy()
#                     y_line = moving_average(y_line, win_size=win_size, bin_width=bins, mode='center')
#                     x_line = _align_centers(centers, y_line, mode='center')
#                     plt.plot(x_line, y_line, lw=1.8, color=col, label=f'{lab} (smoothed)')
                    
#                     # y_line = moving_average(density, bins, win_size)
#                     # c_line = centers[:len(y_line)]
#                     # plt.plot(c_line, y_line, lw=1.8, color=col, label=f'{lab} (smoothed)')

#             else:
#                 # --------- plain distance: counts vs probability ----------
#                 if use_prob_mode:
#                     # probability density: histogram density=True, KDE unscaled
#                     ylab  = "Probability density (1/µm)"
#                     if plot_type in ('hist', 'both'):
#                         plt.hist(dist, bins=edges, density=True, alpha=alpha, color=col)
#                     if plot_type in ('line', 'both'):
#                         dens, _ = np.histogram(dist, bins=edges, density=True)
#                         y_line = dens.copy()
#                         y_line = moving_average(y_line, win_size=win_size, bin_width=bins, mode='center')
#                         x_line = _align_centers(centers, y_line, mode='center')
#                         plt.plot(x_line, y_line, lw=1.8, color=col, label=f'{lab} (smoothed)')
#                         # y_line = moving_average(dens, bins, win_size)
#                         # c_line = centers[:len(y_line)]
#                         # plt.plot(c_line, y_line, lw=1.8, color=col, label=f'{lab} (smoothed)')

#                 else:
#                     # counts per bin
#                     ylab  = f"Synapse count (bin={bins} µm)"
#                     if plot_type in ('hist', 'both'):
#                         plt.bar(centers, counts, width=bins, alpha=alpha, color=col)
#                     if plot_type in ('line', 'both'):
#                         y_line = counts.copy()
#                         y_line = moving_average(y_line, win_size=win_size, bin_width=bins, mode='center')
#                         x_line = _align_centers(centers, y_line, mode='center')
#                         plt.plot(x_line, y_line, lw=1.8, color=col, label=f'{lab} (smoothed)')
#                         # y_line = moving_average(counts, bins, win_size)
#                         # c_line = centers[:len(y_line)]
#                         # plt.plot(c_line, y_line, lw=1.8, color=col, label=f'{lab} (smoothed)')


#         plt.xlabel("Distance from soma (µm)")
#         plt.ylabel(ylab)
#         plt.title("Distance distribution")
#         plt.legend(); plt.tight_layout(); plt.show()
#         return

#     # ================================= Weight only ==========================
#     if props in (('weight',), ('weight_count',), ('weight_probability',)):
#         use_prob_mode = props[0] == 'weight_probability'

#         plt.figure(figsize=fig_sizes)
#         for recs, lab in zip(record_sets, labels):
#             w = np.asarray([(r['weight']) for r in recs])
#             if not w.size:
#                 continue
#             col   = next(COLOR_CYCLER)
#             edges = _auto_edges(w, bins)

#             if use_prob_mode:
#                 ylab = "Probability density"
#                 if plot_type in ('hist', 'both'):
#                     plt.hist(w, bins=edges, density=True, alpha=alpha, color=col)
#                 if plot_type in ('line', 'both'):
#                     dens, _ = np.histogram(w, bins=edges, density=True)
#                     c_line  = (edges[:-1] + edges[1:]) * 0.5
#                     y_line = dens.copy()
#                     y_line = moving_average(y_line, win_size=win_size, bin_width=bins, mode='center')
#                     x_line = _align_centers(centers, y_line, mode='center')
#                     plt.plot(x_line, y_line, lw=1.8, color=col, label=f'{lab} (smoothed)')
#                     # y_line  = moving_average(dens, bins, win_size)
#                     # c_line  = c_line[:len(y_line)]
#                     # plt.plot(c_line, y_line, color=col, lw=1.8, label=f'{lab} (smoothed)')
#             else:
#                 ylab = f"Synapse count (bin={bins})"
#                 if plot_type in ('hist', 'both'):
#                     plt.hist(w, bins=edges, alpha=alpha, color=col)
#                 if plot_type in ('line', 'both'):
#                     cnts, _ = np.histogram(w, bins=edges)
#                     c_line  = (edges[:-1] + edges[1:]) * 0.5
#                     y_line = cnts.copy()
#                     y_line = moving_average(y_line, win_size=win_size, bin_width=bins, mode='center')
#                     x_line = _align_centers(centers, y_line, mode='center')
#                     plt.plot(x_line, y_line, lw=1.8, color=col, label=f'{lab} (smoothed)')
#                     # y_line  = moving_average(cnts, bins, win_size)
#                     # c_line  = c_line[:len(y_line)]
#                     # plt.plot(c_line, y_line, color=col, lw=1.8, label=f'{lab} (smoothed)')

#         plt.xlabel("Synaptic weight")
#         plt.ylabel(ylab)
#         plt.title("Weight distribution")
#         plt.legend(); plt.tight_layout(); plt.show()
#         return

#     # ============================ Weight vs distance ========================
#     if set(props) == {'weight', 'distance'}:
#         plt.figure(figsize=fig_sizes)
#         for recs, lab in zip(record_sets, labels):
#             w = [r['weight']  for r in recs]
#             d = [r['distance'] for r in recs]
#             plt.scatter(d, w, s=5, alpha=1.0, label=lab)

#         plt.xlabel("Distance (µm)")
#         plt.ylabel("Weight (max) (nS)")
#         plt.title("Weight vs Distance")
#         plt.legend(); 
#         plt.tight_layout(); plt.show()
#         return

#     raise ValueError("plotted_props must be "
#                      "('distance',), ('distance_density',), "
#                      "('weight',), or ('weight','distance')")

def plot_syn_records(
        cell,
        syn_records,
        plotted_groups,                 # e.g. ['bg', 'stim'] or ['all']
        plotted_props=('distance',),    # 'distance', 'distance_count',
                                        # 'distance_probability', 'distance_density'/'density',
                                        # 'weight', 'weight_count', 'weight_probability',
                                        # or ('weight','distance')
        color = None,
        plot_type='both',               # 'hist' | 'line' | 'both'
        bins=10.0,                      # bin width (µm or weight units)
        win_size=25,                    # moving-average window (same units as `bins`)
        fig_sizes=(6, 4)):

    # collect records and labels
    if plotted_groups == ['all']:
        record_sets = [list(chain.from_iterable(syn_records.values()))]
    else:
        record_sets = [syn_records[g] for g in plotted_groups]
    labels = plotted_groups

    COLOR_CYCLER = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    alpha = 0.6

    # normalise prop names
    def _norm(s):
        return s.strip().lower().replace(' ', '_')
    props = tuple(_norm(p) for p in plotted_props)
    if props == ('density_count',):     # alias
        props = ('distance_count',)

    def _auto_edges(data, bw):
        lo, hi = data.min(), data.max()
        return np.arange(lo, hi + bw, bw)

    if plot_type not in ('hist', 'line', 'both'):
        raise ValueError("plot_type must be 'hist', 'line', or 'both'")

    # =============================== Distance (and density) ===============================
    if props in (('distance',), ('distance_count',), ('distance_probability',),
                 ('distance_density',), ('density',)):

        use_density_mode = props[0] in ('distance_density', 'density')     # per-µm exposure
        use_prob_mode    = props[0] == 'distance_probability'              # PDF (1/µm)

        plt.figure(figsize=fig_sizes)
        ylab = ""   # set per-loop; used after loop

        for recs, lab in zip(record_sets, labels):
            dist  = np.asarray([r['distance'] for r in recs])
            if not dist.size:
                continue

            if color is not None:
                col = color
            else:
                col    = next(COLOR_CYCLER)
            edges  = _auto_edges(dist, bins)
            centers = (edges[:-1] + edges[1:]) * 0.5

            if use_density_mode:
                # ---- per-µm exposure in each distance bin ----
                len_exposure = np.zeros(len(edges) - 1)  # total dendritic length (µm) in each bin
                h.distance(0, cell.soma[0](0.5))
                for sec in cell.dend:
                    for seg in sec:
                        seg_dist = h.distance(seg)
                        seg_len  = sec.L / sec.nseg
                        i = np.digitize(seg_dist, edges) - 1
                        if 0 <= i < len(len_exposure):
                            len_exposure[i] += seg_len

                # synapse counts per bin
                counts, _ = np.histogram(dist, bins=edges)

                # density: synapses per µm of dendrite in that bin
                with np.errstate(divide='ignore', invalid='ignore'):
                    yvals = np.divide(counts, len_exposure, where=len_exposure > 0)

                ylab = f"Synapses / µm (bin = {bins} µm)"

                # histogram
                if plot_type in ('hist', 'both'):
                    plt.bar(centers, yvals, width=bins, alpha=alpha, color=col)

                # smoothed line (moving average)
                if plot_type in ('line', 'both'):
                    y_line = moving_average(yvals, win_size=win_size, bin_width=bins, mode='center')
                    x_line = _align_centers(centers, y_line, mode='center')
                    plt.plot(x_line, y_line, lw=1.8, color=col, label=f'{lab} (smoothed)')

            else:
                # plain distance: counts or probability
                if use_prob_mode:
                    # probability density (1/µm)
                    dens, _ = np.histogram(dist, bins=edges, density=True)
                    yvals = dens
                    ylab  = "Probability density (1/µm)"

                    if plot_type in ('hist', 'both'):
                        plt.bar(centers, yvals, width=bins, alpha=alpha, color=col)
                    if plot_type in ('line', 'both'):
                        y_line = moving_average(yvals, win_size=win_size, bin_width=bins, mode='center')
                        x_line = _align_centers(centers, y_line, mode='center')
                        plt.plot(x_line, y_line, lw=1.8, color=col, label=f'{lab} (smoothed)')
                else:
                    # counts per bin
                    counts, _ = np.histogram(dist, bins=edges, density=False)
                    yvals = counts
                    ylab  = f"Synapse count (bin = {bins} µm)"

                    if plot_type in ('hist', 'both'):
                        plt.bar(centers, yvals, width=bins, alpha=alpha, color=col)
                    if plot_type in ('line', 'both'):
                        y_line = moving_average(yvals, win_size=win_size, bin_width=bins, mode='center')
                        x_line = _align_centers(centers, y_line, mode='center')
                        plt.plot(x_line, y_line, lw=1.8, color=col, label=f'{lab} (smoothed)')

        plt.xlabel("Distance from soma (µm)")
        plt.ylabel(ylab)
        plt.title("Distance distribution")
        # plt.legend()
        plt.tight_layout()
        plt.show()
        return

    # ================================= Weight only =================================
    if props in (('weight',), ('weight_count',), ('weight_probability',)):
        use_prob_mode = props[0] == 'weight_probability'

        plt.figure(figsize=fig_sizes)
        ylab = ""

        for recs, lab in zip(record_sets, labels):
            w = np.asarray([r['weight'] for r in recs])
            if not w.size:
                continue

            if color is not None:
                col = color
            else:
                col    = next(COLOR_CYCLER)
            edges  = _auto_edges(w, bins)
            centers = (edges[:-1] + edges[1:]) * 0.5

            if use_prob_mode:
                dens, _ = np.histogram(w, bins=edges, density=True)
                yvals = dens
                ylab  = "Probability density"

                if plot_type in ('hist', 'both'):
                    plt.bar(centers, yvals, width=bins, alpha=alpha, color=col)
                if plot_type in ('line', 'both'):
                    y_line = moving_average(yvals, win_size=win_size, bin_width=bins, mode='center')
                    x_line = _align_centers(centers, y_line, mode='center')
                    plt.plot(x_line, y_line, color=col, lw=1.8, label=f'{lab} (smoothed)')
            else:
                cnts, _ = np.histogram(w, bins=edges, density=False)
                yvals = cnts
                ylab  = f"Synapse count (bin = {bins})"

                if plot_type in ('hist', 'both'):
                    plt.bar(centers, yvals, width=bins, alpha=alpha, color=col)
                if plot_type in ('line', 'both'):
                    y_line = moving_average(yvals, win_size=win_size, bin_width=bins, mode='center')
                    x_line = _align_centers(centers, y_line, mode='center')
                    plt.plot(x_line, y_line, color=col, lw=1.8, label=f'{lab} (smoothed)')

        plt.xlabel("Synaptic weight")
        plt.ylabel(ylab)
        plt.title("Weight distribution")
        # plt.legend()
        plt.tight_layout()
        plt.show()
        return

    # ============================ Weight vs distance ============================
    if set(props) == {'weight', 'distance'}:
        plt.figure(figsize=fig_sizes)
        for recs, lab in zip(record_sets, labels):
            w = [r['weight']  for r in recs]
            d = [r['distance'] for r in recs]
            plt.scatter(d, w, s=5, alpha=1.0, label=lab)

        plt.xlabel("Distance (µm)")
        plt.ylabel("Weight (max) (nS)")
        plt.title("Weight vs Distance")
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

    raise ValueError("plotted_props must be "
                     "('distance',), ('distance_count',), ('distance_probability',), "
                     "('distance_density',), ('density',), "
                     "('weight',), ('weight_count',), ('weight_probability',), "
                     "or ('weight','distance')")

# ────────────────────────────────────────────────────────────────────────────
#  Single-simulation plot (Vm, raster, rates)
# ────────────────────────────────────────────────────────────────────────────
def plot_single_run(
        T, V,
        syn_records,
        sim_params,
        *,
        win_size     = None,
        rate_style      = 'line',   # 'hist' | 'line' | 'both'
        raster_style    = 'line',   # 'line' | 'dot'
        col = None,
        in_vivo_curve   = None,    # (time_s, rate_hz) or None
        plot_window = (None,None),
    ):

    """
    T, V : somatic time / voltage vectors (NumPy or h.Vectors)
    syn_records : {'bg_exc':[dict,…], 'bg_inh':[dict,…], 'stim':[dict,…], …}
                  each dict must have 'spike_times' list
    """

    sim_duration_ms = sim_params['tstop']
    bin_width = sim_params['bins']
    stim_start = sim_params['delay'] + 100 # Start/stop of stim manual for now, could be auto?
    stim_stop = sim_params['delay'] + 550

    # Dynamically determine groups and colours
    groups = list(syn_records.keys())
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    g2col  = {g: next(colors) for g in groups}

    # ------------------------- build raster data ---------------------------
    spikes_by_group = {
        g: np.concatenate([r['spike_times'] for r in recs]) if recs else np.array([])
        for g, recs in syn_records.items()
    }
    n_syn = {g: len(syn_records[g]) for g in groups}

    # ----------------------- GridSpec layout -------------------------------
    rows, layout, heights = 1, ['V'], [3]
    if raster_style:
        rows += 1; layout.append('R'); heights.append(3)
    if rate_style:
        rows += 1; layout.append('F'); heights.append(3)

    fig = plt.figure(figsize=(6, 3 * rows))
    gs  = fig.add_gridspec(rows, 1, height_ratios=heights, hspace=0.3)

    # -------------------------- A) Vm trace --------------------------------
    peaks, _ = find_peaks(V, height=-20, distance=2)
    spike_times  = T[peaks]
    print(f"Detected {len(spike_times)} spikes (total avg: {(len(spike_times)/((sim_duration_ms-100)/1000))}) at times (ms):", spike_times)

    axV = fig.add_subplot(gs[0])

    axV.plot(T, V,color = col)
    axV.scatter(spike_times, V[peaks], s=15, color='k', zorder=5)
    for vline in [stim_start,stim_stop]: axV.axvline(x=vline,color='k',linestyle='-',linewidth=1)

    axV.set_xlim(plot_window[0],plot_window[1])
    axV.set_ylabel('Vm (mV)')
    axV.set_title('Cell Output')
    axV.grid()

    row_idx = 1

    # -------------------------- B) raster ----------------------------------
    if 'R' in layout:
        axR = fig.add_subplot(gs[row_idx], sharex=axV)
        row_idx += 1
        def _draw(ax, recs, color, y0):
            for i, r in enumerate(recs):
                y = y0 + i + 1
                t = np.asarray(r['spike_times'])
                if raster_style == 'line':
                    ax.vlines(t, y - .4, y + .4, color=color, lw=0.8)
                else:
                    ax.scatter(t, np.full_like(t, y), s=18, marker='.',
                               color=color)
        y0 = 0
        legend_elems = []
        for g in groups:
            _draw(axR, syn_records[g], g2col[g], y0)
            legend_elems.append(Line2D([0], [0], marker='.',
                                       color=g2col[g], linestyle='',
                                       ms=8, label=g))
            y0 += n_syn[g]

        # plot vertical lines for spike times
        for spk in spike_times:
            axR.axvline(x=spk, color = 'k', linestyle = '--', linewidth = 1)

        for vline in [stim_start,stim_stop]: axR.axvline(x=vline,color='k',linestyle='-',linewidth=1)

        axR.set_xlim(plot_window[0],plot_window[1])
        axR.set_ylim(0.5, y0 + .5)
        axR.set_ylabel('Synapse ID')
        axR.set_title('Cell Input Raster')
        axR.grid(axis='x')
        if len(groups) > 1:
            axR.legend(handles=legend_elems, loc='upper left')

    # -------------------------- C) rates -----------------------------------
    if 'F' in layout:
        axF = fig.add_subplot(gs[row_idx], sharex=axV)
        bins    = np.arange(0, sim_duration_ms + bin_width, bin_width)
        centers = bins[:-1] + .5 * bin_width
        bw_sec  = bin_width / 1000

        rates = {}
        for g in groups:
            counts,_ = np.histogram(spikes_by_group[g], bins=bins)
            rate     = counts / (max(n_syn[g], 1) * bw_sec)
            if win_size:
                rate  = moving_average(rate, win_size=win_size,
                                       bin_width=bin_width,
                                       mode='center')
            rates[g] = rate

        if win_size:
            centers = centers[:len(next(iter(rates.values())))]

        if rate_style in ('hist', 'both'):
            for g in groups:
                axF.bar(centers, rates[g], width=bin_width,
                        color=g2col[g], alpha=.5, label=g)

        if rate_style in ('line', 'both'):
            for g in groups:
                axF.plot(centers, rates[g], lw=1.8, color=g2col[g], label=g)

        # combined
        # print(rates)
        # total = np.sum(list(rates.values()), axis=0)
        # print(total)
        # axF.plot(centers, total, lw=2, color='k', label='TOTAL')

        # ---------- combined curve: correctly averaged across *all* synapses ---
        # keep the per-group spike counts so we can weight by #synapses
        counts_by_g = {g: np.histogram(spikes_by_group[g], bins=bins)[0]
                       for g in groups}

        total_syn   = sum(n_syn.values())
        total_counts = np.sum(list(counts_by_g.values()), axis=0)       # per-bin spikes
        total_rate   = total_counts / (max(total_syn, 1) * bw_sec)      # Hz / synapse

        if win_size:
            total_rate = moving_average(total_rate,
                                        win_size=win_size,
                                        bin_width=bin_width,
                                        mode='center')

        # axF.plot(centers[:len(total_rate)], total_rate,
        #          lw=2, color='k', label='TOTAL')

        # in-vivo data
        if in_vivo_curve is not None:
            axF.plot(in_vivo_curve[0] * 1000, in_vivo_curve[1],
                     'k', lw=2, label='In-vivo')
        
        # plot vertical lines for spike times
        for spk in spike_times:
            axF.axvline(x=spk, color = 'k', linestyle = '--', linewidth = 1)
            
        for vline in [stim_start,stim_stop]: axF.axvline(x=vline,color='k',linestyle='-',linewidth=1)

        axF.set_ylabel('Rate (Hz / synapse)')
        axF.set_xlabel('Time (ms)')
        axF.set_xlim(plot_window[0],plot_window[1])
        axF.set_ylim(bottom=0)
        axF.set_title('Cell Input Average')
        if len(groups) > 1 or in_vivo_curve is not None:
            axF.legend()
        axF.grid()
    

    # plt.tight_layout()
    plt.show()



# ────────────────────────────────────────────────────────────────────────────
#  Param-study plotting helper for shaded plot
# ────────────────────────────────────────────────────────────────────────────
def _mean_band_stats(Y, *, shade=None, norm_fr=None, norm_scope='post_mean'):
    """
    Y: (n_trials, T) array
    shade: None | 'sem' | 'std' | float(multiplier of SEM) | (qlo,qhi) in [0,1]
    norm_scope: 'post_mean' (shift/scale mean and band together) or 'per_trial'
    Returns: mean, lo, hi  (lo/hi are None if shade is None)
    """
    Y = np.asarray(Y, float)
    n = Y.shape[0]

    # per-trial normalization (if requested)
    if norm_fr is not None and norm_scope == 'per_trial':
        Y = np.vstack([
            (y + norm_fr) / max(np.max(y + norm_fr), 1e-12)
            for y in Y
        ])

    mean = Y.mean(axis=0)

    # compute band on current Y
    lo = hi = None
    if shade is not None:
        if isinstance(shade, str):
            if shade == 'sem':
                spread = (Y.std(axis=0, ddof=1) / np.sqrt(n)) if n > 1 else np.zeros_like(mean)
                lo, hi = mean - spread, mean + spread
            elif shade == 'std':
                spread = Y.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
                lo, hi = mean - spread, mean + spread
            else:
                raise ValueError("shade must be 'sem', 'std', float, (qlo,qhi), or None")
        elif isinstance(shade, (int, float)):
            spread = (Y.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)) * float(shade)
            lo, hi = mean - spread, mean + spread
        else:  # (qlo, qhi)
            qlo, qhi = 100*shade[0], 100*shade[1]
            lo, hi = np.percentile(Y, [qlo, qhi], axis=0)

    # post-mean normalization (shift/scale mean and band together)
    if norm_fr is not None and norm_scope == 'post_mean':
        mean0 = mean + norm_fr
        scale = max(np.max(mean0), 1e-12)
        mean  = mean0 / scale
        if lo is not None and hi is not None:
            lo = (lo + norm_fr) / scale
            hi = (hi + norm_fr) / scale

    return mean, lo, hi




# ────────────────────────────────────────────────────────────────────────────
#  Param-study plotting
# ────────────────────────────────────────────────────────────────────────────
def plot_param_study(all_param_data,
                    #  *,
                     param_study = {},
                     sim_params = {},
                     win_size = 25,
                     plot_type='line',
                     plot_bio=None,
                     plot_raster=False,
                     raster_style='line',
                     alpha=0.6,
                     plot_window = None,
                     norm_fr = None,
                     shade_mode = None,
                     set_color = None,
                     save_curve = False, #or filename
                     ):

    ok_rate  = ('hist', 'line', 'both')
    ok_rast  = ('line', 'dot')
    if plot_type not in ok_rate:
        raise ValueError(f"plot_type must be {ok_rate}")
    if raster_style not in ok_rast:
        raise ValueError(f"raster_style must be {ok_rast}")

    sim_duration_ms = sim_params['tstop']
    bin_width = sim_params['bins']
    stim_start = sim_params['delay'] + 100 # Start/stop of stim manual for now, could be auto?
    stim_stop = sim_params['delay'] + 550

    bw_s  = bin_width / 1000
    bins  = np.arange(0, sim_duration_ms + bin_width, bin_width)
    centers = bins[:-1] + .5 * bin_width
    groups = list(all_param_data.keys())
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    g2col  = {g: next(colors) for g in groups}

    # ----------------------------- combined figure if raster -----------------
    if plot_raster:
        fig, (axRate, axRaster) = plt.subplots(
            2, 1, figsize=(6, 8),
            sharex=True, gridspec_kw={'height_ratios': [1, 1]}
        )
    else:
        fig, axRate = plt.subplots(figsize=(6, 4))

    # ----------------------------- rate panel ------------------------------
    for key in groups:
        trial_spikes = all_param_data.get(key, [])

        # Build per-trial rate matrix (N, T)
        per_trial_rates = []
        for tr in trial_spikes:
            tr = np.asarray(tr)
            counts, _ = np.histogram(tr, bins=bins)
            rate = counts / bw_s  # Hz per single trial
            per_trial_rates.append(rate)

        Y = np.vstack(per_trial_rates)  # shape (N_trials, T)
        x = centers.copy()

        # Moving Average
        if win_size:
            Y_smooth = []
            for y in Y:
                ys = moving_average(y, win_size=win_size, bin_width=bin_width, mode='center')
                Y_smooth.append(ys)
            T_new = min(len(y) for y in Y_smooth) # After smoothing, length shrinks: align x
            Y = np.vstack([y[:T_new] for y in Y_smooth])
            drop = (len(x) - T_new) // 2
            x = x[drop : drop + T_new]

        if (set_color is not None) and (len(param_study['param_vals'])==1):
            col = set_color
        else:
            col = g2col[key]

        # # Histogram (Could make only outline, with shaded sem like line)
        # if plot_type in ('hist', 'both',): 
        #     mean_y = Y.mean(axis=0)
        #     axRate.bar(x, mean_y, width=bin_width, color=col, alpha=alpha, label=key)

        # # Line graph
        # if plot_type in ('line', 'both',):
        #     if shade is not None:
        #         _shaded_mean_band(
        #             axRate, x, Y,
        #             label=key, color=col,
        #             shade=shade[0], alpha=shade[1],
        #             norm_fr=norm_fr, norm_scope='post_mean'
        #         )
        #     else: # no shading: just the mean line
        #         axRate.plot(x, Y.mean(axis=0), color=col, lw=2, label=key)

        # stats (normalize or not via norm_fr; shade mode/alpha)
        mean, lo, hi = _mean_band_stats(Y, shade=shade_mode,
            norm_fr=norm_fr, norm_scope='post_mean')

        # histogram (uses normalized mean for consistency)
        if plot_type in ('hist', 'both'):
            axRate.bar(x, mean, width=bin_width, color=col, alpha=alpha, label=key)

        # mean line (always)
        print(type(mean))
        if plot_type in ('line', 'both'):
            axRate.plot(x, mean, color=col, lw=2, label=key)
            # optional shaded band
            if (shade_mode is not None) and (lo is not None) and (hi is not None):
                axRate.fill_between(x, lo, hi, color=col, alpha=0.25, linewidth=0)

    # Plot additional bio curves
    if plot_bio and plot_bio[0]:
        bio_data = plot_bio[2]
        if norm_fr is not None:
            bio_data = normalize(bio_data,norm_offset=-1*bio_data[0])
        axRate.plot(plot_bio[1] * 1000, bio_data, 'k', lw=2, label='In-Vivo Input')

    # Set up fig
    axRate.set_ylabel("Avg rate (Hz)")
    if len(groups) > 1 or (plot_bio and plot_bio[0]):
        axRate.legend()
    axRate.grid()
    for vline in [stim_start,stim_stop]: axRate.axvline(x=vline,color='k',linestyle='-',linewidth=1)
    if plot_window is not None:
        axRate.set_xlim(plot_window['x'][0],plot_window['x'][1])
        axRate.set_ylim(plot_window['y'][0],plot_window['y'][1])

    if len(param_study['param_vals']) > 1:
        axRate.set_title(f"Parametric Analysis (Param: {param_study['param_type']} "
                f"| Trials: {param_study['n_trials']})")
    else:
        axRate.set_title(f"Multi-Trial Average Firing Rate "
                f"(Trials: {param_study['n_trials']} | Win. = {win_size} ms)")


    # ----------------------------- raster ----------------------------------
    if plot_raster:
            y0 = 0
            handles = []
            for key in groups:

                if (set_color is not None) and (len(param_study['param_vals'])==1):
                    col = set_color
                else:
                    col = g2col[key]

                for tr in all_param_data.get(key, []):
                    if raster_style == 'line':
                        axRaster.vlines(tr, y0 + .5, y0 + 1.5, color=col, lw=10)
                    else:
                        axRaster.scatter(tr, np.full_like(tr, y0 + 1),
                                        color=col, s=6, marker='.')
                    y0 += 1
                handles.append(Line2D([0], [0],
                                    color=col,
                                    marker='|' if raster_style == 'line' else '.',
                                    linestyle='', markersize=8, label=key))
                
            axRaster.set_ylim(0.5, y0 + .5)
            axRaster.set_xlabel("Time (ms)")
            axRaster.set_ylabel("Trial #")
            if len(handles) > 1:
                axRaster.legend(handles=handles, loc='upper left')
            axRaster.grid(axis='x')
            for vline in [stim_start,stim_stop]: axRaster.axvline(x=vline,color='k',linestyle='-',linewidth=1)
    else:
        axRate.set_xlabel("Time (ms)")
        

    # plt.tight_layout()


# ────────────────────────────────────────────────────────────────────────────
#  Comparing multiple datasets plotting
# ────────────────────────────────────────────────────────────────────────────
def plot_compare_multi(
        plotted_files,
        sim_params,
        win_size=25, # None
        plot_bio = (False),
        plot_window = None,
        shade_mode = None,
        ):
    

    plt.figure(figsize=(6,4))

    for pfs in plotted_files:
        with open(plotted_files[pfs]['data'], "rb") as f:
            all_param_data = pickle.load(f)
        
        bin_width = sim_params['bins']
        sim_dur = plotted_files[pfs]['dur']
        stim_start = sim_params['delay'] + 100 # Start/stop of stim manual for now, could be auto?
        stim_stop = sim_params['delay'] + 550
        norm_fr = plotted_files[pfs]['norm_fr']

        colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        if plotted_files[pfs]['color'] is not None:
            col = plotted_files[pfs]['color']
        else:
            col = next(colors)

        bw_s  = bin_width / 1000
        bins  = np.arange(0, sim_dur + bin_width, bin_width)
        centers = bins[:-1] + .5 * bin_width
        trial_spikes = all_param_data[list(all_param_data.keys())[0]] # Works for only one param right now
        # Could create color gradient based on col for multiple params in groups

        # Build per-trial rate matrix (N, T) for variability data
        per_trial_rates = []
        for tr in trial_spikes:
            tr = np.asarray(tr)
            counts, _ = np.histogram(tr, bins=bins)
            rate = counts / bw_s  # Hz per single trial
            per_trial_rates.append(rate)

        Y = np.vstack(per_trial_rates)  # shape (N_trials, T)
        x = centers.copy()

        # Moving Average
        if win_size:
            Y_smooth = []
            for y in Y:
                ys = moving_average(y, win_size=win_size, bin_width=bin_width, mode='center')
                Y_smooth.append(ys)
            T_new = min(len(y) for y in Y_smooth) # After smoothing, length shrinks: align x
            Y = np.vstack([y[:T_new] for y in Y_smooth])
            drop = (len(x) - T_new) // 2
            x = x[drop : drop + T_new]

        # get mean & band with your chosen normalization scope
        mean, lo, hi = _mean_band_stats(
            Y, shade=shade_mode,
            norm_fr=norm_fr,           # None => no normalization
            norm_scope='post_mean'     # or 'per_trial' if you prefer
        )

        # 1) plot the mean line (always)
        plt.plot(x, mean, color=col, lw=2, label=f'{pfs} ({len(trial_spikes)} trials)')

        # 2) add shaded band if requested
        if shade_mode is not None and lo is not None and hi is not None:
            plt.fill_between(x, lo, hi, color=col, alpha=0.25, linewidth=0)

        # if shade_mode is not None:
        #     _mean_band_stats(
        #         plt, x, Y,
        #         label=label, color=col,
        #         shade=shade_mode, alpha=0.25,
        #         norm_fr=norm_fr, norm_scope='post_mean'
        #     )
        # else: # no shading: just the mean line
        #     plt.plot(x, Y.mean(axis=0), color=col, lw=2, label=label)

    #  Plot additional bio curves
    if plot_bio and plot_bio[0]:
        bio_data = plot_bio[2]
        if norm_fr is not None:
            bio_data = normalize(bio_data,norm_offset=-1*bio_data[0])
        plt.plot(plot_bio[1] * 1000, bio_data, 'k', lw=2, label='In-Vivo Input')

    for vline in [stim_start,stim_stop]: plt.axvline(x=vline,color='k',linestyle='-',linewidth=1)
    # Set up fig
    plt.ylabel('Avg Rate (Hz)')
    plt.xlabel('Time (ms)')
    plt.legend()
    plt.grid()
    if plot_window is not None:
        plt.xlim(plot_window['x'][0],plot_window['x'][1])
        plt.ylim(plot_window['y'][0],plot_window['y'][1])
    plt.title(f'Comparison of Average Firing Rates (Win: {win_size} ms)')
        
