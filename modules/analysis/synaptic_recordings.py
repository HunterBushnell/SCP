"""Read measured synaptic traces without reconstructing a NEURON model."""
from __future__ import annotations

import numpy as np


def synaptic_recording_payload(results, trial_idx=None):
    payload = results.get('syn_recordings')
    if payload is None:
        entries = results.get('syn_recordings_by_trial') or []
        if trial_idx is None:
            payload = entries[0].get('recordings') if entries else None
        else:
            matches = [entry for entry in entries if entry.get('trial_idx') == int(trial_idx)]
            payload = matches[0].get('recordings') if matches else None
    elif trial_idx not in (None, 0):
        return None
    if payload is None:
        return None
    if payload.get('format_version') != 1:
        raise ValueError('Unsupported synaptic recording format_version; use the matching SCP reader.')
    time = np.asarray(payload.get('T', []), dtype=float)
    if time.ndim != 1 or not np.isfinite(time).all() or (time.size > 1 and np.any(np.diff(time) <= 0)):
        raise ValueError('Synaptic recordings require a finite, increasing one-dimensional time axis.')
    for name, group in payload.get('groups', {}).items():
        for var, values in group.get('aggregate', {}).items():
            array = np.asarray(values, dtype=float)
            if array.ndim != 1 or array.size != time.size:
                raise ValueError(f'Synaptic aggregate {name}.{var} does not match its recorded time axis.')
        for var, values in group.get('samples', {}).items():
            array = np.asarray(values, dtype=float)
            if array.ndim != 2 or array.shape[1] != time.size:
                raise ValueError(f'Synaptic samples {name}.{var} do not match their recorded time axis.')
    return payload


def recorded_group_totals(results, trial_idx=None):
    """Sum complete recorded groups only; never promote samples to totals."""
    payload = synaptic_recording_payload(results, trial_idx)
    if payload is None:
        return None
    time = np.asarray(payload['T'], dtype=float)
    groups = {name: group for name, group in payload.get('groups', {}).items()
              if int(group.get('n_synapses', 0)) > 0}
    totals, incomplete = {}, {}
    for var, key in (('i', 'I'), ('g', 'G')):
        arrays, missing = [], []
        for name, group in groups.items():
            coverage = group.get('coverage', {}).get(var, {})
            n_syn = int(group['n_synapses'])
            if (var not in group.get('aggregate', {}) or group.get('aggregation', {}).get(var) != 'sum'
                    or coverage.get('available_count') != n_syn or coverage.get('aggregate_count') != n_syn):
                missing.append(name)
            else:
                arrays.append(np.asarray(group['aggregate'][var], dtype=float))
        if arrays and not missing:
            totals[key] = np.sum(arrays, axis=0)
        if missing:
            incomplete[key] = missing
    return {'T': time, 'series': totals, 'incomplete_groups': incomplete,
            'recorded_groups': list(groups), 'scope': 'recorded_groups',
            'aggregation_verified': True}


def summarize_synaptic_recordings(results, trial_idx=None):
    """Descriptive per-group coverage/statistics; no new physiological metrics."""
    payload = synaptic_recording_payload(results, trial_idx)
    if payload is None:
        return {'available': False, 'groups': {}, 'trial_idx': trial_idx}
    def stats(values):
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        return {'n_finite': int(finite.size), 'mean': float(np.mean(finite)) if finite.size else None,
                'peak_abs': float(np.max(np.abs(finite))) if finite.size else None}
    groups = {}
    for name, group in payload.get('groups', {}).items():
        variables = {}
        for var in sorted(set(group.get('coverage', {})) | set(group.get('aggregate', {})) | set(group.get('samples', {}))):
            samples = np.asarray(group.get('samples', {}).get(var, []), dtype=float)
            variables[var] = {'unit': payload.get('units', {}).get(var, ''),
                'aggregation': group.get('aggregation', {}).get(var),
                'coverage': group.get('coverage', {}).get(var, {}),
                'aggregate': stats(group['aggregate'][var]) if var in group.get('aggregate', {}) else None,
                'sample_count': int(samples.shape[0]) if samples.ndim == 2 else 0}
        groups[name] = {'n_synapses': group.get('n_synapses'), 'variables': variables,
                        'synapse_ids': group.get('synapse_ids', []), 'mode': group.get('mode')}
    return {'available': bool(groups), 'groups': groups, 'trial_idx': trial_idx,
            'time_n': len(payload['T']), 'scope': 'recorded_groups'}


def format_synaptic_recording_table(summary, *, title='Measured synaptic recordings'):
    lines = [f'**{title}**', '',
             '| Group | Variable | Unit | Available / placed | Aggregate | Mean | Peak abs | Individual samples |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for name, group in summary.get('groups', {}).items():
        for var, item in group['variables'].items():
            aggregate = item['aggregate'] or {}
            def display(value):
                return '—' if value is None else f'{value:.4g}'
            lines.append(f'| {name} | {var} | {item["unit"]} | '
                         f'{item["coverage"].get("available_count", 0)} / {group["n_synapses"]} | '
                         f'{item["aggregation"] if item["aggregate"] is not None else "—"} | '
                         f'{display(aggregate.get("mean"))} | {display(aggregate.get("peak_abs"))} | '
                         f'{item["sample_count"]} |')
    lines += ['', 'Aggregates cover available variables in the recorded groups; '
              'partial coverage is not a whole-cell total. Sampled traces remain individual synapses.']
    return '\n'.join(lines)


def plot_synaptic_recordings(results, *, trial_idx=None, groups=None, variables=('i', 'g'), include_samples=False):
    """Plot each variable on its own native-unit axis, using recorded time."""
    import matplotlib.pyplot as plt
    payload = synaptic_recording_payload(results, trial_idx)
    if payload is None:
        raise ValueError('No measured synaptic recordings for the requested trial. Enable syn_recording before running.')
    names = list(payload.get('groups', {})) if groups is None else ([groups] if isinstance(groups, str) else list(groups))
    unknown = set(names) - set(payload.get('groups', {}))
    if unknown:
        raise ValueError(f'Unknown recorded synaptic groups: {sorted(unknown)}')
    variables = (variables,) if isinstance(variables, str) else tuple(variables)
    if not variables:
        raise ValueError('Select at least one synaptic variable to plot.')
    fig, axes = plt.subplots(len(variables), 1, squeeze=False, sharex=True)
    time = np.asarray(payload['T'], dtype=float)
    for var, ax in zip(variables, axes[:, 0]):
        for name in names:
            group = payload['groups'][name]
            coverage = group.get('coverage', {}).get(var, {})
            if var in group.get('aggregate', {}):
                label = f'{name} {group.get("aggregation", {}).get(var, "aggregate")} '
                label += f'({coverage.get("aggregate_count", 0)}/{group["n_synapses"]})'
                ax.plot(time, group['aggregate'][var], label=label)
            if include_samples:
                for i, values in enumerate(group.get('samples', {}).get(var, [])):
                    # Missing variables may remove selected synapses: use per-variable IDs when available.
                    ids = coverage.get('sample_synapse_ids', group.get('synapse_ids', []))
                    identity = ids[i] if i < len(ids) else i
                    ax.plot(time, values, alpha=.5, label=f'{name} synapse {identity}')
        ax.set_ylabel(f'{var} ({payload.get("units", {}).get(var, "native units")})')
        if ax.lines:
            ax.legend()
        else:
            ax.text(.5, .5, 'No recorded aggregate; enable include_samples for sample-only data.',
                    transform=ax.transAxes, ha='center', wrap=True)
    axes[-1, 0].set_xlabel('Time (ms)')
    fig.tight_layout()
    return fig, axes[:, 0]
