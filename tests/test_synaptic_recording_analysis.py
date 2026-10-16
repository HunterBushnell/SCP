from __future__ import annotations

import copy
import unittest
import numpy as np

from modules.analysis import analysis
from modules.analysis.synaptic_recordings import synaptic_recording_payload, recorded_group_totals


def result():
    payload = {'format_version': 1, 'T': np.arange(4.), 'units': {'i': 'nA', 'g': 'uS'}, 'groups': {}}
    for name, value in (('exc', 1.), ('inh', -2.)):
        payload['groups'][name] = {'n_synapses': 2, 'mode': 'both', 'synapse_ids': [10],
            'aggregate': {'i': np.full(4, value), 'g': np.ones(4)},
            'samples': {'i': np.full((1, 4), value / 2)},
            'aggregation': {'i': 'sum', 'g': 'sum'},
            'coverage': {var: {'available_count': 2, 'aggregate_count': 2, 'sample_count': 1}
                         for var in ('i', 'g')}}
    return {'mode': 'single', 'sim_cfg': {'stim_start_ms': 1., 'stim_duration_ms': 2.},
            'syn_recordings': payload, 'traces': {}}


class SynapticAnalysisTests(unittest.TestCase):
    def test_sums_use_own_time_and_do_not_require_voltage(self):
        data = result()
        totals = recorded_group_totals(data)
        np.testing.assert_array_equal(totals['series']['I'], np.full(4, -1.))
        np.testing.assert_array_equal(totals['series']['G'], np.full(4, 2.))
        summary = analysis.summarize_total_synaptic_traces(data)
        self.assertTrue(summary['aggregation_verified'])
        self.assertEqual(summary['scope'], 'recorded_groups')
        self.assertEqual(summary['series']['I']['mean'], -1.)
        self.assertEqual(summary['time_n'], 4)

    def test_partial_or_sample_only_groups_are_not_promoted_to_totals(self):
        data = result()
        group = data['syn_recordings']['groups']['inh']
        group['coverage']['i']['available_count'] = 1
        self.assertNotIn('I', recorded_group_totals(data)['series'])
        group['aggregate'].pop('g')
        self.assertEqual(recorded_group_totals(data)['series'], {})
        summary = analysis.summarize_synaptic_recordings(data)
        self.assertTrue(summary['available'])
        self.assertEqual(summary['groups']['inh']['variables']['i']['sample_count'], 1)
        table = analysis.format_synaptic_recording_table(summary)
        self.assertIn('1 / 2', table)
        self.assertIn('not a whole-cell total', table)

    def test_legacy_i_g_are_explicitly_unverified(self):
        data = {'mode': 'single', 'sim_cfg': {}, 'traces': {'T': np.arange(4.), 'I': np.ones(4)}}
        summary = analysis.summarize_total_synaptic_traces(data)
        self.assertFalse(summary['aggregation_verified'])
        self.assertEqual(summary['scope'], 'legacy_aggregation_unknown')
        self.assertIn('not verified synaptic totals', analysis.format_total_synaptic_trace_table(summary))

    def test_trial_selection_does_not_substitute_a_different_trial(self):
        data = result()
        payload = data.pop('syn_recordings')
        data['mode'] = 'multi'
        data['syn_recordings_by_trial'] = [{'trial_idx': 3, 'recordings': payload}]
        self.assertIs(synaptic_recording_payload(data, 3), payload)
        self.assertIsNone(synaptic_recording_payload(data, 0))
        self.assertIs(synaptic_recording_payload(data), payload)

    def test_invalid_time_or_shape_is_not_silently_truncated(self):
        for location in ('T', 'aggregate', 'samples'):
            data = result()
            payload = data['syn_recordings']
            if location == 'T':
                payload['T'] = np.array([0., 2., 1., 3.])
            elif location == 'aggregate':
                payload['groups']['exc']['aggregate']['i'] = np.ones(3)
            else:
                payload['groups']['exc']['samples']['i'] = np.ones((1, 3))
            with self.subTest(location=location), self.assertRaises(ValueError):
                synaptic_recording_payload(data)

    def test_plots_use_saved_units_and_sample_identity(self):
        import matplotlib.pyplot as plt
        data = result()
        fig, axes = analysis.plot_synaptic_recordings(data, groups='exc', include_samples=True)
        self.addCleanup(plt.close, fig)
        self.assertEqual(len(axes), 2)
        self.assertIn('nA', axes[0].get_ylabel())
        self.assertIn('uS', axes[1].get_ylabel())
        self.assertIn('synapse 10', axes[0].lines[1].get_label())
        np.testing.assert_array_equal(axes[0].lines[0].get_xdata(), np.arange(4.))


if __name__ == '__main__':
    unittest.main()
