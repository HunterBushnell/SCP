"""Public native examples: provenance, configuration, and non-scientific smoke tests.

ACT's Python package and external source checkout are deliberately not imported.
Numerical checks use temporary protocol overrides, never example acceptance targets.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
ACT_COMMIT = "2b3fdc190107b8242e3496329cfdf240abd09694"
FAMILIES = ("ACT_Spiker", "ACT_BursterCa", "ACT_LAA", "ACT_L5", "ACT_SST")
HOC = FAMILIES[:3]
SENTINEL = "SCP_STAGED_NATIVE_SMOKE="

SMOKE_PROGRAM = textwrap.dedent(r'''
    import hashlib
    import json
    from pathlib import Path
    import re
    import sys
    import tempfile
    import numpy as np

    root, tune = map(Path, sys.argv[1:3])
    sys.path.insert(0, str(root))
    from modules.setup.mechanisms import compile_modfiles
    from modules.model import validate_cell
    from modules.simulation import SimulationSession, SimulationOptions
    from modules.simulation.current_injection import resolve_sim_conditions
    from modules.tuning import run_passive_protocol, run_active_protocol
    from modules.tuning.bmtool_synapse_adapter import _bmtool_cell_facade
    from modules.analysis.analysis import load_cell_and_geometry
    from modules.simulation.result_loading import load_results
    from modules.simulation.result_paths import _result_filename
    from modules.workspaces import infer_workspace, result_workspace_identity

    cfg = json.loads((tune/'cell_configs/cell_config.json').read_text())
    # Check provenance before compilation/construction can do anything.
    for path in list(tune.glob('*PROVENANCE*.json')):
        source = json.loads(path.read_text())
        for entry in source.get('files', []):
            dest = tune/entry.get('destination_path', '')
            expected = entry.get('current_sha256') or entry.get('sha256')
            if expected and dest.is_file():
                assert hashlib.sha256(dest.read_bytes()).hexdigest() == expected, str(dest)
    compile_modfiles(tune, recompile=False, load_dll=True, allow_missing=True, cell_config=cfg)
    overrides = {'tstart': 0., 'tstop': 20., 'dt': .025,
        'stim_start_ms': 5., 'stim_duration_ms': 10., 'n_trials': 1,
        'save_output': False, 'plots_profile': 'off', 'save_output_metrics': False,
        'snapshot': {'enabled': False}, 'append': {'enabled': False},
        'iclamp': {'enabled': True, 'amp_nA': .05, 'delay_ms': 5., 'dur_ms': 10.,
                   'tstop_ms': 20., 'dt_ms': .025, 'record_currents': True}}
    session = SimulationSession.from_tune(tune, options=SimulationOptions(iclamp=True,
        load_mechanisms=False, n_trials=1, sim_overrides=overrides))
    # No checkpoint inheritance during an intrinsic staging assay.
    session.sim_cfg_preview.update(append={'enabled': False}, append_enabled=False, append_to=None)
    session.prepare()
    cell = validate_cell(session.cell)
    initial_geometry = [(float(s.L), float(s.diam), int(s.nseg)) for s in cell.all]
    assert cell.soma and cell.all and all(s in cell.all for s in cell.soma)
    assert len({s.name() for s in cell.all}) == len(cell.all)
    conditions = resolve_sim_conditions(cell, session.sim_cfg)
    def check_trace(t, v, currents):
        assert len(t) == len(v) == 801, (len(t), len(v))
        assert np.isfinite(t).all() and np.isfinite(v).all()
        assert np.isclose(v[0], conditions['v_init_mV'], atol=1e-6), (v[0], conditions)
        assert np.isclose(t[-1], 20.)
        for key, values in currents.items():
            assert len(values) == len(t) and np.isfinite(values).all(), key
    params = {'stim_amp': .05, 'stim_delay': 5., 'stim_dur': 10.,
              'h_tstop': 20., 'h_dt': .025, 'conditions': conditions}
    for runner, amps in ((run_passive_protocol, [-25., -50.]),
                         (run_active_protocol, [50.]),
                         (run_active_protocol, [0., 50., 100.])):
        cell.h.celsius = 5.
        sweeps = runner(cell=cell, sim_params=params, sim_amps=amps)
        for amp in amps:
            check_trace(sweeps['T'][amp], sweeps['V'][amp], sweeps['I'][amp])
        assert np.isclose(cell.h.celsius, conditions['celsius_C'])
    result = session.run()
    check_trace(result['traces']['T'], result['traces']['V'], result['iclamp']['I'])
    assert np.isclose(cell.h.celsius, conditions['celsius_C'])
    identity = infer_workspace(tune).identity
    assert result_workspace_identity(result) == identity
    stem = _result_filename(result['sim_cfg'], 'smoke', '.pkl')
    if identity['project_name']:
        assert stem.startswith(identity['project_name']+'_'+identity['cell_name']+'_')
    inspected, geometry, _ = load_cell_and_geometry(tune, cell_override=cell)
    assert inspected is cell and geometry['meta']['counts']['all_secs'] == len(cell.all)
    facade = _bmtool_cell_facade(cell)
    assert tuple(facade.soma) == tuple(cell.soma) and tuple(facade.all) == tuple(cell.all)
    assert initial_geometry == [(float(s.L), float(s.diam), int(s.nseg)) for s in cell.all]
    mechanisms = sorted({name for s in cell.all for name in s.psection()['density_mechs']})
    assert mechanisms
    if cfg.get('cell_loader') == 'hoc_template':
        text = (tune/cfg['paths']['hoc_template']).read_text()
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
        text = re.sub(r'//[^\n]*', '', text)
        name = cfg['hoc_template']['template_name']
        body = re.search(r'\bbegintemplate\s+'+re.escape(name)+r'\b(.*?)\bendtemplate\s+'+re.escape(name)+r'\b', text, re.S).group(1)
        expected = set(re.findall(r'\binsert\s+(\w+)', body))
        ion_mechanisms = {ion+'_ion' for section in cell.all for ion in section.psection()['ions']}
        assert expected <= set(mechanisms) | ion_mechanisms, (expected-set(mechanisms)-ion_mechanisms, mechanisms)
    # Saving/analysis is checked outside the tune, without retaining a dataset.
    with tempfile.TemporaryDirectory(prefix='scp-staged-result-') as temporary:
        session.result['sim_cfg'].update(output='staging_smoke', output_stem='staging_smoke',
            save_output=True, append_enabled=False, append_to=None, plots_profile='off')
        session.options.output_dir = temporary
        saved = session.save()
        assert saved is not None
        restored = load_results(saved)
        assert result_workspace_identity(restored) == identity
        check_trace(restored['traces']['T'], restored['traces']['V'], {})
    print('SCP_STAGED_NATIVE_SMOKE='+json.dumps({'workspace_identity': identity,
        'conditions': conditions, 'section_count': len(cell.all), 'soma_count': len(cell.soma),
        'geometry': initial_geometry, 'mechanisms': mechanisms,
        'current_trace_count': len(result['iclamp']['I']), 'sample_count': 801,
        'passive_active_fi_iclamp': 'finite', 'saved_result_roundtrip': True,
        'step4_facade': True, 'step6_inspection': True}))
''')


def configurations():
    return [(family, variant, ROOT / 'cells' / family / 'tunes' / variant)
            for family in FAMILIES
            for variant in (('orig', 'tuned', 'reference') if family in HOC else ('orig', 'tuned'))]


class ACTExampleConfigurationTests(unittest.TestCase):
    def test_catalog_and_native_hashes_are_self_contained(self):
        for family, variant, tune in configurations():
            with self.subTest(family=family, tune=variant):
                self.assertTrue(tune.is_dir())
                provenance = json.loads((tune / 'SOURCE_PROVENANCE.json').read_text())
                self.assertEqual(provenance['source_repository']['commit'], ACT_COMMIT)
                self.assertFalse(provenance['compiled_outputs_included'])
                self.assertNotIn('/home/', json.dumps(provenance))
                expected_mods = {'ACT_Spiker': 1, 'ACT_BursterCa': 8, 'ACT_LAA': 10,
                                 'ACT_L5': 16, 'ACT_SST': 16}[family]
                self.assertEqual(sum(entry['destination_path'].endswith('.mod')
                                     for entry in provenance['files']), expected_mods)
                native_files = {path.relative_to(tune).as_posix()
                                for pattern in ('model/*.hoc', 'modfiles/*.mod', '*.swc',
                                                '*.marker', 'manifest.json', '*_fit.json')
                                for path in tune.glob(pattern)}
                self.assertEqual({entry['destination_path'] for entry in provenance['files']}, native_files)
                for entry in provenance['files']:
                    path = tune / entry['destination_path']
                    self.assertTrue(path.resolve().is_relative_to(tune.resolve()))
                    self.assertTrue(path.is_file())
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry['current_sha256'])
                    self.assertEqual(entry['source_sha256'], entry['current_sha256'])
                    self.assertEqual(entry['relationship'], 'exact_copy')
                    self.assertNotIn(Path(entry['destination_path']).suffix, {'.so', '.dll', '.pkl', '.pickle', '.nwb'})

    def test_generic_configs_are_targetless_cell_only_and_not_scientific_protocols(self):
        conditions = {'ACT_Spiker': (6.3, -50.), 'ACT_BursterCa': (6.3, -50.),
                      'ACT_LAA': (31., -70.), 'ACT_L5': (34., -90.),
                      'ACT_SST': (34., -79.65586853027344)}
        from modules.setup.defaults import default_sim_config
        defaults = default_sim_config(cell_name='generic', specimen_id=None, model_type='')
        for family, variant, tune in configurations():
            with self.subTest(family=family, tune=variant):
                cfg = json.loads((tune / 'cell_configs/cell_config.json').read_text())
                sim = json.loads((tune / 'cell_configs/sim_config.json').read_text())
                target = json.loads((tune / 'cell_configs/target_config.json').read_text())
                self.assertEqual((cfg['cell_name'], cfg['tune'], cfg['color']), (family, variant, 'k'))
                self.assertNotIn('project_name', cfg)
                self.assertEqual(target['target_source']['mode'], 'none')
                self.assertFalse((tune / 'cell_configs/syn_config.json').exists())
                self.assertFalse((tune / 'cell_configs/syn_groups').exists())
                self.assertTrue(sim['iclamp']['enabled'])
                self.assertEqual((sim['conditions']['celsius_C'], sim['conditions']['v_init_mV']), conditions[family])
                for key in ('tstart', 'tstop', 'dt', 'stim_start_ms', 'stim_duration_ms'):
                    self.assertEqual(sim[key], defaults[key])
                for key in ('amp_nA', 'delay_ms', 'dur_ms', 'tstop_ms', 'dt_ms'):
                    self.assertEqual(sim['iclamp'][key], defaults['iclamp'][key])
                if family in HOC:
                    self.assertEqual(cfg['cell_loader'], 'hoc_template')
                    self.assertEqual(cfg['hoc_template']['constructor_args'], [])
                else:
                    self.assertEqual(cfg['cell_loader'], 'allen_manifest')
                    self.assertEqual(cfg['tuning']['soma_diam_multiplier'], 1.)

    def test_hoc_working_copies_and_reference_are_distinguished(self):
        for family in HOC:
            orig, tuned, reference = [ROOT / 'cells' / family / 'tunes' / v for v in ('orig', 'tuned', 'reference')]
            a, b, ref = [json.loads((t / 'SOURCE_PROVENANCE.json').read_text()) for t in (orig, tuned, reference)]
            self.assertEqual(a['files'], b['files'])
            self.assertIn('Untuned', b['model_relationship'])
            self.assertIn('known-reference', ref['model_relationship'])
            self.assertTrue(ref['files'][0]['source_path'].endswith('/target_template.hoc'))

    def test_allen_variants_keep_active_manifest_and_matching_sources(self):
        for family in ('ACT_L5', 'ACT_SST'):
            for variant, upstream in [('orig', 'adb'), ('tuned', 'seg_tuned')]:
                tune = ROOT / 'cells' / family / 'tunes' / variant
                provenance = json.loads((tune / 'SOURCE_PROVENANCE.json').read_text())
                self.assertTrue(all(f'/{upstream}/' in entry['source_path'] for entry in provenance['files']))
                manifest = json.loads((tune / 'manifest.json').read_text())
                for row in manifest['biophys']:
                    for relative in row['model_file']:
                        self.assertTrue((tune / relative).is_file())
                cfg = json.loads((tune / 'cell_configs/cell_config.json').read_text())
                self.assertEqual(cfg['allen']['utils'], 'auto' if family == 'ACT_L5' else 'standard')
                if family == 'ACT_L5':
                    self.assertEqual(cfg['allen']['axon_type'], 'stub')

    def test_examples_are_public_but_generated_products_are_ignored(self):
        for family in FAMILIES:
            cell = ROOT / 'cells' / family
            self.assertTrue((cell / 'README.md').is_file())
            self.assertTrue((cell / 'LICENSE.ACT').is_file())
            native = next((cell / 'tunes/orig').rglob('*.mod'))
            result = subprocess.run(['git', 'check-ignore', '--no-index', str(native.relative_to(ROOT))], cwd=ROOT,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stdout)
            result = subprocess.run(['git', 'check-ignore', '--no-index', f'cells/{family}/tunes/orig/modfiles/x86_64/libnrnmech.so'],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_documentation_distinguishes_staging_from_characterization(self):
        catalog = (ROOT / 'docs/guides/act_examples.md').read_text()
        for phrase in ('staged and smoke-tested', 'not completed SCP',
                       'Untuned', 'fresh process', 'target_source.mode: "none"'):
            # Catalog table uses lowercase "untuned" in prose.
            self.assertIn(phrase.lower(), catalog.lower())
        for family in FAMILIES:
            self.assertIn(family, catalog)
            readme = (ROOT / 'cells' / family / 'README.md').read_text()
            self.assertIn('SOURCE_PROVENANCE.json', readme)
            self.assertIn('LICENSE.ACT', readme)
        notices = (ROOT / 'THIRD_PARTY_NOTICES.md').read_text()
        self.assertIn(ACT_COMMIT, notices)
        self.assertIn("does not relicense these native assets", notices)


class ACTExampleRuntimeTests(unittest.TestCase):
    def test_every_tune_in_a_fresh_process(self):
        if importlib.util.find_spec('neuron') is None:
            self.skipTest('NEURON unavailable for native example checks')
        from modules.setup.mechanisms import find_compiled_mechanism_dll, find_nrnivmodl
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
        for family, variant, tune in configurations():
            with self.subTest(family=family, tune=variant), tempfile.TemporaryDirectory(prefix='scp-act-example-') as temporary:
                if family not in HOC and importlib.util.find_spec('allensdk') is None:
                    self.skipTest('AllenSDK unavailable for this Allen example')
                selected = tune
                if find_compiled_mechanism_dll(tune) is None:
                    if find_nrnivmodl() is None or not (shutil.which('c++') or shutil.which('g++')):
                        self.skipTest('Native example libraries missing and nrnivmodl/C++ compiler unavailable')
                    selected = Path(temporary) / family / 'tunes' / variant
                    shutil.copytree(tune, selected)
                result = subprocess.run([sys.executable, '-c', SMOKE_PROGRAM, str(ROOT), str(selected), 'compile'],
                                        cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout[-10000:] + '\n' + result.stderr[-10000:])
                line = next(line for line in result.stdout.splitlines() if line.startswith(SENTINEL))
                report = json.loads(line[len(SENTINEL):])
                self.assertEqual(report['sample_count'], 801)
                self.assertGreater(report['current_trace_count'], 0)
                self.assertEqual(report['workspace_identity']['cell_name'], family)


if __name__ == '__main__':
    unittest.main()
