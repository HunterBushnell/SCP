from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from neuron import h

from modules.model import wrap_cell, validate_cell, load_cell
from modules.model.runtime_cell import prepare_runtime_workspace
from modules.setup import preview_model_sources, register_model_sources, set_model_source_mode, create_working_copy
from modules.setup.step1_prepare import prepare_tune
from modules.simulation.session import SimulationSession, SimulationOptions
from modules.simulation.result_loading import load_results
from modules.tuning import run_passive_protocol, run_active_protocol
from modules.tuning.bmtool_synapse_adapter import _bmtool_cell_facade, prepare_scp_synapse_tuning, create_scp_synapse_tuner
from modules.analysis.analysis import load_cell_and_geometry
from modules.notebooks.pipeline_workflow import prepare_pipeline_notebook, run_fresh_simulation, preview_pipeline_inputs
from scripts.restore_run_state import restore_run_state

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / 'fixtures' / 'custom_models'
CONDITIONS = {'v_init_mV': -70.0, 'celsius_C': 34.0}
SIM = {'conditions': CONDITIONS, 'dt': 0.025, 'tstop': 20.0,
       'plots_profile': 'off',
       'iclamp': {'enabled': True, 'amp_nA': 0.025, 'delay_ms': 5.0, 'dur_ms': 10.0}}


def native_tree(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def factory_config(root, mode='copy', mods=False):
    return {'cell_name': 'Synthetic', 'tune': 'orig', 'cell_loader': 'python_factory',
            'paths': {'python_factory': 'package/cell.py', 'modfiles': 'mechanisms' if mods else None},
            'python_factory': {'callable': 'build_cell', 'args': [], 'kwargs': {'scale': 1.0, 'use_mod': mods},
                               'section_map': {'soma': 'body', 'dend': 'branch', 'all': 'whole_cell'}},
            'model_source': {'mode': mode, 'source_root': str(root), 'folders': ['package', 'mechanisms'],
                             'working_directory': 'package'}}


def scaffold(tune, cfg):
    return prepare_tune(tune_dir=tune, cell_name=cfg['cell_name'], tune_name=tune.name,
        cell_loader=cfg['cell_loader'], source_type='existing', loader_paths=cfg['paths'],
        loader_config={k: cfg[k] for k in ('model_source', 'python_factory', 'hoc_template') if k in cfg},
        v_init_mV=-70.0, celsius_C=34.0, do_synapse_configs=False, target_source_mode='none',
        do_compile_modfiles=False, load_compiled_dll=False, do_validate=False)


class CellAdoptionTests(unittest.TestCase):
    def setUp(self):
        self.soma = h.Section(name='inline_test_body')
        self.soma.L = self.soma.diam = 20
        self.soma.insert('pas')
        self.soma.g_pas, self.soma.e_pas = 0.0001, -70
        self.cell = wrap_cell(self.soma, cell_name='Inline')

    def test_section_owner_and_facade_preserve_native_properties(self):
        before = self.soma.psection()['density_mechs']
        unrelated = h.Section(name='unrelated_inline')
        self.assertIs(self.cell.model, self.soma)
        self.assertEqual(self.cell.all, (self.soma,))
        self.assertEqual(self.cell.dend, ())
        self.assertIs(wrap_cell(self.cell), self.cell)
        self.assertIs(validate_cell(self.cell), self.cell)
        self.assertEqual(before, self.soma.psection()['density_mechs'])
        self.assertNotIn(unrelated, self.cell.all)

    def test_explicit_owner_names_and_collections(self):
        owner = SimpleNamespace(body=self.soma)
        self.assertEqual(wrap_cell(owner, section_map={'soma': 'body'}).all, (self.soma,))
        self.assertEqual(wrap_cell(owner, sections={'soma': self.soma}).all, (self.soma,))
        with self.assertRaises(ValueError):
            wrap_cell(owner, sections={}, section_map={})
        with self.assertRaises(TypeError):
            wrap_cell(owner, sections={'soma': [self.soma(0.5)]})
        with self.assertRaises(ValueError):
            wrap_cell(object())
        with self.assertRaises(AttributeError):
            wrap_cell(owner, section_map={'soma': 'missing'})
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            wrap_cell(SimpleNamespace(soma=self.soma, somatic=h.Section(name='other_body')))
        self.assertEqual(wrap_cell(SimpleNamespace(soma=iter([self.soma]))).all, (self.soma,))
        self.assertEqual(wrap_cell(SimpleNamespace(body=iter([self.soma])),
                                  section_map={'soma': 'body'}).all, (self.soma,))

    def test_passive_active_fi_and_conditions(self):
        params = {'stim_amp': 0.025, 'stim_delay': 5., 'stim_dur': 10.,
                  'h_tstop': 20., 'h_dt': .025, 'conditions': CONDITIONS}
        for runner, amps in ((run_passive_protocol, [-25.]), (run_active_protocol, [25.]),
                             (run_active_protocol, [0., 25.])):
            h.celsius = 10.
            results = runner(cell=self.cell, sim_params=params, sim_amps=amps)
            for amp in amps:
                self.assertEqual(len(results['T'][amp]), 801)
                self.assertEqual(len(results['V'][amp]), 801)
                self.assertTrue(np.isfinite(results['V'][amp]).all())
                self.assertEqual(results['V'][amp][0], -70.)
                for current in results['I'][amp].values():
                    self.assertEqual(len(current), 801)
                    self.assertTrue(np.isfinite(current).all())
            self.assertEqual(h.celsius, 34.)

    def test_duplicate_names_do_not_merge_sections(self):
        first, second = h.Section(name='same_name'), h.Section(name='same_name')
        owner = SimpleNamespace(soma=[first, second])
        cell = wrap_cell(owner)
        self.assertEqual(len(cell.all), 2)
        from modules.model.geometry import define_geometry
        self.assertEqual(define_geometry(cell, {})['meta']['counts']['all_secs'], 2)
        cell.all = (first,)
        with self.assertRaisesRegex(ValueError, 'every mapped section'):
            validate_cell(cell)
        with self.assertRaisesRegex(ValueError, 'None is not a cell'):
            wrap_cell(None)

    def test_auto_workspace_requires_explicit_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp) / 'tune'
            with self.assertRaises(ValueError):
                SimulationSession.from_cell(self.cell, tune_dir=tune)
            self.assertFalse(tune.exists())
            prepare_runtime_workspace(tune, self.cell, SIM)
            cfg = json.loads((tune / 'cell_configs/cell_config.json').read_text())
            self.assertEqual(cfg['cell_loader'], 'runtime_cell')
            with self.assertRaisesRegex(RuntimeError, 'CLI/SLURM'):
                load_cell(cfg, base_dir=tune)

    def test_step5_save_does_not_rebuild_and_is_labeled(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch('modules.simulation.session.load_cell', side_effect=AssertionError('must not rebuild')):
                session = SimulationSession.from_cell(self.cell, tune_dir=tmp, sim_config=SIM,
                    options=SimulationOptions(iclamp=True, force_save=True, output_stem='live'))
                result = session.run()
                saved = session.save()
            self.assertIs(session.cell, self.cell)
            self.assertTrue(np.isfinite(result['traces']['V']).all())
            loaded = load_results(saved)
            self.assertTrue(loaded['meta']['model_provenance']['session_only'])
            run_dir = saved.parent
            self.assertTrue((run_dir / 'model_state.json').is_file())
            artifacts = json.loads((run_dir / 'model_artifacts/manifest.json').read_text())
            self.assertFalse(artifacts['construction_recipe_available'])
            self.assertEqual(artifacts['artifacts'], [])

    def test_override_preserves_existing_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            (tune / 'cell_configs').mkdir()
            path = tune / 'cell_configs/cell_config.json'
            original = {'cell_name': 'Existing', 'cell_loader': 'hoc_template',
                        'paths': {'hoc_template': 'not-loaded.hoc'}}
            path.write_text(json.dumps(original))
            session = SimulationSession.from_cell(self.cell, tune_dir=tune, sim_config=SIM)
            session.run()
            self.assertEqual(json.loads(path.read_text()), original)
            self.assertEqual(session.cell_config['cell_loader'], 'runtime_cell')

    def test_step4_and_step6_adopt_without_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch('modules.tuning.bmtool_synapse_adapter.build_synapse_test_cell', side_effect=AssertionError('must not rebuild')):
                session = prepare_scp_synapse_tuning(cell_name='Inline', tune_dir_override=tmp,
                    cell_override=self.cell, runtime_sim_config=SIM, repo_root=ROOT)
            facade = _bmtool_cell_facade(session.cell)
            self.assertEqual(facade.soma, [self.soma])
            self.assertEqual(facade.all, [self.soma])
            with mock.patch('modules.model.load_cell', side_effect=AssertionError('must not rebuild')):
                cell, geom, _ = load_cell_and_geometry(Path(tmp), cell_override=self.cell)
            self.assertIs(cell, self.cell)
            self.assertEqual(geom['meta']['counts']['all_secs'], 1)

    def test_compact_live_run_is_in_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = prepare_pipeline_notebook(repo_root=ROOT, cell_name='Inline', tune_name='orig',
                tune_dir_override=tmp, cell_override=self.cell, runtime_sim_config=SIM)
            with mock.patch('modules.notebooks.pipeline_workflow.subprocess.Popen', side_effect=AssertionError('no worker')):
                result = run_fresh_simulation(state, iclamp=True, output_stem='compact_live')
            self.assertEqual(result.command, ('in-process',))
            self.assertTrue(result.manifest_path.is_file())
            self.assertTrue(result.results['meta']['model_provenance']['session_only'])


    def test_live_synapses_preview_and_multi_trial_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            # A caller-owned native input remains alive on the adopted owner.
            # This does not extend SCP's existing initW/VecStim synapse contract.
            owner = SimpleNamespace(soma=self.soma, syn=h.Exp2Syn(self.soma(.5)), stim=h.NetStim())
            owner.stim.number, owner.stim.start, owner.stim.interval = 2, 5., 5.
            owner.connection = h.NetCon(owner.stim, owner.syn)
            owner.connection.weight[0] = .0001
            adopted = wrap_cell(owner, cell_name='Inline')
            cfg = deepcopy(SIM)
            cfg['iclamp']['enabled'] = False
            state = prepare_pipeline_notebook(repo_root=ROOT, cell_name='Inline', tune_name='orig',
                tune_dir_override=tmp, cell_override=adopted, runtime_sim_config=cfg)
            synapse_config = {'synthetic_input': {'state': True,
                'syns': {'type': 'Exp2Syn', 'N_syn': 2, 'segs': 'soma',
                         'dist_func': {'kind': 'uniform', 'params': {'c': 1.}},
                         'params': {'wt_mean': .0001, 'wt_std': 0.}},
                'input_blocks': [{'name': 'synthetic', 'start_ms': 0., 'stop_ms': 20.,
                                  'mode': 'homogeneous_poisson', 'rate_hz': 100.}]}}
            (Path(tmp) / 'cell_configs/syn_config.json').write_text(json.dumps(synapse_config))
            with mock.patch('modules.notebooks.pipeline_workflow.subprocess.Popen', side_effect=AssertionError('no worker')):
                preview = preview_pipeline_inputs(state, seed=7)
                self.assertEqual(preview.command, ('in-process',))
                self.assertEqual(preview.summary['total_n_syn'], 2)
                (Path(tmp) / 'cell_configs/syn_config.json').write_text('{}')
                result = run_fresh_simulation(state, iclamp=False, n_trials=2, seed=7, output_stem='synaptic_live')
            self.assertEqual(result.results['mode'], 'multi')
            self.assertIs(state.cell, adopted)
            self.assertIs(state.cell.model, owner)
            for trace in result.results['traces']['V']:
                self.assertTrue(np.isfinite(trace).all())
                self.assertGreater(float(np.max(trace)), -70.)

    def test_optional_bmtool_receives_same_cell_facade(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = prepare_scp_synapse_tuning(cell_name='Inline', tune_dir_override=tmp,
                cell_override=self.cell, runtime_sim_config=SIM, repo_root=ROOT)
            connections = {'synthetic': {'spec_settings': {'level_of_detail': 'Exp2Syn', 'sec_id': 0, 'sec_x': .5}}}
            with mock.patch('modules.tuning.bmtool_synapse_adapter.import_bmtool_synapse_api',
                            return_value=(lambda **kwargs: kwargs, None)):
                tuner = create_scp_synapse_tuner(session, conn_type_settings=connections,
                    connection='synthetic', slider_vars=[], other_vars_to_record=[])
            self.assertIs(tuner['hoc_cell'].soma[0], self.soma)

    def test_runtime_save_does_not_copy_unused_allen_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = SimulationSession.from_cell(self.cell, tune_dir=tmp, sim_config=SIM,
                options=SimulationOptions(iclamp=True, force_save=True))
            session.run()
            with mock.patch('modules.simulation.result_saving._copy_fit_json_sidecar', side_effect=AssertionError('unused recipe')):
                self.assertTrue(session.save().is_file())


class ExternalSourcesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'upstream'
        shutil.copytree(FIXTURE, self.source)
        self.tune = self.root / 'cells/Synthetic/tunes/orig'

    def test_preview_reference_snapshot_and_independent_working_copy(self):
        cfg = factory_config(self.source, mode='reference')
        before = native_tree(self.source)
        preview = preview_model_sources(self.tune, cfg)
        self.assertFalse(self.tune.exists())
        self.assertEqual([p['source_path'] for p in preview['files']], sorted(p['source_path'] for p in preview['files']))
        register_model_sources(self.tune, cfg, dry_run=False)
        self.assertTrue((self.tune / 'model_sources/package/cell.py').is_file())
        self.assertEqual(before, native_tree(self.source))
        set_model_source_mode(self.tune, 'copy')
        copied = create_working_copy(source_tune_dir=self.tune)
        target = Path(copied['target']) / 'model_sources/package/length.txt'
        target.write_text('99\n')
        self.assertEqual((self.tune / 'model_sources/package/length.txt').read_text(), '20.0\n')
        self.assertEqual(before, native_tree(self.source))

    def test_refresh_is_explicit_and_backed_up(self):
        cfg = factory_config(self.source)
        register_model_sources(self.tune, cfg, dry_run=False)
        path = self.source / 'package/length.txt'
        path.write_bytes(b'22.0\r\n')
        with self.assertRaisesRegex(ValueError, 'not overwritten'):
            register_model_sources(self.tune, cfg, dry_run=False)
        result = register_model_sources(self.tune, cfg, dry_run=False, refresh=True)
        self.assertTrue(Path(result['backup_dir']).is_dir())
        self.assertEqual((self.tune / 'model_sources/package/length.txt').read_bytes(), b'22.0\r\n')

    def test_traversal_and_external_symlinks_are_rejected(self):
        # A workspace beneath a broad root is safe when selected native trees are disjoint.
        preview_model_sources(self.source / 'safe_tune', factory_config(self.source))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            preview_model_sources(self.source / 'package/unsafe_tune', factory_config(self.source))
        cfg = factory_config(self.source)
        cfg['model_source']['folders'] = ['../outside']
        with self.assertRaises(ValueError):
            preview_model_sources(self.tune, cfg)
        cfg = factory_config(self.source)
        outside = self.root / 'outside.py'
        outside.write_text('pass')
        (self.source / 'package/escape.py').symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            preview_model_sources(self.tune, cfg)

    def test_generated_files_are_excluded(self):
        for name in ('x86_64', '__pycache__', 'output_data', '.scp_build'):
            folder = self.source / 'package' / name
            folder.mkdir()
            (folder / 'generated.py').write_text('pass')
        paths = [p['source_path'] for p in preview_model_sources(self.tune, factory_config(self.source))['files']]
        self.assertFalse(any('generated.py' in p for p in paths))
        cfg = factory_config(self.source)
        cfg['model_source']['folders'] = ['.git']
        with self.assertRaisesRegex(ValueError, 'excluded'):
            preview_model_sources(self.tune, cfg)

    def test_empty_dependency_directories_are_preserved(self):
        (self.source / 'package/empty_native_folder').mkdir()
        register_model_sources(self.tune, factory_config(self.source), dry_run=False)
        self.assertTrue((self.tune / 'model_sources/package/empty_native_folder').is_dir())

    def test_external_allen_fit_resolution_cannot_normalize_original(self):
        if importlib.util.find_spec('allensdk') is None:
            self.skipTest('Optional Allen loader dependency allensdk is unavailable')
        from modules.setup.fit_json import find_fit_json, coerce_fit_genome_values_to_numeric
        native = self.source / 'allen'
        native.mkdir()
        (native / 'manifest.json').write_text(json.dumps({'biophys': [{'model_file': ['synthetic_fit.json']}]}))
        fit = native / 'synthetic_fit.json'
        fit.write_text(json.dumps({'genome': [{'section': 'soma', 'value': '0.001'}]}))
        cfg = {'cell_loader': 'allen_manifest', 'paths': {'manifest': 'allen/manifest.json', 'modfiles': None},
               'model_source': {'mode': 'reference', 'source_root': str(self.source), 'folders': ['allen']}}
        register_model_sources(self.tune, cfg, dry_run=False)
        self.assertEqual(find_fit_json(self.tune), fit)
        before = fit.read_bytes()
        with self.assertRaisesRegex(ValueError, 'external production fit JSON'):
            coerce_fit_genome_values_to_numeric(self.tune)
        self.assertEqual(fit.read_bytes(), before)
        set_model_source_mode(self.tune, 'copy')
        copied = self.tune / 'model_sources/allen/synthetic_fit.json'
        self.assertEqual(find_fit_json(self.tune), copied)
        coerce_fit_genome_values_to_numeric(self.tune)
        self.assertEqual(json.loads(copied.read_text())['genome'][0]['value'], .001)
        self.assertEqual(fit.read_bytes(), before)

    def test_package_factory_relative_asset_and_state_restoration(self):
        cfg = factory_config(self.source, mode='reference')
        scaffold(self.tune, cfg)
        before, cwd, import_path, bytecode = native_tree(self.source), Path.cwd(), list(sys.path), sys.dont_write_bytecode
        cell = load_cell(cfg, base_dir=self.tune)
        self.assertEqual(cell.soma[0].L, 20.)
        self.assertEqual(cell.dend[0].nseg, 3)
        self.assertEqual(native_tree(self.source), before)
        self.assertEqual(Path.cwd(), cwd)
        self.assertEqual(sys.path, import_path)
        self.assertEqual(sys.dont_write_bytecode, bytecode)
        (self.source / 'package/support.py').write_text('# changed\n')
        with self.assertRaisesRegex(RuntimeError, 'changed.*Restart'):
            load_cell(cfg, base_dir=self.tune)

    def test_standalone_factory_and_contextual_failures(self):
        cfg = {'cell_loader': 'python_factory', 'paths': {'python_factory': str(self.source / 'standalone.py'), 'modfiles': None},
               'python_factory': {'callable': 'build_cell', 'args': [23.]}}
        cell = load_cell(cfg, base_dir=self.tune)
        self.assertEqual(cell.soma[0].L, 23.)
        cfg['python_factory']['callable'] = 'missing'
        with self.assertRaisesRegex(RuntimeError, 'missing.*standalone.py') as caught:
            load_cell(cfg, base_dir=self.tune)
        self.assertIsInstance(caught.exception.__cause__, AttributeError)
        cfg['python_factory']['args'] = [float('nan')]
        with self.assertRaisesRegex(ValueError, 'finite JSON'):
            load_cell(cfg, base_dir=self.tune)

    def test_copy_is_portable_after_upstream_removal(self):
        cfg = factory_config(self.source)
        scaffold(self.tune, cfg)
        shutil.rmtree(self.source)
        cell = load_cell(cfg, base_dir=self.tune)
        self.assertEqual(cell.soma[0].L, 20.)

    def test_invalid_return_import_conflict_and_partial_import(self):
        bad_root = self.root / 'bad_builder'
        bad_root.mkdir()
        cfg = {'cell_loader': 'python_factory', 'paths': {'python_factory': str(bad_root / 'bad.py'), 'modfiles': None},
               'python_factory': {'callable': 'build_cell'}}
        (bad_root / 'bad.py').write_text('def build_cell():\n    return None\n')
        with self.assertRaisesRegex(RuntimeError, 'bad.py.*None is not a cell') as caught:
            load_cell(cfg, base_dir=self.tune)
        self.assertIsInstance(caught.exception.__cause__, ValueError)
        cfg = factory_config(self.source, mode='reference')
        with mock.patch.dict(sys.modules, {'package': SimpleNamespace(__file__='/unrelated/package/__init__.py')}):
            with self.assertRaisesRegex(RuntimeError, 'conflicting models'):
                load_cell(cfg, base_dir=self.tune)
        (self.source / 'package/__init__.py').write_text('raise ImportError("synthetic missing dependency")\n')
        with self.assertRaisesRegex(RuntimeError, 'synthetic missing dependency') as caught:
            load_cell(cfg, base_dir=self.tune)
        self.assertIsInstance(caught.exception.__cause__, ImportError)
        with self.assertRaisesRegex(RuntimeError, 'partially imported.*Restart'):
            load_cell(cfg, base_dir=self.tune)

    def test_artifact_materialization_dry_run_backup_and_reload(self):
        cfg = factory_config(self.source, mode='reference')
        scaffold(self.tune, cfg)
        session = SimulationSession.from_tune(self.tune, options=SimulationOptions(
            iclamp=True, force_save=True, output_stem='reference_run', sim_overrides=SIM))
        session.run()
        saved = session.save()
        destination = self.root / 'materialized'
        destination.mkdir()
        before = native_tree(self.source)
        report = restore_run_state(from_run=saved, to_tune=destination, apply=['model_artifacts'], materialize_external=True)
        self.assertEqual(report.errors, [])
        self.assertEqual(list(destination.iterdir()), [])
        report = restore_run_state(from_run=saved, to_tune=destination, apply=['model_artifacts'], materialize_external=True, dry_run=False)
        self.assertEqual(report.errors, [])
        config = json.loads((destination / 'cell_configs/cell_config.json').read_text())
        self.assertEqual(config['model_source']['mode'], 'copy')
        self.assertEqual(load_cell(config, base_dir=destination).soma[0].L, 20.)
        (destination / 'model_sources/package/length.txt').write_text('99')
        report = restore_run_state(from_run=saved, to_tune=destination, apply=['model_artifacts'], materialize_external=True, dry_run=False)
        self.assertTrue(report.backup_dir.is_dir())
        self.assertEqual(before, native_tree(self.source))
        self.assertTrue((destination / 'MODEL_SOURCE_PROVENANCE.json').is_file())
        prohibited = restore_run_state(from_run=saved, to_tune=self.source, apply=['model_artifacts'],
                                       materialize_external=True, dry_run=False)
        self.assertTrue(prohibited.errors)
        self.assertEqual(before, native_tree(self.source))

    def test_materialization_rejects_tampered_paths_before_writing(self):
        scaffold(self.tune, factory_config(self.source, mode='reference'))
        session = SimulationSession.from_tune(self.tune, options=SimulationOptions(
            iclamp=True, force_save=True, sim_overrides=SIM))
        session.run()
        saved = session.save()
        path = saved.parent / 'model_artifacts/manifest.json'
        manifest = json.loads(path.read_text())
        manifest['artifacts'][-1]['materialize_relative_path'] = '../escape.py'
        path.write_text(json.dumps(manifest))
        destination = self.root / 'unsafe_restore'
        destination.mkdir()
        report = restore_run_state(from_run=saved, to_tune=destination, apply=['model_artifacts'],
                                   materialize_external=True, dry_run=False)
        self.assertTrue(report.errors)
        self.assertEqual(list(destination.iterdir()), [])

    def test_safe_file_symlink_is_materialized_with_logical_layout(self):
        alias = self.source / 'package/aliased_cell.py'
        alias.symlink_to('cell.py')
        cfg = factory_config(self.source, mode='reference')
        cfg['paths']['python_factory'] = 'package/aliased_cell.py'
        scaffold(self.tune, cfg)
        session = SimulationSession.from_tune(self.tune, options=SimulationOptions(
            iclamp=True, force_save=True, sim_overrides=SIM))
        session.run()
        saved = session.save()
        destination = self.root / 'logical_copy'
        destination.mkdir()
        report = restore_run_state(from_run=saved, to_tune=destination, apply=['model_artifacts'],
                                   materialize_external=True, dry_run=False)
        self.assertEqual(report.errors, [])
        copied = destination / 'model_sources/package/aliased_cell.py'
        self.assertFalse(copied.is_symlink())
        self.assertEqual(copied.read_bytes(), alias.read_bytes())
        recipe = json.loads((destination / 'cell_configs/cell_config.json').read_text())
        self.assertEqual(load_cell(recipe, base_dir=destination).soma[0].L, 20.)

    def test_saved_hashes_detect_post_construction_source_edits(self):
        scaffold(self.tune, factory_config(self.source, mode='reference'))
        session = SimulationSession.from_tune(self.tune, options=SimulationOptions(
            iclamp=True, force_save=True, sim_overrides=SIM))
        session.run()
        (self.source / 'package/length.txt').write_text('21.0\n')
        saved = session.save()
        manifest = json.loads((saved.parent / 'model_artifacts/manifest.json').read_text())
        self.assertTrue(manifest['errors'])
        edited = next(item for item in manifest['artifacts'] if item['source_path'].endswith('length.txt'))
        self.assertFalse(edited['matches_construction'])
        self.assertNotEqual(edited['construction_sha256'], edited['sha256'])

    def test_compile_failure_never_writes_upstream(self):
        from modules.setup.mechanisms import compile_modfiles
        scaffold(self.tune, factory_config(self.source, mode='reference', mods=True))
        before = native_tree(self.source)
        with mock.patch('modules.setup.mechanisms.subprocess.check_call', side_effect=subprocess.CalledProcessError(1, 'nrnivmodl')):
            with self.assertRaises(subprocess.CalledProcessError):
                compile_modfiles(self.tune, load_dll=False)
        self.assertEqual(before, native_tree(self.source))
        self.assertFalse((self.source / 'mechanisms/x86_64').exists())

    def test_cli_preview_requires_declared_folders_and_reruns_use_stored_recipe(self):
        command = [sys.executable, 'scripts/step1_prepare.py', '--tune-dir', str(self.tune),
            '--cell', 'Synthetic', '--python-factory-file', 'package/cell.py',
            '--python-factory-callable', 'build_cell', '--model-source-root', str(self.source),
            '--loader-paths-json', '{"modfiles":null}', '--preview-model-sources']
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('explicit --model-source-folders', result.stderr)
        self.assertFalse(self.tune.exists())
        command.extend(['--model-source-folders', 'package'])
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.tune.exists())
        scaffold(self.tune, factory_config(self.source))
        before = native_tree(self.tune / 'model_sources')
        shutil.rmtree(self.source)
        result = subprocess.run([sys.executable, 'scripts/step1_prepare.py', '--tune-dir', str(self.tune),
            '--no-download', '--no-synapse-configs', '--no-compile', '--no-load-dll', '--target-source-mode', 'none'],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(before, native_tree(self.tune / 'model_sources'))

    def test_concurrent_external_compilation_preserves_sibling_include(self):
        from modules.setup.mechanisms import find_nrnivmodl
        if shutil.which('c++') is None or find_nrnivmodl() is None:
            self.skipTest('NEURON compiler unavailable')
        dependencies = self.source / 'dependencies'
        dependencies.mkdir()
        shutil.copy2(self.source / 'mechanisms/leak_parameters.inc', dependencies / 'leak_parameters.inc')
        mod = self.source / 'mechanisms/synthetic_leak.mod'
        mod.write_text(mod.read_text().replace('"leak_parameters.inc"', '"../dependencies/leak_parameters.inc"'))
        cfg = factory_config(self.source, mode='reference', mods=True)
        cfg['model_source']['folders'].append('dependencies')
        scaffold(self.tune, cfg)
        before = native_tree(self.source)
        code = '''from pathlib import Path
import sys
from modules.notebooks.bootstrap import ensure_modfiles
from modules.setup.mechanisms import find_compiled_mechanism_dll
t=Path(sys.argv[1]); ensure_modfiles(t); assert find_compiled_mechanism_dll(t)
'''
        workers = [subprocess.Popen([sys.executable, '-c', code, str(self.tune)], cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True) for _ in range(2)]
        for worker in workers:
            out, err = worker.communicate(timeout=60)
            self.assertEqual(worker.returncode, 0, out + err)
        self.assertEqual(len(list((self.tune / '.scp_build').glob('*/build_manifest.json'))), 1)
        self.assertEqual(before, native_tree(self.source))

    def test_fresh_process_hoc_constructor_relative_asset(self):
        cfg = {'cell_name': 'Synthetic', 'cell_loader': 'hoc_template', 'paths': {'hoc_template': 'hoc/RelativeCell.hoc', 'modfiles': None},
               'hoc_template': {'template_name': 'ScpRelativeAssetCell'},
               'model_source': {'mode': 'reference', 'source_root': str(self.source), 'folders': ['hoc']}}
        scaffold(self.tune, cfg)
        code = 'import json,sys; from pathlib import Path; from modules.model import load_cell; t=Path(sys.argv[1]); c=load_cell(json.loads((t/"cell_configs/cell_config.json").read_text()),base_dir=t); assert c.soma[0].L==20'
        result = subprocess.run([sys.executable, '-c', code, str(self.tune)], cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_fresh_process_isolated_mod_build_and_cache_invalidation(self):
        if shutil.which('c++') is None:
            self.skipTest('C++ compiler unavailable')
        from modules.setup.mechanisms import find_nrnivmodl
        if find_nrnivmodl() is None:
            self.skipTest('nrnivmodl unavailable')
        cfg = factory_config(self.source, mode='reference', mods=True)
        scaffold(self.tune, cfg)
        before = native_tree(self.source)
        code = '''import json,sys
from pathlib import Path
from modules.setup.mechanisms import compile_modfiles, find_compiled_mechanism_dll
from modules.simulation import SimulationSession, SimulationOptions
t=Path(sys.argv[1]); result=compile_modfiles(t,load_dll=False)
assert str(t/'.scp_build') in result['compiled_dir']
assert not compile_modfiles(t,load_dll=False)['compiled_now']
s=SimulationSession.from_tune(t,options=SimulationOptions(iclamp=True,sim_overrides=json.loads(sys.argv[2])))
r=s.run(); assert len(r['traces']['V'])==801
print(find_compiled_mechanism_dll(t))
'''
        args = [sys.executable, '-c', code, str(self.tune), json.dumps(SIM)]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
        result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        library = result.stdout.strip().splitlines()[-1]
        self.assertEqual(before, native_tree(self.source))
        include = self.source / 'mechanisms/leak_parameters.inc'
        include.write_text(include.read_text().replace('0.00001', '0.00002'))
        changed = native_tree(self.source)
        result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotEqual(library, result.stdout.strip().splitlines()[-1])
        self.assertEqual(changed, native_tree(self.source))


class CustomModelDocumentationTests(unittest.TestCase):
    def test_contract_snippet_is_executable(self):
        guide = (ROOT / 'docs/guides/custom_models.md').read_text()
        snippet = re.findall(r'```python\n(.*?)\n```', guide, flags=re.S)[0]
        namespace = {}
        exec(compile(snippet, 'custom_models.md:contract', 'exec'), namespace)
        self.assertIs(validate_cell(namespace['cell']).model, namespace['soma'])

    def test_primary_notebooks_keep_optional_runtime_hooks(self):
        for step, filename in enumerate(('0_pipeline.ipynb', '1_setup.ipynb', '2_passive.ipynb',
                '3_active.ipynb', '4_synapses.ipynb', '5_simulate.ipynb', '6_analysis.ipynb')):
            notebook = json.loads((ROOT / filename).read_text())
            code = '\n'.join(''.join(cell.get('source', [])) for cell in notebook['cells'] if cell['cell_type'] == 'code')
            self.assertIn('CELL_OVERRIDE = None', code, filename)
            if step < 6:
                self.assertIn('cell_override=CELL_OVERRIDE', code, filename)
                self.assertIn('runtime_sim_config=RUNTIME_SIM_CONFIG', code, filename)
            else:
                extra = (ROOT / 'modules/analysis/ui/extra.py').read_text()
                self.assertIn("cell_override=g.get('CELL_OVERRIDE')", extra)
            self.assertIn('custom_models.md', (ROOT / filename).read_text())
        self.assertIn('materialize_external=materialize_external', (ROOT / '7_tools.ipynb').read_text())

    def test_synthetic_walkthrough_in_fresh_process(self):
        code = '''import json
from pathlib import Path
nb=json.loads(Path('extra_notebooks/custom_cells.ipynb').read_text())
scope={}
try:
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            exec(compile(''.join(cell['source']), 'custom_cells.ipynb', 'exec'), scope)
finally:
    if 'workspace' in scope: scope['workspace'].cleanup()
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg'),
            capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cli_rejects_runtime_only_recipe_actionably(self):
        with tempfile.TemporaryDirectory() as tmp:
            section = h.Section(name='runtime_only_cli')
            prepare_runtime_workspace(tmp, wrap_cell(section), SIM)
            result = subprocess.run([sys.executable, 'run_pipeline.py', '--tune-dir', tmp, '--iclamp'],
                cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('CLI/SLURM', result.stdout + result.stderr)
            self.assertIn('runtime-only', result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
