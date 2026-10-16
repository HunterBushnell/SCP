"""Portable project identities in saved results, pooling and restoration."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from modules.simulation.model_artifacts import archive_model_artifacts
from modules.simulation.result_appending import _append_results_to_path, _ensure_multi_results, append_multi_results
from modules.simulation.result_loading import load_results
from modules.simulation.result_paths import _resolve_tune_path, _result_filename
from modules.simulation.result_saving import save_results
from modules.workspaces import WorkspaceMismatchError, result_workspace_identity
from scripts import merge_array_results, merge_two_runs, restore_run_state


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def _result(project=None, *, tune_dir=None, stem='run', cell='PV', tune='tuned'):
    sim = {'project_name': project, 'cell': cell, 'tune': tune,
           'output': stem, 'save_output': True, 'save_model_artifacts': False,
           'save_full_results': True, 'output_format': 'pickle',
           'tstart': 0.0, 'tstop': 10.0, 'dt': 0.1, 'bins': 1.0,
           'n_trials': 1, 'n_traces_to_save': 0, 'n_inputs_to_save': 0}
    if tune_dir is not None:
        sim['tune_dir'] = str(tune_dir)
    return {'mode': 'multi', 'sim_cfg': sim, 'spikes': [np.array([2.0])],
            'traces': {}, 'meta': {'cell_config': {}, 'geometry_config': {}, 'syn_config': {}}}


class ProjectResultTests(unittest.TestCase):
    def test_legacy_filename_and_project_prefix(self):
        self.assertEqual(_result_filename(_result()['sim_cfg'], 'run', '.pkl'), 'PV_tuned_run.pkl')
        self.assertEqual(_result_filename(_result('A1')['sim_cfg'], 'run', '.npz'), 'A1_PV_tuned_run.npz')

    def test_project_result_roundtrips_full_and_sidecar_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            for fmt, full in [('pickle', True), ('npz', True), ('pickle', False)]:
                with self.subTest(fmt=fmt, full=full):
                    result = _result('A1', stem=f'{fmt}_{full}')
                    result['sim_cfg'].update(output_format=fmt, save_full_results=full)
                    saved = Path(save_results(result, base_dir=tmp))
                    expected = {'project_name': 'A1', 'cell_name': 'PV', 'tune_name': 'tuned'}
                    for path in (saved, saved.parent / 'run_manifest.json'):
                        loaded = load_results(path)
                        self.assertEqual(result_workspace_identity(loaded), expected)
                        self.assertEqual(loaded['meta']['workspace_identity'], expected)
                    manifest = json.loads((saved.parent / 'run_manifest.json').read_text())
                    self.assertEqual(manifest['format_version'], 1)
                    self.assertEqual(manifest['workspace_identity'], expected)

    def test_project_path_only_old_result_is_namespaced(self):
        result = _result(tune_dir='/old/server/SCP/projects/A1/cells/PV/tunes/tuned')
        result['sim_cfg'].pop('project_name')
        self.assertEqual(result_workspace_identity(result)['project_name'], 'A1')
        self.assertEqual(_result_filename(result['sim_cfg'], 'old', '.pkl'), 'A1_PV_tuned_old.pkl')

    def test_same_workspace_append_retains_source_lineage(self):
        base, new = _result('A1', stem='first'), _result('A1', stem='second')
        merged = append_multi_results(base, new)
        self.assertEqual(len(merged['spikes']), 2)
        self.assertEqual([item['output'] for item in merged['meta']['append_sources']], ['first', 'second'])
        self.assertEqual(len(base['spikes']), 1)

    def test_cross_workspace_append_is_rejected(self):
        for foreign in (_result('Amygdala'), _result(), _result('A1', cell='SST'), _result('A1', tune='orig')):
            with self.subTest(identity=result_workspace_identity(foreign)):
                with self.assertRaises(WorkspaceMismatchError):
                    append_multi_results(_result('A1'), foreign)

    def test_incomplete_global_legacy_append_warns(self):
        base, new = _result(), _result()
        base['sim_cfg'].pop('cell')
        merged = append_multi_results(base, new)
        self.assertTrue(any('Incomplete' in text for text in merged['meta']['append_warnings']))

    def test_persisted_incomplete_legacy_append_upgrades_only_known_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _result()
            base['sim_cfg'].pop('cell')
            base['sim_cfg'].pop('tune')
            target = Path(save_results(base, base_dir=root / 'runs'))
            current = root / 'cells/PV/tunes/tuned'
            incoming = _result(tune_dir=current)
            _append_results_to_path(incoming, target)
            merged = load_results(target)
            self.assertEqual(result_workspace_identity(merged),
                             {'project_name': None, 'cell_name': 'PV', 'tune_name': 'tuned'})
            self.assertEqual(merged['sim_cfg']['tune_dir'], str(current))
            self.assertTrue(merged['meta']['append_warnings'])
            self.assertEqual(merged['meta']['append_sources'][0]['workspace_identity']['cell_name'], None)
            before = target.read_bytes()
            with self.assertRaises(WorkspaceMismatchError):
                _append_results_to_path(_result('A1'), target)
            self.assertEqual(target.read_bytes(), before)

    def test_incomplete_top_level_identity_is_promoted_consistently(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = _result()
            base['sim_cfg'].pop('cell')
            base['sim_cfg'].pop('tune')
            base['workspace_identity'] = {'project_name': None, 'cell_name': None, 'tune_name': None}
            target = Path(save_results(base, base_dir=tmp))
            _append_results_to_path(_result(), target)
            merged = load_results(target)
            expected = {'project_name': None, 'cell_name': 'PV', 'tune_name': 'tuned'}
            self.assertEqual(merged['workspace_identity'], expected)
            self.assertEqual(result_workspace_identity(merged), expected)

    def test_single_to_multi_coercions_preserve_top_level_identity(self):
        single = _result()
        single['mode'] = 'single'
        single['spikes'] = np.array([2.0])
        single['sim_cfg'] = {}
        single['meta'] = {}
        single['workspace_identity'] = {'project_name': 'A1', 'cell_name': 'PV', 'tune_name': 'tuned'}
        for coerce in (_ensure_multi_results, merge_array_results._ensure_multi, merge_two_runs._ensure_multi):
            with self.subTest(coerce=coerce.__module__):
                coerced = coerce(single)
                self.assertEqual(result_workspace_identity(coerced), single['workspace_identity'])
                self.assertIsNot(coerced['workspace_identity'], single['workspace_identity'])

    def test_lowlevel_single_append_rejects_top_only_foreign_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'base.pkl'
            base, foreign = _result('A1'), _result('Amygdala')
            for result in (base, foreign):
                result['workspace_identity'] = result_workspace_identity(result)
                result['sim_cfg'] = {}
                result['meta'] = {}
                result['mode'] = 'single'
                result['spikes'] = np.array([2.0])
            with target.open('wb') as handle:
                pickle.dump(base, handle)
            before = target.read_bytes()
            with self.assertRaises(WorkspaceMismatchError):
                _append_results_to_path(foreign, target)
            self.assertEqual(target.read_bytes(), before)

    def test_append_save_rejects_foreign_target_before_any_new_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(save_results(_result('A1'), base_dir=tmp))
            before = target.read_bytes()
            foreign = _result('Amygdala', stem='must_not_be_written')
            foreign['sim_cfg']['append_to'] = str(target)
            with self.assertRaises(WorkspaceMismatchError):
                save_results(foreign, base_dir=tmp)
            self.assertFalse((Path(tmp) / 'must_not_be_written').exists())
            self.assertEqual(target.read_bytes(), before)
            with self.assertRaises(WorkspaceMismatchError):
                _append_results_to_path(foreign, target)
            self.assertEqual(target.read_bytes(), before)

    def test_relocated_append_prefers_incoming_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / 'old/projects/A1/cells/PV/tunes/tuned'
            current = Path(tmp) / 'current/projects/A1/cells/PV/tunes/tuned'
            for root, source in [(old, '# old builder\n'), (current, '# current builder\n')]:
                (root / 'model').mkdir(parents=True)
                (root / 'model/builder.py').write_text(source, encoding='utf-8')
            cfg = {'cell_loader': 'python_factory', 'cell_name': 'PV', 'project_name': 'A1',
                   'paths': {'python_factory': 'model/builder.py', 'modfiles': None},
                   'python_factory': {'callable': 'build_cell'}}
            base, new = _result('A1', tune_dir=old), _result('A1', tune_dir=current)
            base['meta'].update(cell_config=cfg, model_provenance={'tag': 'old'})
            new['meta'].update(cell_config=cfg, model_provenance={'tag': 'current'})
            merged = append_multi_results(base, new)
            self.assertEqual(merged['sim_cfg']['tune_dir'], str(current))
            self.assertEqual(merged['meta']['model_provenance']['tag'], 'current')
            self.assertEqual(merged['meta']['append_sources'][0]['model_provenance']['tag'], 'old')
            merged['sim_cfg']['save_model_artifacts'] = True
            info = archive_model_artifacts(merged['sim_cfg'], Path(tmp) / 'archive', cell_config=cfg)
            archive = Path(tmp) / 'archive' / info['filename']
            manifest = json.loads(archive.read_text())
            self.assertEqual(manifest['workspace_identity']['project_name'], 'A1')
            self.assertEqual((archive.parent / manifest['artifacts'][0]['archive_path']).read_text(), '# current builder\n')

    def test_paired_full_file_append_refreshes_current_source_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / 'old/projects/A1/cells/PV/tunes/tuned'
            current = root / 'current/projects/A1/cells/PV/tunes/tuned'
            cfg = {'cell_loader': 'python_factory', 'cell_name': 'PV', 'project_name': 'A1',
                   'paths': {'python_factory': 'model/builder.py', 'modfiles': None},
                   'python_factory': {'callable': 'build_cell'}}
            for tune, value in [(old, b'# old\n'), (current, b'# current\n')]:
                (tune / 'model').mkdir(parents=True)
                (tune / 'model/builder.py').write_bytes(value)
            base, incoming = _result('A1', tune_dir=old), _result('A1', tune_dir=current)
            for result in (base, incoming):
                result['meta']['cell_config'] = cfg
                result['sim_cfg']['save_model_artifacts'] = True
            saved = Path(save_results(base, base_dir=root / 'runs'))
            _append_results_to_path(incoming, saved)
            refreshed = load_results(saved.parent)
            self.assertEqual(refreshed['sim_cfg']['tune_dir'], str(current))
            self.assertEqual(len(refreshed['spikes']), 2)
            artifact_root = saved.parent / 'model_artifacts'
            manifest = json.loads((artifact_root / 'manifest.json').read_text())
            self.assertEqual((artifact_root / manifest['artifacts'][0]['archive_path']).read_bytes(), b'# current\n')

    def test_array_merge_rejects_foreign_namespace_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            parts, output = Path(tmp) / 'parts', Path(tmp) / 'merged'
            save_results(_result('A1', stem='slurm_123_0'), base_dir=parts)
            save_results(_result('Amygdala', stem='slurm_123_1'), base_dir=parts)
            argv = ['merge_array_results.py', '--input-dir', str(parts), '--output-dir', str(output), '--job-id', '123']
            with mock.patch.object(sys, 'argv', argv), self.assertRaises(WorkspaceMismatchError):
                merge_array_results.main()
            self.assertFalse(output.exists())

    def test_array_merge_retains_identity_and_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            parts, output = Path(tmp) / 'parts', Path(tmp) / 'merged'
            for index in range(2):
                save_results(_result('A1', stem=f'slurm_123_{index}'), base_dir=parts)
            argv = ['merge_array_results.py', '--input-dir', str(parts), '--output-dir', str(output), '--job-id', '123']
            with mock.patch.object(sys, 'argv', argv):
                merge_array_results.main()
            merged = load_results(output / 'slurm_merged_123')
            self.assertEqual(len(merged['spikes']), 2)
            self.assertEqual(result_workspace_identity(merged)['project_name'], 'A1')
            self.assertEqual(len(merged['meta']['merge_sources']), 2)

    def test_array_merge_incomplete_base_cannot_hide_known_identity_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            parts, output = Path(tmp) / 'parts', Path(tmp) / 'merged'
            for index, cell in enumerate((None, 'PV', 'SST')):
                result = _result(stem=f'slurm_123_{index}')
                if cell is None:
                    result['sim_cfg'].pop('cell')
                    result['sim_cfg'].pop('tune')
                else:
                    result['sim_cfg']['cell'] = cell
                save_results(result, base_dir=parts)
            argv = ['merge_array_results.py', '--input-dir', str(parts), '--output-dir', str(output), '--job-id', '123']
            with mock.patch.object(sys, 'argv', argv), self.assertRaises(WorkspaceMismatchError):
                merge_array_results.main()
            self.assertFalse(output.exists())

    def test_array_merge_promotes_compatible_incomplete_legacy_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            parts, output = Path(tmp) / 'parts', Path(tmp) / 'merged'
            base = _result(stem='slurm_123_0')
            base['sim_cfg'].pop('cell')
            base['sim_cfg'].pop('tune')
            save_results(base, base_dir=parts)
            save_results(_result(stem='slurm_123_1'), base_dir=parts)
            argv = ['merge_array_results.py', '--input-dir', str(parts), '--output-dir', str(output), '--job-id', '123']
            with mock.patch.object(sys, 'argv', argv):
                merge_array_results.main()
            merged = load_results(output / 'slurm_merged_123')
            self.assertEqual(result_workspace_identity(merged),
                             {'project_name': None, 'cell_name': 'PV', 'tune_name': 'tuned'})
            self.assertTrue(any('Incomplete' in warning for warning in merged['meta']['merge_warnings']))
            self.assertEqual(merged['meta']['merge_sources'][0]['workspace_identity']['cell_name'], None)

    def test_array_merge_rejects_top_only_foreign_single_identities(self):
        with tempfile.TemporaryDirectory() as tmp:
            parts, output = Path(tmp) / 'parts', Path(tmp) / 'merged'
            parts.mkdir()
            for index, project in enumerate(('A1', 'Amygdala')):
                result = _result(project)
                result['workspace_identity'] = result_workspace_identity(result)
                result['sim_cfg'] = {}
                result['meta'] = {}
                result['mode'] = 'single'
                result['spikes'] = np.array([2.0])
                with (parts / f'part_{index}.pkl').open('wb') as handle:
                    pickle.dump(result, handle)
            argv = ['merge_array_results.py', '--input-dir', str(parts), '--output-dir', str(output), '--pattern', '*.pkl']
            with mock.patch.object(sys, 'argv', argv), self.assertRaises(WorkspaceMismatchError):
                merge_array_results.main()
            self.assertFalse(output.exists())

    def test_two_run_merge_cannot_bypass_namespace_with_config_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(save_results(_result('A1', stem='a'), base_dir=tmp))
            b = Path(save_results(_result('Amygdala', stem='b'), base_dir=tmp))
            output = Path(tmp) / 'forbidden'
            report = merge_two_runs.merge_two_runs(run_a=a, run_b=b, output_dir=output,
                                                    dry_run=False, strict_configs=False)
            self.assertTrue(any('different workspaces' in error for error in report.errors))
            self.assertFalse(output.exists())

    def test_deliberate_cross_project_comparison_is_not_pooling(self):
        from modules.analysis.analysis import build_compare_table

        rows = build_compare_table(_result('A1'), _result('Amygdala'))
        self.assertTrue(any(row['path'] == 'results.sim_cfg.project_name' for row in rows))

    def test_cross_project_restore_preserves_destination_routing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'projects/A1/cells/PV/tunes/tuned'
            destination = root / 'projects/Amygdala/cells/VIP/tunes/orig'
            target_cfg = {'cell_name': 'VIP', 'tune': 'orig', 'project_name': 'Amygdala',
                          'paths': {'tune_dir': str(destination)}, 'color': 'k'}
            target_sim = {'cell': 'VIP', 'tune': 'orig', 'project_name': 'Amygdala',
                          'tune_dir': str(destination), 'conditions': {'v_init_mV': -60}}
            _write_json(destination / 'cell_configs/cell_config.json', target_cfg)
            _write_json(destination / 'cell_configs/sim_config.json', target_sim)
            result = _result('A1', tune_dir=source)
            result['sim_cfg']['conditions'] = {'v_init_mV': -70}
            result['meta']['cell_config'] = {'cell_name': 'PV', 'tune': 'tuned', 'project_name': 'A1',
                                           'paths': {'tune_dir': str(source)}, 'color': 'r'}
            run = Path(save_results(result, base_dir=root / 'runs'))
            preview = restore_run_state.restore_run_state(from_run=run, to_tune=destination,
                                                         apply=['sim_config', 'cell_config'])
            self.assertEqual(preview.source_identity['project_name'], 'A1')
            self.assertEqual(preview.destination_identity['project_name'], 'Amygdala')
            self.assertEqual(json.loads((destination / 'cell_configs/cell_config.json').read_text()), target_cfg)
            report = restore_run_state.restore_run_state(from_run=run, to_tune=destination,
                                                        apply=['sim_config', 'cell_config'], dry_run=False)
            self.assertFalse(report.errors)
            self.assertIsNotNone(report.backup_dir)
            restored_cfg = json.loads((destination / 'cell_configs/cell_config.json').read_text())
            restored_sim = json.loads((destination / 'cell_configs/sim_config.json').read_text())
            self.assertEqual(restored_cfg, dict(target_cfg, color='r'))
            self.assertEqual(restored_sim['project_name'], 'Amygdala')
            self.assertEqual(restored_sim['tune_dir'], str(destination))
            self.assertEqual(restored_sim['conditions']['v_init_mV'], -70)

    def test_restore_symlink_guard_does_not_depend_on_backups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / 'projects/A1/cells/PV/tunes/tuned'
            configs = destination / 'cell_configs'
            configs.mkdir(parents=True)
            upstream = root / 'upstream_config.json'
            upstream_data = {'project_name': 'A1', 'cell_name': 'PV', 'tune': 'tuned', 'color': 'k'}
            _write_json(upstream, upstream_data)
            (configs / 'cell_config.json').symlink_to(upstream)
            _write_json(configs / 'sim_config.json', {'tstop': 10.0})
            incoming = _result('A1')
            incoming['sim_cfg']['tstop'] = 20.0
            incoming['meta']['cell_config'] = dict(upstream_data, color='r')
            run = Path(save_results(incoming, base_dir=root / 'runs'))
            report = restore_run_state.restore_run_state(from_run=run, to_tune=destination,
                                                        apply=['sim_config', 'cell_config'], dry_run=False, backup=False)
            self.assertTrue(any('outside target tune' in error for error in report.errors))
            self.assertEqual(json.loads(upstream.read_text()), upstream_data)
            self.assertEqual(json.loads((configs / 'sim_config.json').read_text()), {'tstop': 10.0})

    def test_restore_rejects_escaping_workspace_symlinks_before_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upstream = root / 'upstream'
            upstream_data = {'cell_name': 'PV', 'tune': 'tuned', 'color': 'k'}
            _write_json(upstream / 'cell_configs/cell_config.json', upstream_data)
            before = (upstream / 'cell_configs/cell_config.json').read_bytes()
            destination = root / 'projects/A1/cells/PV/tunes/tuned'
            destination.parent.mkdir(parents=True)
            destination.symlink_to(upstream, target_is_directory=True)
            incoming = _result()
            incoming['meta']['cell_config'] = dict(upstream_data, color='r')
            run = Path(save_results(incoming, base_dir=root / 'runs'))
            for dry_run in (True, False):
                for backup in (True, False):
                    with self.subTest(dry_run=dry_run, backup=backup):
                        with self.assertRaisesRegex(ValueError, 'escapes its namespace'):
                            restore_run_state.restore_run_state(
                                from_run=run, to_tune=destination, apply=['cell_config'],
                                dry_run=dry_run, backup=backup)
                        self.assertEqual((upstream / 'cell_configs/cell_config.json').read_bytes(), before)
                        self.assertFalse((upstream / 'restore_backups').exists())

    def test_restore_rejects_inferred_escaping_workspace_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upstream = root / 'upstream'
            upstream_data = {'cell_name': 'PV', 'tune': 'tuned', 'color': 'k'}
            _write_json(upstream / 'cell_configs/cell_config.json', upstream_data)
            destination = root / 'projects/A1/cells/PV/tunes/tuned'
            destination.parent.mkdir(parents=True)
            destination.symlink_to(upstream, target_is_directory=True)
            incoming = _result('A1', tune_dir=destination)
            incoming['meta']['cell_config'] = dict(upstream_data, color='r')
            run = Path(save_results(incoming, base_dir=root / 'runs'))
            with self.assertRaisesRegex(ValueError, 'escapes its namespace'):
                restore_run_state.restore_run_state(from_run=run, apply=['cell_config'], dry_run=False)
            self.assertEqual(json.loads((upstream / 'cell_configs/cell_config.json').read_text()), upstream_data)

    def test_direct_archive_rejects_conflicting_identity_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tune = root / 'projects/A1/cells/PV/tunes/tuned'
            tune.mkdir(parents=True)
            run = root / 'must_not_exist'
            sim = _result('Amygdala', tune_dir=tune)['sim_cfg']
            sim['save_model_artifacts'] = True
            with self.assertRaises(WorkspaceMismatchError):
                archive_model_artifacts(sim, run, cell_config={'cell_loader': 'runtime_cell'})
            self.assertFalse(run.exists())

    def test_direct_archive_preflights_hashes_and_destination_symlinks(self):
        from modules.simulation import model_artifacts

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tune = root / 'projects/A1/cells/PV/tunes/tuned'
            (tune / 'model').mkdir(parents=True)
            (tune / 'model/builder.py').write_bytes(b'# synthetic builder\n')
            (tune / 'mechanisms').mkdir()
            (tune / 'mechanisms/parameters.inc').write_bytes(b'# synthetic dependency\n')
            cfg = {'cell_loader': 'python_factory', 'paths': {'python_factory': 'model/builder.py', 'modfiles': 'mechanisms'},
                   'python_factory': {'callable': 'build_cell'}}
            sim = _result('A1', tune_dir=tune)['sim_cfg']
            sim['save_model_artifacts'] = True
            run = root / 'must_not_exist'
            original_hash = model_artifacts.sha256_file
            def fail_second_source(path):
                if path.name == 'parameters.inc':
                    raise OSError('unreadable declared source')
                return original_hash(path)
            with mock.patch.object(model_artifacts, 'sha256_file', side_effect=fail_second_source):
                with self.assertRaisesRegex(OSError, 'unreadable'):
                    archive_model_artifacts(sim, run, cell_config=cfg)
            self.assertFalse(run.exists())
            upstream = root / 'upstream'
            upstream.mkdir()
            (upstream / 'sentinel').write_bytes(b'untouched\n')
            run.mkdir()
            (run / 'model_artifacts').symlink_to(upstream, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'destination escapes'):
                archive_model_artifacts(sim, run, cell_config=cfg)
            self.assertEqual(list(upstream.iterdir()), [upstream / 'sentinel'])
            self.assertEqual((upstream / 'sentinel').read_bytes(), b'untouched\n')

    def test_project_materialization_and_legacy_archive_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tune = root / 'projects/A1/cells/PV/tunes/tuned'
            upstream = root / 'upstream'
            upstream.mkdir()
            source = tune / 'model_sources/package/builder.py'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'def build_cell():\r\n    return None\r\n')
            cfg = {'cell_loader': 'python_factory', 'cell_name': 'PV', 'tune': 'tuned', 'project_name': 'A1',
                   'paths': {'python_factory': 'package/builder.py', 'modfiles': None},
                   'python_factory': {'callable': 'build_cell'},
                   'model_source': {'mode': 'copy', 'source_root': str(upstream), 'folders': ['package']}}
            _write_json(tune / 'cell_configs/cell_config.json', cfg)
            result = _result('A1', tune_dir=tune)
            result['sim_cfg']['save_model_artifacts'] = True
            result['meta']['cell_config'] = cfg
            run = Path(save_results(result, base_dir=root / 'runs'))
            archived = json.loads((run.parent / 'model_artifacts/manifest.json').read_text())
            self.assertEqual(archived['format_version'], 1)
            self.assertEqual(archived['artifacts'][0]['target_relative_path'], 'model_sources/package/builder.py')
            destination = root / 'projects/Amygdala/cells/VIP/tunes/orig'
            destination.mkdir(parents=True)
            report = restore_run_state.restore_run_state(from_run=run, to_tune=destination,
                                                        apply=['model_artifacts'], materialize_external=True, dry_run=False)
            self.assertFalse(report.errors)
            recipe = json.loads((destination / 'cell_configs/cell_config.json').read_text())
            self.assertEqual(recipe['workspace_identity'], {'project_name': 'Amygdala', 'cell_name': 'VIP', 'tune_name': 'orig'})
            self.assertEqual(recipe['cell_name'], 'VIP')
            self.assertEqual(recipe['model_source']['mode'], 'copy')
            self.assertEqual((destination / 'model_sources/package/builder.py').read_bytes(), source.read_bytes())
            provenance = json.loads((destination / 'MODEL_SOURCE_PROVENANCE.json').read_text())
            self.assertEqual(provenance['source_workspace_identity']['project_name'], 'A1')
            self.assertEqual(provenance['destination_workspace_identity']['project_name'], 'Amygdala')
            self.assertEqual(list(upstream.iterdir()), [])

    def test_path_resolution_accepts_namespace_shorthand(self):
        path = _resolve_tune_path({'cell': 'PV', 'tune': 'tuned', 'project_name': 'A1'})
        self.assertEqual(path.parts[-6:], ('projects', 'A1', 'cells', 'PV', 'tunes', 'tuned'))


if __name__ == '__main__':
    unittest.main()
