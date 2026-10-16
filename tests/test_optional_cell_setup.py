"""Create-only configuration and snapshot-based working copies use disposable files."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scp_pipeline.setup import (preview_setup_outputs, create_setup_outputs,
    preview_workspace_registration, register_workspace, create_working_copy)
from scp_pipeline.setup.optional_configs import target_payload
from scp_pipeline.setup.scaffold import scaffold_synapse_configs, scaffold_base_configs
from scp_pipeline.setup.target_config import prepare_target_config


class OptionalCellSetupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'native'
        self.source.mkdir()
        self.entry = self.source / 'Cell.hoc'
        self.entry.write_bytes(b'begintemplate OptionalCell\r\npublic soma\r\ncreate soma\r\nendtemplate OptionalCell\r\n')
        self.tune = self.root / 'project/cells/Cell/tunes/orig'
        self.kwargs = dict(project_root=self.root / 'project', tune_dir=self.tune,
            cell_name='Cell', tune_name='orig', conditions={'v_init_mV': -65, 'celsius_C': 30},
            cell_config={'cell_loader': 'hoc_template', 'paths': {'hoc_template': 'Cell.hoc', 'modfiles': None},
                'hoc_template': {'template_name': 'OptionalCell', 'constructor_args': []},
                'model_source': {'mode': 'copy', 'source_root': str(self.source), 'files': ['Cell.hoc']}})

    def register(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return register_workspace(dry_run=False, **dict(self.kwargs, **kwargs))

    def files(self):
        return {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}

    def test_disabled_and_skip_make_no_optional_files_or_directories(self):
        before = self.files()
        self.assertEqual(create_setup_outputs(self.tune)['status'], 'unchanged')
        scaffold_synapse_configs(tune_dir=self.tune, config_mode='skip')
        scaffold_base_configs(tune_dir=self.tune, cell_name='Cell', tune_name='orig', specimen_id=None,
                              model_type='perisomatic', config_mode='skip')
        prepare_target_config(tune_dir=self.tune, config_mode='skip')
        self.assertEqual(self.files(), before)
        self.assertFalse(self.tune.exists())

    def test_registration_optional_preflight_and_create_only(self):
        options = {'target_source_mode': 'manual', 'manual_passive': {'v_rest_mV': -68, 'rin_MOhm': 50},
                   'manual_fi_curve': {'currents_pA': [0, 100], 'rates_Hz': [0, 15]}}
        before = self.files()
        preview = preview_workspace_registration(**self.kwargs, target_options=options,
            synapse_options={'template_kinds': ['input_blocks'], 'weight_style': 'fixed'})
        self.assertEqual(self.files(), before)
        self.assertIn('cell_configs/target_config.json', preview['creates'])
        self.register(target_options=options, synapse_options={'weight_style': 'fixed'})
        target = self.tune / 'cell_configs/target_config.json'
        self.assertEqual(json.loads(target.read_text())['manual']['passive']['v_rest_mV'], -68)
        group = json.loads((self.tune / 'cell_configs/syn_groups/input_blocks_template.json').read_text())
        self.assertTrue(all(not value['state'] for value in group.values()))
        before = self.files()
        create_setup_outputs(self.tune, target_options={'target_source_mode': 'none'}, synapse_options={})
        self.assertEqual(self.files(), before)

    def test_all_operations_preflight_before_registration_writes(self):
        for options in ({'manual_passive': {'rin_MOhm': -1}}, {'manual_fi_curve': {'currents_pA': [1]}},
                        {'passive_trace': {'file': 'missing.csv'}}, {'manual_passive': {'tau_ms': float('nan')}}):
            with self.subTest(options=options):
                before = self.files()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.register(target_options=options)
                self.assertEqual(self.files(), before)
                self.assertFalse(self.tune.exists())

    def test_parent_file_collisions_fail_before_any_source_staging(self):
        self.tune.mkdir(parents=True)
        (self.tune / 'cell_configs').write_text('not a directory')
        before = self.files()
        with self.assertRaisesRegex(ValueError, 'not a directory'):
            self.register(target_options={'target_source_mode': 'manual'})
        self.assertEqual(self.files(), before)
        self.assertFalse((self.tune / 'model_sources').exists())

    def test_invalid_navigation_paths_fail_before_registration_writes(self):
        before = self.files()
        with self.assertRaises((TypeError, ValueError)):
            self.register(workspace_root={'invalid': 'not a path'})
        self.assertEqual(self.files(), before)

    def test_navigation_source_collisions_are_rejected_without_upstream_changes(self):
        root = self.kwargs['project_root']
        root.mkdir()
        navigation = root / '.scp_project.json'
        navigation.write_text('{"format_version":1,"project_name":"Example","workspace_root":"."}')
        config = deepcopy(self.kwargs['cell_config'])
        config['model_source'].update(source_root=str(root), files=['.scp_project.json', 'Cell.hoc'])
        (root / 'Cell.hoc').write_bytes(self.entry.read_bytes())
        before = self.files()
        with self.assertRaisesRegex(ValueError, 'Source/destination overlap'):
            self.register(cell_config=config)
        self.assertEqual(self.files(), before)

    def test_invalid_extraction_fields_rejected_before_writing(self):
        for options in (
            {'target_source_mode': 'traces', 'passive_trace': {'dt_ms': [0.025]}},
            {'target_source_mode': 'traces', 'passive_trace': {'reducer': 'unsupported'}},
            {'target_source_mode': 'allen_nwb', 'allen_nwb': {'active': {'min_current_pA': 10, 'max_current_pA': 1}}},
            {'target_source_mode': 'allen_nwb', 'allen_nwb': {'passive': {'stimulus_names': [12]}}},
        ):
            with self.subTest(options=options):
                before = self.files()
                with self.assertRaisesRegex(ValueError, 'finite numbers|reducer|minimum current|stimulus names'):
                    create_setup_outputs(self.tune, target_options=options)
                self.assertEqual(self.files(), before)

    def test_existing_target_mode_validated_without_rewriting_legacy_configs(self):
        self.register()
        path = self.tune / 'cell_configs/target_config.json'
        config = json.loads((path.parent / 'cell_config.json').read_text())
        path.write_text('{"target_source":"legacy description","manual":{}}\n')
        before = self.files()
        create_setup_outputs(self.tune, cell_config=config, include_required=True)
        self.assertEqual(self.files(), before)
        path.write_text('{"target_source":{"mode":"invalid"}}\n')
        before = self.files()
        with self.assertRaisesRegex(ValueError, 'target_source_mode'):
            create_setup_outputs(self.tune, cell_config=config, include_required=True, synapse_options={})
        self.assertEqual(self.files(), before)

    def test_modes_blank_templates_paths_and_conflicting_inputs(self):
        self.tune.mkdir(parents=True)
        data = self.tune / 'data.csv'
        data.write_text('time_ms,voltage_mV\n0,-65\n')
        for mode in ('manual', 'traces', 'allen_nwb', 'none'):
            payload, inputs, warnings = target_payload({'target_source_mode': mode}, self.tune)
            self.assertEqual(set(payload) & {'manual', 'traces', 'allen_nwb'}, {mode} if mode != 'none' else set())
            self.assertEqual(bool(warnings), mode != 'none')
        payload, inputs, _ = target_payload({'target_source_mode': 'traces',
            'passive_trace': {'file': 'data.csv', 'stim_start_ms': 10, 'stim_stop_ms': 20},
            'include_manual_with_file_source': True, 'manual_passive': {'v_rest_mV': -65}}, self.tune)
        self.assertIn('manual', payload)
        self.assertEqual(inputs[0]['path'], str(data))
        bad_options = [
            {'passive_trace': {'file': 'data.csv'}, 'allen_nwb': {'file': 'data.csv'}},
            {'manual_fi_curve': {'currents_pA': [1], 'rates_Hz': [2], 'csv': 'data.csv'}},
            {'manual_fi_curve': {'currents_pA': [1], 'rates_Hz': [-2]}},
            {'target_source_mode': 'traces', 'passive_trace': {'stim_start_ms': 20, 'stim_stop_ms': 10}},
            {'target_source_mode': 'allen_nwb', 'allen_nwb': {'sweep_ids': [1.2]}},
            {'manual_passive': {'v_rest_mV': True}},
        ]
        for options in bad_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                target_payload(options, self.tune)

    def test_required_missing_configs_only_and_explicit_conditions(self):
        self.register()
        sim = self.tune / 'cell_configs/sim_config.json'
        geom = self.tune / 'cell_configs/geometry.json'
        config = json.loads((self.tune / 'cell_configs/cell_config.json').read_text())
        sim.unlink()
        geom.unlink()
        with self.assertRaises((KeyError, ValueError)):
            preview_setup_outputs(self.tune, cell_config=config, include_required=True)
        create_setup_outputs(self.tune, cell_config=config, include_required=True, conditions=self.kwargs['conditions'])
        self.assertEqual(json.loads(sim.read_text())['conditions'], self.kwargs['conditions'])
        before = self.files()
        create_setup_outputs(self.tune, cell_config=config, include_required=True, conditions={'v_init_mV': -99})
        self.assertEqual(self.files(), before)

    def test_registration_preserves_preexisting_scientific_configs_byte_for_byte(self):
        config_dir = self.tune / 'cell_configs'
        config_dir.mkdir(parents=True)
        sim = config_dir / 'sim_config.json'
        sim.write_text('{"conditions":{"v_init_mV":-77,"celsius_C":25},"tstop":111}\n')
        geometry = config_dir / 'geometry.json'
        geometry.write_text('{"custom": "preserve this byte layout"}\n')
        before = {path: path.read_bytes() for path in (sim, geometry)}
        self.register()
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_reference_source_copied_from_saved_snapshot_and_mode_can_change(self):
        config = deepcopy(self.kwargs['cell_config'])
        config['model_source']['mode'] = 'reference'
        self.register(cell_config=config)
        snapshot = (self.tune / 'model_sources/Cell.hoc').read_bytes()
        self.entry.write_text('different upstream now')
        result = create_working_copy(source_tune_dir=self.tune, content_policy='model', source_mode='copy')
        target = Path(result['target'])
        self.assertEqual((target / 'model_sources/Cell.hoc').read_bytes(), snapshot)
        self.assertEqual(self.entry.read_text(), 'different upstream now')
        config = json.loads((self.tune / 'cell_configs/cell_config.json').read_text())
        self.assertEqual(config['model_source']['mode'], 'reference')

    def test_missing_disabled_template_group_can_be_created_without_overwriting_index(self):
        self.register()
        syn = self.tune / 'cell_configs/syn_config.json'
        syn.write_text('{"group_files":["syn_groups/input_blocks_template.json"]}\n')
        before = syn.read_bytes()
        config = json.loads((self.tune / 'cell_configs/cell_config.json').read_text())
        create_setup_outputs(self.tune, cell_config=config, include_required=True, synapse_options={})
        self.assertEqual(syn.read_bytes(), before)
        self.assertTrue((syn.parent / 'syn_groups/input_blocks_template.json').is_file())

    def test_copy_uses_saved_snapshot_even_changed_or_missing_upstream(self):
        self.register()
        snapshot = (self.tune / 'model_sources/Cell.hoc').read_bytes()
        self.entry.write_text('changed upstream')
        for mode, name in (('inherit', 'tuned'), ('copy', 'copy_again'), ('reference', 'shared')):
            self.entry.unlink(missing_ok=True)
            before = self.files()
            report = create_working_copy(source_tune_dir=self.tune, target_tune_name=name,
                                         content_policy='model', source_mode=mode, dry_run=True)
            self.assertEqual(self.files(), before)
            if mode == 'reference':
                self.assertIn('unavailable', ' '.join(report['warnings']))
            result = create_working_copy(source_tune_dir=self.tune, target_tune_name=name,
                                         content_policy='model', source_mode=mode)
            destination = Path(result['target'])
            self.assertEqual((destination / 'model_sources/Cell.hoc').read_bytes(), snapshot)
            config = json.loads((destination / 'cell_configs/cell_config.json').read_text())
            self.assertEqual(config['tune'], name)
            self.assertEqual(config['model_source']['mode'], 'reference' if mode == 'reference' else 'copy')
            self.assertEqual((self.tune / 'model_sources/Cell.hoc').read_bytes(), snapshot)

    def test_copy_exclusions_dependencies_collisions_and_source_preservation(self):
        self.register()
        for relative in ('output_data/run/results.npy', 'exports/plot.png', 'archives/run.zip',
                         'modfiles/x86_64/libnrnmech.so', '.scp_build/cache', 'model_sources/__pycache__/a.pyc', 'data/rates.csv'):
            path = self.tune / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'data')
        create_setup_outputs(self.tune, target_options={'manual_fi_curve': {'csv': 'data/rates.csv'}})
        before = self.files()
        result = create_working_copy(source_tune_dir=self.tune, content_policy='model')
        target = Path(result['target'])
        self.assertEqual((target / 'data/rates.csv').read_bytes(), b'data')
        for name in ('output_data', 'exports', 'archives', '.scp_build'):
            self.assertFalse((target / name).exists())
        for relative, contents in before.items():
            self.assertEqual((self.root / relative).read_bytes(), contents)
        for name in ('tuned', '../escape', 'orig', '/tmp/out', 'bad/name'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                create_working_copy(source_tune_dir=self.tune, target_tune_name=name, content_policy='model')

    def test_legacy_dry_run_retains_legacy_policy_without_overwriting(self):
        self.register()
        result = self.tune / 'output_data/old_run/results.pkl'
        result.parent.mkdir(parents=True)
        result.write_bytes(b'legacy result')
        before = self.files()
        preview = create_working_copy(source_tune_dir=self.tune, dry_run=True)
        self.assertEqual(preview['content_policy'], 'legacy')
        self.assertIn('output_data/old_run/results.pkl', preview['files'])
        self.assertEqual(self.files(), before)
        target = self.tune.with_name('tuned')
        target.mkdir()
        (target / 'existing').write_text('preserve')
        before = self.files()
        preview = create_working_copy(source_tune_dir=self.tune, dry_run=True, overwrite=True)
        self.assertTrue(preview['would_overwrite'])
        self.assertFalse(preview['overwritten'])
        self.assertEqual(self.files(), before)

    def test_required_input_under_excluded_results_blocks_before_writes(self):
        self.register()
        required = self.tune / 'output_data/rates.csv'
        required.parent.mkdir()
        required.write_text('current,rate\n')
        create_setup_outputs(self.tune, target_options={'manual_fi_curve': {'csv': 'output_data/rates.csv'}})
        before = self.files()
        with self.assertRaisesRegex(ValueError, 'required input.*excluded generated'):
            create_working_copy(source_tune_dir=self.tune, content_policy='model')
        self.assertEqual(self.files(), before)

    def test_external_input_preserved_and_absolute_local_input_rebased(self):
        self.register()
        local = self.tune / 'data/rates.csv'
        local.parent.mkdir()
        local.write_text('current,rate\n')
        create_setup_outputs(self.tune, target_options={'manual_fi_curve': {'csv': str(local)}})
        result = create_working_copy(source_tune_dir=self.tune, content_policy='model')
        target = Path(result['target'])
        config = json.loads((target / 'cell_configs/target_config.json').read_text())
        self.assertEqual(config['manual']['fi_curve']['csv'], str(target / 'data/rates.csv'))

    def test_absolute_tune_local_group_references_follow_the_new_copy(self):
        self.register()
        create_setup_outputs(self.tune, synapse_options={})
        group = self.tune / 'cell_configs/syn_groups/input_blocks_template.json'
        index = self.tune / 'cell_configs/syn_config.json'
        index.write_text(json.dumps({'group_files': [str(group)]}))
        before = index.read_bytes()
        result = create_working_copy(source_tune_dir=self.tune, content_policy='model')
        target = Path(result['target'])
        copied = json.loads((target / 'cell_configs/syn_config.json').read_text())
        self.assertEqual(copied['group_files'], [str(target / group.relative_to(self.tune))])
        self.assertEqual(index.read_bytes(), before)

    def test_project_relative_external_inputs_and_disabled_placeholders_preserved(self):
        self.register()
        external = self.root / 'project/cells/Other/data/rates.csv'
        external.parent.mkdir(parents=True)
        external.write_text('Time,AvgFiringRate\n')
        directory = self.tune / 'cell_configs/syn_groups'
        directory.mkdir()
        group = {'active': {'state': True, 'source': {'path': 'cells/Other/data/rates.csv'}},
                 'disabled': {'state': False, 'source': {'path': 'output_data/missing.csv'}}}
        (directory / 'inputs.json').write_text(json.dumps(group))
        result = create_working_copy(source_tune_dir=self.tune, content_policy='model')
        target = Path(result['target'])
        self.assertEqual(json.loads((target / 'cell_configs/syn_groups/inputs.json').read_text()), group)
        self.assertEqual(external.read_text(), 'Time,AvgFiringRate\n')

    def test_symlink_escape_rejected_and_existing_skip_unchanged(self):
        self.register()
        outside = self.root / 'outside'
        outside.write_bytes(b'outside')
        (self.tune / 'escape').symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'symlink escapes'):
            create_working_copy(source_tune_dir=self.tune, content_policy='model')
        before = self.files()
        scaffold_base_configs(tune_dir=self.tune, cell_name='Cell', tune_name='orig', specimen_id=None,
                              model_type='perisomatic', config_mode='skip', v_init_mV=-99)
        self.assertEqual(self.files(), before)


if __name__ == '__main__':
    unittest.main()
