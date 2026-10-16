"""New-cell setup works in both SCP namespaces and project-owned directories."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import uuid

from scp_pipeline.notebooks import PipelineNotebookUI
from scp_pipeline.projects import PROJECT_FILE, load_project_settings
from scp_pipeline.setup import preview_workspace_registration
from tests.pipeline_ui_test_support import all_descendants


class SCPRegistrationUITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / 'SCP'
        self.root.mkdir()
        (self.root / 'run_pipeline.py').write_text('', encoding='utf-8')
        (self.root / 'modules').mkdir()
        (self.root / 'cells/PV/tunes/tuned').mkdir(parents=True)
        (self.root / 'projects/demo/cells').mkdir(parents=True)
        self.external = self.base / 'external'
        self.source = self.base / 'native'
        self.source.mkdir()
        self.template = 'Registration_' + uuid.uuid4().hex
        self.entry = self.source / 'Cell.hoc'
        self.entry.write_text(f'''begintemplate {self.template}
public body, branches, whole
create body, branches[1]
objref whole
proc init() {{
    whole = new SectionList()
    body {{ L=20 diam=20 nseg=1 insert pas whole.append() }}
    branches[0] {{ L=40 diam=2 nseg=3 insert pas whole.append() }}
    connect branches[0](0), body(1)
}}
endtemplate {self.template}
begintemplate {self.template}_Other
public soma
create soma
proc init() {{ soma {{ L=10 diam=10 insert pas }} }}
endtemplate {self.template}_Other
''', encoding='utf-8')
        self.sections = {'soma': 'body', 'dend': 'branches', 'all': 'whole'}

    def files(self):
        return {p.relative_to(self.base).as_posix(): p.read_bytes()
                for p in self.base.rglob('*') if p.is_file()}

    def ui(self, namespace='demo', *, external=False, restore=False):
        ui = PipelineNotebookUI(self.root, {'project_name': namespace, 'cell_name': None,
                                           'tune_name': None},
                                project_root=self.external if external else None,
                                restore_notebook_settings=restore)
        ui.step1_panel()
        return ui

    def configure(self, ui):
        c = ui.controls
        c['setup_action'].value = 'register'
        c['cell_name'].editor.value = 'NewCell'
        c['tune_name'].editor.value = 'orig'
        c['setup_entry'].value = str(self.entry)
        c['setup_template'].value = self.template
        c['setup_sections'].value = json.dumps(self.sections)
        c['setup_voltage'].value = '-65'
        c['setup_celsius'].value = '30'
        c['output_dir'].value = 'recordings'

    def click(self, ui, key):
        with redirect_stdout(io.StringIO()):
            ui.controls[key].click()

    def test_optional_generation_in_scp_and_project_owned_registration(self):
        for namespace, external in ((None, False), ('demo', False), (None, True)):
            with self.subTest(namespace=namespace, external=external):
                ui = self.ui(namespace, external=external)
                self.configure(ui)
                c = ui.controls
                c['setup_targets_enabled'].value = True
                c['setup_target_manual_passive_v_rest_mV'].value = '-66'
                c['setup_synapses_enabled'].value = True
                before = self.files()
                self.click(ui, 'setup_preview')
                self.assertEqual(ui._status_kinds['setup_preview'], 'complete', ui.outputs['setup_preview'].outputs)
                self.assertEqual(before, self.files())
                with mock.patch('scp_pipeline.model.load_cell') as load:
                    self.click(ui, 'setup_save')
                load.assert_not_called()
                self.assertEqual(ui._status_kinds['setup_save'], 'complete', ui.outputs['setup_save'].outputs)
                tune = ui._selected_tune_dir()
                target = json.loads((tune / 'cell_configs/target_config.json').read_text())
                self.assertEqual(target['manual']['passive']['v_rest_mV'], -66)
                self.assertTrue((tune / 'cell_configs/syn_groups/input_blocks_template.json').is_file())
                self.assertFalse(ui._setup_attempted)

    def test_registration_preview_save_and_reopen_in_every_location(self):
        for namespace, external in ((None, False), ('demo', False), (None, True)):
            with self.subTest(namespace=namespace, external=external):
                ui = self.ui(namespace, external=external)
                self.configure(ui)
                owner = self.external if external else self.root / 'projects/demo' if namespace else self.root
                tune = owner / 'cells/NewCell/tunes/orig'
                before = self.files()
                with mock.patch('scp_pipeline.model.load_cell') as load:
                    self.click(ui, 'setup_preview')
                    self.assertEqual(ui._status_kinds['setup_preview'], 'complete', ui.outputs['setup_preview'].outputs)
                    self.assertEqual(self.files(), before)
                    self.click(ui, 'setup_save')
                load.assert_not_called()
                self.assertEqual(ui._status_kinds['setup_save'], 'complete', ui.outputs['setup_save'].outputs)
                self.assertFalse(ui._setup_attempted)
                self.assertIsNone(ui.pipeline_state)
                self.assertEqual(ui._selected_tune_dir(), tune)
                self.assertEqual(ui.controls['setup_action'].value, 'existing')
                self.assertIn('NewCell', ui.controls['cell_name'].options)
                self.assertIn('orig', ui.controls['tune_name'].options)
                self.assertEqual(ui.controls['cell_name'].dropdown.value, 'NewCell')
                self.assertEqual(ui.controls['tune_name'].dropdown.value, 'orig')
                self.assertEqual(ui.controls['cell_name'].editor.layout.display, 'none')
                config = json.loads((tune / 'cell_configs/cell_config.json').read_text())
                self.assertEqual(config['cell_name'], 'NewCell')
                self.assertEqual(config.get('project_name'), 'external' if external else namespace)
                if external:
                    self.assertEqual(config['workspace_layout'], 'project_owned')
                else:
                    self.assertNotIn('workspace_layout', config)
                    self.assertNotIn('workspace_project_root', config)
                self.assertEqual((tune / 'model_sources/Cell.hoc').read_bytes(), self.entry.read_bytes())
                self.assertFalse((tune / 'cell_configs/syn_config.json').exists())
                settings = load_project_settings(owner)
                self.assertEqual(settings['selected_tune'], 'cells/NewCell/tunes/orig')
                expected_output = (self.external if external else self.root) / 'recordings'
                from scp_pipeline.projects import resolve_project_path
                self.assertEqual(resolve_project_path(owner, settings['output_dir']), expected_output)
                if not external:
                    self.assertEqual(load_project_settings(self.root).get('selected_project'), namespace)
                before_reopen = self.files()
                reopened = self.ui(None, external=external, restore=True)
                self.assertEqual(reopened._selected_tune_dir(), tune)
                self.assertEqual(self.files(), before_reopen)

    def test_new_name_inputs_live_beside_registration_not_daily_navigation(self):
        from ipywidgets.embed import dependency_state
        ui = self.ui()
        c = ui.controls
        c['setup_action'].value = 'register'
        daily = list(all_descendants(c['setup_daily_box']))
        registration = list(all_descendants(c['setup_workspace_box']))
        for key in ('cell_name', 'tune_name'):
            self.assertIn(c[key].dropdown, daily)
            self.assertNotIn(c[key].editor, daily)
            self.assertIn(c[key].editor, registration)
            self.assertNotEqual(c[key].editor.layout.display, 'none')
        self.assertLess(registration.index(c['setup_action']), registration.index(c['cell_name'].editor))
        self.assertLess(registration.index(c['tune_name'].editor), registration.index(c['setup_entry']))
        state = dependency_state([ui.step1_panel()], drop_defaults=False)
        row = c['setup_name_entry_row']
        self.assertEqual(state[row.model_id]['state']['children'],
                         ['IPY_MODEL_' + c[key].editor.model_id for key in ('cell_name', 'tune_name')])
        c['setup_action'].value = 'existing'
        self.assertEqual(row.layout.display, 'none')
        c['setup_action'].value = 'live'
        self.assertNotEqual(row.layout.display, 'none')

    def test_empty_namespace_opens_setup_and_switching_to_existing_hides_new_inputs(self):
        ui = self.ui(None)
        self.assertEqual(ui.controls['setup_action'].value, 'existing')
        before = self.files()
        ui.controls['project_name'].value = 'demo'
        self.assertEqual(ui.controls['setup_action'].value, 'register')
        self.assertTrue(ui.controls['setup_manage_toggle'].value)
        self.assertEqual(ui.controls['cell_name'].value, '')
        self.assertEqual(ui.controls['tune_name'].value, '')
        self.assertEqual(self.files(), before)
        ui.controls['project_name'].value = ''
        self.assertEqual(ui.controls['setup_action'].value, 'existing')
        self.assertEqual(ui.controls['setup_name_entry_row'].layout.display, 'none')

    def test_editing_names_preserves_the_other_name_and_selected_constructor(self):
        ui = self.ui()
        self.configure(ui)
        ui.controls['tune_name'].editor.value = 'working'
        for text in ('N', 'Ne', 'New', 'NewCell'):
            ui.controls['cell_name'].editor.value = text
            self.assertEqual(ui.controls['tune_name'].value, 'working')
            self.assertEqual(ui.controls['setup_template'].value, self.template)
        self.assertEqual(ui._selected_tune_dir(), self.root / 'projects/demo/cells/NewCell/tunes/working')

    def test_discovery_and_loader_specific_controls_are_nonexecuting(self):
        ui = self.ui()
        self.configure(ui)
        c = ui.controls
        before = self.files()
        self.click(ui, 'setup_discover')
        self.assertEqual(ui._status_kinds['setup_discover'], 'complete')
        self.assertEqual([value for _label, value in c['setup_candidates'].options if value],
                         [self.template, self.template + '_Other'])
        c['setup_candidates'].value = self.template
        self.assertEqual(c['setup_template'].value, self.template)
        self.assertEqual(self.files(), before)
        self.assertEqual(c['setup_callable'].layout.display, 'none')
        self.assertEqual(c['setup_kwargs'].layout.display, 'none')
        c['setup_loader'].value = 'python_factory'
        self.assertNotEqual(c['setup_callable'].layout.display, 'none')
        self.assertNotEqual(c['setup_kwargs'].layout.display, 'none')
        self.assertEqual(c['setup_template'].layout.display, 'none')

    def test_explicit_tune_path_wins_over_shorthand_and_keeps_effective_identity(self):
        ui = self.ui()
        self.configure(ui)
        destination = self.root / 'projects/other/cells/Actual/tunes/reference'
        ui.controls['tune_dir_override'].value = str(destination)
        self.click(ui, 'setup_save')
        self.assertEqual(ui._status_kinds['setup_save'], 'complete', ui.outputs['setup_save'].outputs)
        config = json.loads((destination / 'cell_configs/cell_config.json').read_text())
        self.assertEqual((config['project_name'], config['cell_name'], config['tune']),
                         ('other', 'Actual', 'reference'))
        self.assertEqual(ui.controls['project_name'].value, 'other')
        self.assertEqual(ui.controls['cell_name'].value, 'Actual')
        self.assertEqual(ui.controls['tune_name'].value, 'reference')
        self.assertEqual(ui.controls['project_root'].value, str(self.root / 'projects/other'))
        self.assertEqual(ui.controls['project_label'].value, 'other')
        self.assertEqual(load_project_settings(self.root)['selected_project'], 'other')
        self.assertFalse((self.root / 'projects/demo/cells/NewCell').exists())

    def test_errors_are_recoverable_and_existing_recipe_is_not_overwritten(self):
        ui = self.ui()
        self.configure(ui)
        original = self.entry.read_bytes()
        before = self.files()
        self.click(ui, 'step1_run')
        self.assertFalse(ui._setup_attempted)
        self.assertEqual(self.files(), before)
        for key, bad, fixed in (('setup_voltage', '', '-65'), ('setup_args', '{bad', '[]'),
                                ('setup_entry', str(self.source / 'missing.hoc'), str(self.entry))):
            ui.controls[key].value = bad
            self.click(ui, 'setup_preview')
            self.assertEqual(ui._status_kinds['setup_preview'], 'error')
            self.assertFalse(ui._setup_attempted)
            self.assertEqual(self.files(), before)
            ui.controls[key].value = fixed
        self.click(ui, 'setup_save')
        self.assertEqual(ui._status_kinds['setup_save'], 'complete')
        saved = self.files()
        ui.controls['setup_action'].value = 'register'
        self.click(ui, 'setup_save')
        self.assertEqual(ui._status_kinds['setup_save'], 'error')
        self.assertEqual(self.files(), saved)
        self.assertEqual(self.entry.read_bytes(), original)

    def test_reference_mode_snapshots_sources_but_keeps_the_original_as_working_source(self):
        from scp_pipeline.loaders.source_context import effective_source_root
        ui = self.ui()
        self.configure(ui)
        ui.controls['setup_source_mode'].value = 'reference'
        before = self.entry.read_bytes()
        self.click(ui, 'setup_save')
        self.assertEqual(ui._status_kinds['setup_save'], 'complete', ui.outputs['setup_save'].outputs)
        tune = ui._selected_tune_dir()
        config = json.loads((tune / 'cell_configs/cell_config.json').read_text())
        self.assertEqual(effective_source_root(config, tune), self.source)
        self.assertEqual((tune / 'model_sources/Cell.hoc').read_bytes(), before)
        self.assertEqual(self.entry.read_bytes(), before)

    def test_bad_navigation_is_rejected_before_sources_or_configs_are_staged(self):
        ui = self.ui()
        self.configure(ui)
        path = self.root / 'projects/demo' / PROJECT_FILE
        path.write_text('[]', encoding='utf-8')
        before = self.files()
        self.click(ui, 'setup_save')
        self.assertEqual(ui._status_kinds['setup_save'], 'error')
        self.assertEqual(self.files(), before)
        self.assertFalse((path.parent / 'cells/NewCell').exists())

    def test_registration_api_requires_an_explicit_workspace_scope(self):
        with self.assertRaisesRegex(ValueError, 'repo_root.*project_root'):
            preview_workspace_registration(tune_dir=self.root / 'cells/NewCell/tunes/orig',
                                           cell_name='NewCell', tune_name='orig',
                                           cell_config={}, conditions={})

    def test_existing_allen_bundle_stages_only_explicit_native_dependencies(self):
        manifest = self.source / 'manifest.json'
        fit = self.source / 'parameters_fit.json'
        morphology = self.source / 'morphology.swc'
        manifest.write_text(json.dumps({'biophys': [{'model_file': [fit.name]}],
                                        'manifest': [{'key': 'MORPHOLOGY', 'type': 'file',
                                                      'spec': morphology.name}]}), encoding='utf-8')
        fit.write_text('{}', encoding='utf-8')
        morphology.write_text('1 1 0 0 0 5 -1\n', encoding='utf-8')
        ui = self.ui()
        self.configure(ui)
        ui.controls['setup_loader'].value = 'allen_manifest'
        ui.controls['setup_entry'].value = str(manifest)
        ui.controls['setup_files'].value = json.dumps([fit.name, morphology.name])
        original = self.files()
        with mock.patch('scp_pipeline.model.load_cell') as load:
            self.click(ui, 'setup_preview')
            self.assertEqual(ui._status_kinds['setup_preview'], 'complete', ui.outputs['setup_preview'].outputs)
            self.assertEqual(self.files(), original)
            self.click(ui, 'setup_save')
        load.assert_not_called()
        self.assertEqual(ui._status_kinds['setup_save'], 'complete', ui.outputs['setup_save'].outputs)
        tune = ui._selected_tune_dir()
        self.assertEqual(sorted(p.name for p in (tune / 'model_sources').iterdir()),
                         sorted([manifest.name, fit.name, morphology.name]))

    def test_allen_download_registration_in_each_location_remains_nonconstructing(self):
        from scp_pipeline.setup.step1_prepare import prepare_base_configs

        def acquisition_double(**kwargs):
            tune = kwargs['tune_dir']
            tune.mkdir(parents=True)
            (tune / 'manifest.json').write_text('{}', encoding='utf-8')
            if kwargs.get('do_base_configs', True):
                prepare_base_configs(tune_dir=tune, cell_name=kwargs['cell_name'], tune_name=kwargs['tune_name'],
                                     specimen_id=kwargs['specimen_id'], v_init_mV=kwargs['v_init_mV'],
                                     celsius_C=kwargs['celsius_C'])
            return {'status': 'ok', 'tune_dir': str(tune)}

        for namespace, external in ((None, False), ('demo', False), (None, True)):
            with self.subTest(namespace=namespace, external=external):
                ui = self.ui(namespace, external=external)
                self.configure(ui)
                c = ui.controls
                c['setup_loader'].value = 'adb'
                c['setup_specimen'].value = '12345'
                before = self.files()
                with mock.patch('scp_pipeline.setup.prepare_tune', side_effect=acquisition_double) as acquire, \
                        mock.patch('scp_pipeline.model.load_cell') as load:
                    self.click(ui, 'setup_preview')
                    self.assertEqual(ui._status_kinds['setup_preview'], 'complete')
                    acquire.assert_not_called()
                    self.assertEqual(self.files(), before)
                    self.click(ui, 'setup_save')
                self.assertEqual(ui._status_kinds['setup_save'], 'complete', ui.outputs['setup_save'].outputs)
                load.assert_not_called()
                self.assertFalse(ui._setup_attempted)
                self.assertEqual(c['setup_action'].value, 'existing')
                kwargs = acquire.call_args.kwargs
                for key in ('do_compile_modfiles', 'load_compiled_dll', 'do_validate',
                            'do_base_configs', 'do_target_config', 'do_synapse_configs'):
                    self.assertFalse(kwargs[key])
                self.assertEqual((kwargs['v_init_mV'], kwargs['celsius_C']), (-65, 30))
                config = json.loads((ui._selected_tune_dir() / 'cell_configs/cell_config.json').read_text())
                self.assertEqual(config.get('project_name'), 'external' if external else namespace)
                self.assertEqual(config.get('workspace_layout'), 'project_owned' if external else None)

    def test_allen_download_rejects_invalid_inputs_before_acquisition(self):
        ui = self.ui()
        self.configure(ui)
        ui.controls['setup_loader'].value = 'adb'
        before = self.files()
        with mock.patch('scp_pipeline.setup.prepare_tune') as acquire:
            for specimen in ('', '-1', 'not an id'):
                ui.controls['setup_specimen'].value = specimen
                self.click(ui, 'setup_save')
                self.assertEqual(ui._status_kinds['setup_save'], 'error')
            ui.controls['setup_specimen'].value = '12345'
            ui.controls['setup_voltage'].value = 'nan'
            self.click(ui, 'setup_save')
            self.assertEqual(ui._status_kinds['setup_save'], 'error')
        acquire.assert_not_called()
        self.assertEqual(self.files(), before)

    @unittest.skipUnless(importlib.util.find_spec('neuron'), 'NEURON unavailable')
    def test_scp_hoc_and_python_registration_protocols_and_step5_in_fresh_processes(self):
        from scp_pipeline.setup.mechanisms import find_nrnivmodl
        if not find_nrnivmodl() or not (shutil.which('c++') or shutil.which('g++')):
            self.skipTest('nrnivmodl/C++ compiler unavailable for the synthetic MOD dependency check')
        fixture = Path(__file__).resolve().parent / 'fixtures/custom_models/mechanisms'
        shutil.copytree(fixture, self.source / 'mechanisms')
        self.entry.write_text(self.entry.read_text().replace('insert pas', 'insert pas insert scp_synthetic_leak'),
                              encoding='utf-8')
        builder = self.source / 'builder.py'
        builder.write_text('def build():\n    from neuron import h\n'
                           '    cell = h.Section(name="synthetic")\n'
                           '    cell.L = cell.diam = 20\n    cell.insert("pas")\n'
                           '    cell.insert("scp_synthetic_leak")\n    return cell\n',
                           encoding='utf-8')
        native_before = {p.relative_to(self.source).as_posix(): p.read_bytes()
                         for p in self.source.rglob('*') if p.is_file()}
        for loader in ('hoc_template', 'python_factory'):
            with self.subTest(loader=loader):
                program = f'''
from pathlib import Path
import json
import numpy as np
from scp_pipeline.notebooks import PipelineNotebookUI, run_fresh_simulation
from scp_pipeline.protocols import run_passive_stage, run_active_protocol_stage, run_fi_curve_stage
from scp_pipeline.analysis import load_results
from scp_pipeline.workspaces import result_workspace_identity
root = Path({str(self.root)!r})
ui = PipelineNotebookUI(root, {{'project_name':'demo', 'cell_name':None, 'tune_name':None}})
ui.step1_panel()
c = ui.controls
c['setup_action'].value = 'register'
c['cell_name'].editor.value = {loader!r}
c['tune_name'].editor.value = 'orig'
c['setup_loader'].value = {loader!r}
c['setup_entry'].value = {str(self.entry if loader == 'hoc_template' else builder)!r}
c['setup_modfiles'].value = 'mechanisms'
c['setup_template'].value = {self.template!r}
c['setup_sections'].value = {(json.dumps(self.sections) if loader == 'hoc_template' else '{}')!r}
c['setup_callable'].value = 'build'
c['setup_voltage'].value = '-65'
c['setup_celsius'].value = '30'
c['output_dir'].value = 'recordings'
c['setup_targets_enabled'].value = True
c['setup_synapses_enabled'].value = True
c['setup_save'].click()
assert ui._status_kinds['setup_save']=='complete', ui.outputs['setup_save'].outputs
assert not ui._setup_attempted
original = ui._selected_tune_dir()
assert (original / 'cell_configs/target_config.json').is_file()
assert (original / 'cell_configs/syn_config.json').is_file()
c['setup_copy_preview'].click()
assert ui._status_kinds['setup_copy_preview']=='complete', ui.outputs['setup_copy_preview'].outputs
assert not original.with_name('tuned').exists()
c['setup_copy_create'].click()
assert ui._status_kinds['setup_copy_create']=='complete', ui.outputs['setup_copy_create'].outputs
assert ui._selected_tune_dir() == original.with_name('tuned')
assert not ui._setup_attempted
c['step1_run'].click()
assert ui.pipeline_state is not None, ui.step1_load_log
assert ui.pipeline_state.output_dir == root / 'recordings'
cell = ui.pipeline_state.cell
assert len(cell.all) == {2 if loader == 'hoc_template' else 1}
assert len(cell.soma) == 1
assert 'scp_synthetic_leak' in cell.soma[0].psection()['density_mechs']
protocol = {{'stim_delay':5,'stim_dur':20,'h_tstop':30,'h_dt':0.025}}
runs = [run_passive_stage(ui.pipeline_state, amps_pA=[-10], protocol_overrides=protocol),
        run_active_protocol_stage(ui.pipeline_state, active_amps_pA=[10], protocol_overrides=protocol),
        run_fi_curve_stage(ui.pipeline_state, fi_amps_pA=[0,10], protocol_overrides=protocol)]
for run in runs:
    for amp, times in run.records['T'].items():
        assert np.isfinite(times).all() and np.isfinite(run.records['V'][amp]).all()
        assert len(times) == len(run.records['V'][amp])
result = run_fresh_simulation(ui.pipeline_state, iclamp=True, stream_output=False,
    sim_overrides={{'save_plots':False,'save_output_metrics':False,'tstop':5,
                   'iclamp':{{'delay_ms':1,'dur_ms':2,'tstop_ms':5,'record_currents':False}}}})
assert result.manifest_path.is_relative_to(root / 'recordings')
saved = load_results(result.manifest_path)
assert result_workspace_identity(saved) == {{'project_name':'demo','cell_name':{loader!r},'tune_name':'tuned'}}
assert np.isfinite(saved['traces']['V']).all()
assert saved['meta']['conditions'] == {{'v_init_mV':-65.0,'celsius_C':30.0}}
'''
                result = subprocess.run([sys.executable, '-c', program], cwd=self.base,
                                        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg'),
                                        capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(native_before, {p.relative_to(self.source).as_posix(): p.read_bytes()
                                        for p in self.source.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main()
