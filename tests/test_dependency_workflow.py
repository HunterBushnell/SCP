"""Dependency acceptance tests use temporary, non-biological project assets."""
import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid
import zipfile

from scp_pipeline.projects import load_project_settings, save_project_settings
from scp_pipeline.setup import preview_workspace_registration, register_workspace
from scp_pipeline.workspaces import assert_same_workspace, WorkspaceMismatchError

ROOT = Path(__file__).resolve().parents[1]


class DependencyWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'project'
        self.root.mkdir()
        self.name = 'Synthetic_' + uuid.uuid4().hex
        self.source = self.root / 'native'
        self.source.mkdir()
        self.hoc = self.source / 'Cell.hoc'
        self.hoc.write_text(f'''begintemplate {self.name}
public soma, all
create soma
objref all
proc init() {{
    all = new SectionList()
    soma {{ L=20 diam=20 nseg=1 insert pas all.append() }}
}}
endtemplate {self.name}
''')
        self.tune = self.root / 'cells/Cell/tunes/orig'
        self.config = {'cell_loader': 'hoc_template', 'paths': {'hoc_template': 'Cell.hoc', 'modfiles': None},
                       'hoc_template': {'template_name': self.name, 'constructor_args': []},
                       'model_source': {'mode': 'copy', 'source_root': str(self.source), 'files': ['Cell.hoc']}}
        self.arguments = dict(project_root=self.root, tune_dir=self.tune, cell_name='Cell', tune_name='orig',
                              cell_config=self.config, conditions={'v_init_mV': -65, 'celsius_C': 30})

    def register(self):
        with redirect_stdout(io.StringIO()):
            return register_workspace(dry_run=False, **self.arguments)

    def subprocess(self, code, cwd=None):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
        env.pop('SCP_ROOT', None)
        result = subprocess.run([sys.executable, '-c', code], cwd=cwd or self.root,
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_preview_save_and_source_ownership(self):
        before = self.hoc.read_bytes()
        preview = preview_workspace_registration(**self.arguments)
        self.assertFalse(self.tune.exists())
        self.assertEqual(preview['sources']['file_count'], 1)
        self.register()
        self.assertEqual(load_project_settings(self.root)['selected_tune'], 'cells/Cell/tunes/orig')
        self.assertEqual(self.hoc.read_bytes(), before)
        self.assertEqual((self.tune / 'model_sources/Cell.hoc').read_bytes(), before)
        self.assertFalse((self.tune / 'cell_configs/syn_config.json').exists())
        config = json.loads((self.tune / 'cell_configs/cell_config.json').read_text())
        self.assertEqual(config['project_name'], 'project')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.register()

    def test_flat_project_source_selection(self):
        flat = self.root / 'Flat.hoc'
        flat.write_bytes(self.hoc.read_bytes())
        self.config['model_source'].update(source_root=str(self.root), files=['Flat.hoc'])
        self.config['paths']['hoc_template'] = 'Flat.hoc'
        self.register()
        self.assertEqual(sorted(p.name for p in (self.tune / 'model_sources').iterdir()), ['Flat.hoc'])

    def test_settings_are_explicit_and_do_not_edit_science(self):
        self.register()
        config = self.tune / 'cell_configs/sim_config.json'
        before = config.read_bytes()
        save_project_settings(self.root, output_dir=self.root / 'results')
        self.assertEqual(load_project_settings(self.root)['output_dir'], 'results')
        self.assertEqual(config.read_bytes(), before)
        (self.root / '.scp_project.json').write_text('[]')
        with self.assertRaisesRegex(ValueError, 'format_version'):
            load_project_settings(self.root)

    def test_host_modules_do_not_conflict_with_installed_namespace(self):
        host = self.root / 'modules'
        host.mkdir()
        (host / '__init__.py').write_text('host_project = True\n')
        self.subprocess('import modules; import scp_pipeline.model; import scp_pipeline.protocols; '
                        'assert modules.host_project; from scp_pipeline.simulation import SimulationSession')

    def test_legacy_namespace_shares_classes_and_registries(self):
        self.subprocess(f'import sys; sys.path.insert(0, {str(ROOT)!r}); '
                        'import modules.model, scp_pipeline.model; '
                        'import modules.loaders.registry, scp_pipeline.loaders.registry; '
                        'assert modules.model is scp_pipeline.model; '
                        'assert modules.loaders.registry is scp_pipeline.loaders.registry; '
                        'from modules.model.load_cell import load_cell; '
                        'from scp_pipeline.model import load_cell as public; assert public is load_cell')

    def test_checkout_provenance_fallback_matches_package_version(self):
        from scp_pipeline import runtime

        tree = ast.parse((ROOT / 'setup.py').read_text(encoding='utf-8'))
        setup_call = next(node for node in ast.walk(tree)
                          if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                          and node.func.id == 'setup')
        package_version = next(ast.literal_eval(item.value) for item in setup_call.keywords
                               if item.arg == 'version')
        with patch.object(runtime.metadata, 'version',
                          side_effect=runtime.metadata.PackageNotFoundError('scp-pipeline')) as lookup:
            with patch.object(runtime, 'code_root', return_value=self.root):
                provenance = runtime.software_provenance()
        lookup.assert_called_once_with('scp-pipeline')
        self.assertEqual(provenance, {'name': 'scp-pipeline', 'version': package_version})

    def test_preflight_conditions_and_overlap(self):
        from scp_pipeline.simulation.result_paths import _build_output_path
        for stem in ('../native', '/absolute', '.', '..', 'nested/run'):
            with self.assertRaisesRegex(ValueError, 'output_stem'):
                _build_output_path({'output': stem}, self.root / 'recordings')
        self.assertFalse((self.root / 'recordings').exists())
        self.arguments['conditions'] = {}
        with self.assertRaisesRegex(ValueError, 'v_init_mV'):
            preview_workspace_registration(**self.arguments)
        self.arguments['conditions'] = {'v_init_mV': True, 'celsius_C': 30}
        with self.assertRaisesRegex(ValueError, 'v_init_mV'):
            preview_workspace_registration(**self.arguments)
        self.arguments['conditions'] = {'v_init_mV': -65, 'celsius_C': 30}
        self.config['model_source'].update(source_root=str(self.root), folders=['.'])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            preview_workspace_registration(**self.arguments)
        self.assertFalse(self.tune.exists())

    def test_copy_reconstruction_after_relocation_without_upstream(self):
        self.register()
        moved = self.root.parent / 'relocated'
        shutil.copytree(self.root, moved)
        shutil.rmtree(moved / 'native')
        self.subprocess('from pathlib import Path; from scp_pipeline.simulation import SimulationSession; '
                        's=SimulationSession.from_tune(Path("cells/Cell/tunes/orig"), iclamp=True, '
                        'sim_overrides={"tstop": 3, "iclamp":{"dur":1,"delay":1}}); '
                        's.prepare(); r=s.run(); assert r["sim_cfg"]["workspace_identity"]["project_name"]=="project"', cwd=moved)

    def test_external_identity_pooling_guard(self):
        self.register()
        from scp_pipeline.simulation import SimulationSession
        from scp_pipeline.workspaces import apply_workspace_metadata
        session = SimulationSession.from_tune(self.tune)
        left = {'sim_cfg': dict(session.sim_cfg_preview)}
        apply_workspace_metadata(left['sim_cfg'], session.workspace)
        right = json.loads(json.dumps(left))
        right['sim_cfg']['workspace_identity']['project_name'] = 'other'
        right['sim_cfg']['project_name'] = 'other'
        with self.assertRaises(WorkspaceMismatchError):
            assert_same_workspace(left, right)

    def test_ui_validation_is_recoverable_and_save_does_not_load(self):
        from scp_pipeline.notebooks import PipelineNotebookUI
        ui = PipelineNotebookUI(project_root=self.root)
        ui.step1_panel()
        ui.controls['cell_name'].value = 'Cell'
        ui.controls['step1_run'].click()
        self.assertFalse(ui._setup_attempted)
        ui.controls['setup_action'].value = 'register'
        ui.controls['setup_entry'].value = str(self.hoc)
        ui.controls['setup_template'].value = self.name
        ui.controls['setup_voltage'].value = '-65'
        ui.controls['setup_celsius'].value = '30'
        ui.controls['setup_save'].click()
        self.assertFalse(ui._setup_attempted)
        self.assertTrue((self.tune / 'cell_configs/cell_config.json').is_file())

    def test_protocols_and_cli_with_project_owned_output(self):
        self.register()
        self.subprocess('from scp_pipeline.protocols import *; from pathlib import Path; '
                        't=Path("cells/Cell/tunes/orig"); '
                        'before={p:p.read_bytes() for p in (t/"cell_configs").glob("*.json")}; '
                        's=prepare_pipeline_notebook(project_root=Path.cwd(), cell_name="Cell", tune_name="orig"); '
                        'p={"stim_delay":5,"stim_dur":20,"h_tstop":30,"h_dt":0.025}; '
                        'run_passive_stage(s, amps_pA=[-10], protocol_overrides=p); '
                        'run_active_protocol_stage(s, active_amps_pA=[10], protocol_overrides=p); '
                        'run_fi_curve_stage(s, fi_amps_pA=[0,10], protocol_overrides=p); '
                        'assert all(p.read_bytes()==v for p,v in before.items())')
        destination = self.root / 'custom_results'
        save_project_settings(self.root, output_dir=destination)
        result = subprocess.run([sys.executable, '-m', 'scp_pipeline', 'run', '--project-root', str(self.root),
                                 '--iclamp', '--force-save', '--output-stem', 'smoke', '--sim-overrides-json',
                                 '{"tstop":4,"iclamp":{"delay":1,"dur":2}}'],
                                cwd=self.root, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg'),
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(list(destination.glob('*/run_manifest.json')))

    def test_copied_notebook_renders_registers_and_runs_hoc_and_python(self):
        for loader in ('hoc_template', 'python_factory'):
            with self.subTest(loader=loader):
                project = self.root / loader
                project.mkdir()
                notebook = project / '0_pipeline.ipynb'
                shutil.copy2(ROOT / '0_pipeline.ipynb', notebook)
                entry = project / ('Cell.hoc' if loader == 'hoc_template' else 'builder.py')
                if loader == 'hoc_template':
                    entry.write_bytes(self.hoc.read_bytes())
                else:
                    entry.write_text('def build():\n    from neuron import h\n'
                                     '    s=h.Section(name="synthetic")\n'
                                     '    s.L=s.diam=20\n    s.insert("pas")\n    return s\n')
                program = f'''
import json
from pathlib import Path
import numpy as np
from scp_pipeline.notebooks.notebook_settings import write_profile
write_profile(Path('.scp_pipeline_settings.json'), {{'active_amps_pA':[25], 'seed':23}})
notebook=json.loads(Path('0_pipeline.ipynb').read_text())
scope={{'__name__':'__main__'}}
for cell in notebook['cells']:
    if cell['cell_type']=='code':
        exec(compile(''.join(cell['source']), 'copied-notebook', 'exec'), scope)
ui=scope['pipeline_ui']
assert ui.pipeline_state is None
assert not ui.settings['cell_name']
assert ui.settings['active_amps_pA']==[25]
assert ui.controls['active_amps_pA'].value=='25'
assert ui.settings['seed']==23
ui.controls['cell_name'].value='Cell'
ui.controls['tune_name'].editor.value='orig'
ui.controls['setup_action'].value='register'
ui.controls['setup_loader'].value={loader!r}
ui.controls['setup_entry'].value={str(entry)!r}
ui.controls['setup_template'].value={self.name!r}
ui.controls['setup_callable'].value='build'
ui.controls['setup_voltage'].value='-65'
ui.controls['setup_celsius'].value='30'
ui.controls['output_dir'].value='recordings'
ui.controls['setup_preview'].click()
assert ui._status_kinds['setup_preview']=='complete', ui.outputs['setup_preview'].outputs
assert not (Path('cells/Cell/tunes/orig')).exists()
ui.controls['setup_save'].click()
assert ui._status_kinds['setup_save']=='complete', ui.outputs['setup_save'].outputs
assert not ui._setup_attempted
ui.controls['step1_run'].click()
assert ui.pipeline_state is not None, ui.step1_load_log
assert ui.settings['active_amps_pA']==[25]
assert ui.controls['active_amps_pA'].value=='25'
ui.controls['active_amps_pA'].value='33'
for cell in notebook['cells']:
    if cell['cell_type']=='code':
        exec(compile(''.join(cell['source']), 'copied-notebook-rerun', 'exec'), scope)
assert scope['pipeline_ui'] is ui
assert ui.controls['active_amps_pA'].value=='33'
from scp_pipeline.protocols import run_passive_stage, run_active_protocol_stage, run_fi_curve_stage
protocol={{'stim_delay':5,'stim_dur':20,'h_tstop':30,'h_dt':0.025}}
characterizations=[run_passive_stage(ui.pipeline_state,amps_pA=[-10],protocol_overrides=protocol),
    run_active_protocol_stage(ui.pipeline_state,active_amps_pA=[10],protocol_overrides=protocol),
    run_fi_curve_stage(ui.pipeline_state,fi_amps_pA=[0,10],protocol_overrides=protocol)]
for measured in characterizations:
    for amplitude,times in measured.records['T'].items():
        voltage=measured.records['V'][amplitude]
        assert np.isfinite(times).all() and np.isfinite(voltage).all()
        assert len(times)==len(voltage)
        assert all(len(current)==len(times) and np.isfinite(current).all()
                   for current in measured.records['I'][amplitude].values())
from scp_pipeline.notebooks import run_fresh_simulation
from scp_pipeline.analysis import load_results
run=run_fresh_simulation(ui.pipeline_state, iclamp=True, stream_output=False,
    sim_overrides={{'save_plots':False,'save_output_metrics':False,
        'tstop':5,'iclamp':{{'delay_ms':1,'dur_ms':2,'tstop_ms':5,'record_currents':False}}}})
assert run.manifest_path.is_relative_to(Path('recordings').resolve())
result=load_results(run.manifest_path)
assert len(result['traces']['T'])==len(result['traces']['V'])
assert np.isfinite(result['traces']['V']).all()
assert result['meta']['conditions']=={{'v_init_mV':-65.0,'celsius_C':30.0}}
assert result['meta']['scp_software']['version']
assert Path('.scp_project.json').is_file()
'''
                self.subprocess(program, cwd=project)

    def test_editable_install_and_wheel_are_collision_free_and_source_only(self):
        environment = self.root / 'environment'
        created = subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(environment)],
                                 capture_output=True, text=True)
        self.assertEqual(created.returncode, 0, created.stderr)
        python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        installed = subprocess.run([str(python), '-m', 'pip', 'install', '-e', str(ROOT),
                                    '--no-deps', '--no-build-isolation', '--no-index'],
                                   cwd=self.root, capture_output=True, text=True)
        self.assertEqual(installed.returncode, 0, installed.stdout + installed.stderr)
        imported = subprocess.run([str(python), '-c', 'from scp_pipeline.simulation import SimulationSession; '
                                   'from importlib.metadata import distribution; '
                                   'assert distribution("scp-pipeline").version == "0.3.0"'],
                                  cwd=self.root, capture_output=True, text=True)
        self.assertEqual(imported.returncode, 0, imported.stdout + imported.stderr)
        wheels = self.root / 'wheels'
        built = subprocess.run([sys.executable, '-m', 'pip', 'wheel', str(ROOT), '--no-deps',
                                '--no-build-isolation', '--no-index', '--wheel-dir', str(wheels)],
                               cwd=self.root, capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        with zipfile.ZipFile(next(wheels.glob('*.whl'))) as wheel:
            names = wheel.namelist()
        self.assertIn('scp_pipeline/analysis/analysis_defaults.json', names)
        self.assertIn('scp_pipeline/cli/run.py', names)
        self.assertFalse(any(name.startswith(('modules/', 'cells/', 'projects/')) for name in names))
        self.assertFalse(any(name.endswith(('.so', '.dll', '.mod', '.hoc', '.nwb')) for name in names))
        # The same installation works when the dependency checkout is inside a project.
        nested = self.root / 'SCP'
        shutil.copytree(ROOT / 'modules', nested / 'modules', ignore=shutil.ignore_patterns('__pycache__'))
        for filename in ('setup.py', 'pyproject.toml', 'run_pipeline.py'):
            shutil.copy2(ROOT / filename, nested / filename)
        installed = subprocess.run([str(python), '-m', 'pip', 'install', '-e', str(nested),
                                    '--no-deps', '--no-build-isolation', '--no-index'],
                                   cwd=self.root, capture_output=True, text=True)
        self.assertEqual(installed.returncode, 0, installed.stdout + installed.stderr)
        imported = subprocess.run([str(python), '-c', f'from scp_pipeline.runtime import code_root; '
                                   f'assert code_root() == __import__("pathlib").Path({str(nested)!r})'],
                                  cwd=self.root, capture_output=True, text=True)
        self.assertEqual(imported.returncode, 0, imported.stdout + imported.stderr)

    def test_project_path_symlink_guard_is_not_resolved_away(self):
        from scp_pipeline.workspaces import resolve_workspace
        outside = self.root.parent / 'outside'
        outside.mkdir()
        (self.root / 'cells').mkdir()
        (self.root / 'cells/Escaped').symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            resolve_workspace(project_root=self.root, tune_dir_override='cells/Escaped/tunes/orig')

    def test_analysis_defaults_and_source_selection_remain_project_local(self):
        from scp_pipeline.projects import load_analysis_settings
        from scp_pipeline.loaders.source_context import same_source_selection
        self.register()
        defaults = load_analysis_settings(self.root)
        self.assertEqual(defaults['compare_list_paths'], [])
        override = self.root / 'analysis.json'
        override.write_text('{"plots_dpi":123}')
        save_project_settings(self.root, analysis_settings=override)
        self.assertEqual(load_analysis_settings(self.root)['plots_dpi'], 123)
        stored = json.loads((self.tune / 'cell_configs/cell_config.json').read_text())['model_source']
        self.assertTrue(same_source_selection(stored, self.config['model_source'], self.tune))
        changed = dict(stored, files=['Another.hoc'])
        self.assertFalse(same_source_selection(stored, changed, self.tune))
