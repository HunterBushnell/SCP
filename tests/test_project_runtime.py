from __future__ import annotations

import ast
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

import numpy as np
from neuron import h

from modules.model import wrap_cell
from modules.setup import create_working_copy
from modules.setup.step1_prepare import prepare_tune
from modules.setup.scaffold import scaffold_base_configs
from modules.simulation import SimulationOptions, SimulationSession
from modules.simulation.result_loading import load_results
from modules.tuning import run_active_protocol, run_passive_protocol
from modules.tuning.notebook_setup import prepare_tuning_notebook_context, build_tuning_cell
from modules.workspaces import WorkspaceMismatchError, infer_workspace


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
CONDITIONS = {"v_init_mV": -70.0, "celsius_C": 34.0}
SIM = {
    "conditions": CONDITIONS,
    "dt": 0.025,
    "tstart": 0.0,
    "tstop": 20.0,
    "plots_profile": "off",
    "save_output_metrics": False,
    "iclamp": {"enabled": True, "amp_nA": 0.025, "delay_ms": 5.0, "dur_ms": 10.0,
               "tstop_ms": 20.0, "dt_ms": 0.025, "record_currents": False},
}
SENTINEL = "SCP_PROJECT_TEST_RESULT="


def tree_hashes(directory: Path) -> dict[str, str]:
    return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


class ProjectRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="scp-project-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "SCP"
        self.root.mkdir()
        (self.root / "modules").symlink_to(ROOT / "modules", target_is_directory=True)
        (self.root / "scripts").mkdir()
        for source in ("run_pipeline.py", "run_slurm.sh", "scripts/step1_prepare.py", "scripts/check_setup.py"):
            shutil.copy2(ROOT / source, self.root / source)
        legacy = self.root / "cells/PV/tunes/tuned/model"
        legacy.mkdir(parents=True)
        (legacy / "untouched.hoc").write_bytes(b"// Legacy assets must remain untouched.\r\n")
        self.legacy_hashes = tree_hashes(self.root / "cells")
        self.env_patch = mock.patch.dict(os.environ, {"SCP_ROOT": str(self.root), "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg"})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        old_path = sys.path[:]
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), old_path))

    def tearDown(self) -> None:
        self.assertEqual(tree_hashes(self.root / "cells"), self.legacy_hashes,
                         "Project operations changed legacy cell assets")

    def tune(self, project="study_a", tune_name="tuned") -> Path:
        return self.root / "projects" / project / "cells/ExampleCell/tunes" / tune_name

    def prepare_factory(self, project="study_a", tune_name="tuned", *, kwargs=None) -> Path:
        tune = self.tune(project, tune_name)
        (tune / "model").mkdir(parents=True)
        shutil.copy2(FIXTURES / "custom_models/standalone.py", tune / "model/cell.py")
        with redirect_stdout(io.StringIO()):
            prepare_tune(tune_dir=tune, cell_name="ExampleCell", tune_name=tune_name, project_name=project,
                         cell_loader="python_factory", source_type="existing",
                         loader_paths={"python_factory": "model/cell.py", "modfiles": None},
                         loader_config={"python_factory": {"callable": "build_cell", "args": [], "kwargs": kwargs or {}}},
                         v_init_mV=-70.0, celsius_C=34.0,
                         do_synapse_configs=False, target_source_mode="none",
                         do_compile_modfiles=False, load_compiled_dll=False, do_validate=False)
        return tune

    def prepare_hoc(self, project="study_a") -> Path:
        tune = self.tune(project)
        shutil.copytree(FIXTURES / "hoc_template", tune)
        config_path = tune / "cell_configs/cell_config.json"
        config = json.loads(config_path.read_text())
        config.update(cell_name="ExampleCell", tune="tuned", project_name=project)
        config_path.write_text(json.dumps(config))
        return tune

    def subprocess(self, *arguments, expected=0, env=None) -> subprocess.CompletedProcess:
        result = subprocess.run([sys.executable, *map(str, arguments)], cwd=self.root,
                                capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(result.returncode, expected, result.stdout + "\n" + result.stderr)
        return result

    def test_project_scaffold_and_sibling_copy_refresh_only_workspace_metadata(self) -> None:
        orig = self.prepare_factory(tune_name="orig")
        cfg = json.loads((orig / "cell_configs/cell_config.json").read_text())
        self.assertEqual(cfg["project_name"], "study_a")
        self.assertEqual(cfg["tune"], "orig")
        self.assertIsNone(cfg["paths"]["modfiles"])
        self.assertFalse((orig / "cell_configs/syn_config.json").exists())
        cfg["paths"]["tune_dir"] = str(orig)
        cfg["workspace_identity"] = {"project_name": "study_a", "cell_name": "ExampleCell", "tune_name": "orig"}
        (orig / "cell_configs/cell_config.json").write_text(json.dumps(cfg))
        sim_path = orig / "cell_configs/sim_config.json"
        sim = json.loads(sim_path.read_text())
        sim.update(tune_dir=str(orig), cell="ExampleCell", tune="orig", project_name="study_a",
                   workspace_identity=cfg["workspace_identity"])
        sim_path.write_text(json.dumps(sim))
        before = tree_hashes(orig)
        copied = create_working_copy(source_tune_dir=orig)
        tuned = Path(copied["target"])
        self.assertEqual(tuned, self.tune())
        new_cfg = json.loads((tuned / "cell_configs/cell_config.json").read_text())
        new_sim = json.loads((tuned / "cell_configs/sim_config.json").read_text())
        self.assertEqual(new_cfg["project_name"], "study_a")
        self.assertEqual(new_cfg["tune"], "tuned")
        self.assertEqual(new_sim["tune"], "tuned")
        self.assertEqual(new_sim["workspace_identity"]["tune_name"], "tuned")
        self.assertEqual(new_sim["tune_dir"], str(tuned))
        self.assertNotEqual(new_cfg["paths"].get("tune_dir"), str(orig))
        self.assertEqual((tuned / "model/cell.py").read_bytes(), (orig / "model/cell.py").read_bytes())
        self.assertEqual(tree_hashes(orig), before)

    def test_configured_project_reading_infers_optional_metadata_without_writes(self) -> None:
        tune = self.prepare_factory()
        config_path = tune / "cell_configs/cell_config.json"
        config = json.loads(config_path.read_text())
        config.pop("project_name", None)
        config_path.write_text(json.dumps(config))
        before = tree_hashes(tune)
        with redirect_stdout(io.StringIO()):
            session = SimulationSession.from_tune(tune)
            context = prepare_tuning_notebook_context(project_name="study_a", cell_name="ExampleCell", tune_name="tuned",
                repo_root=self.root, require_compiled_modfiles=False, print_summary=False)
        self.assertEqual(session.workspace.identity, context.workspace_identity)
        self.assertEqual(context.project_name, "study_a")
        self.assertEqual(tree_hashes(tune), before)

    def test_complete_metadata_less_project_scaffold_stays_byte_identical(self) -> None:
        tune = self.prepare_factory()
        config_path = tune / "cell_configs/cell_config.json"
        config = json.loads(config_path.read_text())
        config.pop("project_name", None)
        config_path.write_bytes((json.dumps(config, indent=5) + "\n").replace("\n", "\r\n").encode())
        before = tree_hashes(tune)
        with redirect_stdout(io.StringIO()):
            scaffold_base_configs(tune_dir=tune, cell_name="ExampleCell", tune_name="tuned", project_name="study_a",
                specimen_id=None, model_type="perisomatic", cell_loader="python_factory", config_mode="fill",
                loader_paths=config["paths"], loader_config={"python_factory": config["python_factory"]})
        self.assertEqual(tree_hashes(tune), before)

    def save_synaptic_baseline(self, tune: Path) -> Path:
        # Empty inputs exercise the ordinary single-trial path without custom MODs.
        (tune / "cell_configs/syn_config.json").write_text("{}\n")
        overrides = deepcopy(SIM)
        overrides["iclamp"]["enabled"] = False
        with redirect_stdout(io.StringIO()):
            session = SimulationSession.from_tune(tune, options=SimulationOptions(mode="single", n_trials=1,
                force_save=True, output_stem="append_base", sim_overrides=overrides))
            session.run()
            saved = session.save()
        self.assertIsNotNone(saved)
        return Path(saved).parent

    def test_same_identity_append_rebases_saved_location_and_uses_current_recipe(self) -> None:
        original = self.prepare_factory()
        baseline = self.save_synaptic_baseline(original)
        current = self.root.parent / "RelocatedSCP/projects/study_a/cells/ExampleCell/tunes/tuned"
        shutil.copytree(original, current, ignore=shutil.ignore_patterns("output_data", ".scp_build"))
        config_path = current / "cell_configs/cell_config.json"
        config = json.loads(config_path.read_text())
        config["python_factory"]["kwargs"] = {"size": 33.0}
        config_path.write_text(json.dumps(config))
        sim_path = current / "cell_configs/sim_config.json"
        sim = json.loads(sim_path.read_text())
        sim.update(tune_dir=str(original), append={"enabled": True, "path": str(baseline)})
        sim_path.write_text(json.dumps(sim))
        before = tree_hashes(current)
        with redirect_stdout(io.StringIO()):
            session = SimulationSession.from_tune(current, options=SimulationOptions(mode="single", n_trials=1,
                force_save=True, output_stem="relocated_append"))
            result = session.run()
            saved = session.save()
        self.assertEqual(session.cell.soma[0].L, 33.0, "Append must construct the current recipe, not its source run")
        self.assertEqual(result["sim_cfg"]["tune_dir"], str(current))
        self.assertEqual(result["sim_cfg"]["workspace_identity"], infer_workspace(current).identity)
        self.assertEqual(result["meta"]["append_lineage"][-1]["tune_dir"], str(original))
        paths = [entry["path"] for entry in result["meta"]["model_provenance"]["inputs"]]
        self.assertTrue(paths)
        self.assertTrue(all(Path(path).is_relative_to(current) for path in paths))
        self.assertTrue(Path(saved).is_relative_to(current / "output_data"))
        run = Path(saved).parent
        manifest = json.loads((run / "model_artifacts/manifest.json").read_text())
        self.assertEqual(manifest["workspace_identity"], infer_workspace(current).identity)
        self.assertTrue(all(Path(entry["path"]).is_relative_to(current)
                            for entry in manifest["construction_inputs"]))
        after = {path: digest for path, digest in tree_hashes(current).items() if not path.startswith("output_data/")}
        self.assertEqual(after, before, "Preparing/appending must not rewrite current configs or native sources")

    def test_cross_project_session_append_rejects_before_cell_construction_or_save(self) -> None:
        baseline = self.save_synaptic_baseline(self.prepare_factory("study_a"))
        target = self.prepare_factory("study_b")
        sim_path = target / "cell_configs/sim_config.json"
        config = json.loads(sim_path.read_text())
        config["append"] = {"enabled": True, "path": str(baseline)}
        sim_path.write_text(json.dumps(config))
        source_before, target_before = tree_hashes(baseline), tree_hashes(target)
        with redirect_stdout(io.StringIO()), mock.patch("modules.simulation.session.load_cell") as construct:
            session = SimulationSession.from_tune(target, options=SimulationOptions(force_save=True))
            with self.assertRaisesRegex(WorkspaceMismatchError, "different workspaces"):
                session.prepare()
            construct.assert_not_called()
        self.assertFalse((target / "output_data").exists())
        self.assertEqual(tree_hashes(target), target_before)
        self.assertEqual(tree_hashes(baseline), source_before)

    def test_two_projects_python_factories_run_and_save_independently(self) -> None:
        for project, size in (("study_a", 20.0), ("study_b", 30.0)):
            with self.subTest(project=project):
                tune = self.prepare_factory(project, kwargs={"size": size})
                with redirect_stdout(io.StringIO()):
                    session = SimulationSession.from_tune(tune, options=SimulationOptions(
                        iclamp=True, force_save=True, output_stem="same_run", sim_overrides=deepcopy(SIM)))
                    result = session.run()
                    saved = session.save()
                self.assertEqual(session.cell.soma[0].L, size)
                self.assertEqual(len(result["traces"]["T"]), 801)
                self.assertTrue(np.isfinite(result["traces"]["V"]).all())
                self.assertEqual(result["traces"]["V"][0], -70.0)
                self.assertEqual(result["sim_cfg"]["workspace_identity"], infer_workspace(tune).identity)
                self.assertEqual(result["meta"]["workspace_identity"]["project_name"], project)
                self.assertTrue(Path(saved).is_relative_to(tune / "output_data"))
                loaded = load_results(Path(saved).parent)
                self.assertEqual(loaded["sim_cfg"]["workspace_identity"]["project_name"], project)
                self.assertEqual(json.loads((Path(saved).parent / "run_manifest.json").read_text())["workspace_identity"]["project_name"], project)

    def test_selected_workspace_replaces_stale_simulation_routing_in_memory(self) -> None:
        tune = self.prepare_factory()
        sim_path = tune / "cell_configs/sim_config.json"
        config = json.loads(sim_path.read_text())
        config.update(cell="PV", tune="orig", project_name="wrong", tune_dir="/old/SCP/cells/PV/tunes/orig")
        sim_path.write_text(json.dumps(config))
        before = sim_path.read_bytes()
        with redirect_stdout(io.StringIO()):
            session = SimulationSession.from_tune(tune, options=SimulationOptions(iclamp=True, sim_overrides=deepcopy(SIM)))
            result = session.run()
        self.assertEqual(result["sim_cfg"]["cell"], "ExampleCell")
        self.assertEqual(result["sim_cfg"]["tune"], "tuned")
        self.assertEqual(result["sim_cfg"]["project_name"], "study_a")
        self.assertEqual(result["sim_cfg"]["tune_dir"], str(tune))
        self.assertEqual(sim_path.read_bytes(), before)

    def test_custom_tune_parent_keeps_project_identity_without_persisted_namespace(self) -> None:
        ordinary = self.prepare_factory()
        tune = ordinary.parent.parent / "variants/tuned"
        tune.parent.mkdir()
        shutil.move(str(ordinary), str(tune))
        config_path = tune / "cell_configs/cell_config.json"
        config = json.loads(config_path.read_text())
        config.pop("project_name", None)
        config_path.write_text(json.dumps(config))
        before = tree_hashes(tune)
        with redirect_stdout(io.StringIO()):
            context = prepare_tuning_notebook_context(project_name="study_a", cell_name="ExampleCell", tune_name="tuned",
                tunes_parent="variants", repo_root=self.root, require_compiled_modfiles=False, print_summary=False)
            session = SimulationSession.from_tune(tune, options=SimulationOptions(iclamp=True, sim_overrides=deepcopy(SIM)))
            result = session.run()
        expected = {"project_name": "study_a", "cell_name": "ExampleCell", "tune_name": "tuned"}
        self.assertEqual(context.workspace_identity, expected)
        self.assertEqual(session.workspace.identity, expected)
        self.assertEqual(result["sim_cfg"]["workspace_identity"], expected)
        self.assertEqual(tree_hashes(tune), before)

    def test_direct_session_and_cli_reject_escaping_managed_symlink_before_load(self) -> None:
        managed = self.prepare_factory()
        outside = self.root.parent / "Outside/ExampleCell/tunes/tuned"
        outside.parent.mkdir(parents=True)
        shutil.move(str(managed), str(outside))
        managed.symlink_to(outside, target_is_directory=True)
        before = tree_hashes(outside)
        with mock.patch("modules.simulation.session.load_cell") as constructor:
            with self.assertRaisesRegex(ValueError, "escapes"):
                SimulationSession.from_tune(managed)
            constructor.assert_not_called()
        completed = self.subprocess(self.root / "run_pipeline.py", "--tune-dir", managed, "--iclamp", expected=1)
        self.assertIn("escapes", completed.stdout + completed.stderr)
        self.assertEqual(tree_hashes(outside), before)
        self.assertFalse((outside / "output_data").exists())

    def test_live_project_workspace_adopts_exact_cell_and_protocols_remain_scoped(self) -> None:
        soma = h.Section(name="project_live_body")
        soma.L = soma.diam = 20.0
        soma.insert("pas")
        soma.g_pas, soma.e_pas = 0.0001, -70.0
        unrelated = h.Section(name="project_unrelated")
        cell = wrap_cell(soma, cell_name="ExampleCell")
        native_before = (float(soma.L), float(soma.diam), int(soma.nseg), float(soma.g_pas))
        with redirect_stdout(io.StringIO()), mock.patch("modules.simulation.session.load_cell", side_effect=AssertionError("live cell must not be rebuilt")):
            context = prepare_tuning_notebook_context(project_name="study_a", cell_name="ExampleCell", tune_name="tuned",
                repo_root=self.root, cell_override=cell, runtime_sim_config=deepcopy(SIM), print_summary=False)
            adopted = build_tuning_cell(context, cell_override=cell)
            session = SimulationSession.from_cell(adopted, tune_dir=self.tune(), sim_config=deepcopy(SIM),
                options=SimulationOptions(iclamp=True, force_save=True, output_stem="live"))
            result = session.run()
            saved = session.save()
        self.assertIs(session.cell, cell)
        self.assertIs(adopted, cell)
        self.assertEqual(session.cell.all, (soma,))
        self.assertNotIn(unrelated, session.cell.all)
        self.assertEqual((float(soma.L), float(soma.diam), int(soma.nseg), float(soma.g_pas)), native_before)
        self.assertEqual(result["sim_cfg"]["workspace_identity"]["project_name"], "study_a")
        self.assertTrue(result["meta"]["model_provenance"]["session_only"])
        self.assertIsNotNone(saved)
        protocol = {"stim_amp": .025, "stim_delay": 5., "stim_dur": 10., "h_tstop": 20., "h_dt": .025, "conditions": CONDITIONS}
        for runner, amps in ((run_passive_protocol, [-25.]), (run_active_protocol, [0., 25.])):
            with redirect_stdout(io.StringIO()):
                protocol_result = runner(cell=cell, sim_params=protocol, sim_amps=amps)
            for amp in amps:
                self.assertEqual(len(protocol_result["T"][amp]), 801)
                self.assertTrue(np.isfinite(protocol_result["V"][amp]).all())

    def test_project_factory_cli_and_explicit_path_precedence_in_fresh_processes(self) -> None:
        first = self.prepare_factory("study_a")
        second = self.prepare_factory("study_b")
        arguments = ["--iclamp", "--n-trials", "1", "--force-save", "--sim-overrides-json", json.dumps(SIM)]
        self.subprocess(self.root / "run_pipeline.py", "--project", "study_a", "--cell", "ExampleCell", "--tune", "tuned",
                        "--output-stem", "cli_project", *arguments)
        self.subprocess(self.root / "run_pipeline.py", "--project", "study_a", "--cell", "PV", "--tune", "orig",
                        "--tune-dir", second.relative_to(self.root), "--output-stem", "cli_explicit", *arguments)
        for tune, stem, project in ((first, "cli_project", "study_a"), (second, "cli_explicit", "study_b")):
            run = tune / "output_data" / stem
            saved_sim = json.loads((run / "sim_cfg.json").read_text())
            self.assertEqual(saved_sim["workspace_identity"]["project_name"], project)
            self.assertEqual(saved_sim["cell"], "ExampleCell")
            self.assertTrue(np.isfinite(load_results(run)["traces"]["V"]).all())

    def test_copied_identical_hoc_definitions_require_fresh_processes_between_projects(self) -> None:
        first, second = self.prepare_hoc("study_a"), self.prepare_hoc("study_b")
        program = textwrap.dedent(f"""
            import json, sys
            import numpy as np
            from modules.simulation import SimulationOptions, SimulationSession
            session = SimulationSession.from_tune(sys.argv[1], options=SimulationOptions(iclamp=True, sim_overrides={SIM!r}))
            result = session.run()
            assert len(session.cell.all) == 2 and len(session.cell.soma) == 1
            assert len(result['traces']['T']) == 801
            assert np.isfinite(result['traces']['V']).all()
            print({SENTINEL!r} + json.dumps(result['sim_cfg']['workspace_identity']))
        """)
        for project, tune in (("study_a", first), ("study_b", second)):
            completed = self.subprocess("-c", program, tune)
            payload = next(line[len(SENTINEL):] for line in completed.stdout.splitlines() if line.startswith(SENTINEL))
            self.assertEqual(json.loads(payload)["project_name"], project)
        conflict = textwrap.dedent("""
            import json, sys
            from pathlib import Path
            from modules.model import load_cell
            first, second = map(Path, sys.argv[1:])
            a = load_cell(json.loads((first/'cell_configs/cell_config.json').read_text()), base_dir=first)
            try:
                load_cell(json.loads((second/'cell_configs/cell_config.json').read_text()), base_dir=second)
            except RuntimeError as exc:
                assert 'restart' in str(exc).lower()
            else:
                raise AssertionError('Project namespaces must not silently redefine a HOC template')
        """)
        self.subprocess("-c", conflict, first, second)

    def test_setup_cli_and_setup_checker_select_project_without_touching_legacy_assets(self) -> None:
        tune = self.tune("study_a", "orig")
        (tune / "model").mkdir(parents=True)
        shutil.copy2(FIXTURES / "custom_models/standalone.py", tune / "model/cell.py")
        self.subprocess(self.root / "scripts/step1_prepare.py", "--project", "study_a", "--cell", "ExampleCell", "--tune", "orig",
            "--cell-loader", "python_factory", "--source-type", "existing", "--python-factory-file", "model/cell.py",
            "--python-factory-callable", "build_cell", "--v-init-mv", "-70", "--celsius-c", "34",
            "--target-source-mode", "none", "--no-synapse-configs", "--no-compile")
        cfg = json.loads((tune / "cell_configs/cell_config.json").read_text())
        self.assertEqual(cfg["project_name"], "study_a")
        sim_path = tune / "cell_configs/sim_config.json"
        sim = json.loads(sim_path.read_text())
        sim["iclamp"]["enabled"] = True
        sim_path.write_text(json.dumps(sim))
        before = tree_hashes(tune)
        checked = self.subprocess(self.root / "scripts/check_setup.py", "--repo-root", self.root,
                                 "--project", "study_a", "--cell", "ExampleCell", "--tune", "orig", "--steps", "2", "3", "5")
        self.assertIn("errors:   0", checked.stdout)
        self.assertIn(str(tune), checked.stdout)
        self.assertEqual(tree_hashes(tune), before)

    def test_slurm_project_shorthand_and_explicit_override_with_dry_cli_stub(self) -> None:
        first, second = self.prepare_factory("study_a"), self.prepare_factory("study_b")
        capture = self.root.parent / "slurm_arguments.json"
        stub = self.root.parent / "python_stub"
        stub.write_text(f"#!{sys.executable}\n" + textwrap.dedent("""
            import json, os, sys
            if len(sys.argv) > 1 and sys.argv[1].endswith('run_pipeline.py'):
                with open(os.environ['SCP_CAPTURE_ARGS'], 'w') as handle:
                    json.dump(sys.argv[1:], handle)
                raise SystemExit(0)
            os.execv(os.environ['SCP_REAL_PYTHON'], [os.environ['SCP_REAL_PYTHON'], *sys.argv[1:]])
        """))
        stub.chmod(0o755)
        env = dict(os.environ, REPO_ROOT=str(self.root), PYTHON_BIN=str(stub), SCP_REAL_PYTHON=sys.executable,
                   SCP_CAPTURE_ARGS=str(capture), PROJECT="study_a", CELL="ExampleCell", TUNE="tuned",
                   ROTATE_LOGS="0", CLEAN_STATUS="0", N_TRIALS="1", ICLAMP="1", MERGE_ARRAY="0")
        for key in list(env):
            if key.startswith("SLURM_") or key in {"TUNE_DIR", "OUTPUT_DIR", "OUTPUT_STEM"}:
                env.pop(key)
        env["SLURM_JOB_ID"] = "123"
        for explicit, expected in ((None, first), (str(second), second)):
            invocation_env = dict(env)
            if explicit:
                invocation_env["TUNE_DIR"] = explicit
            result = subprocess.run(["bash", str(self.root / "run_slurm.sh")], cwd=self.root,
                                    env=invocation_env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            arguments = json.loads(capture.read_text())
            self.assertEqual(arguments[arguments.index("--tune-dir") + 1], str(expected))
            self.assertEqual(arguments[arguments.index("--output-dir") + 1], str(expected / "output_data"))
            self.assertEqual(list((expected / "output_data").rglob("run_manifest.json")), [],
                             "The wrapper may copy logs, but the dry CLI stub must not save results")
            self.assertEqual(list((expected / "output_data").rglob("traces.npz")), [])


class ProjectNotebookContractTests(unittest.TestCase):
    def test_primary_notebook_defaults_and_project_wiring(self) -> None:
        for name in ("0_pipeline.ipynb", "1_setup.ipynb", "2_passive.ipynb", "3_active.ipynb", "4_synapses.ipynb", "5_simulate.ipynb", "6_analysis.ipynb"):
            with self.subTest(notebook=name):
                notebook = json.loads((ROOT / name).read_text())
                sources = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"]
                code = "\n".join(sources)
                self.assertTrue("project_name" in code, f"{name} does not expose project selection yet")
                if name == "0_pipeline.ipynb":
                    source = next(source for source in sources if "pipeline_settings =" in source)
                    assignment = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Assign)
                                      and any(isinstance(target, ast.Name) and target.id == "pipeline_settings" for target in node.targets))
                    values = {ast.literal_eval(key): value for key, value in zip(assignment.value.keys, assignment.value.values)}
                    self.assertIsNone(ast.literal_eval(values["project_name"]))
                    self.assertIsNone(ast.literal_eval(values["cell_name"]))
                    self.assertIsNone(ast.literal_eval(values["tune_name"]))
                else:
                    trees = [ast.parse(source) for source in sources if not source.lstrip().startswith(("%", "!"))]
                    assignment = next(node for tree in trees for node in ast.walk(tree) if isinstance(node, ast.Assign)
                                      and any(isinstance(target, ast.Name) and target.id == "project_name" for target in node.targets))
                    self.assertIsNone(ast.literal_eval(assignment.value))
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        self.assertIsNone(cell["execution_count"])
                        self.assertEqual(cell["outputs"], [])

    def test_slurm_shell_syntax_and_legacy_sst_defaults_remain(self) -> None:
        script = (ROOT / "run_slurm.sh").read_text()
        self.assertIn("CELL=${CELL:-SST}", script)
        self.assertIn("PROJECT", script)
        result = subprocess.run(["bash", "-n", str(ROOT / "run_slurm.sh")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
