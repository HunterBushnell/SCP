from __future__ import annotations

import json
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from neuron import h

from modules.model.load_cell import load_cell
from modules.model.geometry import define_geometry
from modules.model.synapses import preview_synapses
from modules.analysis.analysis import load_cell_and_geometry
from modules.simulation.cell_runtime import _resolve_recording_site
from modules.simulation.result_saving import save_results
from modules.simulation.session import SimulationOptions, SimulationSession
from modules.tuning.active import active_metric_rows, run_active_protocol
from modules.tuning.bmtool_synapse_adapter import (
    SynapseTuningSession,
    create_scp_synapse_tuner,
    prepare_scp_synapse_tuning,
)
from modules.tuning.passive import run_passive_protocol
from modules.tuning.passive_targets import resolve_passive_tuning_inputs
from scripts.restore_run_state import restore_run_state


FIXTURE_TUNE = Path(__file__).resolve().parent / "fixtures" / "hoc_template"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class GenericWorkflowTests(unittest.TestCase):
    def test_manual_passive_targets_do_not_implicitly_import_act(self) -> None:
        config = _read_json(FIXTURE_TUNE / "cell_configs" / "cell_config.json")
        cell = load_cell(config, base_dir=FIXTURE_TUNE)
        context = SimpleNamespace(
            repo_root=FIXTURE_TUNE.parents[2],
            tune_dir=FIXTURE_TUNE,
            cell_config=config,
        )
        manual = {
            "target_rin_mohm": 100.0,
            "target_tau_ms": 10.0,
            "target_v_rest_mv": -70.0,
        }
        with mock.patch(
            "modules.tuning.passive_targets.import_act_passive_module"
        ) as import_act:
            resolution = resolve_passive_tuning_inputs(
                context=context,
                cell=cell,
                manual_passive_targets=manual,
                use_target_config=False,
                target_source_mode="manual",
                compute_act_proposal=False,
            )

        import_act.assert_not_called()
        self.assertIsNone(resolution.act_passive_module)
        self.assertIsNone(resolution.settable_passive_properties)
        self.assertEqual(resolution.passive_targets["target_rin_mohm"], 100.0)

    def test_passive_and_active_protocols_apply_conditions(self) -> None:
        config = _read_json(FIXTURE_TUNE / "cell_configs" / "cell_config.json")
        conditions = _read_json(FIXTURE_TUNE / "cell_configs" / "sim_config.json")[
            "conditions"
        ]
        cell = load_cell(config, base_dir=FIXTURE_TUNE)
        sim_params = {
            "stim_amp": 0.0,
            "stim_delay": 10.0,
            "stim_dur": 20.0,
            "h_tstop": 40.0,
            "h_dt": 0.025,
            "conditions": conditions,
        }

        passive = run_passive_protocol(cell=cell, sim_params=sim_params, sim_amps=[-25.0])
        self.assertEqual(set(passive), {"T", "V", "F", "I"})
        self.assertGreater(len(passive["T"][-25.0]), 0)
        self.assertAlmostEqual(float(h.celsius), 32.0)

        active = run_active_protocol(cell=cell, sim_params=sim_params, sim_amps=[25.0])
        rows = active_metric_rows(
            looped_records=active,
            sim_params=sim_params,
            sim_amps=[25.0],
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["amp_pA"], 25.0)
        self.assertIn("spike_count", rows[0])

    def test_bmtool_handoff_receives_canonical_cell(self) -> None:
        with self.assertRaisesRegex(ValueError, "cell_name or tune_dir_override"):
            prepare_scp_synapse_tuning(
                repo_root=FIXTURE_TUNE.parents[2],
                resolve_bmtool=False,
            )

        with mock.patch(
            "modules.tuning.bmtool_synapse_adapter.ensure_bmtool_on_syspath"
        ) as ensure_bmtool:
            with self.assertWarnsRegex(RuntimeWarning, "deferred"):
                deferred = prepare_scp_synapse_tuning(
                    tune_dir_override=FIXTURE_TUNE,
                    repo_root=FIXTURE_TUNE.parents[2],
                    load_compiled_dll=False,
                    resolve_bmtool=True,
                )
        ensure_bmtool.assert_not_called()
        self.assertIsNone(deferred.bmtool_path)

        config = _read_json(FIXTURE_TUNE / "cell_configs" / "cell_config.json")
        cell = load_cell(config, base_dir=FIXTURE_TUNE)
        session = SynapseTuningSession(
            repo_root=FIXTURE_TUNE.parents[2],
            cell_name="synthetic_scoped_cell",
            tune_name="hoc_template",
            tune_dir=FIXTURE_TUNE,
            cell_config=config,
            cell=cell,
            mechanism_summary={"status": "skipped"},
            bmtool_path=None,
        )

        class FakeTuner:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        settings = {
            "test": {
                "spec_settings": {"level_of_detail": "ExpSyn", "sec_id": 0},
                "spec_syn_param": {},
            }
        }
        with mock.patch(
            "modules.tuning.bmtool_synapse_adapter.import_bmtool_synapse_api",
            return_value=(FakeTuner, object),
        ):
            tuner = create_scp_synapse_tuner(
                session,
                conn_type_settings=settings,
                connection="test",
            )
        self.assertIs(tuner.kwargs["hoc_cell"].source, cell)
        self.assertEqual(len(tuner.kwargs["hoc_cell"].all), 2)

        invalid = json.loads(json.dumps(settings))
        invalid["test"]["spec_settings"]["sec_id"] = 2
        with mock.patch(
            "modules.tuning.bmtool_synapse_adapter.import_bmtool_synapse_api",
            return_value=(FakeTuner, object),
        ) as import_api:
            with self.assertRaisesRegex(IndexError, "sec_id out of range"):
                create_scp_synapse_tuner(
                    session,
                    conn_type_settings=invalid,
                    connection="test",
                )
        import_api.assert_not_called()

        density_mechanism = json.loads(json.dumps(settings))
        density_mechanism["test"]["spec_settings"]["level_of_detail"] = "pas"
        with mock.patch(
            "modules.tuning.bmtool_synapse_adapter.import_bmtool_synapse_api",
            return_value=(FakeTuner, object),
        ) as import_api:
            with self.assertRaisesRegex(RuntimeError, "not a point-process constructor"):
                create_scp_synapse_tuner(
                    session,
                    conn_type_settings=density_mechanism,
                    connection="test",
                )
        import_api.assert_not_called()

        unavailable = json.loads(json.dumps(settings))
        unavailable["test"]["spec_settings"]["level_of_detail"] = "UnavailableFixtureSynapse"
        with mock.patch(
            "modules.tuning.bmtool_synapse_adapter.import_bmtool_synapse_api",
            return_value=(FakeTuner, object),
        ) as import_api:
            with self.assertRaisesRegex(RuntimeError, "add/compile"):
                create_scp_synapse_tuner(
                    session,
                    conn_type_settings=unavailable,
                    connection="test",
                )
        import_api.assert_not_called()

    def test_manifest_free_iclamp_save_and_model_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp) / "tune"
            shutil.copytree(FIXTURE_TUNE, tune)
            hoc_path = tune / "model" / "ScopedCell.hoc"
            hoc_path.write_text(
                hoc_path.read_text(encoding="utf-8").replace("ScopedCell", "ArchivedCell"),
                encoding="utf-8",
            )
            cell_config_path = tune / "cell_configs" / "cell_config.json"
            cell_config = _read_json(cell_config_path)
            cell_config["hoc_template"]["template_name"] = "ArchivedCell"
            cell_config["paths"]["modfiles"] = "modfiles"
            cell_config_path.write_text(json.dumps(cell_config, indent=2) + "\n", encoding="utf-8")
            mod_dir = tune / "modfiles"
            mod_dir.mkdir()
            mod_source = mod_dir / "Synthetic.mod"
            mod_source.write_text("NEURON { SUFFIX synthetic_fixture }\n", encoding="utf-8")

            options = SimulationOptions(
                iclamp=True,
                force_save=True,
                output_dir=Path(tmp) / "outputs",
                output_stem="generic_iclamp",
                load_mechanisms=False,
            )
            session = SimulationSession.from_tune(tune, options=options)
            result = session.run_and_save()
            self.assertEqual(result["mode"], "iclamp")
            self.assertGreater(len(result["traces"]["T"]), 0)
            self.assertAlmostEqual(float(h.celsius), 32.0)
            self.assertIsNotNone(session.saved_path)

            run_manifest = Path(session.saved_path)
            manifest = _read_json(run_manifest)
            artifact_manifest = run_manifest.parent / manifest["files"]["model_artifacts"]
            archived = _read_json(artifact_manifest)
            archived_targets = {
                entry.get("target_relative_path") for entry in archived["artifacts"]
            }
            self.assertIn("model/ScopedCell.hoc", archived_targets)
            self.assertIn("modfiles/Synthetic.mod", archived_targets)
            for entry in archived["artifacts"]:
                archived_file = artifact_manifest.parent / entry["archive_path"]
                self.assertEqual(
                    entry["sha256"], hashlib.sha256(archived_file.read_bytes()).hexdigest()
                )

            original_hoc = hoc_path.read_bytes()
            original_mod = mod_source.read_bytes()
            hoc_path.write_text("// changed after run\n", encoding="utf-8")
            mod_source.write_text("// changed after run\n", encoding="utf-8")

            preview = restore_run_state(
                from_run=run_manifest,
                to_tune=tune,
                apply=["model_artifacts"],
                dry_run=True,
            )
            self.assertEqual(preview.changed_files, 2)
            self.assertEqual(hoc_path.read_text(encoding="utf-8"), "// changed after run\n")

            applied = restore_run_state(
                from_run=run_manifest,
                to_tune=tune,
                apply=["model_artifacts"],
                dry_run=False,
            )
            self.assertEqual(applied.changed_files, 2)
            self.assertEqual(hoc_path.read_bytes(), original_hoc)
            self.assertEqual(mod_source.read_bytes(), original_mod)
            self.assertTrue(any("Recompile" in warning for warning in applied.warnings))
            self.assertTrue(list(hoc_path.parent.glob("ScopedCell.hoc.bak_*")))
            self.assertTrue(list(mod_source.parent.glob("Synthetic.mod.bak_*")))

    def test_runtime_conditions_recording_and_empty_synapse_target(self) -> None:
        config = _read_json(FIXTURE_TUNE / "cell_configs" / "cell_config.json")
        cell = load_cell(config, base_dir=FIXTURE_TUNE)
        with self.assertRaisesRegex(ValueError, "conditions.celsius_C"):
            run_passive_protocol(
                cell=cell,
                sim_params={
                    "stim_delay": 1.0,
                    "stim_dur": 1.0,
                    "h_tstop": 3.0,
                    "h_dt": 0.025,
                    "conditions": {"v_init_mV": -68.0},
                },
                sim_amps=[0.0],
            )

        segment, label = _resolve_recording_site(
            cell, {"sec": "dend", "idx": 0, "x": 0.5}
        )
        self.assertEqual(segment.sec.name(), cell.dend[0].name())
        self.assertEqual(label, "dend[0](0.500)")

        geometry = define_geometry(cell, {"thresholds_um": {"distal": {"low": 1000}}})
        group_inputs = mock.Mock(spike_trains=[[1.0]], mode="homogeneous", meta={})
        groups = {
            "empty_target": {
                "state": True,
                "syns": {
                    "type": "ExpSyn",
                    "N_syn_resolved": 1,
                    "segs": "distal",
                    "params": {},
                },
            }
        }
        with self.assertRaisesRegex(ValueError, "geometry group 'distal' is empty"):
            preview_synapses(cell, geometry, {}, groups, {"empty_target": group_inputs})

    def test_step6_reload_and_full_results_keep_model_artifacts(self) -> None:
        cell, geometry, _ = load_cell_and_geometry(FIXTURE_TUNE)
        self.assertEqual(len(cell.all), 2)
        self.assertEqual(geometry["meta"]["counts"]["all_secs"], 2)

        with tempfile.TemporaryDirectory() as tmp:
            results = {
                "mode": "single",
                "sim_cfg": {
                    "tune_dir": str(FIXTURE_TUNE),
                    "cell": "synthetic_scoped_cell",
                    "tune": "hoc_template",
                    "output": "full_only",
                    "save_output": True,
                    "save_full_results": True,
                    "save_sidecars": False,
                    "output_format": "pickle",
                },
                "meta": {},
                "traces": {"T": [0.0], "V": [-68.0]},
            }
            saved = save_results(results, base_dir=Path(tmp))
            run_dir = Path(saved).parent
            manifest = _read_json(run_dir / "run_manifest.json")
            self.assertIn("results_pkl", manifest["files"])
            self.assertIn("model_artifacts", manifest["files"])

    def test_restore_rejects_tampering_and_paths_outside_tune(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tune = root / "tune"
            configs = tune / "cell_configs"
            configs.mkdir(parents=True)
            (configs / "cell_config.json").write_text(
                json.dumps({"cell_loader": "hoc_template"}), encoding="utf-8"
            )
            target = tune / "model" / "Cell.hoc"
            target.parent.mkdir()
            target.write_text("original", encoding="utf-8")
            outside = root / "outside.txt"
            outside.write_text("outside", encoding="utf-8")

            run = root / "run"
            archive_root = run / "model_artifacts"
            payload = archive_root / "files" / "Cell.hoc"
            payload.parent.mkdir(parents=True)
            payload.write_text("tampered", encoding="utf-8")
            artifact_manifest = {
                "format_version": 1,
                "loader": "hoc_template",
                "errors": [],
                "artifacts": [
                    {
                        "kind": "hoc_template",
                        "archive_path": "files/Cell.hoc",
                        "target_relative_path": "model/Cell.hoc",
                        "sha256": "0" * 64,
                        "restorable": True,
                    },
                    {
                        "kind": "hoc_template",
                        "archive_path": "files/Cell.hoc",
                        "target_relative_path": "../outside.txt",
                        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                        "restorable": True,
                    },
                ],
            }
            (archive_root / "manifest.json").write_text(
                json.dumps(artifact_manifest), encoding="utf-8"
            )
            (run / "run_manifest.json").write_text(
                json.dumps({"files": {"model_artifacts": "model_artifacts/manifest.json"}}),
                encoding="utf-8",
            )

            report = restore_run_state(
                from_run=run,
                to_tune=tune,
                apply=["model_artifacts"],
                dry_run=False,
            )
            self.assertGreaterEqual(len(report.errors), 2)
            self.assertEqual(target.read_text(encoding="utf-8"), "original")
            self.assertEqual(outside.read_text(encoding="utf-8"), "outside")


if __name__ == "__main__":
    unittest.main()
