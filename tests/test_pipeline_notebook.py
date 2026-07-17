from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import matplotlib
import numpy as np

matplotlib.use("Agg")

from modules.notebooks.pipeline_workflow import (
    RestartKernelRequired,
    _model_source_fingerprint,
    _select_setup_source,
    prepare_interactive_synapse_tuner,
    prepare_pipeline_notebook,
    run_active_stage,
    run_fresh_simulation,
    run_passive_stage,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_TUNE = REPO_ROOT / "tests" / "fixtures" / "hoc_template"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _copy_unique_hoc_tune(parent: Path) -> Path:
    tune = parent / "cells" / "synthetic" / "tunes" / "tuned"
    shutil.copytree(FIXTURE_TUNE, tune)
    suffix = uuid.uuid4().hex[:10]
    template_name = f"PipelineCell_{suffix}"
    hoc_path = tune / "model" / "ScopedCell.hoc"
    hoc_path.write_text(
        hoc_path.read_text(encoding="utf-8").replace("ScopedCell", template_name),
        encoding="utf-8",
    )
    config_path = tune / "cell_configs" / "cell_config.json"
    config = _read_json(config_path)
    config["hoc_template"]["template_name"] = template_name
    _write_json(config_path, config)
    return tune


class PipelineNotebookWorkflowTests(unittest.TestCase):
    def test_existing_allen_fill_preserves_loader_metadata_and_user_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp) / "cells" / "PV" / "tunes" / "custom"
            config_dir = tune / "cell_configs"
            config_dir.mkdir(parents=True)
            (tune / "manifest.json").write_text("{}\n", encoding="utf-8")
            config_path = config_dir / "cell_config.json"
            before = {
                "cell_name": "PV",
                "tune": "custom",
                "cell_loader": "allen_manifest",
                "paths": {"manifest": "manifest.json", "modfiles": None},
                "color": "#123456",
                "tuning": {
                    "soma_diam_multiplier": 3.75,
                    "user_note": "preserve me",
                },
                "custom_user_value": {"enabled": True},
            }
            _write_json(config_path, before)
            fake_context = SimpleNamespace(
                cell_name="PV",
                tune_name="custom",
                cell_config=before,
                sim_config={},
            )

            with (
                mock.patch(
                    "modules.tuning.prepare_tuning_notebook_context",
                    return_value=fake_context,
                ),
                mock.patch("modules.tuning.build_tuning_cell", return_value=object()),
            ):
                prepare_pipeline_notebook(
                    repo_root=REPO_ROOT,
                    cell_name="PV",
                    tune_name="custom",
                    tune_dir_override=tune,
                )

            after = _read_json(config_path)
            self.assertEqual(after["cell_loader"], before["cell_loader"])
            self.assertEqual(after["paths"], before["paths"])
            self.assertEqual(after["color"], before["color"])
            self.assertEqual(after["tuning"], before["tuning"])
            self.assertEqual(after["custom_user_value"], before["custom_user_value"])

    def test_existing_hoc_fill_preserves_loader_metadata_and_builds_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = _copy_unique_hoc_tune(Path(tmp))
            config_path = tune / "cell_configs" / "cell_config.json"
            before = _read_json(config_path)

            from modules.notebooks import helpers

            original_builder = helpers.build_cell_for_notebook
            with mock.patch.object(
                helpers,
                "build_cell_for_notebook",
                wraps=original_builder,
            ) as build_cell:
                state = prepare_pipeline_notebook(
                    repo_root=REPO_ROOT,
                    cell_name="synthetic",
                    tune_name="tuned",
                    tune_dir_override=tune,
                )

            self.assertEqual(build_cell.call_count, 1)
            after = _read_json(config_path)
            self.assertEqual(after["cell_loader"], before["cell_loader"])
            self.assertEqual(after["paths"], before["paths"])
            self.assertEqual(after["hoc_template"], before["hoc_template"])
            self.assertTrue((tune / "cell_configs" / "target_config.json").is_file())
            self.assertTrue((tune / "cell_configs" / "syn_config.json").is_file())
            self.assertIsNotNone(state.cell)
            self.assertEqual(
                state.setup_summary["actions"]["validate"]["load_cell"],
                "deferred to the single shared pipeline build",
            )

    def test_setup_source_infers_manifest_and_rejects_unconfigured_custom_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp) / "raw"
            tune.mkdir()
            (tune / "manifest.json").write_text("{}\n", encoding="utf-8")
            source, loader, config = _select_setup_source(
                tune,
                adb_specimen_id=None,
            )
            self.assertEqual((source, loader, config), ("existing", "allen_manifest", {}))

            (tune / "manifest.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "1_setup.ipynb"):
                _select_setup_source(tune, adb_specimen_id=None)

            (tune / "cell_configs").mkdir()
            _write_json(
                tune / "cell_configs" / "cell_config.json",
                {"cell_name": "unregistered_custom", "paths": {}},
            )
            with self.assertRaisesRegex(FileNotFoundError, "1_setup.ipynb"):
                _select_setup_source(tune, adb_specimen_id=None)

            shutil.rmtree(tune / "cell_configs")
            source, loader, config = _select_setup_source(
                tune,
                adb_specimen_id=123,
            )
            self.assertEqual((source, loader, config), ("adb", "allen_manifest", {}))

    def test_explicit_adb_setup_requests_download_without_forcing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp) / "cells" / "PV" / "tunes" / "tuned"
            fake_context = SimpleNamespace(
                cell_name="PV",
                tune_name="tuned",
                cell_config={"cell_name": "PV", "cell_loader": "allen_manifest"},
                sim_config={},
            )
            setup_summary = {
                "actions": {
                    "mechanisms": {"compile_modfiles": {"status": "ok"}},
                }
            }
            with (
                mock.patch(
                    "modules.setup.step1_prepare.prepare_tune",
                    return_value=setup_summary,
                ) as prepare,
                mock.patch(
                    "modules.setup.validation.validate_tune",
                    return_value={},
                ),
                mock.patch(
                    "modules.tuning.prepare_tuning_notebook_context",
                    return_value=fake_context,
                ),
                mock.patch("modules.tuning.build_tuning_cell", return_value=object()),
                mock.patch(
                    "modules.notebooks.pipeline_workflow._model_source_fingerprint",
                    return_value={"fake": "hash"},
                ),
            ):
                prepare_pipeline_notebook(
                    repo_root=REPO_ROOT,
                    cell_name="PV",
                    tune_name="tuned",
                    tune_dir_override=tune,
                    adb_specimen_id=484635029,
                )

            kwargs = prepare.call_args.kwargs
            self.assertEqual(kwargs["source_type"], "adb")
            self.assertTrue(kwargs["do_download"])
            self.assertFalse(kwargs["force_download"])
            self.assertEqual(kwargs["config_mode"], "fill")
            self.assertFalse(kwargs["do_validate"])

    def test_source_fingerprint_ignores_runtime_configs_and_detects_hoc_edits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = _copy_unique_hoc_tune(Path(tmp))
            state = prepare_pipeline_notebook(
                repo_root=REPO_ROOT,
                cell_name="synthetic",
                tune_name="tuned",
                tune_dir_override=tune,
            )

            for filename in ("target_config.json", "sim_config.json", "syn_config.json"):
                path = tune / "cell_configs" / filename
                config = _read_json(path)
                config["pipeline_test_runtime_edit"] = filename
                _write_json(path, config)
            state.assert_sources_unchanged()

            hoc_path = tune / "model" / "ScopedCell.hoc"
            hoc_path.write_text(
                hoc_path.read_text(encoding="utf-8") + "\n// pipeline source edit\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RestartKernelRequired, "Restart the kernel"):
                state.assert_sources_unchanged()

    def test_fingerprint_tracks_cell_fit_morphology_and_mod_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp) / "tune"
            config_dir = tune / "cell_configs"
            model_dir = tune / "model"
            mod_dir = tune / "modfiles"
            config_dir.mkdir(parents=True)
            model_dir.mkdir()
            mod_dir.mkdir()

            fit_path = model_dir / "fit.json"
            morphology_path = model_dir / "morphology.swc"
            manifest_path = model_dir / "manifest.json"
            mod_path = mod_dir / "pipeline_test.mod"
            fit_path.write_text('{"fit": 1}\n', encoding="utf-8")
            morphology_path.write_text("# synthetic morphology\n", encoding="utf-8")
            mod_path.write_text("NEURON { SUFFIX pipeline_test }\n", encoding="utf-8")
            _write_json(
                manifest_path,
                {
                    "biophys": [{"model_file": ["fit.json"]}],
                    "manifest": [
                        {
                            "type": "file",
                            "key": "MORPHOLOGY",
                            "spec": "morphology.swc",
                        }
                    ],
                },
            )
            cell_config_path = config_dir / "cell_config.json"
            _write_json(
                cell_config_path,
                {
                    "cell_name": "fingerprint_fixture",
                    "cell_loader": "allen_manifest",
                    "paths": {
                        "manifest": "model/manifest.json",
                        "modfiles": "modfiles",
                    },
                },
            )

            previous = _model_source_fingerprint(tune)
            for path, text in (
                (fit_path, '{"fit": 2}\n'),
                (morphology_path, "# edited morphology\n"),
                (mod_path, "NEURON { SUFFIX pipeline_test_edited }\n"),
            ):
                path.write_text(text, encoding="utf-8")
                current = _model_source_fingerprint(tune)
                self.assertNotEqual(current, previous, path.name)
                previous = current

            cell_config = _read_json(cell_config_path)
            cell_config["fingerprint_test"] = True
            _write_json(cell_config_path, cell_config)
            self.assertNotEqual(_model_source_fingerprint(tune), previous)

    def test_passive_act_proposal_is_returned_without_writing_model_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = _copy_unique_hoc_tune(Path(tmp))
            state = prepare_pipeline_notebook(
                repo_root=REPO_ROOT,
                cell_name="synthetic",
                tune_name="tuned",
                tune_dir_override=tune,
            )
            hoc_path = tune / "model" / "ScopedCell.hoc"
            cell_config_path = tune / "cell_configs" / "cell_config.json"
            before = (hoc_path.read_bytes(), cell_config_path.read_bytes())
            proposal = SimpleNamespace(
                e_rev_leak=-70.0,
                g_bar_leak=0.0001,
                Cm=1.0,
            )
            resolution = SimpleNamespace(
                act_passive_module=None,
                passive_targets={
                    "target_rin_mohm": 100.0,
                    "target_tau_ms": 10.0,
                    "target_v_rest_mv": -70.0,
                },
                settable_passive_properties=proposal,
                fit_json_candidates=[],
            )
            amp = -25.0
            records = {
                "T": {amp: np.asarray([0.0, 1.0, 2.0])},
                "V": {amp: np.asarray([-70.0, -71.0, -70.5])},
                "F": {amp: 0.0},
                "I": {amp: {}},
            }
            with (
                mock.patch(
                    "modules.tuning.resolve_passive_tuning_inputs",
                    return_value=resolution,
                ),
                mock.patch(
                    "modules.tuning.run_passive_protocol",
                    return_value=records,
                ) as run_protocol,
            ):
                result = run_passive_stage(
                    state,
                    amps_pA=[amp],
                    compute_act_proposal=True,
                )

            self.assertEqual(len(result.proposal_changes), 3)
            self.assertIs(run_protocol.call_args.kwargs["cell"], state.cell)
            self.assertEqual(before, (hoc_path.read_bytes(), cell_config_path.read_bytes()))

    def test_active_stage_reuses_shared_cell_for_traces_and_fi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = _copy_unique_hoc_tune(Path(tmp))
            state = prepare_pipeline_notebook(
                repo_root=REPO_ROOT,
                cell_name="synthetic",
                tune_name="tuned",
                tune_dir_override=tune,
            )
            target_resolution = SimpleNamespace(
                fi_csv_path=None,
                fi_reference_points=[],
            )
            with (
                mock.patch(
                    "modules.tuning.run_active_protocol",
                    return_value={},
                ) as run_protocol,
                mock.patch(
                    "modules.tuning.active_metric_rows",
                    side_effect=[
                        [{"amp_pA": 150.0, "spike_frequency_hz": 5.0}],
                        [{"amp_pA": 0.0, "spike_frequency_hz": 0.0}],
                    ],
                ),
                mock.patch(
                    "modules.tuning.fi_rows_from_metrics",
                    return_value=[{"amp_pA": 0.0, "spike_frequency_hz": 0.0}],
                ),
                mock.patch(
                    "modules.tuning.resolve_active_tuning_targets",
                    return_value=target_resolution,
                ),
                mock.patch(
                    "modules.tuning.plot_active_trace_check",
                    return_value=object(),
                ),
                mock.patch(
                    "modules.tuning.plot_fi_curve",
                    return_value=object(),
                ),
            ):
                run_active_stage(
                    state,
                    active_amps_pA=[150],
                    fi_amps_pA=[0],
                )

            self.assertEqual(run_protocol.call_count, 2)
            for call in run_protocol.call_args_list:
                self.assertIs(call.kwargs["cell"], state.cell)

    def test_synapse_tuner_reuses_shared_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = _copy_unique_hoc_tune(Path(tmp))
            state = prepare_pipeline_notebook(
                repo_root=REPO_ROOT,
                cell_name="synthetic",
                tune_name="tuned",
                tune_dir_override=tune,
            )
            fake_tuner = object()
            with mock.patch(
                "modules.tuning.create_scp_synapse_tuner",
                return_value=fake_tuner,
            ) as create_tuner:
                tuner = prepare_interactive_synapse_tuner(state)

            self.assertIs(tuner, fake_tuner)
            synapse_session = create_tuner.call_args.args[0]
            self.assertIs(synapse_session.cell, state.cell)

    def test_fresh_iclamp_run_uses_explicit_stem_and_loads_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = _copy_unique_hoc_tune(Path(tmp))
            state = prepare_pipeline_notebook(
                repo_root=REPO_ROOT,
                cell_name="synthetic",
                tune_name="tuned",
                tune_dir_override=tune,
            )
            sim_path = tune / "cell_configs" / "sim_config.json"
            sim_config = _read_json(sim_path)
            sim_config["output_stem"] = "configured_stem_should_not_win"
            _write_json(sim_path, sim_config)

            result = run_fresh_simulation(
                state,
                n_trials=1,
                iclamp=True,
                output_stem="explicit_pipeline_stem",
            )

            self.assertEqual(result.output_stem, "explicit_pipeline_stem")
            self.assertTrue(result.manifest_path.is_file())
            self.assertEqual(result.results["mode"], "iclamp")
            self.assertIn("T", result.results["traces"])


if __name__ == "__main__":
    unittest.main()
