from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from modules.notebooks.pipeline_act import (
    inspect_act_active_stage,
    prepare_act_active_stage,
    run_fresh_act_active,
)
from modules.tuning.act_active import (
    CONFIG_NAME,
    act_config_fingerprint,
    act_output_status,
    default_act_active_config,
    estimate_act_workload,
    prepare_act_active_workspace,
    run_act_active_module,
)


ACTIVE_NAMES = [
    "gbar_Nap",
    "gbar_K_T",
    "gbar_Im_v2",
    "gbar_NaTa",
    "gbar_Kd",
    "gbar_Ca_LVA",
    "gbar_Ca_HVA",
    "gbar_Kv2like",
    "gbar_Kv3_1",
]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _mapping_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_tune(root: Path, cell_name: str = "PV", loader: str = "allen_manifest") -> Path:
    tune = root / "cells" / cell_name / "tunes" / "tuned"
    configs = tune / "cell_configs"
    configs.mkdir(parents=True)
    (tune / "modfiles").mkdir()
    (configs / "cell_config.json").write_text(
        json.dumps(
            {
                "cell_name": cell_name,
                "cell_loader": loader,
                "paths": {"manifest": "manifest.json", "modfiles": "modfiles"},
            }
        ),
        encoding="utf-8",
    )
    (configs / "target_config.json").write_text(
        json.dumps(
            {
                "target_source": {"mode": "manual"},
                "manual": {
                    "fi_curve": {
                        "currents_pA": [0, 100, 200],
                        "rates_Hz": [0, 10, 30],
                        "csv": None,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return tune


def _fake_state(root: Path, tune: Path, cell_name: str = "PV", loader: str = "allen_manifest"):
    segment = SimpleNamespace(
        **{
            name: 0.01
            for name in [*ACTIVE_NAMES, "g_pas", "e_pas", "gbar_Ih"]
        }
    )

    class Section:
        def __call__(self, _position):
            return segment

    return SimpleNamespace(
        repo_root=root,
        tune_dir=tune,
        context=SimpleNamespace(
            cell_name=cell_name,
            tune_name="tuned",
            tune_dir=tune,
            cell_config={"cell_loader": loader},
        ),
        cell=SimpleNamespace(soma=[Section()]),
        assert_sources_unchanged=lambda: None,
    )


class PipelineACTTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("SCP_RUN_ACT_INTEGRATION") == "1",
        "set SCP_RUN_ACT_INTEGRATION=1 for the tiny real ACT smoke test",
    )
    def test_optional_tiny_real_act_smoke(self):
        from modules.notebooks import prepare_pipeline_notebook

        state = prepare_pipeline_notebook(
            repo_root=PROJECT_ROOT, cell_name="PV", tune_name="tuned"
        )
        overrides = {
            "target": {
                "mode": "fi_arrays",
                "fi_currents_pA": [0, 100],
                "fi_frequencies_hz": [0, 3.3],
            },
            "modules": {
                "lto": {
                    "enabled": True,
                    "name": "smoke_lto",
                    "conductances": [
                        {
                            "variable_name": "gbar_Nap",
                            "low": 0.00008,
                            "high": 0.0001,
                            "n_slices": 2,
                        }
                    ],
                }
            },
            "simulation": {
                "h_v_init": -71.0,
                "h_tstop": 300.0,
                "h_dt": 0.1,
                "h_celsius": 37.0,
                "ci_delay_ms": 50.0,
                "ci_dur_ms": 200.0,
            },
            "optimizer": {"n_cpus": 1, "n_estimators": 2, "max_depth": 2},
        }
        with tempfile.TemporaryDirectory(prefix="scp_act_integration_") as tmp:
            prepared = prepare_act_active_stage(
                state, workspace=tmp, overrides=overrides, probe_act=True
            )
            self.assertTrue(prepared.act_available, prepared.act_message)
            result = run_fresh_act_active(
                state, prepared, modules="lto", n_cpus=1
            )
            self.assertIn("gbar_Nap", result.predictions)

    def test_curated_inspection_uses_target_config_and_first_class_workload(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            tune = _write_tune(data_root)
            result = inspect_act_active_stage(
                _fake_state(PROJECT_ROOT, tune), workspace=tune / "scratch_act"
            )
            self.assertEqual(result.config_source, "curated preset")
            self.assertEqual(result.loader_support, "supported")
            self.assertEqual(result.target_mode, "fi_arrays")
            self.assertEqual(result.target_point_count, 3)
            self.assertEqual(result.enabled_modules, ["lto", "spiking", "bursting"])
            self.assertGreater(result.workload["training_traces"], 0)
            self.assertEqual(result.resolved_config["simulation"]["ci_delay_ms"], 200.0)
            self.assertEqual(result.resolved_config["simulation"]["ci_dur_ms"], 1000.0)

    def test_unconfigured_custom_cell_is_actionable(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            tune = _write_tune(data_root, cell_name="Custom", loader="hoc_template")
            with self.assertRaisesRegex(FileNotFoundError, "3_active.ipynb"):
                inspect_act_active_stage(
                    _fake_state(PROJECT_ROOT, tune, "Custom", "hoc_template")
                )

    def test_configured_non_allen_loader_is_labeled_experimental(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            tune = _write_tune(data_root, cell_name="Custom", loader="hoc_template")
            workspace = tune / "act_workspace"
            workspace.mkdir()
            cfg = default_act_active_config(
                repo_root=PROJECT_ROOT,
                tune_dir=tune,
                cell_name="Custom",
                tune_name="tuned",
                workspace=workspace,
            )
            (workspace / CONFIG_NAME).write_text(json.dumps(cfg), encoding="utf-8")
            result = inspect_act_active_stage(
                _fake_state(PROJECT_ROOT, tune, "Custom", "hoc_template")
            )
            self.assertEqual(result.config_source, "existing")
            self.assertEqual(result.loader_support, "experimental")

    def test_fi_csv_target_is_resolved_and_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            tune = _write_tune(data_root)
            (tune / "targets.csv").write_text(
                "amp_pA,spike_frequency_hz\n0,0\n125,8\n250,35\n",
                encoding="utf-8",
            )
            target_path = tune / "cell_configs" / "target_config.json"
            target_path.write_text(
                json.dumps(
                    {
                        "target_source": {"mode": "manual"},
                        "manual": {
                            "fi_curve": {
                                "currents_pA": [],
                                "rates_Hz": [],
                                "csv": "targets.csv",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = inspect_act_active_stage(
                _fake_state(PROJECT_ROOT, tune), workspace=tune / "scratch_act"
            )
            self.assertEqual(result.target_mode, "fi_csv")
            self.assertEqual(result.target_point_count, 3)
            self.assertEqual(
                result.resolved_config["simulation"]["ci_amps_pA"],
                [0.0, 125.0, 250.0],
            )

    def test_mocked_nwb_preparation_records_extracted_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            tune = _write_tune(data_root)
            nwb_path = tune / "cell_ephys.nwb"
            nwb_path.write_bytes(b"mock")
            target_path = tune / "cell_configs" / "target_config.json"
            target_path.write_text(
                json.dumps(
                    {
                        "target_source": {"mode": "allen_nwb"},
                        "allen_nwb": {"file": "cell_ephys.nwb"},
                    }
                ),
                encoding="utf-8",
            )

            def fake_extract(_source, act_target_path, *, summary_path, **_kwargs):
                Path(act_target_path).write_text(
                    "mean_i,spike_frequency\n0,0\n0.1,12\n", encoding="utf-8"
                )
                Path(summary_path).write_text(
                    "current_pA,spike_frequency_hz\n0,0\n100,12\n", encoding="utf-8"
                )
                return {"currents_pA": [0.0, 100.0], "frequencies_hz": [0.0, 12.0]}

            with mock.patch(
                "modules.tuning.act_active.write_allen_nwb_fi_target_csv",
                side_effect=fake_extract,
            ):
                result = prepare_act_active_stage(
                    _fake_state(PROJECT_ROOT, tune),
                    workspace=tune / "scratch_act",
                    probe_act=False,
                )
            self.assertEqual(result.target_mode, "allen_nwb")
            self.assertEqual(result.target_point_count, 2)
            self.assertEqual(
                result.resolved_config["simulation"]["ci_amps_pA"],
                [0.0, 100.0],
            )

    def test_existing_trace_npy_config_is_reused_but_not_generated(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            tune = _write_tune(data_root)
            workspace = tune / "act_workspace"
            workspace.mkdir()
            trace_path = workspace / "target_trace.npy"
            np.save(trace_path, np.zeros((2, 5, 2), dtype=float))
            cfg = default_act_active_config(
                repo_root=PROJECT_ROOT,
                tune_dir=tune,
                cell_name="PV",
                tune_name="tuned",
                workspace=workspace,
            )
            cfg["target"] = {
                "mode": "trace_npy",
                "path": trace_path.name,
                "source_npy": trace_path.name,
            }
            cfg["simulation"]["ci_amps_pA"] = [0.0, 100.0]
            (workspace / CONFIG_NAME).write_text(json.dumps(cfg), encoding="utf-8")
            result = inspect_act_active_stage(_fake_state(PROJECT_ROOT, tune))
            self.assertEqual(result.target_mode, "trace_npy")
            self.assertEqual(result.target_point_count, 2)

            cfg["simulation"]["ci_amps_pA"] = [0.0]
            (workspace / CONFIG_NAME).write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one ci_amps_pA"):
                inspect_act_active_stage(_fake_state(PROJECT_ROOT, tune))

    def test_target_or_act_config_edit_requires_preparation_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            tune = _write_tune(data_root)
            state = _fake_state(PROJECT_ROOT, tune)
            prepared = prepare_act_active_stage(
                state, workspace=tune / "act_workspace", probe_act=False
            )
            target_config = tune / "cell_configs" / "target_config.json"
            target_config.write_text(
                target_config.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "Prepare ACT workspace"):
                run_fresh_act_active(state, prepared, modules="lto", n_cpus=1)

            prepared = prepare_act_active_stage(
                state, workspace=tune / "act_workspace", probe_act=False
            )
            act_config = json.loads(prepared.config_path.read_text(encoding="utf-8"))
            act_config["optimizer"]["random_state"] += 1
            prepared.config_path.write_text(json.dumps(act_config), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "ACT config"):
                run_fresh_act_active(state, prepared, modules="lto", n_cpus=1)

    def test_prepare_preserves_existing_config_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            tune = _write_tune(data_root)
            workspace = tune / "act_workspace"
            prepare_act_active_workspace(
                repo_root=PROJECT_ROOT,
                tune_dir=tune,
                cell_name="PV",
                tune_name="tuned",
                workspace=workspace,
                target_mode="fi_arrays",
                fi_currents_pA=[0, 100],
                fi_frequencies_hz=[0, 12],
            )
            path = workspace / CONFIG_NAME
            cfg = json.loads(path.read_text(encoding="utf-8"))
            cfg["custom_metadata"] = {"keep": True}
            cfg["optimizer"]["n_estimators"] = 17
            path.write_text(json.dumps(cfg), encoding="utf-8")
            prepare_act_active_workspace(
                repo_root=PROJECT_ROOT,
                tune_dir=tune,
                cell_name="PV",
                tune_name="tuned",
                workspace=workspace,
                target_mode=None,
                optimizer={"n_cpus": 1},
                preserve_existing=True,
            )
            updated = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(updated["custom_metadata"], {"keep": True})
            self.assertEqual(updated["optimizer"]["n_estimators"], 17)
            self.assertEqual(updated["optimizer"]["n_cpus"], 1)
            builder = (workspace / "cell_builder.py").read_text(encoding="utf-8")
            self.assertIn("_apply_fixed_predictions", builder)

    def test_workload_and_stale_dependency_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            tune = _write_tune(data_root)
            workspace = tune / "act_workspace"
            cfg = default_act_active_config(
                repo_root=PROJECT_ROOT,
                tune_dir=tune,
                cell_name="PV",
                tune_name="tuned",
                workspace=workspace,
            )
            cfg["target"]["fi_currents_pA"] = [0, 100]
            cfg["target"]["fi_frequencies_hz"] = [0, 10]
            cfg["simulation"]["ci_amps_pA"] = [0, 100]
            workspace.mkdir()
            (workspace / "target_sf.csv").write_text(
                "mean_i,spike_frequency\n0,0\n0.1,10\n", encoding="utf-8"
            )
            (workspace / CONFIG_NAME).write_text(json.dumps(cfg), encoding="utf-8")
            estimate = estimate_act_workload(cfg, modules="lto")
            self.assertEqual(estimate["training_traces"], 250)

            config_hash = act_config_fingerprint(cfg)
            lto = {"gbar_Nap": 0.01}
            (workspace / "prediction_lto.json").write_text(json.dumps(lto), encoding="utf-8")
            (workspace / "metrics_lto.csv").write_text(
                "metric,value\nTrain MAE (g),0.1\n", encoding="utf-8"
            )
            lto_manifest = {
                "status": "complete",
                "config_fingerprint": config_hash,
                "prior_predictions_fingerprint": _mapping_hash({}),
                "prediction_fingerprint": _mapping_hash(lto),
            }
            (workspace / "run_manifest_lto.json").write_text(
                json.dumps(lto_manifest), encoding="utf-8"
            )
            spiking = {"gbar_NaTa": 0.1}
            (workspace / "prediction_spiking.json").write_text(
                json.dumps(spiking), encoding="utf-8"
            )
            (workspace / "metrics_spiking.csv").write_text(
                "metric,value\nTrain MAE (g),0.1\n", encoding="utf-8"
            )
            spiking_manifest = {
                "status": "complete",
                "config_fingerprint": config_hash,
                "prior_predictions_fingerprint": _mapping_hash(lto),
                "prediction_fingerprint": _mapping_hash(spiking),
            }
            (workspace / "run_manifest_spiking.json").write_text(
                json.dumps(spiking_manifest), encoding="utf-8"
            )
            self.assertEqual(act_output_status(cfg)["spiking"]["status"], "current")

            rerun = {"gbar_Nap": 0.02}
            (workspace / "prediction_lto.json").write_text(json.dumps(rerun), encoding="utf-8")
            lto_manifest["prediction_fingerprint"] = _mapping_hash(rerun)
            (workspace / "run_manifest_lto.json").write_text(
                json.dumps(lto_manifest), encoding="utf-8"
            )
            statuses = act_output_status(cfg)
            self.assertEqual(statuses["lto"]["status"], "current")
            self.assertEqual(statuses["spiking"]["status"], "stale")

    def test_later_module_receives_prior_predictions_through_builder_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            tune = _write_tune(data_root)
            workspace = tune / "act_workspace"
            workspace.mkdir()
            cfg = default_act_active_config(
                repo_root=PROJECT_ROOT,
                tune_dir=tune,
                cell_name="PV",
                tune_name="tuned",
                workspace=workspace,
            )
            cfg["modules"] = {
                "lto": {
                    "name": "lto",
                    "conductances": [
                        {
                            "variable_name": "g1",
                            "low": 0.0,
                            "high": 1.0,
                            "n_slices": 2,
                        }
                    ],
                },
                "spiking": {
                    "name": "spiking",
                    "conductances": [
                        {
                            "variable_name": "g2",
                            "low": 0.0,
                            "high": 1.0,
                            "n_slices": 2,
                        }
                    ],
                },
            }
            (workspace / "target_sf.csv").write_text(
                "mean_i,spike_frequency\n0,0\n", encoding="utf-8"
            )
            (workspace / CONFIG_NAME).write_text(json.dumps(cfg), encoding="utf-8")
            (workspace / "prediction_lto.json").write_text(
                json.dumps({"g1": 0.25}), encoding="utf-8"
            )

            cell = SimpleNamespace(prediction={"g2": None})

            class Metrics:
                def to_csv(self, path, index=False):
                    Path(path).write_text("metric,value\nTrain MAE,0.1\n", encoding="utf-8")

            class Module:
                def __init__(self, **kwargs):
                    self.cell = kwargs["cell"]

                def run(self):
                    fixed = json.loads(
                        (workspace / "fixed_predictions.json").read_text(encoding="utf-8")
                    )
                    if fixed != {"g1": 0.25}:
                        raise AssertionError(fixed)
                    self.cell.prediction = {"g2": 0.5}
                    return Metrics()

            api = {"ACTModule": Module}
            with (
                mock.patch(
                    "modules.tuning.act_active._import_act_api", return_value=api
                ),
                mock.patch(
                    "modules.tuning.act_active._import_workspace_builder",
                    return_value=object(),
                ),
                mock.patch(
                    "modules.tuning.act_active._build_act_cell", return_value=cell
                ),
                mock.patch(
                    "modules.tuning.act_active._build_simulation_parameters",
                    return_value=object(),
                ),
                mock.patch(
                    "modules.tuning.act_active._build_optimization_parameters",
                    return_value=object(),
                ),
            ):
                result = run_act_active_module(workspace, "spiking")
            self.assertEqual(result["prediction"], {"g2": 0.5})
            self.assertFalse((workspace / "fixed_predictions.json").exists())


if __name__ == "__main__":
    unittest.main()
