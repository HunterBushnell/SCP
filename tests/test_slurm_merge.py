from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from modules import run_sim
from modules.analysis import analysis
from scripts import merge_array_results


class SlurmMergeTests(unittest.TestCase):
    def test_merged_stem_replaces_stale_array_task_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_data = Path(tmp) / "output_data"
            run_root = output_data / "slurm_123"
            parts_dir = run_root / "parts"

            part_result = {
                "mode": "multi",
                "sim_cfg": {
                    "cell": "SST",
                    "tune": "tuned",
                    "tstart": 0.0,
                    "tstop": 10.0,
                    "dt": 0.1,
                    "bins": 1.0,
                    "n_trials": 1,
                    "n_traces_to_save": 0,
                    "n_inputs_to_save": 0,
                    "output": "slurm_123_0",
                    "output_stem": "slurm_123_0",
                    "save_output": True,
                    "save_model_artifacts": False,
                },
                "spikes": [np.asarray([], dtype=float)],
                "traces": {},
                "meta": {"n_trials": 1, "trial_ids": [0]},
            }
            part_manifest = run_sim.save_results(part_result, base_dir=parts_dir)
            self.assertEqual(Path(part_manifest).parent.name, "slurm_123_0")

            argv = [
                "merge_array_results.py",
                "--input-dir",
                str(parts_dir),
                "--output-dir",
                str(run_root),
                "--job-id",
                "123",
                "--output-stem",
                "results",
            ]
            with mock.patch.object(sys, "argv", argv):
                merge_array_results.main()

            merged_dir = run_root / "results"
            self.assertTrue((merged_dir / "run_manifest.json").is_file())
            self.assertFalse((run_root / "slurm_123_0").exists())

            merged = run_sim.load_results(merged_dir)
            self.assertEqual(merged["sim_cfg"]["output"], "results")
            self.assertEqual(merged["sim_cfg"]["output_stem"], "results")
            self.assertEqual(len(merged["spikes"]), 1)

            discovered = analysis.collect_run_dirs(output_data)
            self.assertEqual(discovered, [merged_dir])


if __name__ == "__main__":
    unittest.main()
