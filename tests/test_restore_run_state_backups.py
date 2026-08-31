from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.restore_run_state import restore_run_state


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class RestoreRunStateBackupTests(unittest.TestCase):
    def test_each_restore_uses_one_timestamped_backup_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            tune_dir = root / "tune"
            config_dir = tune_dir / "cell_configs"

            _write_json(
                run_dir / "run_manifest.json",
                {
                    "files": {
                        "cell_config": "cell_config.json",
                        "syn_config": "syn_config.json",
                    }
                },
            )
            _write_json(run_dir / "cell_config.json", {"setting": "restored"})
            _write_json(
                run_dir / "syn_config.json",
                {
                    "pn_exc": {"weight": 2.0},
                    "bg_exc": {"weight": 3.0},
                },
            )

            _write_json(config_dir / "cell_config.json", {"setting": "current"})
            _write_json(
                config_dir / "syn_config.json",
                {
                    "group_files": [
                        "syn_groups/pn_exc.json",
                        "syn_groups/bg_exc.json",
                    ]
                },
            )
            _write_json(
                config_dir / "syn_groups" / "pn_exc.json",
                {"pn_exc": {"weight": 1.0}},
            )
            _write_json(
                config_dir / "syn_groups" / "bg_exc.json",
                {"bg_exc": {"weight": 1.0}},
            )

            preview = restore_run_state(
                from_run=run_dir,
                to_tune=tune_dir,
                apply=["cell_config", "syn_groups"],
                dry_run=True,
            )
            self.assertIsNone(preview.backup_dir)
            self.assertFalse((tune_dir / "restore_backups").exists())

            first = restore_run_state(
                from_run=run_dir,
                to_tune=tune_dir,
                apply=["cell_config", "syn_groups"],
                dry_run=False,
            )

            self.assertIsNotNone(first.backup_dir)
            first_backup_set = first.backup_dir
            self.assertEqual(first_backup_set.parent, tune_dir / "restore_backups")
            self.assertRegex(first_backup_set.name, r"^\d{8}_\d{6}_\d{6}(?:_\d{3})?$")
            self.assertEqual(
                json.loads((first_backup_set / "cell_configs" / "cell_config.json").read_text()),
                {"setting": "current"},
            )
            self.assertEqual(
                json.loads(
                    (
                        first_backup_set
                        / "cell_configs"
                        / "syn_groups"
                        / "pn_exc.json"
                    ).read_text()
                ),
                {"pn_exc": {"weight": 1.0}},
            )
            self.assertEqual(
                json.loads(
                    (
                        first_backup_set
                        / "cell_configs"
                        / "syn_groups"
                        / "bg_exc.json"
                    ).read_text()
                ),
                {"bg_exc": {"weight": 1.0}},
            )
            self.assertFalse(list(config_dir.rglob("*.bak_*")))
            backup_paths = [
                item.backup_path for item in first.file_reports if item.backup_path
            ]
            self.assertEqual(len(backup_paths), 3)
            self.assertTrue(
                all(path.is_relative_to(first_backup_set) for path in backup_paths)
            )

            _write_json(config_dir / "cell_config.json", {"setting": "current-again"})
            second = restore_run_state(
                from_run=run_dir,
                to_tune=tune_dir,
                apply=["cell_config"],
                dry_run=False,
            )

            self.assertIsNotNone(second.backup_dir)
            self.assertNotEqual(second.backup_dir, first_backup_set)
            self.assertEqual(
                len(list((tune_dir / "restore_backups").iterdir())),
                2,
            )

            no_changes = restore_run_state(
                from_run=run_dir,
                to_tune=tune_dir,
                apply=["cell_config"],
                dry_run=False,
            )
            self.assertIsNone(no_changes.backup_dir)
            self.assertEqual(
                len(list((tune_dir / "restore_backups").iterdir())),
                2,
            )


if __name__ == "__main__":
    unittest.main()
