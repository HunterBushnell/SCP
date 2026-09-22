from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_public_cleanup import (
    _cache_artifacts,
    _copy_verified,
    _validate_public_runs,
)
from scripts.sanitize_public_runs import DEFAULT_PUBLIC_RUNS, sanitize_run
from scripts.verify_public_archive import verify_archive


class PublicRunSanitationTests(unittest.TestCase):
    def test_archive_copy_verifies_exact_tree_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            destination = root / "archive" / "source"
            (source / "empty").mkdir(parents=True)
            (source / "nested").mkdir()
            (source / "nested" / "data.txt").write_text("content", encoding="utf-8")

            files, total = _copy_verified(source, destination)
            self.assertEqual(files, 1)
            self.assertEqual(total, len("content"))
            self.assertTrue((destination / "empty").is_dir())

    def test_cleanup_refuses_missing_curated_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing"
            with self.assertRaisesRegex(FileNotFoundError, "Curated public run"):
                _validate_public_runs({missing})

    def test_cache_discovery_never_enters_existing_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            active_cache = repo / "modules" / "__pycache__"
            archived_cache = repo / "archive" / "older" / "__pycache__"
            active_cache.mkdir(parents=True)
            archived_cache.mkdir(parents=True)

            discovered = _cache_artifacts(repo)
            self.assertIn(active_cache, discovered)
            self.assertNotIn(archived_cache, discovered)

    def test_archive_manifest_verifier_detects_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            archive = repo / "archive" / "release"
            payload = archive / "output_runs" / "example"
            payload.mkdir(parents=True)
            (payload / "result.txt").write_text("result", encoding="utf-8")
            manifest = {
                "records": [{
                    "destination": "archive/release/output_runs/example",
                    "files": 1,
                    "bytes": len("result"),
                }]
            }
            (archive / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            self.assertEqual(verify_archive(repo, archive), [])
            (payload / "result.txt").write_text("changed", encoding="utf-8")
            self.assertRegex(verify_archive(repo, archive)[0], "Archive mismatch")

    def test_bundled_public_runs_are_portable_and_self_consistent(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        forbidden = ("/home/", "/Users/", "C:\\Users\\")

        for relative in DEFAULT_PUBLIC_RUNS:
            with self.subTest(run=relative):
                run = repo / relative
                results = run / "results"
                manifest_path = results / "run_manifest.json"
                self.assertTrue(manifest_path.is_file())
                self.assertFalse((run / "logs").exists())
                self.assertFalse((results / "input_stats.json").exists())

                for json_path in run.rglob("*.json"):
                    serialized = json_path.read_text(encoding="utf-8")
                    for token in forbidden:
                        self.assertNotIn(token, serialized)

                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertNotIn("input_stats", manifest["files"])
                for value in manifest["files"].values():
                    paths = value.values() if isinstance(value, dict) else (value,)
                    for path in paths:
                        self.assertTrue(
                            (results / path).exists(),
                            f"Missing manifest target in {relative}: {path}",
                        )

    def test_sanitizes_paths_manifest_and_local_only_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir) / "SCP"
            run = repo / "cells" / "PV" / "tunes" / "tuned" / "output_data" / "PV5_base"
            results = run / "results"
            plots = run / "plots"
            logs = run / "logs"
            results.mkdir(parents=True)
            plots.mkdir()
            logs.mkdir()
            (results / "input_stats.json").write_text("{}\n", encoding="utf-8")
            (plots / "single_plot.svg").write_text("plot", encoding="utf-8")
            (logs / "job.out").write_text("local", encoding="utf-8")
            (results / "sim_cfg.json").write_text(
                json.dumps({"tune_dir": str(repo / "cells/PV/tunes/tuned")}),
                encoding="utf-8",
            )
            (results / "run_manifest.json").write_text(
                json.dumps({
                    "files": {
                        "input_stats": "input_stats.json",
                        "plots": {"single_plot": str(repo / "somewhere/single_plot.svg")},
                    }
                }),
                encoding="utf-8",
            )

            preview = sanitize_run(run, repo)
            self.assertEqual(len(preview.removed), 2)
            self.assertEqual(len(preview.rewritten), 2)
            self.assertTrue(logs.exists())

            sanitize_run(run, repo, apply=True)
            self.assertFalse(logs.exists())
            self.assertFalse((results / "input_stats.json").exists())
            sim_cfg = json.loads((results / "sim_cfg.json").read_text(encoding="utf-8"))
            self.assertEqual(sim_cfg["tune_dir"], "cells/PV/tunes/tuned")
            manifest = json.loads((results / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertNotIn("input_stats", manifest["files"])
            self.assertEqual(manifest["files"]["plots"]["single_plot"], "../plots/single_plot.svg")

    def test_refuses_unresolved_external_absolute_paths_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir) / "SCP"
            run = repo / "cells" / "PV" / "tunes" / "tuned" / "output_data" / "PV5_base"
            results = run / "results"
            logs = run / "logs"
            results.mkdir(parents=True)
            logs.mkdir()
            (logs / "job.out").write_text("local", encoding="utf-8")
            (results / "meta.json").write_text(
                json.dumps({"external_path": "/private/research/data.csv"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unresolved absolute paths"):
                sanitize_run(run, repo, apply=True)
            self.assertTrue(logs.exists())


if __name__ == "__main__":
    unittest.main()
