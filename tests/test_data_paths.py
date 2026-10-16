from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from modules.data_paths import resolve_data_alias
from modules.input_generation.mode_helpers import _resolve_source_path
from modules.input_generation.mode_precomputed import _mode_precomputed


class RelocatedDataTests(unittest.TestCase):
    def fixture(self, root):
        (root / 'modules').mkdir()
        (root / 'run_pipeline.py').touch()
        target = root / 'cells/Example/data/input.csv'
        target.parent.mkdir(parents=True)
        target.write_bytes(b'Time,AvgFiringRate\n0,1\n')
        alias = root / 'cells/DATA_PATH_ALIASES.json'
        alias.write_text(json.dumps({'aliases': {'external_data/input.csv': {
            'path': 'cells/Example/data/input.csv', 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}}}))
        tune = root / 'cells/Example/tunes/orig'
        tune.mkdir(parents=True)
        return target, tune, alias

    def test_explicit_alias_preserves_old_config_and_original_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, tune, _ = self.fixture(root)
            self.assertEqual(resolve_data_alias('external_data/input.csv', root), target)
            self.assertEqual(_resolve_source_path('external_data/input.csv', {'tune_dir': str(tune)}), target)
            original = root / 'external_data/input.csv'
            original.parent.mkdir()
            original.write_text('an explicitly retained original')
            self.assertEqual(resolve_data_alias(original, root), original)
            self.assertEqual(resolve_data_alias('unknown.csv', root), root / 'unknown.csv')

    def test_alias_cannot_substitute_changed_or_missing_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, _, _ = self.fixture(root)
            target.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                resolve_data_alias('external_data/input.csv', root)
            target.unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'Relocated data'):
                resolve_data_alias('external_data/input.csv', root)

    def test_alias_rejects_traversal_and_escaping_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'repo'
            root.mkdir()
            target, _, manifest = self.fixture(root)
            source = json.loads(manifest.read_text())
            source['aliases']['external_data/input.csv']['path'] = '../outside.csv'
            manifest.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                resolve_data_alias('external_data/input.csv', root)
            outside = Path(tmp) / 'outside.csv'
            outside.write_bytes(target.read_bytes())
            target.unlink()
            target.symlink_to(outside)
            source['aliases']['external_data/input.csv']['path'] = 'cells/Example/data/input.csv'
            manifest.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                resolve_data_alias('external_data/input.csv', root)

    def test_required_precomputed_input_never_becomes_empty_trains(self):
        cfg = {'tstart': 0., 'tstop': 10.}
        with tempfile.TemporaryDirectory() as tmp:
            group = {'syns': {'N_syn_resolved': 2}, 'source': {'path': str(Path(tmp) / 'absent.npz')}}
            with self.assertRaisesRegex(FileNotFoundError, 'not replaced with empty trains'):
                _mode_precomputed(cfg, group, None, np.random.default_rng(0))
            group['source']['trains'] = []
            trains = _mode_precomputed(cfg, group, None, np.random.default_rng(0))
            self.assertEqual(len(trains), 2)  # Deliberate empty inline inputs remain valid.
            group['syns']['N_syn_resolved'] = 0
            del group['source']['trains']
            self.assertEqual(_mode_precomputed(cfg, group, None, np.random.default_rng(0)), [])

    def test_published_legacy_aliases_are_byte_exact(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / 'cells/DATA_PATH_ALIASES.json').read_text())
        self.assertEqual(len(manifest['aliases']), 3)
        for source, entry in manifest['aliases'].items():
            target = resolve_data_alias(source, root)
            self.assertEqual(target, root / entry['path'])
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), entry['sha256'])


if __name__ == '__main__':
    unittest.main()
