"""A failed native HOC load must never reach an unsafe partial constructor."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scp_pipeline.loaders import hoc_template


class HocLoadFailureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'Cell.hoc'
        self.path.write_text('begintemplate SyntheticPartial\nendtemplate SyntheticPartial\n')
        for key, value in (
            ('_LOADED_HOC_SOURCES', set()), ('_LOADED_HOC_HASHES', {}),
            ('_LOADED_TEMPLATE_SOURCES', {}), ('_FAILED_HOC_SOURCES', set()),
        ):
            patcher = mock.patch.object(hoc_template, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_failed_load_with_partial_symbol_is_not_cached_or_retried(self):
        runtime = SimpleNamespace()
        constructor = mock.Mock(side_effect=AssertionError('Do not instantiate'))
        def failed_load(_path):
            runtime.SyntheticPartial = constructor
            return 0.0
        runtime.load_file = mock.Mock(side_effect=failed_load)
        for attempt in range(2):
            with self.assertRaisesRegex(RuntimeError, 'paths.modfiles.*Restart'):
                hoc_template._load_template_definition(runtime, self.path, 'SyntheticPartial')
        runtime.load_file.assert_called_once_with(str(self.path))
        constructor.assert_not_called()
        self.assertIn(self.path, hoc_template._FAILED_HOC_SOURCES)
        self.assertNotIn(self.path, hoc_template._LOADED_HOC_SOURCES)
        self.assertNotIn(self.path, hoc_template._LOADED_HOC_HASHES)
        self.assertNotIn('SyntheticPartial', hoc_template._LOADED_TEMPLATE_SOURCES)

    def test_native_exception_is_preserved_with_context_and_restart_guard(self):
        cause = ValueError('native parse error')
        runtime = SimpleNamespace(load_file=mock.Mock(side_effect=cause))
        with self.assertRaisesRegex(RuntimeError, 'Cell.hoc.*SyntheticPartial') as caught:
            hoc_template._load_template_definition(runtime, self.path, 'SyntheticPartial')
        self.assertIs(caught.exception.__cause__, cause)
        with self.assertRaisesRegex(RuntimeError, 'previously failed.*Restart'):
            hoc_template._load_template_definition(runtime, self.path, 'SyntheticPartial')
        runtime.load_file.assert_called_once()

    def test_successfully_loaded_source_can_correct_a_missing_template_selection(self):
        runtime = SimpleNamespace()
        def successful_load(_path):
            runtime.SyntheticPartial = mock.Mock()
            return 1.0
        runtime.load_file = mock.Mock(side_effect=successful_load)
        with self.assertRaisesRegex(AttributeError, 'WrongTemplate'):
            hoc_template._load_template_definition(runtime, self.path, 'WrongTemplate')
        hoc_template._load_template_definition(runtime, self.path, 'SyntheticPartial')
        runtime.load_file.assert_called_once()
        self.assertNotIn(self.path, hoc_template._FAILED_HOC_SOURCES)

    def test_missing_mechanism_survives_in_real_neuron_fresh_process(self):
        fixture = Path(__file__).resolve().parent / 'fixtures/invalid_hoc/MissingMechanism.hoc'
        program = f'''
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except ImportError:
    pass
from neuron import h
from scp_pipeline.model import load_cell
config = {{'cell_loader': 'hoc_template', 'paths': {{'hoc_template': {str(fixture)!r}, 'modfiles': None}},
          'hoc_template': {{'template_name': 'SCPMissingMechanism', 'constructor_args': []}}}}
for _ in range(2):
    try:
        load_cell(config, base_dir={str(fixture.parent)!r})
    except RuntimeError as exc:
        assert 'paths.modfiles' in str(exc) and 'Restart' in str(exc), str(exc)
    else:
        raise AssertionError('Invalid HOC source was accepted')
probe = h.Section(name='still_alive')
probe.insert('pas')
print('PROCESS_SURVIVED_FAILED_HOC_LOAD')
'''
        result = subprocess.run([sys.executable, '-u', '-X', 'faulthandler', '-c', program],
            cwd=self.path.parent, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PROCESS_SURVIVED_FAILED_HOC_LOAD', result.stdout)
        self.assertIn('not a MECHANISM', result.stderr)


if __name__ == '__main__':
    unittest.main()
