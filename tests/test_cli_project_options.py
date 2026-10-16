"""Project CLI overrides and identity checks must happen before mutations."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scp_pipeline.cli import main
from scp_pipeline.projects import save_project_settings


class ProjectCLIOptionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'project'
        self.tune = self.root / 'cells/Cell/tunes/orig'
        self.config = self.tune / 'cell_configs/cell_config.json'
        self.config.parent.mkdir(parents=True)
        self.config.write_text(json.dumps({'project_name': 'study', 'cell_loader': 'runtime_cell',
                                          'workspace_layout': 'project_owned',
                                          'workspace_project_root': '../../../..'}))
        save_project_settings(self.root, project_name='study', selected_tune=self.tune,
                              output_dir='remembered-results')

    def invoke(self, command, *arguments, result=0):
        seen = []
        module = SimpleNamespace(main=mock.Mock(side_effect=lambda: seen.append(list(sys.argv)) or result))
        with mock.patch('scp_pipeline.cli.importlib.import_module', return_value=module):
            returned = main([command, '--project-root', str(self.root), *arguments])
        return returned, seen, module

    def test_explicit_output_directory_wins_with_both_argument_styles(self):
        for arguments in (('--output-dir', 'chosen-results'), ('--output-dir=chosen-results',)):
            with self.subTest(arguments=arguments):
                _, seen, _ = self.invoke('run', *arguments)
                self.assertEqual(sum(arg == '--output-dir' or arg.startswith('--output-dir=')
                                     for arg in seen[0]), 1)
                self.assertNotIn(str(self.root / 'remembered-results'), seen[0])

    def test_remembered_output_is_used_only_without_an_explicit_override(self):
        _, seen, _ = self.invoke('run')
        index = seen[0].index('--output-dir')
        self.assertEqual(seen[0][index + 1], str(self.root / 'remembered-results'))

    def test_foreign_project_setup_is_rejected_before_dispatch_or_writes(self):
        self.config.write_text(json.dumps({'project_name': 'another-study'}))
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with mock.patch('scp_pipeline.cli.importlib.import_module') as dispatch:
            with self.assertRaisesRegex(ValueError, 'another project.*no setup operations'):
                main(['setup', '--project-root', str(self.root), '--tune-dir', str(self.tune)])
        dispatch.assert_not_called()
        self.assertEqual(before, {p.relative_to(self.root): p.read_bytes()
                                  for p in self.root.rglob('*') if p.is_file()})

    def test_failed_setup_does_not_write_post_dispatch_metadata(self):
        self.config.write_text(json.dumps({'cell_loader': 'runtime_cell'}))
        before = self.config.read_bytes()
        returned, _, _ = self.invoke('setup', result=3)
        self.assertEqual(returned, 3)
        self.assertEqual(self.config.read_bytes(), before)

    def test_help_does_not_require_matching_workspace_identity(self):
        self.config.write_text(json.dumps({'project_name': 'another-study'}))
        returned, _, module = self.invoke('setup', '--help')
        self.assertEqual(returned, 0)
        module.main.assert_called_once()


if __name__ == '__main__':
    unittest.main()
