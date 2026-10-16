"""Exercise real kernel comm events, not only Python-side Button.click()."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from queue import Empty
import sys
import tempfile
import time
import unittest

from scp_pipeline.notebooks._pipeline_ui.text_output import TextOutput


class TextOutputTests(unittest.TestCase):
    def test_log_uses_only_stock_html_state_and_escapes_source_text(self):
        output = TextOutput()
        self.addCleanup(output.close)
        output.append_stdout('<script>not executable</script>\n')
        output.append_stderr('an error & details\n')
        self.assertIn('&lt;script&gt;', output.value)
        self.assertIn('error &amp; details', output.value)
        self.assertNotIn('<script>', output.value)
        self.assertEqual(output._model_name, 'HTMLModel')
        self.assertEqual(output._model_module, '@jupyter-widgets/controls')
        self.assertNotIn('outputs', output.get_state())
        self.assertNotIn('msg_id', output.get_state())
        self.assertEqual([record['name'] for record in output.outputs], ['stdout', 'stderr'])
        output.clear_output(wait=True)
        self.assertEqual(output.outputs, ())
        self.assertEqual(output.value, '')


@unittest.skipUnless(importlib.util.find_spec('ipykernel') and importlib.util.find_spec('jupyter_client'),
                     'Real widget comm checks require ipykernel and jupyter_client')
class SetupWidgetCommTests(unittest.TestCase):
    def setUp(self):
        from jupyter_client import KernelManager

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.entry = self.root / 'native/Cells.hoc'
        self.entry.parent.mkdir()
        self.entry.write_text(''.join(
            f'begintemplate {name}\npublic soma\ncreate soma\n'
            f'proc init() {{ soma {{ insert pas }} }}\nendtemplate {name}\n'
            for name in ('SyntheticFirst', 'SyntheticSecond')))
        self.native_bytes = self.entry.read_bytes()
        self.manager = KernelManager()
        # Use this test environment, never an existing notebook's kernel.
        self.manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        self.manager.start_kernel(cwd=str(self.root), env=dict(
            os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg'))
        self.addCleanup(self.manager.shutdown_kernel, now=True)
        self.client = self.manager.client()
        self.client.start_channels()
        self.addCleanup(self.client.stop_channels)
        self.client.wait_for_ready(timeout=30)
        self.messages = []
        messages = self.execute(f'''
import json
from IPython.display import display
from scp_pipeline.notebooks import PipelineNotebookUI
ui = PipelineNotebookUI(project_root={str(self.root)!r})
panel = ui.step1_panel()
c = ui.controls
c['cell_name'].editor.value = 'Cell'
c['tune_name'].editor.value = 'orig'
c['setup_voltage'].value = '-65'
c['setup_celsius'].value = '30'
def descendants(widget):
    yield widget
    for child in getattr(widget, 'children', ()):
        yield from descendants(child)
advanced = next(widget for widget in descendants(panel)
                if getattr(widget, 'description', '') == 'Advanced model settings')
advanced_box = next(widget for widget in descendants(panel)
                    if advanced in getattr(widget, 'children', ()))
ids = {{key: widget.model_id for key, widget in c.items()}}
ids.update({{key + '_status': widget.model_id for key, widget in ui.statuses.items()}})
ids.update(advanced=advanced.model_id, advanced_layout=advanced_box.children[1].layout.model_id,
           target_layout=c['setup_target_form'].layout.model_id)
display(panel)
print('WIDGET_IDS:' + json.dumps(ids))
''')
        self.ids = json.loads(''.join(message['content']['text'] for message in messages
                                     if message['msg_type'] == 'stream').split('WIDGET_IDS:')[-1])
        self.output_ids = {message['content']['comm_id'] for message in self.messages
                           if message['msg_type'] == 'comm_open'
                           and message['content'].get('data', {}).get('state', {}).get('_model_module')
                           == '@jupyter-widgets/output'}

    def receive_until_idle(self, msg_id):
        messages = []
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                message = self.client.get_iopub_msg(timeout=1)
            except Empty:
                continue
            self.messages.append(message)
            if message.get('parent_header', {}).get('msg_id') != msg_id:
                continue
            messages.append(message)
            if message['msg_type'] == 'error':
                self.fail('\n'.join(message['content']['traceback']))
            if message['msg_type'] == 'status' and message['content']['execution_state'] == 'idle':
                return messages
        self.fail('Disposable test kernel did not return to idle')

    def execute(self, source):
        return self.receive_until_idle(self.client.execute(source))

    def event(self, key, *, state=None):
        data = ({'method': 'custom', 'content': {'event': 'click'}} if state is None
                else {'method': 'update', 'state': state, 'buffer_paths': []})
        message = self.client.session.msg('comm_msg', {'comm_id': self.ids[key], 'data': data})
        self.client.shell_channel.send(message)
        messages = self.receive_until_idle(message['header']['msg_id'])
        # OutputModel updates trigger VS Code Jupyter's special full-handle
        # acknowledgement path. Setup must not enter it, even without capture.
        self.assertFalse([message for message in messages if message['msg_type'] == 'comm_msg'
                          and message['content']['comm_id'] in self.output_ids])
        return messages

    def assert_update(self, messages, key, field, expected):
        states = [message['content']['data'].get('state', {}) for message in messages
                  if message['msg_type'] == 'comm_msg'
                  and message['content']['comm_id'] == self.ids[key]]
        self.assertTrue(any(state.get(field) == expected for state in states), (key, field, states))

    def assert_complete(self, messages, key):
        values = [message['content']['data'].get('state', {}).get('value', '') for message in messages
                  if message['msg_type'] == 'comm_msg'
                  and message['content']['comm_id'] == self.ids[key + '_status']]
        self.assertTrue(any('Complete:' in value and '#18743c' in value for value in values), values)

    def test_discovery_selection_toggles_save_and_copy_keep_publishing_updates(self):
        # Recover from an incorrect click, repeat discovery, then select through
        # the actual Dropdown frontend protocol (index, not Python-only value).
        self.event('setup_discover')
        self.event('setup_entry', state={'value': str(self.entry)})
        for _ in range(2):
            self.assert_complete(self.event('setup_discover'), 'setup_discover')
        self.assert_update(self.event('setup_candidates', state={'index': 2}),
                           'setup_template', 'value', 'SyntheticSecond')
        self.assert_update(self.event('advanced', state={'value': True}), 'advanced_layout', 'display', '')
        self.assert_update(self.event('setup_targets_enabled', state={'value': True}), 'target_layout', 'display', '')
        self.assert_complete(self.event('setup_preview'), 'setup_preview')
        self.assertFalse((self.root / 'cells').exists())
        self.assert_complete(self.event('setup_save'), 'setup_save')
        self.assertTrue((self.root / 'cells/Cell/tunes/orig/cell_configs/target_config.json').is_file())
        self.event('setup_copy_name', state={'value': '../unsafe'})
        self.event('setup_copy_preview')
        self.event('setup_copy_name', state={'value': 'tuned'})
        for _ in range(2):
            self.assert_complete(self.event('setup_copy_preview'), 'setup_copy_preview')
        self.assert_complete(self.event('setup_copy_create'), 'setup_copy_create')
        self.assertTrue((self.root / 'cells/Cell/tunes/tuned/cell_configs/cell_config.json').is_file())
        self.assert_complete(self.event('setup_project_save'), 'setup_project_save')
        self.assert_complete(self.event('notebook_save'), 'notebook_save')
        self.execute('''
assert not ui._setup_attempted
assert ui.pipeline_state is None
assert not c['step1_run'].disabled
assert c['setup_template'].value == 'SyntheticSecond'
if 'neuron' in __import__('sys').modules:
    from neuron import h
    assert not list(h.allsec())
    assert not hasattr(h, 'SyntheticFirst') and not hasattr(h, 'SyntheticSecond')
''')
        self.assertEqual(self.entry.read_bytes(), self.native_bytes)
