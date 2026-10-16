"""Plain setup logs without Jupyter Output's frontend execution machinery."""
from __future__ import annotations

from html import escape

from ipywidgets import HTML
from ipywidgets.widgets.trait_types import TypedTuple
from traitlets import Dict, observe


class TextOutput(HTML):
    """An escaped, selectable text log rendered by the stock HTML widget.

    File-only actions need neither rich display nor output capture. In some
    VS Code Jupyter versions, updates to OutputModel enter a broken message
    acknowledgement path and stall *all* subsequent widget updates. Using HTML
    here avoids that path altogether, not just Output's context manager.

    Keep the small stream-log API used by controllers and diagnostics. The
    Python-only ``outputs`` trait is deliberately not synchronized; only the
    ordinary HTML ``value`` crosses the widget comm.
    """

    outputs = TypedTuple(Dict(), default_value=())
    msg_id = ''  # No capture context or parent-message subscription.

    @observe('outputs')
    def _render_text(self, change):
        text = ''.join(record['text'] for record in change['new'])
        self.value = (
            '<pre style="white-space:pre-wrap;overflow-wrap:anywhere;'
            'color:inherit;background:transparent;margin:0">'
            + escape(text) + '</pre>'
        ) if text else ''

    def append_stdout(self, text):
        self._append_stream('stdout', text)

    def append_stderr(self, text):
        self._append_stream('stderr', text)

    def _append_stream(self, name, text):
        self.outputs = (*self.outputs, {'output_type': 'stream', 'name': name, 'text': str(text)})

    def clear_output(self, *args, **kwargs):
        self.outputs = ()
