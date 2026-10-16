"""Full-list workspace selectors with explicit new-name entry.

Only imported while building the widget UI. The frontend uses stock widgets;
unlike a browser-backed Combobox, selecting a value never filters its menu.
"""
from __future__ import annotations

import ipywidgets as widgets
from ipywidgets.widgets.trait_types import TypedTuple
from traitlets import Bool, Unicode, observe


class WorkspaceNameSelector(widgets.VBox):
    """Keep the existing value/options API without an implicit text filter."""

    value = Unicode()
    options = TypedTuple(Unicode(), default_value=())
    description = Unicode()
    disabled = Bool(False)
    allow_new = Bool(False)
    show_name_editor = Bool(False)
    new_name_option = Bool(True)

    def __init__(self, *, options=(), value='', description='', **kwargs):
        self._updating = True
        self._editing_new = False
        self.dropdown = widgets.Dropdown(layout=widgets.Layout(width='100%'))
        self.editor = widgets.Text(layout=widgets.Layout(width='100%', display='none'))
        super().__init__(children=(self.dropdown, self.editor), options=options,
                         value=value, description=description, **kwargs)
        self.dropdown.observe(self._choose, names='value')
        self.editor.observe(self._edit, names='value')
        self._updating = False
        self._render()

    @observe('value')
    def _value_changed(self, _change):
        if not self._updating:
            self._editing_new = False
            self._render()

    @observe('options', 'description', 'disabled', 'allow_new', 'show_name_editor', 'new_name_option', 'tooltip')
    def _options_changed(self, _change):
        if not self._updating:
            self._render()

    def _render(self):
        if self._updating:
            return
        self._updating = True
        try:
            editing = self.allow_new and (
                self._editing_new or (self.value and self.value not in self.options)
                or not any(self.options))
            placeholder = f'Select {self.description.lower()}…'
            entries = [(name or placeholder, name) for name in self.options]
            if self.allow_new and self.new_name_option:
                entries.append(('Enter new name…', None))
            selected = None if editing else self.value
            if not self.new_name_option and selected not in self.options:
                selected = ''
            if selected not in [value for _label, value in entries]:
                # Preserve explicit code-first values even before discovery.
                entries.append((selected or placeholder, selected))
            self.dropdown.options = entries
            self.dropdown.value = selected
            self.dropdown.description = self.description
            self.dropdown.tooltip = self.tooltip
            self.dropdown.disabled = self.disabled
            self.editor.value = self.value
            self.editor.description = f'New {self.description.lower()}'
            self.editor.placeholder = f'Type a new {self.description.lower()} name'
            self.editor.disabled = self.disabled
            self.editor.layout.display = '' if self.allow_new and (editing or self.show_name_editor) else 'none'
        finally:
            self._updating = False

    def _set_value(self, value, *, editing=False):
        self._updating = True
        try:
            self.value = value
            self._editing_new = editing
        finally:
            self._updating = False
        self._render()

    def _choose(self, change):
        if not self._updating:
            self._set_value(change['new'] or '', editing=change['new'] is None)

    def _edit(self, change):
        if not self._updating:
            self._set_value(change['new'], editing=True)
