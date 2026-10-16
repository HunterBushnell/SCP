"""Step 1 model-selection and quiet-load UI."""

from __future__ import annotations

import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from typing import Any, Sequence

from scp_pipeline.workspaces import discover_cells, discover_projects, discover_tunes

from ..pipeline_workflow import prepare_pipeline_notebook
from .project_setup import build_project_controls, navigation_root, adopt_locations, preview, refresh_summary, _initial_action
from .setup_outputs import guard_draft
from .common import (
    PipelineUIComponent,
    clear_output as _clear_output,
    optional_text as _optional_text,
    quiet_neuron_startup as _quiet_neuron_startup,
)


class Step1UI(PipelineUIComponent):
    def build_panel(self) -> Any:
        if "step1" in self.panels:
            return self.panels["step1"]
        w = self.widgets
        from .name_selector import WorkspaceNameSelector
        projects = [] if self.project_root else discover_projects(self.repo_root)
        preferred_project = str(self.settings.get("project_name") or "")
        if preferred_project and preferred_project not in projects:
            projects = [preferred_project, *projects]
        project = w.Dropdown(
            options=[("Bundled cells", ""), *((name, name) for name in projects)],
            value=preferred_project,
            description="Project",
            layout=w.Layout(width="280px"),
        )
        cells = self._cell_options(self.settings.get("cell_name"))
        cell_value = self.settings.get('cell_name')
        cell = WorkspaceNameSelector(
            options=cells,
            value='' if cell_value is None else str(cell_value or cells[0]),
            description="Cell",
            new_name_option=False,
            layout=w.Layout(width="280px"),
        )
        tunes = self._tune_options(cell.value, self.settings.get("tune_name"))
        tune_value = self.settings.get('tune_name')
        tune = WorkspaceNameSelector(
            options=tunes,
            value='' if tune_value is None else str(tune_value or tunes[0]),
            description="Tune",
            new_name_option=False,
            layout=w.Layout(width="280px"),
        )
        override = w.Text(
            value=_optional_text(self.settings.get("tune_dir_override")),
            description="Tune path",
            placeholder="optional path override",
            layout=w.Layout(width="560px"),
        )
        quiet_output = w.Checkbox(
            value=bool(self.settings.get("quiet_step1_output", True)),
            description="Quiet load",
            indent=False,
            tooltip=(
                "Show only the SCP load summary. Captured setup details remain "
                "available as pipeline_ui.step1_load_log."
            ),
            layout=w.Layout(width="560px"),
        )
        recompile = w.Checkbox(value=bool(self.settings.get('recompile_modfiles', False)),
                              description='Recompile mechanisms', indent=False,
                              tooltip='Build the configured MOD sources locally before loading. Restart after changing mechanisms.')
        loading_toggle = w.ToggleButton(description='Loading options', value=False)
        loading_options = w.VBox([quiet_output, recompile], layout=w.Layout(display='none'))
        loading_toggle.observe(lambda change: setattr(loading_options.layout, 'display', '' if change['new'] else 'none'), names='value')
        run_button = w.Button(
            description="Prepare and load",
            button_style="primary",
            icon="play",
            layout=w.Layout(width="180px"),
        )
        status = w.HTML()
        output = w.Output()

        self.controls.update(
            {
                "project_name": project,
                "cell_name": cell,
                "tune_name": tune,
                "tune_dir_override": override,
                "quiet_step1_output": quiet_output,
                'recompile_modfiles': recompile,
                'loading_options_toggle': loading_toggle,
                'loading_options_box': loading_options,
                "step1_run": run_button,
            }
        )
        self.outputs["step1"] = output
        self.statuses["step1"] = status

        project.observe(self._on_project_changed, names="value")
        cell.observe(self._on_cell_changed, names="value")
        tune.observe(lambda change: self._guard_setup_selection(change, 'tune_name'), names='value')
        override.observe(lambda change: self._guard_setup_selection(change, 'tune_dir_override'), names='value')
        self._observe_value(cell, "cell_name", lambda value: str(value).strip())
        self._observe_value(tune, "tune_name", lambda value: str(value).strip())
        tune.observe(self._release_remembered_tune, names="value")
        self._observe_value(
            override,
            "tune_dir_override",
            lambda value: str(value).strip() or None,
        )
        self._observe_value(quiet_output, "quiet_step1_output", bool)
        self._observe_value(recompile, 'recompile_modfiles', bool)
        run_button.on_click(self._on_prepare)

        project_controls = build_project_controls(self)
        if self.project_root:
            project.layout.display = 'none'
        panel = w.VBox(
            [
                project_controls,
                self._notebook_defaults.build_panel(),
                loading_toggle, loading_options,
                w.HBox([run_button, status]),
                output,
            ]
        )
        self.panels["step1"] = panel
        message = (
            "Complete the new model fields, then Preview model setup and Save cell setup."
            if self.controls['setup_action'].value == 'register'
            else "Choose a cell and tune, then prepare and load it."
        )
        self._set_status("step1", "waiting", message)
        return panel

    def _selected_project(self) -> Any:
        if self.project_root is not None:
            return None
        control = self.controls.get("project_name")
        return (control.value if control is not None else self.settings.get("project_name")) or None

    def _cell_options(self, preferred: Any = None, *, project_name: Any = None) -> list[str]:
        options = discover_cells(navigation_root(self), project_name=project_name or self._selected_project())
        return options or ['']

    def _tune_options(self, cell_name: Any, preferred: Any = None) -> list[str]:
        label = str(cell_name or "").strip()
        if not label:
            return ['']
        options = discover_tunes(navigation_root(self), label, project_name=self._selected_project())
        return options or ['']

    def _on_project_changed(self, change: dict[str, Any]) -> None:
        if self._syncing_model_selection or getattr(self, '_changing_namespace', False):
            return
        if self.project_root is not None:
            self.settings['project_name'] = None
            return
        # Components delegate state to the controller.
        defaults = self._controller.__dict__.get('_notebook_defaults')
        try:
            guard_draft(self)
            if defaults is not None and not defaults.suspended:
                defaults.guard_replace()
            from ..notebook_settings import settings_root
            from scp_pipeline.projects import load_project_settings, resolve_project_path
            from scp_pipeline.workspaces import infer_workspace
            namespace = str(change['new']).strip() or None
            root = settings_root(self.repo_root, project_name=namespace)
            saved = load_project_settings(root)
            selected_path = (resolve_project_path(root, saved['selected_tune'])
                             if saved.get('selected_tune') else None)
            remembered = infer_workspace(selected_path) if selected_path else None
        except (ValueError, OSError) as exc:
            self._changing_namespace = True
            try:
                self.controls['project_name'].value = change.get('old', '')
            finally:
                self._changing_namespace = False
            self._show_validation_error('setup_open', exc, output_key='setup_open')
            return
        self.settings['project_name'] = namespace
        cell = self.controls.get("cell_name")
        if cell is None:
            return
        # Apply one navigation change atomically. Widget option refreshes must
        # not recursively reopen projects or let a previous remembered path win.
        self._syncing_model_selection = True
        try:
            override = self.controls['tune_dir_override']
            explicit = override.value.strip()
            if explicit == self._remembered_tune_override:
                explicit = ''
            self._remembered_tune_override = None
            options = self._cell_options(project_name=namespace)
            preferred_cell = remembered.cell_name if remembered and not explicit else cell.value
            cell.options = options
            if preferred_cell and preferred_cell not in options:
                preferred_cell = 'PV' if 'PV' in options else options[0]
            cell.value = preferred_cell or ''
            tune = self.controls['tune_name']
            options = self._tune_options(cell.value)
            preferred_tune = remembered.tune_name if remembered and not explicit else tune.value
            tune.options = options
            if preferred_tune and preferred_tune not in options:
                preferred_tune = 'tuned' if 'tuned' in options else options[0]
            tune.value = preferred_tune or ''
            override.value = explicit or (str(selected_path) if selected_path else '')
            if not explicit:
                self._remembered_tune_override = override.value or None
            self.settings.update(cell_name=cell.value, tune_name=tune.value,
                                 tune_dir_override=override.value or None)
            self.workspace_root = saved.get('workspace_root') or '.'
            if 'project_root' in self.controls:
                self.controls['project_root'].value = str(root)
                self.controls['workspace_root'].value = self.workspace_root
            if 'project_label' in self.controls:
                self.controls['project_label'].value = saved['project_name']
            output = saved.get('output_dir')
            self.settings['output_dir'] = str(resolve_project_path(root, output)) if output else None
            if 'output_dir' in self.controls:
                self.controls['output_dir'].value = self.settings['output_dir'] or ''
        finally:
            self._syncing_model_selection = False
        if defaults is not None and defaults.enabled and not defaults.suspended:
            defaults.restore_scope()
        if 'setup_action' in self.controls:
            action = _initial_action(self)
            self.controls['setup_action'].value = action
            if action != 'existing':
                self.controls['setup_manage_toggle'].value = True
        refresh_summary(self)

    def _release_remembered_tune(self, _change: dict[str, Any]) -> None:
        """Selectors replace restored navigation, but never a deliberate path override."""
        if self._syncing_model_selection:
            return
        owner = _change.get('owner')
        if owner is not None and owner.value != _change.get('new'):
            return  # Rejected navigation must not discard its explicit path.
        override = self.controls.get('tune_dir_override')
        if override is not None and override.value == self._remembered_tune_override:
            override.value = ''
        self._remembered_tune_override = None

    def _guard_setup_selection(self, change, key):
        if self._syncing_model_selection or getattr(self, '_guarding_setup_draft', False):
            return True
        action = self.controls.get('setup_action')
        if action is None or action.value != 'existing':
            return True
        try:
            guard_draft(self)
        except ValueError as exc:
            self._guarding_setup_draft = True
            self._syncing_model_selection = True
            try:
                self.controls[key].value = change.get('old', '')
            finally:
                self._syncing_model_selection = False
                self._guarding_setup_draft = False
            self._set_status('setup_preview', 'error', str(exc))
            return False
        return True

    def _on_cell_changed(self, change: dict[str, Any]) -> None:
        if self._syncing_model_selection:
            return
        if not self._guard_setup_selection(change, 'cell_name'):
            return
        self._release_remembered_tune(change)
        tune = self.controls.get("tune_name")
        if tune is None:
            return
        options = self._tune_options(change["new"])
        action = self.controls.get('setup_action')
        if action is not None and action.value in {'register', 'live'} and tune.value:
            # Typing a new cell name must not erase the new tune name on each keystroke.
            tune.options = options
            return
        tune.options = options
        if tune.value not in options:
            if tune.value and change['new']:
                tune.value = "tuned" if "tuned" in options else options[0]
            else:
                tune.value = ''

    def _on_prepare(self, _button: Any) -> None:
        if self._setup_attempted:
            self._set_status(
                "step1",
                "warning",
                "Step 1 already ran. Restart the kernel to load another model.",
            )
            return
        try:
            adopt_locations(self)
            if self.controls['setup_action'].value == 'register':
                raise ValueError('Use Preview model setup and Save cell setup before preparing a newly registered model.')
            guard_draft(self)
            cell_name = str(self.controls["cell_name"].value).strip()
            tune_name = str(self.controls["tune_name"].value).strip()
            override = str(self.controls["tune_dir_override"].value).strip() or None
            if not override and (not cell_name or not tune_name):
                raise ValueError("Cell and tune are required when no tune path is supplied.")
            # File/config mistakes must remain recoverable in both workspace
            # modes, before construction choices are locked or NEURON is entered.
            preview(self)
        except Exception as exc:
            self._show_validation_error("step1", exc)
            return

        quiet_output = bool(self.controls["quiet_step1_output"].value)
        self.settings["quiet_step1_output"] = quiet_output
        model_settings = {
            "project_name": self._selected_project(),
            "cell_name": cell_name,
            "tune_name": tune_name,
            "tune_dir_override": override,
            "recompile_modfiles": bool(self.settings.get("recompile_modfiles", False)),
        }
        self.settings.update(deepcopy(model_settings))
        if self.settings.get('output_dir'):
            model_settings['output_dir'] = self.settings['output_dir']
        if self.project_root is not None:
            model_settings.update(project_root=self.project_root, workspace_root=self.workspace_root,
                                  output_dir=self.settings.get('output_dir'))
        if self.cell_override is not None and self.controls['setup_action'].value == 'live':
            model_settings['cell_override'] = self.cell_override
            model_settings['runtime_sim_config'] = self.runtime_sim_config
        self._setup_attempted = True
        self._lock_model_controls()
        self._set_status("step1", "running", "Preparing and loading the tune…")
        load_error = None
        with self.outputs["step1"]:
            _clear_output()
            captured_stdout = StringIO()
            captured_stderr = StringIO()
            captured_warnings: list[Any] = []
            try:
                if quiet_output:
                    with (
                        _quiet_neuron_startup(),
                        warnings.catch_warnings(record=True) as captured_warnings,
                        redirect_stdout(captured_stdout),
                        redirect_stderr(captured_stderr),
                    ):
                        warnings.simplefilter("always")
                        self.pipeline_state = prepare_pipeline_notebook(
                            repo_root=self.repo_root,
                            **model_settings,
                        )
                    self.step1_load_log = self._format_step1_load_log(
                        captured_stdout,
                        captured_stderr,
                        captured_warnings,
                    )
                else:
                    self.step1_load_log = ""
                    self.pipeline_state = prepare_pipeline_notebook(
                        repo_root=self.repo_root,
                        **model_settings,
                    )
                self._print_step1_load_summary(quiet_output=quiet_output)
            except Exception as exc:
                load_error = exc
                if quiet_output:
                    self.step1_load_log = self._format_step1_load_log(
                        captured_stdout,
                        captured_stderr,
                        captured_warnings,
                        traceback_text=traceback.format_exc(),
                    )
                    print(f"{type(exc).__name__}: {exc}")
                else:
                    traceback.print_exc()
                print(
                    "Step 1 may have partially loaded NEURON model state. Restart "
                    "the kernel before trying Step 1 again."
                )
        # Publish outside Output: live Jupyter capture must not hide the reason
        # setup buttons are now locked, including a suppressed native load error.
        if load_error is not None:
            self._restart_required = True
            self._set_status(
                "step1", "error",
                f"Load failed ({type(load_error).__name__}: {load_error}); "
                "restart the kernel before retrying.",
            )
            self._refresh_button_states()
            return

        self._loaded_model_settings = {key: deepcopy(value) for key, value in model_settings.items()
                                       if key not in {'cell_override', 'runtime_sim_config'}}
        try:
            self._refresh_connections()
            self._refresh_input_preview_group_options()
            self._refresh_simulation_controls()
            self._refresh_passive_target_controls()
            self._refresh_active_target_labels()
            self._refresh_act_inspection()
            self._notebook_defaults.watch()
        except Exception as exc:
            self.step1_load_log += '\n[notebook controls]\n' + traceback.format_exc()
            self._restart_required = True
            self.outputs['step1'].append_stdout(
                f'Cell loaded, but refreshing notebook controls failed: {type(exc).__name__}: {exc}\n')
            self._set_status('step1', 'error',
                f'Cell loaded, but notebook controls failed ({exc}). Restart the kernel; do not recreate the workspace.')
            self._refresh_button_states()
            return
        self._set_status(
            "step1",
            "complete",
            f"Ready: {self.pipeline_state.context.cell_name} / "
            f"{self.pipeline_state.context.tune_name}",
        )
        self._refresh_button_states()

        if self.cell_override is not None:
            with self.outputs['step1']:
                print('Live cell adopted: Steps 2–5 execute in this kernel. Saved results are session-only, not reconstructible.')

    @staticmethod
    def _format_step1_load_log(
        captured_stdout: StringIO,
        captured_stderr: StringIO,
        captured_warnings: Sequence[Any],
        *,
        traceback_text: str = "",
    ) -> str:
        parts: list[str] = []
        stdout_text = captured_stdout.getvalue().strip()
        stderr_text = captured_stderr.getvalue().strip()
        warning_text = "".join(
            warnings.formatwarning(
                warning.message,
                warning.category,
                warning.filename,
                warning.lineno,
            )
            for warning in captured_warnings
        ).strip()
        for label, value in (
            ("stdout", stdout_text),
            ("stderr", stderr_text),
            ("warnings", warning_text),
            ("traceback", traceback_text.strip()),
        ):
            if value:
                parts.append(f"[{label}]\n{value}")
        return "\n\n".join(parts)

    def _print_step1_load_summary(self, *, quiet_output: bool) -> None:
        state = self.pipeline_state
        context = state.context
        cell_config = dict(getattr(context, "cell_config", {}) or {})
        print("Pipeline tune loaded")
        print("  Cell:", context.cell_name)
        print("  Project:", getattr(context, "project_name", None) or "Bundled cells")
        print("  Tune:", context.tune_name)
        print("  Tune directory:", state.tune_dir)
        print("  Cell loader:", cell_config.get("cell_loader", "unknown"))
        loader = cell_config.get('cell_loader')
        if getattr(state, 'runtime_only', False):
            print('  Model source:', 'supplied in-memory cell (session-only)')
        elif loader in {'hoc_template', 'python_factory', 'allen_manifest'}:
            role = 'manifest' if loader == 'allen_manifest' else loader
            if (cell_config.get('paths') or {}).get(role):
                from scp_pipeline.loaders.paths import resolve_loader_path
                from scp_pipeline.setup.mechanisms import resolve_modfiles_dir
                print('  Model source:', resolve_loader_path(
                    cell_config, role, base_dir=state.tune_dir, loader_name=loader))
                options = cell_config.get(loader) or {}
                if options.get('template_name') or options.get('callable'):
                    print('  Template / callable:', options.get('template_name') or options.get('callable'))
                print('  Source mode:', (cell_config.get('model_source') or {}).get('mode', 'tune-local'))
                print('  MOD sources:', resolve_modfiles_dir(state.tune_dir, cell_config) or 'built-in mechanisms only')
        multiplier = (cell_config.get("tuning", {}) or {}).get(
            "soma_diam_multiplier"
        )
        if multiplier is not None:
            print("  Soma diameter multiplier:", multiplier)
        validation = getattr(context, "validation", None)
        if validation is not None:
            print(
                "  Step 1 validation:",
                "ok" if getattr(validation, "ok", False) else "check required",
            )
        mechanism_status = dict(
            getattr(state, "mechanism_summary", {}) or {}
        ).get("status")
        if mechanism_status:
            print("  Mechanisms:", mechanism_status)
        cell_class = type(state.cell)
        print("  Shared cell type:", f"{cell_class.__module__}.{cell_class.__name__}")
        if quiet_output and self.step1_load_log:
            print("  Setup details: hidden in pipeline_ui.step1_load_log")

    def sync_from_settings(self, *, act_settings_changed: bool = False) -> None:
        del act_settings_changed
        keys = ("project_name", "cell_name", "tune_name", "tune_dir_override")
        requested = {key: self.settings.get(key) for key in keys}
        self._syncing_model_selection = True
        try:
            project = self.controls.get("project_name")
            if project is not None and not project.disabled:
                desired = str(requested["project_name"] or "")
                projects = [] if self.project_root else discover_projects(self.repo_root)
                if desired and desired not in projects:
                    projects = [desired, *projects]
                project.options = [("Bundled cells", ""), *((name, name) for name in projects)]
                project.value = desired
            cell = self.controls.get('cell_name')
            if cell is not None and not cell.disabled:
                cell.options = self._cell_options(requested['cell_name'])
                cell.value = _optional_text(requested['cell_name'])
            tune = self.controls.get('tune_name')
            if tune is not None and not tune.disabled:
                tune.options = self._tune_options(requested['cell_name'], requested['tune_name'])
                tune.value = _optional_text(requested['tune_name'])
            override = self.controls.get('tune_dir_override')
            if override is not None and not override.disabled:
                override.value = _optional_text(requested['tune_dir_override'])
        finally:
            self._syncing_model_selection = False
        # Widget cascades may temporarily choose a fallback cell/tune. Explicit
        # code-edited selection remains authoritative when synchronizing it.
        self.settings.update(requested)
        quiet = self.controls.get("quiet_step1_output")
        if quiet is not None and not quiet.disabled:
            quiet.value = bool(self.settings.get("quiet_step1_output", True))
        recompile = self.controls.get('recompile_modfiles')
        if recompile is not None and not recompile.disabled:
            recompile.value = bool(self.settings.get('recompile_modfiles', False))
        refresh_summary(self)

    def refresh_button_states(self, *, ready: bool, act_busy: bool) -> None:
        del ready, act_busy
        button = self.controls.get("step1_run")
        if button is not None:
            button.disabled = self._setup_attempted


__all__ = ["Step1UI"]
