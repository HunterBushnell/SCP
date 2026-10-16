"""Public controller facade for the compact SCP pipeline notebook UI."""

from __future__ import annotations

import html
import json
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, MutableMapping, Optional

from scp_pipeline.workspaces import resolve_workspace

from ._pipeline_ui.common import (
    MODEL_SETTING_KEYS,
    PIPELINE_UI_DEFAULTS,
    PipelineUIComponent,
    clear_output as _clear_output,
    import_widgets as _import_widgets,
    parse_float_list as _parse_float_list,
    parse_optional_float as _parse_optional_float,
    parse_optional_int as _parse_optional_int,
)
from ._pipeline_ui.step1 import Step1UI
from ._pipeline_ui.step2 import Step2UI
from ._pipeline_ui.step3 import Step3UI
from ._pipeline_ui.step3_act import Step3ACTUI
from ._pipeline_ui.step4 import Step4UI
from ._pipeline_ui.step5 import Step5UI
from .pipeline_act import (
    PipelineACTEvaluationResult,
    PipelineACTRunResult,
    PipelineACTWorkspaceResult,
)
from .pipeline_workflow import (
    PipelineActiveProtocolResult,
    PipelineFIResult,
    PipelineInputPreviewResult,
    PipelineNotebookState,
    PipelinePassiveProposalResult,
    PipelinePassiveResult,
    PipelineSimulationResult,
    RestartKernelRequired,
)


class PipelineNotebookUI:
    """Public owner of five cached, independent compact-pipeline panels."""

    def __init__(
        self,
        repo_root: str | Path | None = None,
        settings: MutableMapping[str, Any] | None = None,
        *, cell_override: Any = None, runtime_sim_config: Optional[dict[str, Any]] = None,
        project_root: str | Path | None = None, workspace_root: str | Path | None = None,
        restore_notebook_settings: bool = False,
        notebook_settings_path: str | Path | None = None,
    ) -> None:
        settings = {} if settings is None else settings
        if not isinstance(settings, MutableMapping):
            raise TypeError("settings must be a mutable mapping/dict.")
        from scp_pipeline.runtime import code_root
        self.repo_root = Path(repo_root).expanduser().resolve() if repo_root is not None else code_root()
        self.project_root = Path(project_root).expanduser().resolve() if project_root is not None else None
        self.workspace_root = workspace_root
        # Automatically restored navigation is not a user-entered explicit path.
        # Release it when the user chooses another cell, tune, or namespace.
        self._remembered_tune_override = None
        self._syncing_model_selection = False
        if self.project_root is not None:
            from scp_pipeline.projects import PROJECT_FILE, load_project_settings
            saved = load_project_settings(self.project_root)
            settings.setdefault('cell_name', '')
            settings.setdefault('tune_name', 'orig')
            settings['project_name'] = None  # Namespace dropdown is only for SCP-owned workspaces.
            if (self.project_root / PROJECT_FILE).is_file():
                settings['tune_dir_override'] = saved.get('selected_tune')
                self._remembered_tune_override = settings['tune_dir_override']
                settings['output_dir'] = saved.get('output_dir')
                self.workspace_root = workspace_root or saved.get('workspace_root')
                if saved.get('selected_tune'):
                    from scp_pipeline.projects import resolve_project_path
                    from scp_pipeline.workspaces import infer_workspace
                    selected = infer_workspace(resolve_project_path(self.project_root, saved['selected_tune']))
                    settings.update(cell_name=selected.cell_name, tune_name=selected.tune_name)
        elif restore_notebook_settings:
            from .notebook_settings import settings_root
            from scp_pipeline.projects import PROJECT_FILE, load_project_settings, resolve_project_path
            root = settings_root(self.repo_root, project_name=settings.get('project_name'))
            if (root / PROJECT_FILE).is_file():
                saved = load_project_settings(root)
                if settings.get('project_name') is None and saved.get('selected_project'):
                    # Navigation remembers the SCP namespace separately from
                    # the bundled namespace's own selected tune.
                    settings['project_name'] = saved['selected_project']
                    root = settings_root(self.repo_root, project_name=saved['selected_project'])
                    saved = load_project_settings(root)
                if saved.get('selected_tune'):
                    selected_path = resolve_project_path(root, saved['selected_tune'])
                    from scp_pipeline.workspaces import infer_workspace
                    selected = infer_workspace(selected_path)
                    settings['tune_dir_override'] = str(selected_path)
                    self._remembered_tune_override = settings['tune_dir_override']
                    settings.update(cell_name=selected.cell_name, tune_name=selected.tune_name)
                if saved.get('output_dir'):
                    settings['output_dir'] = str(resolve_project_path(root, saved['output_dir']))
        self.settings = settings
        self.cell_override = cell_override
        self.runtime_sim_config = runtime_sim_config
        self._fill_defaults(self.settings)
        self.widgets = _import_widgets()

        self.pipeline_state: Optional[PipelineNotebookState] = None
        self.passive_proposal_result: Optional[PipelinePassiveProposalResult] = None
        self.passive_result: Optional[PipelinePassiveResult] = None
        self.active_result: Optional[PipelineActiveProtocolResult] = None
        self.fi_result: Optional[PipelineFIResult] = None
        self.act_workspace_result: Optional[PipelineACTWorkspaceResult] = None
        self.act_run_result: Optional[PipelineACTRunResult] = None
        self.act_predictions: dict[str, float] = {}
        self.act_evaluation_result: Optional[PipelineACTEvaluationResult] = None
        self.act_job: Any = None
        self.act_log = ""
        self.tuner: Any = None
        self.input_preview_result: Optional[PipelineInputPreviewResult] = None
        self.simulation_result: Optional[PipelineSimulationResult] = None
        self.diagnostics: Any = None
        self.step1_load_log = ""
        self.input_preview_log = ""
        self.simulation_log = ""

        self.panels: dict[str, Any] = {}
        self.controls: dict[str, Any] = {}
        self.outputs: dict[str, Any] = {}
        self.statuses: dict[str, Any] = {}
        self._status_kinds: dict[str, str] = {}
        self._setup_attempted = False
        self._restart_required = False
        self._loaded_model_settings: Optional[dict[str, Any]] = None
        self._syncing_passive_targets = False
        self._syncing_active_current_amp = False
        self._syncing_input_preview_options = False
        self._syncing_simulation_options = False
        self._syncing_diagnostic_options = False
        target_overrides = dict(
            self.settings.get("passive_target_overrides", {}) or {}
        )
        self._passive_target_dirty = {
            field
            for field in (
                "target_rin_mohm",
                "target_tau_ms",
                "target_v_rest_mv",
            )
            if target_overrides.get(field) is not None
        }
        self._passive_target_source_mode = "none"
        self._act_thread: Any = None
        self._act_cancel_requested = False
        self._act_syncing = False
        self._act_conductance_controls: list[dict[str, Any]] = []
        self._act_inspection: Any = None
        self._applied_act_settings_signature = self._act_settings_signature(
            self.settings
        )

        self._components: dict[str, PipelineUIComponent] = {
            "step1": Step1UI(self),
            "step2": Step2UI(self),
            "step3": Step3UI(self),
            "step3_act": Step3ACTUI(self),
            "step4": Step4UI(self),
            "step5": Step5UI(self),
        }
        from ._pipeline_ui.notebook_defaults import NotebookDefaultsUI
        self._notebook_defaults = NotebookDefaultsUI(
            self, enabled=restore_notebook_settings, path=notebook_settings_path)
        self._notebook_defaults.initialize()

    def __getattr__(self, name: str) -> Any:
        """Keep private callback access compatible while components stay private."""

        components = self.__dict__.get("_components", {})
        for component in components.values():
            descriptor = getattr(type(component), name, None)
            if descriptor is not None:
                return descriptor.__get__(component, type(component))
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    @staticmethod
    def _fill_defaults(settings: MutableMapping[str, Any]) -> None:
        for key, value in PIPELINE_UI_DEFAULTS.items():
            settings.setdefault(key, deepcopy(value))

    @staticmethod
    def _act_settings_signature(settings: MutableMapping[str, Any]) -> str:
        return json.dumps(
            {
                "workspace": settings.get("act_workspace_override"),
                "overrides": settings.get("act_overrides") or {},
            },
            sort_keys=True,
            default=str,
        )

    def apply_settings(self, settings: MutableMapping[str, Any]) -> None:
        """Adopt code-edited settings and refresh all unlocked controls."""

        if not isinstance(settings, MutableMapping):
            raise TypeError("settings must be a mutable mapping/dict.")
        self._fill_defaults(settings)
        ignored: list[str] = []
        if self._loaded_model_settings is not None:
            for key in MODEL_SETTING_KEYS:
                loaded = deepcopy(self._loaded_model_settings[key])
                if settings.get(key) != loaded:
                    ignored.append(key)
                    settings[key] = loaded
        new_act_signature = self._act_settings_signature(settings)
        act_settings_changed = (
            self._applied_act_settings_signature != new_act_signature
        )
        self._applied_act_settings_signature = new_act_signature
        self.settings = settings
        self._sync_controls_from_settings(
            act_settings_changed=act_settings_changed
        )
        defaults = self.__dict__.get('_notebook_defaults')
        if defaults is not None:
            defaults.changed()
            defaults.watch()
        if ignored:
            self._set_status(
                "step1",
                "warning",
                "Model selection is locked. Restart the kernel to change: "
                + ", ".join(ignored),
            )

    def step1_panel(self) -> Any:
        panel = self._components["step1"].build_panel()
        self._notebook_defaults.watch()
        return panel

    def step2_panel(self) -> Any:
        panel = self._components["step2"].build_panel()
        self._notebook_defaults.watch()
        return panel

    def step3_panel(self) -> Any:
        panel = self._components["step3"].build_panel()
        self._notebook_defaults.watch()
        return panel

    def step4_panel(self) -> Any:
        panel = self._components["step4"].build_panel()
        self._notebook_defaults.watch()
        return panel

    def step5_panel(self) -> Any:
        panel = self._components["step5"].build_panel()
        self._notebook_defaults.watch()
        return panel

    def save_notebook_settings(self, path=None):
        """Explicitly persist notebook choices, never model/scientific configs."""
        return self._notebook_defaults.save(path)

    def load_notebook_settings(self, path=None, *, discard=False):
        """Restore validated notebook choices without constructing a model."""
        return self._notebook_defaults.load(path, discard=discard)

    def _run_button(self, description: str, *, icon: str = "play") -> Any:
        return self.widgets.Button(
            description=description,
            button_style="primary",
            icon=icon,
            layout=self.widgets.Layout(width="190px"),
        )

    def _selected_tune_dir(self) -> Path:
        if self.pipeline_state is not None:
            return Path(self.pipeline_state.tune_dir).resolve()
        cell_name = self.settings.get('cell_name')
        tune_name = self.settings.get('tune_name')
        override = self.settings.get('tune_dir_override')
        if not override and not (cell_name and tune_name):
            raise ValueError('Select a Cell and Tune in Step 1, or supply an explicit Tune path.')
        return resolve_workspace(
            repo_root=self.repo_root,
            project_name=self.settings.get("project_name"),
            cell_name=cell_name,
            tune_name=tune_name,
            tune_dir_override=override,
            project_root=self.project_root, workspace_root=self.workspace_root,
        ).tune_dir

    @staticmethod
    def _include_preferred(
        options: list[str], preferred: Any, *, fallback: str
    ) -> list[str]:
        value = str(preferred or "").strip()
        if value and value not in options:
            options = [value, *options]
        if not options:
            options = [value or fallback]
        return options

    def _observe_value(self, widget: Any, key: str, parser: Any) -> None:
        def _sync(change: dict[str, Any]) -> None:
            try:
                # A navigation guard may have restored the old value earlier in
                # this notification; never persist the rejected event's value.
                self.settings[key] = parser(widget.value)
            except (TypeError, ValueError):
                return

        widget.observe(_sync, names="value")

    def _observe_valid_float_list(self, widget: Any, key: str) -> None:
        def _sync(change: dict[str, Any]) -> None:
            try:
                self.settings[key] = _parse_float_list(change["new"], name=key)
            except ValueError:
                return

        widget.observe(_sync, names="value")

    def _observe_valid_optional_int(
        self, widget: Any, key: str, *, positive: bool
    ) -> None:
        def _sync(change: dict[str, Any]) -> None:
            try:
                self.settings[key] = _parse_optional_int(
                    change["new"], name=key, positive=positive
                )
            except ValueError:
                return

        widget.observe(_sync, names="value")

    def _observe_valid_optional_float(self, widget: Any, key: str) -> None:
        def _sync(change: dict[str, Any]) -> None:
            try:
                value = _parse_optional_float(change["new"], name=key)
            except ValueError:
                return
            if value is not None:
                self.settings[key] = value

        widget.observe(_sync, names="value")

    def _observe_valid_optional_float_or_none(self, widget: Any, key: str) -> None:
        def _sync(change: dict[str, Any]) -> None:
            try:
                self.settings[key] = _parse_optional_float(change["new"], name=key)
            except ValueError:
                return

        widget.observe(_sync, names="value")

    @staticmethod
    def _fresh_process_log(result: Any) -> str:
        parts: list[str] = []
        command = tuple(getattr(result, "command", ()) or ())
        stdout_text = str(getattr(result, "stdout", "") or "").strip()
        if command:
            parts.append("[command]\n" + " ".join(str(item) for item in command))
        if stdout_text:
            parts.append("[stdout+stderr]\n" + stdout_text)
        return "\n\n".join(parts)

    def _run_stage(
        self,
        step: str,
        running_message: str,
        operation: Any,
        *,
        output_key: Optional[str] = None,
    ) -> None:
        if self.pipeline_state is None:
            self._show_validation_error(
                step,
                RuntimeError("Complete Step 1 first."),
                output_key=output_key,
            )
            return
        button_keys = {
            "step2": ["step2_proposal", "step2_run"],
            "step3": ["step3_active", "step3_fi"],
            "step4": [
                "step4_initialize",
                "step4_single_event",
                "step4_interactive",
            ],
            "step5": ["step5_check_inputs", "step5_run", "step5_plot"],
        }[step]
        for key in button_keys:
            self.controls[key].disabled = True
        self._set_status(step, "running", running_message)
        with self.outputs[output_key or step]:
            _clear_output()
            try:
                operation()
            except RestartKernelRequired as exc:
                traceback.print_exc()
                self._mark_restart_required(str(exc))
                return
            except Exception as exc:
                traceback.print_exc()
                self._set_status(step, "error", f"{type(exc).__name__}: {exc}")
            else:
                self._set_status(step, "complete", "Complete.")
            finally:
                self._refresh_button_states()

    def _show_validation_error(
        self,
        step: str,
        exc: Exception,
        *,
        output_key: Optional[str] = None,
    ) -> None:
        self._set_status(step, "error", str(exc))
        output = self.outputs.get(output_key or step)
        if output is not None:
            # Validation needs plain text, not a parent-message capture. Setup
            # logs use HTML; rich protocol outputs support the same stream API.
            output.outputs = ({'output_type': 'stream', 'name': 'stdout',
                               'text': f"{type(exc).__name__}: {exc}\n"},)

    def _mark_restart_required(self, message: str) -> None:
        self._restart_required = True
        for step in ("step2", "step3", "step4", "step5"):
            self._set_status(step, "error", "Kernel restart required: " + message)
        self._refresh_button_states()

    def _lock_model_controls(self) -> None:
        for key, control in self.controls.items():
            if key.startswith('setup_') or key in {'workspace_mode', 'project_root', 'workspace_root', 'project_label', 'output_dir'}:
                control.disabled = True
        for key in (
            "project_name",
            "cell_name",
            "tune_name",
            "tune_dir_override",
            "quiet_step1_output",
            'recompile_modfiles',
            "step1_run",
        ):
            control = self.controls.get(key)
            if control is not None:
                control.disabled = True

    def _refresh_button_states(self) -> None:
        ready = self.pipeline_state is not None and not self._restart_required
        act_busy = self._act_thread is not None and self._act_thread.is_alive()
        for component in self._components.values():
            component.refresh_button_states(ready=ready, act_busy=act_busy)

    def _sync_controls_from_settings(
        self, *, act_settings_changed: bool = False
    ) -> None:
        for component in self._components.values():
            component.sync_from_settings(
                act_settings_changed=act_settings_changed
            )
        self._refresh_button_states()

    def _set_status(self, step: str, kind: str, message: str) -> None:
        status = self.statuses.get(step)
        if status is None:
            return
        self._status_kinds[step] = kind
        colors = {
            "waiting": "#666",
            "running": "#8a5a00",
            "complete": "#18743c",
            "warning": "#8a5a00",
            "error": "#b42318",
        }
        labels = {
            "waiting": "Status",
            "running": "Running",
            "complete": "Complete",
            "warning": "Attention",
            "error": "Error",
        }
        status.value = (
            f'<span style="color:{colors.get(kind, "#666")}; margin-left:10px">'
            f"<b>{labels.get(kind, kind.title())}:</b> "
            f"{html.escape(str(message))}</span>"
        )


__all__ = ["PIPELINE_UI_DEFAULTS", "PipelineNotebookUI"]
