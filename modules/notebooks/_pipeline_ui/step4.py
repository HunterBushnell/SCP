"""Step 4 BMTool initialization and tuner UI."""

from __future__ import annotations

from typing import Any, Optional

from ..pipeline_workflow import prepare_interactive_synapse_tuner
from .common import PipelineUIComponent


class Step4UI(PipelineUIComponent):
    def build_panel(self) -> Any:
        if "step4" in self.panels:
            return self.panels["step4"]
        w = self.widgets
        enabled = w.Checkbox(
            value=bool(self.settings.get("enable_synapse_tuning")),
            description="Enable BMTool synapse tuning",
            indent=False,
            layout=w.Layout(width="260px"),
        )
        connection = w.Dropdown(
            options=[("Load Step 1 to discover connections", "")],
            value="",
            description="Connection",
            layout=w.Layout(width="360px"),
        )
        initialize = self._run_button("Initialize BMTool", icon="wrench")
        single_event = self._run_button("Single Event")
        interactive = self._run_button("Interactive Tuner")
        status = w.HTML()
        initialize_output = w.Output()
        single_event_output = w.Output()
        interactive_output = w.Output()
        self.controls.update(
            {
                "enable_synapse_tuning": enabled,
                "synapse_connection": connection,
                "step4_initialize": initialize,
                "step4_single_event": single_event,
                "step4_interactive": interactive,
            }
        )
        # Keep the original key as an initialization-output alias for callers
        # that inspected the pre-card Step 4 controller.
        self.outputs["step4"] = initialize_output
        self.outputs["step4_initialize"] = initialize_output
        self.outputs["step4_single_event"] = single_event_output
        self.outputs["step4_interactive"] = interactive_output
        self.statuses["step4"] = status
        self._observe_value(enabled, "enable_synapse_tuning", bool)
        enabled.observe(lambda _change: self._refresh_button_states(), names="value")
        connection.observe(self._on_connection_changed, names="value")
        initialize.on_click(self._on_initialize_tuner)
        single_event.on_click(self._on_single_event)
        interactive.on_click(self._on_interactive_tuner)
        card_layout = w.Layout(
            border="1px solid #d9d9d9",
            padding="8px",
            margin="0 0 8px 0",
        )
        initialize_panel = w.VBox(
            [
                w.HTML("<b>Initialize BMTool</b>"),
                w.HTML(
                    "<span style='font-size:90%; color:#555'>"
                    "Enable this optional stage and choose the configured "
                    "connection used to create the tuner around the shared "
                    "Step 1 cell."
                    "</span>"
                ),
                w.HBox([enabled, connection, initialize]),
                initialize_output,
            ],
            layout=card_layout,
        )
        single_event_panel = w.VBox(
            [
                w.HTML("<b>Single Event</b>"),
                w.HTML(
                    "<span style='font-size:90%; color:#555'>"
                    "Run BMTool's single-synaptic-event response check using "
                    "the initialized tuner."
                    "</span>"
                ),
                single_event,
                single_event_output,
            ],
            layout=card_layout,
        )
        interactive_panel = w.VBox(
            [
                w.HTML("<b>Interactive Tuner</b>"),
                w.HTML(
                    "<span style='font-size:90%; color:#555'>"
                    "Open BMTool's interactive parameter controls. Tuned "
                    "values remain manual until copied into the synapse JSON."
                    "</span>"
                ),
                interactive,
                interactive_output,
            ],
            layout=card_layout,
        )
        panel = w.VBox(
            [initialize_panel, single_event_panel, interactive_panel, status]
        )
        self.panels["step4"] = panel
        self._refresh_button_states()
        return panel

    def _on_connection_changed(self, change: dict[str, Any]) -> None:
        value = str(change["new"] or "").strip()
        self.settings["synapse_connection"] = value or None

    def _on_initialize_tuner(self, _button: Any) -> None:
        connection = str(self.controls["synapse_connection"].value or "").strip()
        self.settings["synapse_connection"] = connection or None
        self._run_stage(
            "step4",
            "Initializing BMTool…",
            lambda: self._initialize_tuner(connection or None),
            output_key="step4_initialize",
        )
        if self.tuner is not None:
            self.controls["synapse_connection"].disabled = True
        self._refresh_button_states()

    def _initialize_tuner(self, connection: Optional[str]) -> None:
        self.tuner = prepare_interactive_synapse_tuner(
            self.pipeline_state,
            connection_override=connection,
        )

    def _on_single_event(self, _button: Any) -> None:
        self._run_stage(
            "step4",
            "Running BMTool single event…",
            lambda: self.tuner.SingleEvent(),
            output_key="step4_single_event",
        )

    def _on_interactive_tuner(self, _button: Any) -> None:
        self._run_stage(
            "step4",
            "Opening BMTool interactive tuner…",
            lambda: self.tuner.InteractiveTuner(),
            output_key="step4_interactive",
        )

    def _refresh_connections(self) -> None:
        connection = self.controls.get("synapse_connection")
        if connection is None or self.pipeline_state is None:
            return
        from modules.tuning import (
            default_synapse_tuning_config,
            load_synapse_tuning_config,
            synapse_tuning_config_path,
        )

        path = synapse_tuning_config_path(self.pipeline_state.tune_dir)
        config = (
            load_synapse_tuning_config(self.pipeline_state.tune_dir)
            if path.is_file()
            else default_synapse_tuning_config()
        )
        names = list(config["connections"])
        preferred = self.settings.get("synapse_connection")
        selected = (
            str(preferred)
            if preferred in names
            else str(config.get("default_connection") or names[0])
        )
        connection.options = [(name, name) for name in names]
        connection.value = selected
        self.settings["synapse_connection"] = selected

    def sync_from_settings(self, *, act_settings_changed: bool = False) -> None:
        del act_settings_changed
        enabled = self.controls.get("enable_synapse_tuning")
        if enabled is not None:
            enabled.value = bool(self.settings.get("enable_synapse_tuning"))
        if (
            "synapse_connection" in self.controls
            and self.pipeline_state is not None
            and self.tuner is None
        ):
            self._refresh_connections()

    def refresh_button_states(self, *, ready: bool, act_busy: bool) -> None:
        if "step4_initialize" not in self.controls:
            return
        enabled = bool(self.controls["enable_synapse_tuning"].value)
        initialized = self.tuner is not None
        self.controls["step4_initialize"].disabled = (
            not ready or not enabled or initialized or act_busy
        )
        self.controls["step4_single_event"].disabled = (
            not ready or not initialized or act_busy
        )
        self.controls["step4_interactive"].disabled = (
            not ready or not initialized or act_busy
        )
        current_kind = self._status_kinds.get("step4")
        if self._restart_required:
            return
        if not ready and current_kind != "error":
            self._set_status("step4", "waiting", "Waiting for Step 1.")
        elif not enabled:
            self._set_status(
                "step4", "waiting", "Optional BMTool stage is disabled."
            )
        elif initialized and current_kind in {None, "waiting", "running"}:
            self._set_status("step4", "complete", "BMTool tuner initialized.")
        elif not initialized and current_kind != "error":
            self._set_status(
                "step4",
                "waiting",
                "Choose a connection and initialize BMTool.",
            )


__all__ = ["Step4UI"]
