"""Step 2 passive proposal and protocol UI."""

from __future__ import annotations

import html
from typing import Any

from ..pipeline_workflow import (
    PASSIVE_PROTOCOL_DEFAULTS,
    compute_passive_proposal,
    run_passive_stage,
)
from .common import (
    PipelineUIComponent,
    format_float_list as _format_float_list,
    optional_text as _optional_text,
    parse_float_list as _parse_float_list,
    parse_optional_float as _parse_optional_float,
)


_PASSIVE_TARGET_FIELDS = (
    ("target_rin_mohm", "Rin (MΩ)"),
    ("target_tau_ms", "Tau (ms)"),
    ("target_v_rest_mv", "Vrest (mV)"),
)
_PASSIVE_PROTOCOL_FIELDS = (
    ("stim_delay", "Delay (ms)"),
    ("stim_dur", "Duration (ms)"),
    ("h_tstop", "Stop (ms)"),
    ("h_dt", "dt (ms)"),
)
_PASSIVE_TARGET_CONTROL_KEYS = {
    "target_rin_mohm": "passive_target_rin_mohm",
    "target_tau_ms": "passive_target_tau_ms",
    "target_v_rest_mv": "passive_target_v_rest_mv",
}
_PASSIVE_PROTOCOL_CONTROL_KEYS = {
    "stim_delay": "passive_stim_delay",
    "stim_dur": "passive_stim_dur",
    "h_tstop": "passive_h_tstop",
    "h_dt": "passive_h_dt",
}


class Step2UI(PipelineUIComponent):
    def build_panel(self) -> Any:
        if "step2" in self.panels:
            return self.panels["step2"]
        w = self.widgets
        source_mode, target_values = self._configured_passive_target_display()
        target_source = w.HTML()
        target_controls: dict[str, Any] = {}
        for field, description in _PASSIVE_TARGET_FIELDS:
            target_controls[field] = w.Text(
                value=_optional_text(target_values.get(field)),
                description=description,
                layout=w.Layout(width="230px"),
                style={"description_width": "90px"},
            )
        self._set_passive_target_source_label(
            target_source,
            source_mode=source_mode,
            target_values=target_values,
        )
        proposal_button = self._run_button("Compute ACT proposal")
        proposal_output = w.Output()

        amps = w.Text(
            value=_format_float_list(self.settings.get("passive_amps_pA")),
            description="Amps (pA)",
            layout=w.Layout(width="360px"),
        )
        timing_toggle = w.ToggleButton(
            value=False,
            description="Show advanced options",
            icon="sliders",
            layout=w.Layout(width="200px"),
        )
        timing_controls: dict[str, Any] = {}
        protocol_overrides = dict(
            self.settings.get("passive_protocol_overrides", {}) or {}
        )
        for field, description in _PASSIVE_PROTOCOL_FIELDS:
            timing_controls[field] = w.Text(
                value=_optional_text(
                    protocol_overrides.get(field, PASSIVE_PROTOCOL_DEFAULTS[field])
                ),
                description=description,
                layout=w.Layout(width="220px"),
                style={"description_width": "95px"},
            )
        timing_box = w.VBox(
            [
                w.HTML(
                    "<span style='font-size:90%; color:#555'>"
                    "Values matching the defaults are omitted from "
                    "<code>passive_protocol_overrides</code>."
                    "</span>"
                ),
                w.HBox(list(timing_controls.values())),
            ],
            layout=w.Layout(display="none"),
        )
        run_button = self._run_button("Run passive")
        status = w.HTML()
        passive_output = w.Output()
        self.controls.update(
            {
                "passive_amps_pA": amps,
                "passive_target_source": target_source,
                "passive_target_rin_mohm": target_controls["target_rin_mohm"],
                "passive_target_tau_ms": target_controls["target_tau_ms"],
                "passive_target_v_rest_mv": target_controls["target_v_rest_mv"],
                "passive_timing_toggle": timing_toggle,
                "passive_timing_box": timing_box,
                "passive_stim_delay": timing_controls["stim_delay"],
                "passive_stim_dur": timing_controls["stim_dur"],
                "passive_h_tstop": timing_controls["h_tstop"],
                "passive_h_dt": timing_controls["h_dt"],
                "step2_proposal": proposal_button,
                "step2_run": run_button,
            }
        )
        self.outputs["step2_proposal"] = proposal_output
        self.outputs["step2_passive"] = passive_output
        self.statuses["step2"] = status
        self._observe_valid_float_list(amps, "passive_amps_pA")
        for field, control in target_controls.items():
            self._observe_passive_target(control, field)
        timing_toggle.observe(self._on_passive_timing_toggled, names="value")
        proposal_button.on_click(self._on_passive_proposal)
        run_button.on_click(self._on_passive)

        card_layout = w.Layout(
            border="1px solid #d9d9d9",
            padding="8px",
            margin="0 0 8px 0",
        )
        proposal_panel = w.VBox(
            [
                w.HTML("<b>ACT passive proposal</b>"),
                target_source,
                w.HTML(
                    "<span style='font-size:90%; color:#555'>"
                    "These editable session values are loaded from the selected "
                    "tune and are never written back automatically."
                    "</span>"
                ),
                w.HBox(list(target_controls.values())),
                proposal_button,
                proposal_output,
            ],
            layout=card_layout,
        )
        passive_panel = w.VBox(
            [
                w.HTML("<b>Passive protocol</b>"),
                w.HBox([amps, timing_toggle, run_button]),
                timing_box,
                passive_output,
            ],
            layout=card_layout,
        )
        panel = w.VBox([proposal_panel, passive_panel, status])
        self.panels["step2"] = panel
        self._refresh_button_states()
        return panel

    def _configured_passive_target_display(self) -> tuple[str, dict[str, Any]]:
        from modules.tuning import (
            load_target_config,
            passive_targets_from_config,
            target_source_mode_from_config,
        )

        config = load_target_config(self._selected_tune_dir())
        source_mode = (
            target_source_mode_from_config(config) if config else "none"
        )
        configured = (
            passive_targets_from_config(config)
            if source_mode == "manual"
            else {field: None for field, _description in _PASSIVE_TARGET_FIELDS}
        )
        overrides = dict(self.settings.get("passive_target_overrides", {}) or {})
        for field in self._passive_target_dirty:
            if field in overrides:
                configured[field] = overrides[field]
        self._passive_target_source_mode = source_mode
        return source_mode, configured

    def _set_passive_target_source_label(
        self,
        control: Any,
        *,
        source_mode: str,
        target_values: dict[str, Any],
    ) -> None:
        complete = all(
            target_values.get(field) is not None
            for field, _description in _PASSIVE_TARGET_FIELDS
        )
        if complete and self._passive_target_dirty:
            detail = "session target overrides loaded"
        elif complete:
            detail = "effective passive values loaded"
        else:
            detail = "complete values will be required for an ACT proposal"
        control.value = (
            "<span style='font-size:90%; color:#555'>"
            f"Target source: <b>{html.escape(str(source_mode))}</b> · "
            f"{html.escape(detail)}"
            "</span>"
        )

    def _refresh_passive_target_controls(self, resolution: Any = None) -> None:
        source_control = self.controls.get("passive_target_source")
        if source_control is None:
            return
        if resolution is None:
            source_mode, values = self._configured_passive_target_display()
        else:
            source_mode = str(getattr(resolution, "target_source_mode", "manual"))
            resolved = dict(getattr(resolution, "passive_targets", {}) or {})
            values = {
                field: resolved.get(field)
                for field, _description in _PASSIVE_TARGET_FIELDS
            }
            self._passive_target_source_mode = source_mode

        self._syncing_passive_targets = True
        try:
            for field, _description in _PASSIVE_TARGET_FIELDS:
                control = self.controls.get(_PASSIVE_TARGET_CONTROL_KEYS[field])
                if control is not None:
                    control.value = _optional_text(values.get(field))
        finally:
            self._syncing_passive_targets = False
        self._set_passive_target_source_label(
            source_control,
            source_mode=source_mode,
            target_values=values,
        )

    def _observe_passive_target(self, widget: Any, field: str) -> None:
        def _sync(change: dict[str, Any]) -> None:
            if self._syncing_passive_targets:
                return
            try:
                value = _parse_optional_float(
                    change["new"],
                    name=dict(_PASSIVE_TARGET_FIELDS)[field],
                )
            except ValueError:
                return
            overrides = dict(
                self.settings.get("passive_target_overrides", {}) or {}
            )
            overrides[field] = value
            self.settings["passive_target_overrides"] = overrides
            self._passive_target_dirty.add(field)

        widget.observe(_sync, names="value")

    def _on_passive_timing_toggled(self, change: dict[str, Any]) -> None:
        visible = bool(change["new"])
        timing_box = self.controls.get("passive_timing_box")
        toggle = self.controls.get("passive_timing_toggle")
        if timing_box is not None:
            timing_box.layout.display = "" if visible else "none"
        if toggle is not None:
            toggle.description = (
                "Hide advanced options" if visible else "Show advanced options"
            )

    def _passive_targets_from_controls(self) -> dict[str, Optional[float]]:
        values: dict[str, Optional[float]] = {}
        for field, description in _PASSIVE_TARGET_FIELDS:
            control = self.controls[_PASSIVE_TARGET_CONTROL_KEYS[field]]
            values[field] = _parse_optional_float(control.value, name=description)
        return values

    def _passive_protocol_overrides_from_controls(self) -> dict[str, float]:
        effective: dict[str, float] = {}
        for field, description in _PASSIVE_PROTOCOL_FIELDS:
            control = self.controls[_PASSIVE_PROTOCOL_CONTROL_KEYS[field]]
            value = _parse_optional_float(control.value, name=description)
            if value is None:
                value = float(PASSIVE_PROTOCOL_DEFAULTS[field])
            effective[field] = float(value)

        if effective["stim_delay"] < 0:
            raise ValueError("Passive delay must be non-negative.")
        if effective["stim_dur"] <= 0 or effective["h_tstop"] <= 0:
            raise ValueError("Passive duration and stop time must be positive.")
        if effective["h_dt"] <= 0:
            raise ValueError("Passive dt must be positive.")
        if effective["h_tstop"] < effective["stim_delay"] + effective["stim_dur"]:
            raise ValueError(
                "Passive stop time must be at least delay + duration."
            )

        overrides = {
            field: value
            for field, value in effective.items()
            if value != float(PASSIVE_PROTOCOL_DEFAULTS[field])
        }
        self.settings["passive_protocol_overrides"] = dict(overrides)
        return overrides

    def _on_passive_proposal(self, _button: Any) -> None:
        try:
            if self._passive_target_source_mode == "manual":
                self._refresh_passive_target_controls()
            targets = self._passive_targets_from_controls()
        except Exception as exc:
            self._show_validation_error(
                "step2",
                exc,
                output_key="step2_proposal",
            )
            return
        self._run_stage(
            "step2",
            "Computing ACT passive proposal…",
            lambda: self._compute_passive_proposal(targets),
            output_key="step2_proposal",
        )

    def _compute_passive_proposal(
        self,
        targets: dict[str, Optional[float]],
    ) -> None:
        self.passive_proposal_result = compute_passive_proposal(
            self.pipeline_state,
            manual_passive_targets=targets,
        )
        self._refresh_passive_target_controls(
            self.passive_proposal_result.resolution
        )

    def _on_passive(self, _button: Any) -> None:
        try:
            if self._passive_target_source_mode == "manual":
                self._refresh_passive_target_controls()
            amps = _parse_float_list(
                self.controls["passive_amps_pA"].value,
                name="Passive amplitudes",
            )
            targets = self._passive_targets_from_controls()
            protocol_overrides = self._passive_protocol_overrides_from_controls()
        except Exception as exc:
            self._show_validation_error(
                "step2",
                exc,
                output_key="step2_passive",
            )
            return
        self.settings["passive_amps_pA"] = amps
        self._run_stage(
            "step2",
            "Running passive tuning…",
            lambda: self._run_passive(amps, targets, protocol_overrides),
            output_key="step2_passive",
        )

    def _run_passive(
        self,
        amps: list[float],
        targets: dict[str, Optional[float]],
        protocol_overrides: dict[str, float],
    ) -> None:
        self.passive_result = run_passive_stage(
            self.pipeline_state,
            amps_pA=amps,
            compute_act_proposal=False,
            manual_passive_targets=targets,
            protocol_overrides=protocol_overrides,
        )
        self._refresh_passive_target_controls(self.passive_result.resolution)

    def sync_from_settings(self, *, act_settings_changed: bool = False) -> None:
        del act_settings_changed
        amps = self.controls.get("passive_amps_pA")
        if amps is not None:
            amps.value = _format_float_list(self.settings.get("passive_amps_pA"))
        target_overrides = dict(
            self.settings.get("passive_target_overrides", {}) or {}
        )
        self._passive_target_dirty = {
            field
            for field, _description in _PASSIVE_TARGET_FIELDS
            if target_overrides.get(field) is not None
        }
        self._refresh_passive_target_controls()
        protocol_overrides = dict(
            self.settings.get("passive_protocol_overrides", {}) or {}
        )
        for field, _description in _PASSIVE_PROTOCOL_FIELDS:
            control = self.controls.get(_PASSIVE_PROTOCOL_CONTROL_KEYS[field])
            if control is not None:
                control.value = _optional_text(
                    protocol_overrides.get(field, PASSIVE_PROTOCOL_DEFAULTS[field])
                )

    def refresh_button_states(self, *, ready: bool, act_busy: bool) -> None:
        for key in ("step2_proposal", "step2_run"):
            if key in self.controls:
                self.controls[key].disabled = not ready or act_busy
        if "step2" in self.statuses and not ready and not self._restart_required:
            self._set_status("step2", "waiting", "Waiting for Step 1.")


__all__ = ["Step2UI"]
