"""Step 3 active-protocol and FI-curve UI."""

from __future__ import annotations

import html
from typing import Any, Sequence

from ..pipeline_workflow import (
    ACTIVE_PROTOCOL_DEFAULTS,
    run_active_protocol_stage,
    run_fi_curve_stage,
)
from .common import (
    PipelineUIComponent,
    format_float_list as _format_float_list,
    optional_text as _optional_text,
    parse_float_list as _parse_float_list,
    parse_optional_float as _parse_optional_float,
)


_ACTIVE_PROTOCOL_FIELDS = (
    ("stim_delay", "Delay (ms)"),
    ("stim_dur", "Duration (ms)"),
    ("h_tstop", "Stop (ms)"),
    ("h_dt", "dt (ms)"),
)
_ACTIVE_PROTOCOL_CONTROL_KEYS = {
    field: f"active_{field}" for field, _description in _ACTIVE_PROTOCOL_FIELDS
}
_FI_PROTOCOL_CONTROL_KEYS = {
    field: f"fi_{field}" for field, _description in _ACTIVE_PROTOCOL_FIELDS
}


class Step3UI(PipelineUIComponent):
    def build_panel(self) -> Any:
        if "step3" in self.panels:
            return self.panels["step3"]
        w = self.widgets
        active_amps = w.Text(
            value=_format_float_list(self.settings.get("active_amps_pA")),
            description="Amps (pA)",
            layout=w.Layout(width="360px"),
        )
        active_timing_toggle = w.ToggleButton(
            value=False,
            description="Show advanced options",
            icon="sliders",
            layout=w.Layout(width="200px"),
        )
        active_overrides = dict(
            self.settings.get("active_protocol_overrides", {}) or {}
        )
        active_timing_controls = self._step3_protocol_controls(active_overrides)
        active_threshold = w.Text(
            value=_optional_text(
                self.settings.get("active_spike_threshold_mV", -20.0)
            ),
            description="Spike threshold (mV)",
            layout=w.Layout(width="260px"),
            style={"description_width": "145px"},
        )
        include_currents = w.Checkbox(
            value=bool(self.settings.get("active_include_currents", True)),
            description="Plot recorded currents",
            indent=False,
            layout=w.Layout(width="220px"),
        )
        active_amp_values = _parse_float_list(
            self.settings.get("active_amps_pA"),
            name="Active amplitudes",
        )
        active_current_amp = w.Dropdown(
            options=self._active_current_amp_options(active_amp_values),
            value=self._resolved_active_current_amp(active_amp_values),
            description="Current sweep",
            layout=w.Layout(width="270px"),
            style={"description_width": "105px"},
            disabled=not include_currents.value,
        )
        active_advanced_box = w.VBox(
            [
                w.HTML(
                    "<span style='font-size:90%; color:#555'>"
                    "Protocol timing starts with SCP's Step 3 defaults. Values "
                    "matching those defaults are omitted from "
                    "<code>active_protocol_overrides</code>."
                    "</span>"
                ),
                w.HBox(list(active_timing_controls.values())),
                active_threshold,
                w.HTML(
                    "<span style='font-size:90%; color:#555'>"
                    "<b>Recorded ionic-current display</b> · choose which "
                    "active sweep supplies the current traces."
                    "</span>"
                ),
                w.HBox([include_currents, active_current_amp]),
            ],
            layout=w.Layout(display="none"),
        )
        fi_amps = w.Text(
            value=_format_float_list(self.settings.get("fi_amps_pA")),
            description="Amps (pA)",
            layout=w.Layout(width="560px"),
        )
        fi_timing_toggle = w.ToggleButton(
            value=False,
            description="Show advanced options",
            icon="sliders",
            layout=w.Layout(width="200px"),
        )
        raw_fi_overrides = self.settings.get("fi_protocol_overrides")
        fi_overrides = dict(
            active_overrides if raw_fi_overrides is None else raw_fi_overrides or {}
        )
        fi_timing_controls = self._step3_protocol_controls(fi_overrides)
        fi_threshold = w.Text(
            value=_optional_text(self.settings.get("fi_spike_threshold_mV", -20.0)),
            description="Spike threshold (mV)",
            layout=w.Layout(width="260px"),
            style={"description_width": "145px"},
        )
        fi_advanced_box = w.VBox(
            [
                w.HTML(
                    "<span style='font-size:90%; color:#555'>"
                    "FI timing is independent from the trace-check timing. Values "
                    "matching SCP defaults are omitted from "
                    "<code>fi_protocol_overrides</code>."
                    "</span>"
                ),
                w.HBox(list(fi_timing_controls.values())),
                fi_threshold,
            ],
            layout=w.Layout(display="none"),
        )
        active_target_source = w.HTML(self._configured_active_target_label())
        active_button = self._run_button("Run active protocol")
        fi_button = self._run_button("Run FI curve")
        active_output = w.Output()
        fi_output = w.Output()

        status = w.HTML()
        self.controls.update(
            {
                "active_amps_pA": active_amps,
                "active_timing_toggle": active_timing_toggle,
                "active_timing_box": active_advanced_box,
                "active_spike_threshold_mV": active_threshold,
                "active_include_currents": include_currents,
                "active_current_display_amp_pA": active_current_amp,
                "fi_amps_pA": fi_amps,
                "fi_timing_toggle": fi_timing_toggle,
                "fi_timing_box": fi_advanced_box,
                "fi_spike_threshold_mV": fi_threshold,
                "fi_target_source": active_target_source,
                "step3_active": active_button,
                "step3_fi": fi_button,
                **{
                    _ACTIVE_PROTOCOL_CONTROL_KEYS[field]: control
                    for field, control in active_timing_controls.items()
                },
                **{
                    _FI_PROTOCOL_CONTROL_KEYS[field]: control
                    for field, control in fi_timing_controls.items()
                },
            }
        )
        self.outputs["step3_active"] = active_output
        self.outputs["step3_fi"] = fi_output
        self.statuses["step3"] = status
        self._observe_valid_float_list(active_amps, "active_amps_pA")
        self._observe_valid_float_list(fi_amps, "fi_amps_pA")
        self._observe_valid_optional_float(
            active_threshold,
            "active_spike_threshold_mV",
        )
        self._observe_valid_optional_float(
            fi_threshold,
            "fi_spike_threshold_mV",
        )
        self._observe_value(include_currents, "active_include_currents", bool)
        active_current_amp.observe(
            self._on_active_current_amp_changed,
            names="value",
        )
        active_amps.observe(
            self._on_active_amplitudes_changed,
            names="value",
        )
        include_currents.observe(
            lambda change: setattr(active_current_amp, "disabled", not change["new"]),
            names="value",
        )
        active_timing_toggle.observe(
            lambda change: self._on_step3_options_toggled("active", change),
            names="value",
        )
        fi_timing_toggle.observe(
            lambda change: self._on_step3_options_toggled("fi", change),
            names="value",
        )
        active_button.on_click(self._on_active_protocol)
        fi_button.on_click(self._on_fi_curve)

        card_layout = w.Layout(
            border="1px solid #d9d9d9",
            padding="8px",
            margin="0 0 8px 0",
        )
        active_panel = w.VBox(
            [
                w.HTML("<b>Active protocol</b>"),
                w.HTML(
                    "<span style='font-size:90%; color:#555'>"
                    "Inspect positive-current voltage traces, recorded currents, "
                    "and per-current firing measurements."
                    "</span>"
                ),
                w.HBox([active_amps, active_timing_toggle, active_button]),
                active_advanced_box,
                active_output,
            ],
            layout=card_layout,
        )
        fi_panel = w.VBox(
            [
                w.HTML("<b>FI curve</b>"),
                active_target_source,
                w.HTML(
                    "<span style='font-size:90%; color:#555'>"
                    "Run a frequency-current sweep and compare it with the "
                    "configured target when available."
                    "</span>"
                ),
                w.HBox([fi_amps, fi_timing_toggle, fi_button]),
                fi_advanced_box,
                fi_output,
            ],
            layout=card_layout,
        )
        act_panel = self._create_step3_act_panel(card_layout)
        panel = w.VBox([active_panel, fi_panel, act_panel, status])
        self.panels["step3"] = panel
        if self.pipeline_state is not None:
            self._refresh_act_inspection()
        self._refresh_button_states()
        return panel

    def _step3_protocol_controls(
        self,
        overrides: MutableMapping[str, Any],
    ) -> dict[str, Any]:
        w = self.widgets
        return {
            field: w.Text(
                value=_optional_text(
                    overrides.get(field, ACTIVE_PROTOCOL_DEFAULTS[field])
                ),
                description=description,
                layout=w.Layout(width="220px"),
                style={"description_width": "95px"},
            )
            for field, description in _ACTIVE_PROTOCOL_FIELDS
        }

    def _configured_active_target_label(self) -> str:
        from modules.tuning import (
            fi_curve_from_config,
            load_target_config,
            manual_fi_csv_from_config,
            target_source_mode_from_config,
        )

        tune_dir = self._selected_tune_dir()
        config = load_target_config(tune_dir)
        source_mode = target_source_mode_from_config(config, default="none")
        detail = "no configured FI target"
        if source_mode == "manual":
            currents, frequencies = fi_curve_from_config(config)
            csv_path = manual_fi_csv_from_config(config, tune_dir)
            if currents and frequencies:
                detail = f"{len(currents)} configured FI target points"
            elif csv_path is not None:
                detail = f"FI target CSV: {Path(csv_path).name}"
            else:
                detail = "manual source has no FI target values"
        elif source_mode == "allen_nwb":
            detail = "Allen NWB target will be resolved at run time"
        elif source_mode == "traces":
            detail = "active trace target will be resolved for ACT"
        return (
            "<span style='font-size:90%; color:#555'>"
            f"Target source: <b>{html.escape(str(source_mode))}</b> · "
            f"{html.escape(detail)}"
            "</span>"
        )

    def _refresh_active_target_labels(self) -> None:
        label = self._configured_active_target_label()
        for key in ("fi_target_source", "act_target_source"):
            control = self.controls.get(key)
            if control is not None:
                control.value = label

    def _on_step3_options_toggled(
        self,
        scope: str,
        change: dict[str, Any],
    ) -> None:
        visible = bool(change["new"])
        options_box = self.controls.get(f"{scope}_timing_box")
        toggle = self.controls.get(f"{scope}_timing_toggle")
        if options_box is not None:
            options_box.layout.display = "" if visible else "none"
        if toggle is not None:
            toggle.description = (
                "Hide advanced options" if visible else "Show advanced options"
            )

    def _step3_protocol_overrides_from_controls(
        self,
        *,
        scope: str,
        setting_key: str,
        label: str,
    ) -> dict[str, float]:
        control_keys = (
            _ACTIVE_PROTOCOL_CONTROL_KEYS
            if scope == "active"
            else _FI_PROTOCOL_CONTROL_KEYS
        )
        effective: dict[str, float] = {}
        for field, description in _ACTIVE_PROTOCOL_FIELDS:
            control = self.controls[control_keys[field]]
            value = _parse_optional_float(control.value, name=description)
            if value is None:
                value = float(ACTIVE_PROTOCOL_DEFAULTS[field])
            effective[field] = float(value)

        if effective["stim_delay"] < 0:
            raise ValueError(f"{label} delay must be non-negative.")
        if effective["stim_dur"] <= 0 or effective["h_tstop"] <= 0:
            raise ValueError(f"{label} duration and stop time must be positive.")
        if effective["h_dt"] <= 0:
            raise ValueError(f"{label} dt must be positive.")
        if effective["h_tstop"] < effective["stim_delay"] + effective["stim_dur"]:
            raise ValueError(f"{label} stop time must be at least delay + duration.")

        overrides = {
            field: value
            for field, value in effective.items()
            if value != float(ACTIVE_PROTOCOL_DEFAULTS[field])
        }
        self.settings[setting_key] = dict(overrides)
        return overrides

    @staticmethod
    def _active_current_amp_options(
        amplitudes: Sequence[float],
    ) -> list[tuple[str, float]]:
        unique: list[float] = []
        for amplitude in amplitudes:
            value = float(amplitude)
            if value not in unique:
                unique.append(value)
        return [(f"{value:g} pA", value) for value in unique]

    def _resolved_active_current_amp(self, amplitudes: Sequence[float]) -> float:
        values = [float(value) for value in amplitudes]
        preferred = self.settings.get("active_current_display_amp_pA")
        if preferred is not None:
            preferred_value = float(preferred)
            if preferred_value in values:
                return preferred_value
            self.settings["active_current_display_amp_pA"] = None
        return values[-1]

    def _refresh_active_current_amp_control(
        self,
        amplitudes: Sequence[float],
    ) -> None:
        control = self.controls.get("active_current_display_amp_pA")
        if control is None:
            return
        self._syncing_active_current_amp = True
        try:
            control.options = self._active_current_amp_options(amplitudes)
            control.value = self._resolved_active_current_amp(amplitudes)
        finally:
            self._syncing_active_current_amp = False

    def _on_active_amplitudes_changed(self, change: dict[str, Any]) -> None:
        try:
            amplitudes = _parse_float_list(
                change["new"],
                name="Active amplitudes",
            )
        except ValueError:
            return
        self._refresh_active_current_amp_control(amplitudes)

    def _on_active_current_amp_changed(self, change: dict[str, Any]) -> None:
        if self._syncing_active_current_amp or change["new"] is None:
            return
        self.settings["active_current_display_amp_pA"] = float(change["new"])

    def _on_active_protocol(self, _button: Any) -> None:
        try:
            active_amps = _parse_float_list(
                self.controls["active_amps_pA"].value,
                name="Active amplitudes",
            )
            protocol_overrides = self._step3_protocol_overrides_from_controls(
                scope="active",
                setting_key="active_protocol_overrides",
                label="Active protocol",
            )
            spike_threshold = _parse_optional_float(
                self.controls["active_spike_threshold_mV"].value,
                name="Active spike threshold",
            )
            if spike_threshold is None:
                spike_threshold = -20.0
            include_currents = bool(
                self.controls["active_include_currents"].value
            )
            current_display_amp = float(
                self.controls["active_current_display_amp_pA"].value
            )
            if current_display_amp not in active_amps:
                raise ValueError(
                    "The recorded-current sweep must be one of the active "
                    "protocol amplitudes."
                )
        except Exception as exc:
            self._show_validation_error(
                "step3",
                exc,
                output_key="step3_active",
            )
            return
        self.settings.update(
            {
                "active_amps_pA": active_amps,
                "active_spike_threshold_mV": spike_threshold,
                "active_include_currents": include_currents,
                "active_current_display_amp_pA": current_display_amp,
            }
        )
        self._run_stage(
            "step3",
            "Running active protocol…",
            lambda: self._run_active_protocol(
                active_amps,
                protocol_overrides,
                spike_threshold,
                include_currents,
                current_display_amp,
            ),
            output_key="step3_active",
        )

    def _run_active_protocol(
        self,
        active_amps: list[float],
        protocol_overrides: dict[str, float],
        spike_threshold: float,
        include_currents: bool,
        current_display_amp: float,
    ) -> None:
        self.active_result = run_active_protocol_stage(
            self.pipeline_state,
            active_amps_pA=active_amps,
            protocol_overrides=protocol_overrides,
            spike_threshold_mV=spike_threshold,
            include_currents=include_currents,
            current_amp_pA=current_display_amp,
        )

    def _on_fi_curve(self, _button: Any) -> None:
        try:
            self._refresh_active_target_labels()
            fi_amps = _parse_float_list(
                self.controls["fi_amps_pA"].value,
                name="FI amplitudes",
            )
            protocol_overrides = self._step3_protocol_overrides_from_controls(
                scope="fi",
                setting_key="fi_protocol_overrides",
                label="FI protocol",
            )
            spike_threshold = _parse_optional_float(
                self.controls["fi_spike_threshold_mV"].value,
                name="FI spike threshold",
            )
            if spike_threshold is None:
                spike_threshold = -20.0
        except Exception as exc:
            self._show_validation_error(
                "step3",
                exc,
                output_key="step3_fi",
            )
            return
        self.settings.update(
            {
                "fi_amps_pA": fi_amps,
                "fi_spike_threshold_mV": spike_threshold,
            }
        )
        self._run_stage(
            "step3",
            "Running FI curve…",
            lambda: self._run_fi_curve(
                fi_amps,
                protocol_overrides,
                spike_threshold,
            ),
            output_key="step3_fi",
        )

    def _run_fi_curve(
        self,
        fi_amps: list[float],
        protocol_overrides: dict[str, float],
        spike_threshold: float,
    ) -> None:
        self.fi_result = run_fi_curve_stage(
            self.pipeline_state,
            fi_amps_pA=fi_amps,
            protocol_overrides=protocol_overrides,
            spike_threshold_mV=spike_threshold,
        )

    def sync_from_settings(self, *, act_settings_changed: bool = False) -> None:
        del act_settings_changed
        for key in ("active_amps_pA", "fi_amps_pA"):
            control = self.controls.get(key)
            if control is not None:
                control.value = _format_float_list(self.settings.get(key))
        active_overrides = dict(
            self.settings.get("active_protocol_overrides", {}) or {}
        )
        raw_fi_overrides = self.settings.get("fi_protocol_overrides")
        fi_overrides = dict(
            active_overrides
            if raw_fi_overrides is None
            else raw_fi_overrides or {}
        )
        for overrides, control_keys in (
            (active_overrides, _ACTIVE_PROTOCOL_CONTROL_KEYS),
            (fi_overrides, _FI_PROTOCOL_CONTROL_KEYS),
        ):
            for field, _description in _ACTIVE_PROTOCOL_FIELDS:
                control = self.controls.get(control_keys[field])
                if control is not None:
                    control.value = _optional_text(
                        overrides.get(field, ACTIVE_PROTOCOL_DEFAULTS[field])
                    )
        for key in ("active_spike_threshold_mV", "fi_spike_threshold_mV"):
            control = self.controls.get(key)
            if control is not None:
                control.value = _optional_text(self.settings.get(key, -20.0))
        include = self.controls.get("active_include_currents")
        if include is not None:
            include.value = bool(self.settings.get("active_include_currents"))
        try:
            amplitudes = _parse_float_list(
                self.settings.get("active_amps_pA"),
                name="Active amplitudes",
            )
        except ValueError:
            amplitudes = []
        if amplitudes:
            self._refresh_active_current_amp_control(amplitudes)
        self._refresh_active_target_labels()

    def refresh_button_states(self, *, ready: bool, act_busy: bool) -> None:
        for key in ("step3_active", "step3_fi"):
            if key in self.controls:
                self.controls[key].disabled = not ready or act_busy
        if "step3" not in self.statuses:
            return
        if not ready and not self._restart_required:
            self._set_status("step3", "waiting", "Waiting for Step 1.")
        elif ready and self._status_kinds.get("step3") in {None, "waiting"}:
            self._set_status(
                "step3",
                "waiting",
                "Ready for active/FI checks or optional ACT preparation.",
            )


__all__ = ["Step3UI"]
