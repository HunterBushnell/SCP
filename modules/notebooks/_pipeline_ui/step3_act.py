"""Experimental ACT controls embedded in the Step 3 panel."""

from __future__ import annotations

import html
import json
import os
import signal
import subprocess
import threading
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from ..pipeline_act import (
    _assert_workspace_prepared,
    display_act_evaluation,
    evaluate_fresh_act_predictions,
    inspect_act_active_stage,
    prepare_act_active_stage,
    run_fresh_act_active,
)
from .common import (
    PipelineUIComponent,
    clear_output as _clear_output,
    format_float_list as _format_float_list,
    optional_text as _optional_text,
    parse_float_list as _parse_float_list,
    parse_optional_float as _parse_optional_float,
    parse_optional_int as _parse_optional_int,
)


class Step3ACTUI(PipelineUIComponent):
    def _create_step3_act_panel(self, card_layout: Any) -> Any:
        w = self.widgets
        experimental_note = w.HTML(
            "<div style='font-size:90%; color:#7a4b00; background:#fff8e6; "
            "border-left:3px solid #d99b22; padding:6px 8px'>"
            "<b>Experimental:</b> this compact ACT active-tuning workflow has "
            "not yet been fully validated. Treat every prediction as a "
            "review-only proposal; the active protocol and FI curve above are "
            "the established Step 3 checks."
            "</div>"
        )
        info = w.HTML(
            "<span style='font-size:90%; color:#555'>Complete Step 1 to inspect ACT support.</span>"
        )
        target_summary = w.HTML()
        workload = w.HTML()
        module = w.Dropdown(
            options=[
                ("Low-threshold (LTO)", "lto"),
                ("Spiking", "spiking"),
                ("Bursting", "bursting"),
                ("All modules", "all"),
            ],
            value="lto",
            description="Module",
            layout=w.Layout(width="300px"),
        )
        cpu_default = self.settings.get("act_n_cpus")
        cpus = w.BoundedIntText(
            value=max(1, min(int(cpu_default or 4), os.cpu_count() or 1)),
            min=1,
            max=max(1, os.cpu_count() or 1),
            description="CPUs",
            layout=w.Layout(width="180px"),
        )
        prepare = self._run_button("Prepare ACT workspace", icon="wrench")
        run = self._run_button("Run selected module")
        cancel = self._run_button("Cancel", icon="stop")
        cancel.button_style = "warning"
        review = self._run_button("Review predictions", icon="table")
        evaluate = self._run_button("Evaluate predictions", icon="flask")
        review_eval = self._run_button("Review evaluation", icon="bar-chart")
        options_toggle = w.ToggleButton(
            value=False,
            description="Show ACT options",
            icon="sliders",
            layout=w.Layout(width="190px"),
        )

        target_mode = w.Dropdown(
            options=[
                ("Configured/default", None),
                ("FI arrays", "fi_arrays"),
                ("FI CSV", "fi_csv"),
                ("Allen NWB", "allen_nwb"),
                ("Existing trace NPY", "trace_npy"),
            ],
            value=None,
            description="Target",
            layout=w.Layout(width="280px"),
        )
        target_currents = w.Text(description="Currents (pA)", layout=w.Layout(width="520px"))
        target_freqs = w.Text(description="Rates (Hz)", layout=w.Layout(width="520px"))
        target_csv = w.Text(description="FI CSV", layout=w.Layout(width="520px"))
        target_nwb = w.Text(description="Allen NWB", layout=w.Layout(width="520px"))
        nwb_stimuli = w.Text(description="Stimuli", layout=w.Layout(width="520px"))
        nwb_negative = w.Checkbox(description="Include negative currents", indent=False)
        nwb_average = w.Checkbox(value=True, description="Average repeats", indent=False)
        nwb_min = w.Text(description="Min pA", layout=w.Layout(width="190px"))
        nwb_max = w.Text(description="Max pA", layout=w.Layout(width="190px"))
        nwb_threshold = w.Text(description="Spike mV", layout=w.Layout(width="190px"))
        nwb_refractory = w.Text(description="Refractory ms", layout=w.Layout(width="210px"))
        target_array_box = w.VBox([target_currents, target_freqs])
        target_csv_box = w.VBox([target_csv])
        target_nwb_box = w.VBox(
            [
                target_nwb,
                nwb_stimuli,
                w.HBox([nwb_negative, nwb_average]),
                w.HBox([nwb_min, nwb_max, nwb_threshold, nwb_refractory]),
            ]
        )

        conductance_box = w.VBox(
            [
                w.HTML(
                    "<span style='font-size:90%; color:#555'>"
                    "Select one module to edit its conductances.</span>"
                )
            ]
        )
        passive_names = w.Text(description="Passive names", layout=w.Layout(width="700px"))
        active_channels = w.Text(description="Active channels", layout=w.Layout(width="700px"))

        sim_controls: dict[str, Any] = {}
        for key, label in (
            ("h_v_init", "Initial mV"),
            ("h_tstop", "Stop ms"),
            ("h_dt", "dt ms"),
            ("h_celsius", "Temp °C"),
            ("ci_delay_ms", "Delay ms"),
            ("ci_dur_ms", "Duration ms"),
        ):
            sim_controls[key] = w.Text(
                description=label,
                layout=w.Layout(width="205px"),
                style={"description_width": "90px"},
            )
        optimizer_controls: dict[str, Any] = {}
        for key, label in (
            ("random_state", "Random seed"),
            ("n_estimators", "Estimators"),
            ("max_depth", "Max depth"),
            ("train_features", "Features"),
            ("spike_threshold", "Spike mV"),
            ("max_n_spikes", "Max spikes"),
        ):
            optimizer_controls[key] = w.Text(
                description=label,
                layout=w.Layout(width="235px" if key != "train_features" else "520px"),
                style={"description_width": "100px"},
            )
        filters = w.SelectMultiple(
            options=[("Saturated traces", "saturated"), ("No-spike traces", "no_spikes")],
            description="Remove",
            layout=w.Layout(width="330px", height="70px"),
        )
        filter_window = w.Text(description="Window ms", layout=w.Layout(width="270px"))
        saturation = w.Text(description="Saturation mV", layout=w.Layout(width="250px"))
        workspace = w.Text(
            value=_optional_text(self.settings.get("act_workspace_override")),
            description="Workspace",
            placeholder="blank = tune/act_workspace",
            layout=w.Layout(width="620px"),
        )
        overwrite = w.Checkbox(
            value=bool(self.settings.get("act_overwrite_outputs", False)),
            description="Overwrite an existing/partial selected-module run",
            indent=False,
            layout=w.Layout(width="440px"),
        )
        advanced = w.VBox(
            [
                w.HTML("<b>Target overrides</b>"),
                target_mode,
                target_array_box,
                target_csv_box,
                target_nwb_box,
                w.HTML("<b>Selected-module conductances</b>"),
                conductance_box,
                w.HTML("<b>ACT cell adapter</b>"),
                passive_names,
                active_channels,
                w.HTML("<b>Simulation</b>"),
                w.HTML(
                    "<span style='font-size:90%; color:#555'>"
                    "Current amplitudes are target-derived.</span>"
                ),
                w.HBox(list(sim_controls.values())[:3]),
                w.HBox(list(sim_controls.values())[3:]),
                w.HTML("<b>Optimizer</b>"),
                w.HBox(list(optimizer_controls.values())[:3]),
                w.HBox(list(optimizer_controls.values())[3:]),
                w.HTML("<b>Filtering</b>"),
                w.HBox([filters, w.VBox([filter_window, saturation])]),
                w.HTML("<b>Workspace and output</b>"),
                workspace,
                overwrite,
            ],
            layout=w.Layout(display="none", border="1px solid #eee", padding="8px"),
        )
        output = w.Output()

        act_controls = {
            "act_experimental_note": experimental_note,
            "act_support_info": info,
            "act_target_source": target_summary,
            "act_workload": workload,
            "act_active_module": module,
            "act_n_cpus": cpus,
            "step3_act_prepare": prepare,
            "step3_act_run": run,
            "step3_act_cancel": cancel,
            "step3_act_review": review,
            "step3_act_evaluate": evaluate,
            "step3_act_review_evaluation": review_eval,
            "act_options_toggle": options_toggle,
            "act_options_box": advanced,
            "act_target_mode": target_mode,
            "act_target_currents": target_currents,
            "act_target_frequencies": target_freqs,
            "act_target_csv": target_csv,
            "act_target_nwb": target_nwb,
            "act_nwb_stimuli": nwb_stimuli,
            "act_nwb_include_negative": nwb_negative,
            "act_nwb_average": nwb_average,
            "act_nwb_min": nwb_min,
            "act_nwb_max": nwb_max,
            "act_nwb_threshold": nwb_threshold,
            "act_nwb_refractory": nwb_refractory,
            "act_target_array_box": target_array_box,
            "act_target_csv_box": target_csv_box,
            "act_target_nwb_box": target_nwb_box,
            "act_conductance_box": conductance_box,
            "act_passive_names": passive_names,
            "act_active_channels": active_channels,
            "act_filter_features": filters,
            "act_filter_window": filter_window,
            "act_saturation_threshold": saturation,
            "act_workspace_override": workspace,
            "act_overwrite_outputs": overwrite,
        }
        act_controls.update({f"act_sim_{key}": value for key, value in sim_controls.items()})
        act_controls.update({f"act_opt_{key}": value for key, value in optimizer_controls.items()})
        self.controls.update(act_controls)
        self.outputs["step3_act"] = output

        module.observe(self._on_act_module_changed, names="value")
        cpus.observe(self._on_act_cpus_changed, names="value")
        target_mode.observe(self._on_act_target_mode_changed, names="value")
        options_toggle.observe(self._on_act_options_toggled, names="value")
        workspace.observe(self._on_act_workspace_changed, names="value")
        overwrite.observe(self._on_act_overwrite_changed, names="value")
        prepare.on_click(self._on_act_prepare)
        run.on_click(self._on_act_run)
        cancel.on_click(self._on_act_cancel)
        review.on_click(self._on_act_review)
        evaluate.on_click(self._on_act_evaluate)
        review_eval.on_click(self._on_act_review_evaluation)

        self._observe_act_override(
            target_currents,
            "target",
            "fi_currents_pA",
            self._act_float_list_or_none,
        )
        self._observe_act_override(
            target_freqs,
            "target",
            "fi_frequencies_hz",
            self._act_float_list_or_none,
        )
        self._observe_act_override(target_csv, "target", "fi_csv_path", self._act_text_or_none)
        self._observe_act_override(target_nwb, "target", "nwb_path", self._act_text_or_none)
        self._observe_act_override(
            nwb_stimuli,
            "target",
            "nwb_stimulus_names",
            self._act_string_list_or_none,
        )
        self._observe_act_override(nwb_negative, "target", "nwb_include_negative_currents", bool)
        self._observe_act_override(nwb_average, "target", "nwb_average_repeats", bool)
        for control, key in (
            (nwb_min, "nwb_min_current_pA"),
            (nwb_max, "nwb_max_current_pA"),
            (nwb_threshold, "nwb_spike_threshold_mV"),
            (nwb_refractory, "nwb_refractory_ms"),
        ):
            self._observe_act_override(
                control, "target", key, self._act_optional_float
            )
        self._observe_act_override(
            passive_names,
            "act_cell",
            "passive",
            self._act_string_list_or_none,
        )
        self._observe_act_override(
            active_channels,
            "act_cell",
            "active_channels",
            self._act_string_list_or_none,
        )
        for key, control in sim_controls.items():
            self._observe_act_override(control, "simulation", key, self._act_optional_float)
        for key, control in optimizer_controls.items():
            parser = self._act_string_list_or_none if key == "train_features" else (
                self._act_optional_int
                if key
                in {"random_state", "n_estimators", "max_depth", "max_n_spikes"}
                else self._act_optional_float
            )
            self._observe_act_override(control, "optimizer", key, parser)
        self._observe_act_override(
            filters,
            "filter",
            "filtered_out_features",
            lambda value: list(value),
        )
        self._observe_act_override(
            filter_window,
            "filter",
            "window_of_inspection",
            self._act_float_list_or_none,
        )
        self._observe_act_override(
            saturation,
            "filter",
            "saturation_threshold",
            self._act_optional_float,
        )
        self._update_act_target_visibility()

        return w.VBox(
            [
                w.HTML("<b>ACT active tuning</b>"),
                experimental_note,
                info,
                target_summary,
                workload,
                w.HBox([module, cpus, options_toggle]),
                advanced,
                w.HBox([prepare, run, cancel]),
                w.HBox([review, evaluate, review_eval]),
                output,
            ],
            layout=card_layout,
        )

    def _on_act_options_toggled(self, change: dict[str, Any]) -> None:
        visible = bool(change["new"])
        box = self.controls.get("act_options_box")
        toggle = self.controls.get("act_options_toggle")
        if box is not None:
            box.layout.display = "" if visible else "none"
        if toggle is not None:
            toggle.description = "Hide ACT options" if visible else "Show ACT options"

    def _on_act_target_mode_changed(self, change: dict[str, Any]) -> None:
        if not self._act_syncing:
            self._set_act_override("target", "mode", change["new"])
            self.act_workspace_result = None
        self._update_act_target_visibility()
        self._refresh_button_states()

    def _update_act_target_visibility(self) -> None:
        mode_control = self.controls.get("act_target_mode")
        if mode_control is None:
            return
        mode = mode_control.value
        if mode is None and getattr(self, "_act_inspection", None) is not None:
            mode = self._act_inspection.target_mode
        for key, shown in (
            ("act_target_array_box", mode in {None, "fi_arrays"}),
            ("act_target_csv_box", mode == "fi_csv"),
            ("act_target_nwb_box", mode == "allen_nwb"),
        ):
            self.controls[key].layout.display = "" if shown else "none"

    def _on_act_module_changed(self, change: dict[str, Any]) -> None:
        if not self._act_syncing:
            self.settings["act_active_module"] = change["new"]
        self._rebuild_act_conductance_editor()
        self._refresh_act_workload_label()
        self._refresh_button_states()

    def _on_act_cpus_changed(self, change: dict[str, Any]) -> None:
        if not self._act_syncing:
            self.settings["act_n_cpus"] = int(change["new"])

    def _on_act_workspace_changed(self, change: dict[str, Any]) -> None:
        if self._act_syncing:
            return
        self.settings["act_workspace_override"] = str(change["new"]).strip() or None
        self._applied_act_settings_signature = self._act_settings_signature(
            self.settings
        )
        self.act_workspace_result = None
        self._refresh_button_states()

    def _on_act_overwrite_changed(self, change: dict[str, Any]) -> None:
        if not self._act_syncing:
            self.settings["act_overwrite_outputs"] = bool(change["new"])
        self._refresh_button_states()

    def _observe_act_override(
        self,
        widget: Any,
        section: str,
        key: str,
        parser: Any,
    ) -> None:
        def _sync(change: dict[str, Any]) -> None:
            if self._act_syncing:
                return
            try:
                value = parser(change["new"])
            except (TypeError, ValueError):
                return
            self._set_act_override(section, key, value)
            self.act_workspace_result = None
            self._refresh_button_states()

        widget.observe(_sync, names="value")

    def _set_act_override(self, section: str, key: str, value: Any) -> None:
        overrides = deepcopy(dict(self.settings.get("act_overrides") or {}))
        section_values = dict(overrides.get(section) or {})
        if value is None:
            section_values.pop(key, None)
        else:
            section_values[key] = value
        if section_values:
            overrides[section] = section_values
        else:
            overrides.pop(section, None)
        self.settings["act_overrides"] = overrides
        self._applied_act_settings_signature = self._act_settings_signature(
            self.settings
        )

    @staticmethod
    def _act_text_or_none(value: Any) -> Optional[str]:
        return str(value).strip() or None

    @staticmethod
    def _act_optional_float(value: Any) -> Optional[float]:
        return _parse_optional_float(value, name="ACT option")

    @staticmethod
    def _act_optional_int(value: Any) -> Optional[int]:
        return _parse_optional_int(value, name="ACT option")

    @staticmethod
    def _act_float_list_or_none(value: Any) -> Optional[list[float]]:
        if value in (None, ""):
            return None
        return _parse_float_list(value, name="ACT values")

    @staticmethod
    def _act_string_list_or_none(value: Any) -> Optional[list[str]]:
        if value in (None, ""):
            return None
        if isinstance(value, str):
            values = [part.strip() for part in value.split(",")]
        else:
            values = [str(part).strip() for part in value]
        values = [value for value in values if value]
        return values or None

    def _refresh_act_inspection(self) -> None:
        if self.pipeline_state is None or "act_support_info" not in self.controls:
            return
        try:
            inspection = inspect_act_active_stage(
                self.pipeline_state,
                workspace=self.settings.get("act_workspace_override"),
                overrides=self.settings.get("act_overrides"),
            )
        except Exception as exc:
            self._act_inspection = None
            self.controls["act_support_info"].value = (
                "<span style='font-size:90%; color:#b42318'><b>Unavailable:</b> "
                + html.escape(str(exc))
                + "</span>"
            )
            self.controls["act_target_source"].value = ""
            self.controls["act_workload"].value = ""
            self._refresh_button_states()
            return
        self._act_inspection = inspection
        self._populate_act_controls(inspection)
        support_color = "#18743c" if inspection.loader_support == "supported" else "#8a5a00"
        self.controls["act_support_info"].value = (
            f"<span style='font-size:90%; color:{support_color}'>"
            f"Configuration: <b>{html.escape(inspection.config_source)}</b> · "
            f"loader <code>{html.escape(inspection.loader_name)}</code> "
            f"(<b>{html.escape(inspection.loader_support)}</b>)</span>"
        )
        self.controls["act_target_source"].value = (
            "<span style='font-size:90%; color:#555'>"
            f"Target: <b>{html.escape(inspection.target_mode)}</b> · "
            f"{inspection.target_point_count} point(s) · "
            f"<code>{html.escape(str(inspection.target_path))}</code></span>"
        )
        self._refresh_act_workload_label()
        self._refresh_button_states()

    def _populate_act_controls(self, result: Any) -> None:
        cfg = result.resolved_config
        target = cfg.get("target", {}) or {}
        sim = cfg.get("simulation", {}) or {}
        opt = cfg.get("optimizer", {}) or {}
        filter_cfg = cfg.get("filter", {}) or {}
        settings_overrides = dict(self.settings.get("act_overrides") or {})
        target_override_mode = (settings_overrides.get("target") or {}).get("mode")
        self._act_syncing = True
        try:
            enabled = list(result.enabled_modules)
            options = [(key.replace("_", " ").title(), key) for key in enabled]
            options.append(("All modules", "all"))
            self.controls["act_active_module"].options = options
            requested = self.settings.get("act_active_module")
            if requested not in [value for _label, value in options]:
                statuses = result.output_status
                requested = next(
                    (
                        key
                        for key in enabled
                        if statuses.get(key, {}).get("status") != "current"
                    ),
                    enabled[0] if enabled else "all",
                )
                self.settings["act_active_module"] = requested
            self.controls["act_active_module"].value = requested
            configured_cpus = int(opt.get("n_cpus") or 4)
            selected_cpus = int(self.settings.get("act_n_cpus") or configured_cpus)
            self.controls["act_n_cpus"].value = max(
                1, min(selected_cpus, self.controls["act_n_cpus"].max)
            )
            self.controls["act_target_mode"].value = target_override_mode
            self.controls["act_target_currents"].value = _format_float_list(
                target.get("fi_currents_pA") or []
            )
            self.controls["act_target_frequencies"].value = _format_float_list(
                target.get("fi_frequencies_hz") or []
            )
            self.controls["act_target_csv"].value = _optional_text(target.get("source_csv"))
            self.controls["act_target_nwb"].value = _optional_text(target.get("source_nwb"))
            nwb = target.get("nwb_options", {}) or {}
            self.controls["act_nwb_stimuli"].value = ", ".join(nwb.get("stimulus_names") or [])
            self.controls["act_nwb_include_negative"].value = bool(
                nwb.get("include_negative_currents", False)
            )
            self.controls["act_nwb_average"].value = bool(nwb.get("average_repeats", True))
            for control_key, value in (
                ("act_nwb_min", nwb.get("min_current_pA")),
                ("act_nwb_max", nwb.get("max_current_pA")),
                ("act_nwb_threshold", nwb.get("spike_threshold_mV")),
                ("act_nwb_refractory", nwb.get("refractory_ms")),
            ):
                self.controls[control_key].value = _optional_text(value)
            self.controls["act_passive_names"].value = ", ".join(
                cfg.get("act_cell", {}).get("passive", []) or []
            )
            self.controls["act_active_channels"].value = ", ".join(
                cfg.get("act_cell", {}).get("active_channels", []) or []
            )
            for key in ("h_v_init", "h_tstop", "h_dt", "h_celsius", "ci_delay_ms", "ci_dur_ms"):
                self.controls[f"act_sim_{key}"].value = _optional_text(sim.get(key))
            for key in (
                "random_state",
                "n_estimators",
                "max_depth",
                "train_features",
                "spike_threshold",
                "max_n_spikes",
            ):
                value = opt.get(key)
                if key == "train_features" and value is not None:
                    value = ", ".join(value)
                self.controls[f"act_opt_{key}"].value = _optional_text(value)
            self.controls["act_filter_features"].value = tuple(
                filter_cfg.get("filtered_out_features") or []
            )
            self.controls["act_filter_window"].value = _format_float_list(
                filter_cfg.get("window_of_inspection") or []
            )
            self.controls["act_saturation_threshold"].value = _optional_text(
                filter_cfg.get("saturation_threshold")
            )
            self.controls["act_workspace_override"].value = _optional_text(
                self.settings.get("act_workspace_override")
            )
            self.controls["act_overwrite_outputs"].value = bool(
                self.settings.get("act_overwrite_outputs", False)
            )
        finally:
            self._act_syncing = False
        self._update_act_target_visibility()
        self._rebuild_act_conductance_editor()

    def _rebuild_act_conductance_editor(self) -> None:
        box = self.controls.get("act_conductance_box")
        inspection = getattr(self, "_act_inspection", None)
        if box is None or inspection is None:
            return
        module_key = self.controls["act_active_module"].value
        if module_key == "all":
            box.children = (
                self.widgets.HTML(
                    "<span style='font-size:90%; color:#555'>"
                    "Select one module to edit its conductances; All runs each "
                    "configured module in order.</span>"
                ),
            )
            self._act_conductance_controls = []
            return
        module_values = (
            (self.settings.get("act_overrides") or {}).get("modules")
            or inspection.resolved_config.get("modules")
            or {}
        )
        spec = module_values.get(module_key, {})
        rows = []
        controls: list[dict[str, Any]] = []
        for index, item in enumerate(spec.get("conductances", []) or []):
            variable = self.widgets.Text(
                value=str(item.get("variable_name") or ""),
                description="Variable",
                layout=self.widgets.Layout(width="245px"),
            )
            blocked = self.widgets.Checkbox(
                value=bool(item.get("blocked", False)), description="Blocked", indent=False
            )
            low = self.widgets.Text(
                value=_optional_text(item.get("low")),
                description="Low",
                layout=self.widgets.Layout(width="180px"),
            )
            high = self.widgets.Text(
                value=_optional_text(item.get("high")),
                description="High",
                layout=self.widgets.Layout(width="180px"),
            )
            variation = self.widgets.Text(
                value=_optional_text(item.get("bounds_variation")),
                description="Variation",
                layout=self.widgets.Layout(width="190px"),
            )
            slices = self.widgets.BoundedIntText(
                value=max(1, int(item.get("n_slices", 1))),
                min=1,
                max=1000,
                description="Slices",
                layout=self.widgets.Layout(width="170px"),
            )
            row_controls = {
                "variable_name": variable,
                "blocked": blocked,
                "low": low,
                "high": high,
                "bounds_variation": variation,
                "n_slices": slices,
            }
            for bounded in (low, high, variation, slices):
                bounded.disabled = blocked.value
            def _toggle_bounds(
                change: dict[str, Any],
                bounded: tuple[Any, ...] = (low, high, variation, slices),
            ) -> None:
                for control in bounded:
                    control.disabled = bool(change["new"])

            blocked.observe(_toggle_bounds, names="value")
            for field, control in row_controls.items():
                control.observe(
                    lambda change, module_key=module_key, index=index, field=field: (
                        self._on_act_conductance_changed(
                            module_key, index, field, change["new"]
                        )
                    ),
                    names="value",
                )
            controls.append(row_controls)
            rows.append(self.widgets.HBox(list(row_controls.values())))
        self._act_conductance_controls = controls
        box.children = tuple(rows) or (
            self.widgets.HTML(
                "<span style='color:#b42318'>"
                "This module has no conductances.</span>"
            ),
        )

    def _on_act_conductance_changed(
        self, module_key: str, index: int, field: str, raw_value: Any
    ) -> None:
        if self._act_syncing:
            return
        try:
            if field == "variable_name":
                value = str(raw_value).strip()
            elif field == "blocked":
                value = bool(raw_value)
            elif field == "n_slices":
                value = int(raw_value)
            else:
                value = _parse_optional_float(raw_value, name=field)
        except ValueError:
            return
        inspection = getattr(self, "_act_inspection", None)
        if inspection is None:
            return
        overrides = deepcopy(dict(self.settings.get("act_overrides") or {}))
        modules = deepcopy(
            overrides.get("modules") or inspection.resolved_config.get("modules") or {}
        )
        modules[module_key]["conductances"][index][field] = value
        overrides["modules"] = modules
        self.settings["act_overrides"] = overrides
        self._applied_act_settings_signature = self._act_settings_signature(
            self.settings
        )
        self.act_workspace_result = None
        self._refresh_act_workload_label()
        self._refresh_button_states()

    def _refresh_act_workload_label(self) -> None:
        control = self.controls.get("act_workload")
        inspection = getattr(self, "_act_inspection", None)
        if control is None or inspection is None:
            return
        from modules.tuning import estimate_act_workload

        cfg = deepcopy(inspection.resolved_config)
        modules_override = (self.settings.get("act_overrides") or {}).get("modules")
        if modules_override:
            cfg["modules"] = deepcopy(modules_override)
        selection = self.controls["act_active_module"].value
        estimate = estimate_act_workload(cfg, modules=selection)
        control.value = (
            "<span style='font-size:90%; color:#555'>"
            f"Estimated workload: <b>{estimate['training_traces']:,}</b> training + "
            f"<b>{estimate['evaluation_traces']:,}</b> evaluation traces "
            f"({estimate['target_points']} target points)</span>"
        )

    def _on_act_prepare(self, _button: Any) -> None:
        self.act_workspace_result = None
        self.act_run_result = None
        self.act_evaluation_result = None
        self._start_act_background(
            action="prepare",
            operation=lambda: prepare_act_active_stage(
                self.pipeline_state,
                workspace=self.settings.get("act_workspace_override"),
                overrides=self.settings.get("act_overrides"),
                probe_act=True,
            ),
            success=self._complete_act_prepare,
        )

    def _complete_act_prepare(self, result: Any) -> None:
        self.act_workspace_result = result
        self._act_inspection = result
        self._populate_act_controls(result)
        print("ACT workspace prepared")
        print("  Workspace:", result.workspace)
        print("  Config:", result.config_path)
        print("  Loader support:", result.loader_support)
        print("  Target:", result.target_mode, f"({result.target_point_count} points)")
        print("  Modules:", ", ".join(result.enabled_modules))
        print("  ACT probe:", result.act_message)
        if not result.act_available:
            raise RuntimeError(result.act_message)

    def _on_act_run(self, _button: Any) -> None:
        if self.act_workspace_result is None:
            self._show_validation_error(
                "step3",
                RuntimeError("Prepare the ACT workspace after the latest option changes."),
                output_key="step3_act",
            )
            return
        module = self.controls["act_active_module"].value
        n_cpus = int(self.controls["act_n_cpus"].value)
        overwrite = bool(self.controls["act_overwrite_outputs"].value)
        self._start_act_background(
            action="run",
            operation=lambda: run_fresh_act_active(
                self.pipeline_state,
                self.act_workspace_result,
                modules=module,
                n_cpus=n_cpus,
                overwrite=overwrite,
                line_callback=self._record_act_line,
                process_callback=self._record_act_process,
            ),
            success=self._complete_act_run,
        )

    def _complete_act_run(self, result: Any) -> None:
        previous_workspace = self.act_workspace_result
        self.act_run_result = result
        self.act_predictions = dict(result.predictions)
        print("ACT module run complete. Predictions remain review-only.")
        self._print_act_review(result.predictions, result.metrics, result.output_status)
        self._refresh_act_inspection()
        refreshed = getattr(self, "_act_inspection", None)
        if refreshed is not None and previous_workspace is not None:
            refreshed.act_available = previous_workspace.act_available
            refreshed.act_message = previous_workspace.act_message
            self.act_workspace_result = refreshed

    def _on_act_cancel(self, _button: Any) -> None:
        process = self.act_job
        if process is None or process.poll() is not None:
            return
        self._act_cancel_requested = True
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:  # pragma: no cover - Windows notebook path
                process.terminate()
        except ProcessLookupError:
            pass
        def _force_kill() -> None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    if os.name != "nt":
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    else:  # pragma: no cover - Windows notebook path
                        process.kill()
                except ProcessLookupError:
                    pass

        threading.Thread(
            target=_force_kill,
            name="scp-act-cancel",
            daemon=True,
        ).start()
        self._set_status("step3", "warning", "Cancelling ACT worker and child processes…")

    def _on_act_review(self, _button: Any) -> None:
        if self.act_workspace_result is None:
            self._show_validation_error(
                "step3",
                RuntimeError("Prepare the ACT workspace before reviewing predictions."),
                output_key="step3_act",
            )
            return
        try:
            self.pipeline_state.assert_sources_unchanged()
            _assert_workspace_prepared(
                self.pipeline_state, self.act_workspace_result
            )
        except Exception as exc:
            self._show_validation_error("step3", exc, output_key="step3_act")
            return
        from modules.tuning import (
            act_output_status,
            collect_act_predictions,
            load_act_module_metrics,
        )

        predictions = collect_act_predictions(self.act_workspace_result.config_path)
        metrics = load_act_module_metrics(self.act_workspace_result.config_path)
        statuses = act_output_status(self.act_workspace_result.config_path)
        self.act_predictions = predictions
        with self.outputs["step3_act"]:
            _clear_output()
            self._print_act_review(predictions, metrics, statuses)

    def _print_act_review(
        self,
        predictions: dict[str, float],
        metrics: dict[str, list[dict[str, Any]]],
        statuses: dict[str, dict[str, Any]],
    ) -> None:
        from modules.tuning import display_tuning_rows, load_act_active_config

        config_path = (
            self.act_workspace_result.config_path
            if self.act_workspace_result is not None
            else self._act_inspection.config_path
        )
        cfg = load_act_active_config(config_path)
        workspace = Path(cfg["workspace"])
        rows = []
        for module, spec in (cfg.get("modules") or {}).items():
            provenance = statuses.get(module, {}).get("status", "missing")
            module_predictions: dict[str, Any] = {}
            module_path = workspace / f"prediction_{module}.json"
            if module_path.is_file():
                try:
                    module_predictions = json.loads(
                        module_path.read_text(encoding="utf-8")
                    )
                except Exception:
                    module_predictions = {}
            for conductance in spec.get("conductances", []) or []:
                name = str(conductance.get("variable_name"))
                rows.append(
                    {
                        "module": module,
                        "conductance": name,
                        "predicted_value": module_predictions.get(
                            name, predictions.get(name)
                        ),
                        "low": conductance.get("low"),
                        "high": conductance.get("high"),
                        "provenance": provenance,
                    }
                )
        display_tuning_rows("ACT predictions (review only; not applied)", rows)
        metric_rows = []
        for module, values in metrics.items():
            for row in values:
                metric_rows.append({"module": module, **row})
        if metric_rows:
            display_tuning_rows("ACT optimization metrics", metric_rows)
        else:
            print("No saved ACT metrics are available yet.")

    def _on_act_evaluate(self, _button: Any) -> None:
        if self.act_workspace_result is None:
            self._show_validation_error(
                "step3", RuntimeError("Prepare the ACT workspace first."), output_key="step3_act"
            )
            return
        predictions = self.act_predictions or None
        self._start_act_background(
            action="evaluate",
            operation=lambda: evaluate_fresh_act_predictions(
                self.pipeline_state,
                self.act_workspace_result,
                predictions=predictions,
                n_cpus=int(self.controls["act_n_cpus"].value),
                overwrite=bool(self.controls["act_overwrite_outputs"].value),
                display=False,
                line_callback=self._record_act_line,
                process_callback=self._record_act_process,
            ),
            success=self._complete_act_evaluation,
        )

    def _complete_act_evaluation(self, result: Any) -> None:
        self.act_evaluation_result = result
        print("ACT prediction evaluation saved:", result.manifest_path)
        print("Click Review evaluation to display the FI plot and comparison table.")

    def _on_act_review_evaluation(self, _button: Any) -> None:
        if self.act_evaluation_result is None:
            self._show_validation_error(
                "step3",
                RuntimeError("Evaluate predictions before reviewing the evaluation."),
                output_key="step3_act",
            )
            return
        try:
            self.pipeline_state.assert_sources_unchanged()
            _assert_workspace_prepared(
                self.pipeline_state, self.act_workspace_result
            )
        except Exception as exc:
            self._show_validation_error("step3", exc, output_key="step3_act")
            return
        with self.outputs["step3_act"]:
            _clear_output()
            display_act_evaluation(self.pipeline_state, self.act_evaluation_result)
            print("Evaluation manifest:", self.act_evaluation_result.manifest_path)

    def _start_act_background(self, *, action: str, operation: Any, success: Any) -> None:
        if self._act_thread is not None and self._act_thread.is_alive():
            self._show_validation_error(
                "step3", RuntimeError("Another ACT job is already running."), output_key="step3_act"
            )
            return
        if self.pipeline_state is None:
            self._show_validation_error(
                "step3", RuntimeError("Complete Step 1 first."), output_key="step3_act"
            )
            return
        self._act_cancel_requested = False
        self.act_log = ""
        self.act_job = None
        self.outputs["step3_act"].clear_output(wait=True)
        self._set_status("step3", "running", f"ACT {action} is running in a fresh process…")

        def _target() -> None:
            try:
                result = operation()
                if self._act_cancel_requested:
                    raise RuntimeError("ACT job cancelled; partial output is marked incomplete.")
                with self.outputs["step3_act"]:
                    success(result)
            except Exception as exc:
                with self.outputs["step3_act"]:
                    if self._act_cancel_requested:
                        print(
                            "ACT job cancelled. Partial artifacts were retained; "
                            "enable overwrite to rerun."
                        )
                        self._set_status(
                            "step3",
                            "warning",
                            "ACT job cancelled; partial output retained.",
                        )
                    else:
                        traceback.print_exc()
                        self._set_status("step3", "error", f"{type(exc).__name__}: {exc}")
            else:
                self._set_status("step3", "complete", f"ACT {action} complete.")
            finally:
                self.act_job = None
                self._refresh_button_states()

        self._act_thread = threading.Thread(target=_target, name=f"scp-act-{action}", daemon=True)
        self._act_thread.start()
        self._refresh_button_states()

    def _record_act_process(self, process: subprocess.Popen[str]) -> None:
        self.act_job = process
        self._refresh_button_states()

    def _record_act_line(self, line: str) -> None:
        self.act_log += line
        curated = (
            "scp act",
            "act workspace",
            "running act",
            "running module",
            "generating conductance",
            "simulat",
            "filter",
            "train",
            "predict",
            "evaluat",
            "complete",
            "done",
            "error",
        )
        if any(token in line.lower() for token in curated):
            self.outputs["step3_act"].append_stdout(line)

    def sync_from_settings(self, *, act_settings_changed: bool = False) -> None:
        if "act_active_module" in self.controls:
            value = self.settings.get("act_active_module")
            available = [
                option[1]
                for option in self.controls["act_active_module"].options
            ]
            if value in available:
                self.controls["act_active_module"].value = value
        if "act_workspace_override" not in self.controls:
            return
        self._act_syncing = True
        try:
            self.controls["act_workspace_override"].value = _optional_text(
                self.settings.get("act_workspace_override")
            )
            self.controls["act_overwrite_outputs"].value = bool(
                self.settings.get("act_overwrite_outputs", False)
            )
            if self.settings.get("act_n_cpus") is not None:
                self.controls["act_n_cpus"].value = max(
                    1,
                    min(
                        int(self.settings["act_n_cpus"]),
                        self.controls["act_n_cpus"].max,
                    ),
                )
        finally:
            self._act_syncing = False
        if act_settings_changed:
            self.act_workspace_result = None
            if self.pipeline_state is not None:
                self._refresh_act_inspection()

    def refresh_button_states(self, *, ready: bool, act_busy: bool) -> None:
        if "step3_act_prepare" not in self.controls:
            return
        inspection = getattr(self, "_act_inspection", None)
        prepared = self.act_workspace_result is not None
        self.controls["step3_act_prepare"].disabled = (
            not ready or inspection is None or act_busy
        )
        selected = self.controls["act_active_module"].value
        output_status = self.act_workspace_result.output_status if prepared else {}
        selected_keys = list(output_status) if selected == "all" else [selected]
        rerun_exists = any(
            output_status.get(key, {}).get("status") != "missing"
            for key in selected_keys
        )
        overwrite = bool(self.controls["act_overwrite_outputs"].value)
        self.controls["step3_act_run"].disabled = (
            not ready
            or not prepared
            or not self.act_workspace_result.act_available
            or act_busy
            or (rerun_exists and not overwrite)
        )
        predictions_available = bool(self.act_predictions) or any(
            item.get("status") in {"current", "stale", "legacy"}
            for item in output_status.values()
        )
        self.controls["step3_act_review"].disabled = (
            not ready or not prepared or act_busy or not predictions_available
        )
        self.controls["step3_act_evaluate"].disabled = (
            not ready
            or not prepared
            or act_busy
            or not predictions_available
            or not self.act_workspace_result.act_available
        )
        self.controls["step3_act_review_evaluation"].disabled = (
            not ready or act_busy or self.act_evaluation_result is None
        )
        self.controls["step3_act_cancel"].disabled = not (
            act_busy
            and self.act_job is not None
            and self.act_job.poll() is None
        )


__all__ = ["Step3ACTUI"]
