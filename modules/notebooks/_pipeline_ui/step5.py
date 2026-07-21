"""Step 5 input-preview, simulation, and result-plotting UI."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from ..pipeline_workflow import preview_pipeline_inputs, run_fresh_simulation
from ..run_diagnostics import show_run_diagnostics
from ..synapse_preview import SYNAPSE_PREVIEW_PLOTS, show_synapse_preview
from .common import (
    PIPELINE_UI_DEFAULTS,
    PipelineUIComponent,
    merge_non_none as _merge_non_none,
    nested_value as _nested_value,
    optional_text as _optional_text,
    parse_optional_float as _parse_optional_float,
    parse_optional_int as _parse_optional_int,
    set_nested_value as _set_nested_value,
)


_DIAGNOSTIC_PLOTS = (
    ("Input rate", "input_rate"),
    ("Input raster", "input_raster"),
    ("Membrane voltage", "membrane_voltage"),
    ("Output firing rate", "output_rate"),
    ("Output raster", "output_raster"),
)


class Step5UI(PipelineUIComponent):
    def build_panel(self) -> Any:
        if "step5" in self.panels:
            return self.panels["step5"]
        w = self.widgets
        trials = w.BoundedIntText(
            value=max(1, int(self.settings.get("n_trials", 1))),
            min=1,
            max=1_000_000,
            description="Trials",
            layout=w.Layout(width="220px"),
        )
        seed = w.Text(
            value=_optional_text(self.settings.get("seed")),
            description="Shared seed",
            placeholder="optional",
            layout=w.Layout(width="240px"),
        )
        mode = w.Dropdown(
            options=[("Synapse inputs", "synapse"), ("Current clamp", "iclamp")],
            value="iclamp" if self.settings.get("run_iclamp") else "synapse",
            description="Mode",
            layout=w.Layout(width="260px"),
        )
        output_stem = w.Text(
            value=_optional_text(self.settings.get("output_stem")),
            description="Output stem",
            placeholder="blank = unique pipeline timestamp",
            layout=w.Layout(width="440px"),
        )
        quiet_preview = w.Checkbox(
            value=bool(self.settings.get("quiet_input_preview_output", True)),
            description="Quiet preview",
            indent=False,
            tooltip=(
                "Hide routine fresh-process output. Full details remain "
                "available as pipeline_ui.input_preview_log."
            ),
            layout=w.Layout(width="170px"),
        )
        quiet_run = w.Checkbox(
            value=bool(self.settings.get("quiet_simulation_output", True)),
            description="Quiet run",
            indent=False,
            tooltip=(
                "Hide routine fresh-process output. Full details remain "
                "available as pipeline_ui.simulation_log."
            ),
            layout=w.Layout(width="150px"),
        )
        sim_cfg = self._effective_simulation_config()
        sim_options_toggle = w.ToggleButton(
            value=False,
            description="Show advanced options",
            icon="sliders",
            layout=w.Layout(width="200px"),
        )
        sim_tstart = w.FloatText(
            value=float(sim_cfg.get("tstart", 0.0)),
            description="Start (ms)",
            layout=w.Layout(width="210px"),
        )
        sim_tstop = w.FloatText(
            value=float(sim_cfg.get("tstop", 1000.0)),
            description="Stop (ms)",
            layout=w.Layout(width="210px"),
        )
        sim_dt = w.FloatText(
            value=float(sim_cfg.get("dt", 0.025)),
            description="dt (ms)",
            layout=w.Layout(width="190px"),
        )
        sim_bins = w.FloatText(
            value=float(sim_cfg.get("bins", 5.0)),
            description="Bin (ms)",
            layout=w.Layout(width="190px"),
        )
        sim_stim_start = w.FloatText(
            value=float(sim_cfg.get("stim_start_ms", 300.0)),
            description="Stim start (ms)",
            layout=w.Layout(width="230px"),
        )
        sim_stim_duration = w.FloatText(
            value=float(sim_cfg.get("stim_duration_ms", 500.0)),
            description="Stim duration (ms)",
            layout=w.Layout(width="250px"),
        )
        sim_v_init = w.Text(
            value=_optional_text(
                _nested_value(sim_cfg, ("conditions", "v_init_mV"))
            ),
            description="Initial V (mV)",
            placeholder="configured/loader default",
            layout=w.Layout(width="280px"),
            style={"description_width": "110px"},
        )
        sim_celsius = w.Text(
            value=_optional_text(
                _nested_value(sim_cfg, ("conditions", "celsius_C"))
            ),
            description="Temperature (°C)",
            placeholder="configured/loader default",
            layout=w.Layout(width="300px"),
            style={"description_width": "125px"},
        )
        sim_jitter = w.Text(
            value=_optional_text(sim_cfg.get("jitter")),
            description="Jitter (ms)",
            placeholder="blank = none/configured",
            layout=w.Layout(width="260px"),
        )
        sim_randomness_mode = w.Dropdown(
            options=[
                ("Random each run", "random"),
                ("Derived/reproducible", "derived"),
                ("Fixed components", "fixed"),
            ],
            value=(
                sim_cfg.get("randomness_mode")
                if sim_cfg.get("randomness_mode") in {"random", "derived", "fixed"}
                else "random"
            ),
            description="Randomness",
            layout=w.Layout(width="290px"),
        )
        iclamp_cfg = sim_cfg.get("iclamp")
        if not isinstance(iclamp_cfg, Mapping):
            iclamp_cfg = {}
        sim_iclamp_amp = w.FloatText(
            value=float(iclamp_cfg.get("amp_nA", 0.2)),
            description="Amplitude (nA)",
            layout=w.Layout(width="240px"),
        )
        sim_iclamp_delay = w.Text(
            value=_optional_text(iclamp_cfg.get("delay_ms")),
            description="Delay (ms)",
            placeholder="blank = stim start",
            layout=w.Layout(width="245px"),
        )
        sim_iclamp_duration = w.Text(
            value=_optional_text(iclamp_cfg.get("dur_ms")),
            description="Duration (ms)",
            placeholder="blank = stim duration",
            layout=w.Layout(width="270px"),
        )
        sim_iclamp_tstop = w.Text(
            value=_optional_text(iclamp_cfg.get("tstop_ms")),
            description="Stop (ms)",
            placeholder="blank = run stop",
            layout=w.Layout(width="240px"),
        )
        sim_iclamp_dt = w.Text(
            value=_optional_text(iclamp_cfg.get("dt_ms")),
            description="dt (ms)",
            placeholder="blank = run dt",
            layout=w.Layout(width="220px"),
        )
        sim_iclamp_currents = w.Checkbox(
            value=bool(iclamp_cfg.get("record_currents", False)),
            description="Record ionic currents",
            indent=False,
            layout=w.Layout(width="210px"),
        )
        sim_traces = w.BoundedIntText(
            value=max(0, int(sim_cfg.get("n_traces_to_save", 1))),
            min=0,
            max=1_000_000,
            description="Trace trials",
            layout=w.Layout(width="220px"),
        )
        sim_inputs = w.Text(
            value=_optional_text(sim_cfg.get("n_inputs_to_save", 1)),
            description="Input trials",
            placeholder="number or all",
            layout=w.Layout(width="220px"),
        )
        sim_save_input_stats = w.Checkbox(
            value=bool(sim_cfg.get("save_input_stats", True)),
            description="Save input statistics",
            indent=False,
            layout=w.Layout(width="190px"),
        )
        sim_input_stats_bin = w.FloatText(
            value=float(sim_cfg.get("input_stats_bin_ms", 5.0)),
            description="Stats bin (ms)",
            layout=w.Layout(width="230px"),
        )
        plot_profile = sim_cfg.get("plots_profile")
        if plot_profile not in {"off", "basic", "inputs", "full"}:
            plot_profile = "basic"
        sim_plots_profile = w.Dropdown(
            options=[
                ("Off", "off"),
                ("Basic output", "basic"),
                ("Output + inputs", "inputs"),
                ("Full", "full"),
            ],
            value=plot_profile,
            description="Auto plots",
            layout=w.Layout(width="270px"),
        )
        sim_save_plots_mode = w.Dropdown(
            options=[("Single combined plot", "single_plot"), ("Default", "default")],
            value=(
                sim_cfg.get("save_plots_mode")
                if sim_cfg.get("save_plots_mode") in {"single_plot", "default"}
                else "single_plot"
            ),
            description="Plot layout",
            layout=w.Layout(width="300px"),
        )
        sim_save_plots_overwrite = w.Checkbox(
            value=bool(sim_cfg.get("save_plots_overwrite", False)),
            description="Overwrite auto plots",
            indent=False,
            layout=w.Layout(width="190px"),
        )
        save_cfg = sim_cfg.get("save")
        if not isinstance(save_cfg, Mapping):
            save_cfg = {}
        output_format = save_cfg.get("format", sim_cfg.get("output_format", "pkl"))
        sim_output_format = w.Dropdown(
            options=[("Pickle", "pkl"), ("NumPy NPZ", "npz")],
            value=output_format if output_format in {"pkl", "npz"} else "pkl",
            description="Full format",
            layout=w.Layout(width="245px"),
        )
        sim_full_results = w.Checkbox(
            value=bool(
                save_cfg.get(
                    "full_results",
                    sim_cfg.get("save_full_results", False),
                )
            ),
            description="Save full result bundle",
            indent=False,
            layout=w.Layout(width="210px"),
        )
        cell_recording = sim_cfg.get("cell_recording")
        if not isinstance(cell_recording, Mapping):
            cell_recording = {}
        recording_vars = cell_recording.get("vars")
        if not isinstance(recording_vars, Mapping):
            recording_vars = {}
        sim_cell_recording = w.Checkbox(
            value=bool(cell_recording.get("enabled", False)),
            description="Detailed cell recording",
            indent=False,
            layout=w.Layout(width="205px"),
        )
        sim_record_ion_currents = w.Checkbox(
            value=bool(recording_vars.get("ion_currents", False)),
            description="Ionic currents",
            indent=False,
            layout=w.Layout(width="150px"),
        )
        sim_record_mech_currents = w.Checkbox(
            value=bool(recording_vars.get("mech_currents", False)),
            description="Mechanism currents",
            indent=False,
            layout=w.Layout(width="180px"),
        )
        sim_record_ion_concentrations = w.Checkbox(
            value=bool(recording_vars.get("ion_concentrations", False)),
            description="Ion concentrations",
            indent=False,
            layout=w.Layout(width="175px"),
        )
        sim_iclamp_box = w.VBox(
            [
                w.HTML("<b>Current clamp protocol</b>"),
                w.HBox(
                    [sim_iclamp_amp, sim_iclamp_delay, sim_iclamp_duration],
                    layout=w.Layout(flex_flow="row wrap"),
                ),
                w.HBox(
                    [sim_iclamp_tstop, sim_iclamp_dt, sim_iclamp_currents],
                    layout=w.Layout(flex_flow="row wrap"),
                ),
            ],
            layout=w.Layout(
                display="" if mode.value == "iclamp" else "none",
                border="1px solid #ececec",
                padding="6px",
                margin="4px 0",
            ),
        )
        sim_advanced_box = w.VBox(
            [
                w.HTML(
                    "<span style='font-size:90%; color:#555'>"
                    "Values are loaded from the current <code>sim_config.json</code>. "
                    "Changes are temporary run overrides and never edit that file. "
                    "Load/append, snapshot, custom recording sites, and specialized "
                    "plot presets remain file-based."
                    "</span>"
                ),
                w.HTML("<b>Timing and runtime conditions</b>"),
                w.HBox(
                    [sim_tstart, sim_tstop, sim_dt, sim_bins],
                    layout=w.Layout(flex_flow="row wrap"),
                ),
                w.HBox(
                    [sim_stim_start, sim_stim_duration, sim_v_init, sim_celsius],
                    layout=w.Layout(flex_flow="row wrap"),
                ),
                w.HTML("<b>Randomness</b>"),
                w.HBox(
                    [sim_randomness_mode, sim_jitter],
                    layout=w.Layout(flex_flow="row wrap"),
                ),
                sim_iclamp_box,
                w.HTML("<b>Saved samples and automatic plots</b>"),
                w.HBox(
                    [sim_traces, sim_inputs, sim_save_input_stats, sim_input_stats_bin],
                    layout=w.Layout(flex_flow="row wrap"),
                ),
                w.HBox(
                    [sim_plots_profile, sim_save_plots_mode, sim_save_plots_overwrite],
                    layout=w.Layout(flex_flow="row wrap"),
                ),
                w.HBox(
                    [sim_output_format, sim_full_results],
                    layout=w.Layout(flex_flow="row wrap"),
                ),
                w.HTML("<b>Detailed cell recording</b>"),
                w.HBox(
                    [
                        sim_cell_recording,
                        sim_record_ion_currents,
                        sim_record_mech_currents,
                        sim_record_ion_concentrations,
                    ],
                    layout=w.Layout(flex_flow="row wrap"),
                ),
            ],
            layout=w.Layout(display="none", padding="4px 0"),
        )
        configured_groups = self._configured_input_preview_groups()
        group_options, selected_groups = self._input_preview_group_selection(
            configured_groups
        )
        preview_groups = w.SelectMultiple(
            options=group_options or [("No configured groups found", "")],
            value=tuple(selected_groups),
            description="Groups",
            tooltip="Select one or more configured synapse groups to display.",
            layout=w.Layout(width="390px", height="88px"),
            style={"description_width": "60px"},
        )
        configured_plots = self.settings.get("input_preview_plots")
        if configured_plots is None:
            configured_plots = SYNAPSE_PREVIEW_PLOTS
        selected_plots = [
            plot
            for plot in configured_plots
            if plot in SYNAPSE_PREVIEW_PLOTS
        ]
        preview_plots = w.SelectMultiple(
            options=[
                ("Weight distribution", "weight_distribution"),
                ("Distance distribution", "distance_distribution"),
                ("Weight vs distance", "weight_vs_distance"),
            ],
            value=tuple(selected_plots),
            description="Plots",
            tooltip="Select the plots to include in the compact preview figure.",
            layout=w.Layout(width="390px", height="88px"),
            style={"description_width": "55px"},
        )
        preview_options_toggle = w.ToggleButton(
            value=False,
            description="Show advanced options",
            icon="sliders",
            layout=w.Layout(width="200px"),
        )
        preview_trial = w.BoundedIntText(
            value=max(0, int(self.settings.get("input_preview_trial_idx", 0))),
            min=0,
            max=1_000_000,
            description="Trial index",
            layout=w.Layout(width="190px"),
        )
        preview_table = w.Checkbox(
            value=bool(self.settings.get("input_preview_show_table", True)),
            description="Show summary table",
            indent=False,
            layout=w.Layout(width="190px"),
        )
        histogram_mode = w.Dropdown(
            options=[("Probability density", "density"), ("Counts", "count")],
            value=(
                "density"
                if self.settings.get("input_preview_histogram_density", True)
                else "count"
            ),
            description="Histograms",
            layout=w.Layout(width="260px"),
        )
        distance_bin = w.FloatText(
            value=float(self.settings.get("input_preview_distance_bin_um", 25.0)),
            description="Distance bin (µm)",
            layout=w.Layout(width="240px"),
            style={"description_width": "125px"},
        )
        weight_bin = w.Text(
            value=_optional_text(self.settings.get("input_preview_weight_bin")),
            description="Weight bin",
            placeholder="blank = automatic",
            layout=w.Layout(width="250px"),
        )
        plot_columns = w.Dropdown(
            options=[
                ("Single row", 3),
                ("Up to 2 per row", 2),
                ("One per row", 1),
            ],
            value=max(
                1,
                min(3, int(self.settings.get("input_preview_plot_columns", 3))),
            ),
            description="Layout",
            layout=w.Layout(width="240px"),
        )
        plot_size = w.Dropdown(
            options=[("Compact", "compact"), ("Standard", "standard")],
            value=(
                self.settings.get("input_preview_plot_size")
                if self.settings.get("input_preview_plot_size")
                in {"compact", "standard"}
                else "compact"
            ),
            description="Plot size",
            layout=w.Layout(width="220px"),
        )
        preview_advanced_box = w.VBox(
            [
                w.HTML(
                    "<span style='font-size:90%; color:#555'>"
                    "These settings change only the preview display and sampled "
                    "trial; they do not edit synapse JSON files."
                    "</span>"
                ),
                w.HBox([preview_trial, preview_table, histogram_mode]),
                w.HBox([distance_bin, weight_bin]),
                w.HBox([plot_columns, plot_size]),
            ],
            layout=w.Layout(display="none"),
        )
        requested_diagnostic_plots = self.settings.get("diagnostic_plots")
        if not isinstance(requested_diagnostic_plots, Sequence) or isinstance(
            requested_diagnostic_plots, (str, bytes, bytearray)
        ):
            requested_diagnostic_plots = PIPELINE_UI_DEFAULTS["diagnostic_plots"]
        diagnostic_plot_values = {value for _label, value in _DIAGNOSTIC_PLOTS}
        selected_diagnostic_plots = tuple(
            value
            for value in requested_diagnostic_plots
            if value in diagnostic_plot_values
        )
        diagnostic_plots = w.SelectMultiple(
            options=_DIAGNOSTIC_PLOTS,
            value=selected_diagnostic_plots,
            description="Plots",
            tooltip="Select the panels in the compact shared-time-axis figure.",
            layout=w.Layout(width="390px", height="120px"),
            style={"description_width": "55px"},
        )
        diagnostic_trial = w.Dropdown(
            options=[("Trial 0", 0)],
            value=0,
            description="Trial",
            disabled=True,
            layout=w.Layout(width="210px"),
        )
        diagnostic_options_toggle = w.ToggleButton(
            value=False,
            description="Show advanced options",
            icon="sliders",
            layout=w.Layout(width="200px"),
        )
        window_mode = self.settings.get("diagnostic_window_mode", "stimulus")
        if window_mode not in {"stimulus", "full", "manual"}:
            window_mode = "stimulus"
        diagnostic_window_mode = w.Dropdown(
            options=[
                ("Around stimulus", "stimulus"),
                ("Full simulation", "full"),
                ("Manual", "manual"),
            ],
            value=window_mode,
            description="Window",
            layout=w.Layout(width="260px"),
        )
        diagnostic_window_start = w.Text(
            value=_optional_text(
                self.settings.get("diagnostic_window_start_ms")
            ),
            description="Start (ms)",
            placeholder="required for manual",
            layout=w.Layout(width="240px"),
        )
        diagnostic_window_stop = w.Text(
            value=_optional_text(
                self.settings.get("diagnostic_window_stop_ms")
            ),
            description="Stop (ms)",
            placeholder="required for manual",
            layout=w.Layout(width="240px"),
        )
        diagnostic_window_padding = w.FloatText(
            value=float(
                self.settings.get("diagnostic_window_padding_ms", 100.0)
            ),
            description="Padding (ms)",
            layout=w.Layout(width="235px"),
        )
        diagnostic_manual_window_box = w.HBox(
            [diagnostic_window_start, diagnostic_window_stop],
            layout=w.Layout(
                display="" if window_mode == "manual" else "none",
                flex_flow="row wrap",
            ),
        )
        diagnostic_padding_box = w.HBox(
            [diagnostic_window_padding],
            layout=w.Layout(
                display="" if window_mode == "stimulus" else "none"
            ),
        )
        configured_rate_bin = self.settings.get("diagnostic_rate_bin_ms")
        if configured_rate_bin is None:
            configured_rate_bin = sim_cfg.get("plots_input_bin_ms")
        if configured_rate_bin is None:
            configured_rate_bin = sim_cfg.get(
                "input_stats_bin_ms", sim_cfg.get("bins", 5.0)
            )
        configured_smoothing = self.settings.get("diagnostic_smoothing_ms")
        if configured_smoothing is None:
            configured_smoothing = sim_cfg.get(
                "plots_input_smooth_ms", sim_cfg.get("plots_win_size", 25.0)
            )
        diagnostic_rate_bin = w.FloatText(
            value=float(configured_rate_bin),
            description="Rate bin (ms)",
            layout=w.Layout(width="235px"),
        )
        diagnostic_smoothing = w.FloatText(
            value=float(configured_smoothing),
            description="Smoothing (ms)",
            layout=w.Layout(width="250px"),
        )
        diagnostic_raster_style = w.Dropdown(
            options=[("Dots", "dot"), ("Lines", "line")],
            value=(
                self.settings.get("diagnostic_raster_style")
                if self.settings.get("diagnostic_raster_style") in {"dot", "line"}
                else "dot"
            ),
            description="Raster",
            layout=w.Layout(width="220px"),
        )
        diagnostic_groups = w.SelectMultiple(
            options=[("Run a simulation to discover groups", "")],
            value=(),
            description="Input groups",
            disabled=True,
            tooltip="Groups used by the input-rate and input-raster panels.",
            layout=w.Layout(width="390px", height="100px"),
            style={"description_width": "90px"},
        )
        diagnostic_show_stimulus = w.Checkbox(
            value=bool(self.settings.get("diagnostic_show_stimulus", True)),
            description="Show stimulus markers",
            indent=False,
            layout=w.Layout(width="190px"),
        )
        diagnostic_figure_size = w.Dropdown(
            options=[("Compact", "compact"), ("Standard", "standard")],
            value=(
                self.settings.get("diagnostic_figure_size")
                if self.settings.get("diagnostic_figure_size")
                in {"compact", "standard"}
                else "compact"
            ),
            description="Figure size",
            layout=w.Layout(width="230px"),
        )
        diagnostic_advanced_box = w.VBox(
            [
                w.HTML(
                    "<span style='font-size:90%; color:#555'>"
                    "These options affect only the displayed figure. Use Step 6 "
                    "for comparisons, normalization, metrics, styling, and export."
                    "</span>"
                ),
                w.HBox(
                    [diagnostic_window_mode, diagnostic_padding_box],
                    layout=w.Layout(flex_flow="row wrap"),
                ),
                diagnostic_manual_window_box,
                w.HBox(
                    [
                        diagnostic_rate_bin,
                        diagnostic_smoothing,
                        diagnostic_raster_style,
                    ],
                    layout=w.Layout(flex_flow="row wrap"),
                ),
                w.HBox(
                    [diagnostic_groups, diagnostic_show_stimulus, diagnostic_figure_size],
                    layout=w.Layout(flex_flow="row wrap"),
                ),
            ],
            layout=w.Layout(display="none", padding="4px 0"),
        )
        check_inputs = self._run_button("Check inputs", icon="search")
        run_button = self._run_button("Run simulation")
        plot_button = self._run_button("Plot results", icon="bar-chart")
        status = w.HTML()
        check_inputs_output = w.Output()
        simulation_output = w.Output()
        plot_output = w.Output()
        self.controls.update(
            {
                "n_trials": trials,
                "seed": seed,
                "run_mode": mode,
                "output_stem": output_stem,
                "quiet_input_preview_output": quiet_preview,
                "quiet_simulation_output": quiet_run,
                "simulation_options_toggle": sim_options_toggle,
                "simulation_options_box": sim_advanced_box,
                "sim_tstart_ms": sim_tstart,
                "sim_tstop_ms": sim_tstop,
                "sim_dt_ms": sim_dt,
                "sim_bins_ms": sim_bins,
                "sim_stim_start_ms": sim_stim_start,
                "sim_stim_duration_ms": sim_stim_duration,
                "sim_v_init_mV": sim_v_init,
                "sim_celsius_C": sim_celsius,
                "sim_jitter_ms": sim_jitter,
                "sim_randomness_mode": sim_randomness_mode,
                "sim_iclamp_box": sim_iclamp_box,
                "sim_iclamp_amp_nA": sim_iclamp_amp,
                "sim_iclamp_delay_ms": sim_iclamp_delay,
                "sim_iclamp_duration_ms": sim_iclamp_duration,
                "sim_iclamp_tstop_ms": sim_iclamp_tstop,
                "sim_iclamp_dt_ms": sim_iclamp_dt,
                "sim_iclamp_record_currents": sim_iclamp_currents,
                "sim_n_traces_to_save": sim_traces,
                "sim_n_inputs_to_save": sim_inputs,
                "sim_save_input_stats": sim_save_input_stats,
                "sim_input_stats_bin_ms": sim_input_stats_bin,
                "sim_plots_profile": sim_plots_profile,
                "sim_save_plots_mode": sim_save_plots_mode,
                "sim_save_plots_overwrite": sim_save_plots_overwrite,
                "sim_output_format": sim_output_format,
                "sim_save_full_results": sim_full_results,
                "sim_cell_recording_enabled": sim_cell_recording,
                "sim_record_ion_currents": sim_record_ion_currents,
                "sim_record_mech_currents": sim_record_mech_currents,
                "sim_record_ion_concentrations": sim_record_ion_concentrations,
                "input_preview_groups": preview_groups,
                "input_preview_plots": preview_plots,
                "input_preview_options_toggle": preview_options_toggle,
                "input_preview_options_box": preview_advanced_box,
                "input_preview_trial_idx": preview_trial,
                "input_preview_show_table": preview_table,
                "input_preview_histogram_mode": histogram_mode,
                "input_preview_distance_bin_um": distance_bin,
                "input_preview_weight_bin": weight_bin,
                "input_preview_plot_columns": plot_columns,
                "input_preview_plot_size": plot_size,
                "diagnostic_plots": diagnostic_plots,
                "diagnostic_trial_idx": diagnostic_trial,
                "diagnostic_options_toggle": diagnostic_options_toggle,
                "diagnostic_options_box": diagnostic_advanced_box,
                "diagnostic_window_mode": diagnostic_window_mode,
                "diagnostic_window_start_ms": diagnostic_window_start,
                "diagnostic_window_stop_ms": diagnostic_window_stop,
                "diagnostic_window_padding_ms": diagnostic_window_padding,
                "diagnostic_manual_window_box": diagnostic_manual_window_box,
                "diagnostic_padding_box": diagnostic_padding_box,
                "diagnostic_rate_bin_ms": diagnostic_rate_bin,
                "diagnostic_smoothing_ms": diagnostic_smoothing,
                "diagnostic_raster_style": diagnostic_raster_style,
                "diagnostic_input_groups": diagnostic_groups,
                "diagnostic_show_stimulus": diagnostic_show_stimulus,
                "diagnostic_figure_size": diagnostic_figure_size,
                "step5_check_inputs": check_inputs,
                "step5_run": run_button,
                "step5_plot": plot_button,
            }
        )
        # Preserve the original key as an alias for the primary simulation
        # action while exposing each card's output independently.
        self.outputs["step5"] = simulation_output
        self.outputs["step5_check_inputs"] = check_inputs_output
        self.outputs["step5_run"] = simulation_output
        self.outputs["step5_plot"] = plot_output
        self.statuses["step5"] = status
        self._observe_value(trials, "n_trials", int)
        self._observe_valid_optional_int(seed, "seed", positive=False)
        mode.observe(self._on_mode_changed, names="value")
        self._observe_value(
            output_stem,
            "output_stem",
            lambda value: str(value).strip() or None,
        )
        self._observe_value(
            quiet_preview,
            "quiet_input_preview_output",
            bool,
        )
        self._observe_value(
            quiet_run,
            "quiet_simulation_output",
            bool,
        )
        sim_options_toggle.observe(
            self._on_simulation_options_toggled,
            names="value",
        )
        for control_name in (
            "sim_tstart_ms",
            "sim_tstop_ms",
            "sim_dt_ms",
            "sim_bins_ms",
            "sim_stim_start_ms",
            "sim_stim_duration_ms",
            "sim_v_init_mV",
            "sim_celsius_C",
            "sim_jitter_ms",
            "sim_randomness_mode",
            "sim_iclamp_amp_nA",
            "sim_iclamp_delay_ms",
            "sim_iclamp_duration_ms",
            "sim_iclamp_tstop_ms",
            "sim_iclamp_dt_ms",
            "sim_iclamp_record_currents",
            "sim_n_traces_to_save",
            "sim_n_inputs_to_save",
            "sim_save_input_stats",
            "sim_input_stats_bin_ms",
            "sim_plots_profile",
            "sim_save_plots_mode",
            "sim_save_plots_overwrite",
            "sim_output_format",
            "sim_save_full_results",
            "sim_cell_recording_enabled",
            "sim_record_ion_currents",
            "sim_record_mech_currents",
            "sim_record_ion_concentrations",
        ):
            self.controls[control_name].observe(
                lambda change, name=control_name: self._on_simulation_option_changed(
                    name, change
                ),
                names="value",
            )
        preview_groups.observe(
            self._on_input_preview_groups_changed,
            names="value",
        )
        preview_plots.observe(
            lambda change: self.settings.__setitem__(
                "input_preview_plots", list(change["new"])
            ),
            names="value",
        )
        preview_options_toggle.observe(
            self._on_input_preview_options_toggled,
            names="value",
        )
        self._observe_value(preview_trial, "input_preview_trial_idx", int)
        self._observe_value(preview_table, "input_preview_show_table", bool)
        histogram_mode.observe(
            lambda change: self.settings.__setitem__(
                "input_preview_histogram_density", change["new"] == "density"
            ),
            names="value",
        )
        self._observe_value(
            distance_bin,
            "input_preview_distance_bin_um",
            float,
        )
        self._observe_valid_optional_float_or_none(
            weight_bin,
            "input_preview_weight_bin",
        )
        self._observe_value(plot_columns, "input_preview_plot_columns", int)
        self._observe_value(plot_size, "input_preview_plot_size", str)
        diagnostic_plots.observe(
            self._on_diagnostic_plots_changed,
            names="value",
        )
        diagnostic_trial.observe(
            self._on_diagnostic_trial_changed,
            names="value",
        )
        diagnostic_options_toggle.observe(
            self._on_diagnostic_options_toggled,
            names="value",
        )
        diagnostic_window_mode.observe(
            self._on_diagnostic_window_mode_changed,
            names="value",
        )
        self._observe_valid_optional_float_or_none(
            diagnostic_window_start,
            "diagnostic_window_start_ms",
        )
        self._observe_valid_optional_float_or_none(
            diagnostic_window_stop,
            "diagnostic_window_stop_ms",
        )
        self._observe_value(
            diagnostic_window_padding,
            "diagnostic_window_padding_ms",
            float,
        )
        self._observe_value(
            diagnostic_rate_bin,
            "diagnostic_rate_bin_ms",
            float,
        )
        self._observe_value(
            diagnostic_smoothing,
            "diagnostic_smoothing_ms",
            float,
        )
        self._observe_value(
            diagnostic_raster_style,
            "diagnostic_raster_style",
            str,
        )
        diagnostic_groups.observe(
            self._on_diagnostic_groups_changed,
            names="value",
        )
        self._observe_value(
            diagnostic_show_stimulus,
            "diagnostic_show_stimulus",
            bool,
        )
        self._observe_value(
            diagnostic_figure_size,
            "diagnostic_figure_size",
            str,
        )
        check_inputs.on_click(self._on_input_preview)
        run_button.on_click(self._on_simulation)
        plot_button.on_click(self._on_plot_results)
        check_inputs_panel = self._build_check_inputs_card()
        simulation_panel = self._build_simulation_card()
        plot_panel = self._build_plot_results_card()
        panel = w.VBox(
            [check_inputs_panel, simulation_panel, plot_panel, status]
        )
        self.panels["step5"] = panel
        self._refresh_diagnostic_controls()
        self._refresh_button_states()
        return panel

    def _step5_card_layout(self) -> Any:
        return self.widgets.Layout(
            border="1px solid #d9d9d9",
            padding="8px",
            margin="0 0 8px 0",
        )

    def _build_check_inputs_card(self) -> Any:
        w = self.widgets
        return w.VBox(
            [
                w.HTML("<b>Check Inputs</b>"),
                w.HTML(
                    "<span style='font-size:90%; color:#555'>"
                    "Preview sampled synapse placement, distance, and weight "
                    "distributions before running. The optional seed is shared "
                    "with the simulation so trial 0 can be reproduced."
                    "</span>"
                ),
                w.HBox(
                    [
                        self.controls["input_preview_groups"],
                        self.controls["input_preview_plots"],
                    ],
                    layout=w.Layout(flex_flow="row wrap"),
                ),
                w.HBox(
                    [
                        self.controls["seed"],
                        self.controls["quiet_input_preview_output"],
                        self.controls["input_preview_options_toggle"],
                        self.controls["step5_check_inputs"],
                    ],
                    layout=w.Layout(flex_flow="row wrap"),
                ),
                self.controls["input_preview_options_box"],
                self.outputs["step5_check_inputs"],
            ],
            layout=self._step5_card_layout(),
        )

    def _build_simulation_card(self) -> Any:
        w = self.widgets
        return w.VBox(
            [
                w.HTML("<b>Run Simulation</b>"),
                w.HTML(
                    "<span style='font-size:90%; color:#555'>"
                    "Run Step 5 in a fresh process, reload the current JSON "
                    "configuration, and save a uniquely named result."
                    "</span>"
                ),
                w.HBox(
                    [
                        self.controls["n_trials"],
                        self.controls["run_mode"],
                        self.controls["quiet_simulation_output"],
                    ]
                ),
                w.HBox(
                    [
                        self.controls["output_stem"],
                        self.controls["simulation_options_toggle"],
                        self.controls["step5_run"],
                    ],
                    layout=w.Layout(flex_flow="row wrap"),
                ),
                self.controls["simulation_options_box"],
                self.outputs["step5_run"],
            ],
            layout=self._step5_card_layout(),
        )

    def _build_plot_results_card(self) -> Any:
        w = self.widgets
        return w.VBox(
            [
                w.HTML("<b>Plot Results</b>"),
                w.HTML(
                    "<span style='font-size:90%; color:#555'>"
                    "Display diagnostics from the latest successful simulation "
                    "without rerunning it."
                    "</span>"
                ),
                w.HBox(
                    [
                        self.controls["diagnostic_plots"],
                        self.controls["diagnostic_trial_idx"],
                    ],
                    layout=w.Layout(flex_flow="row wrap"),
                ),
                w.HBox(
                    [
                        self.controls["diagnostic_options_toggle"],
                        self.controls["step5_plot"],
                    ]
                ),
                self.controls["diagnostic_options_box"],
                self.outputs["step5_plot"],
            ],
            layout=self._step5_card_layout(),
        )

    def _simulation_config(self) -> dict[str, Any]:
        """Read the selected tune's current simulation config for UI defaults."""

        defaults: dict[str, Any] = {
            "tstart": 0.0,
            "tstop": 1000.0,
            "dt": 0.025,
            "bins": 5.0,
            "jitter": None,
            "stim_start_ms": 300.0,
            "stim_duration_ms": 500.0,
            "randomness_mode": "random",
            "trial_randomness": "synapses",
            "n_traces_to_save": 1,
            "n_inputs_to_save": 1,
            "save_input_stats": True,
            "input_stats_bin_ms": 5.0,
            "plots_profile": "basic",
            "save_plots_mode": "single_plot",
            "save_plots_overwrite": False,
            "save": {"format": "pkl", "full_results": False},
            "iclamp": {
                "enabled": False,
                "amp_nA": 0.2,
                "delay_ms": None,
                "dur_ms": None,
                "tstop_ms": None,
                "dt_ms": None,
                "record_currents": False,
            },
            "cell_recording": {
                "enabled": False,
                "n_trials": 1,
                "vars": {
                    "ion_currents": False,
                    "mech_currents": False,
                    "ion_concentrations": False,
                },
            },
            "conditions": {"v_init_mV": None, "celsius_C": None},
        }
        try:
            from modules.input_generation.config import _resolve_config_root

            path = _resolve_config_root(self._selected_tune_dir()) / "sim_config.json"
            with path.open("r", encoding="utf-8") as handle:
                configured = json.load(handle)
            if not isinstance(configured, dict):
                return defaults
        except Exception:
            return defaults
        return dict(_merge_non_none(defaults, configured))

    def _effective_simulation_config(self) -> dict[str, Any]:
        configured = self._simulation_config()
        overrides = self.settings.get("simulation_overrides")
        if not isinstance(overrides, Mapping):
            overrides = {}
        return dict(_merge_non_none(configured, overrides))

    def _set_simulation_override(
        self, path: Sequence[str], value: Any
    ) -> None:
        overrides = self.settings.get("simulation_overrides")
        if not isinstance(overrides, MutableMapping):
            overrides = {}
            self.settings["simulation_overrides"] = overrides
        _set_nested_value(overrides, path, value)

    @staticmethod
    def _parse_inputs_to_save(value: Any) -> Any:
        text = str(value).strip()
        if not text:
            return None
        if text.lower() == "all":
            return "all"
        try:
            parsed = int(text)
        except (TypeError, ValueError) as exc:
            raise ValueError("Input trials must be a nonnegative integer or 'all'.") from exc
        if parsed < 0:
            raise ValueError("Input trials must be a nonnegative integer or 'all'.")
        return parsed

    def _on_simulation_options_toggled(self, change: dict[str, Any]) -> None:
        visible = bool(change["new"])
        options_box = self.controls.get("simulation_options_box")
        toggle = self.controls.get("simulation_options_toggle")
        if options_box is not None:
            options_box.layout.display = "" if visible else "none"
        if toggle is not None:
            toggle.description = (
                "Hide advanced options" if visible else "Show advanced options"
            )

    def _on_simulation_option_changed(
        self, control_name: str, change: dict[str, Any]
    ) -> None:
        if self._syncing_simulation_options:
            return
        direct_float_paths = {
            "sim_tstart_ms": ("tstart",),
            "sim_tstop_ms": ("tstop",),
            "sim_dt_ms": ("dt",),
            "sim_bins_ms": ("bins",),
            "sim_stim_start_ms": ("stim_start_ms",),
            "sim_stim_duration_ms": ("stim_duration_ms",),
            "sim_iclamp_amp_nA": ("iclamp", "amp_nA"),
            "sim_input_stats_bin_ms": ("input_stats_bin_ms",),
        }
        optional_float_paths = {
            "sim_v_init_mV": ("conditions", "v_init_mV"),
            "sim_celsius_C": ("conditions", "celsius_C"),
            "sim_jitter_ms": ("jitter",),
            "sim_iclamp_delay_ms": ("iclamp", "delay_ms"),
            "sim_iclamp_duration_ms": ("iclamp", "dur_ms"),
            "sim_iclamp_tstop_ms": ("iclamp", "tstop_ms"),
            "sim_iclamp_dt_ms": ("iclamp", "dt_ms"),
        }
        choice_paths = {
            "sim_randomness_mode": ("randomness_mode",),
            "sim_save_plots_mode": ("save_plots_mode",),
        }
        bool_paths = {
            "sim_iclamp_record_currents": ("iclamp", "record_currents"),
            "sim_save_input_stats": ("save_input_stats",),
            "sim_save_plots_overwrite": ("save_plots_overwrite",),
            "sim_cell_recording_enabled": ("cell_recording", "enabled"),
            "sim_record_ion_currents": (
                "cell_recording",
                "vars",
                "ion_currents",
            ),
            "sim_record_mech_currents": (
                "cell_recording",
                "vars",
                "mech_currents",
            ),
            "sim_record_ion_concentrations": (
                "cell_recording",
                "vars",
                "ion_concentrations",
            ),
        }
        try:
            if control_name in direct_float_paths:
                self._set_simulation_override(
                    direct_float_paths[control_name], float(change["new"])
                )
            elif control_name in optional_float_paths:
                self._set_simulation_override(
                    optional_float_paths[control_name],
                    _parse_optional_float(change["new"], name=control_name),
                )
            elif control_name in choice_paths:
                self._set_simulation_override(
                    choice_paths[control_name], str(change["new"])
                )
            elif control_name in bool_paths:
                self._set_simulation_override(
                    bool_paths[control_name], bool(change["new"])
                )
            elif control_name == "sim_n_traces_to_save":
                value = int(change["new"])
                self._set_simulation_override(("n_traces_to_save",), value)
                self._set_simulation_override(
                    ("cell_recording", "n_trials"), value
                )
            elif control_name == "sim_n_inputs_to_save":
                self._set_simulation_override(
                    ("n_inputs_to_save",),
                    self._parse_inputs_to_save(change["new"]),
                )
            elif control_name == "sim_plots_profile":
                profile = str(change["new"])
                flags = {
                    "off": (False, False, False),
                    "basic": (True, False, False),
                    "inputs": (True, True, False),
                    "full": (True, True, True),
                }[profile]
                self._set_simulation_override(("plots_profile",), profile)
                for key, value in zip(
                    ("save_plots", "save_plots_inputs", "save_plots_synapses"),
                    flags,
                ):
                    self._set_simulation_override((key,), value)
            elif control_name == "sim_output_format":
                self._set_simulation_override(
                    ("save", "format"), str(change["new"])
                )
            elif control_name == "sim_save_full_results":
                self._set_simulation_override(
                    ("save", "full_results"), bool(change["new"])
                )
        except (TypeError, ValueError):
            # Text widgets may be temporarily incomplete while the user types.
            # The Run button reports a focused validation message if needed.
            return

    def _refresh_iclamp_options_visibility(self) -> None:
        box = self.controls.get("sim_iclamp_box")
        mode = self.controls.get("run_mode")
        if box is not None and mode is not None:
            box.layout.display = "" if mode.value == "iclamp" else "none"

    def _refresh_simulation_controls(self) -> None:
        if "simulation_options_box" not in self.controls:
            return
        sim_cfg = self._effective_simulation_config()
        iclamp_cfg = sim_cfg.get("iclamp")
        if not isinstance(iclamp_cfg, Mapping):
            iclamp_cfg = {}
        save_cfg = sim_cfg.get("save")
        if not isinstance(save_cfg, Mapping):
            save_cfg = {}
        cell_recording = sim_cfg.get("cell_recording")
        if not isinstance(cell_recording, Mapping):
            cell_recording = {}
        recording_vars = cell_recording.get("vars")
        if not isinstance(recording_vars, Mapping):
            recording_vars = {}
        profile = sim_cfg.get("plots_profile")
        if profile not in {"off", "basic", "inputs", "full"}:
            profile = "basic"
        values = {
            "sim_tstart_ms": float(sim_cfg.get("tstart", 0.0)),
            "sim_tstop_ms": float(sim_cfg.get("tstop", 1000.0)),
            "sim_dt_ms": float(sim_cfg.get("dt", 0.025)),
            "sim_bins_ms": float(sim_cfg.get("bins", 5.0)),
            "sim_stim_start_ms": float(sim_cfg.get("stim_start_ms", 300.0)),
            "sim_stim_duration_ms": float(
                sim_cfg.get("stim_duration_ms", 500.0)
            ),
            "sim_v_init_mV": _optional_text(
                _nested_value(sim_cfg, ("conditions", "v_init_mV"))
            ),
            "sim_celsius_C": _optional_text(
                _nested_value(sim_cfg, ("conditions", "celsius_C"))
            ),
            "sim_jitter_ms": _optional_text(sim_cfg.get("jitter")),
            "sim_randomness_mode": (
                sim_cfg.get("randomness_mode")
                if sim_cfg.get("randomness_mode") in {"random", "derived", "fixed"}
                else "random"
            ),
            "sim_iclamp_amp_nA": float(iclamp_cfg.get("amp_nA", 0.2)),
            "sim_iclamp_delay_ms": _optional_text(iclamp_cfg.get("delay_ms")),
            "sim_iclamp_duration_ms": _optional_text(iclamp_cfg.get("dur_ms")),
            "sim_iclamp_tstop_ms": _optional_text(iclamp_cfg.get("tstop_ms")),
            "sim_iclamp_dt_ms": _optional_text(iclamp_cfg.get("dt_ms")),
            "sim_iclamp_record_currents": bool(
                iclamp_cfg.get("record_currents", False)
            ),
            "sim_n_traces_to_save": max(
                0, int(sim_cfg.get("n_traces_to_save", 1))
            ),
            "sim_n_inputs_to_save": _optional_text(
                sim_cfg.get("n_inputs_to_save", 1)
            ),
            "sim_save_input_stats": bool(sim_cfg.get("save_input_stats", True)),
            "sim_input_stats_bin_ms": float(
                sim_cfg.get("input_stats_bin_ms", 5.0)
            ),
            "sim_plots_profile": profile,
            "sim_save_plots_mode": (
                sim_cfg.get("save_plots_mode")
                if sim_cfg.get("save_plots_mode") in {"single_plot", "default"}
                else "single_plot"
            ),
            "sim_save_plots_overwrite": bool(
                sim_cfg.get("save_plots_overwrite", False)
            ),
            "sim_output_format": (
                save_cfg.get("format", sim_cfg.get("output_format", "pkl"))
                if save_cfg.get("format", sim_cfg.get("output_format", "pkl"))
                in {"pkl", "npz"}
                else "pkl"
            ),
            "sim_save_full_results": bool(
                save_cfg.get(
                    "full_results", sim_cfg.get("save_full_results", False)
                )
            ),
            "sim_cell_recording_enabled": bool(
                cell_recording.get("enabled", False)
            ),
            "sim_record_ion_currents": bool(
                recording_vars.get("ion_currents", False)
            ),
            "sim_record_mech_currents": bool(
                recording_vars.get("mech_currents", False)
            ),
            "sim_record_ion_concentrations": bool(
                recording_vars.get("ion_concentrations", False)
            ),
        }
        self._syncing_simulation_options = True
        try:
            for key, value in values.items():
                control = self.controls.get(key)
                if control is not None:
                    control.value = value
        finally:
            self._syncing_simulation_options = False
        self._refresh_iclamp_options_visibility()

    def _validated_simulation_overrides(self) -> dict[str, Any]:
        for key, label in (
            ("sim_v_init_mV", "Initial voltage"),
            ("sim_celsius_C", "Temperature"),
            ("sim_jitter_ms", "Jitter"),
            ("sim_iclamp_delay_ms", "Current-clamp delay"),
            ("sim_iclamp_duration_ms", "Current-clamp duration"),
            ("sim_iclamp_tstop_ms", "Current-clamp stop time"),
            ("sim_iclamp_dt_ms", "Current-clamp dt"),
        ):
            _parse_optional_float(self.controls[key].value, name=label)
        self._parse_inputs_to_save(self.controls["sim_n_inputs_to_save"].value)

        sim_cfg = self._effective_simulation_config()
        tstart = float(sim_cfg.get("tstart", 0.0))
        tstop = float(sim_cfg.get("tstop", 0.0))
        dt = float(sim_cfg.get("dt", 0.0))
        bins = float(sim_cfg.get("bins", 0.0))
        stim_start = float(sim_cfg.get("stim_start_ms", tstart))
        stim_duration = float(sim_cfg.get("stim_duration_ms", 0.0))
        if not all(
            math.isfinite(value)
            for value in (tstart, tstop, dt, bins, stim_start, stim_duration)
        ):
            raise ValueError("Simulation timing values must be finite.")
        if tstop <= tstart:
            raise ValueError("Simulation stop time must be greater than start time.")
        if dt <= 0 or bins <= 0:
            raise ValueError("Simulation dt and bin width must be greater than zero.")
        if stim_duration < 0:
            raise ValueError("Stimulus duration cannot be negative.")
        if stim_start < tstart or stim_start + stim_duration > tstop:
            raise ValueError(
                "The stimulus marker must lie inside the simulation time window."
            )
        jitter = sim_cfg.get("jitter")
        if jitter is not None and float(jitter) < 0:
            raise ValueError("Jitter cannot be negative.")
        if float(sim_cfg.get("input_stats_bin_ms", 5.0)) <= 0:
            raise ValueError("Input-statistics bin width must be greater than zero.")

        if self.controls["run_mode"].value == "iclamp":
            iclamp_cfg = sim_cfg.get("iclamp")
            if not isinstance(iclamp_cfg, Mapping):
                iclamp_cfg = {}
            def _clamp_value(key: str, fallback: float) -> float:
                value = iclamp_cfg.get(key)
                return float(fallback if value in (None, "") else value)

            delay = _clamp_value("delay_ms", stim_start)
            duration = _clamp_value("dur_ms", stim_duration)
            clamp_stop = _clamp_value("tstop_ms", tstop)
            clamp_dt = _clamp_value("dt_ms", dt)
            clamp_amp = float(iclamp_cfg.get("amp_nA", 0.2))
            if not all(
                math.isfinite(value)
                for value in (delay, duration, clamp_stop, clamp_dt, clamp_amp)
            ):
                raise ValueError("Current-clamp values must be finite.")
            if duration <= 0 or clamp_dt <= 0:
                raise ValueError(
                    "Current-clamp duration and dt must be greater than zero."
                )
            if delay < tstart or delay + duration > clamp_stop:
                raise ValueError(
                    "The current-clamp pulse must lie inside its stop time."
                )

        overrides = self.settings.get("simulation_overrides")
        if overrides is None:
            return {}
        if not isinstance(overrides, Mapping):
            raise TypeError("pipeline_settings['simulation_overrides'] must be a mapping.")
        return deepcopy(dict(overrides))

    def _configured_input_preview_groups(self) -> list[tuple[str, bool]]:
        try:
            from modules.input_generation.inputs import check_inputs

            _sim_config, groups_config = check_inputs(
                path=self._selected_tune_dir(),
                verbose=False,
            )
        except Exception:
            return []
        return [
            (str(name), bool((config or {}).get("state", True)))
            for name, config in groups_config.items()
        ]

    def _input_preview_group_selection(
        self,
        configured_groups: Sequence[tuple[str, bool]],
    ) -> tuple[list[tuple[str, str]], list[str]]:
        names = [name for name, _enabled in configured_groups]
        options = [
            (name if enabled else f"{name} (disabled)", name)
            for name, enabled in configured_groups
        ]
        preferred = self.settings.get("input_preview_groups")
        if preferred is None:
            selected = [
                name for name, enabled in configured_groups if enabled
            ] or names
        else:
            preferred_values = (
                [str(preferred)]
                if isinstance(preferred, str)
                else [str(value) for value in preferred]
            )
            selected = [name for name in preferred_values if name in names]
        return options, selected

    def _refresh_input_preview_group_options(self) -> None:
        control = self.controls.get("input_preview_groups")
        if control is None:
            return
        configured_groups = self._configured_input_preview_groups()
        options, selected = self._input_preview_group_selection(configured_groups)
        self._syncing_input_preview_options = True
        try:
            control.options = options or [("No configured groups found", "")]
            control.value = tuple(selected)
        finally:
            self._syncing_input_preview_options = False

    def _on_input_preview_groups_changed(self, change: dict[str, Any]) -> None:
        if self._syncing_input_preview_options:
            return
        self.settings["input_preview_groups"] = [
            str(value) for value in change["new"] if value
        ]

    def _on_input_preview_options_toggled(self, change: dict[str, Any]) -> None:
        visible = bool(change["new"])
        options_box = self.controls.get("input_preview_options_box")
        toggle = self.controls.get("input_preview_options_toggle")
        if options_box is not None:
            options_box.layout.display = "" if visible else "none"
        if toggle is not None:
            toggle.description = (
                "Hide advanced options" if visible else "Show advanced options"
            )

    def _on_diagnostic_options_toggled(self, change: dict[str, Any]) -> None:
        visible = bool(change["new"])
        options_box = self.controls.get("diagnostic_options_box")
        toggle = self.controls.get("diagnostic_options_toggle")
        if options_box is not None:
            options_box.layout.display = "" if visible else "none"
        if toggle is not None:
            toggle.description = (
                "Hide advanced options" if visible else "Show advanced options"
            )

    def _on_diagnostic_plots_changed(self, change: dict[str, Any]) -> None:
        if self._syncing_diagnostic_options:
            return
        self.settings["diagnostic_plots"] = list(change["new"])

    def _on_diagnostic_trial_changed(self, change: dict[str, Any]) -> None:
        if self._syncing_diagnostic_options or change["new"] is None:
            return
        self.settings["diagnostic_trial_idx"] = int(change["new"])

    def _on_diagnostic_groups_changed(self, change: dict[str, Any]) -> None:
        if self._syncing_diagnostic_options:
            return
        self.settings["diagnostic_input_groups"] = [
            str(value) for value in change["new"] if value
        ]

    def _on_diagnostic_window_mode_changed(
        self, change: dict[str, Any]
    ) -> None:
        self.settings["diagnostic_window_mode"] = str(change["new"])
        self._refresh_diagnostic_window_fields()

    def _refresh_diagnostic_window_fields(self) -> None:
        mode_control = self.controls.get("diagnostic_window_mode")
        manual_box = self.controls.get("diagnostic_manual_window_box")
        padding_box = self.controls.get("diagnostic_padding_box")
        if mode_control is None:
            return
        mode = str(mode_control.value)
        if manual_box is not None:
            manual_box.layout.display = "" if mode == "manual" else "none"
        if padding_box is not None:
            padding_box.layout.display = "" if mode == "stimulus" else "none"

    @staticmethod
    def _diagnostic_trial_count(results: Optional[Mapping[str, Any]]) -> int:
        if not isinstance(results, Mapping):
            return 1
        counts = [1]
        traces = results.get("traces")
        if isinstance(traces, Mapping) and isinstance(traces.get("V"), list):
            counts.append(len(traces["V"]))
        for key in ("spikes", "inputs_by_trial", "cell_recordings_by_trial"):
            value = results.get(key)
            if isinstance(value, (list, tuple)):
                counts.append(len(value))
        meta = results.get("meta")
        if isinstance(meta, Mapping):
            summaries = meta.get("input_summaries")
            if isinstance(summaries, list):
                counts.append(len(summaries))
        return max(1, *counts)

    @staticmethod
    def _diagnostic_available_groups(
        results: Optional[Mapping[str, Any]],
    ) -> list[str]:
        if not isinstance(results, Mapping):
            return []
        names: set[str] = set()
        for key in ("inputs", "syn_records"):
            value = results.get(key)
            if isinstance(value, Mapping):
                names.update(str(name) for name in value)
        trials = results.get("inputs_by_trial")
        if isinstance(trials, list):
            for trial in trials:
                if not isinstance(trial, Mapping):
                    continue
                payload = trial.get("inputs", trial)
                if isinstance(payload, Mapping):
                    names.update(str(name) for name in payload)
        meta = results.get("meta")
        if isinstance(meta, Mapping):
            input_stats = meta.get("input_stats")
            if isinstance(input_stats, Mapping):
                group_means = input_stats.get("group_means")
                if isinstance(group_means, Mapping):
                    names.update(str(name) for name in group_means)
            input_summaries = meta.get("input_summaries")
            if isinstance(input_summaries, list):
                for summary in input_summaries:
                    if not isinstance(summary, Mapping):
                        continue
                    groups = summary.get("groups")
                    if isinstance(groups, Mapping):
                        names.update(str(name) for name in groups)
        return sorted(names)

    def _refresh_diagnostic_controls(self) -> None:
        plots_control = self.controls.get("diagnostic_plots")
        if plots_control is None:
            return
        results = (
            getattr(self.simulation_result, "results", None)
            if self.simulation_result is not None
            else None
        )
        mode = results.get("mode") if isinstance(results, Mapping) else None
        plot_options = (
            [("Membrane voltage", "membrane_voltage")]
            if mode == "iclamp"
            else list(_DIAGNOSTIC_PLOTS)
        )
        available_plot_values = {value for _label, value in plot_options}
        requested = self.settings.get("diagnostic_plots")
        if not isinstance(requested, Sequence) or isinstance(
            requested, (str, bytes, bytearray)
        ):
            requested = PIPELINE_UI_DEFAULTS["diagnostic_plots"]
        selected = [value for value in requested if value in available_plot_values]
        if not selected:
            selected = (
                ["membrane_voltage"]
                if mode == "iclamp"
                else list(PIPELINE_UI_DEFAULTS["diagnostic_plots"])
            )

        trial_count = self._diagnostic_trial_count(results)
        trial_requested = max(
            0, int(self.settings.get("diagnostic_trial_idx", 0) or 0)
        )
        trial_selected = min(trial_requested, trial_count - 1)
        groups = self._diagnostic_available_groups(results)
        requested_groups = self.settings.get("diagnostic_input_groups")
        if requested_groups is None:
            selected_groups = groups
        else:
            if isinstance(requested_groups, str):
                requested_groups = [requested_groups]
            selected_groups = [
                str(value) for value in requested_groups if str(value) in groups
            ]

        self._syncing_diagnostic_options = True
        try:
            plots_control.options = plot_options
            plots_control.value = tuple(selected)
            trial_control = self.controls.get("diagnostic_trial_idx")
            if trial_control is not None:
                trial_control.options = [
                    (f"Trial {idx}", idx) for idx in range(trial_count)
                ]
                trial_control.value = trial_selected
                trial_control.disabled = self.simulation_result is None
            group_control = self.controls.get("diagnostic_input_groups")
            if group_control is not None:
                group_control.options = (
                    [(name, name) for name in groups]
                    if groups
                    else [("No saved input groups", "")]
                )
                group_control.value = tuple(selected_groups)
                group_control.disabled = not groups or mode == "iclamp"
            for key in (
                "diagnostic_rate_bin_ms",
                "diagnostic_smoothing_ms",
                "diagnostic_raster_style",
            ):
                self.controls[key].disabled = mode == "iclamp"

            sim_cfg = results.get("sim_cfg", {}) if isinstance(results, Mapping) else {}
            if not isinstance(sim_cfg, Mapping):
                sim_cfg = {}
            if self.settings.get("diagnostic_rate_bin_ms") is None:
                rate_bin = sim_cfg.get("plots_input_bin_ms")
                if rate_bin is None:
                    rate_bin = sim_cfg.get(
                        "input_stats_bin_ms", sim_cfg.get("bins", 5.0)
                    )
                self.controls["diagnostic_rate_bin_ms"].value = float(rate_bin)
            if self.settings.get("diagnostic_smoothing_ms") is None:
                smoothing = sim_cfg.get(
                    "plots_input_smooth_ms", sim_cfg.get("plots_win_size", 25.0)
                )
                self.controls["diagnostic_smoothing_ms"].value = float(smoothing)
        finally:
            self._syncing_diagnostic_options = False

        self.settings["diagnostic_plots"] = list(selected)
        self.settings["diagnostic_trial_idx"] = trial_selected
        self._refresh_diagnostic_window_fields()

    def _validated_diagnostic_options(
        self,
    ) -> tuple[list[str], dict[str, Any]]:
        plots = [str(value) for value in self.controls["diagnostic_plots"].value]
        if not plots:
            raise ValueError("Select at least one plot panel.")
        trial_idx = int(self.controls["diagnostic_trial_idx"].value or 0)
        window_mode = str(self.controls["diagnostic_window_mode"].value)
        window_start = _parse_optional_float(
            self.controls["diagnostic_window_start_ms"].value,
            name="Plot-window start",
        )
        window_stop = _parse_optional_float(
            self.controls["diagnostic_window_stop_ms"].value,
            name="Plot-window stop",
        )
        padding = float(self.controls["diagnostic_window_padding_ms"].value)
        rate_bin = float(self.controls["diagnostic_rate_bin_ms"].value)
        smoothing = float(self.controls["diagnostic_smoothing_ms"].value)
        if not all(
            math.isfinite(value) for value in (padding, rate_bin, smoothing)
        ):
            raise ValueError("Plot padding, rate bin, and smoothing must be finite.")
        if padding < 0:
            raise ValueError("Stimulus-window padding cannot be negative.")
        if rate_bin <= 0:
            raise ValueError("Rate bin width must be greater than zero.")
        if smoothing < 0:
            raise ValueError("Smoothing cannot be negative.")
        if window_mode == "manual":
            if window_start is None or window_stop is None:
                raise ValueError(
                    "Manual plot windows require both start and stop values."
                )
            if window_stop <= window_start:
                raise ValueError(
                    "Manual plot-window stop must be greater than start."
                )
            plot_window: Optional[tuple[float, float]] = (
                window_start,
                window_stop,
            )
            auto_window = False
        else:
            plot_window = None
            auto_window = window_mode == "stimulus"

        groups = [
            str(value)
            for value in self.controls["diagnostic_input_groups"].value
            if value
        ]
        input_panels = bool({"input_rate", "input_raster"}.intersection(plots))
        available_groups = self._diagnostic_available_groups(
            self.simulation_result.results
        )
        if input_panels and available_groups and not groups:
            raise ValueError(
                "Select at least one input group when an input plot is enabled."
            )

        size = str(self.controls["diagnostic_figure_size"].value)
        panel_count = len(plots)
        if size == "standard":
            figsize = (8.0, max(4.0, min(12.0, 1.9 * panel_count)))
        else:
            figsize = (6.0, max(3.0, min(9.0, 1.45 * panel_count)))
        raster_style = str(self.controls["diagnostic_raster_style"].value)
        show_stimulus = bool(
            self.controls["diagnostic_show_stimulus"].value
        )

        self.settings.update(
            {
                "diagnostic_plots": plots,
                "diagnostic_trial_idx": trial_idx,
                "diagnostic_window_mode": window_mode,
                "diagnostic_window_start_ms": window_start,
                "diagnostic_window_stop_ms": window_stop,
                "diagnostic_window_padding_ms": padding,
                "diagnostic_rate_bin_ms": rate_bin,
                "diagnostic_smoothing_ms": smoothing,
                "diagnostic_raster_style": raster_style,
                "diagnostic_input_groups": groups,
                "diagnostic_show_stimulus": show_stimulus,
                "diagnostic_figure_size": size,
            }
        )
        return plots, {
            "trial_idx": trial_idx,
            "top_input_groups": groups or None,
            "raster_input_groups": groups or None,
            "input_bin_ms": rate_bin,
            "input_smooth_ms": smoothing,
            "input_raster_style": raster_style,
            "output_raster_style": raster_style,
            "output_recompute_bin_ms": rate_bin,
            "output_recompute_smooth_ms": smoothing,
            "plot_window": plot_window,
            "auto_plot_window_from_stim": auto_window,
            "plot_window_adjustment_ms": padding,
            "show_stim_lines": show_stimulus,
            "figsize": figsize,
        }

    def _on_mode_changed(self, change: dict[str, Any]) -> None:
        self.settings["run_iclamp"] = change["new"] == "iclamp"
        self._refresh_iclamp_options_visibility()

    def _on_input_preview(self, _button: Any) -> None:
        try:
            self._refresh_input_preview_group_options()
            seed = _parse_optional_int(
                self.controls["seed"].value,
                name="Seed",
            )
            quiet_output = bool(
                self.controls["quiet_input_preview_output"].value
            )
            configured_values = [
                value
                for _label, value in self.controls["input_preview_groups"].options
                if value
            ]
            groups = [
                str(value)
                for value in self.controls["input_preview_groups"].value
                if value
            ]
            if configured_values and not groups:
                raise ValueError("Select at least one synapse group to display.")
            plot_kinds = [
                str(value)
                for value in self.controls["input_preview_plots"].value
            ]
            trial_idx = int(self.controls["input_preview_trial_idx"].value)
            show_table = bool(
                self.controls["input_preview_show_table"].value
            )
            if not plot_kinds and not show_table:
                raise ValueError(
                    "Select at least one preview plot or enable the summary table."
                )
            histogram_density = (
                self.controls["input_preview_histogram_mode"].value == "density"
            )
            distance_bin_um = float(
                self.controls["input_preview_distance_bin_um"].value
            )
            if distance_bin_um <= 0:
                raise ValueError("Distance bin width must be greater than zero.")
            weight_bin = _parse_optional_float(
                self.controls["input_preview_weight_bin"].value,
                name="Weight bin width",
            )
            if weight_bin is not None and weight_bin <= 0:
                raise ValueError("Weight bin width must be greater than zero.")
            plot_columns = int(
                self.controls["input_preview_plot_columns"].value
            )
            plot_size = str(self.controls["input_preview_plot_size"].value)
        except Exception as exc:
            self._show_validation_error(
                "step5",
                exc,
                output_key="step5_check_inputs",
            )
            return
        self.settings["seed"] = seed
        self.settings.update(
            {
                "quiet_input_preview_output": quiet_output,
                "input_preview_groups": groups,
                "input_preview_plots": plot_kinds,
                "input_preview_trial_idx": trial_idx,
                "input_preview_show_table": show_table,
                "input_preview_histogram_density": histogram_density,
                "input_preview_distance_bin_um": distance_bin_um,
                "input_preview_weight_bin": weight_bin,
                "input_preview_plot_columns": plot_columns,
                "input_preview_plot_size": plot_size,
            }
        )
        self.input_preview_result = None
        self.input_preview_log = ""
        self._run_stage(
            "step5",
            "Checking inputs and synapse placement…",
            lambda: self._run_input_preview(
                seed,
                quiet_output,
                groups,
                plot_kinds,
                trial_idx,
                show_table,
                histogram_density,
                distance_bin_um,
                weight_bin,
                plot_columns,
                plot_size,
            ),
            output_key="step5_check_inputs",
        )

    def _run_input_preview(
        self,
        seed: Optional[int],
        quiet_output: bool,
        groups: list[str],
        plot_kinds: list[str],
        trial_idx: int,
        show_table: bool,
        histogram_density: bool,
        distance_bin_um: float,
        weight_bin: Optional[float],
        plot_columns: int,
        plot_size: str,
    ) -> None:
        self.input_preview_result = preview_pipeline_inputs(
            self.pipeline_state,
            seed=seed,
            trial_idx=trial_idx,
            stream_output=not quiet_output,
        )
        self.input_preview_log = self._fresh_process_log(
            self.input_preview_result
        )
        print("Input preview ready")
        print("  Trial:", self.input_preview_result.trial_idx)
        summary = dict(getattr(self.input_preview_result, "summary", {}) or {})
        if summary.get("total_n_syn") is not None:
            print("  Sampled synapses:", summary["total_n_syn"])
        if quiet_output and self.input_preview_log:
            print("  Process details: hidden in pipeline_ui.input_preview_log")
        show_synapse_preview(
            self.input_preview_result.syn_state,
            trial_idx=self.input_preview_result.trial_idx,
            groups=groups or None,
            show_table=show_table,
            show_plots=bool(plot_kinds),
            plot_kinds=plot_kinds,
            histogram_density=histogram_density,
            plot_density=False,
            distance_bin_um=distance_bin_um,
            weight_bin=weight_bin,
            plot_columns=plot_columns,
            figsize=(3.4, 2.8) if plot_size == "compact" else (4.4, 3.4),
        )

    def _on_simulation(self, _button: Any) -> None:
        try:
            trials = int(self.controls["n_trials"].value)
            if trials < 1:
                raise ValueError("Trials must be at least 1.")
            seed = _parse_optional_int(
                self.controls["seed"].value,
                name="Seed",
            )
            quiet_output = bool(
                self.controls["quiet_simulation_output"].value
            )
            sim_overrides = self._validated_simulation_overrides()
        except Exception as exc:
            self._show_validation_error(
                "step5",
                exc,
                output_key="step5_run",
            )
            return
        iclamp = self.controls["run_mode"].value == "iclamp"
        stem = str(self.controls["output_stem"].value).strip() or None
        self.settings.update(
            {
                "n_trials": trials,
                "seed": seed,
                "run_iclamp": iclamp,
                "output_stem": stem,
                "quiet_simulation_output": quiet_output,
            }
        )
        self.simulation_result = None
        self.diagnostics = None
        self.simulation_log = ""
        self._run_stage(
            "step5",
            "Running the fresh-process simulation…",
            lambda: self._run_simulation(
                trials,
                seed,
                iclamp,
                stem,
                quiet_output,
                sim_overrides,
            ),
            output_key="step5_run",
        )

    def _run_simulation(
        self,
        trials: int,
        seed: Optional[int],
        iclamp: bool,
        stem: Optional[str],
        quiet_output: bool,
        sim_overrides: Mapping[str, Any],
    ) -> None:
        self.simulation_result = run_fresh_simulation(
            self.pipeline_state,
            n_trials=trials,
            seed=seed,
            iclamp=iclamp,
            output_stem=stem,
            stream_output=not quiet_output,
            sim_overrides=sim_overrides,
        )
        self.simulation_log = self._fresh_process_log(self.simulation_result)
        print("Simulation complete")
        print("  Saved run manifest:", self.simulation_result.manifest_path)
        print("  Output stem:", self.simulation_result.output_stem)
        mode = dict(getattr(self.simulation_result, "results", {}) or {}).get("mode")
        if mode:
            print("  Mode:", mode)
        if quiet_output and self.simulation_log:
            print("  Process details: hidden in pipeline_ui.simulation_log")
        self._refresh_diagnostic_controls()

    def _on_plot_results(self, _button: Any) -> None:
        if self.simulation_result is None:
            self._show_validation_error(
                "step5",
                RuntimeError("Run a simulation before plotting results."),
                output_key="step5_plot",
            )
            return
        try:
            plots, plot_options = self._validated_diagnostic_options()
        except Exception as exc:
            self._show_validation_error(
                "step5",
                exc,
                output_key="step5_plot",
            )
            return
        self._run_stage(
            "step5",
            "Plotting the latest simulation result…",
            lambda: self._plot_results(plots, plot_options),
            output_key="step5_plot",
        )

    def _plot_results(
        self, plots: Sequence[str], plot_options: Mapping[str, Any]
    ) -> None:
        results = self.simulation_result.results
        self.diagnostics = show_run_diagnostics(
            results,
            diagnostic_plot="custom",
            diagnostic_plots=plots,
            plot_options=plot_options,
            include_inputs=results.get("mode") != "iclamp",
            cell_name=self.pipeline_state.context.cell_name,
            tune_name=self.pipeline_state.context.tune_name,
            repo_root=self.repo_root,
        )
        print("Plotted run manifest:", self.simulation_result.manifest_path)

    def sync_from_settings(self, *, act_settings_changed: bool = False) -> None:
        del act_settings_changed
        for key in ("output_stem", "seed"):
            control = self.controls.get(key)
            if control is not None and not control.disabled:
                control.value = _optional_text(self.settings.get(key))
        for key in ("quiet_input_preview_output", "quiet_simulation_output"):
            control = self.controls.get(key)
            if control is not None:
                control.value = bool(self.settings.get(key, True))
        self._refresh_simulation_controls()
        self._refresh_input_preview_group_options()
        preview_plots = self.controls.get("input_preview_plots")
        if preview_plots is not None:
            configured = self.settings.get("input_preview_plots")
            if configured is None:
                configured = SYNAPSE_PREVIEW_PLOTS
            preview_plots.value = tuple(
                value for value in configured if value in SYNAPSE_PREVIEW_PLOTS
            )
        preview_values = {
            "input_preview_trial_idx": max(
                0, int(self.settings.get("input_preview_trial_idx", 0))
            ),
            "input_preview_show_table": bool(
                self.settings.get("input_preview_show_table", True)
            ),
            "input_preview_histogram_mode": (
                "density"
                if self.settings.get("input_preview_histogram_density", True)
                else "count"
            ),
            "input_preview_distance_bin_um": float(
                self.settings.get("input_preview_distance_bin_um", 25.0)
            ),
            "input_preview_weight_bin": _optional_text(
                self.settings.get("input_preview_weight_bin")
            ),
            "input_preview_plot_columns": max(
                1,
                min(3, int(self.settings.get("input_preview_plot_columns", 3))),
            ),
            "input_preview_plot_size": (
                self.settings.get("input_preview_plot_size")
                if self.settings.get("input_preview_plot_size")
                in {"compact", "standard"}
                else "compact"
            ),
        }
        for key, value in preview_values.items():
            control = self.controls.get(key)
            if control is not None:
                control.value = value
        trials = self.controls.get("n_trials")
        if trials is not None:
            trials.value = max(1, int(self.settings.get("n_trials", 1)))
        mode = self.controls.get("run_mode")
        if mode is not None:
            mode.value = "iclamp" if self.settings.get("run_iclamp") else "synapse"
        if "diagnostic_plots" in self.controls:
            window_mode = self.settings.get("diagnostic_window_mode", "stimulus")
            if window_mode not in {"stimulus", "full", "manual"}:
                window_mode = "stimulus"
            values = {
                "diagnostic_window_mode": window_mode,
                "diagnostic_window_start_ms": _optional_text(
                    self.settings.get("diagnostic_window_start_ms")
                ),
                "diagnostic_window_stop_ms": _optional_text(
                    self.settings.get("diagnostic_window_stop_ms")
                ),
                "diagnostic_window_padding_ms": float(
                    self.settings.get("diagnostic_window_padding_ms", 100.0)
                ),
                "diagnostic_raster_style": (
                    self.settings.get("diagnostic_raster_style")
                    if self.settings.get("diagnostic_raster_style") in {"dot", "line"}
                    else "dot"
                ),
                "diagnostic_show_stimulus": bool(
                    self.settings.get("diagnostic_show_stimulus", True)
                ),
                "diagnostic_figure_size": (
                    self.settings.get("diagnostic_figure_size")
                    if self.settings.get("diagnostic_figure_size")
                    in {"compact", "standard"}
                    else "compact"
                ),
            }
            for key in ("diagnostic_rate_bin_ms", "diagnostic_smoothing_ms"):
                if self.settings.get(key) is not None:
                    values[key] = float(self.settings[key])
            self._syncing_diagnostic_options = True
            try:
                for key, value in values.items():
                    self.controls[key].value = value
            finally:
                self._syncing_diagnostic_options = False
            self._refresh_diagnostic_controls()

    def refresh_button_states(self, *, ready: bool, act_busy: bool) -> None:
        for key in ("step5_check_inputs", "step5_run"):
            if key in self.controls:
                self.controls[key].disabled = not ready or act_busy
        if "step5_plot" in self.controls:
            self.controls["step5_plot"].disabled = (
                not ready or self.simulation_result is None or act_busy
            )
        if "step5" in self.statuses and not ready and not self._restart_required:
            self._set_status("step5", "waiting", "Waiting for Step 1.")


__all__ = ["Step5UI"]
