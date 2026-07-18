"""Small, independent widget panels for the compact SCP pipeline notebook."""

from __future__ import annotations

import html
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, MutableMapping, Optional, Sequence

from .pipeline_workflow import (
    RestartKernelRequired,
    prepare_interactive_synapse_tuner,
    prepare_pipeline_notebook,
    run_active_stage,
    run_fresh_simulation,
    run_passive_stage,
)
from .run_diagnostics import show_run_diagnostics


PIPELINE_UI_DEFAULTS: dict[str, Any] = {
    "cell_name": "PV",
    "tune_name": "tuned",
    "tune_dir_override": None,
    "adb_specimen_id": None,
    "adb_model_type": "perisomatic",
    "recompile_modfiles": False,
    "passive_amps_pA": [-50.0, -100.0],
    "compute_act_passive_proposal": False,
    "passive_protocol_overrides": {},
    "active_amps_pA": [150.0, 300.0],
    "fi_amps_pA": [float(value) for value in range(0, 301, 50)],
    "active_protocol_overrides": {},
    "enable_synapse_tuning": False,
    "synapse_connection": None,
    "n_trials": 1,
    "seed": None,
    "run_iclamp": False,
    "output_stem": None,
    "diagnostic_plot": "standard",
}

_MODEL_SETTING_KEYS = (
    "cell_name",
    "tune_name",
    "tune_dir_override",
    "adb_specimen_id",
    "adb_model_type",
    "recompile_modfiles",
)


def _import_widgets() -> Any:
    try:
        import ipywidgets as widgets
    except Exception as exc:  # pragma: no cover - depends on the notebook environment
        raise RuntimeError(
            "Pipeline widgets require ipywidgets. Run the notebook environment "
            "cell or install the SCP environment before creating PipelineNotebookUI."
        ) from exc
    return widgets


def _clear_output() -> None:
    try:
        from IPython.display import clear_output

        clear_output(wait=True)
    except Exception:
        return


def _parse_float_list(value: Any, *, name: str) -> list[float]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        raw_values: Sequence[Any] = [part.strip() for part in text.split(",")]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_values = list(value)
    else:
        raise ValueError(f"{name} must be a comma-separated list of numbers.")

    values: list[float] = []
    for raw in raw_values:
        if raw in (None, ""):
            continue
        if isinstance(raw, bool):
            raise ValueError(f"{name} must contain numbers, not booleans.")
        try:
            values.append(float(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} contains an invalid number: {raw!r}. "
                "Use comma-separated values such as -50, -100."
            ) from exc
    if not values:
        raise ValueError(f"{name} must contain at least one number.")
    return values


def _format_float_list(value: Any) -> str:
    try:
        values = _parse_float_list(value, name="amplitudes")
    except ValueError:
        return str(value or "")
    return ", ".join(f"{number:g}" for number in values)


def _parse_optional_int(value: Any, *, name: str, positive: bool = False) -> Optional[int]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer or blank.")
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer or blank; got {value!r}.") from exc
    if positive and parsed <= 0:
        raise ValueError(f"{name} must be greater than zero when provided.")
    return parsed


def _optional_text(value: Any) -> str:
    return "" if value in (None, "") else str(value)


class PipelineNotebookUI:
    """Controller for five cached, per-step SCP pipeline widget panels."""

    def __init__(
        self,
        repo_root: str | Path,
        settings: MutableMapping[str, Any],
    ) -> None:
        if not isinstance(settings, MutableMapping):
            raise TypeError("settings must be a mutable mapping/dict.")
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.settings = settings
        self._fill_defaults(self.settings)
        self.widgets = _import_widgets()

        self.pipeline_state: Any = None
        self.passive_result: Any = None
        self.active_result: Any = None
        self.tuner: Any = None
        self.simulation_result: Any = None
        self.diagnostics: Any = None

        self.panels: dict[str, Any] = {}
        self.controls: dict[str, Any] = {}
        self.outputs: dict[str, Any] = {}
        self.statuses: dict[str, Any] = {}
        self._status_kinds: dict[str, str] = {}
        self._setup_attempted = False
        self._restart_required = False
        self._loaded_model_settings: Optional[dict[str, Any]] = None

    @staticmethod
    def _fill_defaults(settings: MutableMapping[str, Any]) -> None:
        for key, value in PIPELINE_UI_DEFAULTS.items():
            settings.setdefault(key, deepcopy(value))

    def apply_settings(self, settings: MutableMapping[str, Any]) -> None:
        """Adopt code-edited settings and refresh all unlocked controls."""

        if not isinstance(settings, MutableMapping):
            raise TypeError("settings must be a mutable mapping/dict.")
        self._fill_defaults(settings)
        ignored: list[str] = []
        if self._loaded_model_settings is not None:
            for key in _MODEL_SETTING_KEYS:
                loaded = deepcopy(self._loaded_model_settings[key])
                if settings.get(key) != loaded:
                    ignored.append(key)
                    settings[key] = loaded
        self.settings = settings
        self._sync_controls_from_settings()
        if ignored:
            self._set_status(
                "step1",
                "warning",
                "Model selection is locked. Restart the kernel to change: "
                + ", ".join(ignored),
            )

    def step1_panel(self) -> Any:
        if "step1" in self.panels:
            return self.panels["step1"]
        w = self.widgets
        cells = self._cell_options(self.settings.get("cell_name"))
        cell = w.Combobox(
            options=cells,
            value=str(self.settings.get("cell_name") or cells[0]),
            ensure_option=False,
            description="Cell",
            layout=w.Layout(width="280px"),
        )
        tunes = self._tune_options(cell.value, self.settings.get("tune_name"))
        tune = w.Combobox(
            options=tunes,
            value=str(self.settings.get("tune_name") or tunes[0]),
            ensure_option=False,
            description="Tune",
            layout=w.Layout(width="280px"),
        )
        override = w.Text(
            value=_optional_text(self.settings.get("tune_dir_override")),
            description="Tune path",
            placeholder="optional path override",
            layout=w.Layout(width="560px"),
        )
        specimen = w.Text(
            value=_optional_text(self.settings.get("adb_specimen_id")),
            description="ADB ID",
            placeholder="optional specimen ID",
            layout=w.Layout(width="280px"),
        )
        model_type = w.Dropdown(
            options=[("Perisomatic", "perisomatic"), ("All active", "all active")],
            value=self._adb_model_type_value(),
            description="ADB model",
            layout=w.Layout(width="280px"),
        )
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
                "cell_name": cell,
                "tune_name": tune,
                "tune_dir_override": override,
                "adb_specimen_id": specimen,
                "adb_model_type": model_type,
                "step1_run": run_button,
            }
        )
        self.outputs["step1"] = output
        self.statuses["step1"] = status

        cell.observe(self._on_cell_changed, names="value")
        self._observe_value(cell, "cell_name", lambda value: str(value).strip())
        self._observe_value(tune, "tune_name", lambda value: str(value).strip())
        self._observe_value(
            override,
            "tune_dir_override",
            lambda value: str(value).strip() or None,
        )
        self._observe_valid_optional_int(specimen, "adb_specimen_id", positive=True)
        self._observe_value(model_type, "adb_model_type", str)
        run_button.on_click(self._on_prepare)

        panel = w.VBox(
            [
                w.HBox([cell, tune]),
                override,
                w.HBox([specimen, model_type]),
                w.HBox([run_button, status]),
                output,
            ]
        )
        self.panels["step1"] = panel
        self._set_status("step1", "waiting", "Choose a tune, then prepare and load it.")
        return panel

    def step2_panel(self) -> Any:
        if "step2" in self.panels:
            return self.panels["step2"]
        w = self.widgets
        amps = w.Text(
            value=_format_float_list(self.settings.get("passive_amps_pA")),
            description="Amps (pA)",
            layout=w.Layout(width="360px"),
        )
        proposal = w.Checkbox(
            value=bool(self.settings.get("compute_act_passive_proposal")),
            description="Compute ACT proposal (review only)",
            indent=False,
            layout=w.Layout(width="280px"),
        )
        run_button = self._run_button("Run passive")
        status = w.HTML()
        output = w.Output()
        self.controls.update(
            {
                "passive_amps_pA": amps,
                "compute_act_passive_proposal": proposal,
                "step2_run": run_button,
            }
        )
        self.outputs["step2"] = output
        self.statuses["step2"] = status
        self._observe_valid_float_list(amps, "passive_amps_pA")
        self._observe_value(proposal, "compute_act_passive_proposal", bool)
        run_button.on_click(self._on_passive)
        panel = w.VBox(
            [w.HBox([amps, proposal]), w.HBox([run_button, status]), output]
        )
        self.panels["step2"] = panel
        self._refresh_button_states()
        return panel

    def step3_panel(self) -> Any:
        if "step3" in self.panels:
            return self.panels["step3"]
        w = self.widgets
        active_amps = w.Text(
            value=_format_float_list(self.settings.get("active_amps_pA")),
            description="Active (pA)",
            layout=w.Layout(width="360px"),
        )
        fi_amps = w.Text(
            value=_format_float_list(self.settings.get("fi_amps_pA")),
            description="FI (pA)",
            layout=w.Layout(width="560px"),
        )
        run_button = self._run_button("Run active / FI")
        status = w.HTML()
        output = w.Output()
        self.controls.update(
            {
                "active_amps_pA": active_amps,
                "fi_amps_pA": fi_amps,
                "step3_run": run_button,
            }
        )
        self.outputs["step3"] = output
        self.statuses["step3"] = status
        self._observe_valid_float_list(active_amps, "active_amps_pA")
        self._observe_valid_float_list(fi_amps, "fi_amps_pA")
        run_button.on_click(self._on_active)
        panel = w.VBox(
            [active_amps, fi_amps, w.HBox([run_button, status]), output]
        )
        self.panels["step3"] = panel
        self._refresh_button_states()
        return panel

    def step4_panel(self) -> Any:
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
        output = w.Output()
        self.controls.update(
            {
                "enable_synapse_tuning": enabled,
                "synapse_connection": connection,
                "step4_initialize": initialize,
                "step4_single_event": single_event,
                "step4_interactive": interactive,
            }
        )
        self.outputs["step4"] = output
        self.statuses["step4"] = status
        self._observe_value(enabled, "enable_synapse_tuning", bool)
        enabled.observe(lambda _change: self._refresh_button_states(), names="value")
        connection.observe(self._on_connection_changed, names="value")
        initialize.on_click(self._on_initialize_tuner)
        single_event.on_click(self._on_single_event)
        interactive.on_click(self._on_interactive_tuner)
        panel = w.VBox(
            [
                w.HBox([enabled, connection]),
                w.HBox([initialize, single_event, interactive]),
                status,
                output,
            ]
        )
        self.panels["step4"] = panel
        self._refresh_button_states()
        return panel

    def step5_panel(self) -> Any:
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
            description="Seed",
            placeholder="optional",
            layout=w.Layout(width="220px"),
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
        diagnostic = w.Dropdown(
            options=[
                ("Standard", "standard"),
                ("Summary", "summary"),
                ("Single plot", "single_plot"),
                ("Counts only", None),
            ],
            value=self._diagnostic_value(),
            description="Diagnostics",
            layout=w.Layout(width="280px"),
        )
        run_button = self._run_button("Run simulation")
        status = w.HTML()
        output = w.Output()
        self.controls.update(
            {
                "n_trials": trials,
                "seed": seed,
                "run_mode": mode,
                "output_stem": output_stem,
                "diagnostic_plot": diagnostic,
                "step5_run": run_button,
            }
        )
        self.outputs["step5"] = output
        self.statuses["step5"] = status
        self._observe_value(trials, "n_trials", int)
        self._observe_valid_optional_int(seed, "seed", positive=False)
        mode.observe(self._on_mode_changed, names="value")
        self._observe_value(
            output_stem,
            "output_stem",
            lambda value: str(value).strip() or None,
        )
        self._observe_value(diagnostic, "diagnostic_plot", lambda value: value)
        run_button.on_click(self._on_simulation)
        panel = w.VBox(
            [
                w.HBox([trials, seed, mode]),
                w.HBox([output_stem, diagnostic]),
                w.HBox([run_button, status]),
                output,
            ]
        )
        self.panels["step5"] = panel
        self._refresh_button_states()
        return panel

    def _run_button(self, description: str, *, icon: str = "play") -> Any:
        return self.widgets.Button(
            description=description,
            button_style="primary",
            icon=icon,
            layout=self.widgets.Layout(width="180px"),
        )

    def _cell_options(self, preferred: Any = None) -> list[str]:
        cells_dir = self.repo_root / "cells"
        options = (
            sorted(path.name for path in cells_dir.iterdir() if path.is_dir())
            if cells_dir.is_dir()
            else []
        )
        return self._include_preferred(options, preferred, fallback="PV")

    def _tune_options(self, cell_name: Any, preferred: Any = None) -> list[str]:
        tunes_dir = self.repo_root / "cells" / str(cell_name).strip() / "tunes"
        options = (
            sorted(path.name for path in tunes_dir.iterdir() if path.is_dir())
            if tunes_dir.is_dir()
            else []
        )
        return self._include_preferred(options, preferred, fallback="tuned")

    @staticmethod
    def _include_preferred(
        options: list[str], preferred: Any, *, fallback: str
    ) -> list[str]:
        chosen = str(preferred).strip() if preferred not in (None, "") else ""
        if chosen and chosen not in options:
            options.append(chosen)
        if not options:
            options = [chosen or fallback]
        return sorted(set(options))

    def _observe_value(self, widget: Any, key: str, convert: Any) -> None:
        def _sync(change: dict[str, Any]) -> None:
            self.settings[key] = convert(change["new"])

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

    def _on_cell_changed(self, change: dict[str, Any]) -> None:
        tune = self.controls.get("tune_name")
        if tune is None:
            return
        options = self._tune_options(change["new"])
        tune.options = options
        if tune.value not in options:
            tune.value = "tuned" if "tuned" in options else options[0]

    def _on_connection_changed(self, change: dict[str, Any]) -> None:
        value = str(change["new"] or "").strip()
        self.settings["synapse_connection"] = value or None

    def _on_mode_changed(self, change: dict[str, Any]) -> None:
        self.settings["run_iclamp"] = change["new"] == "iclamp"

    def _on_prepare(self, _button: Any) -> None:
        if self._setup_attempted:
            self._set_status(
                "step1",
                "warning",
                "Step 1 already ran. Restart the kernel to load another model.",
            )
            return
        try:
            cell_name = str(self.controls["cell_name"].value).strip()
            tune_name = str(self.controls["tune_name"].value).strip()
            override = str(self.controls["tune_dir_override"].value).strip() or None
            if not override and (not cell_name or not tune_name):
                raise ValueError("Cell and tune are required when no tune path is supplied.")
            specimen_id = _parse_optional_int(
                self.controls["adb_specimen_id"].value,
                name="ADB specimen ID",
                positive=True,
            )
        except Exception as exc:
            self._show_validation_error("step1", exc)
            return

        model_settings = {
            "cell_name": cell_name,
            "tune_name": tune_name,
            "tune_dir_override": override,
            "adb_specimen_id": specimen_id,
            "adb_model_type": str(self.controls["adb_model_type"].value),
            "recompile_modfiles": bool(self.settings.get("recompile_modfiles", False)),
        }
        self.settings.update(deepcopy(model_settings))
        self._setup_attempted = True
        self._lock_model_controls()
        self._set_status("step1", "running", "Preparing and loading the tune…")
        with self.outputs["step1"]:
            _clear_output()
            try:
                self.pipeline_state = prepare_pipeline_notebook(
                    repo_root=self.repo_root,
                    **model_settings,
                )
            except Exception:
                traceback.print_exc()
                print(
                    "Step 1 may have partially loaded NEURON model state. Restart "
                    "the kernel before trying Step 1 again."
                )
                self._set_status(
                    "step1", "error", "Load failed; restart the kernel before retrying."
                )
                self._refresh_button_states()
                return

        self._loaded_model_settings = deepcopy(model_settings)
        self._refresh_connections()
        self._set_status(
            "step1",
            "complete",
            f"Ready: {self.pipeline_state.context.cell_name} / "
            f"{self.pipeline_state.context.tune_name}",
        )
        self._refresh_button_states()

    def _on_passive(self, _button: Any) -> None:
        try:
            amps = _parse_float_list(
                self.controls["passive_amps_pA"].value,
                name="Passive amplitudes",
            )
        except Exception as exc:
            self._show_validation_error("step2", exc)
            return
        self.settings["passive_amps_pA"] = amps
        self._run_stage(
            "step2",
            "Running passive tuning…",
            lambda: self._run_passive(amps),
        )

    def _run_passive(self, amps: list[float]) -> None:
        self.passive_result = run_passive_stage(
            self.pipeline_state,
            amps_pA=amps,
            compute_act_proposal=bool(
                self.controls["compute_act_passive_proposal"].value
            ),
            protocol_overrides=dict(
                self.settings.get("passive_protocol_overrides", {}) or {}
            ),
        )

    def _on_active(self, _button: Any) -> None:
        try:
            active_amps = _parse_float_list(
                self.controls["active_amps_pA"].value,
                name="Active amplitudes",
            )
            fi_amps = _parse_float_list(
                self.controls["fi_amps_pA"].value,
                name="FI amplitudes",
            )
        except Exception as exc:
            self._show_validation_error("step3", exc)
            return
        self.settings["active_amps_pA"] = active_amps
        self.settings["fi_amps_pA"] = fi_amps
        self._run_stage(
            "step3",
            "Running active tuning and FI curve…",
            lambda: self._run_active(active_amps, fi_amps),
        )

    def _run_active(self, active_amps: list[float], fi_amps: list[float]) -> None:
        self.active_result = run_active_stage(
            self.pipeline_state,
            active_amps_pA=active_amps,
            fi_amps_pA=fi_amps,
            protocol_overrides=dict(
                self.settings.get("active_protocol_overrides", {}) or {}
            ),
        )

    def _on_initialize_tuner(self, _button: Any) -> None:
        connection = str(self.controls["synapse_connection"].value or "").strip()
        self.settings["synapse_connection"] = connection or None
        self._run_stage(
            "step4",
            "Initializing BMTool…",
            lambda: self._initialize_tuner(connection or None),
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
        )

    def _on_interactive_tuner(self, _button: Any) -> None:
        self._run_stage(
            "step4",
            "Opening BMTool interactive tuner…",
            lambda: self.tuner.InteractiveTuner(),
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
        except Exception as exc:
            self._show_validation_error("step5", exc)
            return
        iclamp = self.controls["run_mode"].value == "iclamp"
        stem = str(self.controls["output_stem"].value).strip() or None
        diagnostic = self.controls["diagnostic_plot"].value
        self.settings.update(
            {
                "n_trials": trials,
                "seed": seed,
                "run_iclamp": iclamp,
                "output_stem": stem,
                "diagnostic_plot": diagnostic,
            }
        )
        self._run_stage(
            "step5",
            "Running the fresh-process simulation…",
            lambda: self._run_simulation(trials, seed, iclamp, stem, diagnostic),
        )

    def _run_simulation(
        self,
        trials: int,
        seed: Optional[int],
        iclamp: bool,
        stem: Optional[str],
        diagnostic: Optional[str],
    ) -> None:
        self.simulation_result = run_fresh_simulation(
            self.pipeline_state,
            n_trials=trials,
            seed=seed,
            iclamp=iclamp,
            output_stem=stem,
        )
        results = self.simulation_result.results
        self.diagnostics = show_run_diagnostics(
            results,
            diagnostic_plot=diagnostic,
            include_inputs=not iclamp,
            cell_name=self.pipeline_state.context.cell_name,
            tune_name=self.pipeline_state.context.tune_name,
            repo_root=self.repo_root,
        )
        print("Saved run manifest:", self.simulation_result.manifest_path)

    def _run_stage(self, step: str, running_message: str, operation: Any) -> None:
        if self.pipeline_state is None:
            self._show_validation_error(step, RuntimeError("Complete Step 1 first."))
            return
        button_keys = {
            "step2": ["step2_run"],
            "step3": ["step3_run"],
            "step4": [
                "step4_initialize",
                "step4_single_event",
                "step4_interactive",
            ],
            "step5": ["step5_run"],
        }[step]
        for key in button_keys:
            self.controls[key].disabled = True
        self._set_status(step, "running", running_message)
        with self.outputs[step]:
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

    def _show_validation_error(self, step: str, exc: Exception) -> None:
        self._set_status(step, "error", str(exc))
        output = self.outputs.get(step)
        if output is not None:
            with output:
                _clear_output()
                print(f"{type(exc).__name__}: {exc}")

    def _mark_restart_required(self, message: str) -> None:
        self._restart_required = True
        for step in ("step2", "step3", "step4", "step5"):
            self._set_status(step, "error", "Kernel restart required: " + message)
        self._refresh_button_states()

    def _lock_model_controls(self) -> None:
        for key in (
            "cell_name",
            "tune_name",
            "tune_dir_override",
            "adb_specimen_id",
            "adb_model_type",
            "step1_run",
        ):
            control = self.controls.get(key)
            if control is not None:
                control.disabled = True

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

    def _refresh_button_states(self) -> None:
        ready = self.pipeline_state is not None and not self._restart_required
        if "step1_run" in self.controls:
            self.controls["step1_run"].disabled = self._setup_attempted
        for key in ("step2_run", "step3_run", "step5_run"):
            if key in self.controls:
                self.controls[key].disabled = not ready
        if "step2" in self.statuses and not ready and not self._restart_required:
            self._set_status("step2", "waiting", "Waiting for Step 1.")
        if "step3" in self.statuses and not ready and not self._restart_required:
            self._set_status("step3", "waiting", "Waiting for Step 1.")
        if "step5" in self.statuses and not ready and not self._restart_required:
            self._set_status("step5", "waiting", "Waiting for Step 1.")

        if "step4_initialize" in self.controls:
            enabled = bool(self.controls["enable_synapse_tuning"].value)
            initialized = self.tuner is not None
            self.controls["step4_initialize"].disabled = (
                not ready or not enabled or initialized
            )
            self.controls["step4_single_event"].disabled = not ready or not initialized
            self.controls["step4_interactive"].disabled = not ready or not initialized
            current_kind = self._status_kinds.get("step4")
            if self._restart_required:
                pass
            elif not ready and current_kind != "error":
                self._set_status("step4", "waiting", "Waiting for Step 1.")
            elif not enabled:
                self._set_status("step4", "waiting", "Optional BMTool stage is disabled.")
            elif initialized and current_kind in {None, "waiting", "running"}:
                self._set_status("step4", "complete", "BMTool tuner initialized.")
            elif not initialized and not self._restart_required and current_kind != "error":
                self._set_status("step4", "waiting", "Choose a connection and initialize BMTool.")

    def _sync_controls_from_settings(self) -> None:
        direct_text = (
            "cell_name",
            "tune_name",
            "tune_dir_override",
            "adb_specimen_id",
            "output_stem",
            "seed",
        )
        for key in direct_text:
            control = self.controls.get(key)
            if control is not None and not control.disabled:
                control.value = _optional_text(self.settings.get(key))
        for key in ("passive_amps_pA", "active_amps_pA", "fi_amps_pA"):
            control = self.controls.get(key)
            if control is not None:
                control.value = _format_float_list(self.settings.get(key))
        for key in (
            "compute_act_passive_proposal",
            "enable_synapse_tuning",
        ):
            control = self.controls.get(key)
            if control is not None:
                control.value = bool(self.settings.get(key))
        if "adb_model_type" in self.controls and not self.controls["adb_model_type"].disabled:
            self.controls["adb_model_type"].value = self._adb_model_type_value()
        if "n_trials" in self.controls:
            self.controls["n_trials"].value = max(1, int(self.settings.get("n_trials", 1)))
        if "run_mode" in self.controls:
            self.controls["run_mode"].value = (
                "iclamp" if self.settings.get("run_iclamp") else "synapse"
            )
        if "diagnostic_plot" in self.controls:
            self.controls["diagnostic_plot"].value = self._diagnostic_value()
        if (
            "synapse_connection" in self.controls
            and self.pipeline_state is not None
            and self.tuner is None
        ):
            self._refresh_connections()
        self._refresh_button_states()

    def _adb_model_type_value(self) -> str:
        value = str(self.settings.get("adb_model_type") or "perisomatic").strip().lower()
        return "all active" if value in {"all active", "all_active", "all-active"} else "perisomatic"

    def _diagnostic_value(self) -> Optional[str]:
        value = self.settings.get("diagnostic_plot", "standard")
        return value if value in {"standard", "summary", "single_plot", None} else "standard"

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
            f"<b>{labels.get(kind, kind.title())}:</b> {html.escape(str(message))}</span>"
        )


__all__ = ["PIPELINE_UI_DEFAULTS", "PipelineNotebookUI"]
