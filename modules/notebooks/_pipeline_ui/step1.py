"""Step 1 model-selection and quiet-load UI."""

from __future__ import annotations

import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from typing import Any, Sequence

from ..pipeline_workflow import prepare_pipeline_notebook
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
                "quiet_step1_output": quiet_output,
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
        self._observe_value(quiet_output, "quiet_step1_output", bool)
        run_button.on_click(self._on_prepare)

        panel = w.VBox(
            [
                w.HBox([cell, tune]),
                override,
                quiet_output,
                w.HBox([run_button, status]),
                output,
            ]
        )
        self.panels["step1"] = panel
        self._set_status("step1", "waiting", "Choose a tune, then prepare and load it.")
        return panel

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

    def _on_cell_changed(self, change: dict[str, Any]) -> None:
        tune = self.controls.get("tune_name")
        if tune is None:
            return
        options = self._tune_options(change["new"])
        tune.options = options
        if tune.value not in options:
            tune.value = "tuned" if "tuned" in options else options[0]

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
        except Exception as exc:
            self._show_validation_error("step1", exc)
            return

        quiet_output = bool(self.controls["quiet_step1_output"].value)
        self.settings["quiet_step1_output"] = quiet_output
        model_settings = {
            "cell_name": cell_name,
            "tune_name": tune_name,
            "tune_dir_override": override,
            "recompile_modfiles": bool(self.settings.get("recompile_modfiles", False)),
        }
        self.settings.update(deepcopy(model_settings))
        self._setup_attempted = True
        self._lock_model_controls()
        self._set_status("step1", "running", "Preparing and loading the tune…")
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
                self._set_status(
                    "step1", "error", "Load failed; restart the kernel before retrying."
                )
                self._refresh_button_states()
                return

        self._loaded_model_settings = deepcopy(model_settings)
        self._refresh_connections()
        self._refresh_input_preview_group_options()
        self._refresh_simulation_controls()
        self._refresh_passive_target_controls()
        self._refresh_active_target_labels()
        self._refresh_act_inspection()
        self._set_status(
            "step1",
            "complete",
            f"Ready: {self.pipeline_state.context.cell_name} / "
            f"{self.pipeline_state.context.tune_name}",
        )
        self._refresh_button_states()

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
        print("  Tune:", context.tune_name)
        print("  Tune directory:", state.tune_dir)
        print("  Cell loader:", cell_config.get("cell_loader", "unknown"))
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
        for key in ("cell_name", "tune_name", "tune_dir_override"):
            control = self.controls.get(key)
            if control is not None and not control.disabled:
                control.value = _optional_text(self.settings.get(key))
        quiet = self.controls.get("quiet_step1_output")
        if quiet is not None and not quiet.disabled:
            quiet.value = bool(self.settings.get("quiet_step1_output", True))

    def refresh_button_states(self, *, ready: bool, act_busy: bool) -> None:
        del ready, act_busy
        button = self.controls.get("step1_run")
        if button is not None:
            button.disabled = self._setup_attempted


__all__ = ["Step1UI"]
