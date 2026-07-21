"""Compact notebook orchestration for the SCP Steps 1--5 workflow.

The public numbered notebooks remain the detailed teaching interfaces.  This
module provides the smaller, progressive workflow used by ``0_pipeline.ipynb``:

* safely fill/validate an already prepared tune,
* construct one in-kernel cell for passive, active, and BMTool checks,
* detect model-source edits that require a kernel restart, and
* preview Step 5 inputs and run the final simulation in clean Python processes.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


PASSIVE_PROTOCOL_DEFAULTS: dict[str, Any] = {
    "stim_amp": -0.1,
    "stim_delay": 300.0,
    "stim_dur": 1000.0,
    "h_tstop": 1500.0,
    "h_dt": 0.025,
}

ACTIVE_PROTOCOL_DEFAULTS: dict[str, Any] = {
    "stim_amp": -0.1,
    "stim_delay": 200.0,
    "stim_dur": 1000.0,
    "h_tstop": 1500.0,
    "h_dt": 0.025,
}


class RestartKernelRequired(RuntimeError):
    """Raised when loaded model sources changed during a notebook session."""


@dataclass
class PipelineNotebookState:
    """Prepared in-kernel state shared by compact Steps 2--4."""

    repo_root: Path
    tune_dir: Path
    context: Any
    cell: Any
    setup_summary: dict[str, Any]
    mechanism_summary: dict[str, Any]
    source_fingerprint: dict[str, str]

    def assert_sources_unchanged(self) -> None:
        """Require a restart when cell-construction inputs changed on disk."""

        try:
            current = _model_source_fingerprint(self.tune_dir)
        except Exception as exc:
            raise RestartKernelRequired(
                "Model source/config files changed or became invalid after the cell "
                "was loaded. Restart the kernel and rerun 0_pipeline.ipynb from the top. "
                f"Current source check failed: {exc}"
            ) from exc
        if current == self.source_fingerprint:
            return

        before = set(self.source_fingerprint)
        after = set(current)
        changed = sorted(
            path
            for path in before & after
            if self.source_fingerprint[path] != current[path]
        )
        added = sorted(after - before)
        removed = sorted(before - after)
        details = []
        if changed:
            details.append("changed=" + ", ".join(changed))
        if added:
            details.append("added=" + ", ".join(added))
        if removed:
            details.append("removed=" + ", ".join(removed))
        suffix = "; ".join(details) or "source set changed"
        raise RestartKernelRequired(
            "Cell/model sources changed after the in-kernel cell was built "
            f"({suffix}). Restart the kernel and rerun 0_pipeline.ipynb from the top "
            "before continuing. sim_config, target_config, geometry, and synapse "
            "config edits do not require a restart."
        )


@dataclass
class PipelinePassiveResult:
    resolution: Any
    sim_params: dict[str, Any]
    records: dict[str, Any]
    metrics: list[dict[str, Any]]
    target_comparison: list[dict[str, Any]]
    proposal_changes: list[dict[str, Any]]
    figure: Any


@dataclass
class PipelinePassiveProposalResult:
    """Review-only ACT passive proposal computed from current targets."""

    resolution: Any
    proposal_changes: list[dict[str, Any]]


@dataclass
class PipelineActiveProtocolResult:
    """Active current-step traces and metrics from the shared cell."""

    sim_params: dict[str, Any]
    records: dict[str, Any]
    metrics: list[dict[str, Any]]
    figure: Any


@dataclass
class PipelineFIResult:
    """FI sweep, configured-target comparison, and plot."""

    sim_params: dict[str, Any]
    records: dict[str, Any]
    metrics: list[dict[str, Any]]
    fi_rows: list[dict[str, float]]
    target_reference: list[tuple[float, float]]
    target_comparison: list[dict[str, Any]]
    figure: Any


@dataclass
class PipelineActiveResult:
    active_sim_params: dict[str, Any]
    active_records: dict[str, Any]
    active_metrics: list[dict[str, Any]]
    fi_sim_params: dict[str, Any]
    fi_records: dict[str, Any]
    fi_metrics: list[dict[str, Any]]
    fi_rows: list[dict[str, float]]
    target_reference: list[tuple[float, float]]
    target_comparison: list[dict[str, Any]]
    active_figure: Any
    fi_figure: Any


@dataclass
class PipelineSimulationResult:
    output_stem: str
    manifest_path: Path
    results: dict[str, Any]
    command: tuple[str, ...]
    stdout: str


@dataclass
class PipelineInputPreviewResult:
    """Preview-only input and synapse records produced in a fresh process."""

    syn_state: dict[str, Any]
    summary: dict[str, Any]
    session_summary: dict[str, Any]
    trial_idx: int
    command: tuple[str, ...]
    stdout: str


def _read_json_dict(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def _infer_labels(tune_dir: Path) -> tuple[str, str]:
    tune = Path(tune_dir).expanduser().resolve()
    if tune.parent.name == "tunes" and tune.parent.parent.name:
        return tune.parent.parent.name, tune.name
    return tune.parent.name or "cell", tune.name


def _resolve_tune_path(
    *,
    repo_root: Path,
    cell_name: Optional[str],
    tune_name: Optional[str],
    tune_dir_override: Optional[str | Path],
) -> Path:
    if tune_dir_override not in (None, ""):
        tune_dir = Path(str(tune_dir_override)).expanduser()
        if not tune_dir.is_absolute():
            tune_dir = repo_root / tune_dir
        return tune_dir.resolve()
    if not cell_name or not tune_name:
        raise ValueError(
            "Provide tune_dir_override or both cell_name and tune_name in the "
            "pipeline settings cell."
        )
    return (repo_root / "cells" / str(cell_name) / "tunes" / str(tune_name)).resolve()


def _select_setup_source(
    tune_dir: Path,
) -> tuple[str, str, dict[str, Any]]:
    """Return ``(source_type, loader, existing_cell_config)`` for setup."""

    cell_config_path = tune_dir / "cell_configs" / "cell_config.json"
    existing = _read_json_dict(cell_config_path) if cell_config_path.is_file() else {}

    from modules.loaders import get_cell_loader_name

    if existing:
        has_explicit_loader = any(
            existing.get(key) not in (None, "")
            for key in ("cell_loader", "loader", "model_loader")
        )
        if not has_explicit_loader:
            paths = existing.get("paths", {}) or {}
            if not isinstance(paths, dict):
                raise TypeError("cell_config.json paths must be an object/dict.")
            manifest = Path(str(paths.get("manifest") or "manifest.json")).expanduser()
            if not manifest.is_absolute():
                manifest = tune_dir / manifest
            if not manifest.is_file():
                raise FileNotFoundError(
                    f"The existing cell_config.json in {tune_dir} has no registered "
                    "loader and no Allen manifest. Use 1_setup.ipynb to register/stage "
                    "this custom model first, then return to 0_pipeline.ipynb."
                )
        loader = get_cell_loader_name(existing)
        return "existing", loader, existing

    if (tune_dir / "manifest.json").is_file():
        return "existing", "allen_manifest", {}
    raise FileNotFoundError(
        f"No configured cell or Allen manifest was found in {tune_dir}. "
        "Use 1_setup.ipynb to register/stage this custom model first, then return "
        "to 0_pipeline.ipynb."
    )


def prepare_pipeline_notebook(
    *,
    repo_root: Optional[str | Path] = None,
    cell_name: Optional[str] = "PV",
    tune_name: Optional[str] = "tuned",
    tune_dir_override: Optional[str | Path] = None,
    recompile_modfiles: bool = False,
) -> PipelineNotebookState:
    """Fill, validate, and build the single cell shared by Steps 2--4.

    Existing config values always win. New model acquisition and loader setup
    belong in ``1_setup.ipynb``; this compact workflow only loads prepared tunes.
    """

    from modules.notebooks.helpers import ensure_scp_repo_on_syspath

    root = ensure_scp_repo_on_syspath(
        Path(repo_root).expanduser() if repo_root is not None else None
    )
    tune_dir = _resolve_tune_path(
        repo_root=root,
        cell_name=cell_name,
        tune_name=tune_name,
        tune_dir_override=tune_dir_override,
    )
    source_type, loader_name, existing_config = _select_setup_source(tune_dir)
    inferred_cell, inferred_tune = _infer_labels(tune_dir)
    resolved_cell = str(cell_name or existing_config.get("cell_name") or inferred_cell)
    resolved_tune = str(tune_name or existing_config.get("tune") or inferred_tune)

    existing_tuning = existing_config.get("tuning", {}) or {}
    soma_multiplier = (
        existing_tuning.get("soma_diam_multiplier")
        if isinstance(existing_tuning, dict)
        else None
    )
    existing_color = existing_config.get("color")

    from modules.setup.step1_prepare import prepare_tune

    setup_summary = prepare_tune(
        tune_dir=tune_dir,
        cell_name=resolved_cell,
        tune_name=resolved_tune,
        source_type=source_type,
        cell_loader=loader_name,
        soma_diam_multiplier=soma_multiplier,
        color=existing_color,
        do_download=False,
        force_download=False,
        do_compile_modfiles=True,
        recompile_modfiles=bool(recompile_modfiles),
        load_compiled_dll=True,
        allow_missing_modfiles=True,
        do_base_configs=True,
        do_target_config=True,
        do_synapse_configs=True,
        config_mode="fill",
        sync_cell_metadata=True,
        do_validate=False,  # validation below deliberately does not build a second cell
        validate_inputs_cfg=True,
        create_tuned_copy=False,
    )

    from modules.setup.validation import validate_tune

    validation = validate_tune(
        tune_dir=tune_dir,
        cell_name=resolved_cell,
        validate_modfiles=True,
        validate_load_cell=False,
        validate_inputs=True,
        validate_synapses=True,
        allow_missing_modfiles=True,
    )
    setup_summary.setdefault("actions", {})["validate"] = {
        "status": "ok",
        "checks": validation,
        "load_cell": "deferred to the single shared pipeline build",
    }

    from modules.tuning import build_tuning_cell, prepare_tuning_notebook_context

    context = prepare_tuning_notebook_context(
        cell_name=resolved_cell,
        tune_name=resolved_tune,
        tune_dir_override=tune_dir,
        repo_root=root,
        require_compiled_modfiles=True,
        require_synapse_config=False,
        print_summary=True,
    )
    cell = build_tuning_cell(context)
    mechanism_summary = (
        setup_summary.get("actions", {})
        .get("mechanisms", {})
        .get("compile_modfiles", {})
    )
    fingerprint = _model_source_fingerprint(tune_dir)

    print("Pipeline setup: ready")
    print("  source:", source_type)
    print("  loader:", loader_name)
    print("  shared cell:", cell)
    return PipelineNotebookState(
        repo_root=root,
        tune_dir=tune_dir,
        context=context,
        cell=cell,
        setup_summary=setup_summary,
        mechanism_summary=dict(mechanism_summary or {}),
        source_fingerprint=fingerprint,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_source_paths(tune_dir: Path) -> list[Path]:
    tune = Path(tune_dir).expanduser().resolve()
    cell_config_path = tune / "cell_configs" / "cell_config.json"
    if not cell_config_path.is_file():
        raise FileNotFoundError(f"Missing cell config: {cell_config_path}")
    cell_config = _read_json_dict(cell_config_path)

    from modules.loaders import discover_cell_source_artifacts
    from modules.setup.mechanisms import resolve_modfiles_dir

    paths = [cell_config_path]
    paths.extend(
        discover_cell_source_artifacts(cell_config, base_dir=tune).values()
    )
    mod_dir = resolve_modfiles_dir(tune, cell_config)
    if mod_dir is not None and mod_dir.is_dir():
        paths.extend(sorted(mod_dir.rglob("*.mod")))

    unique: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Model source disappeared: {path}")
        seen.add(path)
        unique.append(path)
    return unique


def _model_source_fingerprint(tune_dir: Path) -> dict[str, str]:
    return {
        str(path): _sha256_file(path)
        for path in _model_source_paths(tune_dir)
    }


def _refreshed_context(state: PipelineNotebookState) -> Any:
    state.assert_sources_unchanged()
    from modules.tuning import load_tune_configs

    cell_config, sim_config = load_tune_configs(
        state.tune_dir,
        cell_name=state.context.cell_name,
    )
    return replace(
        state.context,
        cell_config=cell_config,
        sim_config=sim_config,
    )


def _protocol_params(
    defaults: Mapping[str, Any],
    *,
    sim_config: Mapping[str, Any],
    overrides: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    params = dict(defaults)
    params["conditions"] = dict(sim_config.get("conditions", {}) or {})
    for key, value in dict(overrides or {}).items():
        if value is not None:
            params[str(key)] = value
    return params


def _display_rows(title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    from modules.tuning import display_tuning_rows

    display_tuning_rows(title, rows)


def compute_passive_proposal(
    state: PipelineNotebookState,
    *,
    manual_passive_targets: Optional[Mapping[str, Any]] = None,
    passive_area_mode: str = "auto",
    passive_area_scale: float = 1.0,
) -> PipelinePassiveProposalResult:
    """Compute and display an ACT passive proposal without running a protocol.

    The proposal is review-only: this function never applies values to model
    sources or mutates the shared in-kernel cell.
    """

    context = _refreshed_context(state)
    from modules.tuning import (
        passive_proposal_changes,
        resolve_passive_tuning_inputs,
    )

    manual_targets = dict(manual_passive_targets or {})
    manual_mode = (
        "manual"
        if manual_targets
        and all(
            manual_targets.get(field) is not None
            for field in (
                "target_rin_mohm",
                "target_tau_ms",
                "target_v_rest_mv",
            )
        )
        else None
    )
    resolution = resolve_passive_tuning_inputs(
        context=context,
        cell=state.cell,
        manual_passive_targets=manual_targets,
        use_target_config=True,
        target_source_mode=manual_mode,
        compute_act_proposal=True,
        passive_area_mode=passive_area_mode,
        passive_area_scale=float(passive_area_scale),
        extract_from_nwb=False,
        apply_extracted_targets_to_config=False,
    )
    changes = passive_proposal_changes(
        settable_passive_properties=resolution.settable_passive_properties,
        target_file=(
            str(resolution.fit_json_candidates[0])
            if resolution.fit_json_candidates
            else "manual_review"
        ),
    )
    _display_rows("ACT passive proposal (review only; not applied)", changes)
    return PipelinePassiveProposalResult(
        resolution=resolution,
        proposal_changes=changes,
    )


def run_passive_stage(
    state: PipelineNotebookState,
    *,
    amps_pA: Sequence[float] = (-50.0, -100.0),
    compute_act_proposal: bool = False,
    manual_passive_targets: Optional[Mapping[str, Any]] = None,
    protocol_overrides: Optional[Mapping[str, Any]] = None,
    passive_area_mode: str = "auto",
    passive_area_scale: float = 1.0,
) -> PipelinePassiveResult:
    """Run the compact Step 2 protocol against the current target config."""

    context = _refreshed_context(state)
    sim_params = _protocol_params(
        PASSIVE_PROTOCOL_DEFAULTS,
        sim_config=context.sim_config,
        overrides=protocol_overrides,
    )
    amp_values = [float(value) for value in amps_pA]
    if not amp_values:
        raise ValueError("amps_pA must contain at least one passive current step.")

    from modules.tuning import (
        compare_passive_targets,
        display_passive_analysis,
        passive_amplitude_colors,
        passive_metric_rows,
        passive_proposal_changes,
        plot_passive_trace_check,
        resolve_passive_tuning_inputs,
        run_passive_protocol,
    )

    manual_targets = dict(manual_passive_targets or {})
    manual_mode = (
        "manual"
        if manual_targets
        and all(
            manual_targets.get(field) is not None
            for field in (
                "target_rin_mohm",
                "target_tau_ms",
                "target_v_rest_mv",
            )
        )
        else None
    )

    try:
        resolution = resolve_passive_tuning_inputs(
            context=context,
            cell=state.cell,
            manual_passive_targets=manual_targets,
            use_target_config=True,
            target_source_mode=manual_mode,
            compute_act_proposal=bool(compute_act_proposal),
            passive_area_mode=passive_area_mode,
            passive_area_scale=float(passive_area_scale),
            extract_from_nwb=False,
            apply_extracted_targets_to_config=False,
        )
    except ValueError as exc:
        if compute_act_proposal or "Missing passive targets" not in str(exc):
            raise
        print(
            "Passive targets are incomplete; running a target-free protocol. "
            "Fill target_config.json before enabling the ACT proposal."
        )
        resolution = resolve_passive_tuning_inputs(
            context=context,
            cell=state.cell,
            use_target_config=True,
            compute_act_proposal=False,
            target_source_mode="none",
            passive_area_mode=passive_area_mode,
            passive_area_scale=float(passive_area_scale),
        )

    records = run_passive_protocol(
        cell=state.cell,
        sim_params=sim_params,
        sim_amps=amp_values,
    )
    metrics = passive_metric_rows(
        act_passive_module=resolution.act_passive_module,
        looped_records=records,
        sim_params=sim_params,
        sim_amps=amp_values,
    )
    comparison = compare_passive_targets(metrics, resolution.passive_targets)
    proposal_changes = (
        passive_proposal_changes(
            settable_passive_properties=resolution.settable_passive_properties,
            target_file=(
                str(resolution.fit_json_candidates[0])
                if resolution.fit_json_candidates
                else "manual_review"
            ),
        )
        if resolution.settable_passive_properties is not None
        else []
    )

    single_trace_color = (
        context.cell_config.get("color") if len(amp_values) == 1 else None
    )
    amplitude_colors = passive_amplitude_colors(
        amp_values,
        single_trace_color=single_trace_color,
    )
    fig = plot_passive_trace_check(
        looped_records=records,
        sim_params=sim_params,
        sim_amps=amp_values,
        cell_name=context.cell_name,
        tune_name=context.tune_name,
        single_trace_color=single_trace_color,
        amplitude_colors=amplitude_colors,
    )

    import matplotlib.pyplot as plt

    plt.show()

    display_passive_analysis(
        metrics,
        comparison,
        amplitude_colors=amplitude_colors,
    )
    if proposal_changes:
        _display_rows("ACT passive proposal (review only; not applied)", proposal_changes)
    return PipelinePassiveResult(
        resolution=resolution,
        sim_params=sim_params,
        records=records,
        metrics=metrics,
        target_comparison=comparison,
        proposal_changes=proposal_changes,
        figure=fig,
    )


def run_active_protocol_stage(
    state: PipelineNotebookState,
    *,
    active_amps_pA: Sequence[float] = (150.0, 300.0),
    protocol_overrides: Optional[Mapping[str, Any]] = None,
    spike_threshold_mV: float = -20.0,
    include_currents: bool = True,
    current_amp_pA: Optional[float] = None,
) -> PipelineActiveProtocolResult:
    """Run and display only the compact active current-step protocol."""

    context = _refreshed_context(state)
    active_amps = [float(value) for value in active_amps_pA]
    if not active_amps:
        raise ValueError("active_amps_pA must contain at least one current step.")
    selected_current_amp = (
        active_amps[-1] if current_amp_pA is None else float(current_amp_pA)
    )
    if selected_current_amp not in active_amps:
        raise ValueError(
            "current_amp_pA must match one of the active_amps_pA values."
        )
    active_params = _protocol_params(
        ACTIVE_PROTOCOL_DEFAULTS,
        sim_config=context.sim_config,
        overrides=protocol_overrides,
    )

    from modules.tuning import (
        active_amplitude_colors,
        active_metric_rows,
        display_active_analysis,
        plot_active_trace_check,
        run_active_protocol,
    )

    active_records = run_active_protocol(
        cell=state.cell,
        sim_params=active_params,
        sim_amps=active_amps,
    )
    active_metrics = active_metric_rows(
        looped_records=active_records,
        sim_params=active_params,
        sim_amps=active_amps,
        threshold_mv=float(spike_threshold_mV),
    )
    amplitude_colors = active_amplitude_colors(
        active_amps,
        single_trace_color=(
            context.cell_config.get("color") if len(active_amps) == 1 else None
        ),
    )
    active_figure = plot_active_trace_check(
        looped_records=active_records,
        sim_params=active_params,
        sim_amps=active_amps,
        cell_name=context.cell_name,
        tune_name=context.tune_name,
        include_currents=bool(include_currents),
        current_amp=selected_current_amp,
        trace_color=context.cell_config.get("color"),
        amplitude_colors=amplitude_colors,
    )

    import matplotlib.pyplot as plt

    plt.show()
    display_active_analysis(
        active_metrics,
        amplitude_colors=amplitude_colors,
    )
    return PipelineActiveProtocolResult(
        sim_params=active_params,
        records=active_records,
        metrics=active_metrics,
        figure=active_figure,
    )


def run_fi_curve_stage(
    state: PipelineNotebookState,
    *,
    fi_amps_pA: Sequence[float] = tuple(range(0, 301, 50)),
    protocol_overrides: Optional[Mapping[str, Any]] = None,
    spike_threshold_mV: float = -20.0,
) -> PipelineFIResult:
    """Run and display only the FI sweep and configured-target comparison."""

    context = _refreshed_context(state)
    fi_amps = [float(value) for value in fi_amps_pA]
    if not fi_amps:
        raise ValueError("fi_amps_pA must contain at least one current step.")
    fi_params = _protocol_params(
        ACTIVE_PROTOCOL_DEFAULTS,
        sim_config=context.sim_config,
        overrides=protocol_overrides,
    )

    from modules.tuning import (
        active_metric_rows,
        compare_fi_targets,
        display_fi_analysis,
        fi_reference_points_from_csv,
        fi_rows_from_metrics,
        fi_series_colors,
        plot_fi_curve,
        resolve_active_tuning_targets,
        run_active_protocol,
    )

    fi_records = run_active_protocol(
        cell=state.cell,
        sim_params=fi_params,
        sim_amps=fi_amps,
    )
    fi_metrics = active_metric_rows(
        looped_records=fi_records,
        sim_params=fi_params,
        sim_amps=fi_amps,
        threshold_mv=float(spike_threshold_mV),
    )
    fi_rows = fi_rows_from_metrics(fi_metrics)
    target_resolution = resolve_active_tuning_targets(
        context=context,
        require_target=False,
    )
    if target_resolution.fi_csv_path is not None:
        target_reference = fi_reference_points_from_csv(
            target_resolution.fi_csv_path
        )
    else:
        target_reference = target_resolution.fi_reference_points
    comparison = compare_fi_targets(fi_rows, target_reference)
    series_colors = fi_series_colors(
        model_color=context.cell_config.get("color"),
    )
    fi_figure = plot_fi_curve(
        fi_rows=fi_rows,
        cell_name=context.cell_name,
        tune_name=context.tune_name,
        bio_reference=target_reference,
        show_bio_reference=bool(target_reference),
        model_color=series_colors["model"],
        reference_color=series_colors["reference"],
    )

    import matplotlib.pyplot as plt

    plt.show()
    display_fi_analysis(
        fi_rows,
        comparison,
        model_color=series_colors["model"],
        reference_color=series_colors["reference"],
    )
    return PipelineFIResult(
        sim_params=fi_params,
        records=fi_records,
        metrics=fi_metrics,
        fi_rows=fi_rows,
        target_reference=target_reference,
        target_comparison=comparison,
        figure=fi_figure,
    )


def run_active_stage(
    state: PipelineNotebookState,
    *,
    active_amps_pA: Sequence[float] = (150.0, 300.0),
    fi_amps_pA: Sequence[float] = tuple(range(0, 301, 50)),
    protocol_overrides: Optional[Mapping[str, Any]] = None,
    spike_threshold_mV: float = -20.0,
    include_currents: bool = True,
    current_amp_pA: Optional[float] = None,
) -> PipelineActiveResult:
    """Run both Step 3 actions for code-based/backward-compatible use."""

    active = run_active_protocol_stage(
        state,
        active_amps_pA=active_amps_pA,
        protocol_overrides=protocol_overrides,
        spike_threshold_mV=spike_threshold_mV,
        include_currents=include_currents,
        current_amp_pA=current_amp_pA,
    )
    fi = run_fi_curve_stage(
        state,
        fi_amps_pA=fi_amps_pA,
        protocol_overrides=protocol_overrides,
        spike_threshold_mV=spike_threshold_mV,
    )
    return PipelineActiveResult(
        active_sim_params=active.sim_params,
        active_records=active.records,
        active_metrics=active.metrics,
        fi_sim_params=fi.sim_params,
        fi_records=fi.records,
        fi_metrics=fi.metrics,
        fi_rows=fi.fi_rows,
        target_reference=fi.target_reference,
        target_comparison=fi.target_comparison,
        active_figure=active.figure,
        fi_figure=fi.figure,
    )


def prepare_interactive_synapse_tuner(
    state: PipelineNotebookState,
    *,
    connection_override: Optional[str] = None,
) -> Any:
    """Create the optional BMTool tuner while reusing the shared pipeline cell."""

    context = _refreshed_context(state)
    from modules.tuning import (
        SynapseTuningSession,
        connection_option,
        connection_settings_for_bmtool,
        create_scp_synapse_tuner,
        default_record_vars_for_connection,
        default_slider_vars_for_connection,
        ensure_synapse_tuning_config,
        general_settings_for_bmtool,
        selected_connection,
    )

    config_path, config, status = ensure_synapse_tuning_config(
        state.tune_dir,
        overwrite=False,
    )
    connection = selected_connection(config, connection_override)
    general_settings = general_settings_for_bmtool(config)
    connection_settings = connection_settings_for_bmtool(config)
    selected_settings = connection_settings[connection]
    slider_vars = connection_option(config, connection, "slider_vars")
    record_vars = connection_option(config, connection, "other_vars_to_record")
    if slider_vars is None:
        slider_vars = default_slider_vars_for_connection(selected_settings)
    if record_vars is None:
        record_vars = default_record_vars_for_connection(selected_settings)

    session = SynapseTuningSession(
        repo_root=state.repo_root,
        cell_name=context.cell_name,
        tune_name=context.tune_name,
        tune_dir=state.tune_dir,
        cell_config=context.cell_config,
        cell=state.cell,
        mechanism_summary=state.mechanism_summary,
        bmtool_path=None,
    )
    tuner = create_scp_synapse_tuner(
        session,
        conn_type_settings=connection_settings,
        connection=connection,
        general_settings=general_settings,
        current_name=config.get("current_name", "i"),
        other_vars_to_record=record_vars,
        slider_vars=slider_vars,
    )
    print("Synapse tuning config:", config_path)
    print("Config status:", status)
    print("Selected connection:", connection)
    print(
        "Mechanism:",
        selected_settings.get("spec_settings", {}).get("level_of_detail"),
    )
    print("Slider variables:", list(slider_vars))
    return tuner


def _unique_pipeline_stem() -> str:
    return datetime.now().strftime("pipeline_%Y%m%d_%H%M%S_%f")


def preview_pipeline_inputs(
    state: PipelineNotebookState,
    *,
    seed: Optional[int] = None,
    trial_idx: int = 0,
    stream_output: bool = True,
) -> PipelineInputPreviewResult:
    """Sample inputs and synapse placement safely in a fresh interpreter."""

    state.assert_sources_unchanged()
    trial = int(trial_idx)
    if trial < 0:
        raise ValueError("trial_idx must be non-negative.")

    with tempfile.TemporaryDirectory(prefix="scp_pipeline_preview_") as temp_dir:
        artifact = Path(temp_dir) / "input_preview.pkl"
        command = [
            sys.executable,
            "-m",
            "modules.notebooks.pipeline_preview_worker",
            "--tune-dir",
            str(state.tune_dir),
            "--trial-idx",
            str(trial),
            "--output",
            str(artifact),
        ]
        if seed is not None:
            command.extend(("--seed", str(int(seed))))

        env = dict(os.environ)
        env["SCP_ROOT"] = str(state.repo_root)
        env["PYTHONUNBUFFERED"] = "1"
        if stream_output:
            print("Checking inputs and synapse placement in a fresh process:")
            print(" ", " ".join(command))
        process = subprocess.Popen(
            command,
            cwd=str(state.repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_lines: list[str] = []
        assert process.stdout is not None
        try:
            for line in process.stdout:
                output_lines.append(line)
                if stream_output:
                    print(line, end="")
        finally:
            process.stdout.close()
        return_code = process.wait()
        stdout = "".join(output_lines)
        if return_code != 0:
            tail = "".join(output_lines[-30:])
            raise RuntimeError(
                "Fresh-process input preview failed with exit code "
                f"{return_code}.\n{tail}"
            )
        if not artifact.is_file():
            raise FileNotFoundError(
                "Input preview completed without producing its preview artifact."
            )
        with artifact.open("rb") as handle:
            payload = pickle.load(handle)

    if not isinstance(payload, dict) or not isinstance(payload.get("syn_state"), dict):
        raise TypeError("Input preview worker returned an invalid artifact.")
    return PipelineInputPreviewResult(
        syn_state=payload["syn_state"],
        summary=dict(payload.get("summary", {}) or {}),
        session_summary=dict(payload.get("session_summary", {}) or {}),
        trial_idx=int(payload.get("trial_idx", trial)),
        command=tuple(command),
        stdout=stdout,
    )


def run_fresh_simulation(
    state: PipelineNotebookState,
    *,
    n_trials: int = 1,
    seed: Optional[int] = None,
    iclamp: bool = False,
    output_stem: Optional[str] = None,
    stream_output: bool = True,
    sim_overrides: Optional[Mapping[str, Any]] = None,
) -> PipelineSimulationResult:
    """Run Step 5 in a fresh interpreter, force-save it, and load the result."""

    state.assert_sources_unchanged()
    stem = str(output_stem or _unique_pipeline_stem()).strip()
    if not stem:
        raise ValueError("output_stem must be non-empty when provided.")
    trials = int(n_trials)
    if trials < 1:
        raise ValueError("n_trials must be at least 1.")

    run_overrides = deepcopy(dict(sim_overrides or {}))
    iclamp_overrides = run_overrides.get("iclamp")
    if not isinstance(iclamp_overrides, dict):
        iclamp_overrides = {}
    iclamp_overrides["enabled"] = bool(iclamp)
    run_overrides["iclamp"] = iclamp_overrides
    try:
        serialized_overrides = json.dumps(
            run_overrides,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("sim_overrides must be JSON serializable.") from exc

    command = [
        sys.executable,
        str(state.repo_root / "run_pipeline.py"),
        "--tune-dir",
        str(state.tune_dir),
        "--n-trials",
        str(trials),
        "--force-save",
        "--output-stem",
        stem,
        "--sim-overrides-json",
        serialized_overrides,
    ]
    if seed is not None:
        command.extend(("--seed", str(int(seed))))
    if iclamp:
        command.append("--iclamp")

    env = dict(os.environ)
    env["SCP_ROOT"] = str(state.repo_root)
    env["PYTHONUNBUFFERED"] = "1"
    if stream_output:
        print("Running Step 5 in a fresh process:")
        print(" ", " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=str(state.repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    try:
        for line in process.stdout:
            output_lines.append(line)
            if stream_output:
                print(line, end="")
    finally:
        process.stdout.close()
    return_code = process.wait()
    stdout = "".join(output_lines)
    if return_code != 0:
        tail = "".join(output_lines[-30:])
        raise RuntimeError(
            f"Fresh Step 5 process failed with exit code {return_code}.\n{tail}"
        )

    manifest_path = state.tune_dir / "output_data" / stem / "run_manifest.json"
    if not manifest_path.is_file():
        candidates = sorted(
            (state.tune_dir / "output_data").glob(f"{stem}*/run_manifest.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                "Step 5 completed but its saved manifest could not be found under "
                f"{state.tune_dir / 'output_data'} for stem {stem!r}."
            )
        manifest_path = candidates[0]

    from modules import run_sim

    results = run_sim.load_results(manifest_path)
    if stream_output:
        print("Loaded fresh Step 5 result:", manifest_path)
    return PipelineSimulationResult(
        output_stem=manifest_path.parent.name,
        manifest_path=manifest_path,
        results=results,
        command=tuple(command),
        stdout=stdout,
    )


__all__ = [
    "ACTIVE_PROTOCOL_DEFAULTS",
    "PASSIVE_PROTOCOL_DEFAULTS",
    "PipelineActiveProtocolResult",
    "PipelineActiveResult",
    "PipelineFIResult",
    "PipelineInputPreviewResult",
    "PipelineNotebookState",
    "PipelinePassiveProposalResult",
    "PipelinePassiveResult",
    "PipelineSimulationResult",
    "RestartKernelRequired",
    "compute_passive_proposal",
    "prepare_interactive_synapse_tuner",
    "prepare_pipeline_notebook",
    "preview_pipeline_inputs",
    "run_active_stage",
    "run_active_protocol_stage",
    "run_fi_curve_stage",
    "run_fresh_simulation",
    "run_passive_stage",
]
