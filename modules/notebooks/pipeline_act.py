"""Fresh-process ACT active tuning for the compact pipeline notebook.

The compact UI is intentionally proposal-only.  This module prepares a
tune-local ACT workspace, validates it against the already loaded Step 1 cell,
and delegates optimization/evaluation to a clean Python process.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .pipeline_workflow import ACTIVE_PROTOCOL_DEFAULTS, PipelineNotebookState


@dataclass
class PipelineACTWorkspaceResult:
    workspace: Path
    config_path: Path
    builder_path: Path
    target_path: Path
    target_mode: str
    target_point_count: int
    loader_name: str
    loader_support: str
    config_source: str
    enabled_modules: list[str]
    workload: dict[str, Any]
    output_status: dict[str, dict[str, Any]]
    config_fingerprint: str
    target_config_fingerprint: str
    resolved_config: dict[str, Any]
    act_available: bool = False
    act_message: str = "ACT availability has not been probed."


@dataclass
class PipelineACTRunResult:
    modules: list[dict[str, Any]]
    predictions: dict[str, float]
    metrics: dict[str, list[dict[str, Any]]]
    output_status: dict[str, dict[str, Any]]
    config_fingerprint: str
    command: tuple[str, ...]
    stdout: str


@dataclass
class PipelineACTEvaluationResult:
    predictions: dict[str, float]
    fi_rows: list[dict[str, float]]
    target_reference: list[tuple[float, float]]
    target_comparison: list[dict[str, Any]]
    fi_path: Path
    output_dir: Path
    manifest_path: Path
    command: tuple[str, ...]
    stdout: str
    figure: Any = None


def inspect_act_active_stage(
    state: PipelineNotebookState,
    *,
    workspace: Optional[str | Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> PipelineACTWorkspaceResult:
    """Resolve and validate a compact ACT workspace without writing it."""

    state.assert_sources_unchanged()
    from modules.loaders import get_cell_loader_name
    from modules.setup.mechanisms import resolve_modfiles_dir
    from modules.tuning import (
        act_config_fingerprint,
        act_output_status,
        default_act_active_config,
        estimate_act_workload,
        load_act_active_config,
        resolve_active_tuning_targets,
        validate_act_module_specs,
    )
    from modules.tuning.act_active import CONFIG_NAME

    tune_dir = Path(state.tune_dir).resolve()
    workspace_path = _resolve_workspace(tune_dir, workspace)
    config_path = workspace_path / CONFIG_NAME
    if config_path.is_file():
        cfg = load_act_active_config(config_path)
        config_source = "existing"
    elif str(state.context.cell_name).strip().upper() in {"PV", "SST"}:
        cfg = default_act_active_config(
            repo_root=state.repo_root,
            tune_dir=tune_dir,
            cell_name=state.context.cell_name,
            tune_name=state.context.tune_name,
            workspace=workspace_path,
        )
        cfg["simulation"].update(
            {
                "h_tstop": float(ACTIVE_PROTOCOL_DEFAULTS["h_tstop"]),
                "h_dt": float(ACTIVE_PROTOCOL_DEFAULTS["h_dt"]),
                "ci_delay_ms": float(ACTIVE_PROTOCOL_DEFAULTS["stim_delay"]),
                "ci_dur_ms": float(ACTIVE_PROTOCOL_DEFAULTS["stim_dur"]),
            }
        )
        config_source = "curated preset"
    else:
        raise FileNotFoundError(
            "Compact ACT auto-configuration is available only for the curated PV/SST "
            "presets. Configure and validate act_active_config.json in 3_active.ipynb "
            f"first, then return here. Expected: {config_path}"
        )

    cfg = deepcopy(cfg)
    cfg.update(
        {
            "repo_root": str(Path(state.repo_root).resolve()),
            "tune_dir": str(tune_dir),
            "workspace": str(workspace_path),
            "cell_name": str(state.context.cell_name),
            "tune_name": str(state.context.tune_name),
        }
    )
    loader_name = get_cell_loader_name(state.context.cell_config)
    cfg["cell_loader"] = loader_name
    cfg["act_loader_status"] = (
        "supported" if loader_name == "allen_manifest" else "experimental"
    )

    normalized = _normalize_overrides(overrides)
    target_overrides = normalized["target"]
    use_existing_trace = (
        config_source == "existing"
        and not target_overrides
        and str((cfg.get("target") or {}).get("mode", "")).lower() == "trace_npy"
    )
    if not use_existing_trace:
        resolution = resolve_active_tuning_targets(
            context=state.context,
            target_mode=target_overrides.get("mode"),
            fi_currents_pA=target_overrides.get("fi_currents_pA"),
            fi_frequencies_hz=target_overrides.get("fi_frequencies_hz"),
            fi_csv_path=target_overrides.get("fi_csv_path"),
            trace_npy_path=target_overrides.get("trace_npy_path"),
            nwb_path=target_overrides.get("nwb_path"),
            require_target=True,
        )
        resolved_target = _target_config_from_resolution(resolution)
        if not resolved_target.get("fi_currents_pA") and cfg["target"].get(
            "fi_currents_pA"
        ):
            resolved_target.pop("fi_currents_pA", None)
            resolved_target.pop("fi_frequencies_hz", None)
        cfg["target"].update(resolved_target)
        nwb_options = dict(cfg["target"].get("nwb_options") or {})
        for source_key, target_key in (
            ("nwb_stimulus_names", "stimulus_names"),
            ("nwb_include_negative_currents", "include_negative_currents"),
            ("nwb_min_current_pA", "min_current_pA"),
            ("nwb_max_current_pA", "max_current_pA"),
            ("nwb_average_repeats", "average_repeats"),
            ("nwb_spike_threshold_mV", "spike_threshold_mV"),
            ("nwb_refractory_ms", "refractory_ms"),
        ):
            if source_key in target_overrides:
                nwb_options[target_key] = deepcopy(target_overrides[source_key])
        if nwb_options:
            cfg["target"]["nwb_options"] = nwb_options
        cfg.setdefault("simulation", {})["ci_amps_pA"] = list(
            resolution.fi_currents_pA
        )
        if resolution.target_mode == "fi_csv" and resolution.fi_csv_path:
            rows = _read_target_csv_points(resolution.fi_csv_path)
            cfg["target"]["fi_currents_pA"] = [row[0] for row in rows]
            cfg["target"]["fi_frequencies_hz"] = [row[1] for row in rows]
            cfg["simulation"]["ci_amps_pA"] = [row[0] for row in rows]
    else:
        _validate_existing_trace_target(cfg, workspace_path)

    _deep_update(cfg.setdefault("act_cell", {}), normalized["act_cell"])
    _deep_update(cfg.setdefault("simulation", {}), normalized["simulation"])
    _deep_update(cfg.setdefault("optimizer", {}), normalized["optimizer"])
    _deep_update(cfg.setdefault("filter", {}), normalized["filter"])
    if normalized["modules"]:
        cfg["modules"] = deepcopy(normalized["modules"])

    # ACT currents always follow the selected target to avoid a length mismatch.
    target_currents = list((cfg.get("target") or {}).get("fi_currents_pA") or [])
    if target_currents:
        cfg["simulation"]["ci_amps_pA"] = target_currents

    validate_act_module_specs(cfg.get("modules") or {})
    _validate_cell_adapter(state.cell, cfg)
    modfiles = resolve_modfiles_dir(tune_dir)
    if modfiles is not None and not Path(modfiles).is_dir():
        raise FileNotFoundError(
            f"Configured ACT mechanism path does not exist: {modfiles}. Complete "
            "Step 1 setup or correct paths.modfiles in 1_setup.ipynb."
        )

    workload = estimate_act_workload(cfg)
    if workload["target_points"] < 1 and cfg["target"].get("mode") != "allen_nwb":
        raise ValueError("ACT target data contains no usable current/frequency points.")
    target_path = _prospective_target_path(cfg, workspace_path)
    output_status = act_output_status(cfg) if config_path.is_file() else {
        key: {
            "status": "missing",
            "prediction_path": str(workspace_path / f"prediction_{key}.json"),
            "metrics_path": str(workspace_path / f"metrics_{key}.csv"),
            "manifest_path": str(workspace_path / f"run_manifest_{key}.json"),
            "provenance_verified": False,
        }
        for key, spec in (cfg.get("modules") or {}).items()
        if spec.get("enabled", True)
    }
    return PipelineACTWorkspaceResult(
        workspace=workspace_path,
        config_path=config_path,
        builder_path=workspace_path / "cell_builder.py",
        target_path=target_path,
        target_mode=str(cfg["target"]["mode"]),
        target_point_count=int(workload["target_points"]),
        loader_name=loader_name,
        loader_support=cfg["act_loader_status"],
        config_source=config_source,
        enabled_modules=[
            str(key)
            for key, spec in (cfg.get("modules") or {}).items()
            if spec.get("enabled", True)
        ],
        workload=workload,
        output_status=output_status,
        config_fingerprint=act_config_fingerprint(cfg),
        target_config_fingerprint=_target_config_fingerprint(tune_dir),
        resolved_config=cfg,
    )


def prepare_act_active_stage(
    state: PipelineNotebookState,
    *,
    workspace: Optional[str | Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    probe_act: bool = True,
) -> PipelineACTWorkspaceResult:
    """Write and probe the resolved compact ACT workspace."""

    inspection = inspect_act_active_stage(
        state, workspace=workspace, overrides=overrides
    )
    cfg = inspection.resolved_config
    target = cfg["target"]
    nwb = target.get("nwb_options", {}) or {}
    from modules.tuning import prepare_act_active_workspace

    prepare_act_active_workspace(
        repo_root=state.repo_root,
        tune_dir=state.tune_dir,
        cell_name=state.context.cell_name,
        tune_name=state.context.tune_name,
        workspace=inspection.workspace,
        target_mode=target.get("mode"),
        fi_currents_pA=target.get("fi_currents_pA"),
        fi_frequencies_hz=target.get("fi_frequencies_hz"),
        fi_csv_path=target.get("source_csv"),
        trace_npy_path=target.get("source_npy"),
        nwb_path=target.get("source_nwb"),
        nwb_stimulus_names=nwb.get("stimulus_names"),
        nwb_include_negative_currents=bool(nwb.get("include_negative_currents", False)),
        nwb_min_current_pA=nwb.get("min_current_pA", 0.0),
        nwb_max_current_pA=nwb.get("max_current_pA"),
        nwb_average_repeats=bool(nwb.get("average_repeats", True)),
        nwb_spike_threshold_mV=float(nwb.get("spike_threshold_mV", -20.0)),
        nwb_refractory_ms=float(nwb.get("refractory_ms", 1.0)),
        passive_names=cfg.get("act_cell", {}).get("passive"),
        active_channels=cfg.get("act_cell", {}).get("active_channels"),
        module_specs=cfg.get("modules"),
        sim_params=cfg.get("simulation"),
        optimizer=cfg.get("optimizer"),
        filter_params=cfg.get("filter"),
        overwrite_config=True,
        preserve_existing=inspection.config_source == "existing",
    )
    prepared = inspect_act_active_stage(
        state, workspace=inspection.workspace, overrides=overrides
    )
    prepared.config_source = inspection.config_source
    from modules.tuning import (
        act_config_fingerprint,
        act_output_status,
        estimate_act_workload,
        load_act_active_config,
    )
    from modules.tuning.act_active import resolve_workspace_path

    actual_cfg = load_act_active_config(prepared.config_path)
    prepared.resolved_config = actual_cfg
    prepared.config_fingerprint = act_config_fingerprint(prepared.config_path)
    prepared.output_status = act_output_status(prepared.config_path)
    prepared.workload = estimate_act_workload(actual_cfg)
    prepared.target_point_count = int(prepared.workload["target_points"])
    prepared.target_mode = str(actual_cfg.get("target", {}).get("mode"))
    prepared.target_path = resolve_workspace_path(
        prepared.workspace, actual_cfg.get("target", {}).get("path", "target_sf.csv")
    )
    if probe_act:
        try:
            payload, _command, _stdout = _run_worker_action(
                repo_root=state.repo_root,
                action="probe",
                request={"config_path": str(prepared.config_path)},
            )
        except Exception as exc:
            prepared.act_available = False
            prepared.act_message = _friendly_act_probe_error(exc)
        else:
            prepared.act_available = bool(payload.get("available"))
            prepared.act_message = str(payload.get("message") or "ACT is available.")
    return prepared


def run_fresh_act_active(
    state: PipelineNotebookState,
    workspace_result: PipelineACTWorkspaceResult,
    *,
    modules: str | Sequence[str],
    n_cpus: Optional[int] = None,
    overwrite: bool = False,
    line_callback: Optional[Callable[[str], None]] = None,
    process_callback: Optional[Callable[[subprocess.Popen[str]], None]] = None,
) -> PipelineACTRunResult:
    """Run ACT optimization in a fresh process and load its saved results."""

    state.assert_sources_unchanged()
    _assert_workspace_prepared(state, workspace_result)
    payload, command, stdout = _run_worker_action(
        repo_root=state.repo_root,
        action="run",
        request={
            "config_path": str(workspace_result.config_path),
            "modules": modules,
            "n_cpus": n_cpus,
            "overwrite": bool(overwrite),
        },
        line_callback=line_callback,
        process_callback=process_callback,
    )
    return PipelineACTRunResult(
        modules=list(payload.get("modules") or []),
        predictions={str(k): float(v) for k, v in (payload.get("predictions") or {}).items()},
        metrics=dict(payload.get("metrics") or {}),
        output_status=dict(payload.get("output_status") or {}),
        config_fingerprint=str(payload.get("config_fingerprint") or ""),
        command=command,
        stdout=stdout,
    )


def evaluate_fresh_act_predictions(
    state: PipelineNotebookState,
    workspace_result: PipelineACTWorkspaceResult,
    *,
    predictions: Optional[Mapping[str, float]] = None,
    n_cpus: Optional[int] = None,
    overwrite: bool = False,
    display: bool = True,
    line_callback: Optional[Callable[[str], None]] = None,
    process_callback: Optional[Callable[[subprocess.Popen[str]], None]] = None,
) -> PipelineACTEvaluationResult:
    """Evaluate merged ACT predictions without modifying tune model sources."""

    state.assert_sources_unchanged()
    _assert_workspace_prepared(state, workspace_result)
    payload, command, stdout = _run_worker_action(
        repo_root=state.repo_root,
        action="evaluate",
        request={
            "config_path": str(workspace_result.config_path),
            "predictions": None if predictions is None else dict(predictions),
            "n_cpus": n_cpus,
            "overwrite": bool(overwrite),
        },
        line_callback=line_callback,
        process_callback=process_callback,
    )
    result = _evaluation_result_from_payload(
        state, payload=payload, command=command, stdout=stdout
    )
    if display:
        display_act_evaluation(state, result)
    return result


def display_act_evaluation(
    state: PipelineNotebookState,
    result: PipelineACTEvaluationResult,
) -> Any:
    """Render ACT evaluation using the standard compact Step 3 FI styling."""

    from modules.tuning import display_fi_analysis, fi_series_colors, plot_fi_curve

    colors = fi_series_colors(model_color=state.context.cell_config.get("color"))
    result.figure = plot_fi_curve(
        fi_rows=result.fi_rows,
        cell_name=state.context.cell_name,
        tune_name=state.context.tune_name,
        bio_reference=result.target_reference,
        show_bio_reference=bool(result.target_reference),
        model_color=colors["model"],
        reference_color=colors["reference"],
    )
    import matplotlib.pyplot as plt

    plt.show()
    display_fi_analysis(
        result.fi_rows,
        result.target_comparison,
        model_color=colors["model"],
        reference_color=colors["reference"],
    )
    return result.figure


def _evaluation_result_from_payload(
    state: PipelineNotebookState,
    *,
    payload: Mapping[str, Any],
    command: tuple[str, ...],
    stdout: str,
) -> PipelineACTEvaluationResult:
    from modules.tuning import compare_fi_targets, load_act_active_config

    cfg = load_act_active_config(payload.get("config_path") or _evaluation_config_path(payload))
    target = cfg.get("target", {}) or {}
    reference = list(
        zip(
            [float(value) for value in target.get("fi_currents_pA") or []],
            [float(value) for value in target.get("fi_frequencies_hz") or []],
        )
    )
    rows = [
        {"amp_pA": float(row["amp_pA"]), "spike_frequency_hz": float(row["spike_frequency_hz"])}
        for row in payload.get("fi_rows") or []
    ]
    return PipelineACTEvaluationResult(
        predictions={str(k): float(v) for k, v in (payload.get("prediction") or {}).items()},
        fi_rows=rows,
        target_reference=reference,
        target_comparison=compare_fi_targets(rows, reference),
        fi_path=Path(payload["fi_path"]),
        output_dir=Path(payload["output_dir"]),
        manifest_path=Path(payload["manifest_path"]),
        command=command,
        stdout=stdout,
    )


def _evaluation_config_path(payload: Mapping[str, Any]) -> Path:
    manifest = Path(payload["manifest_path"])
    return manifest.parent / "act_active_config.json"


def _run_worker_action(
    *,
    repo_root: str | Path,
    action: str,
    request: Mapping[str, Any],
    line_callback: Optional[Callable[[str], None]] = None,
    process_callback: Optional[Callable[[subprocess.Popen[str]], None]] = None,
) -> tuple[dict[str, Any], tuple[str, ...], str]:
    with tempfile.TemporaryDirectory(prefix="scp_pipeline_act_") as temp_dir:
        temp = Path(temp_dir)
        request_path = temp / "request.json"
        result_path = temp / "result.json"
        request_path.write_text(
            json.dumps({"action": action, **dict(request)}, indent=2) + "\n",
            encoding="utf-8",
        )
        command = (
            sys.executable,
            "-u",
            "-m",
            "modules.notebooks.pipeline_act_worker",
            "--request",
            str(request_path),
            "--result",
            str(result_path),
        )
        process = subprocess.Popen(
            command,
            cwd=str(Path(repo_root).resolve()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=(os.name != "nt"),
        )
        if process_callback is not None:
            process_callback(process)
        lines: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            if line_callback is not None:
                line_callback(line)
        return_code = process.wait()
        stdout = "".join(lines)
        if return_code != 0:
            tail = "\n".join(stdout.strip().splitlines()[-20:])
            raise RuntimeError(
                f"ACT {action} worker failed with exit code {return_code}.\n{tail}"
            )
        if not result_path.is_file():
            raise RuntimeError(f"ACT {action} worker produced no result file.")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return payload, command, stdout


def _resolve_workspace(tune_dir: Path, value: Optional[str | Path]) -> Path:
    from modules.tuning import default_act_workspace

    if value in (None, ""):
        return default_act_workspace(tune_dir)
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = tune_dir / path
    return path.resolve()


def _friendly_act_probe_error(exc: Exception) -> str:
    message = str(exc)
    if "No module named 'sklearn'" in message:
        return (
            "ACT was found, but scikit-learn is missing. Update the SCP environment "
            "from environment.yml (or install scikit-learn) and restart the kernel."
        )
    if "No module named 'timeout_decorator'" in message:
        return (
            "ACT was found, but timeout-decorator is missing. Update the SCP "
            "environment from environment.yml and restart the kernel."
        )
    if "ACT repo not found" in message:
        return (
            message
            + " Locally, clone ACT or set SCP_ACT_PATH; Colab can auto-clone during preparation."
        )
    return message


def _target_config_fingerprint(tune_dir: Path) -> str:
    candidates = (
        tune_dir / "cell_configs" / "target_config.json",
        tune_dir / "target_config.json",
    )
    digest = hashlib.sha256()
    for path in candidates:
        if path.is_file():
            digest.update(str(path.resolve()).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _assert_workspace_prepared(
    state: PipelineNotebookState,
    result: PipelineACTWorkspaceResult,
) -> None:
    from modules.tuning import act_config_fingerprint

    current_target = _target_config_fingerprint(Path(state.tune_dir).resolve())
    if current_target != result.target_config_fingerprint:
        raise RuntimeError(
            "target_config.json changed after ACT preparation. Click Prepare ACT workspace again."
        )
    current_config = act_config_fingerprint(result.config_path)
    if current_config != result.config_fingerprint:
        raise RuntimeError(
            "The ACT config or prepared target changed after preparation. "
            "Click Prepare ACT workspace again."
        )


def _normalize_overrides(value: Optional[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    source = dict(value or {})
    return {
        key: deepcopy(dict(source.get(key) or {}))
        for key in ("target", "act_cell", "simulation", "optimizer", "filter", "modules")
    }


def _deep_update(destination: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if value is not None:
            destination[key] = deepcopy(value)


def _target_config_from_resolution(resolution: Any) -> dict[str, Any]:
    target: dict[str, Any] = {
        "mode": resolution.target_mode,
        "fi_currents_pA": list(resolution.fi_currents_pA),
        "fi_frequencies_hz": list(resolution.fi_frequencies_hz),
    }
    if resolution.fi_csv_path is not None:
        target["source_csv"] = str(resolution.fi_csv_path)
    if resolution.trace_npy_path is not None:
        target["source_npy"] = str(resolution.trace_npy_path)
    if resolution.nwb_path is not None:
        target["source_nwb"] = str(resolution.nwb_path)
        target["nwb_options"] = dict(resolution.nwb_options)
    return target


def _read_target_csv_points(path: Path) -> list[tuple[float, float]]:
    from modules.tuning import fi_reference_points_from_csv

    return fi_reference_points_from_csv(path)


def _validate_existing_trace_target(cfg: Mapping[str, Any], workspace: Path) -> None:
    import numpy as np

    target = cfg.get("target", {}) or {}
    raw = target.get("source_npy") or target.get("path")
    if not raw:
        raise ValueError("Existing trace_npy ACT configuration has no target path.")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = workspace / path
    if not path.is_file():
        raise FileNotFoundError(f"Existing ACT trace target not found: {path}")
    data = np.load(path, mmap_mode="r")
    currents = list((cfg.get("simulation") or {}).get("ci_amps_pA") or [])
    if not currents or not data.ndim or int(data.shape[0]) != len(currents):
        raise ValueError(
            "Existing trace_npy ACT configuration must include one ci_amps_pA value "
            "for each target trace. Configure it in 3_active.ipynb."
        )


def _prospective_target_path(cfg: Mapping[str, Any], workspace: Path) -> Path:
    target = cfg.get("target", {}) or {}
    raw = target.get("path")
    if target.get("mode") == "trace_npy":
        raw = target.get("source_npy") or raw
    if raw:
        path = Path(str(raw)).expanduser()
        return path if path.is_absolute() else workspace / path
    return workspace / "target_sf.csv"


def _validate_cell_adapter(cell: Any, cfg: Mapping[str, Any]) -> None:
    soma = getattr(cell, "soma", None)
    if soma is None:
        raise AttributeError("ACT requires the registered loader to expose a soma section.")
    try:
        sections = list(soma) if not callable(soma) else [soma]
        segment = sections[0](0.5)
    except Exception as exc:
        raise AttributeError("ACT could not access soma[0](0.5) on the loaded cell.") from exc
    missing: list[str] = []
    for name in (cfg.get("act_cell") or {}).get("passive", []) or []:
        if name and not _has_nested_attribute(segment, str(name)):
            missing.append(str(name))
    for spec in (cfg.get("modules") or {}).values():
        if not spec.get("enabled", True):
            continue
        for conductance in spec.get("conductances", []) or []:
            name = str(conductance.get("variable_name") or "")
            if name and not _has_nested_attribute(segment, name):
                missing.append(name)
    if missing:
        raise AttributeError(
            "ACT adapter variables are not available at soma[0](0.5): "
            + ", ".join(sorted(set(missing)))
        )


def _has_nested_attribute(obj: Any, name: str) -> bool:
    current = obj
    try:
        for part in str(name).split("."):
            current = getattr(current, part)
    except Exception:
        return False
    return True


__all__ = [
    "PipelineACTEvaluationResult",
    "PipelineACTRunResult",
    "PipelineACTWorkspaceResult",
    "display_act_evaluation",
    "evaluate_fresh_act_predictions",
    "inspect_act_active_stage",
    "prepare_act_active_stage",
    "run_fresh_act_active",
]
