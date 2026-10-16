"""Shared setup helpers for Step 2/3 tuning notebooks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import json
import os
import sys


@dataclass(frozen=True)
class Step1ValidationReport:
    """File/layout validation summary for a prepared tune directory."""

    tune_dir: Path
    checks: Dict[str, bool]
    paths: Dict[str, Path]
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(self.checks.values())

    @property
    def missing(self) -> Dict[str, Path]:
        return {key: self.paths[key] for key, ok in self.checks.items() if not ok}

    def raise_for_missing(self) -> None:
        if self.ok:
            return
        lines = []
        for name, path in self.missing.items():
            detail = self.errors.get(name)
            suffix = f" ({detail})" if detail else ""
            lines.append(f"{name}: {path}{suffix}")
        raise FileNotFoundError(
            "Step 1 validation failed; missing required paths:\n" + "\n".join(lines)
        )


@dataclass(frozen=True)
class TuningNotebookContext:
    """Resolved Step 2/3 notebook context."""

    repo_root: Path
    tune_dir: Path
    cell_name: str
    tune_name: str
    cell_config: Dict[str, Any]
    sim_config: Dict[str, Any]
    validation: Step1ValidationReport

    @property
    def cell_configs_dir(self) -> Path:
        return self.tune_dir / "cell_configs"

    @property
    def output_dir(self) -> Path:
        return self.tune_dir / "output_data"

    @property
    def tuning_exports_dir(self) -> Path:
        return self.tune_dir / "tuning_exports"

    def summary(self) -> Dict[str, Any]:
        summary = {
            "repo_root": str(self.repo_root),
            "tune_dir": str(self.tune_dir),
            "cell_name": self.cell_name,
            "tune_name": self.tune_name,
            "cell_loader": self.cell_config.get("cell_loader"),
            "validation_ok": self.validation.ok,
            "project_name": self.project_name,
            "workspace_identity": self.workspace_identity,
        }
        multiplier = (self.cell_config.get("tuning", {}) or {}).get(
            "soma_diam_multiplier"
        )
        if multiplier is not None:
            summary["soma_diam_multiplier"] = multiplier
        return summary

    @property
    def workspace_identity(self):
        from scp_pipeline.workspaces import infer_workspace
        return infer_workspace(self.tune_dir, cell_config=self.cell_config,
                               cell_name=self.cell_name, tune_name=self.tune_name).identity

    @property
    def project_name(self):
        return self.workspace_identity['project_name']


def _looks_like_repo_root(path: Path) -> bool:
    return (path / "modules").is_dir() and (path / "run_pipeline.py").is_file()


def resolve_repo_root(start: Optional[Path] = None) -> Path:
    """Locate the SCP repository root from a notebook or script context."""
    from scp_pipeline.runtime import code_root
    if start is not None and (Path(start).resolve() == code_root() or _looks_like_repo_root(Path(start))):
        return Path(start).resolve()
    if start is None:
        return code_root()
    env_root = os.environ.get("SCP_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if _looks_like_repo_root(candidate):
            return candidate
        raise FileNotFoundError(f"SCP_ROOT does not point to an SCP repo: {candidate}")

    start_path = (start or Path.cwd()).expanduser().resolve()
    candidates: list[Path] = [start_path, *start_path.parents]
    for base in (start_path, start_path.parent):
        try:
            candidates.extend(child for child in base.iterdir() if child.is_dir())
        except Exception:
            pass

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if _looks_like_repo_root(resolved):
            return resolved

    raise FileNotFoundError(
        f"Could not locate SCP repo root from {start_path}. "
        "Set SCP_ROOT or launch Jupyter from inside the repo."
    )


def ensure_repo_on_syspath(repo_root: Optional[Path] = None) -> Path:
    """Resolve code location without shadowing a host project's imports."""
    return resolve_repo_root(repo_root)


def resolve_tune_dir(
    *,
    repo_root: Optional[Path] = None,
    cell_name: Optional[str] = None,
    tune_name: Optional[str] = None,
    tunes_parent: str = "tunes",
    tune_dir_override: Optional[str | Path] = None,
    project_name: Optional[str] = None,
    project_root: Optional[str | Path] = None,
    workspace_root: Optional[str | Path] = None,
) -> Path:
    """Resolve a tune directory from explicit path or cell/tune labels."""
    from scp_pipeline.workspaces import resolve_workspace
    return resolve_workspace(repo_root=resolve_repo_root(repo_root), project_name=project_name,
        cell_name=cell_name, tune_name=tune_name, tunes_parent=tunes_parent,
        tune_dir_override=tune_dir_override, project_root=project_root, workspace_root=workspace_root).tune_dir


def _read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


def _compiled_mechanism_exists(tune_dir: Path, cell_config: Dict[str, Any]) -> bool:
    from scp_pipeline.setup.mechanisms import find_compiled_mechanism_dll
    if find_compiled_mechanism_dll(tune_dir, cell_config=cell_config) is not None:
        return True
    from scp_pipeline.setup.mechanisms import resolve_modfiles_dir

    mod_dir = resolve_modfiles_dir(tune_dir, cell_config)
    if mod_dir is None:
        return False
    from scp_pipeline.loaders.source_context import within
    if not within(mod_dir, Path(tune_dir).resolve()):
        return False  # Upstream compiled outputs are never SCP's managed build.
    candidates = (
        mod_dir / "x86_64" / ".libs" / "libnrnmech.so",
        mod_dir / "x86_64" / "libnrnmech.so",
        mod_dir / "nrnmech.dll",
        mod_dir / "x86_64" / "special",
    )
    return any(path.exists() for path in candidates)


def validate_step1_outputs(
    tune_dir: Path,
    *,
    require_sim_config: bool = True,
    require_geometry_config: bool = True,
    require_synapse_config: bool = False,
    require_compiled_modfiles: bool = True,
    raise_on_missing: bool = True,
) -> Step1ValidationReport:
    """Validate the Step 1 file/layout contract needed by Steps 2/3."""
    tune_path = Path(tune_dir).expanduser().resolve()
    cell_configs = tune_path / "cell_configs"
    paths: Dict[str, Path] = {
        "tune_dir": tune_path,
        "cell_config": cell_configs / "cell_config.json",
    }
    if require_sim_config:
        paths["sim_config"] = cell_configs / "sim_config.json"
    if require_geometry_config:
        paths["geometry"] = cell_configs / "geometry.json"
    if require_synapse_config:
        paths["syn_config"] = cell_configs / "syn_config.json"

    checks: Dict[str, bool] = {
        "tune_dir": tune_path.is_dir(),
        "cell_config": paths["cell_config"].is_file(),
    }
    errors: Dict[str, str] = {}
    cell_config: Dict[str, Any] = {}
    if paths["cell_config"].is_file():
        from scp_pipeline.loaders import validate_cell_loader_config

        try:
            cell_config = _read_json(paths["cell_config"])
            for role, resolved in validate_cell_loader_config(
                cell_config,
                base_dir=tune_path,
            ).items():
                key = f"loader:{role}"
                paths[key] = resolved
                checks[key] = resolved.is_file()
        except Exception as exc:
            if raise_on_missing:
                raise
            paths["loader_config"] = paths["cell_config"]
            checks["loader_config"] = False
            errors["loader_config"] = f"{type(exc).__name__}: {exc}"

    from scp_pipeline.setup.mechanisms import resolve_modfiles_dir

    mod_dir = resolve_modfiles_dir(tune_path, cell_config)
    mod_sources = (
        sorted(mod_dir.glob("*.mod")) if mod_dir is not None and mod_dir.is_dir() else []
    )
    if mod_sources:
        paths["modfiles"] = mod_dir
        paths["compiled_modfiles"] = mod_dir
        checks["modfiles"] = True
        checks["compiled_modfiles"] = (
            not require_compiled_modfiles
        ) or _compiled_mechanism_exists(tune_path, cell_config)
    if require_sim_config:
        checks["sim_config"] = paths["sim_config"].is_file()
    if require_geometry_config:
        checks["geometry"] = paths["geometry"].is_file()
    if require_synapse_config:
        checks["syn_config"] = paths["syn_config"].is_file()

    report = Step1ValidationReport(
        tune_dir=tune_path,
        checks=checks,
        paths=paths,
        errors=errors,
    )
    if raise_on_missing:
        report.raise_for_missing()
    return report


def load_tune_configs(
    tune_dir: Path,
    *,
    cell_name: Optional[str] = None,
    require_sim_config: bool = True,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load and lightly normalize `cell_config.json` and `sim_config.json`."""
    tune_path = Path(tune_dir).expanduser().resolve()
    cell_config = _read_json(tune_path / "cell_configs" / "cell_config.json")
    if cell_name:
        cell_config.setdefault("cell_name", str(cell_name))
    cell_config.setdefault("cell_loader", "allen_manifest")
    tuning = cell_config.get("tuning")
    if isinstance(tuning, dict) and "soma_diam_multiplier" in tuning:
        tuning["soma_diam_multiplier"] = float(tuning["soma_diam_multiplier"])

    sim_path = tune_path / "cell_configs" / "sim_config.json"
    sim_config = _read_json(sim_path) if sim_path.is_file() else {}
    if require_sim_config and not sim_config:
        raise FileNotFoundError(f"Missing sim_config.json at {sim_path}")
    return cell_config, sim_config


def _infer_cell_tune_from_path(tune_dir: Path) -> tuple[str, str]:
    from scp_pipeline.workspaces import infer_workspace
    workspace = infer_workspace(tune_dir)
    return workspace.cell_name, workspace.tune_name


def prepare_tuning_notebook_context(
    *,
    cell_name: Optional[str] = None,
    tune_name: Optional[str] = None,
    tunes_parent: str = "tunes",
    tune_dir_override: Optional[str | Path] = None,
    repo_root: Optional[Path] = None,
    require_compiled_modfiles: bool = True,
    require_synapse_config: bool = False,
    print_summary: bool = True,
    cell_override: Any = None,
    runtime_sim_config: Optional[Dict[str, Any]] = None,
    project_name: Optional[str] = None,
    project_root: Optional[str | Path] = None,
    workspace_root: Optional[str | Path] = None,
) -> TuningNotebookContext:
    """Resolve, validate, and load common Step 2/3 notebook state."""
    root = ensure_repo_on_syspath(repo_root)
    tune_dir = resolve_tune_dir(
        repo_root=root,
        cell_name=cell_name,
        tune_name=tune_name,
        tunes_parent=tunes_parent,
        tune_dir_override=tune_dir_override,
        project_name=project_name,
        project_root=project_root, workspace_root=workspace_root,
    )
    from scp_pipeline.workspaces import infer_workspace
    workspace = infer_workspace(tune_dir, cell_name=cell_name, tune_name=tune_name, project_name=project_name)
    resolved_cell, resolved_tune = workspace.cell_name, workspace.tune_name

    if cell_override is not None:
        from scp_pipeline.model.runtime_cell import wrap_cell, prepare_runtime_workspace, adopt_cell
        adopted = wrap_cell(cell_override)
        sim_config = prepare_runtime_workspace(tune_dir, adopted, runtime_sim_config)
        cell_config, _ = load_tune_configs(tune_dir, cell_name=resolved_cell)
        cell_config = dict(adopt_cell(adopted, cell_config).config)
        context = TuningNotebookContext(root, tune_dir, resolved_cell, resolved_tune, cell_config,
                                       sim_config, Step1ValidationReport(tune_dir, {'runtime_cell': True}, {}))
        if print_summary:
            print_tuning_context_summary(context)
        return context
    validation = validate_step1_outputs(
        tune_dir,
        require_synapse_config=require_synapse_config,
        require_compiled_modfiles=require_compiled_modfiles,
    )
    cell_config, sim_config = load_tune_configs(tune_dir, cell_name=resolved_cell)

    context = TuningNotebookContext(
        repo_root=root,
        tune_dir=tune_dir,
        cell_name=resolved_cell,
        tune_name=resolved_tune,
        cell_config=cell_config,
        sim_config=sim_config,
        validation=validation,
    )
    if print_summary:
        print_tuning_context_summary(context)
    return context


def print_tuning_context_summary(context: TuningNotebookContext) -> None:
    """Print a concise notebook setup summary."""
    print("SCP repo:", context.repo_root)
    print("Tune dir:", context.tune_dir)
    print("Cell:", context.cell_name, "| Tune:", context.tune_name)
    if context.project_name is not None:
        print('Project:', context.project_name)
    print("Cell loader:", context.cell_config.get("cell_loader"))
    multiplier = (context.cell_config.get("tuning", {}) or {}).get(
        "soma_diam_multiplier"
    )
    if multiplier is not None:
        print("Soma diameter multiplier (Allen adapter):", multiplier)
    print("Step 1 validation:", "ok" if context.validation.ok else "missing files")


def build_tuning_cell(context: TuningNotebookContext | Dict[str, Any], tune_dir: Optional[Path] = None, *, cell_override=None):
    """Build a NEURON cell for Step 2/3 tuning notebooks."""
    if isinstance(context, TuningNotebookContext):
        cell_config = dict(context.cell_config)
        base_dir = context.tune_dir
    else:
        cell_config = dict(context)
        if tune_dir is None:
            raise ValueError("tune_dir is required when context is a cell_config dict.")
        base_dir = Path(tune_dir).expanduser().resolve()

    from scp_pipeline.notebooks.helpers import build_cell_for_notebook

    if cell_override is not None:
        from scp_pipeline.model.runtime_cell import adopt_cell
        cell = adopt_cell(cell_override, cell_config)
    else:
        cell = build_cell_for_notebook(cell_config, base_dir=base_dir)
    if isinstance(context, TuningNotebookContext):
        from scp_pipeline.simulation.current_injection import apply_sim_conditions

        apply_sim_conditions(cell, context.sim_config)
    return cell


def section_summary(cell: Any) -> Dict[str, Any]:
    """Return simple section counts/area metadata for a built cell."""
    summary: Dict[str, Any] = {}
    for attr in ("soma", "dend", "apic", "axon"):
        value = getattr(cell, attr, None)
        if value is None:
            summary[attr] = 0
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            try:
                summary[attr] = len(list(value))
            except TypeError:
                summary[attr] = 1
        else:
            summary[attr] = 1

    h_obj = getattr(cell, "h", None)
    all_sections = list(getattr(cell, "all", ()) or ())
    if h_obj is not None and all_sections:
        try:
            summary["all_sections"] = len(all_sections)
            summary["total_area_um2"] = float(
                sum(h_obj.area(seg.x, sec=sec) for sec in all_sections for seg in sec)
            )
        except Exception:
            pass
    return summary
