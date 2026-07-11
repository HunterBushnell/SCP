"""Shared setup helpers for Step 2/3 tuning notebooks."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterable
from dataclasses import dataclass
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

    @property
    def ok(self) -> bool:
        return all(self.checks.values())

    @property
    def missing(self) -> Dict[str, Path]:
        return {key: self.paths[key] for key, ok in self.checks.items() if not ok}

    def raise_for_missing(self) -> None:
        if self.ok:
            return
        lines = [f"{name}: {path}" for name, path in self.missing.items()]
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
        return {
            "repo_root": str(self.repo_root),
            "tune_dir": str(self.tune_dir),
            "cell_name": self.cell_name,
            "tune_name": self.tune_name,
            "cell_loader": self.cell_config.get("cell_loader"),
            "soma_diam_multiplier": (
                self.cell_config.get("tuning", {}) or {}
            ).get("soma_diam_multiplier"),
            "validation_ok": self.validation.ok,
        }


def _looks_like_repo_root(path: Path) -> bool:
    return (path / "modules").is_dir() and (path / "run_pipeline.py").is_file()


def resolve_repo_root(start: Optional[Path] = None) -> Path:
    """Locate the SCP repository root from a notebook or script context."""
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
    """Resolve SCP repo root and prepend it to `sys.path` if needed."""
    root = resolve_repo_root(repo_root)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    os.environ["SCP_ROOT"] = root_str
    return root


def resolve_tune_dir(
    *,
    repo_root: Optional[Path] = None,
    cell_name: Optional[str] = None,
    tune_name: Optional[str] = None,
    tunes_parent: str = "tunes",
    tune_dir_override: Optional[str | Path] = None,
) -> Path:
    """Resolve a tune directory from explicit path or cell/tune labels."""
    if tune_dir_override:
        path = Path(tune_dir_override).expanduser()
        if not path.is_absolute():
            path = (resolve_repo_root(repo_root) / path).resolve()
        return path.resolve()

    if not cell_name or not tune_name:
        raise ValueError("Provide tune_dir_override or both cell_name and tune_name.")
    root = resolve_repo_root(repo_root)
    return (root / "cells" / str(cell_name) / tunes_parent / str(tune_name)).resolve()


def _read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


def _compiled_mechanism_exists(tune_dir: Path) -> bool:
    mod_dir = tune_dir / "modfiles"
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
        "manifest": tune_path / "manifest.json",
        "modfiles": tune_path / "modfiles",
        "compiled_modfiles": tune_path / "modfiles",
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
        "manifest": paths["manifest"].is_file(),
        "modfiles": paths["modfiles"].is_dir(),
        "compiled_modfiles": (not require_compiled_modfiles) or _compiled_mechanism_exists(tune_path),
        "cell_config": paths["cell_config"].is_file(),
    }
    if require_sim_config:
        checks["sim_config"] = paths["sim_config"].is_file()
    if require_geometry_config:
        checks["geometry"] = paths["geometry"].is_file()
    if require_synapse_config:
        checks["syn_config"] = paths["syn_config"].is_file()

    report = Step1ValidationReport(tune_dir=tune_path, checks=checks, paths=paths)
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
    paths = cell_config.setdefault("paths", {})
    if not isinstance(paths, dict):
        paths = {}
        cell_config["paths"] = paths
    paths.setdefault("manifest", "manifest.json")

    tuning = cell_config.get("tuning")
    if not isinstance(tuning, dict) or "soma_diam_multiplier" not in tuning:
        raise KeyError(
            "cell_configs/cell_config.json must define tuning.soma_diam_multiplier."
        )
    tuning["soma_diam_multiplier"] = float(tuning["soma_diam_multiplier"])

    sim_path = tune_path / "cell_configs" / "sim_config.json"
    sim_config = _read_json(sim_path) if sim_path.is_file() else {}
    if require_sim_config and not sim_config:
        raise FileNotFoundError(f"Missing sim_config.json at {sim_path}")
    return cell_config, sim_config


def _infer_cell_tune_from_path(tune_dir: Path) -> tuple[str, str]:
    parts = tune_dir.resolve().parts
    cell_name = tune_dir.parent.parent.name if tune_dir.parent.name == "tunes" else "cell"
    tune_name = tune_dir.name
    if "cells" in parts and "tunes" in parts:
        i_cells = parts.index("cells")
        i_tunes = parts.index("tunes")
        if i_cells + 1 < len(parts):
            cell_name = parts[i_cells + 1]
        if i_tunes + 1 < len(parts):
            tune_name = parts[i_tunes + 1]
    return cell_name, tune_name


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
) -> TuningNotebookContext:
    """Resolve, validate, and load common Step 2/3 notebook state."""
    root = ensure_repo_on_syspath(repo_root)
    tune_dir = resolve_tune_dir(
        repo_root=root,
        cell_name=cell_name,
        tune_name=tune_name,
        tunes_parent=tunes_parent,
        tune_dir_override=tune_dir_override,
    )
    inferred_cell, inferred_tune = _infer_cell_tune_from_path(tune_dir)
    resolved_cell = str(cell_name or inferred_cell)
    resolved_tune = str(tune_name or inferred_tune)

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
    print("Cell loader:", context.cell_config.get("cell_loader"))
    print(
        "Soma diameter multiplier:",
        (context.cell_config.get("tuning", {}) or {}).get("soma_diam_multiplier"),
    )
    print("Step 1 validation:", "ok" if context.validation.ok else "missing files")


@contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(old))


def build_tuning_cell(context: TuningNotebookContext | Dict[str, Any], tune_dir: Optional[Path] = None):
    """Build a NEURON cell for Step 2/3 tuning notebooks."""
    if isinstance(context, TuningNotebookContext):
        cell_config = dict(context.cell_config)
        base_dir = context.tune_dir
    else:
        cell_config = dict(context)
        if tune_dir is None:
            raise ValueError("tune_dir is required when context is a cell_config dict.")
        base_dir = Path(tune_dir).expanduser().resolve()

    from modules.notebooks.helpers import build_cell_for_notebook

    with _pushd(base_dir):
        return build_cell_for_notebook(cell_config)


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
    if h_obj is not None and hasattr(h_obj, "allsec"):
        try:
            all_sections = list(h_obj.allsec())
            summary["all_sections"] = len(all_sections)
            summary["total_area_um2"] = float(
                sum(h_obj.area(seg.x, sec=sec) for sec in all_sections for seg in sec)
            )
        except Exception:
            pass
    return summary
