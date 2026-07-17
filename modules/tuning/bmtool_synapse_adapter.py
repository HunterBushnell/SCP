"""SCP adapter helpers for BMTool synapse tuning notebooks.

The BMTool chemical synapse tuner expects either BMTool template/modfile paths
or a pre-built NEURON cell. SCP tune directories already know how to build the
cell, so this module keeps that SCP-specific setup out of Step 4 while leaving
the BMTool tuning workflow mostly unchanged.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from modules.model.geometry import cell_sections
from modules.notebooks.helpers import (
    build_synapse_test_cell,
    ensure_external_repo_on_syspath,
    ensure_scp_repo_on_syspath,
    resolve_cell_config_for_notebook,
)
from modules.setup.mechanisms import compile_modfiles, resolve_modfiles_dir


BMTOOL_REPO_URL = "https://github.com/cyneuro/bmtool.git"
BMTOOL_ENV_VARS = ("SCP_BMTOOL_PATH", "BMTOOL_PATH", "BMTOOL_ROOT")


@dataclass
class SynapseTuningSession:
    """Prepared SCP model state for BMTool synapse tuning."""

    repo_root: Path
    cell_name: str
    tune_name: str
    tune_dir: Path
    cell_config: Dict[str, Any]
    cell: Any
    mechanism_summary: Dict[str, Any]
    bmtool_path: Optional[Path] = None


@dataclass
class _BMToolCellFacade:
    """Expose BMTool's required fields without leaking global hoc sections."""

    source: Any
    soma: list[Any]
    all: list[Any]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)


def _bmtool_cell_facade(cell: Any) -> _BMToolCellFacade:
    soma = cell_sections(cell, "soma")
    all_sections = cell_sections(cell, "all")
    if not soma:
        raise ValueError("BMTool synapse tuning requires at least one canonical soma section.")
    if not all_sections:
        raise ValueError("BMTool synapse tuning requires a non-empty canonical all section group.")
    return _BMToolCellFacade(source=cell, soma=soma, all=all_sections)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip() not in {"0", "false", "False", "no", "No", "off", "Off"}


def _in_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ


def _looks_like_bmtool_repo(path: Path) -> bool:
    return (path / "bmtool" / "synapses.py").is_file()


def ensure_bmtool_on_syspath(
    *,
    repo_root: Optional[Path] = None,
    auto_clone: Optional[bool] = None,
    repo_url: str = BMTOOL_REPO_URL,
    target_dir: Optional[Path] = None,
    prepend: bool = True,
) -> Path:
    """Resolve BMTool and add it to ``sys.path``.

    Resolution follows the same local conventions as the rest of SCP:
    environment variables first, then common ``../mods/bmtool`` locations.
    In Colab, BMTool is cloned automatically unless ``SCP_AUTO_CLONE_BMTOOL=0``.
    """

    root = ensure_scp_repo_on_syspath(repo_root)
    if auto_clone is None:
        auto_clone = _env_flag("SCP_AUTO_CLONE_BMTOOL", default=_in_colab())

    try:
        return ensure_external_repo_on_syspath(
            repo_name="bmtool",
            marker_rel=Path("bmtool") / "synapses.py",
            env_vars=BMTOOL_ENV_VARS,
            repo_root=root,
            prepend=prepend,
        )
    except FileNotFoundError:
        if not auto_clone:
            raise

    destination = Path(target_dir) if target_dir else root.parent / "mods" / "bmtool"
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not _looks_like_bmtool_repo(destination):
        clone_cmd = ["git", "clone", "--depth", "1", repo_url, str(destination)]
        subprocess.check_call(clone_cmd)

    if not _looks_like_bmtool_repo(destination):
        raise FileNotFoundError(f"BMTool clone did not contain bmtool/synapses.py: {destination}")

    path_str = str(destination)
    if path_str not in sys.path:
        if prepend:
            sys.path.insert(0, path_str)
        else:
            sys.path.append(path_str)
    return destination


def import_bmtool_synapse_api(*, repo_root: Optional[Path] = None):
    """Import and return BMTool's ``SynapseTuner`` and ``SynapseOptimizer``."""

    ensure_bmtool_on_syspath(repo_root=repo_root)
    from bmtool.synapses import SynapseOptimizer, SynapseTuner

    return SynapseTuner, SynapseOptimizer


def resolve_tune_dir(
    *,
    repo_root: Path,
    cell_name: str,
    tune_name: str,
    tunes_parent: str = "tunes",
    tune_dir_override: Optional[str | Path] = None,
) -> Path:
    """Resolve a Step 4 target tune directory."""

    if tune_dir_override:
        tune_dir = Path(tune_dir_override).expanduser()
        if not tune_dir.is_absolute():
            tune_dir = repo_root / tune_dir
    else:
        tune_dir = repo_root / "cells" / cell_name / tunes_parent / tune_name
    tune_dir = tune_dir.resolve()
    if not tune_dir.is_dir():
        raise FileNotFoundError(f"Tune directory not found: {tune_dir}")
    return tune_dir


def prepare_scp_synapse_tuning(
    *,
    cell_name: Optional[str] = None,
    tune_name: str = "tuned",
    tunes_parent: str = "tunes",
    tune_dir_override: Optional[str | Path] = None,
    repo_root: Optional[Path] = None,
    recompile_modfiles: bool = False,
    load_compiled_dll: bool = True,
    resolve_bmtool: bool = False,
) -> SynapseTuningSession:
    """Compile/load mechanisms and build an SCP cell for BMTool tuning.

    BMTool resolution is deferred by default until a configured mechanism and
    canonical section index have been validated by ``create_scp_synapse_tuner``.
    The legacy ``resolve_bmtool`` flag is retained as a warning-only compatibility
    input; it cannot resolve BMTool early without bypassing that preflight.
    """

    root = ensure_scp_repo_on_syspath(repo_root)
    if not cell_name and not tune_dir_override:
        raise ValueError(
            "prepare_scp_synapse_tuning requires cell_name or tune_dir_override; "
            "SCP does not choose a model default for Step 4."
        )
    requested_cell_name = str(cell_name or "")
    tune_dir = resolve_tune_dir(
        repo_root=root,
        cell_name=requested_cell_name,
        tune_name=tune_name,
        tunes_parent=tunes_parent,
        tune_dir_override=tune_dir_override,
    )
    cell_config = resolve_cell_config_for_notebook(requested_cell_name, tune_dir=tune_dir)
    resolved_cell_name = str(
        cell_config.get("cell_name")
        or requested_cell_name
        or tune_dir.parent.parent.name
    )
    mechanism_summary = compile_modfiles(
        tune_dir,
        recompile=recompile_modfiles,
        load_dll=load_compiled_dll,
        allow_missing=True,
        cell_config=cell_config,
    )

    cell_config.setdefault("paths", {})
    cell_config["paths"].setdefault("tune_dir", str(tune_dir))
    cell_config.setdefault("tune_dir", str(tune_dir))

    cell = build_synapse_test_cell(cell_config, base_dir=tune_dir)

    if resolve_bmtool:
        warnings.warn(
            "resolve_bmtool=True is deferred until create_scp_synapse_tuner() "
            "can validate the selected point process and canonical section index.",
            RuntimeWarning,
            stacklevel=2,
        )
    bmtool_path = None

    return SynapseTuningSession(
        repo_root=root,
        cell_name=resolved_cell_name,
        tune_name=tune_name,
        tune_dir=tune_dir,
        cell_config=cell_config,
        cell=cell,
        mechanism_summary=mechanism_summary,
        bmtool_path=bmtool_path,
    )


def default_slider_vars_for_connection(connection_settings: Dict[str, Any]) -> list[str]:
    """Return a conservative default list of BMTool slider variables."""

    syn_params = connection_settings.get("spec_syn_param", {})
    level = str(connection_settings.get("spec_settings", {}).get("level_of_detail", ""))
    if "GABA_A" in level:
        preferred = [
            "initW",
            "Dep",
            "Fac",
            "Use",
            "tau_r_GABAA",
            "tau_d_GABAA",
            "gmax",
        ]
    else:
        preferred = [
            "initW",
            "Dep",
            "Fac",
            "Use",
            "tau_r_AMPA",
            "tau_d_AMPA",
            "NMDA_ratio",
        ]
    return [name for name in preferred if name in syn_params]


def default_record_vars_for_connection(connection_settings: Dict[str, Any]) -> list[str]:
    """Return optional BMTool variables to record for common SCP mechanisms."""

    level = str(connection_settings.get("spec_settings", {}).get("level_of_detail", ""))
    if "AMPA_NMDA_STP" in level:
        return ["record_Pr", "record_use"]
    if "GABA_A" in level:
        return ["g"]
    return []


def create_scp_synapse_tuner(
    session: SynapseTuningSession,
    *,
    conn_type_settings: Dict[str, Any],
    connection: str,
    general_settings: Optional[Dict[str, Any]] = None,
    current_name: str = "i",
    other_vars_to_record: Optional[Sequence[str]] = None,
    slider_vars: Optional[Sequence[str]] = None,
):
    """Create BMTool's ``SynapseTuner`` for a prepared SCP cell."""

    if connection not in conn_type_settings:
        raise KeyError(
            f"connection={connection!r} not found. Available: {sorted(conn_type_settings)}"
        )

    conn_settings = conn_type_settings[connection]
    hoc_cell = _bmtool_cell_facade(session.cell)
    _validate_bmtool_selection(
        session=session,
        connection=connection,
        connection_settings=conn_settings,
        hoc_cell=hoc_cell,
    )

    if other_vars_to_record is None:
        other_vars_to_record = default_record_vars_for_connection(conn_settings)
    if slider_vars is None:
        slider_vars = default_slider_vars_for_connection(conn_settings)

    SynapseTuner, _ = import_bmtool_synapse_api(repo_root=session.repo_root)
    modfiles_dir = resolve_modfiles_dir(session.tune_dir, session.cell_config)
    mechanisms_dir = modfiles_dir if modfiles_dir is not None else session.tune_dir

    return SynapseTuner(
        mechanisms_dir=str(mechanisms_dir),
        conn_type_settings=conn_type_settings,
        general_settings=general_settings,
        connection=connection,
        current_name=current_name,
        other_vars_to_record=list(other_vars_to_record),
        slider_vars=list(slider_vars),
        hoc_cell=hoc_cell,
    )


def _validate_bmtool_selection(
    *,
    session: SynapseTuningSession,
    connection: str,
    connection_settings: Dict[str, Any],
    hoc_cell: _BMToolCellFacade,
) -> None:
    """Validate one configured selection before importing or cloning BMTool."""
    spec_settings = connection_settings.get("spec_settings")
    if not isinstance(spec_settings, dict):
        raise ValueError(
            f"Connection {connection!r} must define a spec_settings object in "
            "cell_configs/synapse_tuning_config.json."
        )

    raw_sec_id = spec_settings.get("sec_id", 0)
    if isinstance(raw_sec_id, bool):
        raise ValueError(
            f"Connection {connection!r} sec_id must be an integer; got {raw_sec_id!r}."
        )
    try:
        sec_id = int(raw_sec_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Connection {connection!r} sec_id must be an integer; got {raw_sec_id!r}."
        ) from exc
    if isinstance(raw_sec_id, float) and raw_sec_id != sec_id:
        raise ValueError(
            f"Connection {connection!r} sec_id must be an integer; got {raw_sec_id!r}."
        )
    if sec_id < 0 or sec_id >= len(hoc_cell.all):
        raise IndexError(
            "BMTool sec_id out of range for canonical all sections "
            f"(connection={connection!r}, sec_id={sec_id}, n={len(hoc_cell.all)}). "
            "Edit connections.<name>.spec_settings.sec_id in "
            "cell_configs/synapse_tuning_config.json."
        )

    mechanism_name = str(spec_settings.get("level_of_detail", "")).strip()
    if not mechanism_name:
        raise ValueError(
            f"Connection {connection!r} has no level_of_detail mechanism name. "
            "Set connections.<name>.spec_settings.level_of_detail in "
            "cell_configs/synapse_tuning_config.json."
        )

    hoc = getattr(session.cell, "h", None)
    if hoc is None:
        raise ValueError(
            "The loaded SCP cell does not expose its NEURON runtime as cell.h; "
            "reload it through the common SCP loader before starting Step 4."
        )
    try:
        mechanism_constructor = getattr(hoc, mechanism_name)
    except (AttributeError, TypeError):
        raise RuntimeError(
            f"NEURON point-process mechanism {mechanism_name!r} configured for "
            f"connection {connection!r} is unavailable. Edit level_of_detail to "
            "match an installed point process, or add/compile the corresponding "
            "MOD source and restart the Python/Jupyter process before retrying."
        ) from None

    raw_sec_x = spec_settings.get("sec_x", 0.5)
    if isinstance(raw_sec_x, bool) or not isinstance(raw_sec_x, (int, float)):
        raise ValueError(
            f"Connection {connection!r} sec_x must be numeric in [0, 1]; "
            f"got {raw_sec_x!r}."
        )
    sec_x = float(raw_sec_x)
    if sec_x < 0.0 or sec_x > 1.0:
        raise ValueError(
            f"Connection {connection!r} sec_x must be in [0, 1]; got {sec_x}."
        )
    if not callable(mechanism_constructor):
        raise RuntimeError(
            f"NEURON symbol {mechanism_name!r} configured for connection "
            f"{connection!r} is available but is not a point-process constructor. "
            "Edit level_of_detail to name the intended synapse point process."
        )
    try:
        probe = mechanism_constructor(hoc_cell.all[sec_id](sec_x))
    except Exception as exc:
        raise RuntimeError(
            f"NEURON could not construct point process {mechanism_name!r} for "
            f"connection {connection!r} at sec_id={sec_id}, sec_x={sec_x}. "
            "Check level_of_detail and the compiled MOD source, then restart "
            "the Python/Jupyter process before retrying."
        ) from exc
    del probe


def get_tuned_synapse_params(
    tuner: Any,
    *,
    param_names: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Read selected parameter values from the active BMTool tuner synapse."""

    if param_names is None:
        names = []
        for source in (getattr(tuner, "slider_vars", {}), getattr(tuner, "synaptic_props", {})):
            for name in source:
                if name not in names:
                    names.append(name)
    else:
        names = list(param_names)

    params: Dict[str, float] = {}
    for name in names:
        if hasattr(tuner.syn, name):
            value = getattr(tuner.syn, name)
            if isinstance(value, (int, float)):
                params[name] = float(value)
    return params


def print_syn_group_param_block(params: Dict[str, Any]) -> None:
    """Print a copyable ``syns.params`` block for SCP synapse group configs."""

    print('"params": ' + json.dumps(params, indent=2, sort_keys=True))
