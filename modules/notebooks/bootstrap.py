"""Bootstrap helpers for notebook entry points.

This module keeps environment setup, dependency checks, and modfile handling
out of user-facing notebooks. It intentionally uses only Python standard
library imports at module import time so fresh Colab sessions can import it
immediately after cloning SCP.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_STEP5_EXTERNAL_INPUTS = (
    "external_data/pyrFiringRateAvg.csv",
    "external_data/PV_1000tr.pkl",
    "external_data/SST_1000tr.pkl",
)


def is_colab() -> bool:
    """Return True inside a Google Colab runtime."""
    return "COLAB_RELEASE_TAG" in os.environ


def env_flag(name: str, default: bool) -> bool:
    """Parse common truthy/falsy environment-variable values."""
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip() not in {"0", "false", "False", "no", "No", "off", "Off"}


def ensure_python_package(import_name: str, pip_name: Optional[str] = None) -> None:
    """Install a Python package only when its import is currently unavailable."""
    try:
        importlib.import_module(import_name)
        return
    except Exception:
        pass

    package = pip_name or import_name
    print(f"Installing missing package: {package}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def ensure_notebook_dependencies(
    *,
    install_deps: Optional[bool] = None,
    include_system_deps: Optional[bool] = None,
) -> None:
    """Install dependencies needed by Step 5 notebooks when requested.

    By default this installs packages only in Colab. Local users should install
    the project environment outside the notebook.
    """
    in_colab = is_colab()
    if install_deps is None:
        install_deps = env_flag("SCP_INSTALL_DEPS", default=in_colab)
    if include_system_deps is None:
        include_system_deps = in_colab

    if not install_deps:
        return

    for import_name in ("numpy", "pandas", "matplotlib", "scipy", "h5py", "neuron", "allensdk"):
        ensure_python_package(import_name)

    if include_system_deps:
        subprocess.check_call(["apt-get", "update"])
        subprocess.check_call(["apt-get", "install", "-y", "build-essential"])


def check_required_external_inputs(
    repo_root: Path,
    required_inputs: Iterable[str | Path] = DEFAULT_STEP5_EXTERNAL_INPUTS,
    *,
    warn: bool = True,
) -> list[Path]:
    """Return required external input files that are missing."""
    root = Path(repo_root).expanduser().resolve()
    missing: list[Path] = []
    for rel in required_inputs:
        path = Path(rel)
        candidate = path if path.is_absolute() else root / path
        if not candidate.is_file():
            missing.append(candidate)

    if warn and missing:
        print("Missing external inputs:")
        for path in missing:
            print(" -", path)
    return missing


def ensure_modfiles(tune_dir: Path, *, compile_modfiles: bool = True) -> None:
    """Ensure compiled NEURON mechanisms exist for a tune directory."""
    tune_path = Path(tune_dir).expanduser().resolve()
    mod_dir = tune_path / "modfiles"
    dll_candidates = (
        mod_dir / "x86_64" / ".libs" / "libnrnmech.so",
        mod_dir / "x86_64" / "libnrnmech.so",
    )
    if any(path.is_file() for path in dll_candidates):
        return
    if not mod_dir.is_dir():
        print(f"Missing modfiles dir: {mod_dir}")
        return
    if compile_modfiles:
        print("Compiling modfiles with nrnivmodl...")
        subprocess.check_call(["nrnivmodl"], cwd=str(mod_dir))
    else:
        print("Missing compiled modfiles; run nrnivmodl in", mod_dir)


def finish_step5_notebook_setup(
    repo_root: Path,
    *,
    install_deps: Optional[bool] = None,
    check_external_inputs: bool = True,
    print_status: bool = True,
) -> Dict[str, Any]:
    """Finish Step 5 notebook setup after the SCP repo is available."""
    from modules.notebooks.helpers import ensure_scp_repo_on_syspath

    root = ensure_scp_repo_on_syspath(Path(repo_root))
    os.environ["SCP_ROOT"] = str(root)

    ensure_notebook_dependencies(install_deps=install_deps)
    missing_external = (
        check_required_external_inputs(root, warn=True) if check_external_inputs else []
    )

    if print_status:
        print("Runtime:", "Colab" if is_colab() else "local")
        print("SCP repo:", root)

    return {
        "repo_root": root,
        "in_colab": is_colab(),
        "missing_external": missing_external,
    }
