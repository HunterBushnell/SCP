"""Bootstrap helpers for notebook entry points.

This module keeps environment setup, dependency checks, and modfile handling
out of user-facing notebooks. It intentionally uses only Python standard
library imports at module import time so fresh Colab sessions can import it
immediately after cloning SCP.
"""

from __future__ import annotations

import importlib
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_STEP5_EXTERNAL_INPUTS = (
    "external_data/pyrFiringRateAvg.csv",
    "external_data/PVFiringRateAvg.csv",
    "external_data/SSTFiringRateAvg.csv",
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


def _run_pip_install(*packages: str, extra_args: Iterable[str] = ()) -> None:
    """Run pip install with the current notebook Python executable."""
    cmd = [sys.executable, "-m", "pip", "install", *extra_args, *packages]
    subprocess.check_call(cmd)


def _allen_imports_work() -> bool:
    """Return True when the AllenSDK pieces used by SCP can import."""
    required_modules = (
        "allensdk.api.queries.biophysical_api",
        "allensdk.model.biophys_sim.config",
        "allensdk.model.biophysical.utils",
    )
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception:
            return False
    return True


def _allen_imports_work_in_fresh_python() -> bool:
    """Check AllenSDK imports in a fresh Python process after pip changes."""
    code = (
        "import importlib;"
        "mods=("
        "'allensdk.api.queries.biophysical_api',"
        "'allensdk.model.biophys_sim.config',"
        "'allensdk.model.biophysical.utils'"
        ");"
        "[importlib.import_module(m) for m in mods]"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _restart_colab_runtime_after_install() -> None:
    """Restart Colab after dependency installation if configured to do so."""
    if not env_flag("SCP_AUTO_RESTART_COLAB", default=True):
        return
    print(
        "SCP dependencies were installed successfully. Restarting the Colab "
        "runtime now; rerun this first notebook cell after the runtime reconnects."
    )
    os.kill(os.getpid(), signal.SIGKILL)


def _ensure_colab_py312_dependencies() -> None:
    """
    Install SCP dependencies in Colab's Python 3.12 runtime.

    AllenSDK 2.16.2 declares older NumPy/SciPy/Pandas constraints that cannot be
    installed normally on Python 3.12. SCP only needs a narrow AllenSDK subset
    for Allen biophysical model download/loading, so Colab installs compatible
    modern dependencies first and then installs AllenSDK without resolving its
    stale dependency pins.
    """
    if _allen_imports_work():
        return

    print("Installing SCP dependencies for Colab Python 3.12...")
    _run_pip_install(
        "setuptools<81",
        "numpy==1.26.4",
        "scipy==1.12.0",
        "pandas==2.2.3",
        "matplotlib",
        "h5py",
        "neuron==8.2.4",
        "find-libpython",
        "requests",
        "requests-toolbelt",
        "simplejson",
        "six",
        "jinja2",
        "future",
        "cachetools",
        "python-dateutil",
        "psycopg2-binary",
        "SimpleITK",
        "xarray==2024.3.0",
        "pynwb",
        "hdmf",
        "pynrrd",
        "argschema",
        "semver",
        "nest-asyncio",
        "tqdm",
        "aiohttp",
        "tables",
        "seaborn",
        "statsmodels",
        "scikit-image",
        "sqlalchemy",
        "boto3",
        "ndx-events",
        "glymur",
        "scikit-build",
        extra_args=("-q", "--upgrade"),
    )
    _run_pip_install("allensdk==2.16.2", extra_args=("-q", "--no-deps"))

    importlib.invalidate_caches()
    if not _allen_imports_work():
        if _allen_imports_work_in_fresh_python():
            _restart_colab_runtime_after_install()
            raise RuntimeError(
                "SCP dependencies installed successfully, but the current Colab "
                "runtime still has pre-install packages loaded. Restart the "
                "runtime, then rerun the first notebook cell."
            )
        raise RuntimeError(
            "AllenSDK installed, but SCP could not import the required AllenSDK "
            "model-loading modules. Use Runtime > Restart runtime and rerun the "
            "first notebook cell. If this persists, use the local scp-py311 "
            "environment."
        )


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

    if in_colab and sys.version_info >= (3, 12):
        _ensure_colab_py312_dependencies()
    else:
        for import_name in (
            "numpy",
            "pandas",
            "matplotlib",
            "scipy",
            "h5py",
            "neuron",
            "allensdk",
        ):
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


def ensure_modfiles(
    tune_dir: Path,
    *,
    compile_modfiles: bool = True,
    cell_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Ensure configured NEURON mechanisms exist for a tune directory."""
    from modules.setup.mechanisms import (
        find_compiled_mechanism_dll,
        find_nrnivmodl,
        resolve_modfiles_dir,
    )

    tune_path = Path(tune_dir).expanduser().resolve()
    mod_dir = resolve_modfiles_dir(tune_path, cell_config)
    if mod_dir is None:
        print("No configured modfiles directory; using NEURON built-in mechanisms.")
        return
    if find_compiled_mechanism_dll(tune_path, cell_config=cell_config) is not None:
        return
    if not mod_dir.is_dir():
        print(f"Missing modfiles dir: {mod_dir}")
        return
    if not any(mod_dir.glob("*.mod")):
        print(f"No .mod sources in {mod_dir}; using NEURON built-in mechanisms.")
        return
    if compile_modfiles:
        nrnivmodl = find_nrnivmodl()
        if nrnivmodl is None:
            raise FileNotFoundError(
                "nrnivmodl not found on PATH or next to the active Python "
                f"executable: {sys.executable}"
            )
        print("Compiling modfiles with nrnivmodl...")
        subprocess.check_call([nrnivmodl], cwd=str(mod_dir))
    else:
        print("Missing compiled modfiles; run nrnivmodl in", mod_dir)


def finish_step5_notebook_setup(
    repo_root: Path,
    *,
    install_deps: Optional[bool] = None,
    check_external_inputs: bool = False,
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


def finish_step1_notebook_setup(
    repo_root: Path,
    *,
    install_deps: Optional[bool] = None,
    print_status: bool = True,
) -> Dict[str, Any]:
    """Finish Step 1 notebook setup after the SCP repo is available."""
    from modules.notebooks.helpers import ensure_scp_repo_on_syspath

    root = ensure_scp_repo_on_syspath(Path(repo_root))
    os.environ["SCP_ROOT"] = str(root)

    ensure_notebook_dependencies(install_deps=install_deps)

    if print_status:
        print("Runtime:", "Colab" if is_colab() else "local")
        print("SCP repo:", root)

    return {
        "repo_root": root,
        "in_colab": is_colab(),
    }
