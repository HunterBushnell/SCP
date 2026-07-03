"""NEURON mechanism compilation/loading helpers for Step 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import shutil
import subprocess

from .fit_json import mechanisms_declared_in_fit_json


def find_compiled_mechanism_dll(tune_dir: Path) -> Optional[Path]:
    tune_dir = Path(tune_dir)
    candidates = [
        tune_dir / "modfiles" / "x86_64" / ".libs" / "libnrnmech.so",
        tune_dir / "modfiles" / "x86_64" / "libnrnmech.so",
    ]
    for dll in candidates:
        if dll.is_file():
            return dll
    return None


def compile_modfiles(
    tune_dir: Path,
    *,
    recompile: bool = False,
    load_dll: bool = True,
) -> Dict[str, Any]:
    """Compile modfiles (nrnivmodl) and optionally load the produced DLL."""
    tune_dir = Path(tune_dir).expanduser().resolve()
    mod_dir = tune_dir / "modfiles"
    if not mod_dir.is_dir():
        raise FileNotFoundError(f"Missing modfiles directory: {mod_dir}")

    compiled_dir = mod_dir / "x86_64"
    if recompile and compiled_dir.exists():
        shutil.rmtree(compiled_dir)

    dll = find_compiled_mechanism_dll(tune_dir)
    compiled_now = False
    if dll is None:
        subprocess.check_call(["nrnivmodl"], cwd=str(mod_dir))
        compiled_now = True
        dll = find_compiled_mechanism_dll(tune_dir)

    if dll is None:
        raise FileNotFoundError(
            "nrnivmodl finished but compiled mechanism library was not found under "
            f"{compiled_dir}"
        )

    loaded = False
    dll_preloaded = False
    if load_dll:
        from neuron import h

        h.load_file("stdrun.hoc")
        try:
            h.nrn_load_dll(str(dll))
            loaded = True
        except RuntimeError as exc:
            # Common when rerunning Step-1 in a live kernel/session. NEURON can
            # emit only a generic hocobj_call RuntimeError; verify required
            # mechanisms are already present before deciding to continue.
            required_mechs = mechanisms_declared_in_fit_json(tune_dir)
            missing = sorted(m for m in required_mechs if not hasattr(h, m))
            if not missing:
                loaded = True
                dll_preloaded = True
            else:
                raise RuntimeError(
                    "Failed to load compiled NEURON mechanisms from "
                    f"{dll}. Missing mechanisms after load attempt: {missing}. "
                    "If another mechanism library is already loaded in this "
                    "session, restart the kernel/process or rerun with "
                    "load_compiled_dll=False."
                ) from exc

    return {
        "modfiles_dir": str(mod_dir),
        "compiled_dir": str(compiled_dir),
        "dll": str(dll),
        "compiled_now": bool(compiled_now),
        "loaded": bool(loaded),
        "dll_preloaded": bool(dll_preloaded),
    }
