from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from modules.setup.mechanisms import (
    find_compiled_mechanism_dll,
    load_compiled_mechanism_library,
    resolve_modfiles_dir,
)

def _timestamp_stem() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def normalize_tune_dir(
    user_tune_dir: Union[str, Path],
    *,
    base_dir: Optional[Path] = None,
) -> tuple[Path, Optional[str]]:
    """
    Resolve a tune directory and auto-fix the common `tune` -> `tunes` typo.
    """
    raw = Path(user_tune_dir)
    base = Path.cwd() if base_dir is None else Path(base_dir)
    resolved = (raw if raw.is_absolute() else (base / raw)).resolve()
    if resolved.is_dir():
        return resolved, None

    parts = list(resolved.parts)
    if "tune" in parts:
        idx = parts.index("tune")
        alt = Path(*(parts[:idx] + ["tunes"] + parts[idx + 1 :]))
        if alt.is_dir():
            return alt.resolve(), f"Normalized tune directory from '{resolved}' to '{alt.resolve()}'"

    return resolved, None


def infer_cell_name(tune_dir: Path, cell_config: Optional[Dict[str, Any]] = None) -> str:
    """
    Infer the cell label from config first, then from cells/<CELL>/tunes/<TUNE>.
    """
    if cell_config and cell_config.get("cell_name"):
        return str(cell_config["cell_name"])
    if tune_dir.parent.name == "tunes" and tune_dir.parent.parent.name:
        return tune_dir.parent.parent.name
    return tune_dir.parent.name


def load_mechanisms(
    tune_dir: Path,
    cell_config: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Load compiled NEURON mechanisms from the tune's configured source directory.

    Mechanism identity is based on the compiled library path, not on a list of
    model-specific mechanism names. A tune without configured MOD sources is
    allowed to use NEURON's built-in mechanisms.
    """
    tune_dir = Path(tune_dir).expanduser().resolve()
    mod_dir = resolve_modfiles_dir(tune_dir, cell_config)
    if mod_dir is None:
        print("No configured modfiles directory; using NEURON built-in mechanisms.")
        return None
    if not mod_dir.is_dir():
        print(f"No configured MOD source directory; using available NEURON mechanisms: {mod_dir}")
        return None
    if not any(mod_dir.glob("*.mod")):
        print(f"No configured .mod sources; using available NEURON mechanisms: {mod_dir}")
        return None

    dll = find_compiled_mechanism_dll(tune_dir, cell_config=cell_config)
    if dll is not None:
        summary = load_compiled_mechanism_library(dll)
        action = (
            "Mechanisms already loaded from"
            if summary.get("dll_preloaded")
            else "Loaded mechanisms from"
        )
        print(f"{action} {dll}")
        return Path(summary["dll"])

    raise FileNotFoundError(
        "Compiled mechanisms not found. Run `nrnivmodl` inside the configured "
        f"MOD source directory {mod_dir} to build {mod_dir / 'x86_64'}, then rerun."
    )


def _load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _parse_enabled_path(raw: object) -> tuple[bool, Optional[str]]:
    if isinstance(raw, (list, tuple)):
        enabled = bool(raw[0]) if len(raw) >= 1 else False
        path = raw[1] if len(raw) >= 2 else None
        return enabled, path
    if isinstance(raw, dict):
        return bool(raw.get("enabled", False)), raw.get("path")
    if raw in (None, "", False):
        return False, None
    return True, str(raw)


def _resolve_append_target(sim_cfg_raw: Dict[str, Any], output_base: Path) -> Optional[Path]:
    append_raw = sim_cfg_raw.get("append") if "append" in sim_cfg_raw else sim_cfg_raw.get("append_to")
    enabled, path = _parse_enabled_path(append_raw)
    if not enabled or path in (None, "", False):
        return None
    append_path = Path(str(path))
    if not append_path.is_absolute():
        if append_path.parts and append_path.parts[0] == output_base.name:
            append_path = output_base / Path(*append_path.parts[1:])
        else:
            append_path = output_base / append_path
    return append_path
