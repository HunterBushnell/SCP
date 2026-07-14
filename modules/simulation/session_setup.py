from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from modules.loaders import get_cell_loader_name, loader_requires_manifest


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


def load_mechanisms(tune_dir: Path) -> None:
    """
    Load compiled NEURON mechanisms from a tune directory.
    """
    from neuron import h

    mech_names = ("AMPA_NMDA_STP", "GABA_A", "GABA_A_STP", "vecstim", "Ih")
    if any(hasattr(h, mech) for mech in mech_names):
        print("Mechanisms already loaded; skipping duplicate load.")
        return

    candidates = [
        tune_dir / "modfiles" / "x86_64" / ".libs" / "libnrnmech.so",
        tune_dir / "modfiles" / "x86_64" / "libnrnmech.so",
    ]
    for dll in candidates:
        if dll.is_file():
            try:
                h.nrn_load_dll(str(dll))
                print(f"Loaded mechanisms from {dll}")
                return
            except Exception as exc:
                already_loaded = any(hasattr(h, mech) for mech in mech_names)
                if already_loaded:
                    print(f"Mechanisms already loaded (skipping reload of {dll})")
                    return
                raise RuntimeError(
                    f"Found compiled mechanisms at {dll} but failed to load: {exc}"
                ) from exc

    raise FileNotFoundError(
        "Compiled mechanisms not found. Run `nrnivmodl` inside the tune directory "
        f"{tune_dir}/modfiles to build modfiles/x86_64/.libs/libnrnmech.so, then rerun."
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


def resolve_loader_manifest_path(
    *,
    cell_config: Dict[str, Any],
    cell_config_path: Path,
    tune_dir: Path,
) -> None:
    paths = cell_config.setdefault("paths", {})
    loader_name = get_cell_loader_name(cell_config)
    if not loader_requires_manifest(loader_name):
        return

    raw_manifest = Path(str(paths.get("manifest", "manifest.json")))
    if raw_manifest.is_absolute():
        resolved_manifest = raw_manifest
    else:
        candidates = [
            Path.cwd() / raw_manifest,
            cell_config_path.parent / raw_manifest,
            tune_dir / raw_manifest,
        ]
        resolved_manifest = next((p for p in candidates if p.is_file()), candidates[1])

    if not resolved_manifest.is_file():
        raise FileNotFoundError(
            f"Missing manifest.json for cell_loader={loader_name!r}: {resolved_manifest}"
        )
    paths["manifest"] = str(resolved_manifest)
