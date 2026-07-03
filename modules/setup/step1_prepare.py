"""
Step 1 preparation helpers.

This module bootstraps a tune directory so it is ready for the refactored
modules pipeline:
- download Allen bundle files (manifest, morphology, fit json, modfiles),
- optionally compile/load NEURON mechanisms,
- scaffold common config files under cell_configs/,
- run lightweight validation checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import copy
import json
import os
import shutil
import subprocess
from contextlib import contextmanager

from modules.input_generation import inputs
from modules.model.load_cell import load_cell
from modules.setup import download_cell


CONFIG_MODE_VALUES = ("fill", "overwrite", "skip")
DEFAULT_GENOME_SECTION_ORDER = ("glob", "soma", "axon", "apic", "dend")


@contextmanager
def _pushd(path: Path):
    """
    Temporarily change the process working directory.

    AllenSDK manifest loading resolves some resources relative to cwd, so Step-1
    validation uses this context for load_cell smoke tests.
    """
    old = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(old))


def guess_specimen_from_cell(cell_name: str) -> Optional[int]:
    """Return a default specimen_id for common SCP cell labels."""
    key = (cell_name or "").strip().upper()
    defaults = {
        "PV": 484635029,
        "SST": 485466109,
    }
    return defaults.get(key)


def guess_soma_multiplier(cell_name: str) -> float:
    """Return a default soma diameter multiplier for common SCP labels."""
    key = (cell_name or "").strip().upper()
    defaults = {
        "PV": 6.0,
        "SST": 8.0,
    }
    return float(defaults.get(key, 1.0))


def guess_cell_color(cell_name: str) -> str:
    """Return a default plot color used in existing SCP cell_config files."""
    key = (cell_name or "").strip().upper()
    defaults = {
        "PV": "b",
        "SST": "m",
    }
    return defaults.get(key, "k")


def _deep_fill(existing: Any, defaults: Any) -> Any:
    """
    Recursively fill missing keys in `existing` from `defaults`.

    Existing values always win. Lists are not merged element-wise.
    """
    if isinstance(existing, dict) and isinstance(defaults, dict):
        merged = dict(existing)
        for key, val in defaults.items():
            if key in merged:
                merged[key] = _deep_fill(merged[key], val)
            else:
                merged[key] = copy.deepcopy(val)
        return merged
    return existing


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, found {type(data)!r}")
    return data


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _write_scaffold_json(path: Path, defaults: Dict[str, Any], mode: str) -> Tuple[str, Dict[str, Any]]:
    mode = str(mode).strip().lower()
    if mode not in CONFIG_MODE_VALUES:
        raise ValueError(f"config_mode must be one of {CONFIG_MODE_VALUES}, got {mode!r}")

    if not path.exists():
        _write_json(path, defaults)
        return "created", dict(defaults)

    if mode == "skip":
        return "unchanged", _read_json(path)

    if mode == "overwrite":
        _write_json(path, defaults)
        return "overwritten", dict(defaults)

    existing = _read_json(path)
    merged = _deep_fill(existing, defaults)
    if merged != existing:
        _write_json(path, merged)
        return "updated", merged
    return "unchanged", existing


def default_cell_config(
    *,
    cell_name: str,
    tune_name: str,
    specimen_id: int,
    model_type: str,
    soma_diam_multiplier: float,
    color: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "cell_name": cell_name,
        "tune": tune_name,
        "color": color if color is not None else guess_cell_color(cell_name),
        "cell_loader": "allen_manifest",
        "paths": {
            "manifest": "manifest.json",
        },
        "tuning": {
            "soma_diam_multiplier": float(soma_diam_multiplier),
        },
    }


def default_sim_config(*, cell_name: str, specimen_id: int, model_type: str) -> Dict[str, Any]:
    is_sst = (cell_name or "").strip().upper().startswith("SST")
    tstop = 1200.0 if is_sst else 1000.0
    stim_start = 500.0 if is_sst else 300.0

    return {
        "tstart": 0.0,
        "tstop": tstop,
        "dt": 0.025,
        "bins": 5.0,
        "jitter": 100.0,
        "stim_start_ms": stim_start,
        "stim_duration_ms": 500.0,
        "n_trials": 1,
        "n_traces_to_save": 1,
        "n_inputs_to_save": 1,
        "load": [False, None],
        "save": [False, "step1_bootstrap", "pkl", False],
        "append": [False, None],
        "save_input_stats": True,
        "input_stats_bin_ms": 5.0,
        "avg_rate_curve_smooth_ms": 25.0,
        "avg_rate_curve_smooth_mode": "center",
        "plots_win_size": 25.0,
        "plots_input_smooth_ms": 25.0,
        "plots_profile": "off",
        "save_plots_mode": "single_plot",
        "save_plots_single_plot_preset": "modules/analysis/analysis_presets/single_plot.json",
        "save_plots_overwrite": False,
        "randomness_mode": "random",
        "random_seed": None,
        "param_study": {
            "input_type": None,
            "param_type": None,
            "param_vals": [],
            "n_trials": None,
        },
        "iclamp": {
            "enabled": False,
            "amp_nA": 0.2,
            "delay_ms": 200.0,
            "dur_ms": 800.0,
            "tstop_ms": None,
            "dt_ms": None,
            "record_currents": False,
        },
        "cell_recording": {
            "enabled": False,
            "n_trials": 1,
            "sites": [
                {"sec": "soma", "idx": 0, "x": 0.5},
            ],
            "vars": {
                "v": True,
                "i_cap": False,
                "ion_currents": False,
                "mech_currents": False,
                "ion_concentrations": False,
                "ion_reversals": False,
                "mech_conductances": False,
                "mech_states": False,
            },
        },
        "syn_recording": {
            "enabled": False,
            "default_mode": "group",
            "default_sample_per_group": 10,
            "default_vars": {
                "i": True,
                "g": True,
                "i_AMPA": False,
                "i_NMDA": False,
                "g_AMPA": False,
                "g_NMDA": False,
                "record_use": False,
                "record_Pr": False,
            },
            "groups": {},
        },
        "snapshot": {
            "enabled": False,
            "n_trials": 1,
            "save_all_inputs": True,
            "save_all_traces": True,
            "save_syn_records_by_trial": True,
        },
        "soma_diam_multiplier": float(guess_soma_multiplier(cell_name)),
    }


def default_geometry_config(*, cell_name: str) -> Dict[str, Any]:
    label = f"{cell_name}_default_geometry"
    return {
        "distance_origin": {"kind": "soma", "x": 0.5},
        "thresholds_um": {
            "proximal": {"low": 20.0, "high": 100.0},
            "distal": {"low": 100.0, "high": None},
        },
        "label": label,
    }


def default_placeholder_syn_group(*, group_name: str = "placeholder_off") -> Dict[str, Any]:
    return {
        group_name: {
            "state": False,
            "mode": "homogeneous_poisson",
            "source": {
                "freq": 0.0,
                "baseline": 0.0,
                "path": None,
                "bin_ms": 5.0,
            },
            "timing": {
                "onset_ms": 0.0,
                "stim_tstart_ms": None,
                "duration_ms": None,
                "input_stim_tstart_ms": None,
                "input_duration_ms": None,
            },
            "syns": {
                "type": "AMPA_NMDA_STP",
                "N_syn": 0,
                "segs": "all",
                "dist_func": {"kind": None, "params": {}},
                "params": {
                    "wt_mean": 1.0,
                    "wt_std": 0.0,
                    "initW": 1.0,
                },
            },
            "color": "#7f7f7f",
        }
    }


def default_syn_config(*, include_path: str = "syn_groups/placeholder_off.json") -> Dict[str, Any]:
    return {
        "__includes__": [include_path],
    }


@dataclass
class Step1Paths:
    tune_dir: Path
    config_dir: Path
    syn_groups_dir: Path
    manifest: Path
    modfiles_dir: Path


def resolve_step1_paths(tune_dir: Path) -> Step1Paths:
    tune_dir = Path(tune_dir).expanduser().resolve()
    return Step1Paths(
        tune_dir=tune_dir,
        config_dir=tune_dir / "cell_configs",
        syn_groups_dir=tune_dir / "cell_configs" / "syn_groups",
        manifest=tune_dir / "manifest.json",
        modfiles_dir=tune_dir / "modfiles",
    )


def find_fit_json(tune_dir: Path) -> Optional[Path]:
    """
    Locate the Allen fit JSON associated with this tune directory.

    Priority:
    1) `manifest.json` -> `biophys[*].model_file` entry ending in `_fit.json`
    2) fallback to first `*_fit.json` file in tune root
    """
    tune_dir = Path(tune_dir).expanduser().resolve()
    manifest_path = tune_dir / "manifest.json"

    if manifest_path.is_file():
        try:
            manifest_data = _read_json(manifest_path)
            biophys = manifest_data.get("biophys", [])
            if isinstance(biophys, list):
                for entry in biophys:
                    if not isinstance(entry, dict):
                        continue
                    model_file = entry.get("model_file")
                    model_file_items = model_file if isinstance(model_file, list) else [model_file]
                    for item in model_file_items:
                        if not isinstance(item, str):
                            continue
                        cand = Path(item)
                        if not cand.name.endswith("_fit.json"):
                            continue
                        cand = (tune_dir / cand).resolve() if not cand.is_absolute() else cand.resolve()
                        if cand.is_file():
                            return cand
        except Exception:
            # Fall back to glob search below.
            pass

    fit_candidates = sorted(tune_dir.glob("*_fit.json"))
    if fit_candidates:
        return fit_candidates[0].resolve()
    return None


def sort_genome_by_section(
    tune_dir: Path,
    *,
    section_order: Tuple[str, ...] = DEFAULT_GENOME_SECTION_ORDER,
) -> Dict[str, Any]:
    """
    Optionally reorder fit JSON `genome` entries by section.

    Sort key groups entries by `section`, preserving relative order within each
    section bucket. This is cosmetic/readability-oriented and does not alter
    parameter values.
    """
    tune_dir = Path(tune_dir).expanduser().resolve()
    fit_json = find_fit_json(tune_dir)
    if fit_json is None:
        return {
            "status": "skipped",
            "reason": "fit_json_not_found",
        }

    fit_data = _read_json(fit_json)
    genome = fit_data.get("genome", [])
    if not isinstance(genome, list):
        return {
            "status": "skipped",
            "reason": "genome_not_list",
            "fit_json": str(fit_json),
        }

    order = tuple(section_order) if section_order else DEFAULT_GENOME_SECTION_ORDER
    order_map = {sec: idx for idx, sec in enumerate(order)}

    indexed = list(enumerate(genome))

    def _sort_key(item):
        i, entry = item
        section = ""
        if isinstance(entry, dict):
            raw = entry.get("section", "")
            if raw is not None:
                section = str(raw)
        return (order_map.get(section, len(order_map)), section, i)

    sorted_genome = [entry for _, entry in sorted(indexed, key=_sort_key)]
    changed = sorted_genome != genome

    if changed:
        fit_data["genome"] = sorted_genome
        _write_json(fit_json, fit_data)

    return {
        "status": "updated" if changed else "unchanged",
        "fit_json": str(fit_json),
        "n_genome_entries": int(len(genome)),
        "section_order": list(order),
    }


def coerce_fit_genome_values_to_numeric(tune_dir: Path) -> Dict[str, Any]:
    """
    Convert numeric-like string values in fit JSON genome entries to floats.

    Allen all-active bundles often serialize `genome[*].value` as strings.
    This helper normalizes those values so downstream loaders expecting numeric
    types can use the fit JSON directly.
    """
    tune_dir = Path(tune_dir).expanduser().resolve()
    fit_json = find_fit_json(tune_dir)
    if fit_json is None:
        return {
            "status": "skipped",
            "reason": "fit_json_not_found",
        }

    fit_data = _read_json(fit_json)
    genome = fit_data.get("genome", [])
    if not isinstance(genome, list):
        return {
            "status": "skipped",
            "reason": "genome_not_list",
            "fit_json": str(fit_json),
        }

    converted = 0
    skipped = 0

    for entry in genome:
        if not isinstance(entry, dict) or "value" not in entry:
            continue

        value = entry["value"]
        if isinstance(value, bool):
            skipped += 1
            continue
        if isinstance(value, (int, float)):
            continue

        try:
            new_value = float(value)
        except (TypeError, ValueError):
            skipped += 1
            continue

        entry["value"] = new_value
        converted += 1

    if converted > 0:
        fit_data["genome"] = genome
        _write_json(fit_json, fit_data)

    return {
        "status": "updated" if converted > 0 else "unchanged",
        "fit_json": str(fit_json),
        "n_genome_entries": int(len(genome)),
        "n_converted": int(converted),
        "n_skipped_non_numeric": int(skipped),
    }


def mechanisms_declared_in_fit_json(tune_dir: Path) -> set[str]:
    """
    Return non-empty mechanism names declared in fit JSON genome entries.
    """
    tune_dir = Path(tune_dir).expanduser().resolve()
    fit_json = find_fit_json(tune_dir)
    if fit_json is None:
        return set()

    try:
        fit_data = _read_json(fit_json)
    except Exception:
        return set()

    genome = fit_data.get("genome", [])
    if not isinstance(genome, list):
        return set()

    mechs: set[str] = set()
    for entry in genome:
        if not isinstance(entry, dict):
            continue
        mech = entry.get("mechanism")
        if isinstance(mech, str):
            mech = mech.strip()
            if mech:
                mechs.add(mech)
    return mechs


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


def scaffold_common_configs(
    *,
    tune_dir: Path,
    cell_name: str,
    tune_name: str,
    specimen_id: int,
    model_type: str,
    soma_diam_multiplier: float,
    color: Optional[str] = None,
    config_mode: str = "fill",
    sync_cell_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Ensure common pipeline config files exist under cell_configs/.

    config_mode:
      - fill: create missing files and fill missing keys in existing files
      - overwrite: replace existing files with defaults
      - skip: do not modify existing files
    """
    paths = resolve_step1_paths(tune_dir)
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.syn_groups_dir.mkdir(parents=True, exist_ok=True)

    statuses: Dict[str, Any] = {}

    cell_cfg_defaults = default_cell_config(
        cell_name=cell_name,
        tune_name=tune_name,
        specimen_id=specimen_id,
        model_type=model_type,
        soma_diam_multiplier=soma_diam_multiplier,
        color=color,
    )
    cell_cfg_path = paths.config_dir / "cell_config.json"
    status, cell_cfg_data = _write_scaffold_json(cell_cfg_path, cell_cfg_defaults, config_mode)

    if sync_cell_metadata:
        before_sync = copy.deepcopy(cell_cfg_data)
        cell_cfg_data.setdefault("paths", {})
        cell_cfg_data["cell_name"] = cell_name
        cell_cfg_data["tune"] = tune_name
        cell_cfg_data.setdefault("cell_loader", "allen_manifest")
        # Canonical model identity is the tune directory itself.
        cell_cfg_data.pop("specimen_id", None)
        cell_cfg_data.pop("model_type", None)
        if color is not None or "color" not in cell_cfg_data:
            cell_cfg_data["color"] = color if color is not None else guess_cell_color(cell_name)
        cell_cfg_data["paths"]["manifest"] = "manifest.json"
        cell_cfg_data.setdefault("tuning", {})
        cell_cfg_data["tuning"]["soma_diam_multiplier"] = float(soma_diam_multiplier)
        if cell_cfg_data != before_sync:
            _write_json(cell_cfg_path, cell_cfg_data)
            if status == "unchanged":
                status = "updated"

    statuses["cell_config"] = {
        "path": str(cell_cfg_path),
        "status": status,
    }

    sim_cfg_defaults = default_sim_config(
        cell_name=cell_name,
        specimen_id=specimen_id,
        model_type=model_type,
    )
    sim_cfg_defaults["soma_diam_multiplier"] = float(soma_diam_multiplier)
    sim_cfg_path = paths.config_dir / "sim_config.json"
    sim_status, sim_cfg_data = _write_scaffold_json(sim_cfg_path, sim_cfg_defaults, config_mode)
    # Keep sim config focused on simulation controls; remove identity metadata.
    if isinstance(sim_cfg_data, dict):
        before_sim = copy.deepcopy(sim_cfg_data)
        sim_cfg_data.pop("specimen_id", None)
        sim_cfg_data.pop("model_type", None)
        if sim_cfg_data != before_sim:
            _write_json(sim_cfg_path, sim_cfg_data)
            if sim_status == "unchanged":
                sim_status = "updated"
    statuses["sim_config"] = {
        "path": str(sim_cfg_path),
        "status": sim_status,
    }

    geom_cfg_defaults = default_geometry_config(cell_name=cell_name)
    geom_cfg_path = paths.config_dir / "geometry.json"
    geom_status, _ = _write_scaffold_json(geom_cfg_path, geom_cfg_defaults, config_mode)
    statuses["geometry"] = {
        "path": str(geom_cfg_path),
        "status": geom_status,
    }

    placeholder_group_path = paths.syn_groups_dir / "placeholder_off.json"
    placeholder_defaults = default_placeholder_syn_group(group_name="placeholder_off")
    group_status, _ = _write_scaffold_json(placeholder_group_path, placeholder_defaults, config_mode)
    statuses["syn_group_placeholder"] = {
        "path": str(placeholder_group_path),
        "status": group_status,
    }

    syn_cfg_defaults = default_syn_config(include_path="syn_groups/placeholder_off.json")
    syn_cfg_path = paths.config_dir / "syn_config.json"
    syn_status, _ = _write_scaffold_json(syn_cfg_path, syn_cfg_defaults, config_mode)
    statuses["syn_config"] = {
        "path": str(syn_cfg_path),
        "status": syn_status,
    }

    return statuses


def validate_tune(
    *,
    tune_dir: Path,
    cell_name: str,
    soma_diam_multiplier: float,
    validate_modfiles: bool = True,
    validate_load_cell: bool = True,
    validate_inputs: bool = True,
) -> Dict[str, Any]:
    """Run lightweight validation checks for Step-1 output layout."""
    tune_dir = Path(tune_dir).expanduser().resolve()
    paths = resolve_step1_paths(tune_dir)

    checks: Dict[str, Any] = {
        "manifest_exists": paths.manifest.is_file(),
    }
    if not checks["manifest_exists"]:
        raise FileNotFoundError(f"Missing manifest.json at {paths.manifest}")

    if validate_modfiles:
        dll = find_compiled_mechanism_dll(tune_dir)
        checks["compiled_dll"] = str(dll) if dll else None
        if dll is None:
            raise FileNotFoundError(
                "Compiled mechanisms not found. Run compile_modfiles first."
            )

    if validate_load_cell:
        with _pushd(tune_dir):
            cell = load_cell(
                {
                    "cell_name": cell_name,
                    "cell_loader": "allen_manifest",
                    "paths": {"manifest": "manifest.json"},
                    "tuning": {"soma_diam_multiplier": float(soma_diam_multiplier)},
                }
            )
        checks["load_cell"] = {
            "ok": True,
            "Vinit": cell.Vinit,
        }

    if validate_inputs:
        sim_cfg, groups_cfg = inputs.check_inputs(path=tune_dir, verbose=False)
        checks["inputs_check"] = {
            "ok": True,
            "n_groups": int(len(groups_cfg)),
            "n_active_groups": int(
                sum(
                    1
                    for gcfg in groups_cfg.values()
                    if bool(gcfg.get("state", True)) and bool(gcfg.get("mode"))
                )
            ),
            "tstop": float(sim_cfg.get("tstop", 0.0)),
            "dt": float(sim_cfg.get("dt", 0.0)),
        }

    return checks


def prepare_tune(
    *,
    tune_dir: Path,
    cell_name: str,
    tune_name: str,
    specimen_id: int,
    model_type: str = "perisomatic",
    soma_diam_multiplier: Optional[float] = None,
    color: Optional[str] = None,
    do_download: bool = True,
    force_download: bool = False,
    cache_stimulus: bool = False,
    download_match: str = "contains",
    do_compile_modfiles: bool = True,
    recompile_modfiles: bool = False,
    load_compiled_dll: bool = True,
    coerce_genome_values_to_numeric: bool = False,
    sort_genome_entries_by_section: bool = False,
    do_scaffold_configs: bool = True,
    config_mode: str = "fill",
    sync_cell_metadata: bool = True,
    do_validate: bool = True,
    validate_inputs_cfg: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end Step-1 preparation entrypoint.

    Returns a summary dictionary with actions and key paths.
    """
    tune_dir = Path(tune_dir).expanduser().resolve()
    tune_dir.mkdir(parents=True, exist_ok=True)

    soma_mult = (
        float(soma_diam_multiplier)
        if soma_diam_multiplier is not None
        else float(guess_soma_multiplier(cell_name))
    )

    summary: Dict[str, Any] = {
        "tune_dir": str(tune_dir),
        "cell_name": cell_name,
        "tune_name": tune_name,
        "specimen_id": int(specimen_id),
        "model_type": model_type,
        "soma_diam_multiplier": soma_mult,
        "actions": {},
    }

    if do_download:
        dl_info = download_cell.download_ADB_cell(
            specimen_id=int(specimen_id),
            model_type=model_type,
            tunes_dir=str(tune_dir),
            subdir=None,
            cache_stimulus=cache_stimulus,
            match=download_match,
            quiet=False,
            force=force_download,
        )
        summary["actions"]["download"] = {
            "status": "ok",
            "model_id": int(dl_info.get("model_id")),
            "model_name": dl_info.get("model_name"),
            "n_files": int(len(dl_info.get("files", []))),
        }
    else:
        summary["actions"]["download"] = {"status": "skipped"}

    if coerce_genome_values_to_numeric:
        summary["actions"]["coerce_genome_values"] = coerce_fit_genome_values_to_numeric(tune_dir)
    else:
        summary["actions"]["coerce_genome_values"] = {"status": "skipped"}

    if sort_genome_entries_by_section:
        summary["actions"]["sort_genome"] = sort_genome_by_section(tune_dir)
    else:
        summary["actions"]["sort_genome"] = {"status": "skipped"}

    if do_compile_modfiles:
        compile_info = compile_modfiles(
            tune_dir,
            recompile=recompile_modfiles,
            load_dll=load_compiled_dll,
        )
        summary["actions"]["compile_modfiles"] = {
            "status": "ok",
            **compile_info,
        }
    else:
        summary["actions"]["compile_modfiles"] = {"status": "skipped"}

    if do_scaffold_configs:
        cfg_status = scaffold_common_configs(
            tune_dir=tune_dir,
            cell_name=cell_name,
            tune_name=tune_name,
            specimen_id=int(specimen_id),
            model_type=model_type,
            soma_diam_multiplier=soma_mult,
            color=color,
            config_mode=config_mode,
            sync_cell_metadata=sync_cell_metadata,
        )
        summary["actions"]["scaffold_configs"] = {
            "status": "ok",
            "files": cfg_status,
        }
    else:
        summary["actions"]["scaffold_configs"] = {"status": "skipped"}

    if do_validate:
        checks = validate_tune(
            tune_dir=tune_dir,
            cell_name=cell_name,
            soma_diam_multiplier=soma_mult,
            validate_modfiles=True,
            validate_load_cell=True,
            validate_inputs=validate_inputs_cfg,
        )
        summary["actions"]["validate"] = {
            "status": "ok",
            "checks": checks,
        }
    else:
        summary["actions"]["validate"] = {"status": "skipped"}

    return summary
