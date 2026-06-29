#!/usr/bin/env python3
"""
Merge two SCP run outputs with config-first compatibility checks.

Default mode is dry-run:
  - compares key config payloads used by each run
  - reports differences that may make merge unintended
  - only performs write when --write is supplied
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules import randomness as randomness_mod
from modules import run_sim


SIM_CFG_IGNORE_KEYS = {
    "output",
    "output_stem",
    "output_format",
    "save",
    "save_output",
    "save_sidecars",
    "save_full_results",
    "save_fit_json_sidecar",
    "save_syn_records_sidecar",
    "load",
    "load_enabled",
    "append",
    "append_to",
    "append_enabled",
    "_jitter_delay_ms",
    "_jitter_tstart_ms",
    "_jitter_tstop_ms",
    "save_profile",
    "n_trials",
    "n_traces_to_save",
    "n_inputs_to_save",
    "plots_profile",
    "plots_win_size",
    "plots_input_bin_ms",
    "plots_input_smooth_ms",
    "plots_raster_style",
    "save_plots",
    "save_plots_inputs",
    "save_plots_synapses",
}

RANDOMNESS_SETTING_PATHS = (
    "trials",
    "inputs",
    "timing.tstart",
    "timing.tstop",
    "timing.jitter",
    "synapses.placement",
    "synapses.weights",
    "synapses.dynamics",
    "modes.homogeneous_poisson",
    "modes.inhomogeneous_poisson",
    "modes.precomputed",
)


@dataclass
class RunBundle:
    manifest_path: Path
    run_dir: Path
    run_label: str
    manifest_files: Dict[str, str]
    sim_config: Dict[str, Any]
    cell_config: Optional[Any]
    geometry_config: Optional[Any]
    syn_config: Optional[Any]
    results: Dict[str, Any]
    source_tune: Optional[Path]
    load_warnings: List[str] = field(default_factory=list)


@dataclass
class MergeReport:
    run_a: Path
    run_b: Path
    output_dir: Path
    output_stem: str
    dry_run: bool
    strict_configs: bool
    deterministic_expected: bool = False
    deterministic_notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    config_diffs: Dict[str, List[str]] = field(default_factory=dict)
    sim_ignored_diffs: List[str] = field(default_factory=list)
    trials_a: int = 0
    trials_b: int = 0
    merged_trials: int = 0
    output_run_dir: Optional[Path] = None
    results_dir: Optional[Path] = None
    saved_path: Optional[Path] = None
    logs_dir: Optional[Path] = None
    log_files: List[Path] = field(default_factory=list)

    @property
    def can_merge(self) -> bool:
        return not self.errors


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _looks_like_identifier(key: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key))


def _json_path(parts: Sequence[Any]) -> str:
    out = "$"
    for part in parts:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            key = str(part)
            if _looks_like_identifier(key):
                out += f".{key}"
            else:
                out += f"[{json.dumps(key)}]"
    return out


def _summarize_value(value: Any, max_len: int = 120) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        text = repr(value)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _parse_setting_kind(val: Any) -> str:
    if val is None or val is False:
        return "fixed"
    if val is True:
        return "random"
    if isinstance(val, int) and not isinstance(val, bool):
        return "fixed_explicit"
    if isinstance(val, str):
        txt = val.strip().lower()
        if txt in ("fixed", "off", "none", "identical", "same", ""):
            return "fixed"
        if txt in ("derived", "per_trial", "trial"):
            return "derived"
        if txt in ("random", "stochastic", "full"):
            return "random"
    return "unknown"


def _normalize_mode(mode: Any) -> Optional[str]:
    if mode in (None, "", False):
        return None
    txt = str(mode).strip().lower()
    aliases = {
        "fixed": "fixed",
        "off": "fixed",
        "none": "fixed",
        "identical": "fixed",
        "same": "fixed",
        "derived": "derived",
        "per_trial": "derived",
        "trial": "derived",
        "random": "random",
        "stochastic": "random",
        "full": "random",
    }
    return aliases.get(txt, txt)


def _resolve_seed(sim_cfg: Dict[str, Any]) -> Optional[int]:
    if not isinstance(sim_cfg, dict):
        return None
    randomness_cfg = sim_cfg.get("randomness")
    if isinstance(randomness_cfg, dict):
        seed = randomness_mod.get_by_path(randomness_cfg, "global.seed", None)
        if seed not in (None, "", False):
            try:
                return int(seed)
            except Exception:
                return None
    for key in ("seed", "random_seed"):
        value = sim_cfg.get(key)
        if value not in (None, "", False):
            try:
                return int(value)
            except Exception:
                return None
    return None


def _expects_deterministic_equivalence(sim_cfg: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(sim_cfg, dict):
        return False, "sim_config missing."

    mode = _normalize_mode(sim_cfg.get("randomness_mode"))
    if mode == "random":
        return False, "randomness_mode=random."
    if mode == "fixed":
        seed = _resolve_seed(sim_cfg)
        if seed is None:
            return False, "randomness_mode=fixed but no seed is set."
        return True, f"randomness_mode=fixed with seed={seed}."

    sim_local = copy.deepcopy(sim_cfg)
    try:
        randomness_mod.apply_randomness_mode(sim_local)
    except Exception:
        pass

    rand_cfg = sim_local.get("randomness")
    if not isinstance(rand_cfg, dict):
        seed = _resolve_seed(sim_local)
        if seed is None:
            return False, "No randomness config and no seed set."
        return True, f"Legacy seed-based config with seed={seed}."

    global_state = bool(randomness_mod.get_by_path(rand_cfg, "global.state", True))
    seed = _resolve_seed(sim_local)
    if seed is None:
        if global_state:
            return False, "randomness.global.seed is unset (base seed can vary by run)."
        return False, "randomness.global.state=false and seed unset."

    for path in RANDOMNESS_SETTING_PATHS:
        kind = _parse_setting_kind(randomness_mod.get_by_path(rand_cfg, path, None))
        if kind == "random":
            return False, f"randomness.{path}=random."
        if kind == "unknown":
            return False, f"randomness.{path} uses unsupported setting."

    return True, f"No random settings with fixed seed={seed}."


def _resolve_manifest_path(run_path: Path) -> Path:
    p = run_path.expanduser().resolve()
    if p.is_file():
        if p.name == "run_manifest.json":
            return p
        for cand in (p.parent / "run_manifest.json", p.parent.parent / "run_manifest.json"):
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"Could not locate run_manifest.json from file path: {p}")

    if p.is_dir():
        for cand in (p / "run_manifest.json", p / "results" / "run_manifest.json"):
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"No run_manifest.json found under: {p}")

    raise FileNotFoundError(f"Run path not found: {p}")


def _infer_output_data_dir(run_dir: Path) -> Path:
    if run_dir.name == "results":
        return run_dir.parent.parent
    return run_dir.parent


def _infer_run_label(run_dir: Path, files: Dict[str, str], manifest: Dict[str, Any]) -> str:
    if run_dir.name == "results":
        parent_name = run_dir.parent.name
        if parent_name:
            return parent_name
    out_stem = manifest.get("output_stem")
    if isinstance(out_stem, str) and out_stem.strip():
        return out_stem.strip()
    stem = run_dir.name
    if stem:
        return stem
    sim_path = files.get("sim_cfg")
    if isinstance(sim_path, str):
        return Path(sim_path).stem
    return "run"


def _resolve_tune_path_from_sim(sim_cfg: Optional[Dict[str, Any]]) -> Optional[Path]:
    if not isinstance(sim_cfg, dict):
        return None
    tune_dir = sim_cfg.get("tune_dir")
    if not tune_dir:
        return None
    try:
        return Path(str(tune_dir)).expanduser().resolve()
    except Exception:
        return None


def _load_manifest_json_sidecar(run_dir: Path, files: Dict[str, Any], key: str) -> Optional[Any]:
    rel = files.get(key)
    if not isinstance(rel, str):
        return None
    path = (run_dir / rel).resolve()
    if not path.is_file():
        return None
    try:
        return _read_json(path)
    except Exception:
        return None


def _expand_syn_config(groups_cfg_raw: Any, config_root: Path) -> Dict[str, Any]:
    if isinstance(groups_cfg_raw, dict) and "__includes__" not in groups_cfg_raw:
        return groups_cfg_raw

    include_list: List[str] = []
    inline_groups: Dict[str, Any] = {}
    if isinstance(groups_cfg_raw, list):
        include_list = [str(item) for item in groups_cfg_raw]
    elif isinstance(groups_cfg_raw, dict):
        include_list = [str(item) for item in groups_cfg_raw.get("__includes__", []) or []]
        inline_groups = {k: v for k, v in groups_cfg_raw.items() if k != "__includes__"}
    else:
        raise TypeError("syn_config must be dict/list/contains __includes__")

    merged: Dict[str, Any] = {}
    for rel_path in include_list:
        include_path = (config_root / rel_path).expanduser().resolve()
        include_data = _read_json(include_path)
        if not isinstance(include_data, dict):
            raise TypeError(f"Included synapse config {include_path} must be a dict")
        for group_name, group_cfg in include_data.items():
            if group_name in merged:
                raise ValueError(f"Duplicate group '{group_name}' while loading {include_path}")
            merged[group_name] = group_cfg

    for group_name, group_cfg in inline_groups.items():
        if group_name in merged:
            raise ValueError(f"Duplicate inline group '{group_name}'")
        merged[group_name] = group_cfg

    return merged


def _load_with_fallback(
    *,
    run_dir: Path,
    files: Dict[str, Any],
    manifest_key: str,
    source_tune: Optional[Path],
    fallback_rel: str,
) -> Tuple[Optional[Any], Optional[str]]:
    payload = _load_manifest_json_sidecar(run_dir, files, manifest_key)
    if payload is not None:
        return payload, None

    if source_tune is None:
        return None, None

    path = (source_tune / "cell_configs" / fallback_rel).resolve()
    if not path.is_file():
        return None, None

    try:
        return _read_json(path), f"Using source-tune fallback for {manifest_key}: {path}"
    except Exception as exc:
        return None, f"Failed reading source-tune fallback for {manifest_key}: {path} ({exc})"


def _load_run_bundle(run_path: Path) -> RunBundle:
    manifest_path = _resolve_manifest_path(Path(run_path))
    run_dir = manifest_path.parent
    manifest = _read_json(manifest_path)
    files = manifest.get("files", {}) if isinstance(manifest, dict) else {}
    if not isinstance(files, dict):
        files = {}

    results = run_sim.load_results(manifest_path)
    sim_sidecar = _load_manifest_json_sidecar(run_dir, files, "sim_cfg")
    sim_config = sim_sidecar if isinstance(sim_sidecar, dict) else copy.deepcopy(results.get("sim_cfg", {}) or {})
    source_tune = _resolve_tune_path_from_sim(sim_config)

    warnings: List[str] = []

    cell_cfg, msg = _load_with_fallback(
        run_dir=run_dir,
        files=files,
        manifest_key="cell_config",
        source_tune=source_tune,
        fallback_rel="cell_config.json",
    )
    if msg:
        warnings.append(msg)

    geom_cfg, msg = _load_with_fallback(
        run_dir=run_dir,
        files=files,
        manifest_key="geometry_config",
        source_tune=source_tune,
        fallback_rel="geometry.json",
    )
    if msg:
        warnings.append(msg)

    syn_cfg_raw, msg = _load_with_fallback(
        run_dir=run_dir,
        files=files,
        manifest_key="syn_config",
        source_tune=source_tune,
        fallback_rel="syn_config.json",
    )
    if msg:
        warnings.append(msg)

    syn_cfg = syn_cfg_raw
    needs_expand = isinstance(syn_cfg_raw, list) or (
        isinstance(syn_cfg_raw, dict) and "__includes__" in syn_cfg_raw
    )
    if needs_expand:
        if source_tune is None:
            warnings.append(
                "syn_config uses include structure, but source tune is unavailable; comparing raw structure."
            )
        else:
            config_root = source_tune / "cell_configs"
            try:
                syn_cfg = _expand_syn_config(syn_cfg_raw, config_root=config_root)
            except Exception as exc:
                warnings.append(f"Failed to expand syn_config includes for compare: {exc}")
                syn_cfg = syn_cfg_raw

    return RunBundle(
        manifest_path=manifest_path,
        run_dir=run_dir,
        run_label=_infer_run_label(run_dir, files, manifest),
        manifest_files={str(k): str(v) for k, v in files.items() if isinstance(k, str) and isinstance(v, str)},
        sim_config=sim_config,
        cell_config=cell_cfg,
        geometry_config=geom_cfg,
        syn_config=syn_cfg,
        results=results,
        source_tune=source_tune,
        load_warnings=warnings,
    )


def _sanitize_sim_cfg_for_compare(sim_cfg: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, value in (sim_cfg or {}).items():
        if key in SIM_CFG_IGNORE_KEYS:
            continue
        clean[key] = copy.deepcopy(value)
    return clean


def _extract_ignored_sim_cfg(sim_cfg: Dict[str, Any]) -> Dict[str, Any]:
    ignored: Dict[str, Any] = {}
    for key in SIM_CFG_IGNORE_KEYS:
        if key in sim_cfg:
            ignored[key] = copy.deepcopy(sim_cfg[key])
    return ignored


def _diff_values(
    a: Any,
    b: Any,
    *,
    path_parts: Tuple[Any, ...],
    diffs: List[str],
    max_diffs: int,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    if len(diffs) >= max_diffs:
        return

    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        aa = np.asarray(a, dtype=object)
        bb = np.asarray(b, dtype=object)
        path = _json_path(path_parts)
        if aa.shape != bb.shape:
            diffs.append(f"{path}: shape {aa.shape} vs {bb.shape}")
            return
        try:
            if np.allclose(aa.astype(float), bb.astype(float), rtol=rtol, atol=atol):
                return
            max_abs = float(np.max(np.abs(aa.astype(float) - bb.astype(float))))
            diffs.append(f"{path}: arrays differ (max_abs_diff={max_abs:.6g})")
        except Exception:
            if not np.array_equal(aa, bb):
                diffs.append(f"{path}: arrays differ")
        return

    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a.keys()) | set(b.keys()), key=lambda k: str(k))
        for key in keys:
            if len(diffs) >= max_diffs:
                return
            if key not in a:
                diffs.append(f"{_json_path(path_parts + (key,))}: only in B")
                continue
            if key not in b:
                diffs.append(f"{_json_path(path_parts + (key,))}: only in A")
                continue
            _diff_values(
                a[key],
                b[key],
                path_parts=path_parts + (key,),
                diffs=diffs,
                max_diffs=max_diffs,
                rtol=rtol,
                atol=atol,
            )
        return

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f"{_json_path(path_parts)}: length {len(a)} vs {len(b)}")
            return
        for idx, (va, vb) in enumerate(zip(a, b)):
            if len(diffs) >= max_diffs:
                return
            _diff_values(
                va,
                vb,
                path_parts=path_parts + (idx,),
                diffs=diffs,
                max_diffs=max_diffs,
                rtol=rtol,
                atol=atol,
            )
        return

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol):
            diffs.append(f"{_json_path(path_parts)}: {a!r} != {b!r}")
        return

    if a != b:
        diffs.append(f"{_json_path(path_parts)}: {_summarize_value(a)} != {_summarize_value(b)}")


def _ensure_multi(res: Dict[str, Any]) -> Dict[str, Any]:
    mode = res.get("mode", "single")
    if mode == "multi":
        return res
    if mode != "single":
        raise ValueError(f"Unsupported results mode: {mode!r}")

    sim_cfg = copy.deepcopy(res.get("sim_cfg", {}))
    traces = res.get("traces", {}) or {}
    inputs = res.get("inputs", None)
    spikes = np.asarray(res.get("spikes", []), dtype=float)

    multi = {
        "mode": "multi",
        "sim_cfg": sim_cfg,
        "spikes": [spikes],
        "traces": {},
        "inputs_by_trial": None,
        "meta": copy.deepcopy(res.get("meta", {})),
    }
    if "T" in traces and "V" in traces:
        multi["traces"] = {"T": traces["T"], "V": [traces["V"]]}
    if inputs:
        multi["inputs_by_trial"] = [{"trial_idx": 0, "inputs": inputs}]

    meta = multi.setdefault("meta", {})
    meta["n_trials"] = 1
    meta["trial_ids"] = [0]
    return multi


def _compare_payloads(a: Any, b: Any, *, root: str, max_diffs: int) -> List[str]:
    diffs: List[str] = []
    _diff_values(
        a,
        b,
        path_parts=(root,),
        diffs=diffs,
        max_diffs=max_diffs,
        rtol=0.0,
        atol=0.0,
    )
    return diffs


def _is_run_generated_jitter_diff(section: str, diff_line: str) -> bool:
    line = str(diff_line).lower()
    if section == "sim_config":
        return "._jitter_" in line
    if section == "syn_config":
        return ".time_cfg.anchors.jitter_" in line
    return False


def _split_jitter_diffs(section: str, diffs: List[str]) -> Tuple[List[str], List[str]]:
    kept: List[str] = []
    dropped: List[str] = []
    for line in diffs:
        if _is_run_generated_jitter_diff(section, line):
            dropped.append(line)
        else:
            kept.append(line)
    return kept, dropped


def _stable_hash(payload: Any) -> Optional[str]:
    if payload is None:
        return None
    try:
        txt = json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        try:
            txt = repr(payload)
        except Exception:
            return None
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def _default_output_stem(bundle_a: RunBundle, bundle_b: RunBundle) -> str:
    base = f"merged_{bundle_a.run_label}_plus_{bundle_b.run_label}"
    clean = re.sub(r"[^A-Za-z0-9._=-]+", "_", base).strip("_")
    clean = re.sub(r"_+", "_", clean)
    return clean or "merged_runs"


def _resolve_output_run_dir(output_dir: Path, stem: str) -> Tuple[Path, str, bool]:
    run_dir = output_dir / stem
    if not run_dir.exists():
        return run_dir, stem, False
    idx = 1
    while True:
        alt_stem = f"{stem}_{idx}"
        alt_dir = output_dir / alt_stem
        if not alt_dir.exists():
            return alt_dir, alt_stem, True
        idx += 1


def _report_to_jsonable(report: MergeReport) -> Dict[str, Any]:
    return {
        "run_a": str(report.run_a),
        "run_b": str(report.run_b),
        "output_dir": str(report.output_dir),
        "output_stem": report.output_stem,
        "dry_run": bool(report.dry_run),
        "strict_configs": bool(report.strict_configs),
        "deterministic_expected": bool(report.deterministic_expected),
        "deterministic_notes": list(report.deterministic_notes),
        "warnings": list(report.warnings),
        "errors": list(report.errors),
        "config_diffs": dict(report.config_diffs),
        "sim_ignored_diffs": list(report.sim_ignored_diffs),
        "trials_a": int(report.trials_a),
        "trials_b": int(report.trials_b),
        "merged_trials": int(report.merged_trials),
        "output_run_dir": str(report.output_run_dir) if report.output_run_dir else None,
        "results_dir": str(report.results_dir) if report.results_dir else None,
        "saved_path": str(report.saved_path) if report.saved_path else None,
        "logs_dir": str(report.logs_dir) if report.logs_dir else None,
        "log_files": [str(path) for path in report.log_files],
    }


def _write_merge_logs(report: MergeReport) -> None:
    if report.output_run_dir is None:
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = report.output_run_dir / "logs" / f"merge_{stamp}"
    logs_dir.mkdir(parents=True, exist_ok=True)

    report_json_path = logs_dir / "merge_report.json"
    report_txt_path = logs_dir / "merge_report.txt"

    report_json_path.write_text(
        json.dumps(_report_to_jsonable(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"Mode: {'DRY-RUN' if report.dry_run else 'WRITE'}",
        f"run_a: {report.run_a}",
        f"run_b: {report.run_b}",
        f"output_dir: {report.output_dir}",
        f"output_stem: {report.output_stem}",
        f"output_run_dir: {report.output_run_dir}",
        f"results_dir: {report.results_dir}",
        f"saved_path: {report.saved_path}",
        f"deterministic_expected: {report.deterministic_expected}",
        f"warnings: {len(report.warnings)}",
        f"errors: {len(report.errors)}",
    ]
    report_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report.logs_dir = logs_dir
    report.log_files.extend([report_json_path, report_txt_path])


def _apply_config_checks(
    report: MergeReport,
    bundle_a: RunBundle,
    bundle_b: RunBundle,
    *,
    max_diffs: int,
    ignore_stochastic_jitter_diffs: bool,
) -> None:
    report.warnings.extend([f"[run_a] {msg}" for msg in bundle_a.load_warnings])
    report.warnings.extend([f"[run_b] {msg}" for msg in bundle_b.load_warnings])

    sim_a_raw = bundle_a.sim_config or {}
    sim_b_raw = bundle_b.sim_config or {}
    sim_a = _sanitize_sim_cfg_for_compare(sim_a_raw)
    sim_b = _sanitize_sim_cfg_for_compare(sim_b_raw)

    sim_diffs = _compare_payloads(sim_a, sim_b, root="sim_config", max_diffs=max_diffs)
    dropped_sim_jitter: List[str] = []
    if ignore_stochastic_jitter_diffs:
        sim_diffs, dropped_sim_jitter = _split_jitter_diffs("sim_config", sim_diffs)
    report.config_diffs["sim_config"] = sim_diffs

    ignored_a = _extract_ignored_sim_cfg(sim_a_raw)
    ignored_b = _extract_ignored_sim_cfg(sim_b_raw)
    report.sim_ignored_diffs = _compare_payloads(
        ignored_a, ignored_b, root="sim_config_ignored", max_diffs=max_diffs
    )

    if sim_diffs:
        msg = f"sim_config differs on {len(sim_diffs)} compared paths."
        if report.strict_configs:
            report.errors.append(msg)
        else:
            report.warnings.append(msg)
    if dropped_sim_jitter:
        report.warnings.append(
            f"Ignored {len(dropped_sim_jitter)} run-generated jitter diff(s) in sim_config "
            "because randomness is stochastic."
        )

    pairs = (
        ("cell_config", bundle_a.cell_config, bundle_b.cell_config),
        ("geometry_config", bundle_a.geometry_config, bundle_b.geometry_config),
        ("syn_config", bundle_a.syn_config, bundle_b.syn_config),
    )
    for name, payload_a, payload_b in pairs:
        if payload_a is None or payload_b is None:
            msg = (
                f"{name} is missing in "
                f"{'run_a' if payload_a is None else ''}"
                f"{' and ' if payload_a is None and payload_b is None else ''}"
                f"{'run_b' if payload_b is None else ''}; cannot fully verify config parity."
            )
            if report.strict_configs:
                report.errors.append(msg)
            else:
                report.warnings.append(msg)
            report.config_diffs[name] = []
            continue

        diffs = _compare_payloads(payload_a, payload_b, root=name, max_diffs=max_diffs)
        dropped_jitter: List[str] = []
        if ignore_stochastic_jitter_diffs:
            diffs, dropped_jitter = _split_jitter_diffs(name, diffs)
        report.config_diffs[name] = diffs
        if diffs:
            msg = f"{name} differs on {len(diffs)} compared paths."
            if report.strict_configs:
                report.errors.append(msg)
            else:
                report.warnings.append(msg)
        if dropped_jitter:
            report.warnings.append(
                f"Ignored {len(dropped_jitter)} run-generated jitter diff(s) in {name} "
                "because randomness is stochastic."
            )


def _apply_randomness_checks(
    report: MergeReport,
    bundle_a: RunBundle,
    bundle_b: RunBundle,
) -> None:
    det_a, note_a = _expects_deterministic_equivalence(bundle_a.sim_config)
    det_b, note_b = _expects_deterministic_equivalence(bundle_b.sim_config)
    report.deterministic_expected = det_a and det_b
    report.deterministic_notes = [f"run_a: {note_a}", f"run_b: {note_b}"]

    if not report.deterministic_expected:
        report.warnings.append(
            "Randomness settings indicate stochastic variation; seed/input differences are treated as expected."
        )
        return

    seed_a = _resolve_seed(bundle_a.sim_config)
    seed_b = _resolve_seed(bundle_b.sim_config)
    if seed_a != seed_b:
        report.warnings.append(
            f"Deterministic mode expected but sim seeds differ (run_a={seed_a}, run_b={seed_b})."
        )

    meta_a = bundle_a.results.get("meta", {}) or {}
    meta_b = bundle_b.results.get("meta", {}) or {}

    base_seed_a = (meta_a.get("randomness") or {}).get("base_seed_used")
    base_seed_b = (meta_b.get("randomness") or {}).get("base_seed_used")
    if base_seed_a is not None and base_seed_b is not None and base_seed_a != base_seed_b:
        report.warnings.append(
            f"Deterministic mode expected but runtime base_seed_used differs "
            f"(run_a={base_seed_a}, run_b={base_seed_b})."
        )

    input_hash_a = _stable_hash(meta_a.get("input_summaries"))
    input_hash_b = _stable_hash(meta_b.get("input_summaries"))
    if input_hash_a and input_hash_b and input_hash_a != input_hash_b:
        report.warnings.append(
            "Deterministic mode expected but input_summaries differ between runs."
        )


def merge_two_runs(
    *,
    run_a: Path,
    run_b: Path,
    output_dir: Optional[Path] = None,
    output_stem: Optional[str] = None,
    dry_run: bool = True,
    strict_configs: bool = True,
    keep_logs: bool = True,
    max_diffs: int = 200,
) -> MergeReport:
    bundle_a = _load_run_bundle(Path(run_a))
    bundle_b = _load_run_bundle(Path(run_b))

    out_dir = Path(output_dir).expanduser().resolve() if output_dir else _infer_output_data_dir(bundle_a.run_dir)
    stem = str(output_stem).strip() if output_stem not in (None, "") else _default_output_stem(bundle_a, bundle_b)

    run_root, resolved_stem, stem_renamed = _resolve_output_run_dir(out_dir, stem)
    if stem_renamed:
        rename_msg = (
            f"Output run folder already exists for '{stem}'; using '{resolved_stem}' instead."
        )
    else:
        rename_msg = None

    report = MergeReport(
        run_a=bundle_a.manifest_path.parent,
        run_b=bundle_b.manifest_path.parent,
        output_dir=out_dir,
        output_stem=resolved_stem,
        dry_run=bool(dry_run),
        strict_configs=bool(strict_configs),
        output_run_dir=run_root,
        results_dir=run_root / "results",
    )
    if rename_msg:
        report.warnings.append(rename_msg)

    res_a_multi = _ensure_multi(bundle_a.results)
    res_b_multi = _ensure_multi(bundle_b.results)
    report.trials_a = len(res_a_multi.get("spikes", []) or [])
    report.trials_b = len(res_b_multi.get("spikes", []) or [])
    report.merged_trials = report.trials_a + report.trials_b

    _apply_randomness_checks(report, bundle_a, bundle_b)
    _apply_config_checks(
        report,
        bundle_a,
        bundle_b,
        max_diffs=max_diffs,
        ignore_stochastic_jitter_diffs=not report.deterministic_expected,
    )

    if report.dry_run or not report.can_merge:
        return report

    merged = run_sim.append_multi_results(res_a_multi, res_b_multi)
    sim_cfg = merged.setdefault("sim_cfg", {})
    sim_cfg["output"] = "results"
    sim_cfg["save_output"] = True
    sim_cfg["append_enabled"] = False
    sim_cfg.pop("append", None)
    sim_cfg.pop("append_to", None)

    saved = run_sim.save_results(merged, base_dir=run_root)
    report.saved_path = saved
    if keep_logs:
        _write_merge_logs(report)
    return report


def print_report(report: MergeReport, *, max_lines_per_section: int = 40) -> None:
    mode = "DRY-RUN" if report.dry_run else "WRITE"
    print(f"[merge_two_runs] Mode: {mode}")
    print(f"[merge_two_runs] run_a: {report.run_a}")
    print(f"[merge_two_runs] run_b: {report.run_b}")
    print(f"[merge_two_runs] output_dir: {report.output_dir}")
    print(f"[merge_two_runs] output_stem: {report.output_stem}")
    if report.output_run_dir is not None:
        print(f"[merge_two_runs] output_run_dir: {report.output_run_dir}")
    if report.results_dir is not None:
        print(f"[merge_two_runs] results_dir: {report.results_dir}")
    print(
        f"[merge_two_runs] trial counts: run_a={report.trials_a}, run_b={report.trials_b}, merged={report.merged_trials}"
    )
    print("")
    print(
        f"[merge_two_runs] deterministic_equivalence_expected: {report.deterministic_expected}"
    )
    for line in report.deterministic_notes:
        print(f"  - {line}")
    print("")

    for warning in report.warnings:
        print(f"[warning] {warning}")
    for error in report.errors:
        print(f"[error] {error}")
    if report.warnings or report.errors:
        print("")

    for section in ("sim_config", "cell_config", "geometry_config", "syn_config"):
        diffs = report.config_diffs.get(section, [])
        print(f"[compare] {section}: {len(diffs)} difference(s)")
        for line in diffs[:max_lines_per_section]:
            print(f"  - {line}")
        if len(diffs) > max_lines_per_section:
            print(f"  - ... ({len(diffs) - max_lines_per_section} more)")
        print("")

    if report.sim_ignored_diffs:
        print(f"[compare] sim_config ignored-key differences: {len(report.sim_ignored_diffs)}")
        for line in report.sim_ignored_diffs[:max_lines_per_section]:
            print(f"  - {line}")
        if len(report.sim_ignored_diffs) > max_lines_per_section:
            print(f"  - ... ({len(report.sim_ignored_diffs) - max_lines_per_section} more)")
        print("")

    if report.saved_path is not None:
        print(f"[merge_two_runs] saved: {report.saved_path}")
        if report.logs_dir is not None:
            print(f"[merge_two_runs] logs: {report.logs_dir}")
    elif report.dry_run:
        print("[merge_two_runs] Dry-run only. Re-run with --write to merge.")
    elif report.errors:
        print("[merge_two_runs] Merge aborted due to errors.")
    else:
        print("[merge_two_runs] Merge finished.")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge two SCP run outputs with config-first compatibility checks."
    )
    p.add_argument(
        "--run-a",
        required=True,
        help="First run path (run dir, results dir, or run_manifest.json).",
    )
    p.add_argument(
        "--run-b",
        required=True,
        help="Second run path (run dir, results dir, or run_manifest.json).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output_data directory for merged run (default: inferred from run_a).",
    )
    p.add_argument(
        "--output-stem",
        default=None,
        help="Merged run folder stem (default: auto-generated).",
    )
    p.add_argument(
        "--max-diffs",
        type=int,
        default=200,
        help="Max diff lines per config section (default: 200).",
    )
    p.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help="Do not block merge on config differences; report warnings instead.",
    )
    p.add_argument(
        "--no-logs",
        action="store_true",
        help="Do not write logs/<merge_timestamp>/merge_report.* in the merged run folder.",
    )
    p.add_argument(
        "--write",
        action="store_true",
        help="Apply merge and save output. Default is dry-run.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    report = merge_two_runs(
        run_a=Path(args.run_a),
        run_b=Path(args.run_b),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        output_stem=args.output_stem,
        dry_run=not bool(args.write),
        strict_configs=not bool(args.allow_config_mismatch),
        keep_logs=not bool(args.no_logs),
        max_diffs=max(1, int(args.max_diffs)),
    )
    print_report(report)
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
