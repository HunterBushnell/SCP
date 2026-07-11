#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


APPLY_CHOICES = (
    "sim_config",
    "cell_config",
    "geometry",
    "syn_config",
    "syn_groups",
    "fit_json",
)


_GROUP_FILE_KEYS = ("group_files",)


@dataclass
class Change:
    json_path: str
    old: Any
    new: Any


@dataclass
class FileReport:
    path: Path
    kind: str
    status: str
    message: str = ""
    changes: List[Change] = field(default_factory=list)
    backup_path: Optional[Path] = None


@dataclass
class RestoreReport:
    from_run: Path
    to_tune: Path
    dry_run: bool
    apply: Set[str]
    syn_groups: str
    file_reports: List[FileReport] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def changed_files(self) -> int:
        return sum(1 for item in self.file_reports if item.status in {"changed", "would_change"})


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


def _backup_file(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.bak_{stamp}")
    shutil.copy2(path, backup)
    return backup


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


def _overlay_values_keep_structure(
    target: Any,
    source: Any,
    path_parts: Tuple[Any, ...] = (),
) -> Tuple[Any, List[Change]]:
    if isinstance(target, dict):
        if not isinstance(source, dict):
            return copy.deepcopy(target), []
        out: Dict[str, Any] = {}
        changes: List[Change] = []
        for key, target_val in target.items():
            if key in source:
                new_val, nested = _overlay_values_keep_structure(
                    target_val, source[key], path_parts + (key,)
                )
                out[key] = new_val
                changes.extend(nested)
            else:
                out[key] = copy.deepcopy(target_val)
        return out, changes

    if isinstance(target, list):
        if not isinstance(source, list):
            return copy.deepcopy(target), []
        out_list: List[Any] = []
        changes: List[Change] = []
        for idx, target_val in enumerate(target):
            if idx < len(source):
                new_val, nested = _overlay_values_keep_structure(
                    target_val, source[idx], path_parts + (idx,)
                )
                out_list.append(new_val)
                changes.extend(nested)
            else:
                out_list.append(copy.deepcopy(target_val))
        return out_list, changes

    if isinstance(source, (dict, list)):
        return copy.deepcopy(target), []

    new_scalar = copy.deepcopy(source)
    if target != new_scalar:
        return new_scalar, [Change(_json_path(path_parts), target, new_scalar)]
    return copy.deepcopy(target), []


def _normalize_output_format(fmt: Any) -> str:
    if fmt in (None, ""):
        return "pkl"
    text = str(fmt).strip().lower()
    return "pkl" if text == "pickle" else text


def _coerce_save_shape(template: Any, source_sim: Dict[str, Any]) -> Any:
    enabled = bool(source_sim.get("save_output", source_sim.get("output") not in (None, "", False)))
    stem = source_sim.get("output")
    fmt = _normalize_output_format(source_sim.get("output_format", "pkl"))
    full = bool(source_sim.get("save_full_results", False))

    if isinstance(template, list):
        out = copy.deepcopy(template)
        if len(out) >= 1:
            out[0] = enabled
        if len(out) >= 2:
            out[1] = stem
        if len(out) >= 3:
            out[2] = fmt
        if len(out) >= 4:
            out[3] = full
        return out

    if isinstance(template, dict):
        out = copy.deepcopy(template)
        for key in list(out.keys()):
            low = key.lower()
            if low == "enabled":
                out[key] = enabled
            elif low in {"path", "stem", "name", "output"}:
                out[key] = stem
            elif low == "format":
                out[key] = fmt
            elif low in {"full_results", "save_full_results"}:
                out[key] = full
        return out

    return stem


def _coerce_load_shape(template: Any, source_sim: Dict[str, Any]) -> Any:
    enabled = bool(source_sim.get("load_enabled", source_sim.get("load") not in (None, "", False)))
    value = source_sim.get("load")

    if isinstance(template, list):
        out = copy.deepcopy(template)
        if len(out) >= 1:
            out[0] = enabled
        if len(out) >= 2:
            out[1] = value
        return out

    if isinstance(template, dict):
        out = copy.deepcopy(template)
        for key in list(out.keys()):
            low = key.lower()
            if low == "enabled":
                out[key] = enabled
            elif low in {"path", "value", "load"}:
                out[key] = value
        return out

    return value


def _coerce_append_shape(template: Any, source_sim: Dict[str, Any]) -> Any:
    append_path = source_sim.get("append_to", source_sim.get("append"))
    enabled = bool(source_sim.get("append_enabled", append_path not in (None, "", False)))

    if isinstance(template, list):
        out = copy.deepcopy(template)
        if len(out) >= 1:
            out[0] = enabled
        if len(out) >= 2:
            out[1] = append_path
        return out

    if isinstance(template, dict):
        out = copy.deepcopy(template)
        for key in list(out.keys()):
            low = key.lower()
            if low == "enabled":
                out[key] = enabled
            elif low in {"path", "value", "append"}:
                out[key] = append_path
        return out

    return append_path


def _adapt_sim_source_for_target(target_sim: Dict[str, Any], source_sim: Dict[str, Any]) -> Dict[str, Any]:
    src = copy.deepcopy(source_sim)
    if "save" in target_sim:
        src["save"] = _coerce_save_shape(target_sim.get("save"), source_sim)
    if "load" in target_sim:
        src["load"] = _coerce_load_shape(target_sim.get("load"), source_sim)
    if "append" in target_sim:
        src["append"] = _coerce_append_shape(target_sim.get("append"), source_sim)
    return src


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


def _has_group_file_manifest(groups_cfg_raw: Any) -> bool:
    return isinstance(groups_cfg_raw, dict) and (
        "group_files" in groups_cfg_raw or "__includes__" in groups_cfg_raw
    )


def _read_group_file_list(groups_cfg_raw: Dict[str, Any]) -> List[str]:
    if "__includes__" in groups_cfg_raw:
        raise ValueError("syn_config no longer supports '__includes__'; use 'group_files'.")

    raw_paths = groups_cfg_raw.get("group_files", []) or []
    if not isinstance(raw_paths, list):
        raise TypeError("syn_config field 'group_files' must be a list of relative paths.")
    return [str(item) for item in raw_paths]


def _expand_syn_config(groups_cfg_raw: Any, config_root: Path) -> Dict[str, Any]:
    if isinstance(groups_cfg_raw, dict) and not _has_group_file_manifest(groups_cfg_raw):
        return groups_cfg_raw

    include_list: List[str] = []
    inline_groups: Dict[str, Any] = {}
    if isinstance(groups_cfg_raw, list):
        include_list = [str(item) for item in groups_cfg_raw]
    elif isinstance(groups_cfg_raw, dict):
        include_list = _read_group_file_list(groups_cfg_raw)
        inline_groups = {k: v for k, v in groups_cfg_raw.items() if k not in _GROUP_FILE_KEYS}
    else:
        raise TypeError("syn_config must be dict/list/contains group_files")

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


def _find_fit_json_in_tune(tune_dir: Path, preferred_name: Optional[str] = None) -> Optional[Path]:
    if preferred_name:
        preferred = (tune_dir / preferred_name).resolve()
        if preferred.is_file():
            return preferred

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
                    items = model_file if isinstance(model_file, list) else [model_file]
                    for item in items:
                        if not isinstance(item, str):
                            continue
                        cand = Path(item)
                        if not cand.name.endswith("_fit.json"):
                            continue
                        resolved = (tune_dir / cand).resolve() if not cand.is_absolute() else cand.resolve()
                        if resolved.is_file():
                            return resolved
        except Exception:
            pass

    candidates = sorted(tune_dir.glob("*_fit.json"))
    return candidates[0].resolve() if candidates else None


def _collect_source_payloads(
    manifest_path: Path,
    source_tune_override: Optional[Path] = None,
    allow_source_fallback: bool = True,
) -> Tuple[Path, Dict[str, str], Dict[str, Any], List[str]]:
    run_dir = manifest_path.parent
    manifest = _read_json(manifest_path)
    files = manifest.get("files", {}) if isinstance(manifest, dict) else {}
    if not isinstance(files, dict):
        files = {}

    source: Dict[str, Any] = {
        "sim_config": None,
        "cell_config": None,
        "geometry": None,
        "syn_config": None,
        "fit_json": None,
        "fit_json_path": None,
        "source_tune": None,
    }
    warnings: List[str] = []

    def _load_from_manifest_file(key: str) -> Optional[Any]:
        rel = files.get(key)
        if not isinstance(rel, str):
            return None
        cand = (run_dir / rel).resolve()
        if cand.is_file():
            return _read_json(cand)
        warnings.append(f"Manifest referenced missing file for {key}: {cand}")
        return None

    source["sim_config"] = _load_from_manifest_file("sim_cfg")
    source["cell_config"] = _load_from_manifest_file("cell_config")
    source["geometry"] = _load_from_manifest_file("geometry_config")
    source["syn_config"] = _load_from_manifest_file("syn_config")

    source_tune = source_tune_override
    if source_tune is None:
        source_tune = _resolve_tune_path_from_sim(source["sim_config"])
    if source_tune is not None:
        source_tune = source_tune.expanduser().resolve()
    source["source_tune"] = source_tune

    if allow_source_fallback and source_tune is not None:
        config_root = source_tune / "cell_configs"
        fallback_files = {
            "sim_config": config_root / "sim_config.json",
            "cell_config": config_root / "cell_config.json",
            "geometry": config_root / "geometry.json",
            "syn_config": config_root / "syn_config.json",
        }
        for key, path in fallback_files.items():
            if source.get(key) is None and path.is_file():
                source[key] = _read_json(path)
                warnings.append(f"Using source-tune fallback for {key}: {path}")

    fit_rel = files.get("fit_json")
    if isinstance(fit_rel, str):
        fit_path = (run_dir / fit_rel).resolve()
        if fit_path.is_file():
            source["fit_json_path"] = fit_path
            source["fit_json"] = _read_json(fit_path)
        else:
            warnings.append(f"Manifest referenced missing fit_json sidecar: {fit_path}")

    if source["fit_json"] is None and source_tune is not None:
        fit_path = _find_fit_json_in_tune(source_tune)
        if fit_path and fit_path.is_file():
            source["fit_json_path"] = fit_path
            source["fit_json"] = _read_json(fit_path)
            warnings.append(f"Using source-tune fallback for fit_json: {fit_path}")

    return run_dir, files, source, warnings


def _resolve_apply_set(apply: Sequence[str]) -> Set[str]:
    requested: Set[str] = set()
    for item in apply:
        for token in str(item).split(","):
            token = token.strip()
            if not token:
                continue
            if token not in APPLY_CHOICES:
                raise ValueError(f"Unknown apply target '{token}'. Valid: {', '.join(APPLY_CHOICES)}")
            requested.add(token)
    return requested


def _summarize_value(value: Any, max_len: int = 120) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        text = repr(value)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _apply_single_json_file(
    *,
    target_path: Path,
    source_payload: Any,
    dry_run: bool,
    kind: str,
    report: RestoreReport,
    adapt_for_sim: bool = False,
    backup: bool = True,
) -> None:
    if source_payload is None:
        report.file_reports.append(
            FileReport(
                path=target_path,
                kind=kind,
                status="skipped",
                message="No source payload available.",
            )
        )
        return

    if not target_path.is_file():
        report.file_reports.append(
            FileReport(
                path=target_path,
                kind=kind,
                status="missing",
                message="Target file does not exist.",
            )
        )
        return

    target_payload = _read_json(target_path)
    src_payload = copy.deepcopy(source_payload)
    if adapt_for_sim and isinstance(target_payload, dict) and isinstance(src_payload, dict):
        src_payload = _adapt_sim_source_for_target(target_payload, src_payload)

    new_payload, changes = _overlay_values_keep_structure(target_payload, src_payload, ())
    if not changes:
        report.file_reports.append(
            FileReport(
                path=target_path,
                kind=kind,
                status="unchanged",
                message="No value differences on matching keys.",
            )
        )
        return

    file_report = FileReport(
        path=target_path,
        kind=kind,
        status="would_change" if dry_run else "changed",
        changes=changes,
    )

    if not dry_run:
        if backup:
            file_report.backup_path = _backup_file(target_path)
        _write_json(target_path, new_payload)

    report.file_reports.append(file_report)


def _parse_syn_groups(raw: str) -> Optional[Set[str]]:
    text = (raw or "").strip()
    if not text or text.lower() == "all":
        return None
    groups = {item.strip() for item in text.split(",") if item.strip()}
    return groups or None


def _collect_target_group_files(
    target_syn_cfg: Any,
    config_root: Path,
    syn_cfg_path: Path,
) -> Tuple[Dict[str, Path], Dict[Path, Dict[str, Any]], List[str]]:
    group_to_file: Dict[str, Path] = {}
    file_payloads: Dict[Path, Dict[str, Any]] = {}
    warnings: List[str] = []

    include_list: List[str] = []
    inline_groups: Dict[str, Any] = {}

    if isinstance(target_syn_cfg, list):
        include_list = [str(item) for item in target_syn_cfg]
    elif isinstance(target_syn_cfg, dict):
        if _has_group_file_manifest(target_syn_cfg):
            include_list = _read_group_file_list(target_syn_cfg)
            inline_groups = {k: v for k, v in target_syn_cfg.items() if k not in _GROUP_FILE_KEYS}
        else:
            inline_groups = dict(target_syn_cfg)
    else:
        warnings.append(f"Unsupported target syn_config structure: {type(target_syn_cfg)!r}")
        return group_to_file, file_payloads, warnings

    for rel in include_list:
        include_path = (config_root / rel).expanduser().resolve()
        if not include_path.is_file():
            warnings.append(f"Included syn group file not found: {include_path}")
            continue
        payload = _read_json(include_path)
        if not isinstance(payload, dict):
            warnings.append(f"Included syn group file is not a dict: {include_path}")
            continue
        file_payloads[include_path] = payload
        for group_name in payload.keys():
            if group_name in group_to_file:
                warnings.append(f"Duplicate group '{group_name}' found in includes; keeping first mapping.")
                continue
            group_to_file[group_name] = include_path

    if inline_groups:
        if syn_cfg_path not in file_payloads:
            file_payloads[syn_cfg_path] = _read_json(syn_cfg_path)
        for group_name in inline_groups.keys():
            if group_name in group_to_file:
                warnings.append(f"Group '{group_name}' exists both inline and included; keeping included mapping.")
                continue
            group_to_file[group_name] = syn_cfg_path

    return group_to_file, file_payloads, warnings


def _apply_syn_group_updates(
    *,
    target_tune: Path,
    source_syn_payload: Any,
    source_config_root: Optional[Path],
    syn_groups_selector: str,
    dry_run: bool,
    report: RestoreReport,
    backup: bool = True,
) -> None:
    config_root = target_tune / "cell_configs"
    syn_cfg_path = config_root / "syn_config.json"
    if source_syn_payload is None:
        report.file_reports.append(
            FileReport(
                path=syn_cfg_path,
                kind="syn_groups",
                status="skipped",
                message="No source syn_config payload available.",
            )
        )
        return
    if not syn_cfg_path.is_file():
        report.file_reports.append(
            FileReport(
                path=syn_cfg_path,
                kind="syn_groups",
                status="missing",
                message="Target syn_config.json does not exist.",
            )
        )
        return

    target_syn_cfg = _read_json(syn_cfg_path)
    source_groups: Any = source_syn_payload
    needs_expand = isinstance(source_syn_payload, list) or _has_group_file_manifest(source_syn_payload)
    if needs_expand:
        if source_config_root is None:
            report.file_reports.append(
                FileReport(
                    path=syn_cfg_path,
                    kind="syn_groups",
                    status="skipped",
                    message=(
                        "Source syn_config uses group file includes, but source_config_root is unavailable."
                    ),
                )
            )
            return
        try:
            source_groups = _expand_syn_config(source_syn_payload, config_root=source_config_root)
        except Exception as exc:
            report.file_reports.append(
                FileReport(
                    path=syn_cfg_path,
                    kind="syn_groups",
                    status="skipped",
                    message=f"Failed to expand source syn_config includes: {exc}",
                )
            )
            return

    if not isinstance(source_groups, dict):
        report.file_reports.append(
            FileReport(
                path=syn_cfg_path,
                kind="syn_groups",
                status="skipped",
                message=f"Source syn groups must be a dict, got {type(source_groups)!r}",
            )
        )
        return

    group_to_file, file_payloads, warnings = _collect_target_group_files(target_syn_cfg, config_root, syn_cfg_path)
    report.warnings.extend(warnings)

    selected = _parse_syn_groups(syn_groups_selector)
    if selected is None:
        candidate_groups = sorted(set(source_groups.keys()) & set(group_to_file.keys()))
    else:
        candidate_groups = sorted(selected)

    file_changes: Dict[Path, List[Change]] = {}
    for group_name in candidate_groups:
        if group_name not in source_groups:
            report.warnings.append(f"Requested syn group '{group_name}' not found in source run payload.")
            continue
        if group_name not in group_to_file:
            report.warnings.append(f"Requested syn group '{group_name}' not found in target tune mappings.")
            continue
        file_path = group_to_file[group_name]
        payload = file_payloads[file_path]
        target_group = payload.get(group_name)
        source_group = source_groups.get(group_name)
        if not isinstance(target_group, dict) or not isinstance(source_group, dict):
            report.warnings.append(
                f"Skipped syn group '{group_name}' due to non-dict payloads "
                f"(target={type(target_group)!r}, source={type(source_group)!r})."
            )
            continue

        updated_group, changes = _overlay_values_keep_structure(target_group, source_group, (group_name,))
        if changes:
            payload[group_name] = updated_group
            file_payloads[file_path] = payload
            file_changes.setdefault(file_path, []).extend(changes)

    if not file_changes:
        report.file_reports.append(
            FileReport(
                path=syn_cfg_path,
                kind="syn_groups",
                status="unchanged",
                message="No syn-group value changes found for requested groups.",
            )
        )
        return

    for file_path, changes in sorted(file_changes.items(), key=lambda item: str(item[0])):
        file_report = FileReport(
            path=file_path,
            kind="syn_groups",
            status="would_change" if dry_run else "changed",
            changes=changes,
        )
        if not dry_run:
            if backup:
                file_report.backup_path = _backup_file(file_path)
            _write_json(file_path, file_payloads[file_path])
        report.file_reports.append(file_report)


def restore_run_state(
    *,
    from_run: Path,
    to_tune: Optional[Path] = None,
    apply: Sequence[str] = ("sim_config", "cell_config", "geometry", "syn_config", "syn_groups"),
    syn_groups: str = "all",
    dry_run: bool = True,
    source_tune: Optional[Path] = None,
    allow_source_fallback: bool = True,
    backup: bool = True,
) -> RestoreReport:
    manifest_path = _resolve_manifest_path(Path(from_run))
    run_dir, manifest_files, source_payloads, source_warnings = _collect_source_payloads(
        manifest_path,
        source_tune_override=Path(source_tune).expanduser().resolve() if source_tune else None,
        allow_source_fallback=allow_source_fallback,
    )

    apply_set = _resolve_apply_set(apply)
    source_sim = source_payloads.get("sim_config")

    if to_tune is None:
        inferred = _resolve_tune_path_from_sim(source_sim)
        if inferred is None:
            raise ValueError(
                "Could not infer target tune from run sim_cfg['tune_dir']; pass --to-tune explicitly."
            )
        target_tune = inferred
    else:
        target_tune = Path(to_tune).expanduser().resolve()

    report = RestoreReport(
        from_run=run_dir,
        to_tune=target_tune,
        dry_run=dry_run,
        apply=apply_set,
        syn_groups=syn_groups,
        warnings=list(source_warnings),
    )

    if not target_tune.is_dir():
        report.errors.append(f"Target tune directory does not exist: {target_tune}")
        return report

    config_root = target_tune / "cell_configs"
    source_tune_path = source_payloads.get("source_tune")
    if source_tune_path is not None and Path(source_tune_path) != target_tune:
        report.warnings.append(
            f"Source run appears to come from tune '{source_tune_path}', but applying to '{target_tune}'."
        )

    if "sim_config" in apply_set:
        _apply_single_json_file(
            target_path=config_root / "sim_config.json",
            source_payload=source_payloads.get("sim_config"),
            dry_run=dry_run,
            kind="sim_config",
            report=report,
            adapt_for_sim=True,
            backup=backup,
        )

    if "cell_config" in apply_set:
        _apply_single_json_file(
            target_path=config_root / "cell_config.json",
            source_payload=source_payloads.get("cell_config"),
            dry_run=dry_run,
            kind="cell_config",
            report=report,
            backup=backup,
        )

    if "geometry" in apply_set:
        _apply_single_json_file(
            target_path=config_root / "geometry.json",
            source_payload=source_payloads.get("geometry"),
            dry_run=dry_run,
            kind="geometry",
            report=report,
            backup=backup,
        )

    if "syn_config" in apply_set:
        _apply_single_json_file(
            target_path=config_root / "syn_config.json",
            source_payload=source_payloads.get("syn_config"),
            dry_run=dry_run,
            kind="syn_config",
            report=report,
            backup=backup,
        )

    if "syn_groups" in apply_set:
        source_syn = source_payloads.get("syn_config")
        source_config_root = (
            (Path(source_tune_path) / "cell_configs")
            if isinstance(source_tune_path, (str, Path))
            else None
        )
        _apply_syn_group_updates(
            target_tune=target_tune,
            source_syn_payload=source_syn,
            source_config_root=source_config_root,
            syn_groups_selector=syn_groups,
            dry_run=dry_run,
            report=report,
            backup=backup,
        )

    if "fit_json" in apply_set:
        source_fit = source_payloads.get("fit_json")
        source_fit_path = source_payloads.get("fit_json_path")
        preferred_name = Path(source_fit_path).name if isinstance(source_fit_path, Path) else None
        target_fit_path = _find_fit_json_in_tune(target_tune, preferred_name=preferred_name)
        _apply_single_json_file(
            target_path=target_fit_path if target_fit_path else (target_tune / "<missing_fit_json>"),
            source_payload=source_fit,
            dry_run=dry_run,
            kind="fit_json",
            report=report,
            backup=backup,
        )

    if "fit_json" in apply_set and source_payloads.get("fit_json") is None:
        report.warnings.append("fit_json requested but source fit payload could not be resolved.")
    if "fit_json" in apply_set:
        fit_entry = next((item for item in report.file_reports if item.kind == "fit_json"), None)
        if fit_entry and str(fit_entry.path).endswith("<missing_fit_json>"):
            fit_entry.status = "missing"
            fit_entry.message = "Target fit JSON could not be resolved in target tune."

    _ = manifest_files  # keeps explicit that source comes from manifest sidecars
    return report


def print_report(report: RestoreReport) -> None:
    mode = "DRY-RUN" if report.dry_run else "WRITE"
    print(f"[restore_run_state] Mode: {mode}")
    print(f"[restore_run_state] Source run: {report.from_run}")
    print(f"[restore_run_state] Target tune: {report.to_tune}")
    print(f"[restore_run_state] Apply: {', '.join(sorted(report.apply))}")
    print(f"[restore_run_state] Syn groups: {report.syn_groups}")
    print("")

    for warning in report.warnings:
        print(f"[warning] {warning}")
    for error in report.errors:
        print(f"[error] {error}")
    if report.warnings or report.errors:
        print("")

    for item in report.file_reports:
        msg = f" ({item.message})" if item.message else ""
        print(f"[{item.status}] {item.kind}: {item.path}{msg}")
        if item.backup_path is not None:
            print(f"  backup: {item.backup_path}")
        for change in item.changes:
            print(
                f"  - {change.json_path}: "
                f"{_summarize_value(change.old)} -> {_summarize_value(change.new)}"
            )
    if report.file_reports:
        print("")

    print(
        f"[restore_run_state] Files with changes: {report.changed_files} / {len(report.file_reports)}"
    )
    if report.errors:
        print("[restore_run_state] Completed with errors.")
    elif report.dry_run:
        print("[restore_run_state] Dry-run only. Re-run with --write to apply.")
    else:
        print("[restore_run_state] Changes written.")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Restore tune config values from a saved run output while preserving target file structure."
        )
    )
    p.add_argument(
        "--from-run",
        required=True,
        help=(
            "Run path (run_manifest.json, run folder, or parent folder containing results/run_manifest.json)."
        ),
    )
    p.add_argument(
        "--to-tune",
        default=None,
        help="Target tune directory. If omitted, inferred from run sim_cfg['tune_dir'].",
    )
    p.add_argument(
        "--apply",
        default="sim_config,cell_config,geometry,syn_config,syn_groups",
        help=f"Comma-separated subset of: {', '.join(APPLY_CHOICES)}",
    )
    p.add_argument(
        "--syn-groups",
        default="all",
        help="Which syn groups to update when syn_groups is enabled: 'all' or comma list.",
    )
    p.add_argument(
        "--source-tune",
        default=None,
        help="Optional source tune directory override for fallback reads when sidecars are missing.",
    )
    p.add_argument(
        "--no-source-fallback",
        action="store_true",
        help="Disable fallback reads from source tune files if run sidecars are missing.",
    )
    p.add_argument(
        "--write",
        action="store_true",
        help="Apply changes. Default is dry-run.",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak_<timestamp> backups before writing.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    try:
        report = restore_run_state(
            from_run=Path(args.from_run),
            to_tune=Path(args.to_tune) if args.to_tune else None,
            apply=[args.apply],
            syn_groups=args.syn_groups,
            dry_run=not bool(args.write),
            source_tune=Path(args.source_tune) if args.source_tune else None,
            allow_source_fallback=not bool(args.no_source_fallback),
            backup=not bool(args.no_backup),
        )
    except Exception as exc:
        print(f"[error] restore_run_state failed: {exc}")
        return 2

    print_report(report)
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
