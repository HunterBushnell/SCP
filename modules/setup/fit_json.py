"""Allen fit-JSON discovery and normalization helpers for Step 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .json_utils import _read_json, _write_json

DEFAULT_GENOME_SECTION_ORDER = ("glob", "soma", "axon", "apic", "dend")


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
