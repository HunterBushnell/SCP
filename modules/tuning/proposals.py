"""Tuning proposal export helpers for Step 2/3 notebooks."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import json


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def default_proposal_path(
    tune_dir: Path,
    *,
    step: str,
    stem: Optional[str] = None,
) -> Path:
    """Return a default proposal path under `<tune_dir>/tuning_exports/`."""
    safe_step = str(step).strip().replace(" ", "_")
    safe_stem = str(stem).strip().replace(" ", "_") if stem else f"{safe_step}_proposal"
    filename = f"{safe_stem}_{_utc_timestamp()}.json"
    return Path(tune_dir).expanduser().resolve() / "tuning_exports" / filename


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return value


def proposal_change_rows(proposal: Mapping[str, Any]) -> list[Dict[str, Any]]:
    """Normalize proposal changes into table-friendly row dictionaries."""
    rows: list[Dict[str, Any]] = []
    changes = proposal.get("changes", [])
    if isinstance(changes, Mapping):
        for field, new_value in changes.items():
            rows.append(
                {
                    "file": proposal.get("target_file"),
                    "field": field,
                    "old": None,
                    "new": new_value,
                    "note": None,
                }
            )
        return rows

    if not isinstance(changes, Iterable) or isinstance(changes, (str, bytes)):
        return rows

    for change in changes:
        if isinstance(change, Mapping):
            rows.append(
                {
                    "file": change.get("file", proposal.get("target_file")),
                    "field": change.get("field"),
                    "old": change.get("old"),
                    "new": change.get("new"),
                    "note": change.get("note"),
                }
            )
    return rows


def write_tuning_proposal(
    *,
    tune_dir: Path,
    step: str,
    cell_name: str,
    tune_name: str,
    changes: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    target_file: Optional[str | Path] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    notes: Optional[Iterable[str] | str] = None,
    stem: Optional[str] = None,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Write a reviewable proposal JSON without applying changes."""
    tune_path = Path(tune_dir).expanduser().resolve()
    path = Path(output_path).expanduser() if output_path else default_proposal_path(
        tune_path,
        step=step,
        stem=stem,
    )
    if not path.is_absolute():
        path = tune_path / path
    path.parent.mkdir(parents=True, exist_ok=True)

    if notes is None:
        notes_list: list[str] = []
    elif isinstance(notes, str):
        notes_list = [notes]
    else:
        notes_list = [str(note) for note in notes]

    proposal = {
        "schema": "scp_tuning_proposal.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "step": str(step),
        "cell_name": str(cell_name),
        "tune_name": str(tune_name),
        "tune_dir": str(tune_path),
        "target_file": None if target_file is None else str(target_file),
        "changes": _json_safe(changes),
        "metrics": _json_safe(dict(metrics or {})),
        "notes": notes_list,
        "applied": False,
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(proposal, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def print_proposal_summary(proposal_or_path: Mapping[str, Any] | str | Path) -> None:
    """Print a concise text summary of a proposal JSON."""
    if isinstance(proposal_or_path, (str, Path)):
        with Path(proposal_or_path).open("r", encoding="utf-8") as handle:
            proposal = json.load(handle)
    else:
        proposal = dict(proposal_or_path)

    print("Proposal:", proposal.get("schema", "unknown"))
    print("Step:", proposal.get("step"))
    print("Cell/Tune:", proposal.get("cell_name"), proposal.get("tune_name"))
    print("Target file:", proposal.get("target_file"))
    rows = proposal_change_rows(proposal)
    print("Changes:", len(rows))
    for row in rows:
        print(f"- {row.get('field')}: {row.get('old')} -> {row.get('new')}")
