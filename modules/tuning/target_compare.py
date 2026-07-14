"""Compare tuning notebook outputs against `target_config.json` targets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


PASSIVE_COMPARISON_SPECS: tuple[dict[str, str], ...] = (
    {
        "metric": "v_rest_mV",
        "target_key": "target_v_rest_mv",
        "measured_key": "V_rest",
        "unit": "mV",
    },
    {
        "metric": "rin_MOhm",
        "target_key": "target_rin_mohm",
        "measured_key": "R_in_rest_to_final",
        "unit": "MOhm",
    },
    {
        "metric": "tau_ms",
        "target_key": "target_tau_ms",
        "measured_key": "tau_avg",
        "unit": "ms",
    },
)


def compare_passive_targets(
    passive_metric_rows: Sequence[Mapping[str, Any]],
    passive_targets: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return table rows comparing measured passive metrics with targets."""
    rows: list[dict[str, Any]] = []
    for metric_row in passive_metric_rows:
        amp = _optional_float(metric_row.get("amp_pA"))
        for spec in PASSIVE_COMPARISON_SPECS:
            target_value = _optional_float(passive_targets.get(spec["target_key"]))
            if target_value is None:
                continue
            measured_value = _optional_float(metric_row.get(spec["measured_key"]))
            comparison = _comparison_values(measured_value, target_value)
            rows.append(
                {
                    "amp_pA": amp,
                    "metric": spec["metric"],
                    "unit": spec["unit"],
                    "target_value": target_value,
                    "measured_value": measured_value,
                    **comparison,
                }
            )
    return rows


def compare_fi_targets(
    fi_rows: Sequence[Mapping[str, Any]],
    target_points: Sequence[tuple[float, float]] | Sequence[Sequence[float]],
) -> list[dict[str, Any]]:
    """Return table rows comparing FI measurements to target FI points."""
    normalized_points = normalize_fi_reference_points(target_points)
    if not normalized_points:
        return []

    rows: list[dict[str, Any]] = []
    for row in fi_rows:
        amp = _optional_float(row.get("amp_pA"))
        measured = _first_float(
            row,
            ("spike_frequency_hz", "frequency_hz", "spike_frequency", "rate_hz"),
        )
        interp = interpolate_target_fi(amp, normalized_points)
        target = interp["target_frequency_hz"]
        comparison = _comparison_values(measured, target)
        rows.append(
            {
                "amp_pA": amp,
                "target_lookup": interp["target_lookup"],
                "target_frequency_hz": target,
                "measured_frequency_hz": measured,
                **comparison,
            }
        )
    return rows


def interpolate_target_fi(
    amp_pA: Optional[float],
    target_points: Sequence[tuple[float, float]] | Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Interpolate the target FI curve at one current amplitude.

    Extrapolation is intentionally not performed; out-of-range current steps are
    labelled clearly so notebooks do not imply unsupported target values.
    """
    points = normalize_fi_reference_points(target_points)
    if amp_pA is None:
        return {"target_frequency_hz": None, "target_lookup": "missing_amp"}
    if not points:
        return {"target_frequency_hz": None, "target_lookup": "no_targets"}

    target_by_current = _average_duplicate_currents(points)
    currents = np.asarray(sorted(target_by_current), dtype=float)
    rates = np.asarray([target_by_current[current] for current in currents], dtype=float)
    amp = float(amp_pA)

    exact_index = np.flatnonzero(np.isclose(currents, amp, rtol=0.0, atol=1e-9))
    if exact_index.size:
        return {
            "target_frequency_hz": float(rates[int(exact_index[0])]),
            "target_lookup": "exact",
        }
    if currents.size < 2 or amp < float(currents[0]) or amp > float(currents[-1]):
        return {"target_frequency_hz": None, "target_lookup": "out_of_range"}
    return {
        "target_frequency_hz": float(np.interp(amp, currents, rates)),
        "target_lookup": "interpolated",
    }


def normalize_fi_reference_points(
    points: Sequence[tuple[float, float]] | Sequence[Sequence[float]],
) -> list[tuple[float, float]]:
    """Return sorted `(amp_pA, frequency_hz)` target points."""
    normalized: list[tuple[float, float]] = []
    for point in points or []:
        if len(point) < 2:
            continue
        amp = _optional_float(point[0])
        rate = _optional_float(point[1])
        if amp is None or rate is None:
            continue
        normalized.append((amp, rate))
    return sorted(normalized, key=lambda item: item[0])


def fi_reference_points_from_csv(path: str | Path) -> list[tuple[float, float]]:
    """Read a flexible FI CSV into `(amp_pA, frequency_hz)` points."""
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    current_col = _first_present(
        rows[0],
        ("amp_pA", "current_pA", "I_pA", "mean_i", "amp_nA", "current_nA"),
    )
    rate_col = _first_present(
        rows[0],
        ("spike_frequency_hz", "frequency_hz", "freq_hz", "f_hz", "spike_frequency"),
    )
    points: list[tuple[float, float]] = []
    for row in rows:
        current = _optional_float(row.get(current_col))
        rate = _optional_float(row.get(rate_col))
        if current is None or rate is None:
            continue
        if _current_column_is_na(current_col):
            current *= 1000.0
        points.append((current, rate))
    return normalize_fi_reference_points(points)


def _comparison_values(measured: Optional[float], target: Optional[float]) -> dict[str, Any]:
    if measured is None or target is None:
        return {
            "delta": None,
            "abs_delta": None,
            "pct_error": None,
            "status": "missing_value",
        }
    delta = float(measured) - float(target)
    pct_error = None if float(target) == 0.0 else (delta / abs(float(target))) * 100.0
    return {
        "delta": delta,
        "abs_delta": abs(delta),
        "pct_error": pct_error,
        "status": "ok",
    }


def _average_duplicate_currents(points: Sequence[tuple[float, float]]) -> dict[float, float]:
    values: dict[float, list[float]] = {}
    for current, rate in points:
        values.setdefault(float(current), []).append(float(rate))
    return {current: float(np.mean(rates)) for current, rates in values.items()}


def _first_float(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        value = _optional_float(row.get(key))
        if value is not None:
            return value
    return None


def _first_present(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        if key in row:
            return key
    raise ValueError(f"CSV missing required columns. Expected one of: {', '.join(keys)}")


def _current_column_is_na(column: str) -> bool:
    text = str(column).strip().lower()
    return text in {"mean_i", "amp_na", "current_na"} or text.endswith("_na")


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
