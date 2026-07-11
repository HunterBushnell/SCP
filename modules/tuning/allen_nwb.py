"""Allen/ADB NWB helpers for ACT target generation.

The public-facing contract is intentionally narrow: read a user-downloaded
Allen Cell Types current-clamp NWB file and export FI targets that ACT already
accepts robustly (`target_sf.csv` with `mean_i` in nA and `spike_frequency` in
Hz). Raw voltage-trace export can be added later once the ACT trace path is
more explicit about sampling/window semantics.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import csv

import numpy as np


DEFAULT_FI_STIMULUS_NAMES = ("Long Square",)


def summarize_allen_nwb_sweeps(nwb_path: str | Path) -> list[dict[str, Any]]:
    """Return one metadata row per paired stimulus/response sweep."""
    rows = []
    with _open_nwb(nwb_path) as handle:
        acquisition = handle["acquisition/timeseries"]
        stimulus = handle["stimulus/presentation"]
        for key in _sorted_sweep_keys(set(acquisition.keys()) & set(stimulus.keys())):
            response_group = acquisition[key]
            stimulus_group = stimulus[key]
            sample_rate_hz = _sample_rate_hz(response_group, stimulus_group)
            amp_pA = _read_float(response_group.get("aibs_stimulus_amplitude_pa"))
            stimulus_data = _current_pA(stimulus_group["data"][:])
            start_idx, stop_idx = _active_current_window(stimulus_data, amp_pA, sample_rate_hz)
            rows.append(
                {
                    "sweep": _sweep_number(key),
                    "stimulus_name": _read_text(response_group.get("aibs_stimulus_name")),
                    "stimulus_description": _read_text(response_group.get("aibs_stimulus_description")),
                    "amplitude_pA": amp_pA,
                    "sample_rate_hz": sample_rate_hz,
                    "stim_start_ms": start_idx / sample_rate_hz * 1000.0,
                    "stim_stop_ms": stop_idx / sample_rate_hz * 1000.0,
                    "stim_duration_ms": (stop_idx - start_idx) / sample_rate_hz * 1000.0,
                    "n_samples": int(len(stimulus_data)),
                }
            )
    return rows


def extract_allen_nwb_fi_sweeps(
    nwb_path: str | Path,
    *,
    stimulus_names: Sequence[str] = DEFAULT_FI_STIMULUS_NAMES,
    include_negative_currents: bool = False,
    min_current_pA: Optional[float] = 0.0,
    max_current_pA: Optional[float] = None,
    spike_threshold_mV: float = -20.0,
    refractory_ms: float = 1.0,
) -> list[dict[str, Any]]:
    """Extract per-sweep FI measurements from Allen current-clamp sweeps."""
    wanted_names = {str(name).strip() for name in stimulus_names}
    rows = []
    with _open_nwb(nwb_path) as handle:
        acquisition = handle["acquisition/timeseries"]
        stimulus = handle["stimulus/presentation"]
        for key in _sorted_sweep_keys(set(acquisition.keys()) & set(stimulus.keys())):
            response_group = acquisition[key]
            stimulus_group = stimulus[key]
            stimulus_name = _read_text(response_group.get("aibs_stimulus_name"))
            if wanted_names and stimulus_name not in wanted_names:
                continue

            amp_pA = _read_float(response_group.get("aibs_stimulus_amplitude_pa"))
            if not include_negative_currents and amp_pA < 0:
                continue
            if min_current_pA is not None and amp_pA < float(min_current_pA):
                continue
            if max_current_pA is not None and amp_pA > float(max_current_pA):
                continue

            sample_rate_hz = _sample_rate_hz(response_group, stimulus_group)
            voltage_mV = _voltage_mV(response_group["data"][:])
            current_pA = _current_pA(stimulus_group["data"][:])
            n_samples = min(len(voltage_mV), len(current_pA))
            voltage_mV = voltage_mV[:n_samples]
            current_pA = current_pA[:n_samples]
            start_idx, stop_idx = _active_current_window(current_pA, amp_pA, sample_rate_hz)
            spike_count = _count_threshold_crossings(
                voltage_mV,
                start_idx,
                stop_idx,
                threshold_mV=float(spike_threshold_mV),
                refractory_ms=float(refractory_ms),
                sample_rate_hz=sample_rate_hz,
            )
            duration_s = (stop_idx - start_idx) / sample_rate_hz
            rows.append(
                {
                    "sweep": _sweep_number(key),
                    "stimulus_name": stimulus_name,
                    "stimulus_description": _read_text(response_group.get("aibs_stimulus_description")),
                    "amp_pA": amp_pA,
                    "stim_start_ms": start_idx / sample_rate_hz * 1000.0,
                    "stim_stop_ms": stop_idx / sample_rate_hz * 1000.0,
                    "stim_duration_ms": duration_s * 1000.0,
                    "spike_count": spike_count,
                    "spike_frequency_hz": spike_count / duration_s if duration_s > 0 else np.nan,
                    "resting_voltage_mV": float(np.nanmedian(voltage_mV[: max(1, int(0.2 * sample_rate_hz))])),
                    "max_voltage_mV": float(np.nanmax(voltage_mV[start_idx:stop_idx])),
                    "min_voltage_mV": float(np.nanmin(voltage_mV[start_idx:stop_idx])),
                    "sample_rate_hz": sample_rate_hz,
                }
            )
    if not rows:
        raise ValueError(
            "No matching NWB FI sweeps found. Check stimulus_names/current filters "
            f"for {nwb_path!s}."
        )
    return rows


def aggregate_fi_sweeps(
    rows: Sequence[Mapping[str, Any]],
    *,
    average_repeats: bool = True,
) -> list[dict[str, Any]]:
    """Convert per-sweep FI rows into one ACT target row per current."""
    if not average_repeats:
        return [
            {
                "amp_pA": float(row["amp_pA"]),
                "spike_frequency_hz": float(row["spike_frequency_hz"]),
                "n_sweeps": 1,
                "sweeps": str(row["sweep"]),
            }
            for row in sorted(rows, key=lambda item: (float(item["amp_pA"]), int(item["sweep"])))
        ]

    grouped: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["amp_pA"])].append(row)

    out = []
    for amp_pA in sorted(grouped):
        group = grouped[amp_pA]
        freqs = np.asarray([float(row["spike_frequency_hz"]) for row in group], dtype=float)
        out.append(
            {
                "amp_pA": amp_pA,
                "spike_frequency_hz": float(np.nanmean(freqs)),
                "n_sweeps": len(group),
                "sweeps": ",".join(str(int(row["sweep"])) for row in group),
            }
        )
    return out


def write_allen_nwb_fi_target_csv(
    nwb_path: str | Path,
    target_path: str | Path,
    *,
    summary_path: Optional[str | Path] = None,
    stimulus_names: Sequence[str] = DEFAULT_FI_STIMULUS_NAMES,
    include_negative_currents: bool = False,
    min_current_pA: Optional[float] = 0.0,
    max_current_pA: Optional[float] = None,
    average_repeats: bool = True,
    spike_threshold_mV: float = -20.0,
    refractory_ms: float = 1.0,
) -> dict[str, Any]:
    """Write ACT `target_sf.csv` from an Allen NWB file and return metadata."""
    sweep_rows = extract_allen_nwb_fi_sweeps(
        nwb_path,
        stimulus_names=stimulus_names,
        include_negative_currents=include_negative_currents,
        min_current_pA=min_current_pA,
        max_current_pA=max_current_pA,
        spike_threshold_mV=spike_threshold_mV,
        refractory_ms=refractory_ms,
    )
    fi_rows = aggregate_fi_sweeps(sweep_rows, average_repeats=average_repeats)
    target_rows = [
        {
            "mean_i": float(row["amp_pA"]) / 1000.0,
            "spike_frequency": float(row["spike_frequency_hz"]),
        }
        for row in fi_rows
    ]
    _write_rows_csv(target_path, target_rows)
    if summary_path is not None:
        _write_rows_csv(summary_path, fi_rows)
    return {
        "target_path": str(Path(target_path)),
        "summary_path": str(Path(summary_path)) if summary_path is not None else None,
        "n_sweeps": len(sweep_rows),
        "n_targets": len(fi_rows),
        "currents_pA": [float(row["amp_pA"]) for row in fi_rows],
        "frequencies_hz": [float(row["spike_frequency_hz"]) for row in fi_rows],
    }


def _open_nwb(nwb_path: str | Path):
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Reading Allen NWB target data requires h5py. Install h5py or use "
            "ACT_TARGET_MODE='fi_csv' with a pre-extracted target CSV."
        ) from exc
    path = Path(nwb_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Allen NWB file not found: {path}")
    return h5py.File(path, "r")


def _sorted_sweep_keys(keys: Sequence[str]) -> list[str]:
    return sorted(keys, key=_sweep_number)


def _sweep_number(key: str) -> int:
    try:
        return int(str(key).split("_")[-1])
    except ValueError:
        return 10**9


def _sample_rate_hz(*groups: Any) -> float:
    for group in groups:
        if "starting_time" in group and "rate" in group["starting_time"].attrs:
            return float(group["starting_time"].attrs["rate"])
    raise KeyError("Could not find starting_time.attrs['rate'] for NWB sweep.")


def _read_float(dataset: Any) -> float:
    if dataset is None:
        return float("nan")
    value = _read_scalar(dataset)
    return float(value)


def _read_text(dataset: Any) -> str:
    if dataset is None:
        return ""
    value = _read_scalar(dataset)
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _read_scalar(dataset: Any) -> Any:
    value = dataset[()] if hasattr(dataset, "shape") else dataset
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return value


def _voltage_mV(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values * 1000.0 if np.nanmax(np.abs(values)) < 1.0 else values


def _current_pA(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values * 1e12 if np.nanmax(np.abs(values)) < 1e-3 else values


def _active_current_window(current_pA: np.ndarray, amp_pA: float, sample_rate_hz: float) -> tuple[int, int]:
    if np.isfinite(amp_pA) and abs(amp_pA) >= 1.0:
        tolerance = max(2.0, min(20.0, abs(amp_pA) * 0.10))
        idx = np.where(np.abs(current_pA - amp_pA) <= tolerance)[0]
        if idx.size:
            return _longest_contiguous_index_block(idx)

    baseline = float(np.nanmedian(current_pA[: max(1, int(0.05 * sample_rate_hz))]))
    delta = current_pA - baseline
    threshold = max(2.0, 0.05 * float(np.nanmax(np.abs(delta))))
    idx = np.where(np.abs(delta) > threshold)[0]
    if idx.size:
        return _longest_contiguous_index_block(idx)
    return 0, len(current_pA)


def _longest_contiguous_index_block(indices: np.ndarray) -> tuple[int, int]:
    splits = np.where(np.diff(indices) > 1)[0]
    starts = np.r_[0, splits + 1]
    ends = np.r_[splits, len(indices) - 1]
    lengths = ends - starts + 1
    winner = int(np.argmax(lengths))
    return int(indices[starts[winner]]), int(indices[ends[winner]]) + 1


def _count_threshold_crossings(
    voltage_mV: np.ndarray,
    start_idx: int,
    stop_idx: int,
    *,
    threshold_mV: float,
    refractory_ms: float,
    sample_rate_hz: float,
) -> int:
    trace = voltage_mV[start_idx:stop_idx]
    if len(trace) < 2:
        return 0
    above = trace >= threshold_mV
    crossings = np.where((~above[:-1]) & above[1:])[0] + start_idx + 1
    if crossings.size == 0:
        return 0

    refractory_samples = max(1, int(refractory_ms / 1000.0 * sample_rate_hz))
    kept = 0
    previous = -10**12
    for crossing in crossings:
        if int(crossing) - previous >= refractory_samples:
            kept += 1
            previous = int(crossing)
    return kept


def _write_rows_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path_obj.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path_obj
