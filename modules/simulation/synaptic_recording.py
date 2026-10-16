"""Bounded, opt-in recording of native point-process synaptic variables.

Population currents/conductances are exact sums, not sampled estimates, and
native values already include the mechanism's own weight semantics. Probability
and utilization states are averaged instead. Placement metadata remains in the
separate ``syn_records`` result keys.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import math
from typing import Any, Dict

import numpy as np
from neuron import h

from scp_pipeline.input_generation.recording_config import (
    normalize_syn_recording_config,
    resolve_n_syn_recording_trials,
)


SYNAPTIC_RECORDING_UNITS = {
    "i": "nA", "g": "uS", "i_AMPA": "nA", "i_NMDA": "nA",
    "g_AMPA": "uS", "g_NMDA": "uS", "record_use": "1", "record_Pr": "1",
}
_MEAN_VARIABLES = {"record_use", "record_Pr"}


def synaptic_recording_nbytes(value: Any) -> int:
    """Count retained array bytes once; excludes Python/container overhead."""
    seen: set[int] = set()

    def count(item: Any) -> int:
        if id(item) in seen:
            return 0
        seen.add(id(item))
        if isinstance(item, np.ndarray):
            return int(item.nbytes)
        if isinstance(item, dict):
            return sum(count(child) for child in item.values())
        if isinstance(item, (list, tuple)):
            return sum(count(child) for child in item)
        return 0

    return count(value)


def _record_field(record: Any, field: str, default: Any = None) -> Any:
    return record.get(field, default) if isinstance(record, dict) else getattr(record, field, default)


def _reference(synapse: Any, variable: str, context: str) -> Any:
    """Skip an absent optional variable, but never suppress a broken POINTER."""
    try:
        ref = getattr(synapse, f"_ref_{variable}")
    except AttributeError as missing_ref:
        # Some native mechanisms expose a scalar variable but cannot supply a
        # reference until a POINTER is initialized. That is not an absent var.
        try:
            getattr(synapse, variable)
        except AttributeError:
            return None
        except Exception as exc:
            raise RuntimeError(f"Cannot access {context}; check native pointer initialization.") from exc
        raise RuntimeError(f"Cannot resolve {context}; exposed variable has no valid native reference.") from missing_ref
    except Exception as exc:
        raise RuntimeError(f"Cannot access {context}; check native pointer initialization.") from exc
    if ref is None:
        raise RuntimeError(f"Cannot resolve {context}; native reference is null.")
    return ref


def _sample_indices(group: str, seed: int, n_synapses: int, n_sample: int) -> list[int]:
    # Stable hash ranking never touches any RNG stream and needs only O(sample)
    # scratch space (a full random permutation would allocate O(population)).
    digest = hashlib.sha256(f"{seed}:{len(group)}:{group}".encode("utf-8")).digest()
    count = min(n_synapses, n_sample)
    return sorted(heapq.nsmallest(
        count, range(n_synapses),
        key=lambda index: hashlib.sha256(digest + index.to_bytes(8, "little")).digest(),
    ))


def _synapse_groups(cell: Any, syn_state: Dict[str, Any] | None) -> tuple[Dict[str, Any], Dict[str, Any], str]:
    if syn_state is not None:
        return syn_state.get("synapses", {}), syn_state.get("records", {}), "syn_state.records"
    return {"unmapped": list(getattr(cell, "synapses", []) or [])}, {}, "cell.synapses positional (group unavailable)"


@dataclass
class SynapticRecorders:
    payload: Dict[str, Any]
    vectors: Dict[str, Dict[tuple[str, int], Any]]
    time_vector: Any


def prepare_synaptic_recording(
    cell: Any,
    sim_cfg: Dict[str, Any],
    *,
    syn_state: Dict[str, Any] | None = None,
    trial_idx: int = 0,
    retained_bytes: int = 0,
) -> SynapticRecorders | None:
    """Discover references and budget all buffers before allocating a Vector.

    The budget covers this recorder's estimated peak plus prior retained
    synaptic arrays, not whole-process RSS. Native buffers receive a 2x capacity
    allowance; outputs, conversion scratch, and per-vector overhead are included.
    An exact aggregate intentionally needs every available population vector.
    """
    cap = resolve_n_syn_recording_trials(sim_cfg)
    if trial_idx < 0 or trial_idx >= cap:
        return None
    cfg = normalize_syn_recording_config(sim_cfg)
    dt = float(sim_cfg.get("dt", 0.025))
    interval = cfg["record_dt_ms"] or dt
    tstart = float(sim_cfg.get("tstart", 0.0))
    tstop = float(sim_cfg["tstop"])
    if not all(math.isfinite(number) for number in (dt, interval, tstart, tstop)) or dt <= 0 or tstop < tstart:
        raise ValueError("Synaptic recording requires finite times, dt > 0, and tstop >= tstart")
    # Initialization starts at zero before run_cell applies tstart. Include
    # both possible scheduling spans plus endpoint slack in the estimate.
    n_points = int(math.ceil(max(0.0, tstop, tstop - tstart) / interval)) + 2
    groups, records, identity_source = _synapse_groups(cell, syn_state)
    payload: Dict[str, Any] = {
        "format_version": 1,
        "T": None,
        "units": {},
        "groups": {},
        "record_dt_ms": interval,
        "sample_seed": cfg["sample_seed"],
        "identity_source": identity_source,
    }
    references: Dict[str, Dict[tuple[str, int], tuple[Any, str]]] = {}
    n_output_series = 1  # independent time array
    fallback_id = 0
    for name, synapses_raw in groups.items():
        synapses = list(synapses_raw)
        n_synapses = len(synapses)
        group_records = list(records.get(name, []) or [])
        ids = [
            int(_record_field(group_records[index], "syn_id", fallback_id + index))
            if index < len(group_records) else fallback_id + index
            for index in range(n_synapses)
        ]
        fallback_id += n_synapses
        override = cfg["groups"].get(name, {})
        if not override.get("enabled", True) or not n_synapses:
            continue
        mode = override.get("mode", cfg["default_mode"])
        variables = override.get("vars", cfg["default_vars"])
        sample_count = override.get("sample_per_group", cfg["default_sample_per_group"])
        indices = _sample_indices(name, cfg["sample_seed"], n_synapses, sample_count) if mode in {"sample", "both"} else []
        sampled = set(indices)
        group: Dict[str, Any] = {
            "mode": mode,
            "n_synapses": n_synapses,
            "synapse_ids": [ids[index] for index in indices],
            "sample_indices": indices,
            "aggregate": {},
            "samples": {},
            "coverage": {},
            "aggregation": {},
            "sample_metadata": [
                {field: _record_field(group_records[index], field)
                 for field in ("syn_id", "type", "section", "x", "distance", "weight")}
                if index < len(group_records) else {"syn_id": ids[index], "group_index": index}
                for index in indices
            ],
        }
        refs: Dict[tuple[str, int], tuple[Any, str]] = {}
        for variable, enabled in variables.items():
            if not enabled:
                continue
            payload["units"][variable] = SYNAPTIC_RECORDING_UNITS[variable]
            available: Dict[int, tuple[Any, str]] = {}
            for index, synapse in enumerate(synapses):
                record = group_records[index] if index < len(group_records) else None
                mechanism = _record_field(record, "type", type(synapse).__name__)
                context = f"synaptic variable {variable!r}, group {name!r}, syn_id={ids[index]}, mechanism {mechanism!r}"
                ref = _reference(synapse, variable, context)
                if ref is not None:
                    available[index] = (ref, context)
            complete = len(available) == n_synapses
            aggregate = mode in {"group", "both"} and complete
            n_sample_available = sum(index in available for index in indices)
            group["coverage"][variable] = {
                "available_count": len(available),
                "missing_synapse_ids": [ids[index] for index in range(n_synapses) if index not in available],
                "aggregate_count": n_synapses if aggregate else 0,
                "sample_count": n_sample_available,
                "complete": complete,
            }
            if mode in {"group", "both"}:
                group["aggregation"][variable] = "mean" if variable in _MEAN_VARIABLES else "sum"
            if aggregate:
                n_output_series += 1
            if n_sample_available:
                n_output_series += len(indices)  # missing sample rows remain NaN, not zero
            for index, reference in available.items():
                if aggregate or index in sampled:
                    refs[(variable, index)] = reference
        references[name] = refs
        payload["groups"][name] = group

    if not payload["groups"]:
        return None
    n_vectors = 1 + sum(len(refs) for refs in references.values())
    # Include one conversion scratch series and a small per-vector allowance.
    estimated_bytes = int(retained_bytes) + 8 * n_points * (2 * n_vectors + n_output_series + 1) + 512 * n_vectors
    maximum_bytes = int(cfg["max_memory_mb"] * 1024 * 1024)
    if estimated_bytes > maximum_bytes:
        raise MemoryError(
            f"Synaptic recording estimated peak {estimated_bytes / 1024**2:.3f} MiB exceeds "
            f"syn_recording.max_memory_mb={cfg['max_memory_mb']:g}. Increase record_dt_ms, "
            "reduce variables/trials, select sample mode or fewer groups, or explicitly raise the budget. "
            "Exact group aggregates require every population synapse."
        )
    payload["memory"] = {
        "estimated_peak_bytes": estimated_bytes,
        "retained_bytes": int(retained_bytes),
        "max_memory_bytes": maximum_bytes,
        "estimated_samples_per_trace": n_points,
        "native_vector_count": n_vectors,
        "native_capacity_factor": 2,
        "scope": "synaptic recording arrays and native buffers; not process RSS",
    }
    vectors: Dict[str, Dict[tuple[str, int], Any]] = {}
    recording_interval = () if cfg["record_dt_ms"] is None else (interval,)
    time_vector = h.Vector().record(h._ref_t, *recording_interval)
    for name, refs in references.items():
        vectors[name] = {}
        for key, (reference, context) in refs.items():
            try:
                vectors[name][key] = h.Vector().record(reference, *recording_interval)
            except Exception as exc:
                raise RuntimeError(f"Cannot record {context}; check native pointer initialization and mechanism lifetime.") from exc
    return SynapticRecorders(payload=payload, vectors=vectors, time_vector=time_vector)


def finish_synaptic_recording(recorders: SynapticRecorders | None) -> Dict[str, Any] | None:
    """Detach output arrays, retaining no native vectors or point processes."""
    if recorders is None:
        return None
    payload = recorders.payload
    times = np.array(recorders.time_vector, dtype=float)
    payload["T"] = times
    for name, group in payload["groups"].items():
        vectors = recorders.vectors[name]
        for variable, coverage in group["coverage"].items():
            aggregate = None
            if coverage["aggregate_count"]:
                aggregate = np.zeros(times.size, dtype=float)
            sample = None
            if coverage["sample_count"]:
                sample = np.full((len(group["sample_indices"]), times.size), np.nan, dtype=float)
            rows = {index: row for row, index in enumerate(group["sample_indices"])}
            for (name_var, index), vector in vectors.items():
                if name_var != variable:
                    continue
                # A view avoids a second population-sized copy; all retained
                # outputs are separately allocated before native buffers die.
                values = vector.as_numpy() if hasattr(vector, "as_numpy") else np.asarray(vector, dtype=float)
                if values.size != times.size:
                    raise RuntimeError(f"Synaptic recording time mismatch for group {name!r}, variable {variable!r}")
                if aggregate is not None:
                    aggregate += values
                if sample is not None and index in rows:
                    sample[rows[index]] = values
            if aggregate is not None:
                if group["aggregation"][variable] == "mean":
                    aggregate /= coverage["aggregate_count"]
                group["aggregate"][variable] = aggregate
            if sample is not None:
                group["samples"][variable] = sample
    return payload
