"""Block-based timing resolution for synaptic input generation.

Synapse groups expose a user-facing ``input_blocks`` list. Each block defines
its own simulation window, input mode, and source/rate options. This module
validates those blocks, inserts implicit quiescent gaps for metadata, and
returns the normalized ``time_cfg`` consumed by Step 5 input generation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence


def _calculate_timing(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
    *,
    block_jitter_delays_ms: Optional[Mapping[Any, float]] = None,
) -> Dict[str, Any]:
    """Return normalized timing metadata for one synapse group."""

    sim_tstart = _as_float(sim_cfg.get("tstart"), "sim_config['tstart']")
    sim_tstop = _as_float(sim_cfg.get("tstop"), "sim_config['tstop']")
    if sim_tstop <= sim_tstart:
        raise ValueError(
            f"sim_config['tstop'] must be greater than tstart "
            f"(got tstart={sim_tstart}, tstop={sim_tstop})"
        )

    input_blocks = _normalize_input_blocks(
        sim_tstart=sim_tstart,
        sim_tstop=sim_tstop,
        group_cfg=group_cfg,
        block_jitter_delays_ms=block_jitter_delays_ms,
    )
    blocks = _insert_quiescent_gaps(
        sim_tstart=sim_tstart,
        sim_tstop=sim_tstop,
        input_blocks=input_blocks,
    )

    active_blocks = [block for block in input_blocks if block.get("kind") != "quiescent"]
    anchors: Dict[str, Any] = {
        "sim_tstart": sim_tstart,
        "sim_tstop": sim_tstop,
        "jitter_tstart_ms": _as_float_or_none(
            sim_cfg.get("_jitter_tstart_ms"),
            "sim_config['_jitter_tstart_ms']",
        ),
        "input_window_tstart": min((b["t_start"] for b in active_blocks), default=None),
        "input_window_tstop": max((b["t_end"] for b in active_blocks), default=None),
        "block_count": len(active_blocks),
    }

    return {
        "anchors": anchors,
        "blocks": blocks,
        "input_blocks": input_blocks,
    }


def _normalize_input_blocks(
    *,
    sim_tstart: float,
    sim_tstop: float,
    group_cfg: Dict[str, Any],
    block_jitter_delays_ms: Optional[Mapping[Any, float]] = None,
) -> List[Dict[str, Any]]:
    raw_blocks = group_cfg.get("input_blocks", []) or []
    if not isinstance(raw_blocks, list):
        raise ValueError("group input_blocks must be a list of block dictionaries.")

    normalized: List[Dict[str, Any]] = []
    for idx, raw_block in enumerate(raw_blocks):
        if not isinstance(raw_block, dict):
            raise ValueError(f"input_blocks[{idx}] must be a dictionary.")
        if raw_block.get("state", True) is False:
            continue
        block = _normalize_one_input_block(
            sim_tstart=sim_tstart,
            sim_tstop=sim_tstop,
            raw_block=raw_block,
            idx=idx,
            block_jitter_delays_ms=block_jitter_delays_ms,
        )
        if block is not None:
            normalized.append(block)

    normalized.sort(key=lambda block: (block["t_start"], block["t_end"], block["name"]))
    _validate_no_overlaps(normalized)
    return normalized


def _normalize_one_input_block(
    *,
    sim_tstart: float,
    sim_tstop: float,
    raw_block: Dict[str, Any],
    idx: int,
    block_jitter_delays_ms: Optional[Mapping[Any, float]],
) -> Optional[Dict[str, Any]]:
    name = str(raw_block.get("name") or f"block_{idx + 1}")
    mode = raw_block.get("mode")
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError(f"input block {name!r} is missing a string 'mode'.")
    mode = mode.strip()

    start = _as_float(raw_block.get("start_ms"), f"input block {name!r} start_ms")
    stop = _as_float(raw_block.get("stop_ms"), f"input block {name!r} stop_ms")
    if stop <= start:
        raise ValueError(
            f"input block {name!r} must have stop_ms > start_ms "
            f"(got {start} -> {stop})."
        )
    if start < sim_tstart or stop > sim_tstop:
        raise ValueError(
            f"input block {name!r} must lie within the simulation window "
            f"[{sim_tstart}, {sim_tstop}] ms (got {start} -> {stop})."
        )

    jitter_ms = _as_float_or_none(raw_block.get("jitter_ms"), f"input block {name!r} jitter_ms")
    if jitter_ms is not None and jitter_ms < 0.0:
        raise ValueError(f"input block {name!r} jitter_ms must be >= 0.")

    delay = _lookup_block_jitter_delay(block_jitter_delays_ms, idx=idx, name=name)
    if delay is not None:
        if delay < 0.0:
            raise ValueError(f"resolved jitter delay for input block {name!r} must be >= 0.")
        start = start + delay
        stop = min(stop + delay, sim_tstop)
        if stop <= start:
            return None

    role = str(raw_block.get("role") or _infer_block_role(mode, name)).strip().lower()
    source = dict(raw_block.get("source") or {})
    rate_hz = raw_block.get("rate_hz", source.get("freq"))
    duration = float(stop - start)

    if mode == "homogeneous_poisson":
        if rate_hz is None:
            raise ValueError(
                f"homogeneous input block {name!r} requires rate_hz "
                "or source.freq."
            )
        source["freq"] = _as_float(rate_hz, f"input block {name!r} rate_hz")
    elif mode in {"inhomogeneous_poisson", "precomputed"}:
        source = _normalize_source_crop(
            source=source,
            duration_ms=duration,
            block_name=name,
        )

    kind = "baseline" if mode == "homogeneous_poisson" and role in {"baseline", "background"} else "source"

    block: Dict[str, Any] = {
        "name": name,
        "role": role,
        "kind": kind,
        "mode": mode,
        "t_start": float(start),
        "t_end": float(stop),
        "start_ms": float(start),
        "stop_ms": float(stop),
        "source": source,
    }
    if rate_hz is not None:
        block["rate_hz"] = source.get("freq", rate_hz)
    if jitter_ms is not None:
        block["jitter_ms"] = float(jitter_ms)
    if delay is not None:
        block["jitter_delay_ms"] = float(delay)
    return block


def _normalize_source_crop(
    *,
    source: Dict[str, Any],
    duration_ms: float,
    block_name: str,
) -> Dict[str, Any]:
    crop_start = _as_float_or_none(
        source.get("crop_start_ms"),
        f"input block {block_name!r} source.crop_start_ms",
    )
    crop_stop = _as_float_or_none(
        source.get("crop_stop_ms"),
        f"input block {block_name!r} source.crop_stop_ms",
    )
    if crop_start is None:
        crop_start = 0.0
    if crop_start < 0.0:
        raise ValueError(f"input block {block_name!r} source.crop_start_ms must be >= 0.")
    if crop_stop is None:
        crop_stop = crop_start + duration_ms
    if crop_stop <= crop_start:
        raise ValueError(
            f"input block {block_name!r} source.crop_stop_ms must be greater than crop_start_ms."
        )

    crop_duration = crop_stop - crop_start
    if abs(crop_duration - duration_ms) > 1e-6:
        raise ValueError(
            f"input block {block_name!r} source crop duration must match block duration "
            f"for v1 configs (crop={crop_duration} ms, block={duration_ms} ms)."
        )

    source["crop_start_ms"] = float(crop_start)
    source["crop_stop_ms"] = float(crop_stop)
    return source


def _insert_quiescent_gaps(
    *,
    sim_tstart: float,
    sim_tstop: float,
    input_blocks: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    cursor = sim_tstart

    for block in input_blocks:
        start = float(block["t_start"])
        stop = float(block["t_end"])
        if start > cursor:
            blocks.append(
                {
                    "name": f"quiescent_{len(blocks) + 1}",
                    "role": "quiescent",
                    "kind": "quiescent",
                    "mode": "quiescent",
                    "t_start": float(cursor),
                    "t_end": float(start),
                    "start_ms": float(cursor),
                    "stop_ms": float(start),
                }
            )
        blocks.append(dict(block))
        cursor = max(cursor, stop)

    if cursor < sim_tstop:
        blocks.append(
            {
                "name": f"quiescent_{len(blocks) + 1}",
                "role": "quiescent",
                "kind": "quiescent",
                "mode": "quiescent",
                "t_start": float(cursor),
                "t_end": float(sim_tstop),
                "start_ms": float(cursor),
                "stop_ms": float(sim_tstop),
            }
        )
    return blocks


def _validate_no_overlaps(blocks: Sequence[Dict[str, Any]]) -> None:
    previous: Optional[Dict[str, Any]] = None
    for block in blocks:
        if previous is not None and float(block["t_start"]) < float(previous["t_end"]) - 1e-9:
            raise ValueError(
                "input_blocks must not overlap: "
                f"{previous['name']!r} ends at {previous['t_end']} ms, "
                f"but {block['name']!r} starts at {block['t_start']} ms."
            )
        previous = block


def _lookup_block_jitter_delay(
    delays: Optional[Mapping[Any, float]],
    *,
    idx: int,
    name: str,
) -> Optional[float]:
    if not delays:
        return None
    if idx in delays:
        return float(delays[idx])
    if name in delays:
        return float(delays[name])
    return None


def _infer_block_role(mode: str, name: str) -> str:
    name_l = name.lower()
    if "baseline" in name_l:
        return "baseline"
    if "background" in name_l or name_l.startswith("bg_"):
        return "background"
    if "stim" in name_l or mode != "homogeneous_poisson":
        return "stimulus"
    return "input"


def _as_float(val: Any, label: str) -> float:
    if val is None:
        raise ValueError(f"{label} must be numeric, not null.")
    try:
        return float(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric (got {val!r}).") from exc


def _as_float_or_none(val: Any, label: str) -> Optional[float]:
    if val is None:
        return None
    return _as_float(val, label)
