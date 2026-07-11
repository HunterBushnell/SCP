"""Per-group input processing orchestration.

This layer bridges normalized configs, timing resolution, mode handlers, and
GroupInputs packaging. The public entry point remains inputs.generate_inputs().
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import hashlib
import json
import numpy as np

from modules.core import randomness

from . import modes_core as input_modes_core
from .density import _lognormal_mu_sigma, _resolve_n_syn
from .timing import _calculate_timing
from .types import GroupInputs


def _init_rng(
    rng: Optional[np.random.Generator],
    sim_cfg: Dict[str, Any],
) -> np.random.Generator:
    """
    Initialize RNG.

    Priority:
      1) If an explicit Generator is passed in, use it as-is.
      2) Else, if sim_cfg['seed'] is not None, use it to seed a new Generator.
      3) Else, create an unseeded Generator (non-deterministic).
    """
    if rng is not None:
        return rng

    seed = sim_cfg.get("seed", None)
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


def _make_group_signature(
    sim_cfg: Dict[str, Any],
    group_name: str,
    group_cfg: Dict[str, Any],
) -> str:
    """
    Build a stable signature string for a (sim_cfg, group_cfg) pair.

    Used as a key into an inputs cache so that if the sim/group
    parameters are unchanged, we can reuse previously generated
    spike trains instead of regenerating.

    Notes:
      - We include only JSON-serializable pieces; anything non-serializable
        is converted via str(...).
      - Geometry is deliberately NOT included here; if you want geometry-
        specific caching later, you can add a geometry identifier.
    """
    # Keep only the sim fields that actually affect inputs
    sim_subset = {
        "tstart": sim_cfg.get("tstart"),
        "tstop": sim_cfg.get("tstop"),
        "dt": sim_cfg.get("dt"),
        "seed": sim_cfg.get("seed"),
        "jitter": sim_cfg.get("jitter"),
    }

    payload = {
        "sim": sim_subset,
        "group_name": group_name,
        "group_cfg": group_cfg,
    }

    # JSON with sorted keys for stability; default=str to survive odd types
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _build_default_mode_registry() -> Dict[str, Any]:
    """
    Build the default mode registry from the core module.

    This can be extended by user-defined modes at the notebook level.
    """
    return input_modes_core.get_default_mode_registry()


def _resolve_block_jitter_delays_ms(
    sim_cfg: Dict[str, Any],
    group_name: str,
    group_cfg: Dict[str, Any],
    trial_rng: Optional[randomness.TrialRandomness] = None,
) -> Dict[int, float]:
    """Resolve optional per-block jitter delays from ``block['jitter_ms']``."""
    delays: Dict[int, float] = {}
    for idx, block in enumerate(group_cfg.get("input_blocks", []) or []):
        if not isinstance(block, dict) or block.get("state", True) is False:
            continue
        jitter_raw = block.get("jitter_ms")
        if jitter_raw is None:
            continue
        try:
            jitter_ms = float(jitter_raw)
        except Exception as exc:
            raise ValueError(
                f"Group '{group_name}' block {idx}: jitter_ms must be numeric or null "
                f"(got {jitter_raw!r})"
            ) from exc
        if jitter_ms < 0.0:
            raise ValueError(
                f"Group '{group_name}' block {idx}: jitter_ms must be >= 0 "
                f"(got {jitter_ms!r})"
            )
        if jitter_ms <= 0.0:
            continue

        block_name = str(block.get("name") or f"block_{idx + 1}")
        stream = f"block_jitter:{block_name}"
        if trial_rng is not None:
            jitter_rng = trial_rng.rng("inputs", group=group_name, stream=stream)
        else:
            seed = sim_cfg.get("seed", None)
            if seed is None:
                jitter_rng = np.random.default_rng()
            else:
                block_hash = randomness.stable_u32_from_str(
                    f"{group_name}:{block_name}",
                    salt="block_jitter:",
                )
                jitter_rng = np.random.default_rng(int(seed) ^ int(block_hash) ^ 0x5A17A5A1)

        mu, sig = _lognormal_mu_sigma(jitter_ms, jitter_ms)
        delays[idx] = max(0.0, float(jitter_rng.lognormal(mu, sig)) if sig > 0 else jitter_ms)
    return delays


def _process_all_groups(
    sim_cfg: Dict[str, Any],
    groups_cfg: Dict[str, Dict[str, Any]],
    geometry: Optional[Any],
    mode_registry: Mapping[str, Any],
    rng: Optional[np.random.Generator],
    trial_rng: Optional[randomness.TrialRandomness] = None,
    # cache: Optional[Dict[str, GroupInputs]] = None,
) -> Dict[str, GroupInputs]:
    """
    Loop over all groups and generate per-group inputs.

    High-level steps (per group):
      1) decide whether to skip the group (inactive / invalid)
      2) resolve N_syn (including geometry/density when needed)
      3) compute timing configuration for this group
      4) resolve the mode handler from mode_registry
      5) run the handler to obtain spike trains (or reuse from cache)
      6) package into GroupInputs
    """
    inputs: Dict[str, GroupInputs] = {}

    # Optional: per-trial jitter of input start (global), used to delay baseline onset
    jitter_tstart: Optional[float] = None
    jitter_std = sim_cfg.get("jitter", None)
    if jitter_std is not None:
        try:
            jitter_std = float(jitter_std)
        except Exception as exc:
            raise ValueError(f"sim['jitter'] must be numeric or null (got {jitter_std!r})") from exc
        if jitter_std > 0:
            if trial_rng is not None:
                jitter_rng = trial_rng.rng("inputs", stream="jitter")
            else:
                seed = sim_cfg.get("seed", None)
                if seed is None:
                    jitter_rng = np.random.default_rng()
                else:
                    jitter_rng = np.random.default_rng(int(seed) ^ 0xA5A5A5A5)

            tstart = float(sim_cfg.get("tstart", 0.0))
            tstop = float(sim_cfg.get("tstop", tstart))
            mu, sig = _lognormal_mu_sigma(jitter_std, jitter_std)
            jitter_offset = float(jitter_rng.lognormal(mu, sig)) if sig > 0 else jitter_std
            jitter_offset = max(0.0, jitter_offset)
            jitter_tstart = tstart + jitter_offset
            if tstop > tstart:
                jitter_tstart = min(jitter_tstart, tstop)
            sim_cfg["_jitter_delay_ms"] = jitter_offset

    # Stash for timing calculation (used by modes)
    sim_cfg["_jitter_tstart_ms"] = jitter_tstart

    for gname, gcfg in groups_cfg.items():
        # 1) Skip inactive or invalid groups
        if _should_skip_group(gname, gcfg):
            continue

        # OPTIONAL: cache lookup before doing any heavy work
        # sig: Optional[str] = None
        # if cache is not None:
        #     sig = _make_group_signature(sim_cfg, gname, gcfg)
        #     cached = cache.get(sig)
        #     if cached is not None:
        #         inputs[gname] = cached
        #         continue

        # 2) Resolve number of synapses for this group
        #    This also stashes syns["N_syn_resolved"] into the group config.
        n_syn_resolved = _resolve_n_syn(
            sim_cfg=sim_cfg,
            group_cfg=gcfg,
            geometry=geometry,
        )
        if n_syn_resolved < 0:
            raise ValueError(
                f"Group '{gname}': resolved N_syn_resolved < 0 ({n_syn_resolved})."
            )

        # 3) Generate timing configuration for this group
        block_jitter_delays_ms = _resolve_block_jitter_delays_ms(
            sim_cfg=sim_cfg,
            group_name=gname,
            group_cfg=gcfg,
            trial_rng=trial_rng,
        )
        time_cfg = _calculate_timing(
            sim_cfg,
            gcfg,
            block_jitter_delays_ms=block_jitter_delays_ms,
        )
        gcfg["time_cfg"] = time_cfg  # Attach for modes and downstream consumers

        # 4) Choose RNG for this group and run handler(s) to get spike trains
        mode_name = gcfg.get("mode")
        setting_path = "inputs"
        group_rng = rng
        rand_meta = None
        if trial_rng is not None:
            if mode_name:
                sentinel = object()
                mode_setting = trial_rng.setting(f"modes.{mode_name}", sentinel)
                if mode_setting is not sentinel:
                    setting_path = f"modes.{mode_name}"
            group_rng = trial_rng.rng(
                setting_path,
                group=gname,
                stream=mode_name or "inputs",
            )
            rand_meta = {
                "trial_idx": getattr(trial_rng, "trial_idx", None),
                "setting_path": setting_path,
                "group": gname,
                "stream": mode_name or "inputs",
            }
        elif group_rng is None:
            group_rng = _init_rng(None, sim_cfg)

        if _uses_input_blocks(gcfg):
            spike_trains = _run_input_blocks(
                sim_cfg=sim_cfg,
                group_name=gname,
                group_cfg=gcfg,
                geometry=geometry,
                mode_registry=mode_registry,
                rng=group_rng,
            )
        else:
            handler = _resolve_mode_handler(gname, gcfg, mode_registry)
            spike_trains = _run_mode_handler(
                handler=handler,
                sim_cfg=sim_cfg,
                group_name=gname,
                group_cfg=gcfg,
                geometry=geometry,
                rng=group_rng,
            )

        # Optional consistency check: number of trains vs N_syn_resolved
        if len(spike_trains) != n_syn_resolved:
            raise ValueError(
                f"Group '{gname}': handler returned {len(spike_trains)} trains, "
                f"but N_syn_resolved={n_syn_resolved}."
            )

        # 6) Package into GroupInputs
        group_inputs = _build_group_inputs(
            group_name=gname,
            group_cfg=gcfg,
            spike_trains=spike_trains,
            randomness_meta=rand_meta,
        )

        inputs[gname] = group_inputs

        # Store into cache if enabled
        # if cache is not None and sig is not None:
        #     cache[sig] = group_inputs

    return inputs


def _should_skip_group(
    group_name: str,
    group_cfg: Dict[str, Any],
) -> bool:
    """
    Decide whether to skip this group.

    Rules:
      - If group_cfg['state'] is explicitly False, skip it;
      - If 'mode' is missing or empty, skip it;
      - Otherwise, keep it.
    """
    state = group_cfg.get("state", True)
    if state is False:
        return True

    if _uses_input_blocks(group_cfg):
        return False

    mode = group_cfg.get("mode")
    if mode is None or (isinstance(mode, str) and mode.strip() == ""):
        return True

    return False


def _uses_input_blocks(group_cfg: Dict[str, Any]) -> bool:
    blocks = group_cfg.get("input_blocks", []) or []
    return isinstance(blocks, list) and len(blocks) > 0


def _resolve_mode_handler(
    group_name: str,
    group_cfg: Dict[str, Any],
    mode_registry: Mapping[str, Any],
) -> Any:
    """
    Look up the mode handler for this group.

    - Reads group_cfg['mode'] (must be a string).
    - Looks up mode_registry[mode_name].
    - Raises a clear error if the mode is unknown.
    """
    mode_name = group_cfg.get("mode")
    handler = mode_registry.get(mode_name)
    if handler is None:
        raise ValueError(
            f"Group '{group_name}' specifies unknown mode {mode_name!r}; "
            f"make sure it is registered in mode_registry."
        )
    return handler


def _run_mode_handler(
    handler: Any,
    sim_cfg: Dict[str, Any],
    group_name: str,
    group_cfg: Dict[str, Any],
    geometry: Optional[Any],
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Call the mode handler to obtain spike trains.

    Contract:
      - handler(sim_cfg, group_cfg, geometry, rng) -> list[np.ndarray]
      - each array is 1D spike times in ms, in simulation time.

    Note:
      - group_cfg may contain a 'time_cfg' key if timing has been
        resolved upstream; modes are encouraged to use it instead of
        reinterpreting the raw timing fields.
    """
    spike_trains = handler(sim_cfg, group_cfg, geometry, rng)

    if not isinstance(spike_trains, list):
        raise TypeError(
            f"Mode handler for group '{group_name}' must return a list of np.ndarray, "
            f"got {type(spike_trains)!r}"
        )

    for i, arr in enumerate(spike_trains):
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"Mode handler for group '{group_name}' returned element {i} "
                f"of type {type(arr)!r}, expected np.ndarray."
            )

    return spike_trains


def _run_input_blocks(
    *,
    sim_cfg: Dict[str, Any],
    group_name: str,
    group_cfg: Dict[str, Any],
    geometry: Optional[Any],
    mode_registry: Mapping[str, Any],
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Run each normalized input block and merge spikes per synapse."""
    syns = group_cfg.get("syns", {}) or {}
    n_syn = int(syns.get("N_syn_resolved", syns.get("N_syn", 0)) or 0)
    trains_accum: List[List[float]] = [[] for _ in range(max(n_syn, 0))]

    time_cfg = group_cfg.get("time_cfg", {}) or {}
    input_blocks = time_cfg.get("input_blocks", []) or []
    for block in input_blocks:
        mode_name = block.get("mode")
        handler = mode_registry.get(mode_name)
        if handler is None:
            raise ValueError(
                f"Group '{group_name}' input block {block.get('name')!r} "
                f"specifies unknown mode {mode_name!r}."
            )

        block_cfg = _build_block_group_cfg(group_cfg, block, time_cfg)
        block_trains = _run_mode_handler(
            handler=handler,
            sim_cfg=sim_cfg,
            group_name=f"{group_name}:{block.get('name')}",
            group_cfg=block_cfg,
            geometry=geometry,
            rng=rng,
        )
        if len(block_trains) != n_syn:
            raise ValueError(
                f"Group '{group_name}' input block {block.get('name')!r} returned "
                f"{len(block_trains)} trains, expected {n_syn}."
            )

        for idx, train in enumerate(block_trains):
            if train.size:
                trains_accum[idx].extend(train.tolist())

    tstart = float(sim_cfg.get("tstart", 0.0))
    tstop = float(sim_cfg.get("tstop", tstart))
    out: List[np.ndarray] = []
    for spikes in trains_accum:
        if not spikes:
            out.append(np.array([], dtype=float))
            continue
        arr = np.asarray(spikes, dtype=float)
        arr = arr[(arr >= tstart) & (arr <= tstop)]
        arr.sort()
        out.append(arr)
    return out


def _build_block_group_cfg(
    group_cfg: Dict[str, Any],
    block: Dict[str, Any],
    group_time_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Create the mode-handler view for one input block."""
    source = dict(block.get("source", {}) or {})
    block_start = float(block["t_start"])
    block_stop = float(block["t_end"])
    duration = max(0.0, block_stop - block_start)
    group_anchors = group_time_cfg.get("anchors", {}) or {}

    jitter_tstart = None
    if _global_jitter_applies_to_block(block):
        jitter_tstart = group_anchors.get("jitter_tstart_ms")

    anchors = {
        "sim_tstart": group_anchors.get("sim_tstart"),
        "sim_tstop": group_anchors.get("sim_tstop"),
        "onset": block_start,
        "source_tstart": block_start,
        "source_tstop": block_stop,
        "duration_ms": duration,
        "source_trim_ms": source.get("crop_start_ms"),
        "baseline_rate_hz": None,
        "baseline_spec": {"kind": "none"},
        "jitter_tstart_ms": jitter_tstart,
        "block_name": block.get("name"),
        "block_role": block.get("role"),
    }
    block_time_cfg = {
        "anchors": anchors,
        "blocks": [
            {
                "kind": "source",
                "t_start": block_start,
                "t_end": block_stop,
                "name": block.get("name"),
                "role": block.get("role"),
                "mode": block.get("mode"),
            }
        ],
        "input_blocks": [block],
    }

    block_cfg = dict(group_cfg)
    block_cfg["mode"] = block.get("mode")
    block_cfg["source"] = source
    block_cfg["time_cfg"] = block_time_cfg
    return block_cfg


def _global_jitter_applies_to_block(block: Dict[str, Any]) -> bool:
    role = str(block.get("role") or "").strip().lower()
    return block.get("mode") == "homogeneous_poisson" and role in {"baseline", "background"}


def _build_group_inputs(
    group_name: str,
    group_cfg: Dict[str, Any],
    spike_trains: List[np.ndarray],
    randomness_meta: Optional[Dict[str, Any]] = None,
) -> GroupInputs:
    """
    Package per-group data into a GroupInputs object.

    Stores:
      - name
      - mode
      - spike_trains
      - meta: timing anchors, resolved N_syn, and any other useful snapshot info.
    """
    mode = group_cfg.get("mode", "unknown")
    meta: Dict[str, Any] = {}

    # Stash resolved N_syn and timing info if available
    syns = group_cfg.get("syns", {}) or {}
    if "N_syn_resolved" in syns:
        meta["N_syn_resolved"] = int(syns["N_syn_resolved"])

    time_cfg = group_cfg.get("time_cfg")
    if isinstance(time_cfg, dict):
        anchors = time_cfg.get("anchors", {}) or {}
        meta["time_anchors_ms"] = dict(anchors)
        meta["time_blocks"] = time_cfg.get("blocks", [])

    if randomness_meta:
        meta["randomness"] = randomness_meta

    return GroupInputs(
        name=group_name,
        mode=mode,
        spike_trains=spike_trains,
        meta=meta,
    )


def _finalize_inputs(
    sim_cfg: Dict[str, Any],
    groups_cfg: Dict[str, Dict[str, Any]],
    inputs: Dict[str, GroupInputs],
) -> None:
    """
    Final sanity checks / adjustments on assembled inputs.

    Big picture: make sure every active group has a GroupInputs entry and
    that spike times (if any) live inside the global simulation window.
    """
    tstart = float(sim_cfg.get("tstart", 0.0))
    tstop = float(sim_cfg.get("tstop", 0.0))

    for gname, gcfg in groups_cfg.items():
        if _should_skip_group(gname, gcfg):
            continue

        if gname not in inputs:
            raise ValueError(
                f"Active synapse group '{gname}' has no generated inputs. "
                "Check its 'mode', 'source', and mode handler implementation."
            )

        gin = inputs[gname]

        # Check spike times lie within [tstart, tstop]
        for idx, train in enumerate(gin.spike_trains):
            if train.size == 0:
                continue
            if train.min() < tstart - 1e-9 or train.max() > tstop + 1e-9:
                raise ValueError(
                    f"Group '{gname}', train {idx}: spike times must be within "
                    f"[{tstart}, {tstop}] ms (got min={float(train.min()):.3f}, "
                    f"max={float(train.max()):.3f})."
                )

    return
