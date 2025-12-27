# modules_local/randomness.py
"""
Randomness utilities for the SCP pipeline.

Goal: consistent, user-configurable randomness across trials, groups, and purposes
(inputs vs synapse placement vs synapse weights, etc.) with minimal coupling.

Config semantics (applies to any randomness setting value):
- None / False  -> fixed across trials (trial_component = 0)
- "derived"     -> varies per trial, reproducible (trial_component depends on trials setting)
- True          -> fully random (fresh entropy; seeds recorded in metadata)
- int           -> fixed to that explicit seed (independent of global seed; trial_component = 0)

Trials setting ("randomness.trials") controls the trial_component:
- None/False/int -> trial_component = 0 (identical trials)
- "derived"      -> trial_component = trial_idx
- True           -> trial_component = fresh random nonce per trial (recorded)

SeedSequence recipe for deterministic generators:
SeedSequence([ base_seed_used, trial_component, group_hash_u32, stream_hash_u32 ])
(all as uint32 words; base_seed_used expanded to 2x uint32)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List

import hashlib
import secrets
import numpy as np


SettingValue = Union[None, bool, int, str]
U32 = int


def _u32(x: int) -> U32:
    return int(x) & 0xFFFFFFFF


def _split_u64_to_u32_words(x: int) -> List[U32]:
    """Return [low32, high32] for a 64-bit-ish integer."""
    x = int(x) & 0xFFFFFFFFFFFFFFFF
    return [_u32(x), _u32(x >> 32)]


def stable_u32_from_str(s: str, *, salt: str = "") -> U32:
    """
    Stable 32-bit hash for identifiers (group names, stream names).
    Uses SHA256; stable across runs/machines (unlike Python's hash()).
    """
    h = hashlib.sha256((salt + s).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little", signed=False)


def get_by_path(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _parse_setting(val: SettingValue) -> Tuple[str, Optional[int]]:
    """
    Returns (kind, explicit_seed) where kind in:
      - "fixed"         (None/False)
      - "derived"       ("derived")
      - "random"        (True)
      - "fixed_explicit"(int)
    """
    if val is None or val is False:
        return "fixed", None
    if val is True:
        return "random", None
    if isinstance(val, int) and not isinstance(val, bool):
        return "fixed_explicit", int(val)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("derived",):
            return "derived", None
        if v in ("fixed", "identical", "same"):
            return "fixed", None
        if v in ("random", "stochastic"):
            return "random", None
    raise ValueError(f"Invalid randomness setting: {val!r}")


def _normalize_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
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
    return aliases.get(mode, mode)


def apply_randomness_mode(sim_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply a simplified randomness_mode to sim_cfg.

    If sim_cfg["randomness_mode"] is set, this function overwrites
    sim_cfg["randomness"] with a consistent per-component config.
    """
    if not isinstance(sim_cfg, dict):
        return sim_cfg
    mode = sim_cfg.get("randomness_mode")
    if mode in (None, "", False):
        return sim_cfg

    mode = _normalize_mode(mode)
    if mode not in ("fixed", "derived", "random"):
        raise ValueError(f"Invalid randomness_mode: {mode!r}")

    if mode == "fixed":
        setting_val: SettingValue = False
        trials_val: SettingValue = False
    elif mode == "derived":
        setting_val = "derived"
        trials_val = "derived"
    else:
        setting_val = True
        trials_val = True

    rand_cfg = sim_cfg.get("randomness")
    if not isinstance(rand_cfg, dict):
        rand_cfg = {}

    global_cfg = rand_cfg.get("global")
    if not isinstance(global_cfg, dict):
        global_cfg = {}

    seed = global_cfg.get("seed", sim_cfg.get("seed"))
    rand_cfg = {
        "global": {
            "state": True,
            "seed": seed,
        },
        "trials": trials_val,
        "inputs": setting_val,
        "timing": {
            "tstart": setting_val,
            "tstop": setting_val,
            "jitter": setting_val,
        },
        "synapses": {
            "placement": setting_val,
            "weights": setting_val,
            "dynamics": setting_val,
        },
        "modes": {
            "homogeneous_poisson": setting_val,
            "inhomogeneous_poisson": setting_val,
            "precomputed": setting_val,
        },
    }

    sim_cfg["randomness"] = rand_cfg
    sim_cfg["randomness_mode"] = mode
    return sim_cfg


@dataclass
class RandomnessMeta:
    base_seed_used: Optional[int]
    trials_setting: SettingValue
    # trial_nonces[trial_idx] is only populated when trials_setting == True (fully random trials)
    trial_nonces: Dict[int, int]
    # records for any setting==True calls (fully random components)
    random_seeds_used: List[Dict[str, Any]]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "base_seed_used": self.base_seed_used,
            "trials_setting": self.trials_setting,
            "trial_nonces": dict(self.trial_nonces),
            "random_seeds_used": list(self.random_seeds_used),
            "seed_recipe": (
                "SeedSequence([base_seed_used, trial_component, group_hash_u32, stream_hash_u32]) "
                "with base_seed_used expanded to 2x uint32 words"
            ),
        }


class RandomnessManager:
    """
    Owns the per-run base seed, trial nonce bookkeeping, and seed logging for any fully-random paths.

    Typical usage:
      rm = RandomnessManager(sim_cfg)
      trial_rng = rm.trial(trial_idx)
      rng_inputs = trial_rng.rng("inputs", group="pn_exc", stream="mode:homogeneous_poisson")
      rng_place  = trial_rng.rng("synapses.placement", group="pn_exc", stream="placement")
      rng_weight = trial_rng.rng("synapses.weights", group="pn_exc", stream="weights")
    """

    def __init__(self, sim_cfg: Dict[str, Any]):
        self.sim_cfg = sim_cfg
        self.cfg = sim_cfg.get("randomness", {}) if isinstance(sim_cfg, dict) else {}

        global_state = get_by_path(self.cfg, "global.state", True)
        self.global_state = bool(global_state)

        seed_cfg = get_by_path(self.cfg, "global.seed", None)
        self._base_seed_cfg = seed_cfg if seed_cfg is None else int(seed_cfg)

        # Resolve base seed used once per run.
        # If user did not provide a seed, generate one but DO NOT mutate the JSON; record in metadata instead.
        if not self.global_state:
            # "Off" mode: keep seed behavior minimal.
            self.base_seed_used = self._base_seed_cfg
        else:
            self.base_seed_used = (
                self._base_seed_cfg if self._base_seed_cfg is not None else self._fresh_u64()
            )

        self.trials_setting: SettingValue = get_by_path(self.cfg, "trials", "derived")

        self._trial_nonces: Dict[int, int] = {}
        self._random_seeds_used: List[Dict[str, Any]] = []

    def meta(self) -> RandomnessMeta:
        return RandomnessMeta(
            base_seed_used=self.base_seed_used,
            trials_setting=self.trials_setting,
            trial_nonces=self._trial_nonces,
            random_seeds_used=self._random_seeds_used,
        )

    @staticmethod
    def _fresh_u64() -> int:
        return secrets.randbits(64)

    def _trial_component_u32(self, trial_idx: int) -> U32:
        """
        Compute trial_component based on randomness.trials:
          - fixed / fixed_explicit -> 0
          - derived -> trial_idx
          - random -> per-trial nonce (recorded)
        """
        kind, _explicit = _parse_setting(self.trials_setting)

        if kind in ("fixed", "fixed_explicit"):
            return 0

        if kind == "derived":
            return _u32(trial_idx)

        # kind == "random"
        if trial_idx not in self._trial_nonces:
            self._trial_nonces[trial_idx] = self._fresh_u64()
        return _u32(self._trial_nonces[trial_idx])

    def _deterministic_rng(
        self,
        *,
        base_seed: int,
        trial_component: U32,
        group: Optional[str],
        stream: Union[str, int],
    ) -> np.random.Generator:
        group_u32 = stable_u32_from_str(group, salt="group:") if group else 0
        stream_u32 = _u32(stream) if isinstance(stream, int) else stable_u32_from_str(str(stream), salt="stream:")
        base_words = _split_u64_to_u32_words(base_seed)
        ss = np.random.SeedSequence([*base_words, _u32(trial_component), _u32(group_u32), _u32(stream_u32)])
        return np.random.default_rng(ss)

    def trial(self, trial_idx: int) -> "TrialRandomness":
        return TrialRandomness(self, int(trial_idx))


class TrialRandomness:
    """
    A per-trial view of the RandomnessManager.
    """

    def __init__(self, rm: RandomnessManager, trial_idx: int):
        self.rm = rm
        self.trial_idx = trial_idx
        self.trial_component = rm._trial_component_u32(trial_idx)

    def setting(self, path: str, default: SettingValue = None) -> SettingValue:
        return get_by_path(self.rm.cfg, path, default)

    def rng(
        self,
        setting_path: str,
        *,
        group: Optional[str] = None,
        stream: Union[str, int] = "",
        label: Optional[str] = None,
    ) -> np.random.Generator:
        """
        Get a numpy Generator for this trial, for a given setting path (e.g. "inputs" or "synapses.weights").

        - If setting is fixed/null/false -> deterministic using base_seed_used with trial_component=0.
        - If setting is "derived" -> deterministic using base_seed_used with trial_component derived from trials setting.
        - If setting is True -> fully random: create RNG from fresh random seed and record it.
        - If setting is int -> deterministic using that explicit seed with trial_component=0.

        The returned RNG should be used for all draws in that context (avoid calling rng() per synapse).
        """
        if not self.rm.global_state:
            # Minimal behavior when randomness system is disabled.
            if self.rm.base_seed_used is None:
                return np.random.default_rng()
            return np.random.default_rng(int(self.rm.base_seed_used))

        val = self.setting(setting_path, None)
        kind, explicit = _parse_setting(val)

        # Decide base seed source
        if kind == "fixed_explicit":
            base_seed = int(explicit)
            trial_component = 0
        elif kind == "fixed":
            base_seed = int(self.rm.base_seed_used)
            trial_component = 0
        elif kind == "derived":
            base_seed = int(self.rm.base_seed_used)
            trial_component = self.trial_component
        else:
            # kind == "random": fully random (but record seed so it can be reproduced if desired)
            seed64 = self.rm._fresh_u64()
            rec = {
                "trial_idx": self.trial_idx,
                "setting_path": setting_path,
                "group": group,
                "stream": stream,
                "label": label or setting_path,
                "seed_used": int(seed64),
            }
            self.rm._random_seeds_used.append(rec)
            # use seed64 as base_seed, and keep trial_component 0 (we're already random)
            base_seed = int(seed64)
            trial_component = 0

        return self.rm._deterministic_rng(
            base_seed=base_seed,
            trial_component=_u32(trial_component),
            group=group,
            stream=stream,
        )
