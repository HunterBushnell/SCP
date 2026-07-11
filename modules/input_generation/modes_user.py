"""
User-defined synaptic input modes for Step 5.2.3 (input generation).

Usage
-----
- Define one or more functions with the signature:

    def my_mode(
        sim_cfg: Dict[str, Any],
        group_cfg: Dict[str, Any],
        geometry: Optional[Any],
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        ...

  Each function must return a list of 1D NumPy arrays of spike times (ms)
  in simulation time, one array per synapse-equivalent source.

- Register them in get_user_mode_registry() by name, e.g.:

    return {
        "my_custom_mode": my_custom_mode,
    }

- In the notebook, merge with the default registry from input_modes_core
  and pass the combined registry into inputs.generate_inputs(...).

Contract highlights
-------------------
- Use group_cfg["syns"]["N_syn_resolved"] for the final synapse count.
- Use rng for randomness (do not use global np.random).
- Do not mutate sim_cfg or group_cfg.
- Return a list[np.ndarray] with times clipped to [sim_cfg["tstart"], sim_cfg["tstop"]].
"""


from __future__ import annotations

from typing import Any, Dict, Mapping, Optional
import numpy as np


# ---------------------------------------------------------------------
# Example user modes
# ---------------------------------------------------------------------


def my_custom_mode_example(
    sim_cfg: Dict[str, Any],
    group_cfg: Dict[str, Any],
    geometry: Optional[Any],
    rng: np.random.Generator,
) -> Any:
    """
    Example user-defined mode.

    Contract:
    - Read any needed parameters from the block-level view passed by Step 5:
        group_cfg["source"], group_cfg["time_cfg"], group_cfg["syns"], etc.
    - syns["N_syn_resolved"] (if present) gives the final synapse count.
    - Use sim_cfg["tstart"] / sim_cfg["tstop"] to keep spikes in the sim window.
    - Return: list[np.ndarray], where each array is a 1D array of spike
      times (ms, simulation time).

    Placeholder implementation:
    - Returns one empty spike train per synapse (no spikes).
    - Satisfies the Step 5.2.3 mode contract.
    """
    syn_cfg = (group_cfg or {}).get("syns", {}) or {}
    n_syn = int(syn_cfg.get("N_syn_resolved", 0) or 0)
    return [np.array([], dtype=float) for _ in range(n_syn)]


# ---------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------


def get_user_mode_registry() -> Mapping[str, Any]:
    """
    Return a mapping from mode name (as used in syn_config.json 'mode' field)
    to the corresponding handler function defined in this file.

    Edit this function to expose your custom modes.
    """
    return {
        "my_custom_mode": my_custom_mode_example,
        # Add your own modes here, e.g.:
        # "my_burst_mode": my_burst_mode,
    }
