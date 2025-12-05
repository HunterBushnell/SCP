"""
input_modes_user.py

User-defined synaptic input modes for Step 2.3.

Usage
-----
- Define one or more functions with the signature:

    def my_mode(
        sim_cfg: Dict[str, Any],
        group_cfg: Dict[str, Any],
        geometry: Optional[Any],
        rng: np.random.Generator,
    ) -> Any:
        ...

  Each function must return a list of 1D NumPy arrays of spike times (ms)
  in simulation time, one array per synapse (or per “prototype” train).

- Register them in get_user_mode_registry() by name, e.g.:

    return {
        "my_custom_mode": my_custom_mode,
    }

- In the notebook, merge with the default registry from inputs.py and pass
  the combined registry into inputs.generate_inputs(...).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional
import numpy as np


# ---------------------------------------------------------------------
# Example user mode stubs
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
    - Read any needed parameters from group_cfg["source"], group_cfg["timing"],
      or group_cfg["syns"].
    - Use sim_cfg["tstart"] / sim_cfg["tstop"] to keep spikes in the sim window.
    - Return: List[np.ndarray], where each array is a 1D array of spike times (ms).

    Replace this body with your own logic.
    """
    # Placeholder implementation: no spikes.
    spike_trains: list[np.ndarray] = []
    return spike_trains


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
        # "my_custom_mode": my_custom_mode_example,
        # Add your own modes here, e.g.:
        # "my_burst_mode": my_burst_mode,
    }
