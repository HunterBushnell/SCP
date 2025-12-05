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
    ) -> list[np.ndarray]:
        ...

  Each function must return a list of 1D NumPy arrays of spike times (ms)
  in simulation time, **one array per synapse-equivalent source**.

  The core pipeline (inputs._process_all_groups) will:
    - resolve syns["N_syn_resolved"] before calling your mode,
    - check that len(returned_trains) == N_syn_resolved.

- Register them in get_user_mode_registry() by name, e.g.:

    return {
        "my_custom_mode": my_custom_mode,
    }

- In the notebook, merge with the default registry from inputs.py and pass
  the combined registry into inputs.generate_inputs(...).
"""

"""
User-defined input modes for step 2.3 (input generation).

CONTRACT FOR ANY MODE
---------------------
Signature:
    def my_mode(sim_cfg, group_cfg, geometry, rng) -> list[np.ndarray]

Inputs:
    sim_cfg : dict
        Normalized global sim config; must NOT be modified.
    group_cfg : dict
        Normalized group config; includes:
            group_cfg["source"]  – mode-specific inputs (freq, path, etc.)
            group_cfg["timing"]  – onset/duration/timing fields (ms)
            group_cfg["syns"]    – synapse spec incl. "N_syn_resolved" (int)
    geometry : any | None
        Optional cell geometry; use only if your mode is geometry-dependent.
    rng : np.random.Generator
        Use this for all randomness (do not use global np.random).

Outputs:
    list[np.ndarray]
        One 1D float array of spike times (ms, sorted) per synapse-source.
        Length of list MUST equal group_cfg["syns"]["N_syn_resolved"].
        All spike times MUST lie within [sim_cfg["tstart"], sim_cfg["tstop"]].

Typical mode steps (per group):
    1) Read N_syn = group_cfg["syns"]["N_syn_resolved"].
    2) Compute or read the effective time window for this group
       (often via the timing fields in group_cfg["timing"]).
    3) Use group_cfg["source"] + rng (and optionally geometry) to
       generate N_syn spike trains.
    4) Ensure trains are np.ndarray[float], sorted, and time-clipped
       into the sim window.
    5) Return the list of trains; no NEURON calls and no in-place edits
       to sim_cfg / group_cfg.
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
    - Read any needed parameters from:
        group_cfg["source"], group_cfg["timing"], group_cfg["syns"], etc.
    - syns["N_syn_resolved"] (if present) gives the final synapse count.
    - Use sim_cfg["tstart"] / sim_cfg["tstop"] to keep spikes in the sim window.
    - Return: list[np.ndarray], where each array is a 1D array of spike
      times (ms, simulation time).

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
