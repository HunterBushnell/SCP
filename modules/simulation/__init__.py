"""
Step 5 simulation backend.

The public entry point is `SimulationSession`, which prepares and runs a
tune directory from notebooks, CLI scripts, or SLURM wrappers.
"""

from modules.simulation.session import (
    SimulationOptions,
    SimulationSession,
    infer_cell_name,
    load_mechanisms,
    normalize_tune_dir,
)

__all__ = [
    "SimulationOptions",
    "SimulationSession",
    "infer_cell_name",
    "load_mechanisms",
    "normalize_tune_dir",
]
