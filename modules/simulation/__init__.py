"""
Step 5 simulation backend.

The public entry point is `SimulationSession`, which prepares and runs a
tune directory from notebooks, CLI scripts, or SLURM wrappers.
"""

from scp_pipeline.simulation.session import (
    SimulationOptions,
    SimulationSession,
)
from scp_pipeline.simulation.session_setup import (
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
