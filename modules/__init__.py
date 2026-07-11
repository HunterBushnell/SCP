"""
Local modules for the SCP single-cell pipeline.

Primary packages:
- setup: input cell download and tune-directory preparation.
- model: cell loading, geometry grouping, and synapse attachment.
- input_generation: Step 5.2.3 input config normalization and spike-train generation.
- simulation: Step 5 simulation backend for notebooks, CLI, and SLURM.
- analysis: plotting and analysis utilities.
- notebooks: shared notebook setup/build helpers.
- tuning: shared Step 2/3 notebook setup, ACT integration, and proposal export helpers.
- core: shared utilities such as reproducible randomness.

The `modules.run_sim` facade remains as the stable notebook/API entry point
while implementation code lives under `modules.simulation`.
"""

__all__ = ["load_cell"]


def __getattr__(name):
    """Lazy convenience exports without importing NEURON/model code at package import."""
    if name == "load_cell":
        from .model.load_cell import load_cell

        return load_cell
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
