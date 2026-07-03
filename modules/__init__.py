"""
Local modules for the SCP single-cell pipeline.

Primary packages:
- setup: input cell download and tune-directory preparation.
- model: cell loading, geometry grouping, and synapse attachment.
- input_generation: Step 5.2.3 input config normalization and spike-train generation.
- simulation: Step 5 simulation backend for notebooks, CLI, and SLURM.
- analysis: plotting and analysis utilities.
- notebooks: shared notebook setup/build helpers.
- core: shared utilities such as reproducible randomness.

The legacy `modules.run_sim` facade remains for compatibility with analysis and
notebook code that loads/saves Step 5 results.
"""

from .model.load_cell import load_cell  # re-export for convenience
