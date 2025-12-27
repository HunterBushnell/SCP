"""
Local modules for the SCP single-cell pipeline.

Step 5.2 (build + inputs) components:
- load_cell:   Step 5.2.1 — construct the NEURON cell from tuned model files.
- geometry:    Step 5.2.2 — define geometry groups for synapse placement.
- inputs:      Step 5.2.3 — generate spike trains for each synapse group.
- synapses:    Step 5.2.4 — attach synapses and connect spike trains to the cell.
"""

from .load_cell import load_cell  # re-export for convenience
from .bio_curve import load_bio_curve  # re-export for convenience
