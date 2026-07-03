"""Dataclasses returned by synapse attachment/preview routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SynapseRecord:
    syn_id: int
    group: str
    type: str
    weight: float
    distance: float
    section: str
    x: float
    spike_times: List[float]
