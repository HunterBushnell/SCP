"""Data containers shared by input-generation modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class GroupInputs:
    """Materialized spike trains and metadata for one synapse group."""

    name: str
    mode: str
    spike_trains: List[np.ndarray] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
