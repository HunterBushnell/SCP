"""
Helpers for loading bio firing-rate curves from CSV files.

These curves are typically plotted in seconds, while simulations run in ms.
The helper returns time in seconds (consistent with plotting utilities that
multiply by 1000 for ms axes).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def load_bio_curve(
    csv_path: str,
    *,
    time_col: str = "Time",
    rate_col: str = "AvgFiringRate",
    t_min: float = 0.0,
    delay_ms: float = 0.0,
    time_unit: str = "s",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a bio curve CSV and return (time_s, rate_hz).

    Parameters
    ----------
    csv_path : str
        Path to CSV containing time and firing-rate columns.
    time_col, rate_col : str
        Column names for time and firing rate.
    t_min : float
        Filter: keep rows with time > t_min (in the input time_unit).
    delay_ms : float
        Delay to add (ms). Applied after time conversion.
    time_unit : str
        "s" if time_col is already seconds, "ms" if milliseconds.
    """
    df = pd.read_csv(csv_path)
    if time_col not in df or rate_col not in df:
        raise KeyError(
            f"CSV missing required columns: {time_col!r}, {rate_col!r} "
            f"(found: {list(df.columns)})"
        )

    time = df[time_col].to_numpy(dtype=float)
    rate = df[rate_col].to_numpy(dtype=float)

    if t_min is not None:
        mask = time > float(t_min)
        time = time[mask]
        rate = rate[mask]

    time_unit = (time_unit or "s").strip().lower()
    if time_unit == "ms":
        time = time / 1000.0
    elif time_unit != "s":
        raise ValueError(f"time_unit must be 's' or 'ms' (got {time_unit!r})")

    if delay_ms:
        time = time + (float(delay_ms) / 1000.0)

    return time, rate
