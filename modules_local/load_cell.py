"""
Step 5.2.1 — Load Cell

Single entry point:

    cell = load_cell(cell_config)

For now this assumes an Allen biophysical model with a `manifest.json`
and uses AllenSDK's Config + Utils to:

- load the model description,
- generate morphology,
- load cell parameters,
- cast genome values to float,
- optionally scale soma diameter,
- record Vinit.

It does *not* download files, compile MODs, or change directories.
Those belong in the prep steps (e.g., 0_download or the 5_colab setup cell).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from allensdk.model.biophys_sim.config import Config
from allensdk.model.biophysical.utils import Utils


@dataclass
class LoadedCell:
    """
    Minimal wrapper around a NEURON Allen cell.

    Attributes
    ----------
    h : object
        NEURON hoc interpreter from AllenSDK Utils (contains sections, etc.).
    utils : Utils
        AllenSDK Utils instance used to build the cell.
    description : Config
        AllenSDK Config/description object (includes manifest, genome, etc.).
    Vinit : Optional[float]
        Initial membrane potential (v_init) from the config, if present.
    config : dict
        Original cell_config dictionary passed into load_cell.
    """
    h: Any
    utils: Utils
    description: Config
    Vinit: Optional[float]
    config: Dict[str, Any]

    def __repr__(self) -> str:
        label = self.config.get("cell_name", "<unnamed>")
        return f"LoadedCell(label={label!r})"


def load_cell(cell_config: Dict[str, Any]) -> LoadedCell:
    """
    Build and return a NEURON Allen cell based on `cell_config`.

    Parameters
    ----------
    cell_config : dict
        Minimal expected structure:

        {
            "cell_name": "PV" or "SST" or ...,
            "paths": {
                "manifest": Path(...) or "manifest.json",
            },
            "tuning": {
                "soma_diam_multiplier": 1.0,   # optional
            },
        }

        - If `paths["manifest"]` is relative, it is resolved relative to CWD.
        - If `tuning["soma_diam_multiplier"]` is omitted, 1.0 is used.

    Returns
    -------
    LoadedCell
        Wrapper containing NEURON `h`, AllenSDK `utils` and `description`,
        Vinit, and the original `cell_config`.
    """
    cell_name = cell_config.get("cell_name", "<unknown>")

    paths = cell_config.get("paths", {})
    manifest_path = Path(paths.get("manifest", "manifest.json"))

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"load_cell: manifest.json not found at {manifest_path}. "
            "Ensure Step 1 set the working directory or pass the correct path "
            "in cell_config['paths']['manifest']."
        )

    tuning = cell_config.get("tuning", {})
    soma_diam_multiplier = float(tuning.get("soma_diam_multiplier", 1.0))

    # ------------------------------------------------------------------
    # AllenSDK: load description and create Utils
    # ------------------------------------------------------------------
    description = Config().load(str(manifest_path))
    utils = Utils(description)
    h = utils.h

    # Cast all genome values to float (your original pattern)
    genome = utils.description.data.get("genome", [])
    for d in genome:
        if "value" in d:
            d["value"] = float(d["value"])

    # Configure morphology and load cell parameters (your reused pattern)
    morphology_path = description.manifest.get_path("MORPHOLOGY")
    utils.generate_morphology(morphology_path.encode("ascii", "ignore"))
    utils.load_cell_parameters()

    # Optional soma diameter scaling (your PV tuning behavior)
    if hasattr(h, "soma") and len(h.soma) > 0:
        h.soma[0].diam = h.soma[0].diam * soma_diam_multiplier

    # Extract v_init if present
    Vinit: Optional[float] = None
    data = utils.description.data
    try:
        conditions = data.get("conditions", [])
        if conditions and "v_init" in conditions[0]:
            Vinit = float(conditions[0]["v_init"])
    except Exception:
        Vinit = None

    cell = LoadedCell(
        h=h,
        utils=utils,
        description=description,
        Vinit=Vinit,
        config=cell_config,
    )

    print(
        f"Loaded Allen cell for {cell_name!r} from {manifest_path}, "
        f"soma_diam_multiplier={soma_diam_multiplier}, Vinit={Vinit}"
    )

    return cell
