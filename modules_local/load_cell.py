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
import os
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


def _apply_genome_based_parameters(utils: Utils) -> None:
    """
    Fallback parameter loader for all-active Allen models.

    AllenSDK's Utils.load_cell_parameters() expects `passive[0]` to contain
    `e_pas` and `cm`, which is true for perisomatic models but not for some
    all-active fits. In those cases, passive/channel parameters are carried in
    the genome list by section.
    """
    h = utils.h
    data = utils.description.data

    passive_list = data.get("passive", [])
    passive = passive_list[0] if isinstance(passive_list, list) and passive_list else {}
    if not isinstance(passive, dict):
        passive = {}

    genome = data.get("genome", [])
    if not isinstance(genome, list):
        genome = []

    conditions_list = data.get("conditions", [])
    conditions = (
        conditions_list[0]
        if isinstance(conditions_list, list) and conditions_list
        else {}
    )
    if not isinstance(conditions, dict):
        conditions = {}

    h("access soma")

    # Baseline passive setup.
    for sec in h.allsec():
        if "ra" in passive:
            sec.Ra = float(passive["ra"])
        sec.insert("pas")
        if "e_pas" in passive:
            e_pas = float(passive["e_pas"])
            for seg in sec:
                seg.pas.e = e_pas

    # Per-section cm overrides if present in passive block.
    cm_entries = passive.get("cm", [])
    if isinstance(cm_entries, list):
        for cm_cfg in cm_entries:
            if not isinstance(cm_cfg, dict):
                continue
            section = cm_cfg.get("section")
            cm_val = cm_cfg.get("cm")
            if section is None or cm_val is None:
                continue
            h('forsec "' + str(section) + '" { cm = %g }' % float(cm_val))

    # Apply channel/passive/genome parameters.
    for p in genome:
        if not isinstance(p, dict):
            continue
        section = p.get("section")
        name = p.get("name")
        mechanism = p.get("mechanism", "")
        if not name:
            continue

        try:
            val = float(p.get("value"))
        except (TypeError, ValueError):
            continue

        if section == "glob":
            h(str(name) + " = %g " % val)
            continue

        if not section:
            continue

        if mechanism:
            h('forsec "' + str(section) + '" { insert ' + str(mechanism) + " }")
        h('forsec "' + str(section) + '" { ' + str(name) + " = %g }" % val)

    # Reversal potentials, if present.
    erev_entries = conditions.get("erev", [])
    if isinstance(erev_entries, list):
        for erev in erev_entries:
            if not isinstance(erev, dict):
                continue
            section = erev.get("section")
            if not section:
                continue
            if "ek" in erev:
                h('forsec "' + str(section) + '" { ek = %g }' % float(erev["ek"]))
            if "ena" in erev:
                h('forsec "' + str(section) + '" { ena = %g }' % float(erev["ena"]))


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
    # AllenSDK manifests often reference additional files with relative paths
    # (e.g., model_file and morphology entries). Keep CWD pinned to the
    # manifest directory during AllenSDK loading so these lookups resolve.
    original_cwd = Path.cwd()
    manifest_dir = manifest_path.parent
    used_fallback_loader = False
    try:
        os.chdir(manifest_dir)

        description = Config().load(str(manifest_path.name))
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
        try:
            utils.load_cell_parameters()
        except KeyError as exc:
            # All-active fit files can omit passive e_pas/cm and store them in genome.
            if str(exc).strip("'\"") in {"e_pas", "cm"}:
                _apply_genome_based_parameters(utils)
                used_fallback_loader = True
            else:
                raise
    finally:
        os.chdir(original_cwd)

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

    # Backward-compat aliases for legacy notebook code that used direct
    # section lists on the cell object (e.g., cell.soma[0]).
    if not hasattr(cell, "soma") and hasattr(h, "soma"):
        cell.soma = h.soma
    if not hasattr(cell, "dend") and hasattr(h, "dend"):
        cell.dend = h.dend
    if not hasattr(cell, "apic") and hasattr(h, "apic"):
        cell.apic = h.apic
    if not hasattr(cell, "axon") and hasattr(h, "axon"):
        cell.axon = h.axon
    if not hasattr(cell, "all"):
        all_secs = h.SectionList()
        for sec in h.allsec():
            all_secs.append(sec)
        cell.all = all_secs

    print(
        f"Loaded Allen cell for {cell_name!r} from {manifest_path}, "
        f"soma_diam_multiplier={soma_diam_multiplier}, Vinit={Vinit}, "
        f"loader={'genome_fallback' if used_fallback_loader else 'allensdk_default'}"
    )

    return cell
