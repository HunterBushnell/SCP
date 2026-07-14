from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
from typing import Any, Dict, Optional

from allensdk.model.biophys_sim.config import Config
from allensdk.model.biophysical.utils import AllActiveUtils, Utils

from modules.loaders.base import LoadedCell, ensure_section_aliases


PERISOMATIC_MODEL_TYPE = "Biophysical - perisomatic"
ALL_ACTIVE_MODEL_TYPE = "Biophysical - all active"


def _resolve_manifest_path(cell_config: Dict[str, Any]) -> Path:
    paths = cell_config.get("paths", {})
    manifest_path = Path(paths.get("manifest", "manifest.json"))
    if manifest_path.is_file():
        return manifest_path

    base_candidates = []
    for key in ("tune_dir", "base_dir", "root"):
        raw = paths.get(key)
        if raw:
            base_candidates.append(Path(raw))
    raw_tune_dir = cell_config.get("tune_dir")
    if raw_tune_dir:
        base_candidates.append(Path(raw_tune_dir))

    for base in base_candidates:
        candidate = base / manifest_path
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"allen_manifest loader: manifest.json not found at {manifest_path}. "
        "Set cell_config['paths']['manifest'] to an existing file."
    )


def _allen_options(cell_config: Dict[str, Any]) -> Dict[str, Any]:
    opts = cell_config.get("allen", {})
    if not isinstance(opts, dict):
        opts = {}
    return opts


def _manifest_model_type(description: Config) -> Optional[str]:
    try:
        return description.data["biophys"][0].get("model_type")
    except Exception:
        return None


def _configured_model_type(cell_config: Dict[str, Any], description: Config) -> Optional[str]:
    opts = _allen_options(cell_config)
    raw = (
        opts.get("model_type")
        or cell_config.get("allen_model_type")
        or cell_config.get("model_type")
        or "auto"
    )
    if raw in (None, "", "auto"):
        return _manifest_model_type(description)
    return str(raw)


def _configured_axon_type(cell_config: Dict[str, Any], description: Config) -> str:
    opts = _allen_options(cell_config)
    raw = opts.get("axon_type") or cell_config.get("allen_axon_type")
    if raw in (None, "", "auto"):
        try:
            biophys = description.data.get("biophys", [])
            for item in biophys:
                if isinstance(item, dict) and item.get("axon_type"):
                    return str(item["axon_type"])
        except Exception:
            pass
        return "stub"
    return str(raw)


def _configured_utils_strategy(cell_config: Dict[str, Any]) -> str:
    opts = _allen_options(cell_config)
    raw = opts.get("utils") or "standard"
    return str(raw).strip().lower()


def _suppress_allen_output(cell_config: Dict[str, Any]):
    opts = _allen_options(cell_config)
    verbose = bool(opts.get("verbose", cell_config.get("allen_verbose", False)))
    if verbose:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(io.StringIO())


def _apply_genome_based_parameters(utils: Utils) -> None:
    """
    Apply Allen genome/passive entries directly when `Utils.load_cell_parameters()`
    cannot read a bundle's passive-block layout.
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

    for sec in h.allsec():
        if "ra" in passive:
            sec.Ra = float(passive["ra"])
        sec.insert("pas")
        if "e_pas" in passive:
            e_pas = float(passive["e_pas"])
            for seg in sec:
                seg.pas.e = e_pas

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

    for param in genome:
        if not isinstance(param, dict):
            continue
        section = param.get("section")
        name = param.get("name")
        mechanism = param.get("mechanism", "")
        if not name:
            continue

        try:
            val = float(param.get("value"))
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


def _build_utils(cell_config: Dict[str, Any], description: Config) -> tuple[Any, str]:
    model_type = _configured_model_type(cell_config, description)
    strategy = _configured_utils_strategy(cell_config)
    use_all_active = strategy in {"all_active", "allactive", "all-active"}
    if strategy == "auto":
        use_all_active = model_type == ALL_ACTIVE_MODEL_TYPE

    if use_all_active and model_type == ALL_ACTIVE_MODEL_TYPE:
        axon_type = _configured_axon_type(cell_config, description)
        return AllActiveUtils(description, axon_type), f"all_active:{axon_type}"
    return Utils(description), f"standard:{model_type or 'unknown'}"


def _load_parameters(utils: Any, cell_config: Dict[str, Any]) -> str:
    try:
        with _suppress_allen_output(cell_config):
            utils.load_cell_parameters()
        return "allensdk_default"
    except KeyError as exc:
        if str(exc).strip("'\"") in {"e_pas", "cm"}:
            _apply_genome_based_parameters(utils)
            return "genome_fallback"
        raise


def load_cell(cell_config: Dict[str, Any]) -> LoadedCell:
    """
    Build a NEURON cell from an AllenSDK/ADB `manifest.json` bundle.
    """

    cell_name = cell_config.get("cell_name", "<unknown>")
    manifest_path = _resolve_manifest_path(cell_config)

    tuning = cell_config.get("tuning", {})
    soma_diam_multiplier = float(tuning.get("soma_diam_multiplier", 1.0))

    original_cwd = Path.cwd()
    manifest_dir = manifest_path.parent
    try:
        os.chdir(manifest_dir)

        description = Config().load(str(manifest_path.name))
        utils, utils_label = _build_utils(cell_config, description)
        h = utils.h

        genome = utils.description.data.get("genome", [])
        for entry in genome:
            if isinstance(entry, dict) and "value" in entry:
                entry["value"] = float(entry["value"])

        morphology_path = description.manifest.get_path("MORPHOLOGY")
        utils.generate_morphology(morphology_path.encode("ascii", "ignore"))
        parameter_loader = _load_parameters(utils, cell_config)
    finally:
        os.chdir(original_cwd)

    if hasattr(h, "soma") and len(h.soma) > 0:
        h.soma[0].diam = h.soma[0].diam * soma_diam_multiplier

    Vinit: Optional[float] = None
    data = utils.description.data
    try:
        conditions = data.get("conditions", [])
        if conditions and "v_init" in conditions[0]:
            Vinit = float(conditions[0]["v_init"])
    except Exception:
        Vinit = None

    config = dict(cell_config)
    config["cell_loader"] = "allen_manifest"
    config.setdefault("allen_model_type", _manifest_model_type(description))

    cell = LoadedCell(
        h=h,
        utils=utils,
        description=description,
        Vinit=Vinit,
        config=config,
        loader="allen_manifest",
    )
    ensure_section_aliases(cell)

    print(
        f"Loaded Allen cell for {cell_name!r} from {manifest_path}, "
        f"model_type={_manifest_model_type(description)!r}, "
        f"soma_diam_multiplier={soma_diam_multiplier}, Vinit={Vinit}, "
        f"loader={utils_label}/{parameter_loader}"
    )

    return cell
