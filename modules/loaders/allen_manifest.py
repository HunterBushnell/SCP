from __future__ import annotations

import contextlib
import configparser
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from allensdk.model.biophys_sim.config import Config
from allensdk.model.biophysical.utils import AllActiveUtils, Utils

from modules.loaders.base import (
    LoadedCell,
    apply_soma_diameter_multiplier,
    ensure_section_aliases,
)
from modules.loaders.paths import resolve_loader_path


PERISOMATIC_MODEL_TYPE = "Biophysical - perisomatic"
ALL_ACTIVE_MODEL_TYPE = "Biophysical - all active"


def _ensure_configparser_readfp() -> None:
    """
    Restore the deprecated ConfigParser.readfp alias for AllenSDK on Python 3.12.

    AllenSDK 2.16.2 still calls `ConfigParser.readfp()` while loading
    manifest.json. Python 3.12 removed that alias. Mapping it to `read_file`
    lets AllenSDK reach its existing Python-3 fallback path.
    """
    if hasattr(configparser.ConfigParser, "readfp"):
        return

    def readfp(self, fp, filename=None):  # type: ignore[no-untyped-def]
        return self.read_file(fp, source=filename)

    configparser.ConfigParser.readfp = readfp  # type: ignore[attr-defined]


def _resolve_manifest_path(
    cell_config: Dict[str, Any],
    *,
    base_dir: Optional[str | Path] = None,
) -> Path:
    return resolve_loader_path(
        cell_config,
        "manifest",
        default="manifest.json",
        base_dir=base_dir,
        loader_name="allen_manifest",
    )


def validate_config(
    cell_config: Dict[str, Any],
    *,
    base_dir: Optional[str | Path] = None,
) -> Dict[str, Path]:
    """Validate Allen loader paths without constructing the model."""

    return {"manifest": _resolve_manifest_path(cell_config, base_dir=base_dir)}


def discover_source_artifacts(
    cell_config: Dict[str, Any],
    *,
    base_dir: Optional[str | Path] = None,
) -> Dict[str, Path]:
    """Return tune-local native model sources declared by an Allen manifest."""

    artifacts = validate_config(cell_config, base_dir=base_dir)
    manifest_path = artifacts["manifest"]
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not parse Allen manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Allen manifest must contain a JSON object: {manifest_path}")

    declared: list[tuple[str, str]] = []
    for entry in payload.get("biophys", []) or []:
        if not isinstance(entry, dict):
            continue
        raw_files = entry.get("model_file", [])
        if not isinstance(raw_files, list):
            raw_files = [raw_files]
        for raw in raw_files:
            if isinstance(raw, str) and raw and Path(raw).name != manifest_path.name:
                declared.append(("model_file", raw))

    # MORPHOLOGY and MARKER are model morphology sources. Stimulus/output NWB
    # entries are experimental/run data, not native cell-model artifacts.
    for entry in payload.get("manifest", []) or []:
        if not isinstance(entry, dict) or entry.get("type") != "file":
            continue
        key = str(entry.get("key", "")).strip().upper()
        spec = entry.get("spec")
        if key in {"MORPHOLOGY", "MARKER"} and isinstance(spec, str) and spec:
            declared.append((key.lower(), spec))

    seen: set[Path] = {manifest_path}
    counts: Dict[str, int] = {}
    for role, raw in declared:
        source = Path(raw).expanduser()
        if not source.is_absolute():
            source = manifest_path.parent / source
        source = source.resolve()
        if source in seen:
            continue
        if not source.is_file():
            raise FileNotFoundError(
                f"Allen manifest {manifest_path} declares missing model source: {source}"
            )
        seen.add(source)
        index = counts.get(role, 0)
        counts[role] = index + 1
        artifact_role = role if index == 0 else f"{role}_{index}"
        artifacts[artifact_role] = source
    return artifacts


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


def _section_group_name(section: Any) -> str:
    name = str(section.name()).rsplit(".", 1)[-1]
    return name.split("[", 1)[0]


def _sections_for_group(sections: tuple[Any, ...], group: str) -> tuple[Any, ...]:
    return tuple(section for section in sections if _section_group_name(section) == group)


def _set_scoped_genome_parameter(section: Any, name: str, value: float) -> None:
    """Set one Allen genome value without relying on a global ``forsec`` block."""

    try:
        setattr(section, name, value)
        return
    except AttributeError as exc:
        # Some legacy all-active Allen fits spell the sodium reversal as e_na.
        # NEURON exposes that ion property as ``ena`` once a sodium mechanism
        # has been inserted. Keep this compatibility translation Allen-local.
        alias = {"e_na": "ena"}.get(name)
        if alias is None:
            raise AttributeError(
                f"Allen genome parameter {name!r} is not available on "
                f"section {section.name()!r}. Check the fit JSON parameter name "
                "and its declared mechanism."
            ) from exc
        try:
            setattr(section, alias, value)
        except AttributeError as alias_exc:
            raise AttributeError(
                f"Allen genome parameter {name!r} maps to NEURON property "
                f"{alias!r}, but {alias!r} is unavailable on section "
                f"{section.name()!r} after declared mechanisms were inserted."
            ) from alias_exc


def _apply_genome_based_parameters(
    utils: Utils,
    sections: tuple[Any, ...],
    *,
    apply_all_active_conditions: bool = False,
) -> None:
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

    for sec in sections:
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
            for sec in _sections_for_group(sections, str(section)):
                sec.cm = float(cm_val)

    scoped_parameters: list[tuple[Any, str, float]] = []
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

        target_sections = _sections_for_group(sections, str(section))
        for sec in target_sections:
            if mechanism and h.ismembrane(str(mechanism), sec=sec) != 1:
                sec.insert(str(mechanism))
            scoped_parameters.append((sec, str(name), val))

    # Parameter assignment is deliberately a second pass. Legacy fit files can
    # place an ion reversal before the channel entry that creates that ion's
    # NEURON property (for example e_na before Nap/NaTa).
    for sec, name, val in scoped_parameters:
        _set_scoped_genome_parameter(sec, name, val)

    erev_entries = conditions.get("erev", [])
    if isinstance(erev_entries, list):
        for erev in erev_entries:
            if not isinstance(erev, dict):
                continue
            section = erev.get("section")
            if not section:
                continue
            for sec in _sections_for_group(sections, str(section)):
                if "ek" in erev and h.ismembrane("k_ion", sec=sec) == 1:
                    sec.ek = float(erev["ek"])
                if "ena" in erev and h.ismembrane("na_ion", sec=sec) == 1:
                    sec.ena = float(erev["ena"])

    if apply_all_active_conditions:
        if conditions.get("v_init") is not None:
            h.v_init = float(conditions["v_init"])
        if conditions.get("celsius") is not None:
            h.celsius = float(conditions["celsius"])


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


def _load_parameters(
    utils: Any,
    cell_config: Dict[str, Any],
    sections: tuple[Any, ...],
    *,
    force_scoped: bool,
) -> str:
    if force_scoped:
        _apply_genome_based_parameters(
            utils,
            sections,
            apply_all_active_conditions=isinstance(utils, AllActiveUtils),
        )
        return "scoped_genome"
    try:
        with _suppress_allen_output(cell_config):
            utils.load_cell_parameters()
        return "allensdk_default"
    except KeyError as exc:
        if str(exc).strip("'\"") in {"e_pas", "cm"}:
            _apply_genome_based_parameters(
                utils,
                sections,
                apply_all_active_conditions=isinstance(utils, AllActiveUtils),
            )
            return "genome_fallback"
        raise


def load_cell(
    cell_config: Dict[str, Any],
    *,
    base_dir: Optional[str | Path] = None,
) -> LoadedCell:
    """
    Build a NEURON cell from an AllenSDK/ADB `manifest.json` bundle.
    """

    cell_name = cell_config.get("cell_name", "<unknown>")
    artifacts = validate_config(cell_config, base_dir=base_dir)
    manifest_path = artifacts["manifest"]

    original_cwd = Path.cwd()
    manifest_dir = manifest_path.parent
    try:
        os.chdir(manifest_dir)

        _ensure_configparser_readfp()
        description = Config().load(str(manifest_path.name))
        utils, utils_label = _build_utils(cell_config, description)
        h = utils.h
        preexisting_section_names = {str(sec.name()) for sec in h.allsec()}

        genome = utils.description.data.get("genome", [])
        for entry in genome:
            if isinstance(entry, dict) and "value" in entry:
                entry["value"] = float(entry["value"])

        morphology_path = description.manifest.get_path("MORPHOLOGY")
        utils.generate_morphology(morphology_path.encode("ascii", "ignore"))
        model_sections = tuple(
            sec for sec in h.allsec() if str(sec.name()) not in preexisting_section_names
        )
        if not model_sections:
            raise RuntimeError(
                "Allen loader did not create a distinct set of sections. Restart the "
                "Python/Jupyter process before loading another legacy Allen model."
            )
        parameter_loader = _load_parameters(
            utils,
            cell_config,
            model_sections,
            force_scoped=bool(preexisting_section_names),
        )
    finally:
        os.chdir(original_cwd)

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
    paths = dict(config.get("paths", {}) or {})
    paths["manifest"] = str(manifest_path)
    config["paths"] = paths

    scoped_sections = {
        group: tuple(sec for sec in model_sections if _section_group_name(sec) == group)
        for group in ("soma", "dend", "apic", "axon")
    }
    scoped_sections["all"] = model_sections

    cell = LoadedCell(
        h=h,
        utils=utils,
        description=description,
        Vinit=Vinit,
        config=config,
        loader="allen_manifest",
        model=h,
        sections=scoped_sections,
        source_artifacts={role: str(path) for role, path in artifacts.items()},
    )
    ensure_section_aliases(
        cell,
        owner=h,
        allow_global_fallback=False,
        require_soma=True,
    )
    soma_diam_multiplier = apply_soma_diameter_multiplier(cell)

    print(
        f"Loaded Allen cell for {cell_name!r} from {manifest_path}, "
        f"model_type={_manifest_model_type(description)!r}, "
        f"soma_diam_multiplier={soma_diam_multiplier}, Vinit={Vinit}, "
        f"loader={utils_label}/{parameter_loader}"
    )

    return cell
