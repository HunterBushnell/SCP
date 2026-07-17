from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


CANONICAL_SECTION_GROUPS = ("soma", "dend", "apic", "axon", "all")

_DEFAULT_OWNER_SECTION_NAMES: Dict[str, tuple[str, ...]] = {
    "soma": ("soma", "somatic"),
    "dend": ("dend", "basal"),
    "apic": ("apic", "apical"),
    "axon": ("axon", "axonal"),
    "all": ("all", "all_sections", "sections"),
}


def _looks_like_section(value: Any) -> bool:
    """Return whether *value* behaves like one NEURON Section.

    A Section is iterable over segments, so blindly calling ``list(value)``
    would turn a single section into segments rather than a section collection.
    """

    return (
        value is not None
        and callable(getattr(value, "name", None))
        and callable(value)
        and hasattr(value, "nseg")
    )


def coerce_section_collection(value: Any) -> tuple[Any, ...]:
    """Normalize a Section, SectionList, HOC array, or Python iterable."""

    if value is None:
        return ()
    if _looks_like_section(value):
        return (value,)
    try:
        values = tuple(value)
    except TypeError:
        values = (value,)
    return tuple(section for section in values if _looks_like_section(section))


def _section_identity(section: Any) -> tuple[str, Any]:
    try:
        return ("name", str(section.name()))
    except Exception:
        return ("id", id(section))


def unique_sections(*groups: Iterable[Any]) -> tuple[Any, ...]:
    """Return cell sections in first-seen order without duplicates."""

    seen: set[tuple[str, Any]] = set()
    result: list[Any] = []
    for group in groups:
        for section in group:
            identity = _section_identity(section)
            if identity in seen:
                continue
            seen.add(identity)
            result.append(section)
    return tuple(result)


@dataclass
class LoadedCell:
    """Model-neutral wrapper returned by every SCP cell loader.

    ``h`` is the process-global NEURON interpreter. ``model`` is the object
    that owns this cell's sections (``h`` for legacy Allen models and a HOC
    template instance for object-owned models). The canonical section groups
    are always scoped to that model after :func:`ensure_section_aliases` runs.

    The original field order is retained for compatibility with callers that
    constructed ``LoadedCell`` positionally before the model-neutral contract
    was introduced.
    """

    h: Any
    utils: Any = None
    description: Any = None
    Vinit: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)
    loader: str = ""
    model: Any = None
    sections: Dict[str, tuple[Any, ...]] = field(default_factory=dict, repr=False)
    source_artifacts: Dict[str, str] = field(default_factory=dict)
    soma: tuple[Any, ...] = field(default_factory=tuple, repr=False)
    dend: tuple[Any, ...] = field(default_factory=tuple, repr=False)
    apic: tuple[Any, ...] = field(default_factory=tuple, repr=False)
    axon: tuple[Any, ...] = field(default_factory=tuple, repr=False)
    all: tuple[Any, ...] = field(default_factory=tuple, repr=False)

    @property
    def v_init(self) -> Optional[float]:
        """Snake-case alias for the historical ``Vinit`` attribute."""

        return self.Vinit

    def section_group(self, name: str, *, required: bool = False) -> tuple[Any, ...]:
        """Return one canonical, cell-scoped section group."""

        key = str(name).strip().lower()
        aliases = {
            "somatic": "soma",
            "basal": "dend",
            "apical": "apic",
            "axonal": "axon",
            "all_sections": "all",
            "sections": "all",
        }
        key = aliases.get(key, key)
        if key not in CANONICAL_SECTION_GROUPS:
            allowed = ", ".join(CANONICAL_SECTION_GROUPS)
            raise KeyError(f"Unknown section group {name!r}; expected one of: {allowed}")
        value = tuple(self.sections.get(key, getattr(self, key, ())))
        if required and not value:
            raise ValueError(f"Loaded cell has no sections in required group {key!r}.")
        return value

    def __repr__(self) -> str:
        label = self.config.get("cell_name", "<unnamed>")
        suffix = f", loader={self.loader!r}" if self.loader else ""
        return f"LoadedCell(label={label!r}{suffix})"


def _configured_owner_attributes(value: Any, *, group: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        names = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        names = tuple(value)
    else:
        raise TypeError(
            f"section_map[{group!r}] must be a string, a list of strings, or null."
        )
    normalized: list[str] = []
    for raw in names:
        name = str(raw).strip()
        if not name:
            raise ValueError(f"section_map[{group!r}] contains an empty owner attribute.")
        normalized.append(name)
    return tuple(normalized)


def _sections_from_explicit_map(
    owner: Any,
    section_map: Mapping[str, Any],
    group: str,
) -> tuple[Any, ...]:
    names = _configured_owner_attributes(section_map.get(group), group=group)
    groups: list[tuple[Any, ...]] = []
    for name in names:
        if not hasattr(owner, name):
            raise AttributeError(
                f"section_map[{group!r}] references missing model attribute {name!r}."
            )
        sections = coerce_section_collection(getattr(owner, name))
        groups.append(sections)
    return unique_sections(*groups)


def _sections_from_default_aliases(owner: Any, group: str) -> tuple[Any, ...]:
    for name in _DEFAULT_OWNER_SECTION_NAMES[group]:
        if not hasattr(owner, name):
            continue
        sections = coerce_section_collection(getattr(owner, name))
        if sections:
            return sections
    return ()


def ensure_section_aliases(
    cell: LoadedCell,
    *,
    owner: Any = None,
    section_map: Optional[Mapping[str, Any]] = None,
    allow_global_fallback: bool = True,
    require_soma: bool = False,
) -> LoadedCell:
    """Populate canonical section groups on a loaded cell.

    Explicit ``section_map`` entries reference attributes on the section-owning
    model. Omitted entries use common owner aliases. Global ``h.allsec()`` is
    only a final compatibility fallback and should be disabled by object-owned
    loaders so sections from other cells cannot leak into this wrapper.
    """

    resolved_owner = owner if owner is not None else cell.model
    if resolved_owner is None:
        resolved_owner = cell.h
    cell.model = resolved_owner

    mapping: Mapping[str, Any] = section_map or {}
    unknown = sorted(set(mapping) - set(CANONICAL_SECTION_GROUPS))
    if unknown:
        raise ValueError(
            "section_map contains unsupported canonical group(s): "
            + ", ".join(repr(name) for name in unknown)
        )

    groups: Dict[str, tuple[Any, ...]] = {}
    for group in CANONICAL_SECTION_GROUPS:
        if group in mapping:
            groups[group] = _sections_from_explicit_map(resolved_owner, mapping, group)
            continue

        # A loader may explicitly declare an empty optional group. Preserve
        # that emptiness rather than consulting process-global owner aliases.
        if group in cell.sections:
            groups[group] = coerce_section_collection(cell.sections[group])
            continue

        existing = coerce_section_collection(cell.sections.get(group))
        if not existing:
            existing = coerce_section_collection(getattr(cell, group, None))
        groups[group] = existing or _sections_from_default_aliases(resolved_owner, group)

    classified = unique_sections(
        groups["soma"], groups["dend"], groups["apic"], groups["axon"]
    )
    if groups["all"]:
        groups["all"] = unique_sections(groups["all"], classified)
    elif allow_global_fallback and hasattr(cell.h, "allsec"):
        groups["all"] = unique_sections(tuple(cell.h.allsec()), classified)
    elif classified:
        groups["all"] = classified

    if require_soma and not groups["soma"]:
        raise ValueError(
            "Loaded cell has no canonical soma sections. Configure the loader's "
            "section_map.soma entry or expose a common soma/somatic owner attribute."
        )

    cell.sections = groups
    for group, sections in groups.items():
        setattr(cell, group, sections)
    return cell


def apply_soma_diameter_multiplier(cell: LoadedCell) -> float:
    """Apply the legacy soma diameter multiplier through the canonical soma."""

    tuning = cell.config.get("tuning", {})
    if not isinstance(tuning, dict):
        tuning = {}
    multiplier = float(tuning.get("soma_diam_multiplier", 1.0))
    if multiplier <= 0:
        raise ValueError("tuning.soma_diam_multiplier must be greater than zero.")
    soma = cell.section_group("soma", required=True)
    if multiplier != 1.0:
        soma[0].diam = float(soma[0].diam) * multiplier
    return multiplier
