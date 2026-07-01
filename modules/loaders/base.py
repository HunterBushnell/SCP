from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LoadedCell:
    """
    Minimal wrapper around a loaded NEURON cell.

    Loader-specific fields such as `utils` and `description` are intentionally
    optional so non-Allen loaders can return the same shape without depending on
    AllenSDK.
    """

    h: Any
    utils: Any = None
    description: Any = None
    Vinit: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)
    loader: str = ""

    def __repr__(self) -> str:
        label = self.config.get("cell_name", "<unnamed>")
        suffix = f", loader={self.loader!r}" if self.loader else ""
        return f"LoadedCell(label={label!r}{suffix})"


def ensure_section_aliases(cell: LoadedCell) -> LoadedCell:
    """
    Add common section-list aliases expected by existing notebooks/modules.

    Loaders may return richer objects, but Step 5 code currently expects
    `.h`, `.soma`, `.dend`, `.apic`, `.axon`, and `.all` when available.
    """

    h = cell.h
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
    return cell
