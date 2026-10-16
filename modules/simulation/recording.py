"""Shared recording of density-mechanism currents on cell-owned segments."""

from __future__ import annotations

from typing import Any

from neuron import h


# These built-in currents are exposed on the segment without a mechanism suffix.
# An arbitrary bare-name fallback could instead select a summed ion current or
# a variable belonging to another mechanism.
_BUILTIN_CURRENT_REFERENCES = {
    ("extracellular", "i_membrane"): "_ref_i_membrane",
}


def _current_context(seg: Any, mechanism: str, variable: str) -> str:
    return (
        f"current {mechanism}.{variable} in section {seg.sec.name()!r} "
        f"at x={seg.x:g}"
    )


def _resolve_mechanism_current_reference(
    seg: Any,
    mechanism: str,
    variable: str,
) -> Any:
    """Resolve a mechanism current while retaining native-pointer failures."""
    context = _current_context(seg, mechanism, variable)
    missing: AttributeError | None = None
    try:
        owner = getattr(seg, mechanism)
    except AttributeError as exc:
        missing = exc
        owner = None
    except Exception as exc:
        raise RuntimeError(f"Cannot access {context}. Check native model initialization.") from exc

    candidates = []
    if owner is not None:
        candidates.append((owner, f"_ref_{variable}"))
    candidates.append((seg, f"_ref_{variable}_{mechanism}"))
    builtin_ref = _BUILTIN_CURRENT_REFERENCES.get((mechanism, variable))
    if builtin_ref is not None:
        candidates.append((seg, builtin_ref))

    for reference_owner, name in candidates:
        try:
            reference = getattr(reference_owner, name)
        except AttributeError as exc:
            if missing is None:
                missing = exc
        except Exception as exc:
            raise RuntimeError(
                f"Cannot access {context}. Check native pointer initialization."
            ) from exc
        else:
            # Missing-reference tracebacks include caller frames. Keeping one
            # after a successful fallback can retain a previous trial's IClamp
            # until cyclic GC and change the next sweep's voltage trace.
            missing = None
            return reference

    raise RuntimeError(
        f"Cannot resolve a recording reference for {context}. Check the mechanism "
        "variable and native pointer initialization, or disable current recording."
    ) from missing


def _record_mechanism_current(seg: Any, mechanism: str, variable: str) -> Any:
    """Create one recorder, adding context without hiding NEURON failures."""
    reference = _resolve_mechanism_current_reference(seg, mechanism, variable)
    try:
        return h.Vector().record(reference)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot record {_current_context(seg, mechanism, variable)}. "
            "Check native pointer initialization and mechanism lifetime."
        ) from exc
