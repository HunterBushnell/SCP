"""Notebook display helpers for safe synapse-placement previews."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional


def _display_markdown(markdown_text: str) -> None:
    """Display Markdown in notebooks, falling back to plain text in terminals."""
    try:
        from IPython.display import Markdown, display

        display(Markdown(markdown_text))
    except Exception:
        print(markdown_text)


def _record_field(record: Any, field: str) -> Any:
    if isinstance(record, dict):
        return record.get(field)
    return getattr(record, field, None)


def _resolve_groups(records: dict[str, Any], groups: Optional[str | Iterable[str]]) -> list[str]:
    if groups is None:
        return list(records.keys())
    if isinstance(groups, str):
        groups = [groups]
    groups = list(groups)
    if groups == ["all"]:
        return groups
    return [group for group in groups if group in records]


def _values_for(records: dict[str, Any], field: str, groups: list[str]) -> list[float]:
    selected_groups = records.keys() if groups == ["all"] else groups
    values: list[float] = []
    for group in selected_groups:
        for record in records.get(group, []) or []:
            value = _record_field(record, field)
            if value is not None:
                values.append(float(value))
    return values


def _auto_bin_width(values: list[float], *, target_bins: int = 30, fallback: float = 1.0) -> float:
    if not values:
        return float(fallback)
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 0:
        return float(fallback)
    return max(span / max(int(target_bins), 1), span / 1000.0)


def show_synapse_preview(
    session_or_syn_state: Any,
    *,
    trial_idx: int = 0,
    groups: Optional[str | Iterable[str]] = None,
    show_table: bool = True,
    show_plots: bool = True,
    plot_density: bool = False,
    distance_bin_um: float = 25.0,
    weight_bin: Optional[float] = None,
    figsize: tuple[float, float] = (6.0, 4.0),
) -> dict[str, Any]:
    """
    Display a safe Step 5 synapse-placement preview.

    `session_or_syn_state` can be either a prepared `SimulationSession` or an
    existing `syn_state` dict. When a session is provided, this calls
    `session.preview_synapses(...)`, which uses preview-only records and does
    not attach NEURON synapse objects to the cell.
    """
    session = None
    if isinstance(session_or_syn_state, dict) and "records" in session_or_syn_state:
        syn_state = session_or_syn_state
        cell = None
        geom = None
    else:
        session = session_or_syn_state
        if getattr(session, "iclamp_enabled", False):
            print("Synapse preview skipped: IClamp mode does not generate synapses.")
            return {"records": {}, "preview_only": True, "skipped": "iclamp"}
        syn_state = session.preview_synapses(trial_idx=int(trial_idx))
        cell = getattr(session, "cell", None)
        geom = getattr(session, "geom", None)

    records = syn_state.get("records", {}) or {}
    if not records:
        print("Synapse preview found no active synapse records.")
        return syn_state

    plotted_groups = _resolve_groups(records, groups)
    if not plotted_groups:
        print("Synapse preview found no matching groups.")
        return syn_state

    print("Synapse preview complete (no NEURON synapse objects attached).")
    for group in plotted_groups if plotted_groups != ["all"] else records.keys():
        print(f"  {group}: {len(records.get(group, []) or [])} synapses")

    if show_table:
        from modules.analysis import analysis

        summary = analysis.summarize_synapse_records(records, geom=geom)
        _display_markdown(
            analysis.format_synapse_summary_table(
                summary,
                title=f"Synapse preview: trial {int(trial_idx)}",
                groups=None if plotted_groups == ["all"] else plotted_groups,
            )
        )

    if show_plots:
        from modules.analysis import plotting

        if weight_bin is None:
            weight_bin = _auto_bin_width(
                _values_for(records, "weight", plotted_groups),
                target_bins=30,
                fallback=0.1,
            )

        plotting.plot_syn_records(
            cell,
            records,
            plotted_groups=plotted_groups,
            plotted_props=("weight_probability",),
            plot_type="hist",
            bins=float(weight_bin),
            win_size=float(weight_bin),
            fig_sizes=figsize,
        )
        plotting.plot_syn_records(
            cell,
            records,
            plotted_groups=plotted_groups,
            plotted_props=("distance_probability",),
            plot_type="hist",
            bins=float(distance_bin_um),
            win_size=float(distance_bin_um),
            fig_sizes=figsize,
        )
        plotting.plot_syn_records(
            cell,
            records,
            plotted_groups=plotted_groups,
            plotted_props=("weight", "distance"),
            fig_sizes=figsize,
        )
        if plot_density and cell is not None:
            plotting.plot_syn_records(
                cell,
                records,
                plotted_groups=plotted_groups,
                plotted_props=("distance_density",),
                plot_type="hist",
                bins=float(distance_bin_um),
                win_size=float(distance_bin_um),
                fig_sizes=figsize,
            )

    return syn_state
