"""Notebook display helpers for safe synapse-placement previews."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, Optional


SYNAPSE_PREVIEW_PLOTS = (
    "weight_distribution",
    "distance_distribution",
    "weight_vs_distance",
)


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


def _bin_edges(values: list[float], width: float) -> Any:
    import numpy as np

    if not values:
        return np.asarray([0.0, float(width)])
    lo = float(min(values))
    hi = float(max(values))
    if hi <= lo:
        padding = max(abs(lo) * 0.05, float(width) * 0.5, 1e-12)
        lo -= padding
        hi += padding
    start = math.floor(lo / width) * width
    stop = math.ceil(hi / width) * width
    if stop <= start:
        stop = start + width
    return np.arange(start, stop + width * 1.001, width)


def _normalize_plot_kinds(plot_kinds: Optional[str | Iterable[str]]) -> list[str]:
    if plot_kinds is None:
        return list(SYNAPSE_PREVIEW_PLOTS)
    if isinstance(plot_kinds, str):
        requested = [plot_kinds]
    else:
        requested = [str(value) for value in plot_kinds]
    aliases = {
        "weight": "weight_distribution",
        "weights": "weight_distribution",
        "distance": "distance_distribution",
        "placement": "distance_distribution",
        "scatter": "weight_vs_distance",
    }
    normalized: list[str] = []
    for raw in requested:
        value = aliases.get(raw.strip().lower(), raw.strip().lower())
        if value not in SYNAPSE_PREVIEW_PLOTS:
            raise ValueError(
                f"Unknown synapse preview plot {raw!r}. Choose from "
                + ", ".join(SYNAPSE_PREVIEW_PLOTS)
                + "."
            )
        if value not in normalized:
            normalized.append(value)
    return normalized


def _plot_compact_synapse_preview(
    records: dict[str, Any],
    *,
    groups: list[str],
    plot_kinds: list[str],
    histogram_density: bool,
    distance_bin_um: float,
    weight_bin: float,
    plot_columns: int,
    subplot_size: tuple[float, float],
) -> Any:
    import matplotlib.pyplot as plt

    if not plot_kinds:
        return None
    selected_groups = list(records) if groups == ["all"] else list(groups)
    columns = max(1, min(int(plot_columns), len(plot_kinds)))
    rows = int(math.ceil(len(plot_kinds) / columns))
    fig, axes_grid = plt.subplots(
        rows,
        columns,
        figsize=(float(subplot_size[0]) * columns, float(subplot_size[1]) * rows),
        squeeze=False,
    )
    axes = list(axes_grid.flat)
    colors = {
        group: f"C{index % 10}" for index, group in enumerate(selected_groups)
    }
    all_weights = _values_for(records, "weight", selected_groups)
    all_distances = _values_for(records, "distance", selected_groups)
    weight_edges = _bin_edges(all_weights, weight_bin)
    distance_edges = _bin_edges(all_distances, distance_bin_um)

    for axis, plot_kind in zip(axes, plot_kinds):
        for group in selected_groups:
            group_records = records.get(group, []) or []
            distances = [
                float(value)
                for value in (_record_field(record, "distance") for record in group_records)
                if value is not None
            ]
            weights = [
                float(value)
                for value in (_record_field(record, "weight") for record in group_records)
                if value is not None
            ]
            distance_weight_pairs = [
                (float(distance), float(weight))
                for record in group_records
                for distance, weight in [
                    (
                        _record_field(record, "distance"),
                        _record_field(record, "weight"),
                    )
                ]
                if distance is not None and weight is not None
            ]
            color = colors[group]
            if plot_kind == "weight_distribution" and weights:
                axis.hist(
                    weights,
                    bins=weight_edges,
                    density=bool(histogram_density),
                    alpha=0.55,
                    color=color,
                    label=group,
                )
            elif plot_kind == "distance_distribution" and distances:
                axis.hist(
                    distances,
                    bins=distance_edges,
                    density=bool(histogram_density),
                    alpha=0.55,
                    color=color,
                    label=group,
                )
            elif plot_kind == "weight_vs_distance" and distance_weight_pairs:
                axis.scatter(
                    [pair[0] for pair in distance_weight_pairs],
                    [pair[1] for pair in distance_weight_pairs],
                    alpha=0.6,
                    s=18,
                    color=color,
                    label=group,
                )

        if plot_kind == "weight_distribution":
            axis.set_title("Weight distribution")
            axis.set_xlabel("Synaptic weight")
            axis.set_ylabel("Probability density" if histogram_density else "Count")
        elif plot_kind == "distance_distribution":
            axis.set_title("Distance distribution")
            axis.set_xlabel("Distance from soma (µm)")
            axis.set_ylabel("Probability density" if histogram_density else "Count")
        else:
            axis.set_title("Weight vs distance")
            axis.set_xlabel("Distance from soma (µm)")
            axis.set_ylabel("Synaptic weight")
        axis.grid(True, alpha=0.3)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize="small")

    for unused_axis in axes[len(plot_kinds) :]:
        fig.delaxes(unused_axis)
    fig.tight_layout()
    plt.show()
    return fig


def show_synapse_preview(
    session_or_syn_state: Any,
    *,
    trial_idx: int = 0,
    groups: Optional[str | Iterable[str]] = None,
    show_table: bool = True,
    show_plots: bool = True,
    plot_kinds: Optional[str | Iterable[str]] = None,
    histogram_density: bool = True,
    plot_density: bool = False,
    distance_bin_um: float = 25.0,
    weight_bin: Optional[float] = None,
    plot_columns: int = 3,
    figsize: tuple[float, float] = (3.4, 2.8),
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

        summary = syn_state.get("summary")
        if not isinstance(summary, dict):
            summary = analysis.summarize_synapse_records(records, geom=geom)
        _display_markdown(
            analysis.format_synapse_summary_table(
                summary,
                title=f"Synapse preview: trial {int(trial_idx)}",
                groups=None if plotted_groups == ["all"] else plotted_groups,
            )
        )

    if show_plots:
        selected_plots = _normalize_plot_kinds(plot_kinds)
        if float(distance_bin_um) <= 0:
            raise ValueError("distance_bin_um must be greater than zero.")
        if weight_bin is None:
            weight_bin = _auto_bin_width(
                _values_for(records, "weight", plotted_groups),
                target_bins=30,
                fallback=0.1,
            )
        if float(weight_bin) <= 0:
            raise ValueError("weight_bin must be greater than zero.")
        if int(plot_columns) < 1:
            raise ValueError("plot_columns must be at least one.")
        if float(figsize[0]) <= 0 or float(figsize[1]) <= 0:
            raise ValueError("figsize values must be greater than zero.")
        _plot_compact_synapse_preview(
            records,
            groups=plotted_groups,
            plot_kinds=selected_plots,
            histogram_density=bool(histogram_density),
            distance_bin_um=float(distance_bin_um),
            weight_bin=float(weight_bin),
            plot_columns=int(plot_columns),
            subplot_size=figsize,
        )
        if plot_density and cell is not None:
            from modules.analysis import plotting

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


__all__ = ["SYNAPSE_PREVIEW_PLOTS", "show_synapse_preview"]
