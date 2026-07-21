"""Concise, display-only formatting for Steps 2 and 3 tuning results."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any


DEFAULT_DISPLAY_SIGNIFICANT_DIGITS = 4

_PASSIVE_METRIC_LABELS = {
    "v_rest_mV": "Resting voltage (mV)",
    "rin_MOhm": "Input resistance (MΩ)",
    "tau_ms": "Membrane tau (ms)",
}

_PASSIVE_MEASUREMENT_COLUMNS = (
    ("V_rest", "Resting voltage (mV)"),
    ("R_in_rest_to_final", "Input resistance (MΩ)"),
    ("tau_avg", "Membrane tau (ms)"),
    ("tau_rest_to_trough", "Tau to trough (ms)"),
    ("sag_ratio", "Sag ratio"),
)

_PASSIVE_ADDITIONAL_COLUMNS = (
    ("tau_rest_to_trough", "Tau to trough (ms)"),
    ("sag_ratio", "Sag ratio"),
)

_ACTIVE_SUMMARY_COLUMNS = (
    ("spike_count", "Spikes"),
    ("spike_frequency_hz", "Frequency (Hz)"),
    ("first_spike_latency_ms", "First-spike latency (ms)"),
    ("mean_isi_ms", "Mean ISI (ms)"),
    ("adaptation_ratio", "Adaptation ratio"),
)

_ACTIVE_ADDITIONAL_COLUMNS = (
    ("rest_voltage_mv", "Resting voltage (mV)"),
    ("peak_voltage_mv", "Peak voltage (mV)"),
    ("min_voltage_mv", "Minimum voltage (mV)"),
    ("min_isi_ms", "Minimum ISI (ms)"),
    ("isi_cv", "ISI CV"),
)


def compact_display_data(
    value: Any,
    *,
    significant_digits: int = DEFAULT_DISPLAY_SIGNIFICANT_DIGITS,
) -> Any:
    """Return a recursively rounded display copy without mutating source data."""

    digits = int(significant_digits)
    if digits < 1:
        raise ValueError("significant_digits must be at least 1.")
    if value is None or isinstance(value, (str, bytes, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number) or number == 0.0:
            return number
        decimal_places = digits - 1 - math.floor(math.log10(abs(number)))
        return round(number, decimal_places)
    if isinstance(value, Mapping):
        return {
            key: compact_display_data(item, significant_digits=digits)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            compact_display_data(item, significant_digits=digits) for item in value
        )
    if isinstance(value, Sequence):
        return [
            compact_display_data(item, significant_digits=digits) for item in value
        ]
    return value


def format_tuning_value(
    value: Any,
    *,
    significant_digits: int = DEFAULT_DISPLAY_SIGNIFICANT_DIGITS,
) -> str:
    """Format one notebook value with a compact significant-figure limit."""

    digits = int(significant_digits)
    if digits < 1:
        raise ValueError("significant_digits must be at least 1.")
    if isinstance(value, Real) and not isinstance(value, (Integral, bool)):
        number = float(value)
        return f"{number:.{digits}g}" if math.isfinite(number) else str(number)
    return str(compact_display_data(value, significant_digits=digits))


def display_tuning_rows(
    title: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    significant_digits: int = DEFAULT_DISPLAY_SIGNIFICANT_DIGITS,
) -> list[dict[str, Any]]:
    """Display compact result rows while returning the rounded display copy."""

    display_rows = [
        dict(compact_display_data(dict(row), significant_digits=significant_digits))
        for row in rows
    ]
    if not display_rows:
        print(f"{title}: none")
        return display_rows

    print(title + ":")
    try:
        import pandas as pd
        from IPython.display import display

        with pd.option_context(
            "display.float_format",
            lambda number: format_tuning_value(
                number,
                significant_digits=significant_digits,
            ),
        ):
            display(pd.DataFrame(display_rows))
    except Exception:
        for row in display_rows:
            print(json.dumps(row, indent=2, default=str))
    return display_rows


def display_passive_analysis(
    metrics: Sequence[Mapping[str, Any]],
    target_comparison: Sequence[Mapping[str, Any]],
    *,
    amplitude_colors: Mapping[float, str] | None = None,
    significant_digits: int = DEFAULT_DISPLAY_SIGNIFICANT_DIGITS,
) -> dict[str, list[dict[str, Any]]]:
    """Display the shared Step 2 measurements and metric-first comparison."""

    metric_rows = [dict(row) for row in metrics]
    comparison_rows = [dict(row) for row in target_comparison]
    if amplitude_colors is None:
        from .passive import passive_amplitude_colors

        amplitude_colors = passive_amplitude_colors(
            [row.get("amp_pA") for row in metric_rows if row.get("amp_pA") is not None]
        )
    colors = {float(amp): str(color) for amp, color in amplitude_colors.items()}

    unexpected_spikes = [
        {
            "amp_pA": float(row["amp_pA"]),
            "spike_frequency_hz": float(row["spike_frequency_hz"]),
        }
        for row in metric_rows
        if row.get("amp_pA") is not None
        and row.get("spike_frequency_hz") is not None
        and float(row["spike_frequency_hz"]) > 0.0
    ]
    if unexpected_spikes:
        details = ", ".join(
            "{} pA: {} Hz".format(
                format_tuning_value(
                    row["amp_pA"], significant_digits=significant_digits
                ),
                format_tuning_value(
                    row["spike_frequency_hz"],
                    significant_digits=significant_digits,
                ),
            )
            for row in unexpected_spikes
        )
        print(
            "Warning: unexpected spiking during the passive protocol "
            f"({details}). Review the trace before interpreting passive metrics."
        )

    selected_columns = (
        _PASSIVE_ADDITIONAL_COLUMNS if comparison_rows else _PASSIVE_MEASUREMENT_COLUMNS
    )
    measurement_rows = []
    for row in metric_rows:
        displayed = {"Current (pA)": row.get("amp_pA")}
        for source, label in selected_columns:
            displayed[label] = row.get(source)
        if any(displayed[label] is not None for _source, label in selected_columns):
            measurement_rows.append(displayed)

    comparison_display_rows = _passive_comparison_display_rows(
        comparison_rows,
        amplitude_colors=colors,
    )
    measurement_title = (
        "Additional passive diagnostics"
        if comparison_display_rows
        else "Passive measurements"
    )
    if measurement_rows:
        _display_amplitude_table(
            measurement_title,
            measurement_rows,
            amplitude_colors=colors,
            current_column="Current (pA)",
            significant_digits=significant_digits,
        )
    if comparison_display_rows:
        _display_grouped_passive_comparison(
            comparison_display_rows,
            amplitude_colors=colors,
            significant_digits=significant_digits,
        )
    elif metric_rows:
        print("Passive target comparison: no configured passive targets.")

    return {
        "measurements": measurement_rows,
        "target_comparison": comparison_display_rows,
        "unexpected_spikes": unexpected_spikes,
    }


def display_active_analysis(
    metrics: Sequence[Mapping[str, Any]],
    *,
    amplitude_colors: Mapping[float, str] | None = None,
    significant_digits: int = DEFAULT_DISPLAY_SIGNIFICANT_DIGITS,
) -> dict[str, list[dict[str, Any]]]:
    """Display Step 3 active metrics with trace-matched current-row colors."""
    metric_rows = [dict(row) for row in metrics]
    if amplitude_colors is None:
        from .active import active_amplitude_colors

        amplitude_colors = active_amplitude_colors(
            [row.get("amp_pA") for row in metric_rows if row.get("amp_pA") is not None]
        )
    colors = {float(amp): str(color) for amp, color in amplitude_colors.items()}
    summary_rows: list[dict[str, Any]] = []
    additional_rows: list[dict[str, Any]] = []
    for row in metric_rows:
        summary = {"Current (pA)": row.get("amp_pA")}
        for source, label in _ACTIVE_SUMMARY_COLUMNS:
            summary[label] = row.get(source)
        summary_rows.append(summary)
        additional = {"Current (pA)": row.get("amp_pA")}
        for source, label in _ACTIVE_ADDITIONAL_COLUMNS:
            additional[label] = row.get(source)
        additional_rows.append(additional)
    if summary_rows:
        _display_amplitude_table(
            "Active firing summary",
            summary_rows,
            amplitude_colors=colors,
            current_column="Current (pA)",
            significant_digits=significant_digits,
        )
        _display_amplitude_table(
            "Additional active diagnostics",
            additional_rows,
            amplitude_colors=colors,
            current_column="Current (pA)",
            significant_digits=significant_digits,
        )
    else:
        print("Active firing summary: none")
    return {
        "firing_summary": summary_rows,
        "additional_diagnostics": additional_rows,
    }


def display_fi_analysis(
    fi_rows: Sequence[Mapping[str, Any]],
    target_comparison: Sequence[Mapping[str, Any]],
    *,
    model_color: str = "#1f77b4",
    reference_color: str = "#000000",
    significant_digits: int = DEFAULT_DISPLAY_SIGNIFICANT_DIGITS,
) -> list[dict[str, Any]]:
    """Display one compact FI table using the same colors as the FI plot."""
    measurement_by_amp = {
        float(row["amp_pA"]): row.get("spike_frequency_hz")
        for row in fi_rows
        if row.get("amp_pA") is not None
    }
    comparisons = [dict(row) for row in target_comparison]
    display_rows: list[dict[str, Any]] = []
    if comparisons:
        for row in comparisons:
            lookup = str(row.get("target_lookup") or "").replace("_", " ")
            display_rows.append(
                {
                    "Current (pA)": row.get("amp_pA"),
                    "Model frequency (Hz)": row.get("measured_frequency_hz"),
                    "Target frequency (Hz)": row.get("target_frequency_hz"),
                    "Difference (Hz)": row.get("delta"),
                    "Error (%)": row.get("pct_error"),
                    "Target lookup": lookup.title() if lookup else "",
                }
            )
        title = "FI target comparison"
    else:
        display_rows = [
            {
                "Current (pA)": amp,
                "Model frequency (Hz)": frequency,
            }
            for amp, frequency in measurement_by_amp.items()
        ]
        title = "FI measurements"

    if display_rows:
        _display_fi_table(
            title,
            display_rows,
            model_color=str(model_color),
            reference_color=str(reference_color),
            significant_digits=significant_digits,
        )
    else:
        print(title + ": none")
    if not comparisons:
        print("FI target comparison: no configured FI targets.")
    return display_rows


def _passive_comparison_display_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    amplitude_colors: Mapping[float, str],
) -> list[dict[str, Any]]:
    metric_order = {metric: index for index, metric in enumerate(_PASSIVE_METRIC_LABELS)}
    amplitude_order = {
        float(amp): index for index, amp in enumerate(amplitude_colors)
    }

    def _sort_key(row: Mapping[str, Any]) -> tuple[int, int, float]:
        metric = str(row.get("metric") or "")
        amp = float(row.get("amp_pA") or 0.0)
        return (
            metric_order.get(metric, len(metric_order)),
            amplitude_order.get(amp, len(amplitude_order)),
            amp,
        )

    displayed: list[dict[str, Any]] = []
    for row in sorted(rows, key=_sort_key):
        metric = str(row.get("metric") or "metric")
        unit = str(row.get("unit") or "").strip()
        label = _PASSIVE_METRIC_LABELS.get(metric)
        if label is None:
            label = metric.replace("_", " ").title()
            if unit:
                label += f" ({unit})"
        status = str(row.get("status") or "").replace("_", " ").strip()
        displayed.append(
            {
                "Metric": label,
                "Current (pA)": row.get("amp_pA"),
                "Target": row.get("target_value"),
                "Measured": row.get("measured_value"),
                "Difference": row.get("delta"),
                "Error (%)": row.get("pct_error"),
                "Status": status.title() if status else "",
            }
        )
    return displayed


def _display_grouped_passive_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    amplitude_colors: Mapping[float, str],
    significant_digits: int,
) -> None:
    print("Passive target comparison:")
    try:
        import pandas as pd
        from IPython.display import display

        frame = pd.DataFrame(list(rows)).set_index(["Metric", "Current (pA)"])
        styler = _style_amplitude_frame(
            frame,
            amplitude_colors=amplitude_colors,
            amplitude_index_level=1,
            significant_digits=significant_digits,
        )
        display(styler)
    except Exception:
        for row in rows:
            print(json.dumps(dict(row), indent=2, default=str))


def _display_amplitude_table(
    title: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    amplitude_colors: Mapping[float, str],
    current_column: str,
    significant_digits: int,
) -> None:
    print(title + ":")
    try:
        import pandas as pd
        from IPython.display import display

        frame = pd.DataFrame(list(rows)).set_index(current_column)
        styler = _style_amplitude_frame(
            frame,
            amplitude_colors=amplitude_colors,
            amplitude_index_level=0,
            significant_digits=significant_digits,
        )
        display(styler)
    except Exception:
        for row in rows:
            print(json.dumps(dict(row), indent=2, default=str))


def _display_fi_table(
    title: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    model_color: str,
    reference_color: str,
    significant_digits: int,
) -> None:
    print(title + ":")
    try:
        import pandas as pd
        from IPython.display import display

        frame = pd.DataFrame(list(rows)).set_index("Current (pA)")
        amplitude_colors = {
            float(amp): model_color for amp in frame.index.tolist()
        }
        styler = _style_amplitude_frame(
            frame,
            amplitude_colors=amplitude_colors,
            amplitude_index_level=0,
            significant_digits=significant_digits,
        )
        model_column = "Model frequency (Hz)"
        if model_column in frame.columns:
            styler = styler.set_properties(
                subset=[model_column],
                **{"background-color": _blend_hex(model_color, "#ffffff", alpha=0.16)},
            )
        target_column = "Target frequency (Hz)"
        if target_column in frame.columns:
            styler = styler.set_properties(
                subset=[target_column],
                **{
                    "background-color": _blend_hex(
                        reference_color,
                        "#ffffff",
                        alpha=0.12,
                    )
                },
            )
        display(styler)
    except Exception:
        for row in rows:
            print(json.dumps(dict(row), indent=2, default=str))


def _style_amplitude_frame(
    frame: Any,
    *,
    amplitude_colors: Mapping[float, str],
    amplitude_index_level: int,
    significant_digits: int,
) -> Any:
    numeric_columns = [
        column for column in frame.columns if column != "Status"
    ]
    formats = {
        column: (
            lambda value, digits=significant_digits: format_tuning_value(
                value,
                significant_digits=digits,
            )
        )
        for column in numeric_columns
    }
    amplitude_order = {
        float(amp): index for index, amp in enumerate(amplitude_colors)
    }

    def _row_style(row: Any) -> list[str]:
        raw_amp = row.name[amplitude_index_level] if isinstance(row.name, tuple) else row.name
        amp = float(raw_amp)
        color = amplitude_colors.get(amp, "#808080")
        base = "#f7f7f7" if amplitude_order.get(amp, 0) % 2 else "#ffffff"
        tint = _blend_hex(color, base, alpha=0.11)
        return [f"background-color: {tint}"] * len(row)

    def _current_index_style(values: Any) -> list[str]:
        styles = []
        for value in values:
            amp = float(value)
            color = amplitude_colors.get(amp, "#808080")
            base = "#f7f7f7" if amplitude_order.get(amp, 0) % 2 else "#ffffff"
            tint = _blend_hex(color, base, alpha=0.16)
            styles.append(
                f"background-color: {tint}; border-left: 5px solid {color}; "
                "font-weight: 600"
            )
        return styles

    styler = (
        frame.style.format(formats, na_rep="—")
        .apply(_row_style, axis=1)
        .apply_index(_current_index_style, axis=0, level=amplitude_index_level)
        .set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "left")]},
                {"selector": "td", "props": [("text-align", "right")]},
            ]
        )
    )
    return styler


def _blend_hex(foreground: str, background: str, *, alpha: float) -> str:
    from matplotlib.colors import to_hex, to_rgb

    front = to_rgb(foreground)
    back = to_rgb(background)
    mixed = tuple(
        float(alpha) * front_channel + (1.0 - float(alpha)) * back_channel
        for front_channel, back_channel in zip(front, back)
    )
    return to_hex(mixed)
