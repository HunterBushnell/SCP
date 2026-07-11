"""Selection UI and run-resolution helpers for `6_analysis.ipynb`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .. import analysis
from ._engine import (
    HELP_SELECTION,
    _compare_list_dir_options,
    _compare_list_paths_text,
    _maybe_import_display,
    _maybe_import_widgets,
    _print_help,
    compare_enabled,
    resolve_compare,
    resolve_single,
)
from .state import (
    get_selection_from_globals,
    resolve_compare_from_globals,
    resolve_single_from_globals,
    sync_common_from_globals,
)

def build_selection_ui(g: Dict[str, Any]) -> None:
    if not g.get("use_widgets", True):
        print("Widgets disabled (use_widgets=False).")
        return
    widgets = _maybe_import_widgets()
    display, _ = _maybe_import_display()
    if widgets is None or display is None:
        print("Widgets not enabled or ipywidgets unavailable.")
        return

    out_selection = widgets.Output()

    base_dir = g.get("BASE_DIR")
    if base_dir is None:
        base_dir = analysis.find_scp_root(Path.cwd())
        g["BASE_DIR"] = base_dir
    cells_dir = g.get("CELLS_DIR")
    if cells_dir is None:
        cells_dir = Path(base_dir) / "cells"
        g["CELLS_DIR"] = cells_dir

    def _sanitize_options(raw: Any, fallback: Any) -> list[Any]:
        opts = [v for v in (list(raw) if raw is not None else []) if v not in (None, "")]
        if not opts and fallback not in (None, ""):
            opts = [fallback]
        if not opts:
            opts = [""]
        return opts

    def _pick_value(options: list[Any], preferred: Any) -> Any:
        return preferred if preferred in options else options[0]

    cells = _sanitize_options(analysis.list_cells(base_dir), g.get("cell_name"))
    cell_dd = widgets.Dropdown(
        options=cells,
        value=_pick_value(cells, g.get("cell_name")),
        description="Cell",
    )
    tunes = _sanitize_options(analysis.list_tunes(base_dir, cell_dd.value), g.get("tunes_dir"))
    tunes_dd = widgets.Dropdown(
        options=tunes,
        value=_pick_value(tunes, g.get("tunes_dir")),
        description="Tunes",
    )
    models = _sanitize_options(analysis.list_models(base_dir, cell_dd.value, tunes_dd.value), g.get("model_dir"))
    model_dd = widgets.Dropdown(
        options=models,
        value=_pick_value(models, g.get("model_dir")),
        description="Model",
    )

    compare_list_sel = widgets.SelectMultiple(
        options=[],
        value=(),
        description="Compare list",
        rows=10,
    )
    compare_list_sel.layout = widgets.Layout(width="80%", height="200px")
    compare_list_paths_txt = widgets.Textarea(
        value=_compare_list_paths_text(g.get("compare_list_paths", []) or []),
        description="Compare paths",
        layout=widgets.Layout(width="80%", height="90px"),
    )
    compare_list_clear_btn = widgets.Button(description="Clear selection")
    compare_paths_cb = widgets.Checkbox(
        value=bool(g.get("compare_list_paths_enabled", True)),
        description="Use compare paths",
    )
    selection_help_btn = widgets.Button(description="Help")
    selection_help_btn.layout = widgets.Layout(width="80px", flex="0 0 auto")

    def _refresh_runs(*_):
        base = Path(g.get("CELLS_DIR")) / cell_dd.value / tunes_dd.value / model_dd.value / "output_data"
        names = [analysis.run_label(p) for p in analysis.collect_run_candidates(base)]
        options: list[tuple[str, str]] = [(n, n) for n in names]
        options.extend(_compare_list_dir_options(g, base_dir))
        compare_list_sel.options = options
        valid_vals = {val for _, val in options}
        compare_list_sel.value = tuple([n for n in compare_list_sel.value if n in valid_vals])

    def _refresh_models(*_):
        models = _sanitize_options(
            analysis.list_models(base_dir, cell_dd.value, tunes_dd.value),
            g.get("model_dir"),
        )
        model_dd.options = models
        if model_dd.value not in model_dd.options:
            model_dd.value = _pick_value(models, g.get("model_dir"))
        _refresh_runs()

    cell_dd.observe(_refresh_models, names="value")
    tunes_dd.observe(_refresh_models, names="value")
    model_dd.observe(_refresh_runs, names="value")

    def _clear_compare_list(_):
        compare_list_sel.value = ()
        compare_list_paths_txt.value = ""

    compare_list_clear_btn.on_click(_clear_compare_list)

    _refresh_runs()

    save_plots_cb = widgets.Checkbox(value=bool(g.get("save_plots")), description="Save plots")
    save_analysis_cb = widgets.Checkbox(value=bool(g.get("save_analysis")), description="Save analysis JSON")

    g["cell_dd"] = cell_dd
    g["tunes_dd"] = tunes_dd
    g["model_dd"] = model_dd
    g["save_plots_cb"] = save_plots_cb
    g["save_analysis_cb"] = save_analysis_cb
    g["compare_list_sel"] = compare_list_sel
    g["compare_list_paths_txt"] = compare_list_paths_txt
    g["compare_paths_cb"] = compare_paths_cb

    selection_help_btn.on_click(lambda *_: _print_help(out_selection, HELP_SELECTION))
    compare_list_paths_txt.disabled = not compare_paths_cb.value
    compare_paths_cb.observe(lambda *_: setattr(compare_list_paths_txt, "disabled", not compare_paths_cb.value), names="value")

    display(widgets.HBox([cell_dd, tunes_dd, model_dd]))
    display(widgets.HBox([compare_list_sel, compare_list_clear_btn]))
    display(compare_list_paths_txt)
    display(widgets.HBox([save_plots_cb, save_analysis_cb, compare_paths_cb, selection_help_btn]))
    display(out_selection)

__all__ = [
    "HELP_SELECTION",
    "build_selection_ui",
    "compare_enabled",
    "get_selection_from_globals",
    "resolve_compare",
    "resolve_compare_from_globals",
    "resolve_single",
    "resolve_single_from_globals",
    "sync_common_from_globals",
]
