"""Notebook setup and teaching helper functions."""

from .helpers import (
    build_cell_for_notebook,
    build_synapse_test_cell,
    ensure_external_repo_on_syspath,
    ensure_scp_repo_on_syspath,
)
from .bootstrap import (
    check_required_external_inputs,
    ensure_modfiles,
    finish_step5_notebook_setup,
    is_colab,
)
from .synapse_preview import show_synapse_preview
from .run_diagnostics import show_run_diagnostics

__all__ = [
    "build_cell_for_notebook",
    "build_synapse_test_cell",
    "check_required_external_inputs",
    "ensure_external_repo_on_syspath",
    "ensure_modfiles",
    "ensure_scp_repo_on_syspath",
    "finish_step5_notebook_setup",
    "is_colab",
    "show_run_diagnostics",
    "show_synapse_preview",
]
