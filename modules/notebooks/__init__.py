"""Notebook setup and teaching helper functions."""

from .helpers import (
    build_cell_for_notebook,
    build_synapse_test_cell,
    ensure_external_repo_on_syspath,
    ensure_scp_repo_on_syspath,
)

__all__ = [
    "build_cell_for_notebook",
    "build_synapse_test_cell",
    "ensure_external_repo_on_syspath",
    "ensure_scp_repo_on_syspath",
]
