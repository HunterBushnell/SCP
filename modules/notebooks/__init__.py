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
from .pipeline_ui import PIPELINE_UI_DEFAULTS, PipelineNotebookUI
from .pipeline_workflow import (
    PipelineActiveResult,
    PipelineNotebookState,
    PipelinePassiveResult,
    PipelineSimulationResult,
    RestartKernelRequired,
    prepare_interactive_synapse_tuner,
    prepare_pipeline_notebook,
    run_active_stage,
    run_fresh_simulation,
    run_passive_stage,
)

__all__ = [
    "build_cell_for_notebook",
    "build_synapse_test_cell",
    "check_required_external_inputs",
    "ensure_external_repo_on_syspath",
    "ensure_modfiles",
    "ensure_scp_repo_on_syspath",
    "finish_step5_notebook_setup",
    "is_colab",
    "PIPELINE_UI_DEFAULTS",
    "PipelineNotebookUI",
    "PipelineActiveResult",
    "PipelineNotebookState",
    "PipelinePassiveResult",
    "PipelineSimulationResult",
    "RestartKernelRequired",
    "prepare_interactive_synapse_tuner",
    "prepare_pipeline_notebook",
    "run_active_stage",
    "run_fresh_simulation",
    "run_passive_stage",
    "show_run_diagnostics",
    "show_synapse_preview",
]
