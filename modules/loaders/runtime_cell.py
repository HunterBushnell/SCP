"""Metadata-only adapter for cells that exist in a notebook process."""


def validate_config(cell_config, *, base_dir=None):
    return {}


def discover_source_artifacts(cell_config, *, base_dir=None):
    return {}


def load_cell(cell_config, *, base_dir=None):
    raise RuntimeError('This workspace records a runtime-only cell (runtime_cell) that exists only in its notebook process. '
                       'Supply cell_override/from_cell, or register a saved python_factory or HOC template '
                       'before CLI/SLURM execution or cell reconstruction.')
