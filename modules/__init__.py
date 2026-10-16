"""
Local modules for the SCP single-cell pipeline.

Primary packages:
- setup: input cell download and tune-directory preparation.
- model: cell loading, geometry grouping, and synapse attachment.
- input_generation: Step 5.2.3 input config normalization and spike-train generation.
- simulation: Step 5 simulation backend for notebooks, CLI, and SLURM.
- analysis: plotting and analysis utilities.
- notebooks: shared notebook setup/build helpers.
- tuning: shared Step 2/3 notebook setup, ACT integration, and proposal export helpers.
- core: shared utilities such as reproducible randomness.

The `modules.run_sim` facade remains as the stable notebook/API entry point
while implementation code lives under `modules.simulation`.
"""

import importlib
import importlib.abc
import importlib.util
import sys
from pathlib import Path


if __name__ == "modules":
    # Checkout-only compatibility. Installed distributions expose only
    # scp_pipeline; ordinary imports never seize a host project's `modules`.
    try:
        canonical = importlib.import_module("scp_pipeline")
    except ModuleNotFoundError as exc:
        if exc.name != "scp_pipeline":
            raise
        spec = importlib.util.spec_from_file_location(
            "scp_pipeline", __file__, submodule_search_locations=[str(Path(__file__).parent)])
        canonical = importlib.util.module_from_spec(spec)
        sys.modules["scp_pipeline"] = canonical
        spec.loader.exec_module(canonical)

    class _LegacyLoader(importlib.abc.Loader):
        def __init__(self, target):
            self.target = target
            self.identity = {key: getattr(target, key, None) for key in
                             ("__name__", "__package__", "__spec__", "__loader__")}

        def create_module(self, spec):
            return self.target

        def exec_module(self, module):
            module.__dict__.update(self.identity)
            # Package initialization may export a function sharing its child's
            # name (model.load_cell). Pre-alias loaded descendants so a later
            # legacy import cannot overwrite that canonical function export.
            prefix = self.identity['__name__'] + '.'
            for name, child in list(sys.modules.items()):
                if name.startswith(prefix):
                    sys.modules['modules' + name[len('scp_pipeline'):]] = child

    class _LegacyFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if not fullname.startswith("modules."):
                return None
            name = "scp_pipeline" + fullname[len("modules"):]
            module = importlib.import_module(name)
            return importlib.util.spec_from_loader(
                fullname, _LegacyLoader(module), is_package=hasattr(module, "__path__"))

    sys.meta_path.insert(0, _LegacyFinder())
    sys.modules["modules"] = canonical
    for name, child in list(sys.modules.items()):
        if name.startswith('scp_pipeline.'):
            sys.modules['modules' + name[len('scp_pipeline'):]] = child


__all__ = ["load_cell"]


def __getattr__(name):
    """Lazy convenience exports without importing NEURON/model code at package import."""
    if name == "load_cell":
        from .model.load_cell import load_cell

        return load_cell
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
