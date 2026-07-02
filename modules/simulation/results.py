"""
Compatibility facade for Step 5 result I/O helpers.

New code should prefer focused modules under `modules.simulation`; this module
keeps existing imports working while saving, loading, path, and append helpers
live in smaller files.
"""

from .result_appending import (
    _append_results_to_path,
    _ensure_multi_results,
    append_multi_results,
)
from .result_loading import _load_from_manifest, load_old_multi_results, load_results
from .result_paths import (
    _build_output_path,
    _copy_fit_json_sidecar,
    _find_fit_json_path,
    _json_default,
    _resolve_tune_path,
    _sha256_file,
    _write_json,
)
from .result_saving import (
    _save_sidecars,
    _write_results_file,
    _write_results_to_run_dir,
    save_results,
    save_results_with_name,
)


__all__ = [name for name in globals() if not name.startswith("__")]
