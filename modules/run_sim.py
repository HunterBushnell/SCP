"""
Stable facade for Step 5 simulation helpers.

New code should prefer focused modules under `modules.simulation`; this module
keeps the notebook, script, and analysis import surface concise.
"""

from .simulation.cell_runtime import (
    _build_cell_recorders_for_site,
    _get_cell_recording_cfg,
    _get_soma_segment,
    _normalize_runtime_recording_site,
    _parse_bool_like,
    _resolve_recording_site,
    run_cell,
)
from .simulation.current_injection import (
    _get_hoc,
    get_frequency,
    get_rec_vars_for_i_in_sec,
    looped_current_injection,
    plot_looped_currents,
    run_FI,
    run_current_injection,
    run_iclamp_test,
)
from .simulation.result_helpers import (
    _aggregate_input_stats,
    _resolve_inputs_to_save,
    _resolve_trace_trials_to_save,
    _smooth_rate_curve,
)
from .simulation.result_paths import (
    _build_output_path,
    _copy_fit_json_sidecar,
    _find_fit_json_path,
    _json_default,
    _resolve_tune_path,
    _sha256_file,
    _write_json,
)
from .simulation.result_loading import _load_from_manifest, load_results
from .simulation.results import (
    _append_results_to_path,
    _ensure_multi_results,
    _save_sidecars,
    _write_results_file,
    _write_results_to_run_dir,
    append_multi_results,
    save_results,
    save_results_with_name,
)
from .simulation.runner import (
    _infer_mode,
    run_multi,
    run_param,
    run_sim,
    run_single,
    summarize_results,
)
from .simulation.snapshots import (
    _apply_snapshot_deterministic,
    _collect_env_snapshot,
    _collect_mechanism_info,
    _collect_neuron_state,
    _collect_versions,
    _snapshot_cfg,
    _snapshot_netcon_state,
    _snapshot_synapse_params,
)
from .simulation.trial_helpers import (
    _as_bool,
    _clear_cell_state,
    _coerce_bin_width,
    _compute_input_stats_for_trial,
    _detect_spikes,
    _prepare_input_stats_bins,
    _set_trace_trials_to_save,
    _warn_preexisting_synapses,
)


__all__ = [name for name in globals() if not name.startswith('__')]
