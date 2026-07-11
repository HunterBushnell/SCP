"""Default Step 1 config templates for newly prepared tune directories."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


DEFAULT_SYNAPSE_TEMPLATE_KINDS = (
    "input_blocks",
)

SYNAPSE_TEMPLATE_FILENAMES = {
    "input_blocks": "input_blocks_template.json",
}


def guess_specimen_from_cell(cell_name: str) -> Optional[int]:
    """Return a default specimen_id for common SCP cell labels."""
    key = (cell_name or "").strip().upper()
    defaults = {
        "PV": 484635029,
        "SST": 485466109,
    }
    return defaults.get(key)


def guess_soma_multiplier(cell_name: str) -> float:
    """Return a default soma diameter multiplier for common SCP labels."""
    key = (cell_name or "").strip().upper()
    defaults = {
        "PV": 6.0,
        "SST": 8.0,
    }
    return float(defaults.get(key, 1.0))


def guess_cell_color(cell_name: str) -> str:
    """Return a default plot color used in existing SCP cell_config files."""
    key = (cell_name or "").strip().upper()
    defaults = {
        "PV": "royalblue",
        "SST": "mediumorchid",
    }
    return defaults.get(key, "k")


def default_cell_config(
    *,
    cell_name: str,
    tune_name: str,
    specimen_id: Optional[int],
    model_type: str,
    soma_diam_multiplier: float,
    color: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "cell_name": cell_name,
        "tune": tune_name,
        "color": color if color is not None else guess_cell_color(cell_name),
        "cell_loader": "allen_manifest",
        "paths": {
            "manifest": "manifest.json",
        },
        "tuning": {
            "soma_diam_multiplier": float(soma_diam_multiplier),
        },
    }


def default_sim_config(*, cell_name: str, specimen_id: Optional[int], model_type: str) -> Dict[str, Any]:
    return {
        "tstart": 0.0,
        "tstop": 1000.0,
        "dt": 0.025,
        "bins": 5.0,
        "jitter": None,
        "stim_start_ms": 300.0,
        "stim_duration_ms": 500.0,
        "n_trials": 1,
        "n_traces_to_save": 1,
        "n_inputs_to_save": 1,
        "load": {
            "enabled": False,
            "path": None,
        },
        "save": {
            "enabled": False,
            "stem": "results",
            "format": "pkl",
            "full_results": False,
        },
        "append": {
            "enabled": False,
            "path": None,
        },
        "save_input_stats": True,
        "input_stats_bin_ms": 5.0,
        "avg_rate_curve_smooth_ms": 25.0,
        "avg_rate_curve_smooth_mode": "center",
        "plots_win_size": 25.0,
        "plots_input_bin_ms": None,
        "plots_input_smooth_ms": 25.0,
        "plots_raster_style": "dot",
        "plots_profile": "basic",
        "save_plots_mode": "single_plot",
        "save_plots_single_plot_preset": "modules/analysis/analysis_presets/single_plot.json",
        "save_plots_overwrite": False,
        "randomness_mode": "random",
        "seed": None,
        "iclamp": {
            "enabled": False,
            "amp_nA": 0.2,
            "delay_ms": None,
            "dur_ms": None,
            "tstop_ms": None,
            "dt_ms": None,
            "record_currents": False,
        },
        "cell_recording": {
            "enabled": False,
            "n_trials": 1,
            "sites": [
                {"sec": "soma", "idx": 0, "x": 0.5},
            ],
            "vars": {
                "v": True,
                "i_cap": False,
                "ion_currents": False,
                "mech_currents": False,
                "ion_concentrations": False,
                "ion_reversals": False,
                "mech_conductances": False,
                "mech_states": False,
            },
        },
        "syn_recording": {
            "enabled": False,
            "default_mode": "group",
            "default_sample_per_group": 10,
            "default_vars": {
                "i": True,
                "g": True,
                "i_AMPA": False,
                "i_NMDA": False,
                "g_AMPA": False,
                "g_NMDA": False,
                "record_use": False,
                "record_Pr": False,
            },
            "groups": {},
        },
        "snapshot": {
            "enabled": False,
            "n_trials": 1,
            "save_all_inputs": True,
            "save_all_traces": True,
            "save_syn_records_by_trial": True,
        },
    }


def default_geometry_config(*, cell_name: str) -> Dict[str, Any]:
    label = f"{cell_name}_default_geometry"
    return {
        "distance_origin": {"kind": "soma", "x": 0.5},
        "thresholds_um": {
            "proximal": {"low": 20.0, "high": 100.0},
            "distal": {"low": 100.0, "high": None},
        },
        "label": label,
    }


def default_synapse_weight_params(*, weight_style: str = "distributed") -> Dict[str, Any]:
    """Return generic weight params for generated synapse templates."""
    style = str(weight_style or "distributed").strip().lower()
    if style == "fixed":
        return {"initW": 1.0}
    if style == "distributed":
        return {"wt_mean": 1.0, "wt_std": 0.0}
    raise ValueError("weight_style must be 'fixed' or 'distributed'")


def _default_syns_template(
    *,
    syn_type: str,
    n_syn: Optional[int],
    segs: str,
    dist_func: Dict[str, Any],
    weight_style: str,
) -> Dict[str, Any]:
    return {
        "type": syn_type,
        "N_syn": n_syn,
        "segs": segs,
        "dist_func": dist_func,
        "params": default_synapse_weight_params(weight_style=weight_style),
    }


def _default_input_blocks_template() -> list[Dict[str, Any]]:
    return [
        {
            "name": "pre_baseline",
            "role": "baseline",
            "start_ms": 0.0,
            "stop_ms": 300.0,
            "mode": "homogeneous_poisson",
            "rate_hz": 2.0,
        },
        {
            "name": "stimulus",
            "role": "stimulus",
            "start_ms": 300.0,
            "stop_ms": 800.0,
            "mode": "inhomogeneous_poisson",
            "source": {
                "path": "external_data/pyrFiringRateAvg.csv",
                "time_col": "Time",
                "rate_col": "AvgFiringRate",
                "bin_ms": 5.0,
                "crop_start_ms": 0.0,
                "crop_stop_ms": 500.0,
            },
        },
        {
            "name": "post_baseline",
            "role": "baseline",
            "start_ms": 800.0,
            "stop_ms": 1000.0,
            "mode": "homogeneous_poisson",
            "rate_hz": 2.0,
        },
    ]


def default_syn_group_template(
    *,
    template_kind: str,
    group_name: Optional[str] = None,
    weight_style: str = "distributed",
) -> Dict[str, Any]:
    """Return one disabled synapse-group template for Step 1 scaffolding."""
    kind = str(template_kind or "").strip().lower()
    if kind not in DEFAULT_SYNAPSE_TEMPLATE_KINDS:
        allowed = ", ".join(DEFAULT_SYNAPSE_TEMPLATE_KINDS)
        raise ValueError(f"Unknown synapse template kind {template_kind!r}; expected one of: {allowed}")

    name = group_name or f"{kind}_template"

    return {
        name: {
            "state": False,
            "color": "#1f77b4",
            "input_blocks": _default_input_blocks_template(),
            "syns": _default_syns_template(
                syn_type="AMPA_NMDA_STP",
                n_syn=0,
                segs="all",
                dist_func={"kind": "uniform", "params": {"c": 1.0}},
                weight_style=weight_style,
            ),
        }
    }


def default_syn_group_templates(
    *,
    template_kinds: Optional[Iterable[str]] = None,
    weight_style: str = "distributed",
) -> Dict[str, Dict[str, Any]]:
    """Return filename -> template payload for selected Step 1 synapse templates."""
    kinds = list(DEFAULT_SYNAPSE_TEMPLATE_KINDS if template_kinds is None else template_kinds)
    templates: Dict[str, Dict[str, Any]] = {}
    for kind_raw in kinds:
        kind = str(kind_raw).strip().lower()
        if not kind:
            continue
        filename = SYNAPSE_TEMPLATE_FILENAMES.get(kind)
        if filename is None:
            allowed = ", ".join(DEFAULT_SYNAPSE_TEMPLATE_KINDS)
            raise ValueError(f"Unknown synapse template kind {kind_raw!r}; expected one of: {allowed}")
        templates[filename] = default_syn_group_template(
            template_kind=kind,
            weight_style=weight_style,
        )
    return templates


def default_syn_config(*, include_path: str = "syn_groups/input_blocks_template.json") -> Dict[str, Any]:
    return {
        "group_files": [include_path],
    }


def default_syn_config_for_templates(*, filenames: Iterable[str]) -> Dict[str, Any]:
    return {
        "group_files": [f"syn_groups/{filename}" for filename in filenames],
    }
