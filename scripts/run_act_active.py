#!/usr/bin/env python3
"""Prepare or run ACT active-tuning modules from an SCP tune workspace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from modules.tuning import (
    ACT_ACTIVE_CONFIG_NAME,
    default_act_workspace,
    evaluate_act_predictions,
    prepare_act_active_workspace,
    resolve_active_tuning_targets,
    resolve_repo_root,
    resolve_tune_dir,
    run_act_active_modules,
    workspace_summary,
)


def _parse_float_list(raw: Optional[str]) -> Optional[list[float]]:
    if raw in (None, ""):
        return None
    return [float(part.strip()) for part in str(raw).split(",") if part.strip()]


def _parse_str_list(raw: Optional[str]) -> Optional[list[str]]:
    if raw in (None, ""):
        return None
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _resolve_config_ref(args: argparse.Namespace, repo_root: Path) -> Path:
    if args.config:
        return Path(args.config).expanduser().resolve()
    if args.workspace:
        return Path(args.workspace).expanduser().resolve()
    tune_dir = resolve_tune_dir(
        repo_root=repo_root,
        cell_name=args.cell,
        tune_name=args.tune,
        tunes_parent=args.tunes_parent,
        tune_dir_override=args.tune_dir,
    )
    return default_act_workspace(tune_dir)


def _prepare_if_requested(args: argparse.Namespace, repo_root: Path, config_ref: Path) -> Path:
    if args.config and not args.prepare:
        return config_ref

    workspace = config_ref if config_ref.name != ACT_ACTIVE_CONFIG_NAME else config_ref.parent
    config_path = workspace / ACT_ACTIVE_CONFIG_NAME
    should_prepare = args.prepare or not config_path.exists()
    if not should_prepare:
        return config_path

    tune_dir = resolve_tune_dir(
        repo_root=repo_root,
        cell_name=args.cell,
        tune_name=args.tune,
        tunes_parent=args.tunes_parent,
        tune_dir_override=args.tune_dir,
    )
    target_resolution = resolve_active_tuning_targets(
        context=SimpleNamespace(tune_dir=tune_dir),
        target_mode=args.target_mode,
        fi_currents_pA=_parse_float_list(args.fi_currents_pa),
        fi_frequencies_hz=_parse_float_list(args.fi_frequencies_hz),
        fi_csv_path=args.fi_csv,
        trace_npy_path=args.trace_npy,
        nwb_path=args.nwb,
        require_target=True,
    )
    nwb_options = dict(target_resolution.nwb_options)
    explicit_stimulus_names = _parse_str_list(args.nwb_stimulus_names)
    if explicit_stimulus_names is not None:
        nwb_options["stimulus_names"] = explicit_stimulus_names
    if args.nwb_include_negative_currents:
        nwb_options["include_negative_currents"] = True
    if args.nwb_min_current_pa is not None:
        nwb_options["min_current_pA"] = args.nwb_min_current_pa
    if args.nwb_max_current_pa is not None:
        nwb_options["max_current_pA"] = args.nwb_max_current_pa
    if args.nwb_keep_repeats:
        nwb_options["average_repeats"] = False
    if args.nwb_spike_threshold_mv is not None:
        nwb_options["spike_threshold_mV"] = args.nwb_spike_threshold_mv
    if args.nwb_refractory_ms is not None:
        nwb_options["refractory_ms"] = args.nwb_refractory_ms
    summary = prepare_act_active_workspace(
        repo_root=repo_root,
        tune_dir=tune_dir,
        cell_name=args.cell,
        tune_name=args.tune,
        workspace=workspace,
        target_mode=target_resolution.target_mode,
        fi_currents_pA=target_resolution.fi_currents_pA,
        fi_frequencies_hz=target_resolution.fi_frequencies_hz,
        fi_csv_path=target_resolution.fi_csv_path,
        trace_npy_path=target_resolution.trace_npy_path,
        nwb_path=target_resolution.nwb_path,
        nwb_stimulus_names=nwb_options["stimulus_names"],
        nwb_include_negative_currents=nwb_options["include_negative_currents"],
        nwb_min_current_pA=nwb_options["min_current_pA"],
        nwb_max_current_pA=nwb_options["max_current_pA"],
        nwb_average_repeats=nwb_options["average_repeats"],
        nwb_spike_threshold_mV=nwb_options["spike_threshold_mV"],
        nwb_refractory_ms=nwb_options["refractory_ms"],
        overwrite_config=True,
    )
    print(json.dumps(summary, indent=2))
    return Path(summary["config"])


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", default="PV", help="Cell label under cells/<CELL>.")
    parser.add_argument("--tune", default="tuned", help="Tune label under cells/<CELL>/tunes/<TUNE>.")
    parser.add_argument("--tunes-parent", default="tunes", help="Tune parent folder name.")
    parser.add_argument("--tune-dir", default=None, help="Explicit tune directory override.")
    parser.add_argument("--workspace", default=None, help="Explicit act_workspace directory.")
    parser.add_argument("--config", default=None, help="Path to act_active_config.json or workspace directory.")
    parser.add_argument("--repo-root", default=None, help="Explicit SCP repository root.")

    parser.add_argument("--prepare", action="store_true", help="Create/update workspace config and target files.")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare workspace and exit.")
    parser.add_argument("--run", action="store_true", help="Run ACT modules.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate saved predictions with ACT FI runs.")
    parser.add_argument("--module", default="all", help="Module to run: all, lto, spiking, or bursting.")
    parser.add_argument("--n-cpus", type=int, default=None, help="Override configured ACT CPU count.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing ACT outputs for reruns.")

    parser.add_argument(
        "--target-mode",
        choices=["fi_arrays", "fi_csv", "allen_nwb", "trace_npy"],
        default=None,
        help="Target-data source used when preparing the workspace. Defaults to target_config.json.",
    )
    parser.add_argument("--fi-currents-pa", default=None, help="Comma-separated FI currents in pA.")
    parser.add_argument("--fi-frequencies-hz", default=None, help="Comma-separated FI firing rates in Hz.")
    parser.add_argument("--fi-csv", default=None, help="CSV with FI currents/frequencies.")
    parser.add_argument("--nwb", default=None, help="Allen/ADB ephys NWB file for extracting FI targets.")
    parser.add_argument(
        "--nwb-stimulus-names",
        default=None,
        help="Comma-separated NWB stimulus names to use for FI extraction.",
    )
    parser.add_argument(
        "--nwb-include-negative-currents",
        action="store_true",
        help="Include negative-current sweeps when extracting NWB FI targets.",
    )
    parser.add_argument("--nwb-min-current-pa", type=float, default=None, help="Minimum NWB current amplitude in pA.")
    parser.add_argument("--nwb-max-current-pa", type=float, default=None, help="Maximum NWB current amplitude in pA.")
    parser.add_argument(
        "--nwb-keep-repeats",
        action="store_true",
        help="Keep repeated sweeps as separate target rows instead of averaging by current amplitude.",
    )
    parser.add_argument("--nwb-spike-threshold-mv", type=float, default=None, help="Spike threshold for NWB FI extraction.")
    parser.add_argument("--nwb-refractory-ms", type=float, default=None, help="Spike refractory window for NWB FI extraction.")
    parser.add_argument("--trace-npy", default=None, help="ACT-compatible target trace .npy file.")

    args = parser.parse_args(argv)
    repo_root = resolve_repo_root(Path(args.repo_root).expanduser() if args.repo_root else REPO_ROOT_HINT)
    config_ref = _resolve_config_ref(args, repo_root)
    config_path = _prepare_if_requested(args, repo_root, config_ref)

    if args.prepare_only:
        return 0

    if args.run:
        modules = args.module if args.module == "all" else [args.module]
        results = run_act_active_modules(
            config_path,
            modules=modules,
            n_cpus=args.n_cpus,
            overwrite=args.overwrite,
        )
        print(json.dumps(results, indent=2))

    if args.evaluate:
        result = evaluate_act_predictions(
            config_path,
            n_cpus=args.n_cpus,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, indent=2))

    if not args.run and not args.evaluate:
        print(json.dumps(workspace_summary(config_path), indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
