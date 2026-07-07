"""Step 1 tune-preparation orchestration.

This module remains the notebook/script import surface. Implementation details are
split into focused setup modules for defaults, scaffolding, mechanisms, fit JSON
cleanup, and validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from modules.setup.adb import download_ADB_cell

from .defaults import (
    default_cell_config,
    default_geometry_config,
    default_placeholder_syn_group,
    default_sim_config,
    default_syn_config,
    guess_cell_color,
    guess_soma_multiplier,
    guess_specimen_from_cell,
)
from .fit_json import (
    coerce_fit_genome_values_to_numeric,
    find_fit_json,
    mechanisms_declared_in_fit_json,
    sort_genome_by_section,
)
from .mechanisms import compile_modfiles, find_compiled_mechanism_dll
from .paths import Step1Paths, resolve_step1_paths
from .scaffold import scaffold_common_configs
from .validation import validate_tune


def prepare_tune(
    *,
    tune_dir: Path,
    cell_name: str,
    tune_name: str,
    specimen_id: int,
    model_type: str = "perisomatic",
    soma_diam_multiplier: Optional[float] = None,
    color: Optional[str] = None,
    do_download: bool = True,
    force_download: bool = False,
    cache_stimulus: bool = False,
    download_match: str = "contains",
    do_compile_modfiles: bool = True,
    recompile_modfiles: bool = False,
    load_compiled_dll: bool = True,
    coerce_genome_values_to_numeric: bool = False,
    sort_genome_entries_by_section: bool = False,
    do_scaffold_configs: bool = True,
    config_mode: str = "fill",
    sync_cell_metadata: bool = True,
    do_validate: bool = True,
    validate_inputs_cfg: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end Step-1 preparation entrypoint.

    Returns a summary dictionary with actions and key paths.
    """
    tune_dir = Path(tune_dir).expanduser().resolve()
    tune_dir.mkdir(parents=True, exist_ok=True)

    soma_mult = (
        float(soma_diam_multiplier)
        if soma_diam_multiplier is not None
        else float(guess_soma_multiplier(cell_name))
    )

    summary: Dict[str, Any] = {
        "tune_dir": str(tune_dir),
        "cell_name": cell_name,
        "tune_name": tune_name,
        "specimen_id": int(specimen_id),
        "model_type": model_type,
        "soma_diam_multiplier": soma_mult,
        "actions": {},
    }

    if do_download:
        dl_info = download_ADB_cell(
            specimen_id=int(specimen_id),
            model_type=model_type,
            tunes_dir=str(tune_dir),
            subdir=None,
            cache_stimulus=cache_stimulus,
            match=download_match,
            quiet=False,
            force=force_download,
        )
        summary["actions"]["download"] = {
            "status": "ok",
            "model_id": int(dl_info.get("model_id")),
            "model_name": dl_info.get("model_name"),
            "n_files": int(len(dl_info.get("files", []))),
        }
    else:
        summary["actions"]["download"] = {"status": "skipped"}

    if coerce_genome_values_to_numeric:
        summary["actions"]["coerce_genome_values"] = coerce_fit_genome_values_to_numeric(tune_dir)
    else:
        summary["actions"]["coerce_genome_values"] = {"status": "skipped"}

    if sort_genome_entries_by_section:
        summary["actions"]["sort_genome"] = sort_genome_by_section(tune_dir)
    else:
        summary["actions"]["sort_genome"] = {"status": "skipped"}

    if do_compile_modfiles:
        compile_info = compile_modfiles(
            tune_dir,
            recompile=recompile_modfiles,
            load_dll=load_compiled_dll,
        )
        summary["actions"]["compile_modfiles"] = {
            "status": "ok",
            **compile_info,
        }
    else:
        summary["actions"]["compile_modfiles"] = {"status": "skipped"}

    if do_scaffold_configs:
        cfg_status = scaffold_common_configs(
            tune_dir=tune_dir,
            cell_name=cell_name,
            tune_name=tune_name,
            specimen_id=int(specimen_id),
            model_type=model_type,
            soma_diam_multiplier=soma_mult,
            color=color,
            config_mode=config_mode,
            sync_cell_metadata=sync_cell_metadata,
        )
        summary["actions"]["scaffold_configs"] = {
            "status": "ok",
            "files": cfg_status,
        }
    else:
        summary["actions"]["scaffold_configs"] = {"status": "skipped"}

    if do_validate:
        checks = validate_tune(
            tune_dir=tune_dir,
            cell_name=cell_name,
            soma_diam_multiplier=soma_mult,
            validate_modfiles=True,
            validate_load_cell=True,
            validate_inputs=validate_inputs_cfg,
        )
        summary["actions"]["validate"] = {
            "status": "ok",
            "checks": checks,
        }
    else:
        summary["actions"]["validate"] = {"status": "skipped"}

    return summary
