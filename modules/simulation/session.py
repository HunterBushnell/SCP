from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Union

from modules.core import randomness
from modules.input_generation import inputs
from modules.input_generation.config import (
    _inject_path_metadata,
    _normalize_sim_config,
    _resolve_config_root,
)
from modules.model import geometry, synapses
from modules.model.load_cell import load_cell

from .current_injection import run_iclamp_test
from .multi_run import run_multi
from .result_loading import load_results
from .result_saving import save_results
from .single_run import run_single
from .session_setup import (
    _load_json_dict,
    _resolve_append_target,
    _timestamp_stem,
    infer_cell_name,
    load_mechanisms,
    normalize_tune_dir,
    resolve_loader_manifest_path,
)


def _merge_overrides(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively merge non-None notebook/CLI overrides into a config dict."""
    if not overrides:
        return base
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            nested = dict(base.get(key, {}) or {})
            base[key] = _merge_overrides(nested, value)
        elif isinstance(value, dict):
            base[key] = _merge_overrides({}, value)
        else:
            base[key] = value
    return base


@dataclass
class SimulationOptions:
    mode: Optional[str] = None
    n_trials: Optional[int] = None
    seed: Optional[int] = None
    trial_offset: Optional[int] = None
    sim_overrides: Optional[Dict[str, Any]] = None
    iclamp: bool = False
    snapshot: bool = False
    force_save: bool = False
    output_dir: Optional[Union[str, Path]] = None
    output_stem: Optional[str] = None
    load_mechanisms: bool = True


@dataclass
class SimulationSession:
    tune_dir: Path
    options: SimulationOptions = field(default_factory=SimulationOptions)
    config_root: Path = field(init=False)
    sim_path: Path = field(init=False)
    syn_path: Path = field(init=False)
    cell_config_path: Path = field(init=False)
    geom_config_path: Path = field(init=False)
    sim_cfg_preview: Dict[str, Any] = field(default_factory=dict, init=False)
    cell_config: Dict[str, Any] = field(default_factory=dict, init=False)
    geom_config: Dict[str, Any] = field(default_factory=dict, init=False)
    sim_cfg_override: Optional[Dict[str, Any]] = field(default=None, init=False)
    cell: Any = field(default=None, init=False)
    geom: Any = field(default=None, init=False)
    sim_cfg: Optional[Dict[str, Any]] = field(default=None, init=False)
    groups_cfg: Optional[Dict[str, Any]] = field(default=None, init=False)
    inputs_by_group: Optional[Dict[str, Any]] = field(default=None, init=False)
    randomness_manager: Optional[randomness.RandomnessManager] = field(default=None, init=False)
    result: Optional[Dict[str, Any]] = field(default=None, init=False)
    saved_path: Optional[Path] = field(default=None, init=False)
    iclamp_cfg: Dict[str, Any] = field(default_factory=dict, init=False)
    iclamp_enabled: bool = field(default=False, init=False)
    snapshot_enabled: bool = field(default=False, init=False)
    prepared: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.tune_dir = Path(self.tune_dir).resolve()

    @classmethod
    def from_tune(
        cls,
        tune_dir: Union[str, Path],
        options: Optional[SimulationOptions] = None,
        **option_overrides: Any,
    ) -> "SimulationSession":
        if options is None:
            options = SimulationOptions(**option_overrides)
        elif option_overrides:
            options = replace(options, **option_overrides)
        return cls(Path(tune_dir), options=options).load_configs()

    @property
    def cell_name(self) -> str:
        return infer_cell_name(self.tune_dir, self.cell_config)

    @property
    def tune_name(self) -> str:
        return self.tune_dir.name

    @property
    def output_dir(self) -> Path:
        if self.options.output_dir:
            return Path(self.options.output_dir)
        return self.tune_dir / "output_data"

    def load_configs(self) -> "SimulationSession":
        self.config_root = _resolve_config_root(self.tune_dir)
        self.sim_path = self.config_root / "sim_config.json"
        self.syn_path = self.config_root / "syn_config.json"
        self.cell_config_path = self.tune_dir / "cell_configs" / "cell_config.json"
        self.geom_config_path = self.tune_dir / "cell_configs" / "geometry.json"

        if not self.sim_path.is_file():
            raise FileNotFoundError(
                f"Missing sim_config.json in {self.config_root}. Resolved tune_dir: {self.tune_dir}"
            )
        if not self.syn_path.is_file():
            raise FileNotFoundError(f"Missing syn_config.json in {self.config_root}")

        self.sim_cfg_preview = _load_json_dict(self.sim_path)
        self.cell_config = _load_json_dict(self.cell_config_path)
        self.geom_config = _load_json_dict(self.geom_config_path)

        self.cell_config.setdefault("cell_name", infer_cell_name(self.tune_dir, self.cell_config))
        paths = self.cell_config.setdefault("paths", {})
        if not isinstance(paths, dict):
            paths = {}
            self.cell_config["paths"] = paths
        paths.setdefault("manifest", "manifest.json")
        paths.setdefault("tune_dir", str(self.tune_dir))
        self._resolve_loader_paths()

        return self

    def _resolve_loader_paths(self) -> None:
        resolve_loader_manifest_path(
            cell_config=self.cell_config,
            cell_config_path=self.cell_config_path,
            tune_dir=self.tune_dir,
        )

    def _resolve_append_override(self) -> None:
        append_target = _resolve_append_target(self.sim_cfg_preview, self.tune_dir / "output_data")
        self.sim_cfg_override = None
        if append_target is None:
            return

        if append_target.name == "run_manifest.json":
            append_target = append_target.parent
        if not append_target.exists():
            print(f"append_to target not found yet: {append_target} (using local sim_config.json)")
            return

        base_results = load_results(append_target)
        base_sim_cfg = base_results.get("sim_cfg", {}) or {}
        base_cell = base_sim_cfg.get("cell")
        base_tune = base_sim_cfg.get("tune")
        if base_cell and base_cell != self.cell_name:
            raise ValueError(
                f"append_to points to cell {base_cell!r} but tune_dir is {self.cell_name!r}"
            )
        if base_tune and base_tune != self.tune_name:
            raise ValueError(
                f"append_to points to tune {base_tune!r} but tune_dir is {self.tune_name!r}"
            )

        self.sim_cfg_override = copy.deepcopy(base_sim_cfg)
        self.sim_cfg_override["append"] = (
            self.sim_cfg_preview.get("append")
            if "append" in self.sim_cfg_preview
            else self.sim_cfg_preview.get("append_to")
        )

    def _apply_snapshot_option(self) -> None:
        sim_cfg_for_cell = self.sim_cfg_override or self.sim_cfg_preview
        snapshot_raw = sim_cfg_for_cell.get("snapshot", None)
        self.snapshot_enabled = False
        if isinstance(snapshot_raw, dict):
            self.snapshot_enabled = bool(snapshot_raw.get("enabled", False))
        elif snapshot_raw is True:
            self.snapshot_enabled = True

        if self.options.snapshot:
            if not isinstance(snapshot_raw, dict):
                snapshot_raw = {}
            snapshot_raw["enabled"] = True
            sim_cfg_for_cell["snapshot"] = snapshot_raw
            self.snapshot_enabled = True
            if self.sim_cfg_override is None:
                self.sim_cfg_override = sim_cfg_for_cell

    def _apply_cell_tuning_fallback(self) -> None:
        sim_cfg_for_cell = self.sim_cfg_override or self.sim_cfg_preview
        tuning = self.cell_config.setdefault("tuning", {})
        if not isinstance(tuning, dict):
            tuning = {}
            self.cell_config["tuning"] = tuning
        if "soma_diam_multiplier" not in tuning:
            tuning["soma_diam_multiplier"] = sim_cfg_for_cell.get("soma_diam_multiplier", 1.0)
        if sim_cfg_for_cell.get("soma_diam_multiplier") is not None:
            tuning["soma_diam_multiplier"] = sim_cfg_for_cell.get("soma_diam_multiplier", 1.0)

    def _apply_run_overrides(self, sim_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Apply options that affect input generation and simulation behavior."""
        _merge_overrides(sim_cfg, self.options.sim_overrides)

        if self.options.n_trials is not None:
            sim_cfg["n_trials"] = int(self.options.n_trials)
        if self.options.trial_offset is not None:
            sim_cfg["trial_offset"] = int(self.options.trial_offset)

        if self.options.seed is not None:
            sim_cfg["seed"] = int(self.options.seed)
            rand_cfg = sim_cfg.get("randomness", {})
            if not isinstance(rand_cfg, dict):
                rand_cfg = {}
            global_cfg = rand_cfg.get("global")
            if not isinstance(global_cfg, dict):
                global_cfg = {}
            global_cfg["seed"] = int(self.options.seed)
            rand_cfg["global"] = global_cfg
            sim_cfg["randomness"] = rand_cfg

        return sim_cfg

    def _apply_options_to_sim_cfg(self, sim_cfg: Dict[str, Any]) -> Dict[str, Any]:
        self._apply_run_overrides(sim_cfg)

        save_output = sim_cfg.get("save_output", True)
        if save_output is None:
            save_output = True
        save_output = bool(save_output)
        if self.options.force_save:
            save_output = True
        sim_cfg["save_output"] = save_output

        output_stem = sim_cfg.get("output_stem")
        if output_stem not in (None, ""):
            sim_cfg["output"] = output_stem

        if save_output:
            if not sim_cfg.get("output"):
                sim_cfg["output"] = self.options.output_stem or _timestamp_stem()
            elif self.options.output_stem:
                sim_cfg["output"] = self.options.output_stem
        elif self.options.output_stem and not sim_cfg.get("output"):
            sim_cfg["output"] = self.options.output_stem

        return sim_cfg

    def prepare(self, *, generate_inputs: bool = True) -> "SimulationSession":
        if self.options.load_mechanisms:
            load_mechanisms(self.tune_dir)

        self._resolve_append_override()
        self._apply_snapshot_option()
        self._apply_cell_tuning_fallback()

        self.cell = load_cell(self.cell_config)
        self.geom = geometry.define_geometry(self.cell, self.geom_config)

        sim_cfg_for_mode = self._apply_run_overrides(
            copy.deepcopy(self.sim_cfg_override or self.sim_cfg_preview)
        )
        self.iclamp_cfg = sim_cfg_for_mode.get("iclamp", {}) or {}
        self.iclamp_enabled = bool(self.iclamp_cfg.get("enabled", False)) or bool(self.options.iclamp)
        if self.snapshot_enabled and self.iclamp_enabled:
            print("Snapshot mode ignored because IClamp is enabled.")
            self.snapshot_enabled = False
            sim_cfg_for_cell = self.sim_cfg_override or self.sim_cfg_preview
            if isinstance(sim_cfg_for_cell.get("snapshot"), dict):
                sim_cfg_for_cell["snapshot"]["enabled"] = False

        if self.iclamp_enabled:
            sim_cfg_raw = copy.deepcopy(self.sim_cfg_override or self.sim_cfg_preview)
            self._apply_run_overrides(sim_cfg_raw)
            self.sim_cfg = _normalize_sim_config(sim_cfg_raw)
            _inject_path_metadata(self.sim_cfg, self.config_root)
            self.sim_cfg = self._apply_options_to_sim_cfg(self.sim_cfg)
            self.groups_cfg = {}
            self.inputs_by_group = {}
        elif generate_inputs:
            sim_cfg_for_inputs = self._apply_run_overrides(
                copy.deepcopy(self.sim_cfg_override or self.sim_cfg_preview)
            )
            self.sim_cfg, self.groups_cfg, self.inputs_by_group = inputs.generate_inputs(
                path=self.tune_dir,
                geometry=self.geom,
                seed_override=self.options.seed,
                sim_cfg_override=sim_cfg_for_inputs,
            )
            self.sim_cfg = self._apply_options_to_sim_cfg(self.sim_cfg)

        self.prepared = True
        return self

    def run(self, *, trial_callback: Optional[Any] = None) -> Dict[str, Any]:
        if not self.prepared:
            self.prepare()
        if self.sim_cfg is None:
            raise RuntimeError("SimulationSession.run called before simulation config was prepared.")

        if self.iclamp_enabled:
            self.result = run_iclamp_test(
                self.cell,
                self.sim_cfg,
                iclamp_cfg=self.iclamp_cfg,
            )
            return self.result

        if self.groups_cfg is None or self.inputs_by_group is None:
            raise RuntimeError("Synapse-driven run requires groups_cfg and inputs_by_group.")

        self.randomness_manager = randomness.RandomnessManager(self.sim_cfg)
        mode = self.options.mode
        if mode is None:
            n_trials_eff = int(self.sim_cfg.get("n_trials", 1) or 1)
            mode = "multi" if n_trials_eff > 1 else "single"

        if mode == "single":
            self.result = run_single(
                cell=self.cell,
                geom=self.geom,
                sim_cfg=self.sim_cfg,
                groups_cfg=self.groups_cfg,
                inputs_by_group=self.inputs_by_group,
                rm=self.randomness_manager,
            )
        elif mode == "multi":
            self.result = run_multi(
                cell=self.cell,
                geom=self.geom,
                sim_cfg=self.sim_cfg,
                groups_cfg=self.groups_cfg,
                inputs_by_group=self.inputs_by_group,
                rm=self.randomness_manager,
                trial_callback=trial_callback,
            )
        else:
            raise ValueError(f"Unsupported simulation mode: {mode!r}")

        meta = self.result.setdefault("meta", {})
        meta["randomness"] = self.randomness_manager.meta().as_dict()
        if self.snapshot_enabled:
            meta["cell_config"] = copy.deepcopy(self.cell_config)
            meta["geometry_config"] = copy.deepcopy(self.geom_config)

        return self.result

    def save(self) -> Optional[Path]:
        if self.result is None:
            raise RuntimeError("SimulationSession.save called before run().")
        self.saved_path = save_results(self.result, base_dir=self.output_dir)
        return self.saved_path

    def run_and_save(self, *, trial_callback: Optional[Any] = None) -> Dict[str, Any]:
        result = self.run(trial_callback=trial_callback)
        self.save()
        return result

    def preview_synapses(self, *, trial_idx: int = 0) -> Dict[str, Any]:
        if not self.prepared:
            self.prepare()
        if self.iclamp_enabled:
            raise RuntimeError("IClamp mode does not generate synapses.")
        if self.sim_cfg is None or self.groups_cfg is None or self.inputs_by_group is None:
            raise RuntimeError("Cannot preview synapses before inputs are prepared.")

        rm = randomness.RandomnessManager(self.sim_cfg)
        trial_rng = rm.trial(int(trial_idx))
        return synapses.preview_synapses(
            cell=self.cell,
            geom=self.geom,
            sim_cfg=self.sim_cfg,
            groups_cfg=self.groups_cfg,
            inputs_by_group=self.inputs_by_group,
            trial_rng=trial_rng,
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "tune_dir": str(self.tune_dir),
            "config_root": str(self.config_root),
            "cell": self.cell_name,
            "tune": self.tune_name,
            "iclamp_enabled": self.iclamp_enabled,
            "snapshot_enabled": self.snapshot_enabled,
            "mode": self.options.mode,
            "n_trials": None if self.sim_cfg is None else self.sim_cfg.get("n_trials"),
            "output": None if self.sim_cfg is None else self.sim_cfg.get("output"),
            "output_dir": str(self.output_dir),
            "saved_path": None if self.saved_path is None else str(self.saved_path),
        }
