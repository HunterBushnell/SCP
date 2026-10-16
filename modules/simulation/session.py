from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Union

from scp_pipeline.core import randomness
from scp_pipeline.input_generation import inputs
from scp_pipeline.input_generation.config import (
    _inject_path_metadata,
    _normalize_sim_config,
    _resolve_config_root,
)
from scp_pipeline.model import geometry, synapses
from scp_pipeline.model.load_cell import load_cell

from .current_injection import apply_sim_conditions, run_iclamp_test
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
    cell_override: Any = field(default=None, repr=False)
    runtime_sim_config: Optional[Dict[str, Any]] = field(default=None, repr=False)
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
    prepared_inputs: list[Dict[str, Any]] = field(default_factory=list, init=False)
    workspace: Any = field(default=None, init=False)
    append_lineage: list[Dict[str, Any]] = field(default_factory=list, init=False)
    append_warnings: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        from scp_pipeline.workspaces import infer_workspace
        self.tune_dir = infer_workspace(self.tune_dir).tune_dir

    @classmethod
    def from_tune(
        cls,
        tune_dir: Union[str, Path],
        options: Optional[SimulationOptions] = None,
        cell_override: Any = None,
        runtime_sim_config: Optional[Dict[str, Any]] = None,
        **option_overrides: Any,
    ) -> "SimulationSession":
        if options is None:
            options = SimulationOptions(**option_overrides)
        elif option_overrides:
            options = replace(options, **option_overrides)
        return cls(Path(tune_dir), options=options, cell_override=cell_override,
                   runtime_sim_config=runtime_sim_config).load_configs()

    @classmethod
    def from_cell(cls, cell, *, tune_dir, sim_config=None, geometry_config=None, options=None):
        from scp_pipeline.model.runtime_cell import wrap_cell, prepare_runtime_workspace
        adopted = wrap_cell(cell)
        prepare_runtime_workspace(tune_dir, adopted, sim_config, geometry_config)
        return cls.from_tune(tune_dir, options=options, cell_override=adopted,
                             runtime_sim_config=sim_config)

    @property
    def cell_name(self) -> str:
        return infer_cell_name(self.tune_dir, self.cell_config)

    @property
    def tune_name(self) -> str:
        return self.workspace.tune_name if self.workspace is not None else self.tune_dir.name

    @property
    def output_dir(self) -> Path:
        if self.options.output_dir:
            output = Path(self.options.output_dir).expanduser().resolve()
            if self.cell_config.get('model_source'):
                from scp_pipeline.loaders.source_context import guard_source_destinations
                guard_source_destinations(self.cell_config, self.tune_dir, [output])
            return output
        return self.tune_dir / "output_data"

    def load_configs(self) -> "SimulationSession":
        if self.cell_override is not None:
            from scp_pipeline.model.runtime_cell import wrap_cell, prepare_runtime_workspace
            self.cell_override = wrap_cell(self.cell_override)
            prepare_runtime_workspace(self.tune_dir, self.cell_override, self.runtime_sim_config)
        self.config_root = _resolve_config_root(self.tune_dir)
        self.sim_path = self.config_root / "sim_config.json"
        self.syn_path = self.config_root / "syn_config.json"
        self.cell_config_path = self.tune_dir / "cell_configs" / "cell_config.json"
        self.geom_config_path = self.tune_dir / "cell_configs" / "geometry.json"

        if not self.sim_path.is_file():
            raise FileNotFoundError(
                f"Missing sim_config.json in {self.config_root}. Resolved tune_dir: {self.tune_dir}"
            )
        self.sim_cfg_preview = _load_json_dict(self.sim_path)
        self.cell_config = _load_json_dict(self.cell_config_path)
        self.geom_config = _load_json_dict(self.geom_config_path)
        from scp_pipeline.workspaces import infer_workspace
        self.workspace = infer_workspace(self.tune_dir, cell_config=self.cell_config)
        if self.cell_override is not None:
            from scp_pipeline.model.runtime_cell import adopt_cell
            _merge_overrides(self.sim_cfg_preview, self.runtime_sim_config)
            effective = dict(self.cell_config)
            effective['cell_name'] = self.cell_override.config.get('cell_name') or effective.get('cell_name')
            self.cell_override = adopt_cell(self.cell_override, effective)
            self.cell_config = dict(self.cell_override.config)

        self.cell_config.setdefault("cell_name", infer_cell_name(self.tune_dir, self.cell_config))
        paths = self.cell_config.setdefault("paths", {})
        if not isinstance(paths, dict):
            paths = {}
            self.cell_config["paths"] = paths
        paths["tune_dir"] = str(self.tune_dir)
        if "tune_dir" in self.cell_config:
            self.cell_config["tune_dir"] = str(self.tune_dir)

        return self

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
        from scp_pipeline.workspaces import assert_same_workspace, result_workspace_identity
        self.append_warnings = assert_same_workspace(
            {'workspace_identity': self.workspace.identity}, base_results, context='append_to')
        self.append_lineage = list((base_results.get('meta') or {}).get('append_lineage') or [])
        self.append_lineage.append({'run_path': str(append_target),
            'tune_dir': base_sim_cfg.get('tune_dir'), 'workspace_identity': result_workspace_identity(base_results)})

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

    def _validate_cell_tuning(self) -> None:
        tuning = self.cell_config.get("tuning")
        if tuning is None:
            return
        if not isinstance(tuning, dict):
            raise TypeError("cell_config tuning must be an object when provided.")
        if "soma_diam_multiplier" in tuning:
            tuning["soma_diam_multiplier"] = float(tuning["soma_diam_multiplier"])

    def _apply_run_overrides(self, sim_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Apply options that affect input generation and simulation behavior."""
        _merge_overrides(sim_cfg, self.options.sim_overrides)
        from scp_pipeline.workspaces import apply_workspace_metadata
        apply_workspace_metadata(sim_cfg, self.workspace)

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

        # An explicit runtime option is the highest-precedence output-name
        # control. Keep both normalized aliases aligned because result saving
        # also accepts legacy configs that still contain ``output_stem``.
        if self.options.output_stem:
            sim_cfg["output"] = self.options.output_stem
            sim_cfg["output_stem"] = self.options.output_stem

        if save_output:
            if not sim_cfg.get("output"):
                sim_cfg["output"] = self.options.output_stem or _timestamp_stem()
        elif self.options.output_stem and not sim_cfg.get("output"):
            sim_cfg["output"] = self.options.output_stem

        return sim_cfg

    def prepare(self, *, generate_inputs: bool = True) -> "SimulationSession":
        self._resolve_append_override()
        if self.options.load_mechanisms and self.cell_override is None:
            load_mechanisms(self.tune_dir, self.cell_config)

        self._apply_snapshot_option()
        self._validate_cell_tuning()

        sim_cfg_for_mode = self._apply_run_overrides(
            copy.deepcopy(self.sim_cfg_override or self.sim_cfg_preview)
        )
        self.iclamp_cfg = sim_cfg_for_mode.get("iclamp", {}) or {}
        self.iclamp_enabled = bool(self.iclamp_cfg.get("enabled", False)) or bool(self.options.iclamp)

        from .model_artifacts import construction_inputs
        self.prepared_inputs = construction_inputs(self.cell_config, self.tune_dir)
        self.cell = self.cell_override if self.cell_override is not None else load_cell(self.cell_config, base_dir=self.tune_dir)
        if construction_inputs(self.cell_config, self.tune_dir) != self.prepared_inputs:
            raise RuntimeError('Model inputs changed during construction. Restart and prepare again with stable sources.')
        apply_sim_conditions(self.cell, sim_cfg_for_mode)
        self.geom = geometry.define_geometry(self.cell, self.geom_config)

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

        from scp_pipeline.loaders import get_cell_loader_name
        from scp_pipeline.model.runtime_cell import describe_cell
        construction = get_cell_loader_name(self.cell_config)
        provenance = {'construction_kind': construction,
                      'cell_name': self.cell_config.get('cell_name'),
                      'source_mode': 'session_only' if construction == 'runtime_cell' else (self.cell_config.get('model_source') or {}).get('mode', 'tune_local'),
                      'construction_recipe_available': construction != 'runtime_cell',
                      'session_only': construction == 'runtime_cell', 'complete_checkpoint': False,
                      'inputs': copy.deepcopy(self.prepared_inputs)}
        state = describe_cell(self.cell) if construction == 'runtime_cell' else None
        def annotate(result):
            meta = result.setdefault('meta', {})
            meta['model_provenance'] = provenance
            meta['workspace_identity'] = self.workspace.identity
            if self.append_lineage:
                meta['append_lineage'] = copy.deepcopy(self.append_lineage)
            if self.append_warnings:
                meta.setdefault('append_warnings', []).extend(self.append_warnings)
            meta['cell_config'] = copy.deepcopy(self.cell_config)
            meta['geometry_config'] = copy.deepcopy(self.geom_config)
            if state is not None:
                meta['model_state'] = state
            return result

        if self.iclamp_enabled:
            self.result = run_iclamp_test(
                self.cell,
                self.sim_cfg,
                iclamp_cfg=self.iclamp_cfg,
            )
            return annotate(self.result)

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

        return annotate(self.result)

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
            "project_name": self.workspace.project_name,
            "workspace_identity": self.workspace.identity,
            "construction_kind": self.cell_config.get('cell_loader', 'allen_manifest'),
            "execution": 'in-process adopted cell' if self.cell_override is not None else 'configured recipe',
            "construction_recipe_available": self.cell_override is None,
            "iclamp_enabled": self.iclamp_enabled,
            "snapshot_enabled": self.snapshot_enabled,
            "mode": self.options.mode,
            "n_trials": None if self.sim_cfg is None else self.sim_cfg.get("n_trials"),
            "output": None if self.sim_cfg is None else self.sim_cfg.get("output"),
            "output_dir": str(self.output_dir),
            "saved_path": None if self.saved_path is None else str(self.saved_path),
        }
