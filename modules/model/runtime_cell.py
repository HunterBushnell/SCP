"""Adopt existing compartmental NEURON objects without reconstructing them."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from scp_pipeline.loaders.base import CANONICAL_SECTION_GROUPS, LoadedCell, ensure_section_aliases, unique_sections


def _sections(value):
    from neuron import nrn
    if value is None:
        return ()
    try:
        values = (value,) if isinstance(value, nrn.Section) else tuple(value)
    except TypeError as exc:
        raise TypeError('Expected a NEURON Section or a collection of Sections.') from exc
    if any(not isinstance(section, nrn.Section) for section in values):
        raise TypeError('Canonical section collections must contain NEURON Sections, not segments or other values.')
    return values


def validate_cell(cell):
    """Validate the normalized interface without initializing or mutating it."""
    from neuron import h
    if not isinstance(cell, LoadedCell) or cell.h is not h:
        raise TypeError('Use wrap_cell() to provide the NEURON runtime and canonical section collections.')
    if cell.model is None:
        raise ValueError('A loaded cell must retain its model/section owner.')
    for group in CANONICAL_SECTION_GROUPS:
        _sections(getattr(cell, group))
    if not cell.soma or not cell.all:
        raise ValueError('A compatible cell requires nonempty canonical soma and all collections.')
    identities = {hash(section) for section in cell.all}
    if any(hash(section) not in identities for group in CANONICAL_SECTION_GROUPS for section in getattr(cell, group)):
        raise ValueError('Canonical all must contain every mapped section.')
    return cell


def wrap_cell(model, *, sections=None, section_map=None, cell_name=None):
    """Wrap a Section, owner object, or LoadedCell, retaining native properties."""
    from neuron import h, nrn
    if model is None or (isinstance(model, LoadedCell) and model.model is None):
        raise ValueError('Provide a Section or its owning object; None is not a cell.')
    if sections is not None and section_map is not None:
        raise ValueError('Provide sections or section_map, not both.')
    h.load_file('stdrun.hoc')
    cell = model if isinstance(model, LoadedCell) else LoadedCell(h=h, model=model,
        loader='runtime_cell', config={'cell_loader': 'runtime_cell', 'paths': {'modfiles': None}})
    if model is h and sections is None:
        if section_map is None:
            raise ValueError('For the global h namespace, explicitly map the sections belonging to this cell.')
        section_map = {group: section_map.get(group) for group in CANONICAL_SECTION_GROUPS}
    if isinstance(model, nrn.Section):
        if sections is not None or section_map is not None:
            raise ValueError('A single Section already defines soma/all; do not also provide a mapping.')
        sections = {'soma': model, 'all': model}
    if sections is not None:
        if not isinstance(sections, dict) or set(sections) - set(CANONICAL_SECTION_GROUPS):
            raise ValueError('sections must map canonical soma/dend/apic/axon/all names to Sections.')
        cell.sections = {key: _sections(sections.get(key)) for key in CANONICAL_SECTION_GROUPS}
    if section_map is not None:
        if not isinstance(section_map, dict):
            raise TypeError('section_map must be a dictionary of owner attribute names.')
        for group, names in section_map.items():
            if group not in CANONICAL_SECTION_GROUPS:
                raise ValueError(f'Unknown canonical group {group!r}.')
            names = [names] if isinstance(names, str) else ([] if names is None else names)
            if not isinstance(names, (list, tuple)) or any(not isinstance(name, str) or not name for name in names):
                raise TypeError(f'section_map.{group} must contain owner attribute names.')
            values = [_sections(getattr(cell.model, name)) for name in names]
            cell.sections[group] = unique_sections(*values)
        # Store normalized values so one-shot iterables are never consumed twice.
        section_map = None
    elif sections is None:
        for group, aliases in {'soma': ('soma', 'somatic'), 'dend': ('dend', 'basal'),
                               'apic': ('apic', 'apical'), 'axon': ('axon', 'axonal'),
                               'all': ('all', 'all_sections', 'sections')}.items():
            if group in cell.sections or getattr(cell, group, ()):
                continue
            candidates = [_sections(getattr(cell.model, alias)) for alias in aliases
                          if hasattr(cell.model, alias)]
            identities = {tuple(hash(section) for section in values) for values in candidates if values}
            if len(identities) > 1:
                raise ValueError(f'Ambiguous owner aliases for {group}; provide sections or section_map explicitly.')
            cell.sections[group] = next((values for values in candidates if values), ())
    ensure_section_aliases(cell, owner=cell.model, section_map=section_map,
                          allow_global_fallback=False, require_soma=True)
    if cell_name is not None:
        cell.config = dict(cell.config, cell_name=str(cell_name))
    return validate_cell(cell)


def adopt_cell(model, config=None):
    """Label a supplied object as runtime-only without modifying its recipe."""
    cell = wrap_cell(model)
    effective = deepcopy(config or cell.config or {})
    if cell.config.get('cell_name'):
        effective['cell_name'] = cell.config['cell_name']
    effective['cell_loader'] = 'runtime_cell'
    effective['paths'] = {'modfiles': None}
    effective.pop('model_source', None)
    effective.pop('hoc_template', None)
    effective.pop('python_factory', None)
    cell.config = effective
    cell.loader = 'runtime_cell'
    cell.source_artifacts = {}
    return cell


def prepare_runtime_workspace(tune_dir, cell, sim_config=None, geometry_config=None, *, include_target_config=False, dry_run=False):
    """Create only missing bookkeeping configs; never replace a stored recipe."""
    from scp_pipeline.setup.defaults import default_sim_config, default_geometry_config
    from scp_pipeline.simulation.current_injection import validate_required_sim_conditions
    from scp_pipeline.simulation.session import _merge_overrides
    from scp_pipeline.workspaces import infer_workspace
    workspace = infer_workspace(tune_dir)
    tune = workspace.tune_dir
    configs = tune / 'cell_configs'
    from scp_pipeline.loaders.source_context import within
    if not within(configs.resolve(), tune):
        raise ValueError('Runtime config destination escapes the selected workspace.')
    label = cell.config.get('cell_name') or (tune.parent.parent.name if tune.parent.name == 'tunes' else tune.name)
    runtime_config = {'cell_name': label, 'tune': tune.name, 'color': 'k',
                      'cell_loader': 'runtime_cell', 'paths': {}}
    workspace = infer_workspace(tune, cell_name=label)
    if workspace.project_name is not None:
        runtime_config['project_name'] = workspace.project_name
    stored_sim = json.loads((configs / 'sim_config.json').read_text()) if (configs / 'sim_config.json').is_file() else default_sim_config(cell_name=label, specimen_id=None, model_type='')
    effective_sim = _merge_overrides(deepcopy(stored_sim), sim_config)
    validate_required_sim_conditions(runtime_config, effective_sim)
    geometry = geometry_config or default_geometry_config(cell_name=label)
    payloads = {'cell_config.json': runtime_config, 'sim_config.json': effective_sim, 'geometry.json': geometry}
    if include_target_config:
        payloads['target_config.json'] = {'schema_version': 1, 'target_source': {'mode': 'none'}}
    for filename, value in payloads.items():
        path = configs / filename
        if not within(path.resolve(), tune):
            raise ValueError(f'Runtime config destination escapes the workspace: {path}')
        for parent in path.parents:
            if parent.exists() and not parent.is_dir():
                raise ValueError(f'Runtime config parent is not a directory: {parent}')
            if parent == tune:
                break
        if path.exists() and not isinstance(json.loads(path.read_text()), dict):
            raise ValueError(f'Expected a configuration object: {path}')
        json.dumps(value, allow_nan=False)
    if dry_run:
        return effective_sim
    configs.mkdir(parents=True, exist_ok=True)
    for filename, value in payloads.items():
        path = configs / filename
        if not path.exists():
            path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return effective_sim


def describe_cell(cell):
    """Descriptive pre-run inventory, not a resumable NEURON checkpoint."""
    result = []
    for section in cell.all:
        info = section.psection()
        result.append({'name': section.name(), 'L_um': float(section.L), 'diam_um': float(section.diam),
                       'nseg': int(section.nseg), 'Ra_ohm_cm': float(section.Ra),
                       'cm_uF_cm2': info['cm'], 'density_mechanisms': info['density_mechs']})
    return {'description_only': True, 'complete_checkpoint': False, 'sections': result}
