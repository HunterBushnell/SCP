# Step 6 UI Package

This package organizes the notebook-facing UI helpers used by `6_analysis.ipynb`.

## Public Sections

- `selection.py`: output tree/run selection and run-resolution helpers.
- `outputs.py`: output firing-rate/ISI plotting UI and execution helpers.
- `inputs.py`: input plotting UI and execution helpers.
- `metrics.py`: output metrics tables and metric-distribution helpers.
- `extra.py`: optional Extra Analysis tools.

## Stable Facade

`modules.analysis.analysis_ui` is the stable facade used by notebooks:

```python
from modules.analysis import analysis_ui
analysis_ui.build_outputs_ui(globals())
```

New backend code can import a narrower section when that is clearer:

```python
from modules.analysis.ui import outputs
outputs.build_outputs_ui(globals())
```

## Implementation Boundary

`_engine.py` contains the implementation shared by the section modules. It is
kept private so the public surface can stay organized while the notebook API
remains stable.
