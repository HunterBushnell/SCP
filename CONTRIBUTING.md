# Contributing to SCP

SCP is intended to stay notebook-friendly, modular, and reusable for single-cell
modeling projects. Contributions should keep that user workflow clear.

## Before Changing Code

- Open or reference an issue when the change is not obvious.
- Prefer small, focused changes over broad rewrites.
- Keep notebook-facing code concise; move reusable machinery into `modules/`.
- Avoid committing private, collaborator-specific, or unpublished raw data unless
  it has been explicitly approved for release.

## Data and Generated Files

- Do not commit downloaded NWB files, compiled NEURON artifacts, logs, or scratch
  outputs.
- Only commit `output_data/` folders when they are curated public examples.
- Keep large datasets outside the repo and document where users should obtain
  them.

## Validation

Before submitting a change, run the checks that apply to your edit:

```bash
python scripts/check_notebooks.py
python scripts/check_setup.py --steps 1 2 3 4 5 --cell PV --tune tuned --compile-modfiles
python scripts/check_setup.py --steps 1 2 3 4 5 --cell SST --tune tuned --compile-modfiles
python -m compileall -q modules scripts run_pipeline.py
```

For simulation or analysis changes, also run at least one small Step 5 simulation
and confirm Step 6 can load the result.

## Style Guidelines

- Keep public notebooks readable for users who are not terminal-heavy Python
  developers.
- Put reusable functions in `modules/`, not repeated notebook cells.
- Keep configs explicit, documented, and compatible with the notebook workflow.
- Update relevant docs when user-facing behavior changes.
