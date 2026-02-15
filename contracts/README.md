# Contracts Index

These contracts describe Step 5.2.3/5.2.4 behavior and run-output schemas.
They are descriptive references; public inputs/outputs remain authoritative.

## Step Mapping
- Legacy 2.3 -> SCP 5.2.3 (Generate inputs).
- Legacy 2.4 -> SCP 5.2.4 (Build/attach synapses).

## Current References
- `pvsst_results_outputs.contract.v1.md`: output/run artifact schema.
- `pvsst_step2.3_inputs_modes.contract.v1.md`: mode handler interface expectations.
- `mode_contract.md`: compact mode contract summary.
- `5_local_randomness.md`: randomness design notes.

## Historical Drafts (Context)
- `step2.3_input_generation.contract.v1.md`
- `pvsst_step2.3_inputs.contract.v2.md`
- `pvsst_step2.3_inputs.contract.v3.md`
- `pvsst_step2.3_inputs.contract.v4.md`

## Notes
- Some historical 2.3 drafts reflect pre-refactor config layouts.
- For current runtime behavior, cross-check:
  - `modules_local/inputs.py`
  - `docs/configs_reference.md`
  - `docs/step_5_local_pipeline.md`
