# Contracts Index

Contracts are developer/design references. Public usage is documented in
`docs/`, especially:

- `../docs/pipeline_overview.md`
- `../docs/guides/step_1_setup.md`
- `../docs/guides/step_5_simulate.md`
- `../docs/reference/configs_reference.md`
- `../docs/reference/outputs_layout.md`

## Current Design References

- `pvsst_results_outputs.contract.v1.md`: run-output artifact schema.
- `pvsst_step2.3_inputs_modes.contract.v1.md`: input mode handler expectations.
- `mode_contract.md`: compact input mode contract summary.
- `5_simulate_randomness.md`: randomness design notes.

## Historical Drafts

These files record design history and may describe earlier internal layouts:

- `archive/step2.3_input_generation.contract.v1.md`
- `archive/pvsst_step2.3_inputs.contract.v2.md`
- `archive/pvsst_step2.3_inputs.contract.v3.md`
- `archive/pvsst_step2.3_inputs.contract.v4.md`

## Runtime Code

For implementation details, cross-check:

- `modules/input_generation/`
- `modules/simulation/`
- `modules/setup/`
