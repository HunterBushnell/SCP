# ACT BursterCa — staged HOC example

Single-compartment `BursterCa` from ACT commit
`2b3fdc190107b8242e3496329cfdf240abd09694`. The eight intrinsic MOD sources
are preserved with their original names, bytes, and credits; unused synaptic
and GUI/driver files are not imported. See `LICENSE.ACT`, native headers,
and each tune's `SOURCE_PROVENANCE.json`.

`orig` preserves ACT's starting `template.hoc`; `tuned` is initially an
identical **untuned working copy**. `reference` preserves the known-target
`target_template.hoc`, not a recovered ACT optimization.

Initial conditions are 6.3°C / −50 mV. Target mode is `none`, synapses are not
configured, and generic SCP IClamp fields are placeholders, not an accepted
characterization protocol. Choose `cell_name = "ACT_BursterCa"`, a tune,
and `project_name = None`; use `--iclamp` for CLI cell-only runs.
Restart processes when changing copied native definitions.

This is staged and smoke-tested, not a completed scientific walkthrough.
No firing phenotype, biological-property match, or tuning quality is an
acceptance assertion. See the [ACT catalog](../../docs/guides/act_examples.md).
