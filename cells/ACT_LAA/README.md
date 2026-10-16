# ACT LAA — staged HOC example

Single-compartment native `Cell_A` from ACT commit
`2b3fdc190107b8242e3496329cfdf240abd09694`, using its ten intrinsic MOD
sources. Source bytes, original contributor credits, and mechanism equations
are preserved. `LICENSE.ACT` is supplied without replacing inherited headers
or asserting SCP's MIT license applies to all native assets. Each tune's
`SOURCE_PROVENANCE.json` records its source paths and hashes.

`orig` is ACT's starting `template.hoc`; `tuned` is an identical, initially
**untuned working copy**; `reference` is its known-target
`target_template.hoc`, not an exported optimized fit. Segmented and
known-passive-only experimental variants are not imported.

Explicit conditions are 31°C / −70 mV. Targets are `none`, synapses are not
configured, and current/timing settings remain generic SCP placeholders.
Select `cell_name = "ACT_LAA"`, a tune, and `project_name = None`; use
`--iclamp` for CLI cell-only runs. A new kernel/process is needed when
switching definitions named `Cell_A` or conflicting mechanism libraries.

This is staged and smoke-tested, not independently biologically validated
or a completed manual-tuning example. See the
[ACT catalog](../../docs/guides/act_examples.md) for later development.
