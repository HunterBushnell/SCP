# ACT SST — staged Allen-manifest example

Allen model `476686112`, supplied through ACT at commit
`2b3fdc190107b8242e3496329cfdf240abd09694`. This is distinct from SCP's
legacy SST model. Each tune preserves its native manifest, selected fit,
morphology/marker, and matching 16-MOD library.

- `orig`: pristine Allen bundle from ACT's `data/SST/adb`.
- `tuned`: ACT's `data/SST/seg_tuned` with its active manifest-selected fit.

The active tuned fit equals ACT's `AFTER_ACT` checkpoint, not its differently
valued `FINAL` checkpoint. No alternate checkpoint is silently substituted.
Segmented mechanisms and tuned fit remain paired. Native morphology can emit
the inherited multi-root SWC warning; it is not edited to suppress the warning.

SCP uses standard Allen utilities with soma multiplier 1.0. Conditions are
34°C / −79.65586853027344 mV. Targets are `none`, synapses are not configured,
and timing/current fields remain generic placeholders rather than accepted
scientific protocols. Select `cell_name = "ACT_SST"`, a tune, and
`project_name = None`; use `--iclamp` for CLI cell-only runs. Restart when
switching conflicting native definitions.

This family is staged and smoke-tested, not a completed SCP walkthrough or
independent biological/tuning validation. ACT's package is not required;
NEURON/AllenSDK are. See `LICENSE.ACT`, native credits, tune-local
`SOURCE_PROVENANCE.json`, and the [ACT catalog](../../docs/guides/act_examples.md).
