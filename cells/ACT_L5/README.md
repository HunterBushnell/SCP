# ACT L5 — staged Allen-manifest example

Allen model `497232429`, supplied through ACT at commit
`2b3fdc190107b8242e3496329cfdf240abd09694`. This Rorb all-active model is
distinct from SCP's legacy PN example. Each tune contains its native manifest,
manifest-selected fit, morphology/marker, and matching 16-MOD library.

- `orig`: pristine Allen bundle from ACT's `data/L5/adb`.
- `tuned`: imported ACT variant from `data/L5/seg_tuned`.

ACT's altered `orig` starting variant and optimizer checkpoints are not
imported. The tuned mechanisms and fit are a matched set; do not swap a fit
into another variant's library. An orig-to-tuned overwrite would discard the
imported tuned variant rather than simply initialize a new working directory.

SCP uses the native Allen-manifest adapter with all-active utilities and stub
axon. Explicit conditions are 34°C / −90 mV, with soma multiplier 1.0 and
neutral plot color. Targets are `none`, synapses are not configured, and
generic IClamp controls are placeholders—not source-paper protocols.

Select `cell_name = "ACT_L5"`, a tune, and `project_name = None`, or run
`python run_pipeline.py --cell ACT_L5 --tune orig --iclamp` from the SCP root.
NEURON/AllenSDK are required; ACT and BMTool are not. Use fresh processes
when switching native variants.

The family is staged and smoke-tested, not a completed SCP characterization
dataset or independent assessment of ACT's tuning. Native headers,
`LICENSE.ACT`, and tune-local `SOURCE_PROVENANCE.json` retain source lineage and credits.
See the [ACT catalog](../../docs/guides/act_examples.md).
