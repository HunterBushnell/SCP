# ACT Spiker — staged HOC example

Single-compartment `Spiker` from ACT commit
`2b3fdc190107b8242e3496329cfdf240abd09694`, with one native `hh_spiker.mod`.
Native files and credits are unchanged. The source credits Tyler B and a
NEURON tutorial; see `LICENSE.ACT`, source headers, and each tune's
`SOURCE_PROVENANCE.json`.

| Tune | Meaning |
|---|---|
| `orig` | ACT starting `template.hoc` |
| `tuned` | Identical, initially **untuned** editable working copy |
| `reference` | Known-reference `target_template.hoc`, not an optimized export |

Initial conditions are 6.3°C / −50 mV. Target mode is `none`, synapses are not
configured, and IClamp timing/current values are generic SCP placeholders—not
a validated scientific protocol. Core use requires NEURON, not ACT or BMTool.

Choose `cell_name = "ACT_Spiker"`, `tune_name = "orig"`, and
`project_name = None`; stored configs own loader paths and conditions.
From the SCP root: `python run_pipeline.py --cell ACT_Spiker --tune orig --iclamp`.
Restart the kernel/process when switching copied HOC/MOD definitions.

This family is staged and smoke-tested, not a completed SCP tuning walkthrough
or independent biological validation. No scientific targets or datasets have
been generated. See the [ACT catalog](../../docs/guides/act_examples.md) for
setup, process boundaries, provenance, and later manual tuning.
