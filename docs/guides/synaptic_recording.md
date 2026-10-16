# Measured synaptic recordings

Work in progress: implementation is paused pending review, not release-approved.
Capture remains disabled by default. See the
[resume plan](../project/synaptic_recording_update.md) for status and known work.

Step 5 can record native point-process currents, conductances, and exposed
short-term-plasticity state. This is distinct from `syn_records`, which records
placement, weights, and delivered input times rather than measured traces.
Neither ACT nor BMTool is needed for recording.

## Enable and bound capture

Edit `cell_configs/sim_config.json`, for example:

```json
"syn_recording": {
  "enabled": true,
  "n_trials": 1,
  "default_mode": "sample",
  "default_sample_per_group": 5,
  "sample_seed": 0,
  "record_dt_ms": 0.5,
  "max_memory_mb": 256,
  "default_vars": {
    "i": true, "g": true,
    "i_AMPA": false, "i_NMDA": false,
    "g_AMPA": false, "g_NMDA": false,
    "record_use": false, "record_Pr": false
  },
  "groups": {}
}
```

This is a recording-settings excerpt, not a complete simulation config or a
scientific stimulation protocol. Capture is disabled by default. `n_trials`
limits recorded trials independently of voltage storage; `null` follows
`n_traces_to_save`, and `0` allocates no synaptic recorders. An explicit positive
cap permits synaptic capture even when voltage traces are not retained.
IClamp-only runs do not attach synapses or produce these measurements.

`record_dt_ms: null` records at simulation `dt`. A chosen recording interval
must be at least `dt`; it changes recording resolution, not the integration
timestep. Measured traces carry their own time axis. Sampling uses a separate,
deterministic group/name seed and does not consume placement, weight, or input
randomness. It is not a claim that a small sample represents the population.

## Modes and aggregation

- `group`: record every synapse needed for exact group aggregates.
- `sample`: retain individual traces for at most `default_sample_per_group`.
- `both`: exact group aggregates plus that deterministic individual sample.

Currents (`i`, `i_AMPA`, `i_NMDA`) are summed in nA; conductances (`g`,
`g_AMPA`, `g_NMDA`) are summed in µS. The dimensionless `record_use` and
`record_Pr` states are averaged, not summed. These names use standard native
synaptic units; custom mechanisms must expose compatible variables. Native
values already include their mechanism's weight semantics and are not
multiplied by weights a second time.

Per-group overrides inherit global defaults:

```json
"groups": {
  "pn_exc": {"mode": "both", "sample_per_group": 3,
             "vars": {"record_use": true, "record_Pr": true}},
  "bg_inh": {"enabled": false}
}
```

Only supported native variables are recorded. Missing variables are reported
with coverage counts/IDs, not replaced by zero. An aggregate is omitted unless
that variable is available for **every** synapse in the group. Missing sampled
rows are NaN and retain their selected synapse identities. Exposed but invalid
pointers or failed recorders raise contextual errors rather than being skipped.
Steps that use arbitrary caller-owned synapses without group metadata label
their positional identities as `unmapped` instead of inventing biological groups.

## Memory limits

Exact aggregates require native recorders for all contributing synapses, even
though only a sum is saved. A large population at full simulation resolution
can need substantial memory. Before vector allocation, SCP estimates native
buffer capacity, output arrays, conversion scratch, and previously retained
synaptic trials. If the budget is exceeded, reduce variables/groups/trials,
use sample mode, increase `record_dt_ms`, or deliberately raise the limit.

This is a conservative recorder-array estimate, not a guarantee about total
process memory. Snapshot mode does not silently enable recording or lift the
memory limit. No thread-based parallel NEURON recording is introduced.

## Save, inspect, and plot

Single trials use `syn_recordings`; multi-trial runs use
`syn_recordings_by_trial` entries with explicit `trial_idx`. Payloads contain
their own `T`, units, group mode/counts, selected IDs/metadata, coverage,
aggregation method, and aggregate/sample arrays. Placement `syn_records*`
remains separate. Manifest sidecars, full pickle/NPZ results, appending, and
array/two-run merges preserve the measured recordings and their trial indices.
Append retention uses the destination recording cap; incompatible measured
time axes are rejected rather than silently combined.

In Step 6, the recording tables report group coverage and sample counts. For
explicit plotting from a saved run:

```python
from modules.simulation.result_loading import load_results
from modules.analysis import analysis

results = load_results("cells/ExampleCell/tunes/tuned/output_data/example_run")
summary = analysis.summarize_synaptic_recordings(results, trial_idx=0)
print(analysis.format_synaptic_recording_table(summary))
fig, axes = analysis.plot_synaptic_recordings(
    results, trial_idx=0, variables=("i", "g"), include_samples=True,
)
```

Trial selection uses the recorded trial ID, not an index into a truncated
sample list. Analysis works without reconstructing the cell or retaining Vm.
Sums across recorded groups are labeled with that scope, not claimed to cover
unrecorded groups or the whole cell. Legacy `traces.I/G` lacks a verified
aggregation contract and is labeled as such; historical records are not
retroactively reinterpreted as totals or rewritten.
