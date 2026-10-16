# Cell-scoped data and public baselines

Public PV/SST simulations currently use the saved `PV5_base` / `SST5_base`
**synaptic** settings. Their intrinsic models remain the existing tuned cells;
this is not a switch to the original Allen models. The selected circuit inputs,
weights, background rates, and SST VIP timing follow those baseline records.
The default remains one trial, not the archived 2,000-trial experiment.

PV's cross-cell PV/SST inhibitory groups and SST's VIP group are explicitly
disabled in this baseline. Their absence is intentional, not a missing-data
fallback. Public historical runs/comparison entries remain available and are
not rewritten to pretend they were produced by the current config.

## Data belongs with its consumer

```text
cells/PV/data/                         # public PV input/reference CSVs
cells/SST/data/                        # public SST input/reference CSVs
cells/PN/data/                         # public PN input template dependency
projects/bV1/external_data/            # independent private research copies
```

PV and SST each carry a byte-identical copy of the shared pyramidal-rate input,
so their active configs do not depend on another example's data folder. The
PV/SST reference CSVs live with their corresponding cells. A project may keep
shared dependencies at its project root rather than duplicating them in every
cell. Project copies are independent; editing public data does not edit bV1.

Paths beginning with `cells/` or `projects/` are SCP-root-relative. Other input
paths keep their existing tune-relative semantics. A new generic scaffold uses
the disabled placeholder `data/input_rates.csv`; it does not silently choose
the PV or PN dataset. Explicitly stage/select suitable data for a new cell.

## Historical references and missing inputs

The old root `external_data/` CSVs have moved. Their three explicit aliases in
`cells/DATA_PATH_ALIASES.json` resolve to the same source bytes using SHA-256
verification. Existing files retain precedence; unknown paths are not guessed.
If a relocation target has changed, replay fails instead of substituting a new
dataset. Historical saved configs and artifacts retain their original paths.

For active input groups, a missing required precomputed source now raises an
actionable error; it no longer becomes empty spike trains. Supply the intended
dataset, correct its path, or deliberately disable the group. Explicit inline
empty trains and disabled/zero-synapse groups remain valid. The setup checker
checks enabled input dependencies for Step 5; IClamp does not require them.

The private bV1 research setup is unchanged, including its unresolved
`SST5.3_tuned` input. The public baseline does not require that input. This
update does not recover datasets, tune cells, or reproduce a paper experiment.
