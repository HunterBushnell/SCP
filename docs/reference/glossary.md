# Glossary

## Cell

The NEURON model for a specific cell type or model target, such as PV or SST.

## Tune

A specific set of model files, mechanisms, configs, and outputs for a cell.

## Tune Directory

The folder for one tune:

```text
cells/<CELL>/tunes/<TUNE>/
```

## Trial

One simulation run within a single- or multi-trial batch.

## Synapse Group

A named group of synapses with shared placement, mechanism, and input settings,
such as `pn_exc` or `bg_inh`.

## Input Block

An explicit active input window for a synapse group. Blocks define start/stop
times, mode, and source/rate settings.

## Input Mode

The method used to generate spike trains for a synapse group block, such as
`homogeneous_poisson`, `inhomogeneous_poisson`, or `precomputed`.

## Snapshot

A richer capture of inputs, traces, and metadata for comparison/debugging.

## Run Manifest

The metadata/index file saved in each output run folder.
