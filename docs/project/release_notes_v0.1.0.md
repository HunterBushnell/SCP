# SCP v0.1.0 Release Notes

SCP v0.1.0 is the initial public preview release of the Single Cell Pipeline. This
release focuses on a notebook-first workflow for preparing, tuning, simulating,
and analyzing NEURON/BMTK-compatible single-cell models.

## Highlights

- Notebook workflow for Steps 1–7:
  - setup and cell/config preparation,
  - passive property review/tuning,
  - active property review/tuning,
  - synapse review/tuning,
  - simulation execution,
  - analysis and plotting,
  - utility tools.
- Local and Google Colab-compatible notebook bootstraps.
- Colab Python 3.12 bootstrap compatibility for optional IPython autoreload,
  dependency installation, and AllenSDK model loading.
- Allen Cell Types/Allen Database model setup support.
- ACT integration points for passive and active target workflows.
- BMTool-based synapse tuning support with SCP config adapters.
- Config-driven simulation setup with notebook overrides for common options.
- Manual, trace-file, and Allen NWB target configuration support.
- Local CLI and SLURM entry points for larger simulation batches.
- Curated PV and SST baseline example outputs for analysis demonstrations.
- PN, PV, and SST example cell/tune scaffolds.
- Public notebook defaults standardized around the PV tuned example.
- Allen/ADB NWB target configs use tune-local NWB filenames.

## Example Data

This release includes concise public example targets and baseline simulation
outputs for demonstrating the pipeline. Downloaded NWB files, compiled NEURON
artifacts, scratch logs, and non-curated simulation outputs are intentionally
ignored by Git.

## Known Scope

- The core notebook workflow and PV/SST example simulation paths have been
  validated. The main notebook workflow has also been validated through Step 6
  in Google Colab.
- ACT active tuning is exposed as an integration path, but model-specific tuning
  performance depends on ACT and the selected biological targets. Step 3.4 ACT
  active auto-tuning remains pending deeper ACT workflow validation.
- Step 7 utility tools load in Colab, but individual utilities are tested as
  needed.
- Additional ADB-target tuned examples may be added after further tuning and
  validation.
