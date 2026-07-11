# Public Release Cleanup Notes

This note tracks public-release cleanup decisions that may need final review before
tagging a release. It is not part of the user workflow.

## Current Decisions

| Area | Current public-repo state | Follow-up |
| --- | --- | --- |
| Historical scratch archive | Historical notebooks/scripts are not part of the main public workflow. Useful segmentation content is available as `extra_notebooks/act_segmentation.ipynb`. | Confirm no additional archived utilities should be restored. |
| Saved simulation outputs | Existing saved outputs are not required for the code to run and are excluded from Git by default. | Decide whether to ship approved PV/SST outputs or generate smaller public examples. |
| Large/intermediate external data | Keep only concise public CSV examples in `external_data/` unless additional data are approved. | Advisor/lab review before adding paper-specific or collaborator-derived data. |
| Compiled mechanisms | Generated `x86_64/` mechanism builds are ignored. | Users compile locally with Step 1, `check_setup.py --compile-modfiles`, or `nrnivmodl`. |
| Logs/status files | Local run logs/status artifacts are ignored except placeholders needed to keep folders. | No public replacement needed. |

## Kept Public External CSVs

- `external_data/pyrFiringRateAvg.csv`
- `external_data/PVFiringRateAvg.csv`
- `external_data/SSTFiringRateAvg.csv`

## Release Reminder

Before making the repo public, verify whether the final release should include:

- no saved outputs,
- approved PV/SST saved outputs from the existing examples, or
- newly generated small demonstration outputs.
