# Project Workspaces

Use this optional namespace for SCP work on separate projects:

```text
projects/<project>/cells/<cell>/tunes/<tune>/
```

Public examples live under `cells/`. No private project models are bundled
here; a fresh public clone includes this guide, not locally excluded research
workspaces. Existing cells can be copied or migrated deliberately using the
same tune layout; there is no automatic migration.

In the local research checkout, bV1 holds independent PV/SST copies, LUT holds
EUSmn/HYPO/PGN, A1 uses ET/IT/PV/SST, and amygdala contains isolated native
cell copies. These roots are locally excluded. The legacy public PV/SST/PN
examples remain temporarily available alongside staged ACT examples; this is
an example-development transition, not a Git-history removal operation.

Native model sources and tune configs are Git-eligible, just as under `cells/`.
Generated results, exports, ACT workspaces, backups, and compiled mechanisms are
ignored. If an entire project is private, add its exact directory to your local
Git exclusion file; do not rely on generated-output rules to hide its sources.

See [Project workspaces](../docs/guides/project_workspaces.md) for selection,
local exclusions, external model sources, and result identity.
