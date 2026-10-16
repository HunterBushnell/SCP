# Third-Party and Model-Asset Notices

The root [`LICENSE`](LICENSE) applies to SCP-authored source code and
documentation. It does not relicense the model, morphology, mechanism, or other
assets identified below. An asset's own file header or applicable source terms
take precedence over SCP's root license.

## Allen Cell Types Database model assets

The Allen-manifest bundles under `cells/PN/`, `cells/PV/`, and `cells/SST/`
contain model assets obtained from the Allen Cell Types Database through
AllenSDK. These include manifests, model parameter files, morphologies, marker
files, and Allen model mechanism sources. Tune-local SCP configuration and
documentation files are not included in this notice merely because they share
the same cell directory.

The source metadata recorded in the bundles is:

| SCP cell | Allen specimen ID | Allen model ID | Recorded model type and name |
|---|---:|---:|---|
| PN | `382982932` | `497233292` | `all active`; `Biophysical - all active_Nr5a1-Cre;Ai14-177333.03.01.01` |
| PV | `484635029` | `485602029` | `perisomatic`; `Biophysical - perisomatic_Pvalb-IRES-Cre;Ai14-201791.05.01.01` |
| SST | `485466109` | `496538951` | `all active`; `Biophysical - all active_Sst-IRES-Cre;Ai14-202729.03.02.01` |

Each tune records this information in `.adb_download_meta.json`. The specimen
records can be reached at:

- `https://celltypes.brain-map.org/experiment/electrophysiology/382982932`
- `https://celltypes.brain-map.org/experiment/electrophysiology/484635029`
- `https://celltypes.brain-map.org/experiment/electrophysiology/485466109`

SCP makes no additional license assertion for these source assets. Consult the
source records and any terms applicable to their use.

## BBP/EPFL synapse mechanisms

The following mechanism files retain headers identifying copyright
`BBP/EPFL 2005-2021` and the Creative Commons
[Attribution-NonCommercial-ShareAlike 4.0 International
license](https://creativecommons.org/licenses/by-nc-sa/4.0/) (CC BY-NC-SA
4.0):

- `cells/PV/tunes/tuned/modfiles/AMPA_NMDA_STP.mod`
- `cells/PV/tunes/tuned/modfiles/GABA_A.mod`
- `cells/PV/tunes/tuned_adb/modfiles/AMPA_NMDA_STP.mod`
- `cells/PV/tunes/tuned_adb/modfiles/GABA_A.mod`
- `cells/SST/tunes/tuned/modfiles/AMPA_NMDA_STP.mod`
- `cells/SST/tunes/tuned/modfiles/GABA_A.mod`
- `cells/SST/tunes/tuned/modfiles/GABA_A_STP.mod`
- `cells/SST/tunes/tuned_adb/modfiles/AMPA_NMDA_STP.mod`
- `cells/SST/tunes/tuned_adb/modfiles/GABA_A.mod`
- `cells/SST/tunes/tuned_adb/modfiles/GABA_A_STP.mod`

The same terms apply to any archived copies produced under
`output_data/**/model_artifacts/`. These mechanisms are not covered by SCP's
MIT license.

## ACT-sourced native examples

The public `cells/ACT_Spiker/`, `cells/ACT_BursterCa/`, `cells/ACT_LAA/`,
`cells/ACT_L5/`, and `cells/ACT_SST/` families copy native sources from
[ACT](https://github.com/V-Marco/ACT), pinned to commit
`2b3fdc190107b8242e3496329cfdf240abd09694`. Each tune records source-relative
paths, source/current SHA-256 hashes, and its variant relationship in
`SOURCE_PROVENANCE.json`. ACT itself is not a runtime dependency for these
native examples.

ACT's MIT license notice, copyright Vladimir Omelyusik (2025), is retained as
`LICENSE.ACT` in each family. Native file headers and upstream credits remain
unchanged; this notice does not assert that every inherited model, mechanism,
or Allen dataset has been relicensed by ACT or SCP. In particular, Spiker
credits Tyler B and a NEURON tutorial; its Hodgkin–Huxley mechanism retains
its original attributions. BursterCa/LAA mechanisms retain their source-model
and contributor credits, including Migliore where identified in the headers.
The Allen L5/SST bundles retain their native fit, morphology, and mechanism
sources and are subject to the source records/terms applicable to those assets.
SCP's root MIT license does not relicense these native assets.

The ACT SST model is `476686112`; ACT L5 is `497232429`. They are distinct
from SCP's legacy Allen models listed above. Staging and numerical smoke
checks are not independent biological-validation claims. See the
[ACT example catalog](docs/guides/act_examples.md).

## LUT-derived HOC and MOD sources

Earlier SCP versions included HOC and MOD model sources under the following
paths, derived from the [LUT repository](https://github.com/HunterBushnell/LUT).
Those sources were redistributed with permission at the time of inclusion:

- `cells/EUSmn/tunes/*/model/`
- `cells/EUSmn/tunes/*/modfiles/`
- `cells/HYPO/tunes/*/model/`
- `cells/HYPO/tunes/*/modfiles/`
- `cells/PGN/tunes/*/model/`
- `cells/PGN/tunes/*/modfiles/`

Those live examples are now maintained in a locally excluded LUT project,
not bundled in the current public tree. This relocation does not erase
earlier commits/releases or change their original notices. The historical
source lineage is pinned to LUT commit
`f9739d8aa7f94eac67fcaa67e8e04e26787dee0f`. EUSmn begins from the LUT PUD
template; the tuned EUSmn, HYPO, and PGN HOC templates are manually derived
tunes. Exact source and current-artifact hashes are recorded in each tune's
`SOURCE_PROVENANCE.json`.

Permission to redistribute these files does not relicense them under SCP's MIT
license, and SCP asserts no additional upstream license terms for them.

## `vecstim.mod`

Copies of `vecstim.mod` are present in the PV and SST tuned/tuned-ADB mechanism
bundles. Its upstream provenance and license have not been resolved. SCP's root
MIT license makes no license assertion for `vecstim.mod`; users should verify
that their intended use and redistribution are permitted.

## Archived copies

When a listed model or mechanism source is copied into an
`output_data/**/model_artifacts/` snapshot, that copy retains the same notice
and license status as its source file.
