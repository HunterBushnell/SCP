# Installation

## Local Conda Setup

```bash
git clone <SCP_REPO_URL>
cd SCP
conda env create -f environment.yml
conda activate scp-py311
# Suggested full notebook setup, including optional ACT and BMTool:
python -m pip install -e '.[act,bmtool]' -c requirements.txt
python -m ipykernel install --user --name scp-py311 --display-name "scp-py311"
```

## Optional venv Setup

Conda is preferred because NEURON/AllenSDK dependencies are easier to reproduce,
but a venv can work on compatible systems:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pip install -e '.[act,bmtool]' -c requirements.txt
python -m ipykernel install --user --name scp-venv --display-name "Python (SCP venv)"
```

## BMTool: suggested install-once dependency

BMTool is optional for core SCP use, but recommended for the full Step 4
notebook workflow. The setup commands above install its pinned package
(`bmtool==0.8.6`) into the environment using SCP's `bmtool` extra. For cell-only
use, omit the extras and install SCP with `python -m pip install -e .`.

To add BMTool to an already configured environment, run from the SCP checkout:

```bash
conda activate scp-py311
python -m pip install -r requirements-bmtool.txt
```

The optional requirements file constrains resolution to SCP's existing
scientific pins, including NEURON 8.2.4. It installs BMTool's declared Python
dependencies without upgrading those pins; a conflict fails rather than
silently replacing them. This is a suggested extra step after
`conda env create`, not a mandatory dependency in the base environment.

Installation is once **per Python environment**, not per project. After a
kernel restart, notebooks, CLI checks, and copied compact notebooks use the
same `bmtool` import. There is no required checkout location or per-project
path setting. BMTool itself documents [package installation on PyPI](https://pypi.org/project/bmtool/).

SCP prefers the environment package over automatically discovered checkouts.
It imports BMTool only when the integration is requested, and validates
`SynapseTuner(hoc_cell=..., ...)` and `SynapseOptimizer(tuner)` before use. A
broken/incompatible installed package produces an actionable error; SCP does
not silently download a different copy to replace it. Normal local/project
notebooks do not install or upgrade dependencies automatically.

For BMTool development, editable-install a checkout at **any** location in the
same environment instead. Retain that checkout and use the same constraints:

```bash
python -m pip install -e /path/to/bmtool -c /path/to/SCP/requirements.txt
```

## ACT: suggested install-once source dependency

ACT is optional for target-derived passive proposals and active optimization;
ordinary passive/active/FI measurements do not need it. The suggested full
setup above includes the `act` extra. To add only ACT to an existing SCP
environment, run from the SCP checkout:

```bash
conda activate scp-py311
python -m pip install -r requirements-act.txt
```

This installs Automatic Cell Tuner from the pinned
[V-Marco/ACT revision](https://github.com/V-Marco/ACT/tree/2b3fdc190107b8242e3496329cfdf240abd09694)
(`2b3fdc190107b8242e3496329cfdf240abd09694`) with SCP's scientific dependency
constraints. Git and network access are needed for this one-time source
installation. Do **not** substitute `pip install act`: the import/distribution
name alone does not identify this source project or revision.

Only ACT's Python package is installed, not its example datasets or experiment
directories. A regular installation does not require retaining a source
checkout; an editable developer installation does. Both are shared across all
projects using that Python environment. Restart kernels/workers after installing
or changing the dependency.

SCP checks the selected package's passive or active APIs only when requested,
without constructing cells or running optimization. Passive helpers do not
import the active optimizer. Broken imports, unrelated `act` packages, and API
mismatches are reported directly; they do not cause a replacement download.
This installation change does not expand ACT's model support: configured
non-Allen optimization remains experimental, and runtime-only notebook cells
still require a persisted builder before fresh-process optimization.

Known optimization limitation at this pin: the optional real optimizer smoke
test reaches ACT's custom-builder copying code and fails with
`TypeError: 'tuple' object is not callable`. `ACTModule._simulate_cells` passes
the stored `(builder, kwargs)` tuple back to `set_custom_cell_builder` without
unpacking it. This reproduces in the pristine checkout as well as the installed
package; installation, passive helpers, API checks, and fresh-process builder
preparation work independently. Resolving the upstream optimizer issue is a
separate [roadmap item](project/roadmap.md), not a package-install workaround or
a claim that full optimization passed.

For development, editable-install a deliberate checkout at any location:

```bash
python -m pip install -e /path/to/ACT -c /path/to/SCP/requirements.txt
```

Use the pinned revision for reproducibility, or record the revision you are
deliberately testing. Installation does not move generated ACT builders,
targets, optimization settings, proposals, or outputs out of their selected
project/tune workspace. Normal notebook actions never run pip.

## External Repositories

ACT and BMTool normally use environment-installed packages. Old source
checkouts remain supported as an advanced/compatibility option; neither tool
requires this directory layout:

```text
<SCP parent>/
├── SCP/
└── mods/
    ├── ACT/
    └── bmtool/
```

Notebook bootstrap and ordinary passive/active/FI/simulation actions do not
download these repositories.

To reuse checkouts stored elsewhere:

```bash
export SCP_ACT_PATH=/path/to/ACT
export SCP_BMTOOL_PATH=/path/to/bmtool
```

Explicit tool-path overrides take precedence over the corresponding environment
package; unset them to return to ordinary installed-package use. A bad explicit
path is an error, not a reason to substitute another copy. Without an override,
SCP tries the importable package, then existing legacy `mods/` locations.

To choose checkout destinations, use `SCP_ACT_DIR` and
`SCP_BMTOOL_DIR`. Cloning is **off by default** for both tools;
`SCP_AUTO_CLONE_ACT=1` or `SCP_AUTO_CLONE_BMTOOL=1` explicitly enables legacy
last-resort cloning. That advanced fallback follows the selected repository
branch, not the pinned installation recipe; use the suggested requirements for
reproducibility. The setup checker never clones.
Repository URLs and
branches can be overridden with `SCP_ACT_REPO_URL` /
`SCP_ACT_REPO_BRANCH` and `SCP_BMTOOL_REPO_URL` /
`SCP_BMTOOL_REPO_BRANCH`.

ACT is optional for Step 2 target-derived proposals and Step 3 optimization.
Core passive sweeps and manual active/FI checks run without ACT. BMTool is
optional and used only when Step 4 synapse tuning is requested. SCP's environment
includes ACT's Python-side `scikit-learn` and `timeout-decorator` dependencies;
after adding ACT to an older environment, update from `environment.yml` before
using the compact optimizer controls.

SCP treats optional tools as read-only dependencies. Generated ACT
workspaces and all SCP adapters remain inside the selected workspace/SCP package;
SCP does not patch files in ACT or BMTool.

## Validate the Workspace

First register the named interface in the activated environment with
`python -m pip install -e .` (or the suggested `.[act,bmtool]` command above)
from the SCP checkout. This exposes `scp_pipeline`
and `scp-pipeline` without replacing scientific dependency versions. Reinstall
if the checkout moves or packaging metadata changes; restart kernels after code
updates. See the [project-owned dependency workflow](guides/dependency_workflow.md).

Run the setup checker:

```bash
python scripts/check_setup.py --steps 1 2 3 4 5 --cell PV --tune tuned --compile-modfiles
```

ACT and BMTool remain optional in this check. Add `--check-act` and/or
`--check-bmtool` only when validating those external integrations.
Both checks verify imports and the required stage-specific APIs, not just the
presence of a source folder:

```bash
scp-pipeline check --steps 4 --cell PV --tune tuned --check-bmtool
scp-pipeline check --steps 2 3 --cell PV --tune tuned --check-act
```

Run notebook checks:

```bash
python scripts/check_notebooks.py
```

If a tune contains custom `.mod` sources and is missing compiled mechanisms,
build them once (using that tune's configured MOD source directory):

```bash
cd cells/PV/tunes/tuned/modfiles
nrnivmodl
```

or use `--compile-modfiles` with `scripts/check_setup.py`, or let Step 1 compile
them. Models that use only built-in NEURON mechanisms need no compilation.
Compiled `x86_64/` folders are generated artifacts and are ignored by Git.

## Colab

The root notebooks can bootstrap a fresh Colab runtime:

- `0_pipeline.ipynb` (compact Steps 1–5 entry point)
- `1_setup.ipynb`
- `2_passive.ipynb`
- `3_active.ipynb`
- `4_synapses.ipynb`
- `5_simulate.ipynb`
- `6_analysis.ipynb`
- `7_tools.ipynb`

Useful environment overrides:

- `SCP_REPO_URL`
- `SCP_REPO_BRANCH`
- `SCP_REPO_DIR`
- `SCP_GIT_TOKEN`, `SCP_GITHUB_TOKEN`, or `GITHUB_TOKEN`
- `SCP_ACT_REPO_URL`, `SCP_ACT_REPO_BRANCH`, `SCP_ACT_DIR`, `SCP_ACT_PATH`
- `SCP_BMTOOL_REPO_URL`, `SCP_BMTOOL_REPO_BRANCH`,
  `SCP_BMTOOL_DIR`, `SCP_BMTOOL_PATH`

For private repositories, store a GitHub token in Colab secrets and set
`SCP_GIT_TOKEN` before running the notebook bootstrap cell.

For BMTool in Colab, complete the normal bootstrap and explicitly install
`bmtool==0.8.6` with `%pip install bmtool==0.8.6`, then restart/reconnect as
needed. Do not apply the Python 3.11 requirements constraints to Colab's newer
Python compatibility environment. BMTool initialization does not run pip.

For ACT in Colab, complete bootstrap and explicitly install the same pinned
source with `%pip install "act @ git+https://github.com/V-Marco/ACT.git@2b3fdc190107b8242e3496329cfdf240abd09694"`.
Use the Colab compatibility environment rather than Python 3.11 constraints,
and restart/reconnect as needed. ACT initialization does not install packages
or clone repositories by default.

## Large Local Data

Do not commit downloaded Allen/ADB ephys `.nwb` files, saved simulation outputs,
compiled mechanisms, or notebook scratch exports. The default `.gitignore`
excludes these generated/local artifacts.
