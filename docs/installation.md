# Installation

## Local Conda Setup

```bash
git clone <SCP_REPO_URL>
cd SCP
conda env create -f environment.yml
conda activate scp-py311
python -m ipykernel install --user --name scp-py311 --display-name "scp-py311"
```

## Optional venv Setup

Conda is preferred because NEURON/AllenSDK dependencies are easier to reproduce,
but a venv can work on compatible systems:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name scp-venv --display-name "Python (SCP venv)"
```

## External Repositories

ACT and BMTool are optional. You do not need either one for the core SCP
workflow. When an ACT proposal/optimization or BMTool tuner action is first
requested, SCP checks configured and common locations and, if necessary,
clones a fresh checkout to the sibling `mods/` directory:

```text
<SCP parent>/
├── SCP/
└── mods/
    ├── ACT/
    └── bmtool/
```

Notebook bootstrap and ordinary passive/active/FI/simulation actions do not
download these tools. This keeps them optional while making the first
tool-specific action work the same way locally and in Colab.

To reuse checkouts stored elsewhere:

```bash
export SCP_ACT_PATH=/path/to/ACT
export SCP_BMTOOL_PATH=/path/to/bmtool
```

To choose installation destinations, use `SCP_ACT_DIR` and
`SCP_BMTOOL_DIR`. Set `SCP_AUTO_CLONE_ACT=0` or
`SCP_AUTO_CLONE_BMTOOL=0` to require manual provisioning. Repository URLs and
branches can be overridden with `SCP_ACT_REPO_URL` /
`SCP_ACT_REPO_BRANCH` and `SCP_BMTOOL_REPO_URL` /
`SCP_BMTOOL_REPO_BRANCH`.

ACT is optional for Step 2 target-derived proposals and Step 3 optimization.
Core passive sweeps and manual active/FI checks run without ACT. BMTool is
optional and used only when Step 4 synapse tuning is requested. SCP's environment
includes ACT's Python-side `scikit-learn` and `timeout-decorator` dependencies;
after adding ACT to an older environment, update from `environment.yml` before
using the compact optimizer controls.

SCP imports these repositories as read-only dependencies. Generated ACT
workspaces and all SCP adapters remain inside the selected tune/SCP checkout;
SCP does not patch files in ACT or BMTool.

## Validate the Workspace

Run the setup checker:

```bash
python scripts/check_setup.py --steps 1 2 3 4 5 --cell PV --tune tuned --compile-modfiles
```

ACT and BMTool remain optional in this check. Add `--check-act` and/or
`--check-bmtool` only when validating those external integrations.

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

## Large Local Data

Do not commit downloaded Allen/ADB ephys `.nwb` files, saved simulation outputs,
compiled mechanisms, or notebook scratch exports. The default `.gitignore`
excludes these generated/local artifacts.
