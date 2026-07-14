# Installation

## Local Conda Setup

```bash
git clone <SCP_REPO_URL>
cd SCP
conda env create -f environment.yml
conda activate scp-py311
python -m ipykernel install --user --name scp-py311 --display-name "Python (SCP)"
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

Steps 2-4 use or reserve integrations with external tools:

```bash
mkdir -p ../mods
git clone https://github.com/V-Marco/ACT.git ../mods/ACT
git clone https://github.com/cyneuro/bmtool.git ../mods/bmtool
```

If stored elsewhere:

```bash
export SCP_ACT_PATH=/path/to/ACT
export SCP_BMTOOL_PATH=/path/to/bmtool
```

ACT is required for Step 2 passive tuning and optional Step 3 ACT active tuning.
Step 3's manual active sweep/FI checks can run without ACT. BMTool is used by
Step 4 synapse tuning.

## Validate the Workspace

Run the setup checker:

```bash
python scripts/check_setup.py --steps 1 2 3 4 5 --cell PV --tune tuned --compile-modfiles
```

Run notebook checks:

```bash
python scripts/check_notebooks.py
```

If a tune is missing compiled mechanisms, build them once:

```bash
cd cells/PV/tunes/tuned/modfiles
nrnivmodl
```

or use `--compile-modfiles` with `scripts/check_setup.py`, or let Step 1 compile
them. Compiled `x86_64/` folders are generated artifacts and are ignored by Git.

## Colab

The root notebooks can bootstrap a fresh Colab runtime:

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
- `SCP_BMTOOL_PATH`

For private repositories, store a GitHub token in Colab secrets and set
`SCP_GIT_TOKEN` before running the notebook bootstrap cell.

## Large Local Data

Do not commit downloaded Allen/ADB ephys `.nwb` files, saved simulation outputs,
compiled mechanisms, or notebook scratch exports. The default `.gitignore`
excludes these generated/local artifacts.
