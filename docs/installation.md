Installation

Recommended local setup (Conda)
1. Clone SCP and enter the repo:
   `git clone <SCP_REPO_URL> && cd SCP`
2. Create and activate the environment:
   `conda env create -f environment.yml`
   `conda activate scp-py311`
3. Register a Jupyter kernel:
   `python -m ipykernel install --user --name scp-py311 --display-name "Python (SCP)"`

Alternative local setup (venv + pip)
1. Create and activate a virtual environment:
   `python -m venv .venv`
   `source .venv/bin/activate`
2. Install dependencies:
   `pip install -r requirements.txt`
3. Register a Jupyter kernel:
   `python -m ipykernel install --user --name scp-venv --display-name "Python (SCP venv)"`

External repos for steps 1-4
1. Clone ACT and bmtool next to SCP (default discovery path):
   `mkdir -p ../mods`
   `git clone https://github.com/V-Marco/ACT.git ../mods/ACT`
   `git clone https://github.com/cyneuro/bmtool.git ../mods/bmtool`
2. If stored elsewhere, set env vars before running notebooks:
   `export SCP_ACT_PATH=/path/to/ACT`
   `export SCP_BMTOOL_PATH=/path/to/bmtool`

Validate environment + workspace
1. Run the setup checker:
   `python scripts/check_setup.py --steps 0 1 2 3 4 5 --cell PV --tune seg_tuned`
2. If mechanisms are missing, build during check:
   `python scripts/check_setup.py --steps 5 --cell PV --tune seg_tuned --compile-modfiles`
3. Lint notebooks for portability and duplicate-key config issues:
   `python scripts/check_notebooks.py`

Manual modfile build (per tune)
- `cd <tune_dir>/modfiles && nrnivmodl`

Colab/Linux bootstrapped notebooks
- `colab_notebooks/2_colab.ipynb`, `colab_notebooks/3_colab.ipynb`, and `colab_notebooks/5_colab.ipynb` install dependencies and clone required repos in a fresh Colab session.
- Optional Colab environment overrides:
  - SCP repo: `SCP_REPO_URL`, `SCP_REPO_BRANCH`, `SCP_REPO_DIR`, `SCP_GIT_TOKEN`.
  - ACT repo (Steps 2/3): `SCP_ACT_REPO_URL`, `SCP_ACT_REPO_BRANCH`, `SCP_ACT_DIR`, `SCP_ACT_PATH`.
