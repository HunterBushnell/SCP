Colab Notebooks Quickstart

Notebooks in this folder
- `2_colab.ipynb`: Step 2 passive tuning (bootstrapped).
- `3_colab.ipynb`: Step 3 active tuning (bootstrapped).
- `5_colab.ipynb`: Step 5 full pipeline (bootstrapped).

What "bootstrapped" means
- The notebook can clone SCP in-session.
- The notebook can install missing Python/system dependencies.
- The notebook can compile NEURON mechanisms when needed.

Run in Colab (public repo)
1. Open a Colab runtime.
2. Upload/open one notebook from this folder.
3. Run cells from top to bottom.

Run in Colab (private repo)
1. Create a GitHub token with read access to the private repo.
2. In Colab, store the token in Secrets (recommended) or set env vars in a cell.
3. Run a setup cell before the notebook bootstrap cell:

```python
import os
from google.colab import userdata

os.environ["SCP_GIT_TOKEN"] = userdata.get("SCP_GIT_TOKEN")
os.environ["SCP_REPO_URL"] = "https://github.com/<org>/<private-scp-repo>.git"
os.environ["SCP_REPO_BRANCH"] = "main"      # optional
os.environ["SCP_REPO_DIR"] = "/content/SCP" # optional
os.environ["SCP_AUTO_CLONE"] = "1"
```

Optional ACT overrides (Step 2/3)
- `SCP_ACT_REPO_URL`
- `SCP_ACT_REPO_BRANCH`
- `SCP_ACT_DIR`
- `SCP_ACT_PATH`
- `SCP_AUTO_CLONE_ACT` (`1` by default)

Common issues
- `SCP_REPO_DIR exists but is not an SCP checkout`
  - Delete the folder or set a different `SCP_REPO_DIR`.
- Clone/auth errors on private repos
  - Verify token scope and org SSO authorization.
- `nrnivmodl`/mechanism errors
  - Re-run bootstrap cell, then re-run the compile/load cell.

Recommended class flow
1. Run once with defaults.
2. Confirm one successful baseline run.
3. Change one parameter block at a time.
