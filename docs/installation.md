Installation

SCP assumes a working NEURON environment and compiled modfiles.

Local environment
- Python 3.9+ recommended.
- Core packages: numpy, scipy, pandas, matplotlib.
- NEURON with Python bindings must be installed and importable.

Modfiles
- Each tune has a `modfiles/` folder.
- Build once per tune:
  `cd <tune_dir>/modfiles && nrnivmodl`

Colab/Linux (bootstrapped)
- `5_colab.ipynb` can bootstrap a clean environment
  (install deps, compile modfiles, and run the pipeline end-to-end).
