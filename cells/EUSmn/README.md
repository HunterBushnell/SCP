# EUSmn cell

This bundle is a single-compartment EUS motor-neuron example loaded through
SCP's `hoc_template` adapter.

- `tunes/orig` preserves the LUT PUD template used as the starting model.
- `tunes/tuned` contains the manually derived EUSmn template and tune settings.
- Runtime conditions are `-70 mV`/`34 °C` for `orig` and `-55 mV`/`34 °C` for
  `tuned`.

From the SCP repository root, compile and check the tuned bundle with:

```bash
python scripts/check_setup.py --cell EUSmn --tune tuned --steps 1 2 3 6 7 --compile-modfiles
python run_pipeline.py --cell EUSmn --tune tuned --iclamp --sim-overrides-json '{"save_output": false}'
```

The tune is provided as a reproducible manual-tuning example, not as an
independent biological-validation claim. See each tune's
`SOURCE_PROVENANCE.json` for exact source lineage and hashes.

The HOC and MOD model sources are redistributed with permission, are excluded
from SCP's MIT license, and have no upstream license asserted here.
