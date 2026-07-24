# HYPO cell

This bundle is a single-compartment sympathetic preganglionic-neuron example
loaded through SCP's `hoc_template` adapter.

- `tunes/orig` preserves the LUT source template.
- `tunes/tuned` contains the manually tuned derivative.
- Both tunes use runtime conditions of `-62 mV` and `31 °C`.

From the SCP repository root, compile and check the tuned bundle with:

```bash
python scripts/check_setup.py --cell HYPO --tune tuned --steps 1 2 3 6 7 --compile-modfiles
python run_pipeline.py --cell HYPO --tune tuned --iclamp --sim-overrides-json '{"save_output": false}'
```

The tune is provided as a reproducible manual-tuning example, not as an
independent biological-validation claim. See each tune's
`SOURCE_PROVENANCE.json` for exact source lineage and hashes.

The HOC and MOD model sources are redistributed with permission, are excluded
from SCP's MIT license, and have no upstream license asserted here.
