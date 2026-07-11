# Example Run

Goal: run one PV tuned example, save it, and inspect outputs.

## Check Setup

```bash
python scripts/check_setup.py --steps 5 --cell PV --tune seg_tuned --compile-modfiles
```

## Run One Trial

```bash
python run_pipeline.py \
  --tune-dir cells/PV/tunes/seg_tuned \
  --n-trials 1 \
  --force-save \
  --output-stem example_pv_run
```

`--force-save` is included because public example configs may keep
`save.enabled` disabled by default to avoid accidental output growth.

## Find Outputs

```text
cells/PV/tunes/seg_tuned/output_data/example_pv_run/run_manifest.json
```

## Inspect Outputs

Open `6_analysis.ipynb` and select:

```python
cell_name = "PV"
tunes_dir = "tunes"
model_dir = "seg_tuned"
run_single_stem = "example_pv_run"
```

If modfiles are missing and you did not use `--compile-modfiles`:

```bash
cd cells/PV/tunes/seg_tuned/modfiles
nrnivmodl
```
