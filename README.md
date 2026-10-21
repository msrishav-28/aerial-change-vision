# aerial-change-vision

Tile-level **EO change** (optional SAR / drone later) with scene and AOI holdouts.

Civilian AOIs only. This is **not** ATR, targeting, VisDrone-mAP-as-a-product,
or a continent-scale foundation model.

Until `experiments/` has a run on real tiles, there are **no IoU claims**.

## Question

Did this place change between date A and date B, when the test scene was
never in training?

Protocol: [`docs/QUESTION.md`](docs/QUESTION.md), [`docs/TASK.md`](docs/TASK.md),
[`docs/LEAKAGE.md`](docs/LEAKAGE.md), [`docs/NOT_CLAIMED.md`](docs/NOT_CLAIMED.md).

## Setup

```bash
git clone https://github.com/msrishav-28/aerial-change-vision
cd aerial-change-vision
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e . && pip install -e ".[ingest]"
pytest -q
python scripts/check_leakage.py
python scripts/train_baseline.py --epochs 1 --allow-placeholder
python scripts/train_baseline.py --model threshold --allow-placeholder
```

Settings live in `configs/default.yaml`; unknown or missing keys fail on load.
CLI flags override the config. Every run writes a unique, never-overwritten
record under `experiments/`.

Example chips are placeholders: missing chip files fail loudly unless
`--allow-placeholder` is passed; placeholder runs are stamped
`placeholder_data: true`; they are smoke tests, never results. Do not quote them.

## Real data (OSCD)

`data/` ships a manifest over the OSCD change-detection dataset via a public
mirror: 24 Sentinel-2 region pairs (native 10 m crops), 13-band verified
ingest, binary human change masks, official 14/10 train/test split kept
intact. Licences and disclosed deviations: [`docs/DATA_CARD.md`](docs/DATA_CARD.md).
Regenerate locally (raw parquets are gitignored):

```bash
.venv/Scripts/python.exe scripts/build_oscd_manifest.py   # writes 726 chips + manifest
python scripts/check_leakage.py --manifest data/processed/oscd_manifest.csv
python scripts/train_baseline.py --config configs/oscd.yaml --model threshold --seed 42
python scripts/train_baseline.py --config configs/oscd.yaml --model pair --seed 42
python scripts/train_baseline.py --config configs/oscd.yaml --model pair --seed 42 \
  --holdout-aoi oscd_train_03,oscd_train_08
```

## Rules baked into tests

- split by `scene_id` (or `flight_id` for drone)
- chips from one scene cannot appear in two splits
- `date_t0` strictly before `date_t1`, or the run fails
- labels containing tank / missile / warship / enemy **fail the build**
- report AUPRC and precision at FPR ≤ 0.05, not accuracy

## Results

Real-tile baselines (2026-10-21), official OSCD test split, 230 chips,
10 held-out regions; mean over seeds 42/43/44 — full provenance in the
linked run records under `experiments/`:

| model | test AUPRC | prec @ FPR ≤ 0.05 (val-chosen) |
|---|---|---|
| threshold (mean abs diff) | 0.849 (prevalence 0.817) | 0.847 ± 0.021 |
| pair (2-conv CNN) | 0.839 ± 0.010 | 0.821 ± 0.048 |
| AOI holdout: threshold | 0.902 (prevalence 0.774) | 1.0 / 1.0 / 1.0 |
| AOI holdout: pair | 0.910 ± 0.004 | 0.929 / nan / 0.915 |

Neither baseline separates changes much beyond base rate, and the learned
pair model does **not** beat the naive difference baseline on this data.
The AOI-holdout rows test two regions never seen in training (easier, much
smaller sample; one seed's strict threshold flags nothing → precision
undefined, reported as nan). Read with
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) (crops, RGB-only,
clouds/seasonality unhandled). AUPRC of a random scorer equals prevalence,
not 0.5.

## Next

1. Log the AOI-holdout summary in `experiments/` (runs complete; README note pending).
2. Report recall at the FPR ≤ 0.05 operating point alongside precision.
3. Use the remaining 10 bands (NIR/NDVI) to separate vegetation phenology from change.
4. Optional: same footprint Sentinel-1, report optical/SAR disagreement.

Method transfer for multi-sensor internships: [`docs/AURIC_TRANSFER.md`](docs/AURIC_TRANSFER.md).
Keep that file about *splits and sensors*, not product ontology.

## License

MIT for code. Imagery licences stay with the source (Copernicus / OSCD / etc.).
