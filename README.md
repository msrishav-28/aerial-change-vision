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
pip install -r requirements.txt && pip install -e .
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

## Rules baked into tests

- split by `scene_id` (or `flight_id` for drone)
- chips from one scene cannot appear in two splits
- `date_t0` strictly before `date_t1`, or the run fails
- labels containing tank / missile / warship / enemy **fail the build**
- report AUPRC and precision at FPR ≤ 0.05, not accuracy

## Next

1. Download OSCD or chip two Sentinel-2 dates over **one** AOI (ponds, reservoir).
2. Write `data/processed/manifest.csv` with real `scene_id` / `aoi_id`.
3. Log threshold baseline vs `pair` model vs an AOI holdout in `experiments/`.
4. Optional: same footprint Sentinel-1, report optical/SAR disagreement.

Method transfer for multi-sensor internships: [`docs/AURIC_TRANSFER.md`](docs/AURIC_TRANSFER.md).
Keep that file about *splits and sensors*, not product ontology.

## License

MIT for code. Imagery licences stay with the source (Copernicus / OSCD / etc.).
