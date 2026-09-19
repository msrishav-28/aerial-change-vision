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
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest -q
python scripts/check_leakage.py
python scripts/train_baseline.py --epochs 1
```

Example chips are placeholders. Do not quote that run.

## Rules baked into tests

- split by `scene_id` (or `flight_id` for drone)
- chips from one scene cannot appear in two splits
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
