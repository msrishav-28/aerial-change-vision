# Experiments

One JSON per run, named `run_<UTC timestamp>_<model>_seed<seed>.json`.
Run records are never overwritten.

Each record carries: UTC start/end, exact command, config file and effective
config, manifest path + sha256, split mode + seed + scene-to-split hash, git
SHA + dirty flag, library versions, platform, device, per-epoch validation
history, final validation and test metrics (AUPRC; precision at FPR <= 0.05
with the operating threshold chosen on validation only), and a
`placeholder_data` flag.

Smoke runs on the example manifest are stamped `placeholder_data: true` and
are never results.

## Real-tile runs (2026-10-21)

Data: OSCD mirror crops, 726 chips, 128x128 at 10 m, official 14/10 region
split with validation carved from the official train pool
(docs/DATA_CARD.md). All records below have `placeholder_data: false`.
AUPRC must be read against chip prevalence (a random scorer scores
~prevalence, not 0.5): val prevalence 0.64-0.75 by split carve, test
prevalence 0.817.

Official test split (10 regions, 230 chips), mean across seeds 42/43/44:

- threshold baseline: test AUPRC 0.849 (identical across seeds, the scorer
  is deterministic); test precision at FPR <= 0.05 via val-chosen
  threshold 0.824 / 0.874 / 0.842 (mean 0.847).
- pair model (2-conv CNN): test AUPRC 0.827 / 0.852 / 0.839 (mean 0.839);
  test precision at FPR <= 0.05 via val-chosen threshold 0.750 / 0.852 /
  0.862 (mean 0.821).

Reading: neither baseline separates changes much beyond base rate on this
data (AUPRC barely above prevalence); the learned pair model does NOT beat
the naive difference baseline here. Records:
`run_20261021T0733*_threshold_seed4{2,3,4}.json`,
`run_20261021T0734*_pair_seed4{2,3,4}.json`.

AOI holdout (regions `oscd_train_03` + `oscd_train_08`, 146 chips,
prevalence 0.774, held out entirely from training; training pool restricted
to the official train regions so runs stay comparable to the table above):

- threshold: test AUPRC 0.902 (identical across seeds); test precision at
  FPR <= 0.05 via val-chosen threshold 1.0 for all three seeds.
- pair: test AUPRC 0.904 / 0.911 / 0.914 (mean 0.910); test precision at
  FPR <= 0.05 0.929 / nan / 0.915. The nan is real: seed 43's
  validation-chosen threshold flagged zero test chips at that strict
  operating point, and precision is undefined — reported as-is rather than
  silently dropped.

Reading: both models do better on these two unseen regions than on the
official test split (0.90-0.91 vs 0.849 AUPRC), but two large high-contrast
regions are an easier and much smaller sample; treat as indicative, not as
proof of geographic generalization. The pair model again ties the naive
baseline. Records: `run_20261021T075920Z_pair_seed42`,
`run_20261021T080033Z_pair_seed43`, `run_20261021T080133Z_pair_seed44`,
`run_20261021T080240Z`..`080253Z_threshold_seed4{2,3,4}` (all with
`split.mode = aoi_holdout`).

Earlier broad-pool holdout records (`run_20261021T0750*`,
`run_20261021T0755*`) trained on 18 regions including the 10 official-test
regions; kept for the record, superseded by the restricted runs above and
not quoted as results.
