# Experiments

One JSON per run, named `run_<UTC timestamp>_<model>_seed<seed>.json`.
Run records are never overwritten.

Each record carries: UTC start/end, exact command, config file and effective
config, manifest path + sha256, split seed + scene-to-split hash, git SHA +
dirty flag, library versions, platform, device, per-epoch validation history,
final validation and test metrics (AUPRC; precision at FPR <= 0.05 with the
operating threshold chosen on validation only), and a `placeholder_data` flag.

Smoke runs on the example manifest are stamped `placeholder_data: true` and
are never results. No README numbers until a real-tile run exists here.
