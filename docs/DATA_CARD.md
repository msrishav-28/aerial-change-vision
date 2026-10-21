# DATA_CARD — OSCD via public mirror

## What this data is

Tile-level change-detection pairs from the Onera Satellite Change Detection
(OSCD) dataset: 24 registered Sentinel-2 image pairs (13 spectral bands) with
binary human-annotated change masks (urban changes: new buildings, roads).
Official dataset: 14 train / 10 test region pairs, Daudt et al. 2018,
https://rcdaudt.github.io/oscd/ (paper: arXiv:1810.08468).

## Source and provenance

- Downloaded mirror: Hugging Face dataset `blanchon/OSCD_MSI` (parquet),
  last modified 2023-12-05, files:
  - `data/train-00000-of-00001.parquet` (273,527,703 bytes)
    sha256 `b6d6ddeefbf0d9a3994acecd2d3d124f23258b3724d134f0d25f633b87560996`
  - `data/test-00000-of-00001.parquet` (124,258,410 bytes)
    sha256 `eb62220dc006298e790287f9b51810c4443f48884018f24bf13ec4e6828f63ca`
- The mirror card declares `license: unknown`; we therefore rely on the
  ORIGINAL OSCD terms below.

## Licences (must be respected downstream)

- Change masks / annotations: CC BY-NC-SA 4.0 — attribution + share-alike +
  NON-COMMERCIAL use only.
- Sentinel-2 imagery: open data, Copernicus / ESA
  (https://sentinel.esa.int/web/sentinel/missions/sentinel-2).
- The manifest records this per row in the `license` column.

## Verification performed before ingest (2026-10-21)

- Row counts match the official split: 14 train + 10 test = 24 pairs.
- Every region has exactly 13 bands; per-band value ranges match Sentinel-2
  L1C/L2A reflectance scaled x10000.
- Every region has a mask of the same height/width as its images; masks are
  binary (0/1) after thresholding; region change fractions 0.3%-10%.
- Visual inspection of a rendered region: individual rooftops, roads and
  bridges resolve at ~1-3 px — consistent with native ~10 m GSD.

## Deviations from the official OSCD distribution (disclosed)

- The mirror stores SUB-AREA CROPS of the official regions at native 10 m
  resolution, not the full 10,000x10,000 frames. Region crops range from
  241x385 to 1180x1070 px (about 0.9-12.6 km^2; official frames are
  100 km^2 each). Roughly 0.04% of official pixel area.
- Region names are not present in the mirror; rows carry a fixed but unknown
  ordering. We assign deterministic synthetic ids `oscd_train_00`..`13`,
  `oscd_test_00`..`09` by row index (same parquet file + row order = same
  region).
- Acquisition dates are not present; `date_t0`/`date_t1` are blank in the
  manifest (the loader's both-or-neither rule allows this). Temporal
  analysis is therefore impossible; the paper documents 2015-2018.
- Georeferencing/CRS is not present; chips are pixel-space only.

## Transformations applied at ingest (scripts/build_oscd_manifest.py)

- Band selection: RGB = bands B04, B03, B02 (true color). The other 10 bands
  were available and dropped; revisit before any spectral work.
- Scaling: reflectance DN clipped to [0, 10000] and linearly mapped to
  uint8 [0, 255] (value 10000 -> 255).
- Tiling: each region symmetric-padded to a multiple of 128 px and cut into
  128x128 chips (1.28 km at 10 m). Masks zero-padded, so padded areas can
  never create positive labels. 726 chips total (496 official-train,
  230 official-test).
- Chip label: `change` if any mask pixel in the chip is set, else
  `no_change`. Changes cluster spatially, so chip-level prevalence is high
  (555/726 change) even though pixel-level change is 0.3-10% per region.

## Splits and leakage

- The official train/test assignment is kept exactly: `split` column =
  train (14 regions) / test (10 regions) in the manifest; the 10 test
  regions are never trained on.
- Validation is carved out of the official-train regions at run time
  (seeded, scene-level): 11 train / 3 val / 10 test regions under the
  default seed.
- One region = one scene_id = one aoi_id, so scene-level and AOI-level
  holdouts coincide; `--holdout-aoi` holds out entire regions.
- Chips from one region never appear in two splits (enforced by
  `assigned_splits` disjointness assertions).
- Boundary buffer: chips within a region are adjacent by construction and
  always share the region's split; no cross-split adjacency is possible,
  so the buffer check is satisfied by construction and not separately
  computed (no real-world coordinates are available in the mirror).

## Regenerating

```
# raw parquets must be under data/raw/OSCD_MSI/ (gitignored)
.venv/Scripts/python.exe scripts/build_oscd_manifest.py
# verifies 14/10 region counts, then writes
# data/processed/oscd/chips/*.pt and data/processed/oscd_manifest.csv
```

The manifest CSV is committed so run records can pin its sha256; chip .pt
files are local artifacts (gitignored).
