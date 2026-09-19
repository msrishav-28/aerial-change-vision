# Tasks

Pick **one** primary task. Do not start all three.

## A — Bi-temporal change (default)

Input: co-registered pair `(t0, t1)` plus optional Sentinel-1 pair.
Output: change mask or change / no-change chip label.
Hold out an entire AOI or an entire date pair.

Intended public sets: OSCD, SECOND, or two Sentinel-2 dates you chip yourself
for **one** Indian AOI (ponds, reservoir water, airfield civil apron).

## B — Small-object count (optional later)

Input: one nadir or oblique frame from a **flight**.
Output: object counts in civilian classes only (`vehicle`, `person`, `boat`, `other`).
Split by `flight_id`, bin by object area and altitude / GSD.

VisDrone / UAVDT are allowed **only** with flight-level splits and a failure gallery.
mAP on a random image split is not a result.

## C — Optical + SAR agreement (optional)

Same footprint, S2 change vs S1 change. Report disagreement, not a fused 99% IoU.

## Baselines (A)

1. Threshold on `|t1 - t0|` or NDWI / NDVI delta
2. Siamese tiny CNN on the pair
3. Then one holdout AOI
