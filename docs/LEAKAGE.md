# Leakage rules

Encoded in `src/aerial_change/leakage.py` and tests.

## Forbidden

1. Putting chips from the same `scene_id` in train and test.
2. Overlapping chip windows that straddle the train/test AOI boundary
   without an exclusion buffer.
3. Splitting drone data by *image* so adjacent frames of one flight leak.
4. Using the post-event mask as an input channel.
5. Training on date t1 RGB and testing on a crop from the same t1 scene.

## Required

- Split key is `scene_id` (change) or `flight_id` (drone).
- Optional `aoi_id` holdout for domain shift (train district A, test district B).
- Chip size and pair order `(t0, t1)` are fixed.
- Record GSD / altitude when known.

## Why

A random 80/20 on VisDrone or OSCD chips is the aerial twin of
pothole-YOLO mAP. Adjacent tiles share the same field and the same date.
