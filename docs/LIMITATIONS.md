# LIMITATIONS — read before quoting any number

Every result in `experiments/` must be read with these limits in mind.

## Data

- **Crops, not full scenes.** The mirror stores sub-area crops of the
  official OSCD regions (about 0.9-12.6 km^2 each, ~0.04% of the official
  100 km^2 frames). Chips total 726; official-test is 230 chips from 10
  regions. Small test sets: one region's idiosyncrasy moves AUPRC by
  several points. See docs/DATA_CARD.md.
- **No acquisition dates.** Temporal/seasonal analysis is impossible; the
  pairs are 2015-2018 per the OSCD paper.
- **No georeferencing.** Chips are pixel-space; no CRS or real-world
  coordinates exist in the mirror.

## Physics the baselines do not handle

- **Clouds and shadows.** Optical Sentinel-2 scenes differ in cloud and
  shadow cover between dates; both look like "change" to a difference
  model and are not labelled in the masks, so part of the label noise is
  systematic.
- **Seasonal vegetation.** Green-up, senescence and mowing between dates
  produce large RGB differences with no human-annotated change. RGB-only
  input (3 of 13 bands) cannot separate phenology from change; NIR/NDVI
  would.
- **Water levels and tides.** Several regions contain rivers; shoreline
  and turbidity changes are unlabelled appearance change.
- **Sun angle and illumination.** Between-date lighting shifts inflate
  |t1 - t0| everywhere; the threshold baseline is directly exposed to
  this, and the CNN partly learns around it only where training scenes
  show it (an AOI-holdout risk).

## Method

- **Baselines are intentionally trivial.** threshold = mean |t1 - t0| per
  chip; pair = a 2-conv CNN on stacked dates. They establish the floor,
  not the frontier. No pretraining, no augmentation, RGB uint8 only.
- **High chip prevalence.** Changes cluster spatially, so 76% of chips
  contain some change and AUPRC must be compared against prevalence
  (AUPRC of a random scorer = prevalence), never against 0.5.
- **Operating point.** "Precision at FPR <= 0.05" selects the threshold on
  validation only; at that strict point recall is very low (few chips are
  flagged at all) and is not yet reported per run.
- **Val carve variance.** Validation is 3 regions carved from the official
  14-train pool per seed; different carves select different thresholds,
  which is the main source of spread in test precision at FPR <= 0.05.
- **AOI holdout size.** The geography-holdout mode holds out 2 regions
  from the 14-region official train pool; a 2-region sample of "unseen
  geography" is indicative, not conclusive.
- **Non-commercial labels.** OSCD masks are CC BY-NC-SA 4.0: results and
  masks must not be used for commercial products without a separate
  licence.
