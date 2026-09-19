# Data card

## Change detection (task A)

| source | notes |
|---|---|
| OSCD | Sentinel-2 urban change pairs, public research set |
| SECOND | aerial change, check licence |
| Self-chipped S2 | one AOI, two dates, document tile IDs and cloud filter |

## Drone (task B, optional)

VisDrone / UAVDT / CARPK: civilian traffic and parking only.
Keep class names generic. Do not rename `car` to a weapons class.

## Manifest (`data/processed/manifest.csv`)

| column | meaning |
|---|---|
| chip_id | row id |
| scene_id | **split key** for change |
| flight_id | **split key** for drone |
| aoi_id | optional geography holdout |
| path_t0, path_t1 | pair paths or `.pt` |
| path_mask | optional change mask |
| label | `change` / `no_change` or class name |
| date_t0, date_t1 | ISO dates |
| gsd_m | metres per pixel if known |
| sensor | `s2` / `s1` / `drone` |
| license | source licence |
