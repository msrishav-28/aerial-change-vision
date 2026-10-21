import pandas as pd
import pytest

from aerial_change.splits import (
    holdout_aoi,
    load_manifest,
    scene_level_split,
    validate_pair_dates,
)


def _toy():
    rows = []
    for scene in range(12):
        for chip in range(3):
            rows.append(
                {
                    "chip_id": f"s{scene}_{chip}",
                    "scene_id": f"s{scene}",
                    "aoi_id": "east" if scene < 3 else "west",
                    "path_t0": f"t0_{scene}_{chip}.pt",
                    "path_t1": f"t1_{scene}_{chip}.pt",
                    "label": "change" if scene % 2 == 0 else "no_change",
                }
            )
    return pd.DataFrame(rows)


def test_scene_split_does_not_break_a_scene():
    parts = scene_level_split(_toy(), seed=0)
    for name, part in parts.items():
        for sid, g in part.groupby("scene_id"):
            assert set(g["split"]) == {name}
    ids = {k: set(v["scene_id"]) for k, v in parts.items()}
    assert ids["train"].isdisjoint(ids["test"])


def test_aoi_holdout():
    train, test = holdout_aoi(_toy(), ["east"])
    assert set(test["aoi_id"]) == {"east"}
    assert set(train["scene_id"]).isdisjoint(set(test["scene_id"]))


def test_validate_pair_dates_rejects_bad_order():
    df = pd.DataFrame(
        [
            {
                "chip_id": "c0",
                "scene_id": "s0",
                "label": "change",
                "date_t0": "2024-06-01",
                "date_t1": "2024-01-01",
            }
        ]
    )
    with pytest.raises(ValueError, match="strictly before"):
        validate_pair_dates(df)


def test_validate_pair_dates_rejects_equal_dates():
    df = pd.DataFrame(
        [
            {
                "chip_id": "c0",
                "scene_id": "s0",
                "label": "change",
                "date_t0": "2024-06-01",
                "date_t1": "2024-06-01",
            }
        ]
    )
    with pytest.raises(ValueError, match="strictly before"):
        validate_pair_dates(df)


def test_validate_pair_dates_rejects_one_sided():
    df = pd.DataFrame(
        [
            {
                "chip_id": "c0",
                "scene_id": "s0",
                "label": "change",
                "date_t0": "2024-01-01",
                "date_t1": None,
            }
        ]
    )
    with pytest.raises(ValueError, match="only one date"):
        validate_pair_dates(df)


def test_validate_pair_dates_rejects_one_column_only():
    df = pd.DataFrame(
        [{"chip_id": "c0", "scene_id": "s0", "label": "change", "date_t0": "2024-01-01"}]
    )
    with pytest.raises(ValueError, match="both are required"):
        validate_pair_dates(df)


def test_validate_pair_dates_accepts_valid_and_missing():
    df = pd.DataFrame(
        [
            {
                "chip_id": "c0",
                "scene_id": "s0",
                "label": "change",
                "date_t0": "2024-01-01",
                "date_t1": "2024-06-01",
            },
            {"chip_id": "c1", "scene_id": "s0", "label": "no_change"},
        ]
    )
    validate_pair_dates(df)


def test_load_manifest_validates_dates(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text(
        "chip_id,scene_id,label,date_t0,date_t1\n"
        "c0,s0,change,2024-06-01,2024-01-01\n"
    )
    with pytest.raises(ValueError, match="strictly before"):
        load_manifest(str(path))
