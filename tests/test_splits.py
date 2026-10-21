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


# ---- assigned splits (precomputed, e.g. the official OSCD train/test) ----


def _assigned_toy():
    df = _toy()
    df["split"] = ["train"] * 18 + ["val"] * 9 + ["test"] * 9
    return df


def test_split_manifest_uses_assigned_column():
    from aerial_change.splits import split_manifest

    parts = split_manifest(_assigned_toy())
    assert set(parts) == {"train", "val", "test"}
    assert len(parts["train"]) == 18
    assert len(parts["val"]) == 9
    assert len(parts["test"]) == 9


def test_split_manifest_rejects_scene_in_two_splits():
    from aerial_change.splits import split_manifest

    df = _assigned_toy()
    df.loc[0, "split"] = "test"  # scene s0 now in train and test
    with pytest.raises(Exception, match="leakage"):
        split_manifest(df)


def test_split_manifest_rejects_unknown_split_value():
    from aerial_change.splits import split_manifest

    df = _assigned_toy()
    df.loc[0, "split"] = "holdout"
    with pytest.raises(ValueError, match="split"):
        split_manifest(df)


def test_split_manifest_requires_train_and_test():
    from aerial_change.splits import split_manifest

    df = _assigned_toy()
    df = df[df["split"] != "test"]
    with pytest.raises(ValueError, match="test"):
        split_manifest(df)


def test_split_manifest_falls_back_to_seeded_split():
    from aerial_change.splits import split_manifest

    parts = split_manifest(_toy(), seed=0)
    assert set(parts) == {"train", "val", "test"}


# ---- AOI holdout run mode ----


def test_holdout_aoi_split_keeps_aoi_scenes_out_of_training():
    from aerial_change.splits import holdout_aoi_split

    df = _toy()
    parts = holdout_aoi_split(df, aoi_values=["east"], seed=0)
    assert set(parts["test"]["aoi_id"]) == {"east"}
    assert "east" not in set(parts["train"]["aoi_id"])
    assert "east" not in set(parts["val"]["aoi_id"])
    train_scenes = set(parts["train"]["scene_id"])
    val_scenes = set(parts["val"]["scene_id"])
    test_scenes = set(parts["test"]["scene_id"])
    assert train_scenes.isdisjoint(val_scenes)
    assert train_scenes.isdisjoint(test_scenes)
    assert val_scenes.isdisjoint(test_scenes)
    assert test_scenes == {"s0", "s1", "s2"}


def test_holdout_aoi_split_rejects_empty_holdout():
    from aerial_change.splits import holdout_aoi_split

    with pytest.raises(ValueError, match="aoi"):
        holdout_aoi_split(_toy(), aoi_values=["nonexistent"], seed=0)


def test_split_manifest_carves_val_from_official_train():
    from aerial_change.splits import split_manifest

    df = _toy()
    df["split"] = ["train"] * 27 + ["test"] * 9  # scenes s0..s8 pool, s9..s11 test
    parts = split_manifest(df, seed=0)
    assert set(parts) == {"train", "val", "test"}
    val_scenes = set(parts["val"]["scene_id"])
    assert val_scenes
    assert val_scenes <= {"s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"}
    assert set(parts["test"]["scene_id"]) == {"s9", "s10", "s11"}
    all_ids = (
        set(parts["train"]["scene_id"])
        | set(parts["val"]["scene_id"])
        | set(parts["test"]["scene_id"])
    )
    assert all_ids == set(df["scene_id"])
    again = split_manifest(df, seed=0)
    assert set(again["val"]["scene_id"]) == val_scenes


def test_aoi_holdout_pool_excludes_assigned_test():
    from aerial_change.splits import aoi_holdout_pool

    df = _assigned_toy()
    pool = aoi_holdout_pool(df)
    assert set(pool["split"]) <= {"train", "val"}
    assert set(pool["scene_id"]) <= set(df[df["split"] != "test"]["scene_id"])
    # no split column: pool is everything
    assert len(aoi_holdout_pool(_toy())) == len(_toy())
