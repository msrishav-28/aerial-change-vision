import pandas as pd

from aerial_change.splits import holdout_aoi, scene_level_split


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
