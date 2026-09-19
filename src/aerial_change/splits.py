from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from aerial_change.leakage import assert_disjoint
from aerial_change.schema import REQUIRED_COLUMNS


def load_manifest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df["chip_id"] = df["chip_id"].astype(str)
    df["scene_id"] = df["scene_id"].astype(str)
    return df


def scene_level_split(
    df: pd.DataFrame,
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.2,
) -> dict[str, pd.DataFrame]:
    scenes = df.drop_duplicates("scene_id")[["scene_id"]].copy()
    if "label" in df.columns:
        # stratify on majority label per scene when possible
        lab = (
            df.groupby("scene_id")["label"]
            .agg(lambda s: s.value_counts().index[0])
            .rename("label")
        )
        scenes = scenes.merge(lab, on="scene_id", how="left")
        strat = scenes["label"] if scenes["label"].nunique() > 1 else None
    else:
        strat = None
    trainval, test = train_test_split(
        scenes, test_size=test_size, random_state=seed, stratify=strat
    )
    strat_tv = trainval["label"] if strat is not None else None
    rel_val = val_size / max(1.0 - test_size, 1e-6)
    train, val = train_test_split(
        trainval, test_size=rel_val, random_state=seed, stratify=strat_tv
    )
    assert_disjoint(train["scene_id"], val["scene_id"], "train/val")
    assert_disjoint(train["scene_id"], test["scene_id"], "train/test")
    assert_disjoint(val["scene_id"], test["scene_id"], "val/test")
    keys = {
        "train": set(train["scene_id"]),
        "val": set(val["scene_id"]),
        "test": set(test["scene_id"]),
    }
    out = {}
    for name, ids in keys.items():
        part = df[df["scene_id"].isin(ids)].copy()
        part["split"] = name
        out[name] = part
    return out


def holdout_aoi(df: pd.DataFrame, aoi_values: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = df["aoi_id"].astype(str).isin([str(v) for v in aoi_values])
    test, train = df[mask].copy(), df[~mask].copy()
    assert_disjoint(train["scene_id"], test["scene_id"], "aoi holdout")
    return train, test
