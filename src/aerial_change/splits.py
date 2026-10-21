from __future__ import annotations

from datetime import date
from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from aerial_change.leakage import assert_disjoint
from aerial_change.schema import REQUIRED_COLUMNS

_MISSING = {"", "nan", "none", "nat"}


def _to_date(value) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _MISSING:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        try:
            return date.fromisoformat(text.split(" ")[0])
        except ValueError:
            raise ValueError(f"unparseable date: {value!r}") from None


def validate_pair_dates(df: pd.DataFrame) -> None:
    if "date_t0" not in df.columns and "date_t1" not in df.columns:
        return
    if "date_t0" not in df.columns or "date_t1" not in df.columns:
        raise ValueError("manifest has only one of date_t0/date_t1; both are required together")
    for idx, row in df.iterrows():
        t0 = _to_date(row["date_t0"])
        t1 = _to_date(row["date_t1"])
        if t0 is None and t1 is None:
            continue
        if t0 is None or t1 is None:
            raise ValueError(
                f"row {idx}: only one date present "
                f"(date_t0={row['date_t0']!r}, date_t1={row['date_t1']!r})"
            )
        if t0 >= t1:
            raise ValueError(f"row {idx}: date_t0 {t0} must be strictly before date_t1 {t1}")


def load_manifest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df["chip_id"] = df["chip_id"].astype(str)
    df["scene_id"] = df["scene_id"].astype(str)
    validate_pair_dates(df)
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


VALID_SPLITS = ("train", "val", "test")


def assigned_splits(
    df: pd.DataFrame, seed: int = 42, val_size: float = 0.15
) -> dict[str, pd.DataFrame]:
    values = set(df["split"].astype(str))
    bad = values - set(VALID_SPLITS)
    if bad:
        raise ValueError(f"unknown split values: {sorted(bad)}")
    if "train" not in values or "test" not in values:
        raise ValueError("assigned splits must contain at least train and test")
    parts = {}
    for name in VALID_SPLITS:
        part = df[df["split"].astype(str) == name].copy()
        if len(part) or name in values:
            parts[name] = part
    if "val" not in values:
        pool = parts["train"]
        scenes = pool.drop_duplicates("scene_id")[["scene_id"]].reset_index(drop=True)
        tv, val = train_test_split(scenes, test_size=val_size, random_state=seed)
        val_scenes = set(val["scene_id"])
        parts["train"] = pool[~pool["scene_id"].isin(val_scenes)].copy()
        parts["val"] = pool[pool["scene_id"].isin(val_scenes)].copy()
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert_disjoint(parts[a]["scene_id"], parts[b]["scene_id"], f"{a}/{b}")
    return parts


def split_manifest(df: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    if "split" in df.columns:
        return assigned_splits(df, seed=seed)
    return scene_level_split(df, seed=seed)


def holdout_aoi_split(
    df: pd.DataFrame,
    aoi_values: Sequence[str],
    seed: int = 42,
    val_size: float = 0.15,
) -> dict[str, pd.DataFrame]:
    pool, test = holdout_aoi(df, aoi_values)
    if test.empty:
        raise ValueError(f"aoi holdout matched no rows: {list(aoi_values)}")
    scenes = pool.drop_duplicates("scene_id")[["scene_id"]].reset_index(drop=True)
    tv, val = train_test_split(scenes, test_size=val_size, random_state=seed)
    train_scenes = set(tv["scene_id"])
    val_scenes = set(val["scene_id"])
    test_scenes = set(test["scene_id"])
    assert_disjoint(train_scenes, val_scenes, "train/val")
    assert_disjoint(train_scenes, test_scenes, "train/test")
    assert_disjoint(val_scenes, test_scenes, "val/test")
    train = pool[pool["scene_id"].isin(train_scenes)].copy()
    val_df = pool[pool["scene_id"].isin(val_scenes)].copy()
    return {"train": train, "val": val_df, "test": test}


def holdout_aoi(df: pd.DataFrame, aoi_values: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = df["aoi_id"].astype(str).isin([str(v) for v in aoi_values])
    test, train = df[mask].copy(), df[~mask].copy()
    assert_disjoint(train["scene_id"], test["scene_id"], "aoi holdout")
    return train, test
