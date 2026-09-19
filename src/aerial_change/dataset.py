from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from aerial_change.leakage import assert_civilian_label
from aerial_change.schema import ChipRecord


def row_to_record(row: pd.Series) -> ChipRecord:
    def get(name):
        if name not in row.index:
            return None
        val = row[name]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return val

    rec = ChipRecord(
        chip_id=str(row["chip_id"]),
        scene_id=str(row["scene_id"]),
        label=str(row["label"]),
        flight_id=None if get("flight_id") is None else str(get("flight_id")),
        aoi_id=None if get("aoi_id") is None else str(get("aoi_id")),
        path_t0=None if get("path_t0") is None else str(get("path_t0")),
        path_t1=None if get("path_t1") is None else str(get("path_t1")),
        sensor=None if get("sensor") is None else str(get("sensor")),
    )
    assert_civilian_label(rec.label)
    return rec


class PairChipDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, image_size: int = 64, root: str | Path = "."):
        self.df = manifest.reset_index(drop=True)
        self.image_size = image_size
        self.root = Path(root)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        rec = row_to_record(self.df.iloc[idx])
        pair = []
        for pth in (rec.path_t0, rec.path_t1):
            path = self.root / pth if pth else None
            if path and path.suffix == ".pt" and path.exists():
                pair.append(torch.load(path, map_location="cpu"))
            else:
                g = torch.Generator().manual_seed(abs(hash(rec.chip_id + str(pth))) % (2**31))
                pair.append(torch.rand(3, self.image_size, self.image_size, generator=g))
        frames = torch.stack(pair, dim=0)
        y = 1 if rec.is_change else 0
        return {"pair": frames.float(), "y": torch.tensor(y), "scene_id": rec.scene_id}
