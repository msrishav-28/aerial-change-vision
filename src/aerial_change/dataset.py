from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from aerial_change.leakage import assert_civilian_label
from aerial_change.schema import ChipRecord


def row_to_record(row: pd.Series) -> ChipRecord:
    def get(name: str):
        if name not in row.index:
            return None
        val = row[name]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return val

    def get_str(name: str) -> str | None:
        val = get(name)
        return None if val is None else str(val)

    def get_float(name: str) -> float | None:
        val = get(name)
        return None if val is None else float(val)

    rec = ChipRecord(
        chip_id=str(row["chip_id"]),
        scene_id=str(row["scene_id"]),
        label=str(row["label"]),
        flight_id=get_str("flight_id"),
        aoi_id=get_str("aoi_id"),
        path_t0=get_str("path_t0"),
        path_t1=get_str("path_t1"),
        path_mask=get_str("path_mask"),
        date_t0=get_str("date_t0"),
        date_t1=get_str("date_t1"),
        gsd_m=get_float("gsd_m"),
        sensor=get_str("sensor"),
        license=get_str("license"),
    )
    assert_civilian_label(rec.label)
    return rec


class PairChipDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        image_size: int = 64,
        root: str | Path = ".",
        allow_placeholder: bool = False,
    ):
        self.df = manifest.reset_index(drop=True)
        self.image_size = image_size
        self.root = Path(root)
        self.allow_placeholder = allow_placeholder
        self.placeholder_chip_ids: set[str] = set()

    @property
    def used_placeholder(self) -> bool:
        return bool(self.placeholder_chip_ids)

    def __len__(self) -> int:
        return len(self.df)

    def _load_chip(self, chip_id: str, rel_path: str | None) -> torch.Tensor:
        path = self.root / rel_path if rel_path else None
        if path is not None and path.suffix == ".pt" and path.exists():
            return torch.load(path, map_location="cpu")
        if not self.allow_placeholder:
            raise FileNotFoundError(
                f"chip file missing for {chip_id}: {rel_path!r}; "
                "pass --allow-placeholder only for smoke runs on the example manifest"
            )
        self.placeholder_chip_ids.add(chip_id)
        # md5, not hash(): placeholder content must be identical across processes
        digest = hashlib.md5(f"{chip_id}:{rel_path}".encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:4], "big"))
        return torch.rand(3, self.image_size, self.image_size, generator=generator)

    def __getitem__(self, idx: int):
        rec = row_to_record(self.df.iloc[idx])
        pair = [
            self._load_chip(rec.chip_id, rec.path_t0),
            self._load_chip(rec.chip_id, rec.path_t1),
        ]
        frames = torch.stack(pair, dim=0)
        y = 1 if rec.is_change else 0
        return {"pair": frames.float(), "y": torch.tensor(y), "scene_id": rec.scene_id}
