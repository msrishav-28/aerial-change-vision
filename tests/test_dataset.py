import pandas as pd
import pytest
import torch

from aerial_change.dataset import PairChipDataset, row_to_record


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chip_id": "c0",
                "scene_id": "s0",
                "label": "change",
                "path_t0": "missing_t0.pt",
                "path_t1": "missing_t1.pt",
            },
            {
                "chip_id": "c1",
                "scene_id": "s0",
                "label": "no_change",
                "path_t0": "missing_t0b.pt",
                "path_t1": "missing_t1b.pt",
            },
        ]
    )


def test_missing_file_fails_loudly(tmp_path):
    ds = PairChipDataset(_manifest(), root=tmp_path)
    with pytest.raises(FileNotFoundError, match="c0"):
        _ = ds[0]


def test_placeholder_is_opt_in_and_stamped():
    ds = PairChipDataset(_manifest(), allow_placeholder=True)
    item = ds[0]
    assert item["pair"].shape == (2, 3, 64, 64)
    assert item["y"].item() == 1
    assert ds.used_placeholder
    assert ds.placeholder_chip_ids == {"c0"}


def test_placeholder_is_deterministic():
    a = PairChipDataset(_manifest(), allow_placeholder=True)[0]["pair"]
    b = PairChipDataset(_manifest(), allow_placeholder=True)[0]["pair"]
    assert torch.equal(a, b)


def test_row_to_record_populates_all_fields():
    row = pd.Series(
        {
            "chip_id": "c9",
            "scene_id": "s9",
            "label": "change",
            "flight_id": None,
            "aoi_id": "west",
            "path_t0": "a.pt",
            "path_t1": "b.pt",
            "path_mask": "m.png",
            "date_t0": "2024-01-01",
            "date_t1": "2024-06-01",
            "gsd_m": 10.0,
            "sensor": "s2",
            "license": "example",
        }
    )
    rec = row_to_record(row)
    assert rec.aoi_id == "west"
    assert rec.path_mask == "m.png"
    assert rec.date_t0 == "2024-01-01"
    assert rec.date_t1 == "2024-06-01"
    assert rec.gsd_m == 10.0
    assert rec.sensor == "s2"
    assert rec.license == "example"
    assert rec.flight_id is None
