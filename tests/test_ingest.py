import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from aerial_change.ingest import (
    chip_label,
    chip_windows,
    ingest_region,
    pad_to_grid,
    select_rgb_uint8,
)
from aerial_change.splits import load_manifest


def scaled(v: int) -> int:
    return int(np.floor(min(max(v, 0), 10000) / 10000.0 * 255.0 + 0.5))


def test_select_rgb_picks_b04_b03_b02_order():
    img = np.zeros((13, 2, 3), dtype=np.uint16)
    img[1] = 1000  # B02 blue
    img[2] = 2000  # B03 green
    img[3] = 3000  # B04 red
    out = select_rgb_uint8(img)
    assert out.shape == (3, 2, 3)
    assert out.dtype == np.uint8
    assert np.all(out[0] == scaled(3000))
    assert np.all(out[1] == scaled(2000))
    assert np.all(out[2] == scaled(1000))


def test_select_rgb_clips_above_10000():
    img = np.zeros((13, 1, 1), dtype=np.uint16)
    img[3] = 20000
    out = select_rgb_uint8(img)
    assert int(out[0, 0, 0]) == 255


def test_select_rgb_rejects_wrong_band_count():
    with pytest.raises(ValueError, match="13"):
        select_rgb_uint8(np.zeros((12, 4, 4), dtype=np.uint16))


def test_pad_reflects_image_and_preserves_original():
    arr = np.arange(6, dtype=np.uint16).reshape(2, 3)
    out = pad_to_grid(arr, chip=4, mode="symmetric")
    assert out.shape == (4, 4)
    assert np.array_equal(out[:2, :3], arr)
    assert out[2:, 3:].min() > 0 or out[2:, :3].min() > 0


def test_pad_mask_fills_with_zero():
    mask = np.array([[1]])
    out = pad_to_grid(mask, chip=4, mode="constant")
    assert out.shape == (4, 4)
    assert out[0, 0] == 1
    assert out.sum() == 1


def test_chip_windows_tiles_exact_grid():
    origins = chip_windows(h=12, w=12, chip=4)
    assert origins == [(r, c) for r in (0, 4, 8) for c in (0, 4, 8)]
    assert chip_windows(h=12, w=8, chip=4) == [
        (r, c) for r in (0, 4, 8) for c in (0, 4)
    ]


def test_chip_label_any_positive_pixel_is_change():
    assert chip_label(np.zeros((4, 4), dtype=np.uint8)) == "no_change"
    m = np.zeros((4, 4), dtype=np.uint8)
    m[2, 3] = 1
    assert chip_label(m) == "change"


def test_ingest_region_writes_chips_and_rows(tmp_path: Path):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 3000, size=(13, 20, 24), dtype=np.uint16)
    img2 = rng.integers(0, 3000, size=(13, 20, 24), dtype=np.uint16)
    mask = np.zeros((20, 24), dtype=np.uint8)
    mask[18:20, 0:2] = 1  # lands in chip at origin (16, 0)

    rows = ingest_region(
        img_t0=img,
        img_t1=img2,
        mask=mask,
        scene_id="oscd_train_00",
        chip=16,
        out_dir=tmp_path,
        rel_prefix="data/processed/oscd/chips",
        split="train",
    )

    assert len(rows) == 4
    change_rows = [r for r in rows if r["label"] == "change"]
    assert len(change_rows) == 1
    assert change_rows[0]["chip_id"] == "oscd_train_00_r001c000"

    for r in rows:
        assert r["scene_id"] == "oscd_train_00"
        assert r["aoi_id"] == "oscd_train_00"
        assert r["split"] == "train"
        assert r["gsd_m"] == 10.0
        assert r["date_t0"] == "" and r["date_t1"] == ""
        name_t0 = r["path_t0"].split("/")[-1]
        name_t1 = r["path_t1"].split("/")[-1]
        name_mask = r["path_mask"].split("/")[-1]
        assert (tmp_path / "oscd_train_00" / name_t0).exists()
        assert (tmp_path / "oscd_train_00" / name_t1).exists()
        assert (tmp_path / "oscd_train_00" / name_mask).exists()

    first = rows[0]
    t0 = torch.load(tmp_path / "oscd_train_00" / first["path_t0"].split("/")[-1], map_location="cpu")
    t1 = torch.load(tmp_path / "oscd_train_00" / first["path_t1"].split("/")[-1], map_location="cpu")
    m = torch.load(tmp_path / "oscd_train_00" / first["path_mask"].split("/")[-1], map_location="cpu")
    assert t0.shape == (3, 16, 16) and t0.dtype == torch.uint8
    assert t1.shape == (3, 16, 16) and t1.dtype == torch.uint8
    assert m.shape == (16, 16) and m.dtype == torch.uint8


def test_ingest_region_pixel_values_round_trip():
    import tempfile

    out_dir = Path(tempfile.mkdtemp())
    img = np.zeros((13, 16, 16), dtype=np.uint16)
    img[3, 0, 0] = 5000  # B04
    img[2, 0, 0] = 2500  # B03
    img[1, 0, 0] = 0  # B02
    mask = np.zeros((16, 16), dtype=np.uint8)

    rows = ingest_region(
        img_t0=img,
        img_t1=img.copy(),
        mask=mask,
        scene_id="s",
        chip=16,
        out_dir=out_dir,
        rel_prefix="x",
        split="train",
    )
    t0 = torch.load(out_dir / "s" / f"{rows[0]['chip_id']}_t0.pt", map_location="cpu")
    assert int(t0[0, 0, 0]) == scaled(5000)
    assert int(t0[1, 0, 0]) == scaled(2500)
    assert int(t0[2, 0, 0]) == scaled(0)


def _tmp() -> Path:
    import tempfile

    d = Path(tempfile.mkdtemp())
    return d


def test_ingest_region_rejects_mask_shape_mismatch():
    img = np.zeros((13, 20, 24), dtype=np.uint16)
    bad_mask = np.zeros((20, 25), dtype=np.uint8)
    with pytest.raises(ValueError, match="mask"):
        ingest_region(
            img_t0=img,
            img_t1=img.copy(),
            mask=bad_mask,
            scene_id="s",
            chip=16,
            out_dir=_tmp(),
            rel_prefix="x",
            split="train",
        )


# ---- end-to-end through a synthetic parquet with the mirror's schema ----


def _png_bytes(mask: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _write_parquet(path: Path, regions: list[tuple[np.ndarray, np.ndarray]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    nested = pa.list_(pa.list_(pa.list_(pa.uint16())))
    mask_type = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
    cols = {"image1": [], "image2": [], "mask": []}
    for img, mask in regions:
        cols["image1"].append(img.tolist())
        cols["image2"].append(img.tolist())
        cols["mask"].append({"bytes": _png_bytes(mask), "path": None})
    table = pa.table(
        {
            "image1": pa.array(cols["image1"], type=nested),
            "image2": pa.array(cols["image2"], type=nested),
            "mask": pa.array(cols["mask"], type=mask_type),
        }
    )
    pq.write_table(table, path)


def _region(h: int, w: int, change_rc: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    img = np.full((13, h, w), 2000, dtype=np.uint16)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[change_rc] = 1
    return img, mask


def test_build_manifest_end_to_end_through_dataset(tmp_path: Path):
    from aerial_change.dataset import PairChipDataset
    from aerial_change.ingest import build_manifest

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_parquet(raw_dir / "train.parquet", [_region(16, 16, (3, 5))])
    _write_parquet(raw_dir / "test.parquet", [_region(12, 12, (1, 1))])

    out_manifest = tmp_path / "manifest.csv"
    df = build_manifest(
        raw={"train": raw_dir / "train.parquet", "test": raw_dir / "test.parquet"},
        out_chips=tmp_path / "chips",
        out_manifest=out_manifest,
        chip_size=8,
        rel_prefix="chips",
    )

    assert len(df) == 8
    assert set(df["split"]) == {"train", "test"}
    assert set(df["scene_id"]) == {"oscd_train_00", "oscd_test_00"}
    assert set(df["label"]) <= {"change", "no_change"}
    assert (df["label"] == "change").sum() == 2

    loaded = load_manifest(str(out_manifest))
    train_rows = loaded[loaded["split"] == "train"].reset_index(drop=True)
    ds = PairChipDataset(train_rows, image_size=8, root=tmp_path, allow_placeholder=False)
    assert not ds.used_placeholder
    assert len(ds) == 4
    item = ds[0]
    assert item["pair"].shape == (2, 3, 8, 8)
    assert item["pair"].dtype == torch.float32
    assert item["y"].item() in (0, 1)
