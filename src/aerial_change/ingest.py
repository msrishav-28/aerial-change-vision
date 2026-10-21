from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

RGB_BANDS = (3, 2, 1)  # B04, B03, B02
N_BANDS = 13
SCALE_MAX = 10000.0
GSD_M = 10.0
SENSOR = "sentinel-2"
LICENSE = "OSCD labels: CC BY-NC-SA 4.0; imagery: ESA Sentinel-2 (open)"

COLUMNS = [
    "chip_id",
    "scene_id",
    "label",
    "split",
    "aoi_id",
    "path_t0",
    "path_t1",
    "path_mask",
    "date_t0",
    "date_t1",
    "gsd_m",
    "sensor",
    "license",
]


def select_rgb_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3 or img.shape[0] != N_BANDS:
        raise ValueError(f"expected 13-band image, got shape {img.shape}")
    rgb = img[list(RGB_BANDS)].astype(np.float32)
    rgb = np.clip(rgb, 0.0, SCALE_MAX) / SCALE_MAX * 255.0
    return np.floor(rgb + 0.5).astype(np.uint8)


def pad_to_grid(arr: np.ndarray, chip: int, mode: str) -> np.ndarray:
    if mode not in {"symmetric", "constant"}:
        raise ValueError(f"pad mode must be 'symmetric' or 'constant', got {mode!r}")
    h, w = arr.shape[-2], arr.shape[-1]
    ph = (chip - h % chip) % chip
    pw = (chip - w % chip) % chip
    if ph == 0 and pw == 0:
        return arr
    spec = [(0, 0)] * (arr.ndim - 2) + [(0, ph), (0, pw)]
    if mode == "constant":
        return np.pad(arr, spec, mode="constant", constant_values=0)
    return np.pad(arr, spec, mode="symmetric")


def chip_windows(h: int, w: int, chip: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(0, h, chip) for c in range(0, w, chip)]


def chip_label(mask_chip: np.ndarray) -> str:
    return "change" if (mask_chip > 0).any() else "no_change"


def ingest_region(
    img_t0: np.ndarray,
    img_t1: np.ndarray,
    mask: np.ndarray,
    scene_id: str,
    chip: int,
    out_dir: Path,
    rel_prefix: str,
    split: str,
) -> list[dict]:
    if img_t0.shape != img_t1.shape:
        raise ValueError(
            f"date images differ in shape: {img_t0.shape} vs {img_t1.shape}"
        )
    if mask.shape != img_t0.shape[1:]:
        raise ValueError(
            f"mask shape {mask.shape} does not match image spatial shape {img_t0.shape[1:]}"
        )

    rgb0 = pad_to_grid(select_rgb_uint8(img_t0), chip, "symmetric")
    rgb1 = pad_to_grid(select_rgb_uint8(img_t1), chip, "symmetric")
    mbinary = pad_to_grid((mask > 0).astype(np.uint8), chip, "constant")

    scene_dir = Path(out_dir) / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for r0, c0 in chip_windows(rgb0.shape[-2], rgb0.shape[-1], chip):
        r_i, c_i = r0 // chip, c0 // chip
        chip_id = f"{scene_id}_r{r_i:03d}c{c_i:03d}"
        stack = {
            "_t0.pt": rgb0[:, r0 : r0 + chip, c0 : c0 + chip],
            "_t1.pt": rgb1[:, r0 : r0 + chip, c0 : c0 + chip],
            "_mask.pt": mbinary[r0 : r0 + chip, c0 : c0 + chip],
        }
        paths = {}
        for suffix, block in stack.items():
            name = f"{chip_id}{suffix}"
            torch.save(torch.from_numpy(block.copy()), scene_dir / name)
            paths[suffix] = f"{rel_prefix}/{scene_id}/{name}"
        rows.append(
            {
                "chip_id": chip_id,
                "scene_id": scene_id,
                "label": chip_label(mbinary[r0 : r0 + chip, c0 : c0 + chip]),
                "split": split,
                "aoi_id": scene_id,
                "path_t0": paths["_t0.pt"],
                "path_t1": paths["_t1.pt"],
                "path_mask": paths["_mask.pt"],
                "date_t0": "",
                "date_t1": "",
                "gsd_m": GSD_M,
                "sensor": SENSOR,
                "license": LICENSE,
            }
        )
    return rows


def decode_region(table, row_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def decode_image(column) -> np.ndarray:
        bands = column.combine_chunks()[row_idx].values
        return np.stack(
            [
                np.stack(
                    [r.values.to_numpy(zero_copy_only=False) for r in bands[b].values]
                )
                for b in range(len(bands))
            ]
        )

    item = table.column("mask").combine_chunks()[row_idx]
    png = Image.open(io.BytesIO(item["bytes"].as_py()))
    mask = (np.array(png) > 0).astype(np.uint8)
    return decode_image(table.column("image1")), decode_image(table.column("image2")), mask


def build_manifest(
    raw: dict[str, Path],
    out_chips: Path,
    out_manifest: Path,
    chip_size: int,
    rel_prefix: str,
) -> pd.DataFrame:
    import pyarrow.parquet as pq

    rows: list[dict] = []
    for split, path in raw.items():
        table = pq.read_table(path)
        for i in range(table.num_rows):
            img_t0, img_t1, mask = decode_region(table, i)
            rows.extend(
                ingest_region(
                    img_t0=img_t0,
                    img_t1=img_t1,
                    mask=mask,
                    scene_id=f"oscd_{split}_{i:02d}",
                    chip=chip_size,
                    out_dir=out_chips,
                    rel_prefix=rel_prefix,
                    split=split,
                )
            )
    df = pd.DataFrame(rows, columns=COLUMNS)
    out_chips = Path(out_chips)
    out_manifest = Path(out_manifest)
    out_chips.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_manifest, index=False)
    return df
