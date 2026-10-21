#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from aerial_change.ingest import build_manifest

EXPECTED_REGIONS = {"train": 14, "test": 10}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/OSCD_MSI")
    parser.add_argument("--out-chips", default="data/processed/oscd/chips")
    parser.add_argument("--out-manifest", default="data/processed/oscd_manifest.csv")
    parser.add_argument("--chip-size", type=int, default=128)
    parser.add_argument("--rel-prefix", default="data/processed/oscd/chips")
    args = parser.parse_args()

    raw = {
        "train": Path(args.raw_dir) / "train.parquet",
        "test": Path(args.raw_dir) / "test.parquet",
    }
    for split, path in raw.items():
        n = pq.ParquetFile(path).metadata.num_rows
        if n != EXPECTED_REGIONS[split]:
            raise SystemExit(
                f"{split}: expected {EXPECTED_REGIONS[split]} regions per official "
                f"OSCD spec, found {n} in {path}"
            )

    df = build_manifest(
        raw=raw,
        out_chips=Path(args.out_chips),
        out_manifest=Path(args.out_manifest),
        chip_size=args.chip_size,
        rel_prefix=args.rel_prefix,
    )
    print("regions:", df["scene_id"].nunique())
    print("chips:", len(df))
    print("labels:", df["label"].value_counts().to_dict())
    print("per-split chips:", df.groupby("split").size().to_dict())


if __name__ == "__main__":
    main()
