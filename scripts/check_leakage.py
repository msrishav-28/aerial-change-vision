#!/usr/bin/env python3
from __future__ import annotations

import argparse

from aerial_change.dataset import row_to_record
from aerial_change.leakage import assert_civilian_label, assert_disjoint
from aerial_change.splits import load_manifest, scene_level_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/processed/example_manifest.csv")
    args = p.parse_args()
    df = load_manifest(args.manifest)
    for _, row in df.iterrows():
        rec = row_to_record(row)
        assert_civilian_label(rec.label)
    parts = scene_level_split(df)
    assert_disjoint(parts["train"]["scene_id"], parts["test"]["scene_id"])
    print("ok", {k: v["scene_id"].nunique() for k, v in parts.items()})


if __name__ == "__main__":
    main()
