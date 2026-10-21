import hashlib
from datetime import datetime, timezone

import pandas as pd

from aerial_change.runs import git_state, manifest_sha256, run_out_path, split_hash


def test_run_out_path_unique(tmp_path):
    now = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)
    first = run_out_path(tmp_path, "pair", 42, now)
    assert first.name == "run_20261020T120000Z_pair_seed42.json"
    first.write_text("{}")
    second = run_out_path(tmp_path, "pair", 42, now)
    assert first != second


def test_split_hash_stable_and_sensitive():
    parts_a = {"train": pd.DataFrame({"scene_id": ["s1", "s2"]})}
    parts_b = {"train": pd.DataFrame({"scene_id": ["s1", "s2"]})}
    parts_c = {
        "train": pd.DataFrame({"scene_id": ["s1"]}),
        "test": pd.DataFrame({"scene_id": ["s2"]}),
    }
    assert split_hash(parts_a) == split_hash(parts_b)
    assert split_hash(parts_a) != split_hash(parts_c)


def test_manifest_sha256(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("hello")
    assert manifest_sha256(path) == hashlib.sha256(b"hello").hexdigest()


def test_git_state_shape():
    state = git_state()
    assert "sha" in state
    assert "dirty" in state
