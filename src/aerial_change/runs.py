from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch


def run_out_path(directory: str | Path, model: str, seed: int, now: datetime) -> Path:
    candidate = Path(directory) / f"run_{now.strftime('%Y%m%dT%H%M%SZ')}_{model}_seed{seed}.json"
    counter = 1
    while candidate.exists():
        candidate = Path(directory) / (
            f"run_{now.strftime('%Y%m%dT%H%M%SZ')}_{model}_seed{seed}_{counter}.json"
        )
        counter += 1
    return candidate


def manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_hash(parts: dict[str, pd.DataFrame]) -> str:
    scene_to_split: dict[str, str] = {}
    for name, part in parts.items():
        for scene in part["scene_id"].astype(str).unique():
            scene_to_split[scene] = name
    blob = json.dumps(scene_to_split, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def git_state() -> dict:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"sha": sha, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {"sha": None, "dirty": None, "reason": "not a git repository (or git unavailable)"}


def library_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }
