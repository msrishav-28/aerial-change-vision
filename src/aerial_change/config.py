from __future__ import annotations

from pathlib import Path

import yaml

ALLOWED_KEYS = (
    "seed",
    "image_size",
    "batch_size",
    "epochs",
    "lr",
    "model",
    "manifest",
    "allow_placeholder",
)

MODELS = ("pair", "threshold")


def load_config(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")
    unknown = sorted(set(raw) - set(ALLOWED_KEYS))
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    missing = [key for key in ALLOWED_KEYS if key not in raw]
    if missing:
        raise ValueError(f"missing config keys: {missing}")
    _validate(raw)
    return raw


def _validate(cfg: dict) -> None:
    if cfg["model"] not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {cfg['model']!r}")
    if cfg["seed"] < 0:
        raise ValueError("seed must be >= 0")
    if cfg["image_size"] <= 0:
        raise ValueError("image_size must be > 0")
    if cfg["batch_size"] <= 0:
        raise ValueError("batch_size must be > 0")
    if cfg["epochs"] < 0:
        raise ValueError("epochs must be >= 0")
    if cfg["lr"] <= 0:
        raise ValueError("lr must be > 0")
    if not isinstance(cfg["allow_placeholder"], bool):
        raise ValueError("allow_placeholder must be a boolean")
