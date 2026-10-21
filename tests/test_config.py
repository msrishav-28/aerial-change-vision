import yaml

from aerial_change.config import ALLOWED_KEYS, load_config


def _write_config(tmp_path, **overrides):
    cfg = {
        "seed": 42,
        "image_size": 64,
        "batch_size": 4,
        "epochs": 10,
        "lr": 0.001,
        "model": "pair",
        "manifest": "data/processed/example_manifest.csv",
        "allow_placeholder": False,
    }
    cfg.update(overrides)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_load_config_happy(tmp_path):
    cfg = load_config(_write_config(tmp_path))
    assert cfg["seed"] == 42
    assert set(cfg) == set(ALLOWED_KEYS)


def test_unknown_key_fails(tmp_path):
    path = _write_config(tmp_path, oops=1)
    try:
        load_config(path)
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown key")


def test_missing_key_fails(tmp_path):
    path = _write_config(tmp_path)
    data = yaml.safe_load(path.read_text())
    del data["lr"]
    path.write_text(yaml.safe_dump(data))
    try:
        load_config(path)
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing key")


def test_bad_model_fails(tmp_path):
    path = _write_config(tmp_path, model="resnet")
    try:
        load_config(path)
    except ValueError as exc:
        assert "model" in str(exc)
    else:
        raise AssertionError("expected ValueError for bad model")
