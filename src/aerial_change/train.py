from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from aerial_change.baselines import threshold_scores
from aerial_change.config import load_config
from aerial_change.dataset import PairChipDataset
from aerial_change.metrics import (
    binary_report,
    precision_at_threshold,
    select_threshold_at_fpr,
)
from aerial_change.models import build_model
from aerial_change.runs import (
    git_state,
    library_versions,
    manifest_sha256,
    run_out_path,
    split_hash,
)
from aerial_change.splits import (
    aoi_holdout_pool,
    holdout_aoi_split,
    load_manifest,
    split_manifest,
)

MAX_FPR = 0.05


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def collect_scores(
    model, model_name: str, loader: DataLoader, device
) -> tuple[np.ndarray, np.ndarray]:
    ys, ss = [], []
    for batch in loader:
        pairs = batch["pair"].to(device)
        if model_name == "threshold":
            scores = threshold_scores(pairs)
        else:
            scores = torch.softmax(model(pairs), dim=-1)[:, 1]
        ys.append(batch["y"].numpy())
        ss.append(scores.cpu().numpy())
    y = np.concatenate(ys) if ys else np.array([])
    s = np.concatenate(ss) if ss else np.array([])
    return y, s


def evaluate_split(model, model_name: str, loader: DataLoader, device) -> dict:
    y, s = collect_scores(model, model_name, loader, device)
    out = binary_report(y, s)
    threshold = select_threshold_at_fpr(y, s, MAX_FPR)
    if threshold is not None:
        out["prec_at_fpr_0.05"] = precision_at_threshold(y, s, threshold)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None, choices=["pair", "threshold"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--allow-placeholder", action="store_true")
    parser.add_argument(
        "--holdout-aoi",
        default=None,
        help="comma-separated aoi_id values held out entirely as the test split",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = args.manifest or cfg["manifest"]
    model_name = args.model or cfg["model"]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["batch_size"]
    lr = args.lr if args.lr is not None else cfg["lr"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    allow_placeholder = args.allow_placeholder or cfg["allow_placeholder"]
    effective = {
        **cfg,
        "manifest": manifest,
        "model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "allow_placeholder": allow_placeholder,
    }

    seed_everything(seed)
    utc_start = datetime.now(timezone.utc)

    df = load_manifest(manifest)
    if args.holdout_aoi:
        aois = [a.strip() for a in args.holdout_aoi.split(",") if a.strip()]
        parts = holdout_aoi_split(aoi_holdout_pool(df), aois, seed=seed)
        split_mode = "aoi_holdout"
    else:
        parts = split_manifest(df, seed=seed)
        split_mode = "assigned" if "split" in df.columns else "seeded"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {
        name: PairChipDataset(
            parts[name], image_size=cfg["image_size"], allow_placeholder=allow_placeholder
        )
        for name in ("train", "val", "test")
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size)

    history: list[dict] = []
    model = None
    if model_name == "pair":
        model = build_model("pair").to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                logits = model(batch["pair"].to(device))
                loss = loss_fn(logits, batch["y"].to(device))
                loss.backward()
                optimizer.step()
            row = {"epoch": epoch, **evaluate_split(model, model_name, val_loader, device)}
            history.append(row)
            print(row)

    y_val, s_val = collect_scores(model, model_name, val_loader, device)
    y_test, s_test = collect_scores(model, model_name, test_loader, device)
    threshold = select_threshold_at_fpr(y_val, s_val, MAX_FPR)
    val_report = binary_report(y_val, s_val)
    test_report = binary_report(y_test, s_test)
    if threshold is not None:
        val_report["prec_at_fpr_0.05"] = precision_at_threshold(y_val, s_val, threshold)
        test_report["prec_at_fpr_0.05_val_threshold"] = precision_at_threshold(
            y_test, s_test, threshold
        )

    n_placeholder = sum(len(ds.placeholder_chip_ids) for ds in datasets.values())
    utc_end = datetime.now(timezone.utc)

    record = {
        "utc_start": utc_start.isoformat(),
        "utc_end": utc_end.isoformat(),
        "command": " ".join(sys.argv),
        "config_file": args.config,
        "config": effective,
        "manifest": {"path": manifest, "sha256": manifest_sha256(manifest)},
        "split": {
            "seed": seed,
            "mode": split_mode,
            "aois": args.holdout_aoi,
            "hash": split_hash(parts),
            "n_scenes": {k: int(v["scene_id"].nunique()) for k, v in parts.items()},
        },
        "model": model_name,
        "git": git_state(),
        "versions": library_versions(),
        "platform": platform.platform(),
        "device": str(device),
        "history": history,
        "val": val_report,
        "test": test_report,
        "threshold": {"selected_on": "val", "max_fpr": MAX_FPR, "value": threshold},
        "placeholder_data": n_placeholder > 0,
        "n_placeholder": n_placeholder,
        "note": (
            "Placeholder chips were substituted for missing files; smoke test only, not a result."
            if n_placeholder
            else "All chips loaded from manifest files."
        ),
    }

    out = Path(args.out) if args.out else run_out_path("experiments", model_name, seed, utc_start)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing run record: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str))
    print(f"wrote {out}")
    print(
        f"placeholder_data={record['placeholder_data']} "
        f"val={val_report} test={test_report}"
    )


if __name__ == "__main__":
    main()
