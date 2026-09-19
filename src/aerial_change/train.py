from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from aerial_change.dataset import PairChipDataset
from aerial_change.metrics import binary_report, precision_at_fpr
from aerial_change.models import build_model
from aerial_change.splits import load_manifest, scene_level_split


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        logits = model(batch["pair"].to(device))
        prob = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        ys.append(batch["y"].numpy())
        ps.append(prob)
    y = np.concatenate(ys) if ys else np.array([])
    p = np.concatenate(ps) if ps else np.array([])
    out = binary_report(y, p)
    if len(y):
        out["prec_at_fpr_0.05"] = precision_at_fpr(y, p, 0.05)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/processed/example_manifest.csv")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="experiments/last_run.json")
    args = p.parse_args()

    df = load_manifest(args.manifest)
    parts = scene_level_split(df, seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("pair").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(PairChipDataset(parts["train"]), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(PairChipDataset(parts["val"]), batch_size=args.batch_size)

    history = []
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            logits = model(batch["pair"].to(device))
            loss = loss_fn(logits, batch["y"].to(device))
            loss.backward()
            opt.step()
        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch, **val}
        history.append(row)
        print(row)

    payload = {
        "seed": args.seed,
        "device": str(device),
        "n_scenes": {k: int(v["scene_id"].nunique()) for k, v in parts.items()},
        "history": history,
        "note": "Placeholder chips if GeoTIFFs are missing. Not a result.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
