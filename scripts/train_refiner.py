#!/usr/bin/env python3
"""Train the crop U-Net refiner on local CPU/MPS. Does not read test labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco
from filament.paths import SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.refiner import CropRefinerDataset, TinyUNet, tversky_loss


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--medoid-only", action="store_true")
    args = parser.parse_args()

    device = pick_device(args.device)
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    train_files = sorted(fn for fn, fold in fold_of.items() if fold != 0)
    ds = CropRefinerDataset(
        views, by_file, train_files, TRAIN_IMG_DIR, medoid_only=args.medoid_only
    )
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)
    model = TinyUNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in tqdm(loader, desc=f"refiner {epoch}/{args.epochs}"):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = bce(logits, y) + tversky_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item()) * x.size(0)
            n += x.size(0)
        mean_loss = running / max(1, n)
        history.append({"epoch": epoch, "loss": mean_loss})
        print(json.dumps(history[-1]))

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    out = WEIGHTS_DIR / "refiner.pt"
    torch.save(model.state_dict(), out)
    meta = {
        "device": device,
        "epochs": args.epochs,
        "n_crops": len(ds),
        "weights": str(out),
        "history": history,
    }
    (WEIGHTS_DIR / "refiner_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("device", "epochs", "n_crops", "weights")}, indent=2))


if __name__ == "__main__":
    main()
