#!/usr/bin/env python3
"""Train ResNet34 full-disk U-Net. Gated. Do not start while v1 owns MPS.

Start only after eval_unet_merge writes RESEARCH_HOLD, RESEARCH_HOLD_SPINE_V2,
or KILL_WORSE_OR_DESERT. Never overwrites unet_semantic. Never submits.
"""

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
sys.path.insert(0, str(ROOT / "scripts"))

from filament.coco_io import grouped_folds, load_coco
from filament.paths import ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.refiner import tversky_loss
from filament.semantic_dataset import FullDiskSemanticDataset
from filament.unet_resnet import build_resnet34_unet
from gate_unet_family import GATE, refuse_resnet
from train_unet import pick_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gate-json", default=str(GATE))
    args = parser.parse_args()

    reason = refuse_resnet(Path(args.gate_json))
    if reason:
        raise SystemExit(reason)

    device = pick_device(args.device)
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    train_files = sorted(fn for fn, fold in fold_of.items() if fold != 0)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    train_ds = FullDiskSemanticDataset(
        views, by_file, train_files, TRAIN_IMG_DIR, imgsz=args.imgsz, augment=True
    )
    val_ds = FullDiskSemanticDataset(
        views, by_file, val_files, TRAIN_IMG_DIR, imgsz=args.imgsz, augment=False
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_resnet34_unet(in_ch=3, pretrained=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss()

    out_dir = WEIGHTS_DIR / "unet_resnet34"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    history = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y, _ in tqdm(train_loader, desc=f"resnet34 {epoch}/{args.epochs}"):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = bce(logits, y) + tversky_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item()) * x.size(0)
            n += x.size(0)
        train_loss = running / max(1, n)

        model.eval()
        val_running = 0.0
        val_n = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = bce(logits, y) + tversky_loss(logits, y)
                val_running += float(loss.item()) * x.size(0)
                val_n += x.size(0)
        val_loss = val_running / max(1, val_n)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save(model.state_dict(), out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
        curve = {
            "family": "unet_resnet34_s1024",
            "best_val_loss": best_val,
            "best_epoch": int(min(history, key=lambda r: r["val_loss"])["epoch"]),
            "history": history,
            "val_loss_is_not_official_pq": True,
            "overwrote_unet_semantic": False,
        }
        (out_dir / "train_curve.json").write_text(json.dumps(curve, indent=2))
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "unet_resnet34_train_curve.json").write_text(json.dumps(curve, indent=2))

    print(json.dumps({"best": str(best_path), "device": device, "do_not_submit": True}, indent=2))


if __name__ == "__main__":
    main()
