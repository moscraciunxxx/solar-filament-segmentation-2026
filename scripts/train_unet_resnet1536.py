#!/usr/bin/env python3
"""Train ResNet34 full-disk U-Net at 1536. Never overwrites unet_resnet34.

Dest is WEIGHTS_DIR/unet_resnet34_1536. Val loss is not official PQ. Never submits.
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
from gate_unet_family import RESNET_GATE, live_mps_owner, refuse_resnet
from train_unet import pick_device

DEST_NAME = "unet_resnet34_1536"
FORBIDDEN = "unet_resnet34"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gate-json", default=str(RESNET_GATE))
    args = parser.parse_args()

    out_dir = WEIGHTS_DIR / DEST_NAME
    if DEST_NAME == FORBIDDEN or FORBIDDEN in Path(out_dir).parts and out_dir.name == FORBIDDEN:
        raise SystemExit("Refusing 1536 train: dest would overwrite unet_resnet34.")
    if out_dir.resolve() == (WEIGHTS_DIR / FORBIDDEN).resolve():
        raise SystemExit("Refusing 1536 train: dest resolved to unet_resnet34.")
    owner = live_mps_owner()
    if owner:
        raise SystemExit(f"Refusing 1536 train: {owner} owns MPS.")
    reason = refuse_resnet(Path(args.gate_json))
    if reason:
        raise SystemExit(reason)

    device = pick_device(args.device)
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    train_files = sorted(fn for fn, fold in fold_of.items() if fold != 0)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    train_ds = FullDiskSemanticDataset(
        views, by_file, train_files, TRAIN_IMG_DIR, imgsz=int(args.imgsz), augment=True
    )
    val_ds = FullDiskSemanticDataset(
        views, by_file, val_files, TRAIN_IMG_DIR, imgsz=int(args.imgsz), augment=False
    )
    train_loader = DataLoader(train_ds, batch_size=int(args.batch), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_resnet34_unet(in_ch=3, pretrained=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss()

    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    history = []
    best_val = float("inf")

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y, _ in tqdm(train_loader, desc=f"resnet34_1536 {epoch}/{args.epochs}"):
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
            "family": "unet_resnet34_s1536",
            "best_val_loss": best_val,
            "best_epoch": int(min(history, key=lambda r: r["val_loss"])["epoch"]),
            "history": history,
            "val_loss_is_not_official_pq": True,
            "overwrote_unet_resnet34": False,
            "imgsz": int(args.imgsz),
        }
        (out_dir / "train_curve.json").write_text(json.dumps(curve, indent=2) + "\n")
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "unet_resnet34_1536_train_curve.json").write_text(json.dumps(curve, indent=2) + "\n")

    print(json.dumps({"best": str(best_path), "device": device, "do_not_submit": True}, indent=2))


if __name__ == "__main__":
    main()
