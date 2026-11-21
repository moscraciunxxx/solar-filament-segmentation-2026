#!/usr/bin/env python3
"""TinyUNet on disk-flatten+CLAHE+Sobel. Gated. Do not start before patch merge.

Weights: WEIGHTS_DIR/unet_3ch_disk — never unet_semantic / unet_resnet34 /
unet_patch_dhebe. Not yolo_seg_phys. Never submits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gate_unet_family import PATCH_GATE, refuse_3ch

FAMILY = "unet_3ch_disk_clahe_sobel"
WEIGHTS_NAME = "unet_3ch_disk"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-json", default=str(PATCH_GATE))
    parser.add_argument("--dry-refuse", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--base", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reason = refuse_3ch(Path(args.gate_json))
    if args.dry_refuse:
        if reason:
            print(reason)
        raise SystemExit(0)
    if reason:
        raise SystemExit(reason)

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from filament.coco_io import grouped_folds, load_coco
    from filament.paths import ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
    from filament.refiner import tversky_loss
    from filament.semantic_dataset import FullDiskSemanticDataset
    from filament.unet import build_fulldisk_unet
    from train_unet import pick_device

    out_dir = WEIGHTS_DIR / WEIGHTS_NAME
    if out_dir.name != WEIGHTS_NAME or "unet_semantic" in out_dir.parts:
        raise SystemExit("Refusing: weights dir must be unet_3ch_disk, never unet_semantic.")

    device = pick_device(args.device)
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    train_files = sorted(fn for fn, fold in fold_of.items() if fold != 0)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    train_ds = FullDiskSemanticDataset(
        views, by_file, train_files, TRAIN_IMG_DIR, imgsz=args.imgsz, augment=True, stack="disk_clahe_sobel"
    )
    val_ds = FullDiskSemanticDataset(
        views, by_file, val_files, TRAIN_IMG_DIR, imgsz=args.imgsz, augment=False, stack="disk_clahe_sobel"
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_fulldisk_unet(in_ch=3, base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss()
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    history = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y, _ in tqdm(train_loader, desc=f"3ch {epoch}/{args.epochs}"):
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
            "family": FAMILY,
            "best_val_loss": best_val,
            "best_epoch": int(min(history, key=lambda r: r["val_loss"])["epoch"]),
            "history": history,
            "val_loss_is_not_official_pq": True,
            "overwrote_unet_semantic": False,
            "not_yolo_phys": True,
        }
        (out_dir / "train_curve.json").write_text(json.dumps(curve, indent=2))
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "unet_3ch_disk_train_curve.json").write_text(json.dumps(curve, indent=2))

    print(json.dumps({"best": str(best_path), "device": device, "do_not_submit": True}, indent=2))


if __name__ == "__main__":
    main()
