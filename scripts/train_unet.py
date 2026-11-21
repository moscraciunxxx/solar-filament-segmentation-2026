#!/usr/bin/env python3
"""Train full-disk gray TinyUNet. Do not start while a YOLO job owns MPS.

v1 input is gray-as-RGB. Official selection is scripts/eval_unet.py instance PQ,
not this script's train loss. Never reads test labels. Never submits.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco
from filament.paths import ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.refiner import tversky_loss
from filament.semantic_dataset import FullDiskSemanticDataset
from filament.unet import build_fulldisk_unet


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def yolo_train_alive() -> bool:
    proc = subprocess.run(["pgrep", "-f", "train_yolo.py"], capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        return False
    pids = [int(x) for x in proc.stdout.split() if x.strip().isdigit() and int(x) != os.getpid()]
    return bool(pids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument(
        "--allow-alongside-yolo",
        action="store_true",
        help="Unsafe. Default refuses to start while train_yolo.py is alive.",
    )
    args = parser.parse_args()

    if yolo_train_alive() and not args.allow_alongside_yolo:
        raise SystemExit("Refusing to start U-Net while train_yolo.py owns MPS. Wait for faint merge PQ.")

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

    model = build_fulldisk_unet(base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss()

    out_dir = WEIGHTS_DIR / "unet_semantic"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    history = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y, _ in tqdm(train_loader, desc=f"unet {epoch}/{args.epochs}"):
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
            "family": "unet_fulldisk_s1024",
            "best_val_loss": best_val,
            "best_epoch": int(min(history, key=lambda r: r["val_loss"])["epoch"]),
            "history": history,
            "val_loss_is_not_official_pq": True,
        }
        (out_dir / "train_curve.json").write_text(json.dumps(curve, indent=2))
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "unet_train_curve.json").write_text(json.dumps(curve, indent=2))

    meta = {
        "family": "unet_fulldisk_s1024",
        "input": "gray-as-RGB",
        "device": device,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "best": str(best_path),
        "val_loss_is_not_official_pq": True,
        "official_eval": "scripts/eval_unet.py --max-val 0",
        "do_not_submit": True,
        "history": history,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "unet_train_meta.json").write_text(json.dumps({k: meta[k] for k in meta if k != "history"}, indent=2))
    print(json.dumps({k: meta[k] for k in ("best", "device", "n_train", "n_val")}, indent=2))


if __name__ == "__main__":
    main()
