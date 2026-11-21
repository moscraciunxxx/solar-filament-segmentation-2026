#!/usr/bin/env python3
"""Train ConvNeXt-T U-Net on the CLV flat field and the soft annotator mean.

Folds 1-4 only. Fold 0 is not in the loss and does not pick the checkpoint.
After epoch 3, threshold 0.5 + connected components is scored on fold 0.
If that PQ is under 0.35 the run stops.

--resume-epoch3 continues epochs 4-8 from the saved epoch-3 weights. It runs
only when the min-area rescore set resume_epochs_4_to_8. Adam moments restart
because they were not saved. Never submits.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_soft_clv import CC_KILL, FAMILY, score_threshold_cc
from filament.coco_io import grouped_folds, load_coco
from filament.paths import ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.soft_clv_dataset import SoftClvDataset
from filament.unet_convnext import ConvNeXtUNet, soft_bce_dice

DEST = WEIGHTS_DIR / FAMILY


def _busy() -> str | None:
    import subprocess

    mine = {os.getpid(), os.getppid()}
    try:
        text = subprocess.check_output(["pgrep", "-af", "train_yolo|train_unet|train_condinst|train_soft_clv"], text=True)
    except subprocess.CalledProcessError:
        return None
    for line in text.splitlines():
        if "pgrep" in line:
            continue
        pid = int(line.split(maxsplit=1)[0])
        if pid not in mine:
            return line.strip()
    return None


def main() -> None:
    epochs = 8
    imgsz = 1536
    if _busy():
        raise SystemExit(f"MPS busy: {_busy()}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device != "mps":
        raise SystemExit("MPS is required")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    train_files = sorted(name for name, fold in fold_of.items() if fold != 0)
    val_files = sorted(name for name, fold in fold_of.items() if fold == 0)
    if len(val_files) != 142 or any(fold_of[name] == 0 for name in train_files):
        raise SystemExit("fold split is wrong")
    train_ds = SoftClvDataset(views, by_file, train_files, TRAIN_IMG_DIR, imgsz, augment=True)
    loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    resume = "--resume-epoch3" in sys.argv
    DEST.mkdir(parents=True, exist_ok=True)
    if resume:
        decision = json.loads((ARTIFACTS / "soft_clv_epoch3_minarea.json").read_text())
        if not decision.get("resume_epochs_4_to_8") or decision.get("fold0_used_to_pick_min_area"):
            raise SystemExit("min-area gate did not clear; the epoch-3 kill stands")
        curve = json.loads((DEST / "train_curve.json").read_text())
        if len(curve.get("history", [])) != 3:
            raise SystemExit("resume expects the three-epoch checkpoint")
        snapshot = DEST / "epoch3.pt"
        if not snapshot.is_file():
            shutil.copy2(DEST / "best.pt", snapshot)
        model = ConvNeXtUNet(pretrained=False).to(device)
        model.load_state_dict(torch.load(DEST / "best.pt", map_location="cpu", weights_only=False))
        best = float(curve["best_train_loss"])
        best_epoch = 3
        history = list(curve["history"])
        start_epoch = 4
    elif (DEST / "best.pt").is_file():
        raise SystemExit("epoch-3 checkpoint exists; refusing a from-scratch retrain")
    else:
        model = ConvNeXtUNet(pretrained=True).to(device)
        best = float("inf")
        best_epoch = 0
        history = []
        start_epoch = 1
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for image, target, _name in loader:
            image = image.to(device)
            target = target.to(device)
            loss = soft_bce_dice(model(image), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            seen += 1
        train_loss = total / max(seen, 1)
        row = {"epoch": epoch, "train_loss": train_loss, "n": seen}
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save(model.state_dict(), DEST / "last.pt")
        if train_loss < best:
            best = train_loss
            best_epoch = epoch
            torch.save(model.state_dict(), DEST / "best.pt")
        curve = {
            "family": FAMILY,
            "imgsz": imgsz,
            "target": "soft mean of all annotators",
            "input": "center-to-limb flat field repeated to 3 channels",
            "best_train_loss": best,
            "best_epoch": best_epoch,
            "history": history,
            "fold0_in_loss": False,
            "optimizer_state_resumed": False,
        }
        (DEST / "train_curve.json").write_text(json.dumps(curve, indent=2) + "\n")
        if epoch == 3 and not resume:
            model.eval()
            report = score_threshold_cc(model, val_files, device)
            report["kill_if_under"] = CC_KILL
            report["continue"] = report["pq"] >= CC_KILL
            (ARTIFACTS / "soft_clv_epoch3_cc.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"epoch3_cc": report}), flush=True)
            if report["pq"] < CC_KILL:
                print(json.dumps({"killed": True, "reason": "epoch3 threshold+CC under 0.35"}), flush=True)
                return
    print(json.dumps({"best": str(DEST / "best.pt"), "do_not_submit": True}), flush=True)


if __name__ == "__main__":
    main()
