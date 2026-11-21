#!/usr/bin/env python3
"""Mask R-CNN, ConvNeXt-Tiny ImageNet backbone, 1536 input, raw medoid labels.

Train is grouped folds 1-3. The checkpoint is the lowest loss on fold 4.
Fold 0 is not read for training or checkpoint selection. Does not submit.
Does not read test labels.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco, medoid_view_id, rasterize_polygons
from filament.paths import IMAGE_SIZE, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR

FAMILY = "maskrcnn_convnext_tiny_1536_raw_medoid"
IMGSZ = 1536


def family_for_holdout(holdout: int) -> str:
    if holdout == 4:
        return FAMILY
    return f"{FAMILY}_ho{holdout}"


def build_model():
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
    from torchvision.models.detection import MaskRCNN
    from torchvision.models.detection.backbone_utils import BackboneWithFPN

    body = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    backbone = BackboneWithFPN(
        body.features,
        return_layers={"1": "0", "3": "1", "5": "2", "7": "3"},
        in_channels_list=[96, 192, 384, 768],
        out_channels=256,
    )
    return MaskRCNN(backbone, num_classes=2, min_size=IMGSZ, max_size=IMGSZ)


class MedoidSet(Dataset):
    def __init__(self, files: list[str], views, by_file, augment: bool) -> None:
        self.files = files
        self.views = views
        self.by_file = by_file
        self.augment = augment
        self.medoid = {name: medoid_view_id(views, by_file[name]) for name in files}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        name = self.files[index]
        image = cv2.imread(str(TRAIN_IMG_DIR / name), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(name)
        if image.shape[0] != IMAGE_SIZE or image.shape[1] != IMAGE_SIZE:
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        view = self.views[self.medoid[name]]
        masks = []
        for filament in view.filaments:
            if len(filament.polygon) < 3:
                continue
            mask = rasterize_polygons([filament.polygon], size=IMAGE_SIZE)
            if int(mask.sum()) < 16:
                continue
            masks.append(mask)
        if masks:
            mask_t = torch.from_numpy(np.stack(masks))
        else:
            mask_t = torch.zeros((0, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
        tensor = torch.from_numpy(image[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        if self.augment and random.random() < 0.5:
            tensor = torch.flip(tensor, [2])
            mask_t = torch.flip(mask_t, [2])
        if self.augment and random.random() < 0.5:
            tensor = torch.flip(tensor, [1])
            mask_t = torch.flip(mask_t, [1])
        boxes = []
        kept = []
        for mask in mask_t:
            ys, xs = torch.nonzero(mask, as_tuple=True)
            if len(xs) == 0:
                continue
            boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
            kept.append(mask)
        if kept:
            mask_t = torch.stack(kept)
            box_t = torch.tensor(boxes, dtype=torch.float32)
            label_t = torch.ones((len(kept),), dtype=torch.int64)
        else:
            mask_t = torch.zeros((0, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
            box_t = torch.zeros((0, 4), dtype=torch.float32)
            label_t = torch.zeros((0,), dtype=torch.int64)
        target = {"boxes": box_t, "labels": label_t, "masks": mask_t}
        return tensor, target, name


def _loss(model, image, target, device) -> torch.Tensor:
    losses = model([image.to(device)], [{k: v.to(device) for k, v in target.items()}])
    return sum(losses.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--holdout-fold", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.holdout_fold not in (1, 2, 3, 4):
        raise SystemExit("holdout fold must be 1, 2, 3, or 4")
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    train_files = sorted(
        fn for fn, fold in fold_of.items() if fold in (1, 2, 3, 4) and fold != args.holdout_fold
    )
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == args.holdout_fold)
    if not train_files or not val_files:
        raise SystemExit("empty split")
    train_set = MedoidSet(train_files, views, by_file, augment=True)
    val_set = MedoidSet(val_files, views, by_file, augment=False)
    out = WEIGHTS_DIR / family_for_holdout(args.holdout_fold)
    if not args.smoke and (out / "best.pt").exists() and args.holdout_fold != 4:
        raise SystemExit(f"checkpoint already exists: {out / 'best.pt'}")
    out.mkdir(parents=True, exist_ok=True)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    if args.smoke:
        model.train()
        loss = _loss(model, *train_set[0][:2], device)
        loss.backward()
        print(f"smoke_ok loss {float(loss.detach()):.4f} device {device}", flush=True)
        return
    best = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_set)))
        random.shuffle(order)
        total = 0.0
        t0 = time.perf_counter()
        for step, index in enumerate(order, start=1):
            opt.zero_grad(set_to_none=True)
            loss = _loss(model, *train_set[index][:2], device)
            loss.backward()
            opt.step()
            total += float(loss)
            if step % 25 == 0:
                print(f"epoch {epoch} step {step}/{len(order)} loss {total / step:.4f}", flush=True)
        model.train()
        val_total = 0.0
        for index in range(len(val_set)):
            loss = _loss(model, *val_set[index][:2], device)
            val_total += float(loss.detach())
        val_loss = val_total / max(1, len(val_set))
        train_loss = total / max(1, len(order))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "seconds": time.perf_counter() - t0,
        }
        history.append(row)
        print("epoch_done", json.dumps(row), flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if best is None or val_loss < best:
            best = val_loss
            stale = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, out / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early_stop epoch {epoch}", flush=True)
                break
    meta = {
        "family": family_for_holdout(args.holdout_fold),
        "backbone": "convnext_tiny ImageNet-1K",
        "imgsz": IMGSZ,
        "labels": "raw medoid",
        "train_folds": sorted({1, 2, 3, 4} - {args.holdout_fold}),
        "checkpoint_fold": args.holdout_fold,
        "fold0_used_for_selection": False,
        "best_val_loss": best,
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("DONE train_maskrcnn_convnext", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
