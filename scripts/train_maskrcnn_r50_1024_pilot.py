#!/usr/bin/env python3
"""Fold-safe Mask R-CNN pilot. Medoid masks, train images outside fold 0.

Scores official fold-0 PQ at the end. Never submits. Never reads test labels.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco, medoid_view_id
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import (
    ARTIFACTS,
    FAST_ARTIFACTS,
    IMAGE_SIZE,
    SEED,
    TRAIN_IMG_DIR,
    WEIGHTS_DIR,
)
from filament.rle import encode_rle

IMGSZ = 1024
EPOCHS = 10
BATCH = 1
LR = 0.0005
SCORE_THRESH = 0.35
MIN_AREA = 96
NMS_IOU = 0.5
DEST = WEIGHTS_DIR / "maskrcnn_r50_1024"
FAMILY = "maskrcnn_r50_1024"


def _set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


class MedoidFilaments(Dataset):
    def __init__(self, files: list[str], views, by_file) -> None:
        self.files = files
        self.views = views
        self.by_file = by_file
        self.scale = IMGSZ / float(IMAGE_SIZE)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        file_name = self.files[index]
        bgr = cv2.imread(str(TRAIN_IMG_DIR / file_name), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(file_name)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (IMGSZ, IMGSZ), interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(small).permute(2, 0, 1).float() / 255.0
        mid = medoid_view_id(self.views, self.by_file[file_name])
        masks = []
        boxes = []
        for filament in self.views[mid].filaments:
            poly = np.round(filament.polygon * self.scale).astype(np.int32)
            if len(poly) < 3:
                continue
            mask = np.zeros((IMGSZ, IMGSZ), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 1)
            if int(mask.sum()) < 16:
                continue
            ys, xs = np.nonzero(mask)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            if x1 <= x0 or y1 <= y0:
                continue
            masks.append(mask)
            boxes.append([x0, y0, x1, y1])
        if masks:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.ones(len(boxes), dtype=torch.int64),
                "masks": torch.tensor(np.stack(masks), dtype=torch.uint8),
            }
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, IMGSZ, IMGSZ), dtype=torch.uint8),
            }
        target["image_id"] = torch.tensor([index])
        return image, target


def _collate(batch):
    return tuple(zip(*batch))


def _build_model() -> torch.nn.Module:
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, 2)
    return model


def _nms_keep(masks: list[np.ndarray], scores: list[float]) -> list[int]:
    order = sorted(range(len(masks)), key=lambda i: scores[i], reverse=True)
    kept: list[int] = []
    for i in order:
        ok = True
        for j in kept:
            inter = np.logical_and(masks[i], masks[j]).sum()
            union = np.logical_or(masks[i], masks[j]).sum()
            if union and inter / union > NMS_IOU:
                ok = False
                break
        if ok:
            kept.append(i)
    return kept


def _predict_rles(model, device, file_name: str) -> list[str]:
    bgr = cv2.imread(str(TRAIN_IMG_DIR / file_name), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (IMGSZ, IMGSZ), interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(small).permute(2, 0, 1).float() / 255.0
    model.eval()
    with torch.no_grad():
        out = model([image.to(device)])[0]
    scores = out["scores"].detach().cpu().numpy()
    raw_masks = out["masks"].detach().cpu().numpy()
    keep = [i for i, score in enumerate(scores) if score >= SCORE_THRESH]
    bin_masks = []
    kept_scores = []
    for i in keep:
        small_mask = raw_masks[i, 0] > 0.5
        full = cv2.resize(small_mask.astype(np.uint8), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        if int(full.sum()) < MIN_AREA:
            continue
        bin_masks.append(full.astype(bool))
        kept_scores.append(float(scores[i]))
    chosen = _nms_keep(bin_masks, kept_scores)
    return [encode_rle(bin_masks[i].astype(np.uint8)) for i in chosen]


def _score(model, device, views, by_file, val_files: list[str]) -> dict:
    frames = []
    for file_name in val_files:
        rles = _predict_rles(model, device, file_name)
        frames.append(pred_to_df(Path(file_name).stem, rles))
        print(f"score {file_name} n={len(rles)}", flush=True)
    pred_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    pred_csv = FAST_ARTIFACTS / f"{FAMILY}_fold0_conf{SCORE_THRESH:.2f}_pred.csv"
    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_csv, index=False)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    scored = score_pq(gt_df, pred_df)
    payload = {
        "family": FAMILY,
        "imgsz": IMGSZ,
        "score_thresh": SCORE_THRESH,
        "min_area": MIN_AREA,
        "nms_iou": NMS_IOU,
        "fold": 0,
        "n_val": len(val_files),
        "pq": scored.pq,
        "tp": scored.tp,
        "fp": scored.fp,
        "fn": scored.fn,
        "mean_tp_iou": scored.mean_tp_iou,
        "n_pred": int(len(pred_df)),
        "pred_csv": str(pred_csv),
        "test_labels_used": False,
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "slot1_stays": 56173362,
        "slot2_stays": 56148889,
    }
    text = json.dumps(payload, indent=2) + "\n"
    (ARTIFACTS / f"{FAMILY}_fold0_official.json").write_text(text)
    (FAST_ARTIFACTS / f"{FAMILY}_fold0_official.json").write_text(text)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    _set_seed()
    if (DEST / "last.pt").exists():
        raise SystemExit(f"refuse to overwrite {DEST / 'last.pt'}")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    train_files = sorted(fn for fn, fold in fold_of.items() if fold != 0)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    device = torch.device("mps")
    model = _build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    loader = DataLoader(
        MedoidFilaments(train_files, views, by_file),
        batch_size=BATCH,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
    )
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"train_images {len(train_files)} val_images {len(val_files)} lr {LR}", flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        skipped_empty = 0
        for step, (images, targets) in enumerate(loader, start=1):
            if any(t["boxes"].numel() == 0 for t in targets):
                skipped_empty += 1
                continue
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at epoch {epoch} step {step}")
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            losses.append(float(loss.detach().cpu()))
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {losses[-1]:.4f}", flush=True)
        mean_loss = float(np.mean(losses)) if losses else 0.0
        print(f"epoch {epoch} skipped_empty {skipped_empty}", flush=True)
        torch.save({"epoch": epoch, "model": model.state_dict(), "mean_loss": mean_loss}, DEST / "last.pt")
        print(f"epoch {epoch}/{EPOCHS} loss {mean_loss:.4f}", flush=True)
    _score(model, device, views, by_file, val_files)


if __name__ == "__main__":
    main()
