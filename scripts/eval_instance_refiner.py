#!/usr/bin/env python3
"""Second-stage refiner for kept instances. Does not start without --start.

Crops are taken at full resolution around each instance, then resized to 256.
The target is the raw annotator soft majority, not the eroded medoid.
Train and threshold selection use folds 1-4. Fold 0 is scored once.
Never submits. Never reads test labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco, medoid_view_id, rasterize_polygons
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, IMAGE_SIZE, SEED, TRAIN_IMG_DIR
from filament.rle import decode_rle, encode_rle
from filament.coco_io import gt_records_for_files

NET = 256
MARGIN = 32
EPOCHS = 6
THRESHOLDS = (0.35, 0.50, 0.65)
OUT = ARTIFACTS / "instance_refiner_fold0.json"
WEIGHTS = FAST_ARTIFACTS / "instance_refiner.pt"
CHAMPION_DIR = FAST_ARTIFACTS / "tile_addon" / "champion"


class Refiner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec = nn.Sequential(
            nn.Conv2d(48, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        return self.dec(torch.cat([self.up(b), a], dim=1))


def _box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    y0 = max(int(ys.min()) - MARGIN, 0)
    y1 = min(int(ys.max()) + MARGIN + 1, IMAGE_SIZE)
    x0 = max(int(xs.min()) - MARGIN, 0)
    x1 = min(int(xs.max()) + MARGIN + 1, IMAGE_SIZE)
    return y0, y1, x0, x1


def _medoid_mask(views, image_ids: list[str]) -> np.ndarray:
    view = views[medoid_view_id(views, image_ids)]
    return rasterize_polygons([f.polygon for f in view.filaments])


def _soft_majority(views, image_ids: list[str]) -> np.ndarray:
    acc = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    n = 0
    for image_id in image_ids:
        acc += rasterize_polygons([f.polygon for f in views[image_id].filaments])
        n += 1
    if n == 0:
        return acc
    return (acc.astype(np.int16) * 2 >= n).astype(np.uint8)


def _instances_for(file_name: str, source: str) -> list[np.ndarray]:
    stem = Path(file_name).stem
    if source == "champion":
        rles = json.loads((CHAMPION_DIR / f"{stem}.json").read_text())
        return [decode_rle(rle) for rle in rles]
    pred = pd.read_csv(FAST_ARTIFACTS / "tile_addon_fold0_pred.csv")
    # Tune images are not in the fold-0 csv. Tile extras live in assembled json.
    extra_path = FAST_ARTIFACTS / "tile_addon" / "assembled" / f"{stem}.json"
    champ = json.loads((CHAMPION_DIR / f"{stem}.json").read_text())
    if not extra_path.exists():
        return [decode_rle(rle) for rle in champ]
    frozen = json.loads((ARTIFACTS / "tile_addon_tune_folds14.json").read_text())["frozen"]
    key = f"{frozen['kind']}_t{frozen['tile']}_c{frozen['conf']}_v{frozen['votes']}_a{frozen['min_area']}"
    extra = json.loads(extra_path.read_text()).get(key, [])
    return [decode_rle(rle) for rle in champ + extra]


class CropSet(Dataset):
    def __init__(self, items: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        stacked, target = self.items[index]
        return torch.from_numpy(stacked), torch.from_numpy(target).unsqueeze(0)


def _crop_pair(gray: np.ndarray, inst: np.ndarray, target: np.ndarray):
    box = _box(inst)
    if box is None:
        return None
    y0, y1, x0, x1 = box
    g = cv2.resize(gray[y0:y1, x0:x1], (NET, NET), interpolation=cv2.INTER_AREA)
    m = cv2.resize(inst[y0:y1, x0:x1], (NET, NET), interpolation=cv2.INTER_NEAREST)
    t = cv2.resize(target[y0:y1, x0:x1], (NET, NET), interpolation=cv2.INTER_NEAREST)
    stacked = np.stack([g.astype(np.float32) / 255.0, m.astype(np.float32)], axis=0)
    return stacked, t.astype(np.float32), box


def _collect(files: list[str], views, by_file, source: str, target_name: str):
    items = []
    meta = []
    for file_name in files:
        gray = cv2.imread(str(TRAIN_IMG_DIR / file_name), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        if gray.shape[0] != IMAGE_SIZE:
            gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        image_ids = by_file[file_name]
        target = _medoid_mask(views, image_ids) if target_name == "medoid" else _soft_majority(views, image_ids)
        for inst in _instances_for(file_name, source):
            packed = _crop_pair(gray, inst, target)
            if packed is None:
                continue
            stacked, t, box = packed
            items.append((stacked, t))
            meta.append((file_name, inst, box))
    return items, meta


def _loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    dice = 1 - (2 * inter + 1) / (prob.sum() + target.sum() + 1)
    return bce + dice


def _refine_mask(model, device, gray: np.ndarray, inst: np.ndarray, threshold: float) -> np.ndarray:
    packed = _crop_pair(gray, inst, inst)
    if packed is None:
        return inst
    stacked, _t, (y0, y1, x0, x1) = packed
    x = torch.from_numpy(stacked).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    small = cv2.resize((prob >= threshold).astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    out[y0:y1, x0:x1] = small
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--source", choices=("champion", "tile"), default="champion")
    parser.add_argument("--target", choices=("soft", "medoid"), default="soft")
    args = parser.parse_args()
    out = ARTIFACTS / ("instance_refiner_medoid_fold0.json" if args.target == "medoid" else "instance_refiner_fold0.json")
    weights = FAST_ARTIFACTS / ("instance_refiner_medoid.pt" if args.target == "medoid" else "instance_refiner.pt")
    if not args.start:
        print(json.dumps({"dry": True, "source": args.source, "target": args.target, "do_not_submit": True}))
        return
    if out.exists():
        raise SystemExit(f"fold 0 already scored: {out}")
    if args.source == "tile" and not (ARTIFACTS / "tile_addon_fold0.json").exists():
        raise SystemExit("tile fold 0 is not scored yet")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    tune_files = sorted(fn for fn, fold in fold_of.items() if fold != 0)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    items, _meta = _collect(tune_files, views, by_file, args.source, args.target)
    if not items:
        raise SystemExit("no crops")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    loader = DataLoader(CropSet(items), batch_size=8, shuffle=True)
    model = Refiner().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = _loss(model(x), y)
            loss.backward()
            opt.step()
            total += float(loss.detach())
        print(f"epoch {epoch + 1} loss {total / max(len(loader), 1):.4f}", flush=True)
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights)
    model.eval()

    def predict(files: list[str], threshold: float) -> pd.DataFrame:
        frames = []
        for file_name in files:
            gray = cv2.imread(str(TRAIN_IMG_DIR / file_name), cv2.IMREAD_GRAYSCALE)
            rles = []
            for inst in _instances_for(file_name, args.source):
                rles.append(encode_rle(_refine_mask(model, device, gray, inst, threshold)))
            frames.append(pred_to_df(Path(file_name).stem, rles))
        return pd.concat(frames, ignore_index=True)

    gt_tune = gt_annotations_to_df(gt_records_for_files(views, by_file, tune_files))
    tune_rows = []
    best = None
    for threshold in THRESHOLDS:
        pred = predict(tune_files, threshold)
        scored = score_pq(gt_tune, pred)
        row = {"threshold": threshold, "pq": scored.pq, "tp": scored.tp, "fp": scored.fp, "fn": scored.fn, "n_pred": int(len(pred))}
        tune_rows.append(row)
        print("tune", json.dumps(row), flush=True)
        if best is None or row["pq"] > best["pq"]:
            best = row
    gt_val = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    pred = predict(val_files, best["threshold"])
    pred_name = "instance_refiner_medoid_fold0_pred.csv" if args.target == "medoid" else "instance_refiner_fold0_pred.csv"
    pred_csv = FAST_ARTIFACTS / pred_name
    pred.to_csv(pred_csv, index=False)
    scored = score_pq(gt_val, pred)
    payload = {
        "family": "instance_refiner_medoid" if args.target == "medoid" else "instance_refiner",
        "source": args.source,
        "target": "raw medoid outline" if args.target == "medoid" else "raw annotator soft majority, not eroded medoid",
        "tune_grid": tune_rows,
        "frozen_threshold": best["threshold"],
        "fold": 0,
        "pq": scored.pq,
        "tp": scored.tp,
        "fp": scored.fp,
        "fn": scored.fn,
        "n_pred": int(len(pred)),
        "mean_tp_iou": scored.mean_tp_iou,
        "champion_pq": 0.4741684688395591,
        "delta_pq": scored.pq - 0.4741684688395591,
        "kept": bool(scored.pq > 0.4741684688395591 + 0.01),
        "pred_csv": str(pred_csv),
        "test_labels_used": False,
        "kaggle_submit_called": False,
        "do_not_submit": True,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("DONE refiner", json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
