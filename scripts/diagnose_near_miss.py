#!/usr/bin/env python3
"""Where official PQ dies: unmatched GT max-IoU histogram (near-miss vs true miss)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco, rasterize_polygons
from filament.infer import infer_yolo_image
from filament.metric import overlap_matrices
from filament.paths import ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.rle import rles_to_layers


def main() -> None:
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS_DIR / "yolo11s_seg" / "weights" / "best.pt"))
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)[:48]

    near = []
    miss = []
    hit = []
    pred_areas = []
    for file_name in tqdm(val_files, desc="near-miss"):
        rles = infer_yolo_image(model, TRAIN_IMG_DIR / file_name, conf=0.25, merge_gap=0)
        pred = torch.from_numpy(rles_to_layers(rles))
        pred_areas.extend([int(p.sum()) for p in pred])
        for image_id in by_file[file_name]:
            gt_masks = [rasterize_polygons([f.polygon]) for f in views[image_id].filaments]
            if not gt_masks:
                continue
            gt = torch.from_numpy(np.stack(gt_masks).astype(np.float32))
            if pred.shape[0] == 0:
                miss.extend([0.0] * gt.shape[0])
                continue
            iou, _ = overlap_matrices(gt, pred)
            mx = iou.max(dim=1).values.numpy()
            for v in mx:
                if v > 0.5:
                    hit.append(float(v))
                elif v >= 0.30:
                    near.append(float(v))
                else:
                    miss.append(float(v))

    out = {
        "n_images": len(val_files),
        "tp_like": len(hit),
        "near_miss_0.30_0.50": len(near),
        "hard_miss": len(miss),
        "mean_tp_iou": float(np.mean(hit)) if hit else 0.0,
        "mean_near": float(np.mean(near)) if near else 0.0,
        "pred_area_median": float(np.median(pred_areas)) if pred_areas else 0.0,
        "refiner_opportunity": len(near) / max(1, len(hit) + len(near) + len(miss)),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "near_miss.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
