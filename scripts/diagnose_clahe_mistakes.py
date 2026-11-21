#!/usr/bin/env python3
"""Fold-0 mistake profile for yolo11s_seg_clahe vs GT and vs Slot1 erode.

CPU. Existing pred CSVs only. No train, no test labels, no submit.
Matching: IoU>0.5, not Hungarian, every annotator-image.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco, rasterize_polygons
from filament.metric import overlap_matrices
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED
from filament.rle import encode_rle, rles_to_layers

CLAHE = FAST_ARTIFACTS / "yolo11s_seg_clahe_fold0_conf0.35_pred.csv"
SLOT1 = FAST_ARTIFACTS / "slot1_erode_3x1_fold0_pred.csv"
AREA_P25 = 642.0
UNIDENT = 3
IOU_THR = 0.5
OUT = ARTIFACTS / "clahe_mistakes_wave44.json"


def _by_stem(path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(path)
    out: dict[str, list[str]] = defaultdict(list)
    for fid, rle in zip(df["filament_id"], df["segmentation_rle"]):
        out[str(fid).rsplit("_", 1)[0]].append(str(rle))
    return out


def _layers(rles: list[str]) -> torch.Tensor:
    if not rles:
        return torch.zeros((0, 2048, 2048), dtype=torch.float32)
    return torch.from_numpy(rles_to_layers(rles))


def main() -> None:
    clahe = _by_stem(CLAHE)
    slot1 = _by_stem(SLOT1)
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)

    n_gt = 0
    bins = {"hit": 0, "near_0.30_0.50": 0, "weak_0.10_0.30": 0, "hard_miss": 0}
    faint_fn = 0
    large_fn = 0
    faint_tp = 0
    large_tp = 0
    slot1_hits_clahe_fn = 0
    both_fn = 0
    clahe_only_tp = 0
    max_iou_near: list[float] = []

    for file_name in val_files:
        stem = Path(file_name).stem
        c_layers = _layers(clahe.get(stem, []))
        s_layers = _layers(slot1.get(stem, []))
        for image_id in by_file[file_name]:
            view = views[image_id]
            if not view.filaments:
                continue
            gt_rles = []
            faint = []
            for fil in view.filaments:
                gt_rles.append(encode_rle(rasterize_polygons([fil.polygon])))
                faint.append(float(fil.area) <= AREA_P25 or int(fil.category_id) == UNIDENT)
            n_gt += len(gt_rles)
            gt = _layers(gt_rles)
            if c_layers.shape[0] == 0:
                mx = torch.zeros(gt.shape[0])
                hit = torch.zeros(gt.shape[0], dtype=torch.bool)
            else:
                iou, _ = overlap_matrices(gt, c_layers)
                mx = iou.max(dim=1).values
                hit = mx > IOU_THR
            if s_layers.shape[0] == 0:
                s_hit = torch.zeros(gt.shape[0], dtype=torch.bool)
            else:
                siou, _ = overlap_matrices(gt, s_layers)
                s_hit = siou.max(dim=1).values > IOU_THR
            for i, is_faint in enumerate(faint):
                v = float(mx[i])
                if v > 0.5:
                    bins["hit"] += 1
                    if is_faint:
                        faint_tp += 1
                    else:
                        large_tp += 1
                    if not bool(s_hit[i]):
                        clahe_only_tp += 1
                else:
                    if 0.30 <= v <= 0.5:
                        bins["near_0.30_0.50"] += 1
                        max_iou_near.append(v)
                    elif v >= 0.10:
                        bins["weak_0.10_0.30"] += 1
                    else:
                        bins["hard_miss"] += 1
                    if is_faint:
                        faint_fn += 1
                    else:
                        large_fn += 1
                    if bool(s_hit[i]):
                        slot1_hits_clahe_fn += 1
                    else:
                        both_fn += 1

    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": "clahe_fold0_mistake_profile",
        "skill": "hidden-exam-discipline",
        "family": "yolo11s_seg_clahe",
        "official_pq": 0.35711071281140794,
        "official": {"tp": 710, "fp": 311, "fn": 879, "n_pred": 624},
        "slot1": {"pq": 0.4365, "tp": 949, "fp": 385, "fn": 640},
        "n_gt": n_gt,
        "iou_bins_per_gt": bins,
        "fn_faint": faint_fn,
        "fn_large": large_fn,
        "tp_faint": faint_tp,
        "tp_large": large_tp,
        "clahe_fn_that_slot1_hits": slot1_hits_clahe_fn,
        "fn_both": both_fn,
        "clahe_tp_slot1_misses": clahe_only_tp,
        "near_miss_mean_iou": (sum(max_iou_near) / len(max_iou_near)) if max_iou_near else 0.0,
        "morph_can_fix_hard_miss": False,
        "did_not_submit": True,
        "kaggle_submit_called": False,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
