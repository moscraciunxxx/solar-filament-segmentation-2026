#!/usr/bin/env python3
"""Claim each 2048 mask's image basin and score fold 0.

Also records how close the false negatives already are to the parent,
without using that histogram to choose a cutoff. Pad is 48. No sweep.
Does not read test labels. Does not submit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.disjoint_add import mask_iou
from filament.marker_watershed import PAD, refine_rles
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR
from filament.rle import decode_rle

PARENT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
OUT = ARTIFACTS / "stream_marker_fold0.json"
EXPECTED_PQ = 0.4387235694048039
PUBLIC_GAP = 0.069


def _row(br, n_pred: int) -> dict:
    return {
        "pq": br.pq,
        "tp": br.tp,
        "fp": br.fp,
        "fn": br.fn,
        "mean_tp_iou": br.mean_tp_iou,
        "n_pred": n_pred,
        "n_annotator_images": br.n_annotator_images,
    }


def _by_stem(frame: pd.DataFrame) -> dict[str, list[str]]:
    stems = frame["filament_id"].str.split("_", n=1).str[0]
    grouped: dict[str, list[str]] = {}
    for stem, rle in zip(stems, frame["segmentation_rle"], strict=True):
        grouped.setdefault(stem, []).append(rle)
    return grouped


def _fn_iou_hist(gt_df: pd.DataFrame, parent: pd.DataFrame) -> dict:
    gt_keys = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_by = _by_stem(parent)
    bins = {"zero": 0, "under_0.1": 0, "from_0.1_to_0.3": 0, "from_0.3_to_0.5": 0, "over_0.5": 0}
    for key in gt_keys.unique():
        image_id = key.split("-", maxsplit=1)[1]
        gt_rles = gt_df.loc[gt_keys == key, "segmentation_rle"].tolist()
        pred_rles = pred_by.get(image_id, [])
        pred_masks = [decode_rle(rle) for rle in pred_rles]
        for rle in gt_rles:
            gt_mask = decode_rle(rle)
            best = max((mask_iou(gt_mask, pred) for pred in pred_masks), default=0.0)
            if best == 0.0:
                bins["zero"] += 1
            elif best < 0.1:
                bins["under_0.1"] += 1
            elif best < 0.3:
                bins["from_0.1_to_0.3"] += 1
            elif best <= 0.5:
                bins["from_0.3_to_0.5"] += 1
            else:
                bins["over_0.5"] += 1
    return bins


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(name for name, fold in fold_of.items() if fold == 0)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    parent = pd.read_csv(PARENT)
    parent_row = _row(score_pq(gt_df, parent), len(parent))
    if abs(parent_row["pq"] - EXPECTED_PQ) > 1e-6:
        raise SystemExit(f"parent rescore drifted: {parent_row}")
    parent_by = _by_stem(parent)
    frames = []
    changed = 0
    for name in val_files:
        stem = Path(name).stem
        image = cv2.imread(str(TRAIN_IMG_DIR / name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise SystemExit(f"missing {name}")
        rles = parent_by.get(stem, [])
        refined = refine_rles(image, rles)
        changed += sum(a != b for a, b in zip(rles, refined, strict=True))
        if refined:
            frames.append(pred_to_df(stem, refined))
    pred = pd.concat(frames, ignore_index=True)
    scored = _row(score_pq(gt_df, pred), len(pred))
    forecast = scored["pq"] - PUBLIC_GAP
    payload = {
        "split": "grouped year x count fold 0",
        "seed": SEED,
        "hypothesis": "marker watershed: each 2048 mask claims its basin inside a 48-pixel pad",
        "pad": PAD,
        "masks_whose_rle_changed": changed,
        "parent_gt_max_iou_counts": _fn_iou_hist(gt_df, parent),
        "parent": parent_row,
        "marker": scored,
        "lift_vs_2048": scored["pq"] - parent_row["pq"],
        "public_gap_used": PUBLIC_GAP,
        "public_forecast": forecast,
        "promote": bool(forecast > 0.6),
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
