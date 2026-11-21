#!/usr/bin/env python3
"""Drop a faint 2048 mask only when yolo11n does not overlap it.

The contrast floor is the 10th percentile of medoid filaments on folds
other than 0. The witness cut is 0.5. No sweep. Does not read test labels.
Does not submit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.contrast_floor import mask_contrast
from filament.faint_unseen import WITNESS_IOU, drop_faint_unseen
from filament.medoid_gt import medoid_gt_records
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR, YOLO_DIR
from filament.preprocess import read_gray
from filament.rle import decode_rle

PARENT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
WITNESS = FAST_ARTIFACTS / "yolo11n_seg_erode_gt_fold0_conf0.35_pred.csv"
OUT = ARTIFACTS / "stream_faint_unseen_fold0.json"
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


def _image_path(file_name: str, fold: int) -> Path:
    split = "val" if fold == 0 else "train"
    cached = YOLO_DIR / "images" / split / file_name
    if cached.exists():
        return cached
    return TRAIN_IMG_DIR / file_name


def contrast_floor(views, by_file, fold_of) -> tuple[float, int]:
    values: list[float] = []
    for name, fold in sorted(fold_of.items()):
        if fold == 0:
            continue
        image = read_gray(_image_path(name, fold))
        for rec in medoid_gt_records(views, by_file[name]):
            contrast = mask_contrast(decode_rle(rec["rle"]), image)
            if contrast is not None:
                values.append(contrast)
    if len(values) < 30:
        raise SystemExit(f"too few training filaments to set the floor: {len(values)}")
    return float(np.quantile(values, 0.10)), len(values)


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    floor, n_train = contrast_floor(views, by_file, fold_of)
    val_files = sorted(name for name, fold in fold_of.items() if fold == 0)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    parent = pd.read_csv(PARENT)
    first = _row(score_pq(gt_df, parent), len(parent))
    second = _row(score_pq(gt_df, parent), len(parent))
    if first != second:
        raise SystemExit(f"parent scored twice and differed: {first} vs {second}")
    if abs(first["pq"] - EXPECTED_PQ) > 1e-6:
        raise SystemExit(f"parent rescore drifted: {first}")
    parent_by = _by_stem(parent)
    witness_by = _by_stem(pd.read_csv(WITNESS))
    frames = []
    dropped = 0
    for name in val_files:
        stem = Path(name).stem
        image = read_gray(_image_path(name, 0))
        kept, n_dropped = drop_faint_unseen(
            parent_by.get(stem, []),
            image,
            witness_by.get(stem, []),
            floor,
            WITNESS_IOU,
        )
        dropped += n_dropped
        if kept:
            frames.append(pred_to_df(stem, kept))
    pred = pd.concat(frames, ignore_index=True) if frames else parent.iloc[0:0]
    scored = _row(score_pq(gt_df, pred), len(pred))
    forecast = scored["pq"] - PUBLIC_GAP
    payload = {
        "split": "grouped year x count fold 0",
        "seed": SEED,
        "hypothesis": "drop a 2048 mask below the training contrast floor only when yolo11n does not overlap it above 0.5",
        "contrast_floor": floor,
        "witness_iou": WITNESS_IOU,
        "n_train_filaments": n_train,
        "masks_dropped": dropped,
        "parent": first,
        "parent_second_call": second,
        "faint_unseen": scored,
        "lift_vs_2048": scored["pq"] - first["pq"],
        "fp_delta": scored["fp"] - first["fp"],
        "fn_delta": scored["fn"] - first["fn"],
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
