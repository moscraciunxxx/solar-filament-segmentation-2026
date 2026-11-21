#!/usr/bin/env python3
"""Add coherent dark lanes whose size and depth match training-fold misses.

Cuts come from medoid filaments the m@1280 model missed on folds other than
0. Fold 0 is only scored. No sweep. Does not read test labels. Does not submit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.coherent_lane import WINDOW, propose_coherent_rles
from filament.medoid_gt import medoid_gt_records
from filament.metric import gt_annotations_to_df, overlap_matrices, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR, YOLO_DIR
from filament.preprocess import read_gray
from filament.rle import decode_rle, rles_to_layers

PARENT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
TRAIN_PRED = FAST_ARTIFACTS / "yolo_yolo11m_seg_1280_foldne0_conf0.35_sz1280_m0_base.csv"
OUT = ARTIFACTS / "stream_coherent_lane_fold0.json"
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


def _image_path(file_name: str) -> Path:
    cached = YOLO_DIR / "images" / "train" / file_name
    if cached.exists():
        return cached
    return TRAIN_IMG_DIR / file_name


def _best(gt_rles: list[str], pred_rles: list[str]) -> list[float]:
    if not pred_rles:
        return [0.0] * len(gt_rles)
    iou, _ = overlap_matrices(
        torch.from_numpy(rles_to_layers(gt_rles)),
        torch.from_numpy(rles_to_layers(pred_rles)),
    )
    return iou.max(dim=1).values.tolist()


def train_gates(views, by_file, fold_of, pred_by) -> dict:
    window = WINDOW if WINDOW % 2 else WINDOW + 1
    depths: list[float] = []
    areas: list[int] = []
    for name, fold in sorted(fold_of.items()):
        if fold == 0:
            continue
        gray = read_gray(_image_path(name))
        blurred = cv2.blur(gray, (window, window))
        dip = blurred.astype(np.float32) - gray.astype(np.float32)
        records = medoid_gt_records(views, by_file[name])
        rles = [rec["rle"] for rec in records]
        stem = Path(name).stem
        best = _best(rles, pred_by.get(stem, []))
        for rle, value in zip(rles, best, strict=True):
            if value > 0.5:
                continue
            mask = decode_rle(rle)
            area = int(mask.sum())
            if area < 10:
                continue
            areas.append(area)
            depths.append(float(np.median(dip[mask > 0])))
    if len(depths) < 30:
        raise SystemExit(f"too few training misses to set cuts: {len(depths)}")
    return {
        "n_train_misses": len(depths),
        "response_min": float(np.quantile(depths, 0.25)),
        "min_area": int(np.quantile(areas, 0.10)),
        "max_area": int(np.quantile(areas, 0.90)),
    }


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    gates = train_gates(views, by_file, fold_of, _by_stem(pd.read_csv(TRAIN_PRED)))
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
    frames = []
    added = 0
    for name in val_files:
        stem = Path(name).stem
        image = read_gray(TRAIN_IMG_DIR / name)
        extra = propose_coherent_rles(
            image,
            parent_by.get(stem, []),
            response_min=gates["response_min"],
            min_area=gates["min_area"],
            max_area=gates["max_area"],
        )
        added += len(extra)
        kept = parent_by.get(stem, []) + extra
        if kept:
            frames.append(pred_to_df(stem, kept))
    pred = pd.concat(frames, ignore_index=True)
    scored = _row(score_pq(gt_df, pred), len(pred))
    forecast = scored["pq"] - PUBLIC_GAP
    payload = {
        "split": "grouped year x count fold 0",
        "seed": SEED,
        "hypothesis": "add a 96-pixel coherent dark lane whose depth and area match training-fold misses and that sits 64 px off the 2048 parent",
        "gates": gates,
        "window": WINDOW,
        "masks_added": added,
        "parent": first,
        "parent_second_call": second,
        "lanes": scored,
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
