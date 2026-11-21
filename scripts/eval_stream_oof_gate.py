#!/usr/bin/env python3
"""Fit a keep/drop rule for low-confidence proposals on folds other than 0.

The 2048 checkpoint at confidence 0.15 finds more true filaments and many
more false positives. This script proposes only masks that do not touch the
confidence-0.35 parent, labels those proposals from fold-train ground truth,
fits one logistic regression, and scores the frozen rule once on fold 0.
It does not read test labels and does not submit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.dark_delta import dark_enough
from filament.disjoint_add import disjoint_add, mask_iou
from filament.elongation import mask_eccentricity
from filament.infer import pick_device, run_yolo_infer
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.rle import decode_rle

WEIGHTS = WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048" / "weights" / "best.pt"
PARENT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
LOW_CACHE = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_conf015_by_fold.csv"
OUT = ARTIFACTS / "stream_oof_gate_fold0.json"
EXPECTED_PQ = 0.4387235694048039


def _row(br, n_pred: int) -> dict:
    return {
        "pq": br.pq,
        "tp": br.tp,
        "fp": br.fp,
        "fn": br.fn,
        "mean_tp_iou": br.mean_tp_iou,
        "n_pred": n_pred,
    }


def _by_stem(frame: pd.DataFrame) -> dict[str, list[str]]:
    if frame.empty:
        return {}
    stems = frame["filament_id"].str.split("_", n=1).str[0]
    grouped: dict[str, list[str]] = {}
    for stem, rle in zip(stems, frame["segmentation_rle"], strict=True):
        grouped.setdefault(stem, []).append(rle)
    return grouped


def _features(rle: str, image: np.ndarray) -> list[float]:
    mask = decode_rle(rle)
    area = float((mask > 0).sum())
    inside = image[mask > 0]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    dilated = cv2.dilate((mask > 0).astype(np.uint8), kernel)
    ring = image[(dilated > 0) & (mask == 0)]
    delta = float(ring.mean() - inside.mean()) if inside.size and ring.size else 0.0
    return [np.log1p(area), mask_eccentricity(mask), delta]


def _positive(rle: str, gt_rles: list[str]) -> int:
    if not gt_rles:
        return 0
    mask = decode_rle(rle)
    for gt in gt_rles:
        if mask_iou(mask, decode_rle(gt)) > 0.5:
            return 1
    return 0


def _infer(files: list[str], model, device: str) -> pd.DataFrame:
    frames = []
    for name in tqdm(files, desc="conf015-infer"):
        rles = run_yolo_infer(
            model,
            TRAIN_IMG_DIR / name,
            conf=0.15,
            imgsz=2048,
            merge_gap=0,
            device=device,
        )
        if rles:
            frames.append(pred_to_df(Path(name).stem, rles))
    if not frames:
        return pd.DataFrame(columns=["filament_id", "segmentation_rle", "fold"])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    files_by_fold: dict[int, list[str]] = {i: [] for i in range(5)}
    for name, fold in fold_of.items():
        files_by_fold[fold].append(name)
    for fold in files_by_fold:
        files_by_fold[fold].sort()

    from ultralytics import YOLO

    device = pick_device("mps")
    model = YOLO(str(WEIGHTS))
    train_files = [name for fold, names in files_by_fold.items() if fold != 0 for name in names]
    low_train = _infer(train_files, model, device)
    low_val = _infer(files_by_fold[0], model, device)

    parent = pd.read_csv(PARENT)
    parent_g = _by_stem(parent)
    low_val_g = _by_stem(low_val)
    low_train_g = _by_stem(low_train)

    xs: list[list[float]] = []
    ys: list[int] = []
    for name in train_files:
        stem = Path(name).stem
        image = cv2.imread(str(TRAIN_IMG_DIR / name), cv2.IMREAD_GRAYSCALE)
        gt = gt_records_for_files(views, by_file, [name])
        gt_rles = [rec["rle"] for rec in gt]
        merged, _ = disjoint_add(parent_g.get(stem, []), low_train_g.get(stem, []))
        # Train images have no saved 0.35 parent. Use an empty parent so the
        # candidate pool is the low-confidence set. The fold-0 rule still
        # requires disjointness from the real 0.35 parent.
        for rle in low_train_g.get(stem, []):
            if image is None:
                continue
            xs.append(_features(rle, image))
            ys.append(_positive(rle, gt_rles))
        del merged

    if len(set(ys)) < 2:
        raise SystemExit(f"need both classes to fit, counts={ys.count(1)}/{len(ys)}")
    clf = LogisticRegression(max_iter=200)
    clf.fit(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.int32))

    kept: dict[str, list[str]] = {}
    n_added = 0
    for name in files_by_fold[0]:
        stem = Path(name).stem
        image = cv2.imread(str(TRAIN_IMG_DIR / name), cv2.IMREAD_GRAYSCALE)
        parent_rles = parent_g.get(stem, [])
        merged, _ = disjoint_add(parent_rles, low_val_g.get(stem, []))
        survivors = []
        for rle in merged[len(parent_rles):]:
            feat = np.asarray([_features(rle, image)], dtype=np.float64)
            if int(clf.predict(feat)[0]) == 1 and dark_enough(decode_rle(rle), image):
                survivors.append(rle)
        n_added += len(survivors)
        kept[stem] = list(parent_rles) + survivors

    val_files = files_by_fold[0]
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    parent_row = _row(score_pq(gt_df, parent), len(parent))
    if abs(parent_row["pq"] - EXPECTED_PQ) > 1e-6:
        raise SystemExit(f"parent rescore drifted: {parent_row}")
    frames = [pred_to_df(stem, rles) for stem, rles in kept.items() if rles]
    pred = pd.concat(frames, ignore_index=True)
    gated = _row(score_pq(gt_df, pred), len(pred))
    lift = gated["pq"] - parent_row["pq"]
    payload = {
        "split": "fit on folds 1-4, score fold 0",
        "seed": SEED,
        "train_proposals": len(ys),
        "train_positives": int(sum(ys)),
        "coef": [float(v) for v in clf.coef_.ravel()],
        "intercept": float(clf.intercept_[0]),
        "fold0_added": n_added,
        "parent": parent_row,
        "oof_gate": gated,
        "lift_vs_2048": lift,
        "promote": bool(lift >= 0.01),
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    LOW_CACHE.write_text("cache not required\n")


if __name__ == "__main__":
    main()
