#!/usr/bin/env python3
"""Add patches that match real training-fold miss filaments.

Eight templates come from folds 1-3. The correlation cut is fixed from
fold 4 positives and quiet-Sun crops before fold 0 is scored. No sweep.
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
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.miss_template import (
    bank_threshold,
    expand_angles,
    max_correlation,
    oriented_patch,
    propose_template_rles,
    rasterize_label,
)
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, FAST_CACHE, SEED, TRAIN_IMG_DIR
from filament.preprocess import read_gray

PARENT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
POS_DIR = FAST_CACHE / "yolo_seg_miss"
NEG_DIR = FAST_CACHE / "yolo_seg_miss_neg"
OUT = ARTIFACTS / "stream_miss_template_fold0.json"
EXPECTED_PQ = 0.4387235694048039
PUBLIC_GAP = 0.069
N_TEMPLATES = 8


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


def _gray_and_mask(image_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    label_path = image_path.parents[2] / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
    if gray is None or not label_path.exists():
        return None
    return gray, rasterize_label(label_path.read_text(), gray.shape[0])


def _bank() -> list[tuple[np.ndarray, np.ndarray]]:
    images = sorted((POS_DIR / "images" / "train").glob("*_m*.jpg"))
    if len(images) < N_TEMPLATES:
        raise SystemExit(f"need at least {N_TEMPLATES} miss crops, found {len(images)}")
    picks = np.linspace(0, len(images) - 1, N_TEMPLATES).astype(int)
    bank: list[tuple[np.ndarray, np.ndarray]] = []
    for index in picks:
        loaded = _gray_and_mask(images[int(index)])
        if loaded is None:
            continue
        gray, mask = loaded
        patch = oriented_patch(gray, mask)
        if patch is None:
            continue
        bank.extend(expand_angles(*patch))
    if len(bank) < 4:
        raise SystemExit(f"template bank too small: {len(bank)}")
    return bank


def _scores(paths: list[Path], bank: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    values = []
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        values.append(max_correlation(gray, bank))
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    bank = _bank()
    pos_paths = sorted((POS_DIR / "images" / "val").glob("*_m*.jpg"))
    neg_paths = sorted((NEG_DIR / "images" / "val").glob("*_q*.jpg"))
    pos_scores = _scores(pos_paths, bank)
    neg_scores = _scores(neg_paths, bank)
    threshold = bank_threshold(pos_scores, neg_scores)
    print(
        json.dumps(
            {
                "n_templates": len(bank),
                "n_pos": int(pos_scores.size),
                "n_neg": int(neg_scores.size),
                "pos_p10": float(np.quantile(pos_scores, 0.10)),
                "neg_p95": float(np.quantile(neg_scores, 0.95)),
                "threshold": threshold,
            }
        ),
        flush=True,
    )
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
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
        extra = propose_template_rles(
            read_gray(TRAIN_IMG_DIR / name),
            parent_by.get(stem, []),
            bank,
            threshold,
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
        "hypothesis": "add a mask pasted from a training-fold miss template when its correlation clears the fold-4 holdout cut and it shares no pixel with the 2048 parent",
        "n_templates": len(bank),
        "n_pos_scores": int(pos_scores.size),
        "n_neg_scores": int(neg_scores.size),
        "pos_p10": float(np.quantile(pos_scores, 0.10)),
        "neg_p95": float(np.quantile(neg_scores, 0.95)),
        "threshold": threshold,
        "masks_added": added,
        "parent": first,
        "parent_second_call": second,
        "templated": scored,
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
