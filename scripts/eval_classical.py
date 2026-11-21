#!/usr/bin/env python3
"""Tune and score the classical detector on a grouped holdout using official PQ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.classical import ClassicalConfig, detect_classical
from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.infer import masks_to_rles
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, SEED, TRAIN_IMG_DIR
from filament.preprocess import read_gray


def predict_files(file_names, cfg: ClassicalConfig) -> pd.DataFrame:
    frames = []
    for file_name in tqdm(file_names, desc="classical"):
        image = read_gray(TRAIN_IMG_DIR / file_name)
        rles = masks_to_rles(detect_classical(image, cfg))
        frames.append(pred_to_df(Path(file_name).stem, rles))
    if not frames:
        return pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=80, help="Cap val images for a fast PQ probe")
    parser.add_argument("--dark-percentile", type=float, default=92.5)
    parser.add_argument("--blur-sigma", type=float, default=36.0)
    parser.add_argument("--min-area", type=int, default=96)
    args = parser.parse_args()

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = [fn for fn, fold in fold_of.items() if fold == args.fold]
    val_files.sort()
    if args.max_val and len(val_files) > args.max_val:
        val_files = val_files[: args.max_val]

    cfg = ClassicalConfig(
        dark_percentile=args.dark_percentile,
        blur_sigma=args.blur_sigma,
        min_area=args.min_area,
    )
    pred_df = predict_files(val_files, cfg)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    breakdown = score_pq(gt_df, pred_df)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = {
        "fold": args.fold,
        "n_val_images": len(val_files),
        "n_pred": int(len(pred_df)),
        "n_gt": int(len(gt_df)),
        "config": cfg.__dict__,
        "pq": breakdown.pq,
        "tp": breakdown.tp,
        "fp": breakdown.fp,
        "fn": breakdown.fn,
        "mean_tp_iou": breakdown.mean_tp_iou,
    }
    out_path = ARTIFACTS / f"classical_fold{args.fold}_pq.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
