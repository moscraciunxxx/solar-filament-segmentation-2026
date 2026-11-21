#!/usr/bin/env python3
"""Score yolo11s union-dedup 2048 alone on fold 0. Conf 0.35. Once. No submit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.infer import pick_device, run_yolo_infer
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, FAST_CACHE, SEED, TRAIN_IMG_DIR

FAMILY = "yolo11s_seg_union_dedup_2048"
WEIGHTS = FAST_CACHE / "weights" / FAMILY / "weights" / "best.pt"
OUT = ARTIFACTS / f"{FAMILY}_fold0.json"
SINGLE_TTA_PQ = 0.4651884607358067
CHAMPION_PQ = 0.4741684688395591


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"fold 0 already scored: {OUT}")
    if not WEIGHTS.is_file():
        raise SystemExit(f"missing {WEIGHTS}")
    from ultralytics import YOLO

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(name for name, fold in fold_of.items() if fold == 0)
    if len(val_files) != 142:
        raise SystemExit(f"fold 0 has {len(val_files)} images")
    device = pick_device("mps")
    model = YOLO(str(WEIGHTS))
    frames = []
    for file_name in tqdm(val_files, desc="union dedup fold0"):
        path = TRAIN_IMG_DIR / file_name
        rles = run_yolo_infer(model, path, conf=0.35, imgsz=2048, merge_gap=0, device=device)
        frames.append(pred_to_df(Path(file_name).stem, rles))
    pred = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    gt = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    scored = score_pq(gt, pred)
    pred_csv = FAST_ARTIFACTS / f"{FAMILY}_fold0_conf0.35_pred.csv"
    pred.to_csv(pred_csv, index=False)
    payload = {
        "family": FAMILY,
        "weights": str(WEIGHTS),
        "labels": "union of annotator instances deduped at IoU>0.5",
        "imgsz": 2048,
        "conf": 0.35,
        "fold": 0,
        "pq": scored.pq,
        "tp": scored.tp,
        "fp": scored.fp,
        "fn": scored.fn,
        "n_pred": int(len(pred)),
        "mean_tp_iou": scored.mean_tp_iou,
        "single_model_tta_pq": SINGLE_TTA_PQ,
        "delta_vs_single_tta": scored.pq - SINGLE_TTA_PQ,
        "champion_pq": CHAMPION_PQ,
        "beats_single_tta_by_0.01": bool(scored.pq > SINGLE_TTA_PQ + 0.01),
        "pred_csv": str(pred_csv),
        "test_labels_used": False,
        "kaggle_submit_called": False,
        "do_not_submit": True,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print("DONE union_dedup_fold0", json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
