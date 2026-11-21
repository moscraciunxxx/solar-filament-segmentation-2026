#!/usr/bin/env python3
"""Score the flat-field yolo11s erode-gt run alone on fold 0.

Same conf 0.35 decode as the parent score path. Does not merge prediction
files and does not submit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.condinst import should_promote
from filament.infer import pick_device, run_yolo_infer
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, FAST_CACHE, SEED

FAMILY = "yolo11s_seg_erode_gt_2048_clv"
PARENT_PQ = 0.4387235694048039
PARENT_CSV = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
IMAGE_DIR = FAST_CACHE / "yolo_seg_erode_gt_clv" / "images" / "val"
WEIGHTS = FAST_CACHE / "weights" / FAMILY / "weights" / "best.pt"


def _image(file_name: str) -> Path:
    stem = Path(file_name).stem
    path = IMAGE_DIR / f"{stem}.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
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
    for file_name in tqdm(val_files, desc="clv yolo fold0"):
        rles = run_yolo_infer(
            model,
            _image(file_name),
            conf=0.35,
            imgsz=2048,
            merge_gap=0,
            device=device,
        )
        frames.append(pred_to_df(Path(file_name).stem, rles))
    pred = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    gt = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    scored = score_pq(gt, pred)
    parent = score_pq(gt, pd.read_csv(PARENT_CSV))
    if abs(parent.pq - PARENT_PQ) > 1e-6:
        raise SystemExit(f"parent rescore drifted: {parent.pq}")
    pred_csv = FAST_ARTIFACTS / f"{FAMILY}_fold0_conf0.35_pred.csv"
    pred.to_csv(pred_csv, index=False)
    payload = {
        "family": FAMILY,
        "weights": str(WEIGHTS),
        "imgsz": 2048,
        "conf": 0.35,
        "input": "center-to-limb flat field, three channels",
        "merged_with_parent": False,
        "pq": scored.pq,
        "tp": scored.tp,
        "fp": scored.fp,
        "fn": scored.fn,
        "n_pred": len(pred),
        "mean_tp_iou": scored.mean_tp_iou,
        "parent_pq": parent.pq,
        "parent_tp": parent.tp,
        "parent_fp": parent.fp,
        "parent_fn": parent.fn,
        "promote": should_promote(scored.pq, scored.tp, same_hit_set=True),
        "pred_csv": str(pred_csv),
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    out = ARTIFACTS / f"{FAMILY}_fold0.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
