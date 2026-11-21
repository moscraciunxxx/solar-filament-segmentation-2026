#!/usr/bin/env python3
"""Run the 2048 weights again after painting out the parent masks.

The second pass can only see the disk that the first pass left empty.
A rediscovery that lands on a parent mask is the same object and is not
counted twice. Confidence and image size stay at the parent's 0.35 / 2048.
Does not read test labels. Does not submit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.disjoint_add import disjoint_add
from filament.inpaint import inpaint_parent
from filament.infer import pick_device, run_yolo_infer
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR

PARENT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
WEIGHTS = WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048" / "weights" / "best.pt"
OUT = ARTIFACTS / "stream_second_pass_fold0.json"
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

    from ultralytics import YOLO

    device = pick_device("mps")
    model = YOLO(str(WEIGHTS))
    frames = []
    added = 0
    for name in tqdm(val_files, desc="second-pass"):
        stem = Path(name).stem
        image = cv2.imread(str(TRAIN_IMG_DIR / name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise SystemExit(f"missing {name}")
        parent_rles = parent_by.get(stem, [])
        painted = inpaint_parent(image, parent_rles)
        tmp = FAST_ARTIFACTS / "_second_pass_tmp.png"
        cv2.imwrite(str(tmp), painted)
        new_rles = run_yolo_infer(
            model, tmp, conf=0.35, imgsz=2048, merge_gap=0, device=device
        )
        kept, n_added = disjoint_add(parent_rles, new_rles)
        added += n_added
        if kept:
            frames.append(pred_to_df(stem, kept))
    pred = pd.concat(frames, ignore_index=True)
    scored = _row(score_pq(gt_df, pred), len(pred))
    forecast = scored["pq"] - PUBLIC_GAP
    payload = {
        "split": "grouped year x count fold 0",
        "seed": SEED,
        "hypothesis": "second forward pass of the 2048 weights after parent masks are painted to the outside median; rediscoveries on a parent mask are not double-counted",
        "instances_added": added,
        "parent": parent_row,
        "second_pass": scored,
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
