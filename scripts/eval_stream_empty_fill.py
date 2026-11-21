#!/usr/bin/env python3
"""Fill fold-0 images the 2048 model left blank and score them.

The fill is a pixel in both the year-shift and all-annotator models.
Other images stay unchanged. No sweep. Does not read test labels. Does not submit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.empty_fill import fill_empty
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED

PARENT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
SHIFT = FAST_ARTIFACTS / "yolo11l_seg_shift_erode3x1_fold0_pred.csv"
ALL = FAST_ARTIFACTS / "yolo11s_seg_all_fold0_conf0.40_pred.csv"
OUT = ARTIFACTS / "stream_empty_fill_fold0.json"
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
    first = _row(score_pq(gt_df, parent), len(parent))
    second = _row(score_pq(gt_df, parent), len(parent))
    if first != second:
        raise SystemExit(f"parent scored twice and differed: {first} vs {second}")
    if abs(first["pq"] - EXPECTED_PQ) > 1e-6:
        raise SystemExit(f"parent rescore drifted: {first}")
    parent_by = _by_stem(parent)
    shift_by = _by_stem(pd.read_csv(SHIFT))
    all_by = _by_stem(pd.read_csv(ALL))
    frames = []
    filled_images = 0
    added = 0
    for name in val_files:
        stem = Path(name).stem
        base = parent_by.get(stem, [])
        pieces = fill_empty(base, shift_by.get(stem, []), all_by.get(stem, []))
        if not base and pieces:
            filled_images += 1
            added += len(pieces)
        if pieces:
            frames.append(pred_to_df(stem, pieces))
    pred = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["filament_id", "segmentation_rle"]
    )
    scored = _row(score_pq(gt_df, pred), len(pred))
    forecast = scored["pq"] - PUBLIC_GAP
    payload = {
        "split": "grouped year x count fold 0",
        "seed": SEED,
        "hypothesis": "on images with no 2048 mask, keep a pixel only when year-shift and the all-annotator model both mark it",
        "images_filled": filled_images,
        "masks_added": added,
        "parent": first,
        "parent_second_call": second,
        "filled": scored,
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
