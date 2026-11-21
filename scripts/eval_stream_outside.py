#!/usr/bin/env python3
"""Drop parent masks nobody else sees, and add witnessed all-annotator masks that miss the parent.

Targets filaments outside the 35-pair year-shift band. Cuts are fixed.
Does not read test labels. Does not submit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.outside_band import WITNESS_IOU, edit_outside_rles
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED

PARENT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
SHIFT = FAST_ARTIFACTS / "yolo11l_seg_shift_erode3x1_fold0_pred.csv"
SMALL = FAST_ARTIFACTS / "yolo11n_seg_erode_gt_fold0_conf0.35_pred.csv"
ALL = FAST_ARTIFACTS / "yolo11s_seg_all_fold0_conf0.40_pred.csv"
OUT = ARTIFACTS / "stream_outside_fold0.json"
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
    shift_by = _by_stem(pd.read_csv(SHIFT))
    small_by = _by_stem(pd.read_csv(SMALL))
    all_by = _by_stem(pd.read_csv(ALL))
    frames = []
    dropped = 0
    added = 0
    stems = set(parent_by) | set(shift_by) | set(small_by) | set(all_by)
    for stem in sorted(stems):
        if stem not in {Path(name).stem for name in val_files}:
            continue
        edited, n_dropped, n_added = edit_outside_rles(
            parent_by.get(stem, []),
            shift_by.get(stem, []),
            small_by.get(stem, []),
            all_by.get(stem, []),
        )
        dropped += n_dropped
        added += n_added
        if edited:
            frames.append(pred_to_df(stem, edited))
    pred = pd.concat(frames, ignore_index=True)
    scored = _row(score_pq(gt_df, pred), len(pred))
    forecast = scored["pq"] - PUBLIC_GAP
    payload = {
        "split": "grouped year x count fold 0",
        "seed": SEED,
        "hypothesis": "drop a 2048 mask unseen by shift, yolo11n, and the all-annotator model; add an all-annotator mask only when it misses the parent and shift or yolo11n overlaps it above 0.5",
        "witness_iou": WITNESS_IOU,
        "masks_dropped": dropped,
        "masks_added": added,
        "parent": parent_row,
        "outside": scored,
        "lift_vs_2048": scored["pq"] - parent_row["pq"],
        "fp_delta": scored["fp"] - parent_row["fp"],
        "fn_delta": scored["fn"] - parent_row["fn"],
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
