#!/usr/bin/env python3
"""Gate the disjoint year-shift adds by a pre-registered darkness margin.

Parent masks are always kept. An added mask stays only when the JPEG inside
the mask is at least 12 gray levels darker than the ring around it.
Does not read test labels. Does not submit. One cut, no sweep.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.dark_delta import MIN_DELTA, dark_enough
from filament.disjoint_add import disjoint_add
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR
from filament.rle import decode_rle

PARENT_2048 = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
PARENT_SHIFT = FAST_ARTIFACTS / "yolo11l_seg_shift_erode3x1_fold0_pred.csv"
OUT = ARTIFACTS / "stream_dark_delta_fold0.json"
EXPECTED_PQ = 0.4387235694048039


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


def _by_stem(df: pd.DataFrame) -> dict[str, list[str]]:
    stems = df["filament_id"].str.split("_", n=1).str[0]
    grouped: dict[str, list[str]] = {}
    for stem, rle in zip(stems, df["segmentation_rle"], strict=True):
        grouped.setdefault(stem, []).append(rle)
    return grouped


def _frame(grouped: dict[str, list[str]]) -> pd.DataFrame:
    frames = [pred_to_df(stem, rles) for stem, rles in grouped.items() if rles]
    if not frames:
        return pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(name for name, fold in fold_of.items() if fold == 0)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    jpeg_of = {Path(name).stem: TRAIN_IMG_DIR / name for name in val_files}

    parent_g = _by_stem(pd.read_csv(PARENT_2048))
    shift_g = _by_stem(pd.read_csv(PARENT_SHIFT))
    kept: dict[str, list[str]] = {}
    n_added = 0
    n_kept = 0
    for stem, parent_rles in parent_g.items():
        merged, n_add = disjoint_add(parent_rles, shift_g.get(stem, []))
        n_added += n_add
        added = merged[len(parent_rles):]
        image = cv2.imread(str(jpeg_of[stem]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise SystemExit(f"missing jpeg for {stem}")
        survivors = []
        for rle in added:
            if dark_enough(decode_rle(rle), image, MIN_DELTA):
                survivors.append(rle)
        n_kept += len(survivors)
        kept[stem] = list(parent_rles) + survivors

    pred = _frame(kept)
    parent_df = pd.read_csv(PARENT_2048)
    parent_row = _row(score_pq(gt_df, parent_df), len(parent_df))
    if abs(parent_row["pq"] - EXPECTED_PQ) > 1e-6:
        raise SystemExit(f"parent rescore drifted: {parent_row}")
    gated = _row(score_pq(gt_df, pred), len(pred))
    lift = gated["pq"] - parent_row["pq"]
    payload = {
        "split": "grouped year x count fold 0",
        "seed": SEED,
        "min_delta": MIN_DELTA,
        "chosen_after_score": False,
        "shift_instances_added_before_gate": n_added,
        "shift_instances_kept": n_kept,
        "parent": parent_row,
        "dark_delta": gated,
        "lift_vs_2048": lift,
        "promote": bool(lift >= 0.01),
        "test_labels_used": False,
        "kaggle_submit_called": False,
        "statement": "Parent masks stay. Only IoU-0 year-shift adds that are 12 gray levels darker than their ring are kept.",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
