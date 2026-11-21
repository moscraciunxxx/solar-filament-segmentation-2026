#!/usr/bin/env python3
"""Score one pre-registered disjoint add on locked grouped fold 0.

Keep every 2048 prediction. Add a year-shift instance only when its mask
has IoU exactly 0 with every 2048 mask on the same image. Not a full OR.
Does not train, does not read test labels, does not submit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.disjoint_add import disjoint_add
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED

PARENT_2048 = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
PARENT_SHIFT = FAST_ARTIFACTS / "yolo11l_seg_shift_erode3x1_fold0_pred.csv"
OUT = ARTIFACTS / "stream_disjoint_fold0.json"

# Locked rescore of the 2048 parent. Used only as a loader check.
EXPECTED_PQ = 0.4387235694048039
EXPECTED_TP = 949
EXPECTED_FP = 410
EXPECTED_FN = 640
PROMOTE_LIFT = 0.01

STATEMENT = (
    "This is not a full OR of the 2048 parent and the year-shift file."
)


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


def _parent_matches(row: dict) -> bool:
    return (
        row["tp"] == EXPECTED_TP
        and row["fp"] == EXPECTED_FP
        and row["fn"] == EXPECTED_FN
        and abs(row["pq"] - EXPECTED_PQ) < 1e-6
    )


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(name for name, fold in fold_of.items() if fold == 0)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))

    parent = pd.read_csv(PARENT_2048)
    shift = pd.read_csv(PARENT_SHIFT)
    parent_row = _row(score_pq(gt_df, parent), len(parent))
    if not _parent_matches(parent_row):
        payload = {
            "bug": "parent rescore is far from the locked 2048 fold-0 score; disjoint add was not scored",
            "parent": parent_row,
            "expected_parent_pq": EXPECTED_PQ,
            "expected_tp": EXPECTED_TP,
            "expected_fp": EXPECTED_FP,
            "expected_fn": EXPECTED_FN,
            "promote": False,
            "test_labels_used": False,
            "statement": STATEMENT,
        }
        OUT.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        sys.exit(2)

    parent_by = _by_stem(parent)
    shift_by = _by_stem(shift)
    merged: dict[str, list[str]] = {}
    added = 0
    for stem in set(parent_by) | set(shift_by):
        kept, n_added = disjoint_add(parent_by.get(stem, []), shift_by.get(stem, []))
        parent_rles = parent_by.get(stem, [])
        if kept[: len(parent_rles)] != parent_rles:
            raise SystemExit(f"parent masks were not kept intact for {stem}")
        added += n_added
        if kept:
            merged[stem] = kept
    merged_df = _frame(merged)
    disjoint_row = _row(score_pq(gt_df, merged_df), len(merged_df))
    promote = bool(disjoint_row["pq"] >= parent_row["pq"] + PROMOTE_LIFT)

    payload = {
        "split": "grouped year x count fold 0",
        "seed": SEED,
        "n_val_images": len(val_files),
        "rule": "IoU == 0 against every 2048 mask on the same image; no threshold sweep",
        "parent_file": str(PARENT_2048),
        "shift_file": str(PARENT_SHIFT),
        "parent": parent_row,
        "disjoint_add": disjoint_row,
        "shift_instances_in_file": int(len(shift)),
        "shift_instances_added": added,
        "promote": promote,
        "test_labels_used": False,
        "kaggle_submit_called": False,
        "statement": STATEMENT,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
