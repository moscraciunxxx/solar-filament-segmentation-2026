#!/usr/bin/env python3
"""Score the 2048 predictions against medoid masks only on locked fold 0.

Every annotator is the official number. This run keeps one medoid view per
JPEG. It does not read test labels and it does not submit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco
from filament.medoid_gt import medoid_gt_records
from filament.metric import gt_annotations_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED

PARENT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
OUT = ARTIFACTS / "parent_vs_medoid_fold0.json"
ALL_ANNOTATOR_PQ = 0.4387235694048039


def _row(br, n_pred: int, n_gt: int) -> dict:
    return {
        "pq": br.pq,
        "tp": br.tp,
        "fp": br.fp,
        "fn": br.fn,
        "mean_tp_iou": br.mean_tp_iou,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "n_annotator_images": br.n_annotator_images,
    }


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(name for name, fold in fold_of.items() if fold == 0)
    records = []
    for name in val_files:
        records.extend(medoid_gt_records(views, by_file[name]))
    gt_df = gt_annotations_to_df(records)
    parent = pd.read_csv(PARENT)
    first = _row(score_pq(gt_df, parent), len(parent), len(gt_df))
    second = _row(score_pq(gt_df, parent), len(parent), len(gt_df))
    if first != second:
        raise SystemExit(f"medoid score differed across calls: {first} vs {second}")
    formula = first["mean_tp_iou"] * first["tp"] / (
        first["tp"] + 0.5 * first["fp"] + 0.5 * first["fn"]
    )
    if abs(formula - first["pq"]) > 1e-9:
        raise SystemExit(f"formula {formula} != pq {first['pq']}")
    payload = {
        "split": "grouped year x count fold 0, medoid annotator only",
        "seed": SEED,
        "hypothesis": "score the unchanged 2048 predictions against the medoid mask of each validation JPEG",
        "parent_csv": str(PARENT),
        "all_annotator_pq": ALL_ANNOTATOR_PQ,
        "medoid": first,
        "medoid_second_call": second,
        "formula_pq": formula,
        "delta_vs_all_annotator": first["pq"] - ALL_ANNOTATOR_PQ,
        "test_labels_used": False,
        "kaggle_submit_called": False,
        "promote": False,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
