#!/usr/bin/env python3
"""Disjoint-add yolo11m erode-gt onto the 2048 parent. Locked fold 0.

Not the year-shift file, not a second confidence of the 2048 weights,
not a watershed, and not a darkness or logistic gate. One rule: IoU == 0.
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
from filament.disjoint_add import disjoint_add
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED

PARENT_2048 = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
CANDIDATE = FAST_ARTIFACTS / "yolo11m_seg_erode_gt_fold0_conf0.35_pred.csv"
OUT = ARTIFACTS / "stream_m_disjoint_fold0.json"
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
    parent = pd.read_csv(PARENT_2048)
    cand = pd.read_csv(CANDIDATE)
    parent_row = _row(score_pq(gt_df, parent), len(parent))
    if abs(parent_row["pq"] - EXPECTED_PQ) > 1e-6:
        raise SystemExit(f"parent rescore drifted: {parent_row}")
    parent_by = _by_stem(parent)
    cand_by = _by_stem(cand)
    merged: dict[str, list[str]] = {}
    added = 0
    for stem in set(parent_by) | set(cand_by):
        parent_rles = parent_by.get(stem, [])
        kept, n_added = disjoint_add(parent_rles, cand_by.get(stem, []))
        added += n_added
        if kept:
            merged[stem] = kept
    frames = [pred_to_df(stem, rles) for stem, rles in merged.items() if rles]
    pred = pd.concat(frames, ignore_index=True)
    gated = _row(score_pq(gt_df, pred), len(pred))
    lift = gated["pq"] - parent_row["pq"]
    payload = {
        "split": "grouped year x count fold 0",
        "seed": SEED,
        "candidate": str(CANDIDATE),
        "hypothesis": "yolo11m erode-gt masks added only when IoU with every 2048 mask is 0",
        "parent": parent_row,
        "m_disjoint": gated,
        "instances_added": added,
        "lift_vs_2048": lift,
        "promote": bool(lift >= 0.01),
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
