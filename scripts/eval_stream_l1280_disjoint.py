#!/usr/bin/env python3
"""Add plain yolo11l-1280 masks that do not touch the 2048 parent.

This checkpoint is not the year-shift model, not an erode-gt sibling, and
not the physics, residual, or tile stacks already scored. One rule: IoU == 0.
No sweep. Does not read test labels. Does not submit.
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

PARENT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
CANDIDATE = FAST_ARTIFACTS / "yolo11l_seg_1280_fold0_conf0.35_pred.csv"
OUT = ARTIFACTS / "stream_l1280_disjoint_fold0.json"
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
    if not CANDIDATE.exists():
        raise SystemExit(f"missing {CANDIDATE}")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(name for name, fold in fold_of.items() if fold == 0)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    parent = pd.read_csv(PARENT)
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
        if kept[: len(parent_rles)] != parent_rles:
            raise SystemExit(f"parent masks were not kept intact for {stem}")
        added += n_added
        if kept:
            merged[stem] = kept
    frames = [pred_to_df(stem, rles) for stem, rles in merged.items() if rles]
    pred = pd.concat(frames, ignore_index=True)
    scored = _row(score_pq(gt_df, pred), len(pred))
    forecast = scored["pq"] - PUBLIC_GAP
    payload = {
        "split": "grouped year x count fold 0",
        "seed": SEED,
        "candidate": str(CANDIDATE),
        "hypothesis": "plain yolo11l-1280 masks added only when IoU with every 2048 mask is 0",
        "parent": parent_row,
        "l1280_disjoint": scored,
        "instances_added": added,
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
