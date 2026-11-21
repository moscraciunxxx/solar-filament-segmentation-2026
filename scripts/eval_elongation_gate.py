#!/usr/bin/env python3
"""Score the locked fold-0 parents and one pre-registered elongation gate.

Does not train, does not read test labels, does not submit.
Parents: yolo11s erode-gt 2048 (public 0.37) and yolo11l year-shift erode.
The gate drops round masks from the 2048 file only. Thresholds live in
filament.elongation and are not chosen from this score.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.elongation import MIN_AREA, MIN_ECCENTRICITY, filter_rles
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED

PARENT_2048 = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
PARENT_SHIFT = FAST_ARTIFACTS / "yolo11l_seg_shift_erode3x1_fold0_pred.csv"
OUT = ARTIFACTS / "elongation_gate_fold0.json"


def _row(br, n_pred: int) -> dict:
    denom = br.tp + 0.5 * br.fp + 0.5 * br.fn
    reconstructed = (br.mean_tp_iou * br.tp / denom) if denom else 0.0
    return {
        "pq": br.pq,
        "tp": br.tp,
        "fp": br.fp,
        "fn": br.fn,
        "mean_tp_iou": br.mean_tp_iou,
        "reconstructed_pq": reconstructed,
        "n_pred": n_pred,
        "n_annotator_images": br.n_annotator_images,
    }


def _by_stem(df: pd.DataFrame) -> dict[str, list[str]]:
    stems = df["filament_id"].str.rsplit("_", n=1).str[0]
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

    parent = pd.read_csv(PARENT_2048)
    shift = pd.read_csv(PARENT_SHIFT)
    run_a = _row(score_pq(gt_df, parent), len(parent))
    run_b = _row(score_pq(gt_df, parent), len(parent))
    shift_row = _row(score_pq(gt_df, shift), len(shift))

    grouped = _by_stem(parent)
    dropped = 0
    filtered: dict[str, list[str]] = {}
    for stem, rles in grouped.items():
        kept = filter_rles(rles)
        dropped += len(rles) - len(kept)
        if kept:
            filtered[stem] = kept
    filtered_df = _frame(filtered)
    gate_row = _row(score_pq(gt_df, filtered_df), len(filtered_df))
    lift = gate_row["pq"] - run_a["pq"]
    promote = bool(lift >= 0.01 and gate_row["fp"] < run_a["fp"])

    payload = {
        "split": "grouped year x count fold 0",
        "seed": SEED,
        "n_val_images": len(val_files),
        "parent_file": str(PARENT_2048),
        "shift_file": str(PARENT_SHIFT),
        "gate": {
            "min_eccentricity": MIN_ECCENTRICITY,
            "min_area": MIN_AREA,
            "chosen_after_score": False,
            "dropped_predictions": dropped,
        },
        "parent_2048_run_1": run_a,
        "parent_2048_run_2": run_b,
        "shift_erode": shift_row,
        "elongation_gate": gate_row,
        "lift_vs_2048": lift,
        "promote": promote,
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
