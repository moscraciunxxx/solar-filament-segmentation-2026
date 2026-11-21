#!/usr/bin/env python3
"""Score one pre-registered watershed split on locked grouped fold 0.

Parent: yolo11s erode-gt 2048 conf 0.35. The split reads predicted masks
only. Peak distance is 48 pixels and is not chosen from this score.
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
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, SEED
from filament.rle import decode_rle, encode_rle
from filament.watershed_split import MIN_PEAK_DISTANCE, check_split_contract, split_mask

PARENT = Path(
    "~/.cache/filament-segmentation-2026/artifacts/"
    "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
)
OUT = ARTIFACTS / "stream_watershed_fold0.json"
EXPECTED_PQ = 0.4387235694048039
EXPECTED_TP = 949
EXPECTED_FP = 410
EXPECTED_FN = 640
EXPECTED_VAL_IMAGES = 142
FP_EXPLODE_MULTIPLE = 2


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


def _split_grouped(grouped: dict[str, list[str]]) -> tuple[dict[str, list[str]], int]:
    out: dict[str, list[str]] = {}
    n_split = 0
    for stem, rles in grouped.items():
        kept: list[str] = []
        for rle in rles:
            pieces = split_mask(decode_rle(rle), MIN_PEAK_DISTANCE)
            if len(pieces) <= 1:
                kept.append(rle)
                continue
            n_split += 1
            kept.extend(encode_rle(piece) for piece in pieces)
        if kept:
            out[stem] = kept
    return out, n_split


def main() -> None:
    if MIN_PEAK_DISTANCE != 48:
        raise SystemExit(f"peak distance must stay pre-registered at 48, got {MIN_PEAK_DISTANCE}")
    check_split_contract()

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(name for name, fold in fold_of.items() if fold == 0)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))

    parent = pd.read_csv(PARENT)
    parent_row = _row(score_pq(gt_df, parent), len(parent))
    parent_ok = (
        len(val_files) == EXPECTED_VAL_IMAGES
        and parent_row["tp"] == EXPECTED_TP
        and parent_row["fp"] == EXPECTED_FP
        and parent_row["fn"] == EXPECTED_FN
        and abs(parent_row["pq"] - EXPECTED_PQ) <= 1e-6
    )
    if not parent_ok:
        bug = {
            "bug": "parent rescore does not match the locked fold-0 loader check",
            "n_val_images": len(val_files),
            "parent": parent_row,
            "expected": {
                "pq": EXPECTED_PQ,
                "tp": EXPECTED_TP,
                "fp": EXPECTED_FP,
                "fn": EXPECTED_FN,
                "n_val_images": EXPECTED_VAL_IMAGES,
            },
            "test_labels_used": False,
        }
        print(json.dumps(bug, indent=2))
        raise SystemExit(2)

    grouped, n_split = _split_grouped(_by_stem(parent))
    split_df = _frame(grouped)
    split_row = _row(score_pq(gt_df, split_df), len(split_df))
    lift = split_row["pq"] - parent_row["pq"]
    fp_exploded = bool(split_row["fp"] >= FP_EXPLODE_MULTIPLE * parent_row["fp"])
    promote = bool(lift >= 0.01 and not fp_exploded)

    payload = {
        "split": "grouped year x count fold 0",
        "hypothesis": "watershed split of one predicted mask when two distance-transform peaks are at least 48 pixels apart",
        "seed": SEED,
        "n_val_images": len(val_files),
        "parent_file": str(PARENT),
        "min_peak_distance": MIN_PEAK_DISTANCE,
        "chosen_after_score": False,
        "parent": parent_row,
        "split_score": split_row,
        "n_masks_split": n_split,
        "lift_vs_parent": lift,
        "fp_explode_multiple": FP_EXPLODE_MULTIPLE,
        "fp_exploded": fp_exploded,
        "promote": promote,
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
