#!/usr/bin/env python3
"""Score the eroded medoid outlines as if they were predictions.

This is a ceiling for the eroded training target on locked fold 0. It uses
fold-0 medoid polygons, so it is not a submission and promote is forced
false. It does not read Kaggle test labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco, medoid_view_id, rasterize_polygons
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.rle import encode_rle
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED
from filament.yolo_export import ERODE_GT_ITERS, ERODE_GT_K, erode_instance_polygon

PARENT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
OUT = ARTIFACTS / "stream_erode_ceiling_fold0.json"
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
    frames = []
    dropped = 0
    kept = 0
    for name in val_files:
        stem = Path(name).stem
        view = views[medoid_view_id(views, by_file[name])]
        rles = []
        for filament in view.filaments:
            contour = erode_instance_polygon(filament.polygon, k=ERODE_GT_K, iters=ERODE_GT_ITERS)
            if contour is None:
                dropped += 1
                continue
            mask = rasterize_polygons([contour])
            if int(mask.sum()) == 0:
                dropped += 1
                continue
            rles.append(encode_rle(mask))
            kept += 1
        if rles:
            frames.append(pred_to_df(stem, rles))
    pred = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["filament_id", "segmentation_rle"]
    )
    scored = _row(score_pq(gt_df, pred), len(pred))
    formula = scored["mean_tp_iou"] * scored["tp"] / (
        scored["tp"] + 0.5 * scored["fp"] + 0.5 * scored["fn"]
    )
    if scored["tp"] and abs(formula - scored["pq"]) > 1e-9:
        raise SystemExit(f"formula check failed: {formula} vs {scored['pq']}")
    payload = {
        "split": "grouped year x count fold 0",
        "seed": SEED,
        "hypothesis": "ceiling only: predict the fold-0 medoid after the locked 3x3 ellipse erosion. Not a model and not a submission.",
        "oracle": True,
        "erode_k": ERODE_GT_K,
        "erode_iters": ERODE_GT_ITERS,
        "instances_kept": kept,
        "instances_dropped": dropped,
        "parent": first,
        "parent_second_call": second,
        "erode_ceiling": scored,
        "formula_pq": formula,
        "public_gap_used": PUBLIC_GAP,
        "public_forecast": scored["pq"] - PUBLIC_GAP,
        "promote": False,
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
