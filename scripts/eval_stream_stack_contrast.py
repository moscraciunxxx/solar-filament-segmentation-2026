#!/usr/bin/env python3
"""Drop faint masks from the whole year-shift stack.

The stack is the faint vote, then continued-fine-tune masks with no shared
pixel, then year-shift masks with no shared pixel. Every kept mask is then
dropped when its ring contrast is below the locked training floor. The floor
value and the 31-pixel ring stay fixed. Does not read test labels. Does not
submit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.contrast_floor import drop_faint_rles, mask_contrast
from filament.disjoint_add import disjoint_add
from filament.medoid_gt import medoid_gt_records
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR, YOLO_DIR
from filament.preprocess import read_gray
from filament.rle import decode_rle
from filament.small_drop import MIN_AREA
from filament.vote import MIN_VOTES, components, vote_map

PARENT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
FAINT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_faint_fold0_conf0.35_pred.csv"
SMALL = FAST_ARTIFACTS / "yolo11n_seg_erode_gt_fold0_conf0.35_pred.csv"
E30 = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_faint_e30_fold0_conf0.35_pred.csv"
SHIFT = FAST_ARTIFACTS / "yolo11l_seg_shift_erode3x1_fold0_pred.csv"
OUT = ARTIFACTS / "stream_stack_contrast_fold0.json"
EXPECTED_PQ = 0.4387235694048039
EXPECTED_STACK_PQ = 0.4572839656476841
RECORDED_FLOOR = 8.381084949352285
PUBLIC_GAP = 0.069
PUBLIC_BAR = 0.6


def should_promote(pq: float, gap: float = PUBLIC_GAP, bar: float = PUBLIC_BAR) -> bool:
    return (pq - gap) > bar


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


def _image_path(file_name: str, fold: int) -> Path:
    split = "val" if fold == 0 else "train"
    cached = YOLO_DIR / "images" / split / file_name
    if cached.exists():
        return cached
    return TRAIN_IMG_DIR / file_name


def contrast_floor(views, by_file, fold_of) -> tuple[float, int]:
    values: list[float] = []
    for name, fold in sorted(fold_of.items()):
        if fold == 0:
            continue
        image = read_gray(_image_path(name, fold))
        for rec in medoid_gt_records(views, by_file[name]):
            contrast = mask_contrast(decode_rle(rec["rle"]), image)
            if contrast is not None:
                values.append(contrast)
    if len(values) < 30:
        raise SystemExit(f"too few training filaments to set the floor: {len(values)}")
    return float(np.quantile(values, 0.10)), len(values)


def main() -> None:
    for path in (PARENT, FAINT, SMALL, E30, SHIFT):
        if not path.is_file():
            raise SystemExit(f"missing prediction file: {path}")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    floor, n_train = contrast_floor(views, by_file, fold_of)
    if abs(floor - RECORDED_FLOOR) > 1e-6:
        raise SystemExit(f"contrast floor drifted: {floor} vs {RECORDED_FLOOR}")
    val_files = sorted(name for name, fold in fold_of.items() if fold == 0)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    parent = pd.read_csv(PARENT)
    first = _row(score_pq(gt_df, parent), len(parent))
    second = _row(score_pq(gt_df, parent), len(parent))
    if first != second:
        raise SystemExit(f"parent scored twice and differed: {first} vs {second}")
    if abs(first["pq"] - EXPECTED_PQ) > 1e-6:
        raise SystemExit(f"parent rescore drifted: {first}")
    groups = [
        _by_stem(pd.read_csv(FAINT)),
        _by_stem(parent),
        _by_stem(pd.read_csv(SMALL)),
    ]
    e30_by = _by_stem(pd.read_csv(E30))
    shift_by = _by_stem(pd.read_csv(SHIFT))
    stack_frames = []
    kept_frames = []
    added_e30 = 0
    added_shift = 0
    dropped = 0
    for name in val_files:
        stem = Path(name).stem
        layers = [group.get(stem, []) for group in groups]
        if any(layers):
            binary = vote_map(layers, MIN_VOTES)
            vote_rles = components(binary, MIN_AREA)
        else:
            vote_rles = []
        with_e30, n_e30 = disjoint_add(vote_rles, e30_by.get(stem, []))
        merged, n_shift = disjoint_add(with_e30, shift_by.get(stem, []))
        added_e30 += n_e30
        added_shift += n_shift
        if merged:
            stack_frames.append(pred_to_df(stem, merged))
            image = read_gray(_image_path(name, 0))
            kept, n_dropped = drop_faint_rles(merged, image, floor)
            dropped += n_dropped
            if kept:
                kept_frames.append(pred_to_df(stem, kept))
    empty = pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    stack = pd.concat(stack_frames, ignore_index=True) if stack_frames else empty
    pred = pd.concat(kept_frames, ignore_index=True) if kept_frames else empty
    stack_scored = _row(score_pq(gt_df, stack), len(stack))
    if abs(stack_scored["pq"] - EXPECTED_STACK_PQ) > 1e-6:
        raise SystemExit(f"stack rescore drifted: {stack_scored}")
    scored = _row(score_pq(gt_df, pred), len(pred))
    formula = scored["mean_tp_iou"] * scored["tp"] / (
        scored["tp"] + 0.5 * scored["fp"] + 0.5 * scored["fn"]
    )
    if scored["tp"] and abs(formula - scored["pq"]) > 1e-9:
        raise SystemExit(f"formula check failed: {formula} vs {scored['pq']}")
    forecast = scored["pq"] - PUBLIC_GAP
    payload = {
        "split": "grouped year x count fold 0",
        "seed": SEED,
        "hypothesis": "year-shift stack, then drop every mask fainter than the training contrast floor",
        "min_votes": MIN_VOTES,
        "min_area": MIN_AREA,
        "contrast_floor": floor,
        "n_train_filaments": n_train,
        "masks_added_e30": added_e30,
        "masks_added_shift": added_shift,
        "masks_dropped_faint": dropped,
        "parent": first,
        "parent_second_call": second,
        "stack": stack_scored,
        "stack_contrast": scored,
        "formula_pq": formula,
        "lift_vs_stack": scored["pq"] - stack_scored["pq"],
        "fp_delta": scored["fp"] - stack_scored["fp"],
        "fn_delta": scored["fn"] - stack_scored["fn"],
        "public_gap_used": PUBLIC_GAP,
        "public_forecast": forecast,
        "promote": should_promote(scored["pq"]),
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
