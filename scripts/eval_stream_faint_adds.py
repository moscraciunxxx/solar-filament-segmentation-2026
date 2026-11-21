#!/usr/bin/env python3
"""Drop a unanimous add only when it is fainter than the locked contrast floor.

The base file stays, including its faint masks. The floor is the recorded
10th percentile from training folds other than 0. The 31-pixel ring is
unchanged. Does not read test labels. Does not submit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.agreed_add import add_unanimous_masks, drop_faint_adds
from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.contained_swap import contained_swap_masks
from filament.contrast_floor import mask_contrast
from filament.disjoint_add import disjoint_add
from filament.holes import fill_holes
from filament.interior_drop import drop_strict_interior
from filament.medoid_gt import medoid_gt_records
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.outside_band import drop_unseen_unless_triple_touch
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR, YOLO_DIR
from filament.preprocess import read_gray
from filament.rle import decode_rle, encode_rle
from filament.small_drop import MIN_AREA, drop_small_rles
from filament.tips import TIP_PX, extend_tips_inside
from filament.vote import MIN_VOTES, components, vote_map

PARENT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
FAINT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_faint_fold0_conf0.35_pred.csv"
SMALL = FAST_ARTIFACTS / "yolo11n_seg_erode_gt_fold0_conf0.35_pred.csv"
E30 = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_faint_e30_fold0_conf0.35_pred.csv"
SHIFT = FAST_ARTIFACTS / "yolo11l_seg_shift_erode3x1_fold0_pred.csv"
FULL = FAST_ARTIFACTS / "yolo11s_seg_full_2048_fold0_conf0.35_pred.csv"
ALL = FAST_ARTIFACTS / "yolo11s_seg_all_fold0_conf0.40_pred.csv"
LARGE = FAST_ARTIFACTS / "yolo11l_seg_1280_fold0_conf0.35_pred.csv"
RCNN = FAST_ARTIFACTS / "maskrcnn_r50_1024_fold0_conf0.35_pred.csv"
OUT = ARTIFACTS / "stream_faint_adds_fold0.json"
EXPECTED_PQ = 0.4387235694048039
EXPECTED_UNANIMOUS_PQ = 0.4639777995958413
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


def _vote(layers: list[list[str]]) -> list[str]:
    if not any(layers):
        return []
    return components(vote_map(layers, MIN_VOTES), MIN_AREA)


def _encode(masks: list[np.ndarray]) -> list[str]:
    return [encode_rle(mask) for mask in masks if int(mask.sum()) > 0]


def _decode(grouped: dict[str, list[str]], stem: str) -> list[np.ndarray]:
    return [decode_rle(rle) for rle in grouped.get(stem, [])]


def _best_partner(mask: np.ndarray, partners: list[np.ndarray]) -> np.ndarray:
    core = mask > 0
    area = int(core.sum())
    best = None
    best_iou = 0.0
    for partner in partners:
        cover = partner > 0
        inter = int(np.logical_and(core, cover).sum())
        if inter == 0:
            continue
        union = area + int(cover.sum()) - inter
        if union <= 0:
            continue
        iou = inter / union
        if iou > best_iou:
            best_iou = iou
            best = partner
    if best is None:
        return np.zeros_like(mask)
    return best


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
    for path in (PARENT, FAINT, SMALL, E30, SHIFT, FULL, ALL, LARGE, RCNN):
        if not path.is_file():
            raise SystemExit(f"missing prediction file: {path}")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    floor, n_train = contrast_floor(views, by_file, fold_of)
    if abs(floor - RECORDED_FLOOR) > 1e-6:
        raise SystemExit(f"contrast floor drifted: {floor} vs {RECORDED_FLOOR}")
    val_files = sorted(name for name, fold in fold_of.items() if fold == 0)
    if len(val_files) != 142:
        raise SystemExit(f"fold 0 should be 142 images, got {len(val_files)}")
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
    full_by = _by_stem(pd.read_csv(FULL))
    all_by = _by_stem(pd.read_csv(ALL))
    large_by = _by_stem(pd.read_csv(LARGE))
    rcnn_by = _by_stem(pd.read_csv(RCNN))
    small_by = groups[2]
    vote_sets = [groups[0], e30_by, full_by]
    unanimous_frames = []
    faint_frames = []
    masks_added = 0
    masks_kept = 0
    for name in val_files:
        stem = Path(name).stem
        layers = [group.get(stem, []) for group in groups]
        if any(layers):
            vote_rles = components(vote_map(layers, MIN_VOTES), MIN_AREA)
        else:
            vote_rles = []
        with_e30, _n_e30 = disjoint_add(vote_rles, e30_by.get(stem, []))
        merged, _n_shift = disjoint_add(with_e30, shift_by.get(stem, []))
        if not merged:
            triple: list[np.ndarray] = []
        else:
            bases = [decode_rle(rle) for rle in merged]
            partners = [decode_rle(rle) for rle in _vote([group.get(stem, []) for group in vote_sets])]
            swapped = contained_swap_masks(bases, partners)
            interior = drop_strict_interior(swapped)
            extended = [extend_tips_inside(mask, _best_partner(mask, partners)) for mask in interior]
            filled = [fill_holes(mask) for mask in extended]
            base_masks = [decode_rle(rle) for rle in drop_small_rles(_encode(filled), MIN_AREA)]
            triple = drop_unseen_unless_triple_touch(
                base_masks,
                _decode(shift_by, stem),
                _decode(small_by, stem),
                _decode(all_by, stem),
                _decode(large_by, stem),
                _decode(rcnn_by, stem),
                _decode(full_by, stem),
            )
        agreed, n_added = add_unanimous_masks(
            triple,
            [
                _decode(large_by, stem),
                _decode(rcnn_by, stem),
                _decode(full_by, stem),
            ],
        )
        image = read_gray(_image_path(name, 0))
        filtered, n_kept = drop_faint_adds(triple, agreed[len(triple) :], image, floor)
        masks_added += n_added
        masks_kept += n_kept
        agreed_rles = _encode(agreed)
        if agreed_rles:
            unanimous_frames.append(pred_to_df(stem, agreed_rles))
        filtered_rles = _encode(filtered)
        if filtered_rles:
            faint_frames.append(pred_to_df(stem, filtered_rles))
    empty = pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    unanimous_pred = pd.concat(unanimous_frames, ignore_index=True) if unanimous_frames else empty
    pred = pd.concat(faint_frames, ignore_index=True) if faint_frames else empty
    unanimous_scored = _row(score_pq(gt_df, unanimous_pred), len(unanimous_pred))
    if abs(unanimous_scored["pq"] - EXPECTED_UNANIMOUS_PQ) > 1e-6:
        raise SystemExit(f"unanimous rescore drifted: {unanimous_scored}")
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
        "hypothesis": "drop a unanimous add only when its contrast is below the locked training floor",
        "tip_px": TIP_PX,
        "min_votes": MIN_VOTES,
        "min_area": MIN_AREA,
        "contrast_floor": floor,
        "n_train_filaments": n_train,
        "masks_added": masks_added,
        "masks_kept": masks_kept,
        "parent": first,
        "parent_second_call": second,
        "unanimous_add": unanimous_scored,
        "faint_adds": scored,
        "formula_pq": formula,
        "lift_vs_best": scored["pq"] - unanimous_scored["pq"],
        "fp_delta": scored["fp"] - unanimous_scored["fp"],
        "fn_delta": scored["fn"] - unanimous_scored["fn"],
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
