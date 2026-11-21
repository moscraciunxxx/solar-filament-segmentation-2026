#!/usr/bin/env python3
"""Extend interior-drop tips only where the faint/continued/full vote agrees.

The distance stays at 8 pixels along the major axis. A new pixel is kept
only when the highest-overlap vote mask already marks it. A vote mask that
misses the filament is ignored. Does not read test labels. Does not submit.
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
from filament.contained_swap import contained_swap_masks
from filament.disjoint_add import disjoint_add
from filament.interior_drop import drop_strict_interior
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED
from filament.rle import decode_rle, encode_rle
from filament.small_drop import MIN_AREA
from filament.tips import TIP_PX, extend_tips_inside
from filament.vote import MIN_VOTES, components, vote_map

PARENT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
FAINT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_faint_fold0_conf0.35_pred.csv"
SMALL = FAST_ARTIFACTS / "yolo11n_seg_erode_gt_fold0_conf0.35_pred.csv"
E30 = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_faint_e30_fold0_conf0.35_pred.csv"
SHIFT = FAST_ARTIFACTS / "yolo11l_seg_shift_erode3x1_fold0_pred.csv"
FULL = FAST_ARTIFACTS / "yolo11s_seg_full_2048_fold0_conf0.35_pred.csv"
OUT = ARTIFACTS / "stream_tip_inside_fold0.json"
EXPECTED_PQ = 0.4387235694048039
EXPECTED_INTERIOR_PQ = 0.4590659230132008
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


def main() -> None:
    for path in (PARENT, FAINT, SMALL, E30, SHIFT, FULL):
        if not path.is_file():
            raise SystemExit(f"missing prediction file: {path}")
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
    groups = [
        _by_stem(pd.read_csv(FAINT)),
        _by_stem(parent),
        _by_stem(pd.read_csv(SMALL)),
    ]
    e30_by = _by_stem(pd.read_csv(E30))
    shift_by = _by_stem(pd.read_csv(SHIFT))
    full_by = _by_stem(pd.read_csv(FULL))
    vote_sets = [groups[0], e30_by, full_by]
    interior_frames = []
    tip_frames = []
    added_e30 = 0
    added_shift = 0
    masks_dropped = 0
    masks_extended = 0
    for name in val_files:
        stem = Path(name).stem
        layers = [group.get(stem, []) for group in groups]
        if any(layers):
            vote_rles = components(vote_map(layers, MIN_VOTES), MIN_AREA)
        else:
            vote_rles = []
        with_e30, n_e30 = disjoint_add(vote_rles, e30_by.get(stem, []))
        merged, n_shift = disjoint_add(with_e30, shift_by.get(stem, []))
        added_e30 += n_e30
        added_shift += n_shift
        if not merged:
            continue
        bases = [decode_rle(rle) for rle in merged]
        partners = [decode_rle(rle) for rle in _vote([group.get(stem, []) for group in vote_sets])]
        swapped = contained_swap_masks(bases, partners)
        kept = drop_strict_interior(swapped)
        masks_dropped += len([mask for mask in swapped if int(mask.sum()) > 0]) - len(kept)
        extended = [extend_tips_inside(mask, _best_partner(mask, partners)) for mask in kept]
        masks_extended += sum(
            int(not np.array_equal(before > 0, after > 0))
            for before, after in zip(kept, extended, strict=True)
        )
        interior_rles = _encode(kept)
        tip_rles = _encode(extended)
        if interior_rles:
            interior_frames.append(pred_to_df(stem, interior_rles))
        if tip_rles:
            tip_frames.append(pred_to_df(stem, tip_rles))
    empty = pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    interior = pd.concat(interior_frames, ignore_index=True) if interior_frames else empty
    pred = pd.concat(tip_frames, ignore_index=True) if tip_frames else empty
    interior_scored = _row(score_pq(gt_df, interior), len(interior))
    if abs(interior_scored["pq"] - EXPECTED_INTERIOR_PQ) > 1e-6:
        raise SystemExit(f"interior rescore drifted: {interior_scored}")
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
        "hypothesis": "extend an interior-drop mask 8 pixels along its major axis only where the faint/continued/full vote already marks the pixel",
        "tip_px": TIP_PX,
        "min_votes": MIN_VOTES,
        "min_area": MIN_AREA,
        "masks_added_e30": added_e30,
        "masks_added_shift": added_shift,
        "masks_dropped": masks_dropped,
        "masks_extended": masks_extended,
        "parent": first,
        "parent_second_call": second,
        "interior_drop": interior_scored,
        "tip_inside": scored,
        "formula_pq": formula,
        "lift_vs_interior": scored["pq"] - interior_scored["pq"],
        "fp_delta": scored["fp"] - interior_scored["fp"],
        "fn_delta": scored["fn"] - interior_scored["fn"],
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
