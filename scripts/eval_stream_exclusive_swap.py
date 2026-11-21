#!/usr/bin/env python3
"""Keep the faint/continued/full swap, then fill only the masks it left.

The primary partner is the vote that scored 0.45902. The secondary partner
is the faint/continued/raw-shift vote. A secondary mask may replace a stack
mask only when the primary vote did not, and only when it is strictly larger
and covers every pixel. Votes stay at 2 and the component floor stays at 96.
Does not read test labels. Does not submit.
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
from filament.exclusive_swap import exclusive_contained_swap
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED
from filament.rle import decode_rle, encode_rle
from filament.small_drop import MIN_AREA
from filament.vote import MIN_VOTES, components, vote_map

PARENT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
FAINT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_faint_fold0_conf0.35_pred.csv"
SMALL = FAST_ARTIFACTS / "yolo11n_seg_erode_gt_fold0_conf0.35_pred.csv"
E30 = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_faint_e30_fold0_conf0.35_pred.csv"
SHIFT = FAST_ARTIFACTS / "yolo11l_seg_shift_erode3x1_fold0_pred.csv"
FULL = FAST_ARTIFACTS / "yolo11s_seg_full_2048_fold0_conf0.35_pred.csv"
RAW = FAST_ARTIFACTS / "yolo11l_seg_shift_fold0_conf0.35_pred.csv"
OUT = ARTIFACTS / "stream_exclusive_swap_fold0.json"
EXPECTED_PQ = 0.4387235694048039
EXPECTED_STACK_PQ = 0.4572839656476841
EXPECTED_VOTE_SWAP_PQ = 0.4590209919721129
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


def _changed(before: list[np.ndarray], after: list[np.ndarray]) -> int:
    return sum(
        int(not np.array_equal(left > 0, right > 0))
        for left, right in zip(before, after, strict=True)
    )


def main() -> None:
    for path in (PARENT, FAINT, SMALL, E30, SHIFT, FULL, RAW):
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
    raw_by = _by_stem(pd.read_csv(RAW))
    primary_sets = [groups[0], e30_by, full_by]
    secondary_sets = [groups[0], e30_by, raw_by]
    stack_frames = []
    primary_frames = []
    swap_frames = []
    added_e30 = 0
    added_shift = 0
    changed_primary = 0
    changed = 0
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
        stack_frames.append(pred_to_df(stem, merged))
        bases = [decode_rle(rle) for rle in merged]
        primary = [decode_rle(rle) for rle in _vote([group.get(stem, []) for group in primary_sets])]
        secondary = [decode_rle(rle) for rle in _vote([group.get(stem, []) for group in secondary_sets])]
        first_swap = contained_swap_masks(bases, primary)
        both = exclusive_contained_swap(bases, primary, secondary)
        changed_primary += _changed(bases, first_swap)
        changed += _changed(bases, both)
        primary_kept = _encode(first_swap)
        kept = _encode(both)
        if primary_kept:
            primary_frames.append(pred_to_df(stem, primary_kept))
        if kept:
            swap_frames.append(pred_to_df(stem, kept))
    empty = pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    stack = pd.concat(stack_frames, ignore_index=True) if stack_frames else empty
    primary_pred = pd.concat(primary_frames, ignore_index=True) if primary_frames else empty
    pred = pd.concat(swap_frames, ignore_index=True) if swap_frames else empty
    stack_scored = _row(score_pq(gt_df, stack), len(stack))
    if abs(stack_scored["pq"] - EXPECTED_STACK_PQ) > 1e-6:
        raise SystemExit(f"stack rescore drifted: {stack_scored}")
    primary_scored = _row(score_pq(gt_df, primary_pred), len(primary_pred))
    if abs(primary_scored["pq"] - EXPECTED_VOTE_SWAP_PQ) > 1e-6:
        raise SystemExit(f"primary swap drifted: {primary_scored}")
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
        "hypothesis": "keep the faint/continued/full containment swap, then let a faint/continued/raw-shift vote replace only a stack mask that swap left untouched",
        "min_votes": MIN_VOTES,
        "min_area": MIN_AREA,
        "masks_added_e30": added_e30,
        "masks_added_shift": added_shift,
        "masks_changed_primary": changed_primary,
        "masks_changed": changed,
        "parent": first,
        "parent_second_call": second,
        "stack": stack_scored,
        "contained_swap_vote": primary_scored,
        "exclusive_swap": scored,
        "formula_pq": formula,
        "lift_vs_stack": scored["pq"] - stack_scored["pq"],
        "lift_vs_primary": scored["pq"] - primary_scored["pq"],
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
