#!/usr/bin/env python3
"""Three-seed ensemble of the parent recipe, fold 0 only.

Seeds 20260910 (parent), 20260911, and 20260912. Same 8 views x 3 scales.
Fusion is the frozen parent TTA setting: IoU 0.5, min votes 8. Not retuned.
Fold 0 is scored once. Never submits. Never reads test labels.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.metric import gt_annotations_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, WEIGHTS_DIR

spec = importlib.util.spec_from_file_location("eval_parent_tta", ROOT / "scripts" / "eval_parent_tta.py")
tta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tta)

FROZEN_IOU = 0.5
# Min votes 8 was the single-model freeze (24 views). It is not valid as a
# 72-voter threshold. This grid is ranked on folds 1-4 only.
VOTE_GRID = (8, 16, 24)
TIE_PQ = 0.005
MODELS = (
    ("s20260910", WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048" / "weights" / "best.pt", FAST_ARTIFACTS / "tta_parent_2048"),
    ("s20260911", WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048_s20260911" / "weights" / "best.pt", FAST_ARTIFACTS / "tta_s20260911"),
    ("s20260912", WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048_s20260912" / "weights" / "best.pt", FAST_ARTIFACTS / "tta_s20260912"),
)
PARENT_TTA = {
    "pq": 0.4651884607358067,
    "tp": 1018,
    "fp": 409,
    "fn": 571,
    "n_pred": 877,
    "mean_tp_iou": 0.6891003917383071,
}


def _payloads(model_name: str, weights: Path, cache: Path, files: list[str]) -> dict:
    if not weights.exists():
        raise SystemExit(f"missing {weights}")
    cache.mkdir(parents=True, exist_ok=True)
    tta.CACHE = cache
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out = {}
    t0 = time.perf_counter()
    for i, file_name in enumerate(files, start=1):
        out[file_name] = tta._load_or_infer(model, file_name)
        if i == 1 or i % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"{model_name} cached {i}/{len(files)} elapsed_min {elapsed / 60:.1f}",
                flush=True,
            )
    del model
    return out


def _ensure_cache(model_name: str, weights: Path, cache: Path, files: list[str]) -> None:
    if not weights.exists():
        raise SystemExit(f"missing {weights}")
    cache.mkdir(parents=True, exist_ok=True)
    tta.CACHE = cache
    missing = [fn for fn in files if not tta._cache_path(fn).exists()]
    print(f"{model_name} missing {len(missing)}/{len(files)}", flush=True)
    if not missing:
        return
    from ultralytics import YOLO

    model = YOLO(str(weights))
    t0 = time.perf_counter()
    for i, file_name in enumerate(missing, start=1):
        tta._load_or_infer(model, file_name)
        if i == 1 or i % 10 == 0:
            print(
                f"{model_name} inferred {i}/{len(missing)} elapsed_min {(time.perf_counter() - t0) / 60:.1f}",
                flush=True,
            )
    del model


def _read_merged(file_name: str) -> dict[str, list[str]]:
    payload: dict[str, list[str]] = {}
    for name, _, cache in MODELS:
        tta.CACHE = cache
        one = json.loads(tta._cache_path(file_name).read_text())
        for key, rles in one.items():
            payload[f"{name}_{key}"] = rles
    return payload


def _groups(masks: list, iou_thr: float) -> list[list[int]]:
    import numpy as np

    if not masks:
        return []
    boxes = [tta._box(m) for m in masks]
    areas = [int(m.sum()) for m in masks]
    order = sorted(range(len(masks)), key=lambda i: areas[i], reverse=True)
    used = [False] * len(masks)
    groups: list[list[int]] = []
    for i in order:
        if used[i] or boxes[i] is None:
            continue
        group = [i]
        used[i] = True
        ax0, ay0, ax1, ay1 = boxes[i]
        for j in order:
            if used[j] or boxes[j] is None:
                continue
            bx0, by0, bx1, by1 = boxes[j]
            if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
                continue
            inter = np.logical_and(masks[i], masks[j]).sum()
            union = np.logical_or(masks[i], masks[j]).sum()
            if union and (inter / union) >= iou_thr:
                group.append(j)
                used[j] = True
        groups.append(group)
    return groups


def _instances_for_votes(masks: list, groups: list[list[int]], min_votes: int):
    import numpy as np

    from filament.infer import yolo_masks_to_instances

    kept = []
    for group in groups:
        if len(group) < min_votes:
            continue
        vote = np.zeros((tta.IMAGE_SIZE, tta.IMAGE_SIZE), dtype=np.uint8)
        for k in group:
            vote += masks[k]
        merged = (vote >= max(1, (len(group) + 1) // 2)).astype(np.uint8)
        if int(merged.sum()) >= 96:
            kept.append(merged)
    return yolo_masks_to_instances(kept, merge_gap=0)


def _vote_frames(files: list[str]) -> dict[int, object]:
    from filament.rle import encode_rle

    buckets = {votes: [] for votes in VOTE_GRID}
    t0 = time.perf_counter()
    for i, file_name in enumerate(files, start=1):
        masks = tta._decode_all(_read_merged(file_name))
        groups = _groups(masks, FROZEN_IOU)
        for votes in VOTE_GRID:
            inst = _instances_for_votes(masks, groups, votes)
            buckets[votes].append(tta.pred_to_df(Path(file_name).stem, [encode_rle(m) for m in inst]))
        if i == 1 or i % 25 == 0:
            print(f"fused {i}/{len(files)} elapsed_min {(time.perf_counter() - t0) / 60:.1f}", flush=True)
    import pandas as pd

    out = {}
    for votes, frames in buckets.items():
        out[votes] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    return out


def _row(votes: int, pred, scored) -> dict:
    return {
        "iou_thr": FROZEN_IOU,
        "min_votes": votes,
        "vote_pool": len(tta.VIEWS) * len(tta.SCALES) * len(MODELS),
        "pq": scored.pq,
        "tp": scored.tp,
        "fp": scored.fp,
        "fn": scored.fn,
        "mean_tp_iou": scored.mean_tp_iou,
        "n_pred": int(len(pred)),
    }


def _select(rows: list[dict]) -> dict:
    best_pq = max(float(r["pq"]) for r in rows)
    band = [r for r in rows if best_pq - float(r["pq"]) <= TIE_PQ]
    chosen = dict(max(band, key=lambda r: int(r["min_votes"])))
    chosen["selection"] = (
        "highest min_votes among folds 1-4 PQ within 0.005 of the best. "
        "IoU 0.5 is carried from the single-model freeze. Vote count is not."
    )
    chosen["best_tune_pq"] = best_pq
    chosen["n_in_band"] = len(band)
    chosen["tune_is_in_sample"] = True
    return chosen


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    tune_files = sorted(fn for fn, fold in fold_of.items() if fold != 0)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    print(
        f"tune_images {len(tune_files)} fold0_images {len(val_files)} "
        f"iou {FROZEN_IOU} votes {VOTE_GRID} pool {len(tta.VIEWS) * len(tta.SCALES) * len(MODELS)}",
        flush=True,
    )
    for name, weights, cache in MODELS:
        _ensure_cache(name, weights, cache, tune_files + val_files)

    gt_tune = gt_annotations_to_df(gt_records_for_files(views, by_file, tune_files))
    tune_preds = _vote_frames(tune_files)
    tune_rows = []
    for votes in VOTE_GRID:
        scored = score_pq(gt_tune, tune_preds[votes])
        row = _row(votes, tune_preds[votes], scored)
        tune_rows.append(row)
        print("tune", json.dumps(row), flush=True)
    frozen = _select(tune_rows)
    tune_path = ARTIFACTS / "ensemble3_tta_tune_folds14.json"
    tune_path.write_text(
        json.dumps(
            {
                "grid": tune_rows,
                "frozen": frozen,
                "vote_pool": 72,
                "min_votes_8_origin": "carried from the 24-view single-model grid; re-ranked here",
                "tune_split": "grouped folds 1-4, in-sample for every seed",
                "test_labels_used": False,
            },
            indent=2,
        )
        + "\n"
    )
    print("frozen", json.dumps(frozen), flush=True)

    gt_val = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    val_preds = _vote_frames(val_files)
    chosen_votes = int(frozen["min_votes"])
    pred = val_preds[chosen_votes]
    scored = score_pq(gt_val, pred)
    pred_csv = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_ensemble3_tta_fold0_pred.csv"
    pred.to_csv(pred_csv, index=False)
    ensemble = _row(chosen_votes, pred, scored)
    ensemble["pred_csv"] = str(pred_csv)
    ensemble["delta_pq_vs_parent"] = ensemble["pq"] - tta.PARENT_PQ
    ensemble["delta_tp_vs_parent"] = ensemble["tp"] - tta.PARENT["tp"]
    ensemble["delta_pq_vs_parent_tta"] = ensemble["pq"] - PARENT_TTA["pq"]
    print("fold0", json.dumps(ensemble), flush=True)

    payload = {
        "family": "yolo11s_seg_erode_gt_2048_ensemble3_tta",
        "seeds": [20260910, 20260911, 20260912],
        "views_per_model": len(tta.VIEWS) * len(tta.SCALES),
        "vote_pool": 72,
        "iou_carried_from_single_model": FROZEN_IOU,
        "vote_grid_ranked_on": "grouped folds 1-4",
        "frozen": frozen,
        "tune_grid": tune_rows,
        "fold": 0,
        "parent_alone": tta.PARENT,
        "parent_tta_single": PARENT_TTA,
        "ensemble": ensemble,
        "pq": ensemble["pq"],
        "tp": ensemble["tp"],
        "fp": ensemble["fp"],
        "fn": ensemble["fn"],
        "mean_tp_iou": ensemble["mean_tp_iou"],
        "n_pred": ensemble["n_pred"],
        "test_labels_used": False,
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "beats_parent_fold0": bool(ensemble["pq"] > tta.PARENT_PQ),
        "beats_parent_tta_fold0": bool(ensemble["pq"] > PARENT_TTA["pq"]),
        "promote_look": bool(
            (ensemble["pq"] >= tta.PARENT_PQ + 0.015 and ensemble["tp"] > tta.PARENT["tp"])
            or ensemble["pq"] >= 0.46
        ),
    }
    text = json.dumps(payload, indent=2) + "\n"
    (ARTIFACTS / "yolo11s_seg_erode_gt_2048_ensemble3_tta_fold0_official.json").write_text(text)
    print(text, flush=True)
    print(
        f"DONE ensemble pq {ensemble['pq']} tp {ensemble['tp']} fp {ensemble['fp']} "
        f"fn {ensemble['fn']} n_pred {ensemble['n_pred']} mean_tp_iou {ensemble['mean_tp_iou']} "
        f"votes {chosen_votes}",
        flush=True,
    )


if __name__ == "__main__":
    main()
