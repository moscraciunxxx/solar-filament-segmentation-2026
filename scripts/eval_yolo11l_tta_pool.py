#!/usr/bin/env python3
"""Frozen 8x3 TTA for the existing yolo11l erode-gt 2048 weights.

The 40-epoch batch-1 run is the parent recipe and is not retrained.
Fold 0 at conf 0.35 is already scored and is not scored again.
TTA is IoU 0.5, min votes 8, not retuned. If that fold-0 PQ beats
0.4651884607358067, the l votes join the 3 yolo11s seeds (96 voters).
Min votes are re-picked on folds 1-4 from 24/32/48 only, then fold 0
is scored once. Kept only if that PQ beats 0.4741684688395591.
Never submits. Never reads test labels.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, WEIGHTS_DIR
from filament.rle import encode_rle

spec = importlib.util.spec_from_file_location("eval_parent_tta", ROOT / "scripts" / "eval_parent_tta.py")
tta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tta)

pool_spec = importlib.util.spec_from_file_location("eval_pool_3s_plus_m", ROOT / "scripts" / "eval_pool_3s_plus_m.py")
pool = importlib.util.module_from_spec(pool_spec)
pool_spec.loader.exec_module(pool)

SINGLE_TTA_PQ = 0.4651884607358067
THREE_SEED_PQ = 0.4741684688395591
FROZEN_IOU = 0.5
FROZEN_VOTES = 8
VOTE_GRID = (24, 32, 48)
TIE_PQ = 0.005
L_WEIGHTS = WEIGHTS_DIR / "yolo11l_seg_erode_gt_2048" / "weights" / "best.pt"
L_CACHE = FAST_ARTIFACTS / "tta_yolo11l_2048"
TTA_JSON = ARTIFACTS / "yolo11l_seg_erode_gt_2048_tta_fold0_official.json"
TUNE_JSON = ARTIFACTS / "pool_3s_plus_l_tune_folds14.json"
FOLD0_JSON = ARTIFACTS / "pool_3s_plus_l_fold0.json"
CACHES = (
    ("s20260910", FAST_ARTIFACTS / "tta_parent_2048"),
    ("s20260911", FAST_ARTIFACTS / "tta_s20260911"),
    ("s20260912", FAST_ARTIFACTS / "tta_s20260912"),
    ("l20260910", L_CACHE),
)


def _cache_file(cache: Path, file_name: str) -> Path:
    return cache / f"{Path(file_name).stem}.json"


def _ensure_l(files: list[str]) -> None:
    if not L_WEIGHTS.is_file():
        raise SystemExit(f"missing {L_WEIGHTS}")
    L_CACHE.mkdir(parents=True, exist_ok=True)
    tta.CACHE = L_CACHE
    missing = [fn for fn in files if not _cache_file(L_CACHE, fn).exists()]
    print(f"l_tta missing {len(missing)}/{len(files)}", flush=True)
    if not missing:
        return
    from ultralytics import YOLO

    model = YOLO(str(L_WEIGHTS))
    t0 = time.perf_counter()
    for i, file_name in enumerate(missing, start=1):
        tta._load_or_infer(model, file_name)
        if i == 1 or i % 10 == 0 or i == len(missing):
            print(
                f"l_tta inferred {i}/{len(missing)} elapsed_min {(time.perf_counter() - t0) / 60:.1f}",
                flush=True,
            )
    del model
    import torch

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _read(file_name: str) -> dict[str, list[str]]:
    payload: dict[str, list[str]] = {}
    stem = Path(file_name).stem
    for name, cache in CACHES:
        path = _cache_file(cache, file_name)
        if not path.exists():
            raise SystemExit(f"missing cache {path}")
        one = json.loads(path.read_text())
        for key, rles in one.items():
            payload[f"{name}_{key}"] = rles
    return payload


def _row(votes: int, n_pool: int, pred, scored) -> dict:
    return {
        "iou_thr": FROZEN_IOU,
        "min_votes": votes,
        "vote_pool": n_pool,
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
        "Grid is 24/32/48. IoU 0.5 is carried."
    )
    chosen["best_tune_pq"] = best_pq
    chosen["n_in_band"] = len(band)
    chosen["tune_is_in_sample"] = True
    return chosen


def score_tta(gt_val, val_files: list[str]) -> dict:
    if TTA_JSON.exists():
        saved = json.loads(TTA_JSON.read_text())
        print("tta_cached", json.dumps(saved["l_tta"]), flush=True)
        return saved
    _ensure_l(val_files)
    tta.CACHE = L_CACHE
    t0 = time.perf_counter()
    frames = []
    for i, file_name in enumerate(val_files, start=1):
        payload = json.loads(tta._cache_path(file_name).read_text())
        masks = tta.fuse_masks(tta._decode_all(payload), FROZEN_IOU, FROZEN_VOTES)
        frames.append(pred_to_df(Path(file_name).stem, [encode_rle(m) for m in masks]))
        if i == 1 or i % 25 == 0:
            print(f"l_tta fused {i}/{len(val_files)} elapsed_min {(time.perf_counter() - t0) / 60:.1f}", flush=True)
    pred = pd.concat(frames, ignore_index=True)
    pred_csv = FAST_ARTIFACTS / "yolo11l_seg_erode_gt_2048_tta_fold0_pred.csv"
    pred.to_csv(pred_csv, index=False)
    scored = score_pq(gt_val, pred)
    row = _row(FROZEN_VOTES, 24, pred, scored)
    payload = {
        "family": "yolo11l_seg_erode_gt_2048_tta",
        "weights": str(L_WEIGHTS),
        "retrained": False,
        "alone_fold0_pq": 0.4195790726186592,
        "frozen": {"iou_thr": FROZEN_IOU, "min_votes": FROZEN_VOTES, "retuned_on_fold0": False},
        "fold": 0,
        "l_tta": row,
        "single_model_tta_pq": SINGLE_TTA_PQ,
        "delta_pq_vs_single_tta": row["pq"] - SINGLE_TTA_PQ,
        "beats_single_tta": bool(row["pq"] > SINGLE_TTA_PQ),
        "pred_csv": str(pred_csv),
        "test_labels_used": False,
        "kaggle_submit_called": False,
        "do_not_submit": True,
    }
    TTA_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print("TTA", json.dumps(row), flush=True)
    return payload


def _frames(files: list[str]) -> dict[int, pd.DataFrame]:
    buckets = {votes: [] for votes in VOTE_GRID}
    t0 = time.perf_counter()
    for i, file_name in enumerate(files, start=1):
        masks = tta._decode_all(_read(file_name))
        groups = pool._groups_fast(masks, FROZEN_IOU)
        for votes in VOTE_GRID:
            inst = pool.ens._instances_for_votes(masks, groups, votes)
            buckets[votes].append(pred_to_df(Path(file_name).stem, [encode_rle(m) for m in inst]))
        if i == 1 or i % 25 == 0:
            print(f"fused {i}/{len(files)} elapsed_min {(time.perf_counter() - t0) / 60:.1f}", flush=True)
    return {votes: pd.concat(frames, ignore_index=True) for votes, frames in buckets.items()}


def score_pool(views, by_file, tune_files: list[str], val_files: list[str]) -> None:
    if FOLD0_JSON.exists():
        print("pool_cached", FOLD0_JSON.read_text(), flush=True)
        return
    for name, cache in CACHES[:3]:
        missing = [fn for fn in tune_files + val_files if not _cache_file(cache, fn).exists()]
        if missing:
            raise SystemExit(f"{name} missing {len(missing)} cached views")
    _ensure_l(tune_files + val_files)
    spot = tta._decode_all(_read(tune_files[0]))
    if pool._groups_fast(spot, FROZEN_IOU) != pool.ens._groups(spot, FROZEN_IOU):
        raise SystemExit("fast groups diverged")
    print("groups_match", tune_files[0], flush=True)
    gt_tune = gt_annotations_to_df(gt_records_for_files(views, by_file, tune_files))
    tune_preds = _frames(tune_files)
    tune_rows = []
    for votes in VOTE_GRID:
        row = _row(votes, 96, tune_preds[votes], score_pq(gt_tune, tune_preds[votes]))
        tune_rows.append(row)
        print("tune", json.dumps(row), flush=True)
    frozen = _select(tune_rows)
    TUNE_JSON.write_text(json.dumps({"grid": tune_rows, "frozen": frozen, "test_labels_used": False}, indent=2) + "\n")
    print("frozen", json.dumps(frozen), flush=True)
    gt_val = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    val_preds = _frames(val_files)
    chosen = int(frozen["min_votes"])
    pred = val_preds[chosen]
    scored = _row(chosen, 96, pred, score_pq(gt_val, pred))
    pred_csv = FAST_ARTIFACTS / "pool_3s_plus_l_fold0_pred.csv"
    pred.to_csv(pred_csv, index=False)
    kept = bool(scored["pq"] > THREE_SEED_PQ)
    payload = {
        "family": "pool_3s_plus_l",
        "vote_pool": 96,
        "vote_grid": list(VOTE_GRID),
        "frozen": frozen,
        "tune_grid": tune_rows,
        "fold": 0,
        "ensemble": scored,
        "three_seed_pq": THREE_SEED_PQ,
        "delta_pq_vs_three_seed": scored["pq"] - THREE_SEED_PQ,
        "kept": kept,
        "pred_csv": str(pred_csv),
        "test_labels_used": False,
        "kaggle_submit_called": False,
        "do_not_submit": True,
    }
    FOLD0_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print("FOLD0", json.dumps(scored), flush=True)
    print(f"DONE pool pq {scored['pq']} votes {chosen} kept {kept}", flush=True)


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    tune_files = sorted(fn for fn, fold in fold_of.items() if fold != 0)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    if len(tune_files) != 565 or len(val_files) != 142:
        raise SystemExit(f"unexpected split tune {len(tune_files)} val {len(val_files)}")
    gt_val = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    tta_payload = score_tta(gt_val, val_files)
    if not tta_payload["beats_single_tta"]:
        print(
            f"DONE stop l_tta {tta_payload['l_tta']['pq']} does not beat {SINGLE_TTA_PQ}",
            flush=True,
        )
        return
    print(f"GATE l_tta {tta_payload['l_tta']['pq']} beats {SINGLE_TTA_PQ}", flush=True)
    score_pool(views, by_file, tune_files, val_files)


if __name__ == "__main__":
    main()
