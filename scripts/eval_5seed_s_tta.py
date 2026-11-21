#!/usr/bin/env python3
"""Five yolo11s seeds, 8x3 TTA, 120 voters.

Seeds 20260910, 20260911, 20260912, 20260913, 20260914.
IoU stays 0.5. Min votes are re-picked on folds 1-4 from (30, 40, 60),
the same 1/4, 1/3, and 1/2 fractions as the 96-voter grid 24/32/48.
Fold 0 is scored once against the 3-seed PQ 0.4741684688395591.
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

THREE_SEED_PQ = 0.4741684688395591
FROZEN_IOU = 0.5
VOTE_GRID = (30, 40, 60)
TIE_PQ = 0.005
TUNE_JSON = ARTIFACTS / "pool_5seed_s_tune_folds14.json"
FOLD0_JSON = ARTIFACTS / "pool_5seed_s_fold0.json"
MODELS = (
    ("s20260910", WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048" / "weights" / "best.pt", FAST_ARTIFACTS / "tta_parent_2048"),
    ("s20260911", WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048_s20260911" / "weights" / "best.pt", FAST_ARTIFACTS / "tta_s20260911"),
    ("s20260912", WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048_s20260912" / "weights" / "best.pt", FAST_ARTIFACTS / "tta_s20260912"),
    ("s20260913", WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048_s20260913" / "weights" / "best.pt", FAST_ARTIFACTS / "tta_s20260913"),
    ("s20260914", WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048_s20260914" / "weights" / "best.pt", FAST_ARTIFACTS / "tta_s20260914"),
)


def _ensure(files: list[str]) -> None:
    from ultralytics import YOLO

    for name, weights, cache in MODELS:
        if not weights.is_file():
            raise SystemExit(f"missing {weights}")
        cache.mkdir(parents=True, exist_ok=True)
        tta.CACHE = cache
        missing = [fn for fn in files if not tta._cache_path(fn).exists()]
        print(f"{name} missing {len(missing)}/{len(files)}", flush=True)
        if not missing:
            continue
        model = YOLO(str(weights))
        t0 = time.perf_counter()
        for i, file_name in enumerate(missing, start=1):
            tta._load_or_infer(model, file_name)
            if i == 1 or i % 10 == 0 or i == len(missing):
                print(
                    f"{name} inferred {i}/{len(missing)} elapsed_min {(time.perf_counter() - t0) / 60:.1f}",
                    flush=True,
                )
        del model


def _read(file_name: str) -> dict[str, list[str]]:
    payload: dict[str, list[str]] = {}
    stem = Path(file_name).stem
    for name, _, cache in MODELS:
        one = json.loads((cache / f"{stem}.json").read_text())
        for key, rles in one.items():
            payload[f"{name}_{key}"] = rles
    return payload


def _row(votes: int, pred, scored) -> dict:
    return {
        "iou_thr": FROZEN_IOU,
        "min_votes": votes,
        "vote_pool": 120,
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
        "Grid is 30/40/60, the 1/4, 1/3, and 1/2 fractions of 120 voters. IoU 0.5 is carried."
    )
    chosen["best_tune_pq"] = best_pq
    chosen["n_in_band"] = len(band)
    chosen["tune_is_in_sample"] = True
    return chosen


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


def main() -> None:
    if FOLD0_JSON.exists():
        raise SystemExit(f"fold 0 already scored: {FOLD0_JSON}")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    tune_files = sorted(fn for fn, fold in fold_of.items() if fold != 0)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    if len(tune_files) != 565 or len(val_files) != 142:
        raise SystemExit(f"unexpected split tune {len(tune_files)} val {len(val_files)}")
    _ensure(tune_files + val_files)
    import torch

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    if TUNE_JSON.exists():
        saved = json.loads(TUNE_JSON.read_text())
        tune_rows = saved["grid"]
        frozen = saved["frozen"]
        print("tune_cached", json.dumps(frozen), flush=True)
    else:
        gt_tune = gt_annotations_to_df(gt_records_for_files(views, by_file, tune_files))
        tune_preds = _frames(tune_files)
        tune_rows = []
        for votes in VOTE_GRID:
            row = _row(votes, tune_preds[votes], score_pq(gt_tune, tune_preds[votes]))
            tune_rows.append(row)
            print("tune", json.dumps(row), flush=True)
        frozen = _select(tune_rows)
        TUNE_JSON.write_text(
            json.dumps({"grid": tune_rows, "frozen": frozen, "vote_pool": 120, "test_labels_used": False}, indent=2)
            + "\n"
        )
        print("frozen", json.dumps(frozen), flush=True)

    gt_val = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    val_preds = _frames(val_files)
    chosen = int(frozen["min_votes"])
    pred = val_preds[chosen]
    scored = _row(chosen, pred, score_pq(gt_val, pred))
    pred_csv = FAST_ARTIFACTS / "pool_5seed_s_fold0_pred.csv"
    pred.to_csv(pred_csv, index=False)
    payload = {
        "family": "pool_5seed_s",
        "seeds": [20260910, 20260911, 20260912, 20260913, 20260914],
        "vote_pool": 120,
        "vote_grid": list(VOTE_GRID),
        "frozen": frozen,
        "tune_grid": tune_rows,
        "fold": 0,
        "ensemble": scored,
        "three_seed_pq": THREE_SEED_PQ,
        "delta_pq_vs_three_seed": scored["pq"] - THREE_SEED_PQ,
        "beats_three_seed": bool(scored["pq"] > THREE_SEED_PQ),
        "pred_csv": str(pred_csv),
        "test_labels_used": False,
        "kaggle_submit_called": False,
        "do_not_submit": True,
    }
    FOLD0_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print("FOLD0", json.dumps(scored), flush=True)
    print(
        f"DONE 5seed pq {scored['pq']} votes {chosen} beats {payload['beats_three_seed']} "
        f"delta {scored['pq'] - THREE_SEED_PQ}",
        flush=True,
    )


if __name__ == "__main__":
    main()
