#!/usr/bin/env python3
"""Add yolo11m TTA votes to the 3-seed yolo11s pool. 96 voters.

IoU stays 0.5. Min votes are re-picked on folds 1-4 from (24, 32, 48) only.
Fold 0 is scored once. Kept only if that PQ beats the 3-seed score
0.4741684688395591. Reuses cached views. Never submits. Never reads test labels.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
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

ens_spec = importlib.util.spec_from_file_location("eval_seed_ensemble", ROOT / "scripts" / "eval_seed_ensemble.py")
ens = importlib.util.module_from_spec(ens_spec)
ens_spec.loader.exec_module(ens)

THREE_SEED_PQ = 0.4741684688395591
FROZEN_IOU = 0.5
VOTE_GRID = (24, 32, 48)
TIE_PQ = 0.005
M_WEIGHTS = WEIGHTS_DIR / "yolo11m_seg_erode_gt_2048" / "weights" / "best.pt"
M_CACHE = FAST_ARTIFACTS / "tta_yolo11m_2048"
MPS_FLAG = ARTIFACTS / "pool_3s_plus_m_mps_free"
TUNE_JSON = ARTIFACTS / "pool_3s_plus_m_tune_folds14.json"
FOLD0_JSON = ARTIFACTS / "pool_3s_plus_m_fold0.json"
CACHES = (
    ("s20260910", FAST_ARTIFACTS / "tta_parent_2048"),
    ("s20260911", FAST_ARTIFACTS / "tta_s20260911"),
    ("s20260912", FAST_ARTIFACTS / "tta_s20260912"),
    ("m20260910", M_CACHE),
)


def _cache_file(cache: Path, file_name: str) -> Path:
    return cache / f"{Path(file_name).stem}.json"


def _ensure_m(files: list[str]) -> None:
    if not M_WEIGHTS.is_file():
        raise SystemExit(f"missing {M_WEIGHTS}")
    M_CACHE.mkdir(parents=True, exist_ok=True)
    tta.CACHE = M_CACHE
    missing = [fn for fn in files if not _cache_file(M_CACHE, fn).exists()]
    print(f"m_tta missing {len(missing)}/{len(files)}", flush=True)
    if not missing:
        return
    from ultralytics import YOLO

    model = YOLO(str(M_WEIGHTS))
    t0 = time.perf_counter()
    for i, file_name in enumerate(missing, start=1):
        tta._load_or_infer(model, file_name)
        if i == 1 or i % 10 == 0 or i == len(missing):
            print(
                f"m_tta inferred {i}/{len(missing)} elapsed_min {(time.perf_counter() - t0) / 60:.1f}",
                flush=True,
            )
    del model
    import torch

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _release_mps() -> None:
    MPS_FLAG.write_text("yolo11m TTA cache complete; MPS released for seed training\n")
    print("MPS_FREE", flush=True)


def _read_four(file_name: str) -> dict[str, list[str]]:
    payload: dict[str, list[str]] = {}
    stem = Path(file_name).stem
    for name, cache in CACHES:
        path = cache / f"{stem}.json"
        if not path.exists():
            raise SystemExit(f"missing cache {path}")
        one = json.loads(path.read_text())
        for key, rles in one.items():
            payload[f"{name}_{key}"] = rles
    return payload


def _groups_fast(masks: list, iou_thr: float) -> list[list[int]]:
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
        mi = masks[i]
        ai = areas[i]
        for j in order:
            if used[j] or boxes[j] is None:
                continue
            bx0, by0, bx1, by1 = boxes[j]
            if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
                continue
            ix0, iy0 = max(ax0, bx0), max(ay0, by0)
            ix1, iy1 = min(ax1, bx1) + 1, min(ay1, by1) + 1
            inter = int(np.logical_and(mi[iy0:iy1, ix0:ix1], masks[j][iy0:iy1, ix0:ix1]).sum())
            union = ai + areas[j] - inter
            if union and (inter / union) >= iou_thr:
                group.append(j)
                used[j] = True
        groups.append(group)
    return groups


def _row(votes: int, pred, scored) -> dict:
    return {
        "iou_thr": FROZEN_IOU,
        "min_votes": votes,
        "vote_pool": 96,
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
        "Grid is exactly 24/32/48. IoU 0.5 is carried."
    )
    chosen["best_tune_pq"] = best_pq
    chosen["n_in_band"] = len(band)
    chosen["tune_is_in_sample"] = True
    return chosen


def _frames(files: list[str]) -> dict[int, pd.DataFrame]:
    buckets = {votes: [] for votes in VOTE_GRID}
    t0 = time.perf_counter()
    for i, file_name in enumerate(files, start=1):
        masks = tta._decode_all(_read_four(file_name))
        groups = _groups_fast(masks, FROZEN_IOU)
        for votes in VOTE_GRID:
            inst = ens._instances_for_votes(masks, groups, votes)
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
    for name, cache in CACHES[:3]:
        missing = [fn for fn in tune_files + val_files if not _cache_file(cache, fn).exists()]
        if missing:
            raise SystemExit(f"{name} missing {len(missing)} cached views")
    _ensure_m(tune_files + val_files)
    _release_mps()

    spot = tta._decode_all(_read_four(tune_files[0]))
    if _groups_fast(spot, FROZEN_IOU) != ens._groups(spot, FROZEN_IOU):
        raise SystemExit("fast groups diverged from the 3-seed grouper")
    print("groups_match", tune_files[0], flush=True)

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
            json.dumps(
                {
                    "grid": tune_rows,
                    "frozen": frozen,
                    "vote_pool": 96,
                    "tune_split": "grouped folds 1-4",
                    "test_labels_used": False,
                },
                indent=2,
            )
            + "\n"
        )
        print("frozen", json.dumps(frozen), flush=True)

    gt_val = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    val_preds = _frames(val_files)
    chosen = int(frozen["min_votes"])
    pred = val_preds[chosen]
    scored = _row(chosen, pred, score_pq(gt_val, pred))
    pred_csv = FAST_ARTIFACTS / "pool_3s_plus_m_fold0_pred.csv"
    pred.to_csv(pred_csv, index=False)
    kept = bool(scored["pq"] > THREE_SEED_PQ)
    payload = {
        "family": "pool_3s_plus_m",
        "models": [name for name, _ in CACHES],
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
    print(
        f"DONE pool pq {scored['pq']} votes {chosen} kept {kept} "
        f"delta {scored['pq'] - THREE_SEED_PQ}",
        flush=True,
    )


if __name__ == "__main__":
    main()
