#!/usr/bin/env python3
"""Score yolo11m erode-gt 2048 on fold 0, then frozen 8x3 TTA.

Alone: confidence 0.35, image size 2048, one view. TTA uses the parent
freeze, IoU 0.5 and min votes 8, and is not retuned. If that TTA fold-0 PQ
beats the single-model TTA at 0.4651884607358067, the m cache joins the
three yolo11s seeds. Min votes are re-picked on folds 1-4 only, then fold 0
is scored once. Never submits. Never reads test labels.
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
from filament.infer import pick_device, run_yolo_infer
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.rle import encode_rle
from gate_unet_family import live_mps_owner

spec = importlib.util.spec_from_file_location("eval_parent_tta", ROOT / "scripts" / "eval_parent_tta.py")
tta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tta)

ens_spec = importlib.util.spec_from_file_location("eval_seed_ensemble", ROOT / "scripts" / "eval_seed_ensemble.py")
ens = importlib.util.module_from_spec(ens_spec)
ens_spec.loader.exec_module(ens)

WEIGHTS = WEIGHTS_DIR / "yolo11m_seg_erode_gt_2048" / "weights" / "best.pt"
PARENT_CSV = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
ALONE_JSON = ARTIFACTS / "yolo11m_seg_erode_gt_2048_fold0.json"
TTA_JSON = ARTIFACTS / "yolo11m_seg_erode_gt_2048_tta_fold0_official.json"
ENS_JSON = ARTIFACTS / "yolo11m_plus_3seed_tta_fold0_official.json"
M_CACHE = FAST_ARTIFACTS / "tta_yolo11m_2048"
PARENT_PQ = 0.4387235694048039
SINGLE_TTA_PQ = 0.4651884607358067
CONF = 0.35
IMGSZ = 2048
FROZEN_IOU = 0.5
FROZEN_VOTES = 8
# Predeclared before any 4-model fold-0 look. IoU stays 0.5.
VOTE_GRID = (8, 16, 24, 32, 40)
TIE_PQ = 0.005
SEED_CACHES = (
    ("s20260910", FAST_ARTIFACTS / "tta_parent_2048"),
    ("s20260911", FAST_ARTIFACTS / "tta_s20260911"),
    ("s20260912", FAST_ARTIFACTS / "tta_s20260912"),
    ("m20260910", M_CACHE),
)


def _row(scored, n_pred: int) -> dict:
    return {
        "pq": scored.pq,
        "tp": scored.tp,
        "fp": scored.fp,
        "fn": scored.fn,
        "mean_tp_iou": scored.mean_tp_iou,
        "n_pred": n_pred,
    }


def _check_parent(gt_df) -> dict:
    parent = pd.read_csv(PARENT_CSV)
    first = _row(score_pq(gt_df, parent), len(parent))
    second = _row(score_pq(gt_df, parent), len(parent))
    if first != second:
        raise SystemExit(f"parent scored twice and differed: {first} vs {second}")
    if abs(first["pq"] - PARENT_PQ) > 1e-6:
        raise SystemExit(f"parent rescore drifted: {first}")
    return first


def _formula(row: dict) -> None:
    if not row["tp"]:
        return
    formula = row["mean_tp_iou"] * row["tp"] / (row["tp"] + 0.5 * row["fp"] + 0.5 * row["fn"])
    if abs(formula - row["pq"]) > 1e-9:
        raise SystemExit(f"formula check failed: {formula} vs {row['pq']}")


def score_alone(gt_df, val_files: list[str], parent: dict) -> dict:
    if ALONE_JSON.exists():
        saved = json.loads(ALONE_JSON.read_text())
        print("alone_cached", json.dumps(saved["m_alone"]), flush=True)
        return saved
    from ultralytics import YOLO

    device = pick_device("auto")
    model = YOLO(str(WEIGHTS))
    frames = []
    t0 = time.perf_counter()
    for i, name in enumerate(val_files, start=1):
        rles = run_yolo_infer(model, TRAIN_IMG_DIR / name, conf=CONF, imgsz=IMGSZ, device=device)
        frames.append(pred_to_df(Path(name).stem, rles))
        if i == 1 or i % 25 == 0:
            print(f"alone {i}/{len(val_files)} elapsed_min {(time.perf_counter() - t0) / 60:.1f}", flush=True)
    del model
    pred = pd.concat(frames, ignore_index=True)
    pred_csv = FAST_ARTIFACTS / "yolo11m_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
    pred.to_csv(pred_csv, index=False)
    scored = _row(score_pq(gt_df, pred), len(pred))
    _formula(scored)
    payload = {
        "split": "grouped year x count fold 0",
        "seed": SEED,
        "weights": str(WEIGHTS),
        "conf": CONF,
        "imgsz": IMGSZ,
        "n_val": len(val_files),
        "parent": parent,
        "m_alone": scored,
        "pred_csv": str(pred_csv),
        "delta_pq_vs_parent": scored["pq"] - PARENT_PQ,
        "test_labels_used": False,
        "kaggle_submit_called": False,
        "do_not_submit": True,
    }
    ALONE_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print("ALONE", json.dumps(scored), flush=True)
    return payload


def _cache_m(files: list[str]) -> None:
    M_CACHE.mkdir(parents=True, exist_ok=True)
    tta.CACHE = M_CACHE
    missing = [fn for fn in files if not tta._cache_path(fn).exists()]
    print(f"m_tta missing {len(missing)}/{len(files)}", flush=True)
    if not missing:
        return
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))
    t0 = time.perf_counter()
    for i, file_name in enumerate(missing, start=1):
        tta._load_or_infer(model, file_name)
        if i == 1 or i % 10 == 0 or i == len(missing):
            print(
                f"m_tta inferred {i}/{len(missing)} elapsed_min {(time.perf_counter() - t0) / 60:.1f}",
                flush=True,
            )
    del model


def score_tta_fold0(gt_df, val_files: list[str]) -> dict:
    if TTA_JSON.exists():
        saved = json.loads(TTA_JSON.read_text())
        print("tta_cached", json.dumps(saved["m_tta"]), flush=True)
        return saved
    _cache_m(val_files)
    tta.CACHE = M_CACHE
    t0 = time.perf_counter()
    frames = []
    for i, file_name in enumerate(val_files, start=1):
        payload = json.loads(tta._cache_path(file_name).read_text())
        masks = tta.fuse_masks(tta._decode_all(payload), FROZEN_IOU, FROZEN_VOTES)
        frames.append(pred_to_df(Path(file_name).stem, [encode_rle(m) for m in masks]))
        if i == 1 or i % 25 == 0:
            print(f"m_tta fused {i}/{len(val_files)} elapsed_min {(time.perf_counter() - t0) / 60:.1f}", flush=True)
    pred = pd.concat(frames, ignore_index=True)
    pred_csv = FAST_ARTIFACTS / "yolo11m_seg_erode_gt_2048_tta_fold0_pred.csv"
    pred.to_csv(pred_csv, index=False)
    scored = _row(score_pq(gt_df, pred), len(pred))
    _formula(scored)
    payload = {
        "family": "yolo11m_seg_erode_gt_2048_tta",
        "weights": str(WEIGHTS),
        "frozen": {"iou_thr": FROZEN_IOU, "min_votes": FROZEN_VOTES, "retuned_on_fold0": False},
        "fold": 0,
        "m_tta": scored,
        "single_model_tta_pq": SINGLE_TTA_PQ,
        "delta_pq_vs_single_tta": scored["pq"] - SINGLE_TTA_PQ,
        "beats_single_tta": bool(scored["pq"] > SINGLE_TTA_PQ),
        "pred_csv": str(pred_csv),
        "test_labels_used": False,
        "kaggle_submit_called": False,
        "do_not_submit": True,
    }
    TTA_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print("TTA", json.dumps(scored), flush=True)
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


def _read_four(file_name: str) -> dict[str, list[str]]:
    payload: dict[str, list[str]] = {}
    stem = Path(file_name).stem
    for name, cache in SEED_CACHES:
        one = json.loads((cache / f"{stem}.json").read_text())
        for key, rles in one.items():
            payload[f"{name}_{key}"] = rles
    return payload


def _select(rows: list[dict]) -> dict:
    best_pq = max(float(r["pq"]) for r in rows)
    band = [r for r in rows if best_pq - float(r["pq"]) <= TIE_PQ]
    chosen = dict(max(band, key=lambda r: int(r["min_votes"])))
    chosen["selection"] = (
        "highest min_votes among folds 1-4 PQ within 0.005 of the best. "
        "IoU 0.5 is carried. Vote count is re-picked on this 96-voter pool."
    )
    chosen["best_tune_pq"] = best_pq
    chosen["n_in_band"] = len(band)
    chosen["tune_is_in_sample"] = True
    return chosen


def score_ensemble(views, by_file, tune_files: list[str], val_files: list[str]) -> None:
    if ENS_JSON.exists():
        print("ensemble_cached", ENS_JSON.read_text(), flush=True)
        return
    _cache_m(tune_files + val_files)
    for name, cache in SEED_CACHES:
        missing = [fn for fn in tune_files + val_files if not (cache / f"{Path(fn).stem}.json").exists()]
        if missing:
            raise SystemExit(f"{name} missing {len(missing)} caches, first {missing[0]}")
    spot_masks = tta._decode_all(_read_four(tune_files[0]))
    if _groups_fast(spot_masks, FROZEN_IOU) != ens._groups(spot_masks, FROZEN_IOU):
        raise SystemExit("fast groups diverged from eval_seed_ensemble._groups")
    print("groups_match", tune_files[0], flush=True)

    def frames_for(files: list[str]) -> dict[int, pd.DataFrame]:
        buckets = {votes: [] for votes in VOTE_GRID}
        t0 = time.perf_counter()
        for i, file_name in enumerate(files, start=1):
            masks = tta._decode_all(_read_four(file_name))
            groups = _groups_fast(masks, FROZEN_IOU)
            for votes in VOTE_GRID:
                inst = ens._instances_for_votes(masks, groups, votes)
                buckets[votes].append(pred_to_df(Path(file_name).stem, [encode_rle(m) for m in inst]))
            if i == 1 or i % 25 == 0:
                print(f"pool fused {i}/{len(files)} elapsed_min {(time.perf_counter() - t0) / 60:.1f}", flush=True)
        out = {}
        for votes, frames in buckets.items():
            out[votes] = pd.concat(frames, ignore_index=True)
        return out

    gt_tune = gt_annotations_to_df(gt_records_for_files(views, by_file, tune_files))
    tune_preds = frames_for(tune_files)
    tune_rows = []
    for votes in VOTE_GRID:
        row = {"iou_thr": FROZEN_IOU, "min_votes": votes, "vote_pool": 96}
        row.update(_row(score_pq(gt_tune, tune_preds[votes]), len(tune_preds[votes])))
        tune_rows.append(row)
        print("tune", json.dumps(row), flush=True)
    frozen = _select(tune_rows)
    (ARTIFACTS / "yolo11m_plus_3seed_tta_tune_folds14.json").write_text(
        json.dumps({"grid": tune_rows, "frozen": frozen, "test_labels_used": False}, indent=2) + "\n"
    )
    print("frozen", json.dumps(frozen), flush=True)

    gt_val = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    val_preds = frames_for(val_files)
    chosen = int(frozen["min_votes"])
    pred = val_preds[chosen]
    scored = _row(score_pq(gt_val, pred), len(pred))
    _formula(scored)
    pred_csv = FAST_ARTIFACTS / "yolo11m_plus_3seed_tta_fold0_pred.csv"
    pred.to_csv(pred_csv, index=False)
    payload = {
        "family": "yolo11m_plus_3seed_tta",
        "models": [name for name, _ in SEED_CACHES],
        "vote_pool": 96,
        "vote_grid_ranked_on": "grouped folds 1-4",
        "frozen": frozen,
        "tune_grid": tune_rows,
        "fold": 0,
        "ensemble": scored,
        "pred_csv": str(pred_csv),
        "single_model_tta_pq": SINGLE_TTA_PQ,
        "three_seed_tta_pq": 0.4741684688395591,
        "delta_pq_vs_single_tta": scored["pq"] - SINGLE_TTA_PQ,
        "delta_pq_vs_three_seed": scored["pq"] - 0.4741684688395591,
        "test_labels_used": False,
        "kaggle_submit_called": False,
        "do_not_submit": True,
    }
    ENS_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print("FOLD0", json.dumps(scored), flush=True)


def main() -> None:
    owner = live_mps_owner()
    if owner and "eval_yolo11m_erode_2048.py" not in owner:
        raise SystemExit(f"refusing to score while {owner} owns MPS")
    if not WEIGHTS.is_file():
        raise SystemExit(f"missing weights: {WEIGHTS}")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(name for name, fold in fold_of.items() if fold == 0)
    tune_files = sorted(name for name, fold in fold_of.items() if fold != 0)
    if len(val_files) != 142 or len(tune_files) != 565:
        raise SystemExit(f"unexpected split val {len(val_files)} tune {len(tune_files)}")
    gt_val = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    parent = _check_parent(gt_val)
    score_alone(gt_val, val_files, parent)
    tta_payload = score_tta_fold0(gt_val, val_files)
    if not tta_payload["beats_single_tta"]:
        print(
            f"DONE stop m_tta {tta_payload['m_tta']['pq']} does not beat {SINGLE_TTA_PQ}",
            flush=True,
        )
        return
    print(f"GATE m_tta {tta_payload['m_tta']['pq']} beats {SINGLE_TTA_PQ}", flush=True)
    score_ensemble(views, by_file, tune_files, val_files)
    print("DONE ensemble", flush=True)


if __name__ == "__main__":
    main()
