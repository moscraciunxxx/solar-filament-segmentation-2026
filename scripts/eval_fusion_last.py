#!/usr/bin/env python3
"""One fusion test on the frozen 3-seed 72-voter cache.

The cache stores RLEs only, so a confidence-weighted vote is not run.
Two score-free variants are ranked on folds 1-4. Fold 0 is scored once,
for the winner only.

higher_iou: champion grouper at IoU 0.7, min votes still 24.
soft_nms: area order, Gaussian decay exp(-IoU^2 / 0.5). A mask is blended
into the larger instance and removed when decay < 0.5. Otherwise it can
start its own instance. This is not model-confidence soft-NMS.

Kept only if fold-0 PQ beats 0.4741684688395591 and the hit GT ids differ
from the 3-seed champion. Never submits. Never reads test labels.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.metric import gt_annotations_to_df, overlap_matrices, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED
from filament.rle import encode_rle, rles_to_layers

spec = importlib.util.spec_from_file_location("eval_parent_tta", ROOT / "scripts" / "eval_parent_tta.py")
tta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tta)

pool_spec = importlib.util.spec_from_file_location("eval_pool_3s_plus_m", ROOT / "scripts" / "eval_pool_3s_plus_m.py")
pool = importlib.util.module_from_spec(pool_spec)
pool_spec.loader.exec_module(pool)

CHAMPION_PQ = 0.4741684688395591
CHAMPION_CSV = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_ensemble3_tta_fold0_pred.csv"
VOTES = 24
SIGMA = 0.5
DECAY_CUT = 0.5
TIE_PQ = 0.005
OUT = ARTIFACTS / "fusion_last_fold0.json"
TUNE_JSON = ARTIFACTS / "fusion_last_tune_folds14.json"
TUNE_DIR = FAST_ARTIFACTS / "fusion_last_tune"
FOLD_DIR = FAST_ARTIFACTS / "fusion_last_fold0"
CACHES = (
    ("s20260910", FAST_ARTIFACTS / "tta_parent_2048"),
    ("s20260911", FAST_ARTIFACTS / "tta_s20260911"),
    ("s20260912", FAST_ARTIFACTS / "tta_s20260912"),
)


def _read(file_name: str) -> dict[str, list[str]]:
    payload: dict[str, list[str]] = {}
    stem = Path(file_name).stem
    for name, cache in CACHES:
        one = json.loads((cache / f"{stem}.json").read_text())
        for key, rles in one.items():
            payload[f"{name}_{key}"] = rles
    return payload


def _iou(masks, boxes, areas, i: int, j: int) -> float:
    ax0, ay0, ax1, ay1 = boxes[i]
    bx0, by0, bx1, by1 = boxes[j]
    if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
        return 0.0
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1) + 1, min(ay1, by1) + 1
    inter = int(np.logical_and(masks[i][iy0:iy1, ix0:ix1], masks[j][iy0:iy1, ix0:ix1]).sum())
    union = areas[i] + areas[j] - inter
    return inter / union if union else 0.0


def _soft_nms(masks: list[np.ndarray]):
    from filament.infer import yolo_masks_to_instances

    if not masks:
        return []
    boxes = [tta._box(m) for m in masks]
    areas = [int(m.sum()) for m in masks]
    order = sorted(range(len(masks)), key=lambda i: areas[i], reverse=True)
    used = [False] * len(masks)
    kept = []
    for i in order:
        if used[i] or boxes[i] is None:
            continue
        used[i] = True
        members = [(i, 1.0)]
        for j in order:
            if used[j] or boxes[j] is None:
                continue
            overlap = _iou(masks, boxes, areas, i, j)
            if overlap <= 0:
                continue
            decay = math.exp(-(overlap * overlap) / SIGMA)
            if decay < DECAY_CUT:
                members.append((j, decay))
                used[j] = True
        vote = np.zeros(masks[i].shape, dtype=np.float32)
        weight = 0.0
        for k, w in members:
            vote += masks[k] * w
            weight += w
        merged = (vote >= 0.5 * weight).astype(np.uint8)
        if int(merged.sum()) >= 96:
            kept.append(merged)
    return yolo_masks_to_instances(kept, merge_gap=0)


def _variants(file_name: str, names: tuple[str, ...]) -> dict[str, list[str]]:
    masks = tta._decode_all(_read(file_name))
    out: dict[str, list[str]] = {}
    if "higher_iou_0.7" in names:
        groups = pool._groups_fast(masks, 0.7)
        higher = pool.ens._instances_for_votes(masks, groups, VOTES)
        out["higher_iou_0.7"] = [encode_rle(m) for m in higher]
    if "soft_nms" in names:
        out["soft_nms"] = [encode_rle(m) for m in _soft_nms(masks)]
    return out


def _part_path(directory: Path, file_name: str) -> Path:
    return directory / f"{Path(file_name).stem}.json"


def _load_part(directory: Path, file_name: str, names: tuple[str, ...]) -> dict[str, list[str]] | None:
    path = _part_path(directory, file_name)
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    if any(name not in saved for name in names):
        return None
    return {name: saved[name] for name in names}


def _write_part(directory: Path, file_name: str, payload: dict[str, list[str]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = _part_path(directory, file_name)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def _frames(files: list[str], names: tuple[str, ...], directory: Path) -> dict[str, pd.DataFrame]:
    buckets = {name: [] for name in names}
    t0 = time.perf_counter()
    reused = 0
    for i, file_name in enumerate(files, start=1):
        one = _load_part(directory, file_name, names)
        if one is None:
            one = _variants(file_name, names)
            _write_part(directory, file_name, one)
        else:
            reused += 1
        stem = Path(file_name).stem
        for name in names:
            buckets[name].append(pred_to_df(stem, one[name]))
        if i == 1 or i % 25 == 0:
            print(
                f"fused {i}/{len(files)} reused {reused} elapsed_min {(time.perf_counter() - t0) / 60:.1f}",
                flush=True,
            )
    print(f"fused_done {len(files)} reused {reused}", flush=True)
    return {name: pd.concat(frames, ignore_index=True) for name, frames in buckets.items()}


def _row(name: str, pred, scored) -> dict:
    return {
        "variant": name,
        "pq": scored.pq,
        "tp": scored.tp,
        "fp": scored.fp,
        "fn": scored.fn,
        "mean_tp_iou": scored.mean_tp_iou,
        "n_pred": int(len(pred)),
    }


def _hit_ids(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> set[str]:
    gt_keys = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_keys = pred_df["filament_id"].str.split("_", n=1).str[0]
    hits: set[str] = set()
    for annotator_image in gt_keys.unique():
        image_id = annotator_image.split("-", maxsplit=1)[1]
        gt_rows = gt_df.loc[gt_keys == annotator_image]
        pred_rles = pred_df.loc[pred_keys == image_id, "segmentation_rle"].tolist()
        ids = gt_rows["filament_id"].tolist()
        gt_rles = gt_rows["segmentation_rle"].tolist()
        if not gt_rles or not pred_rles:
            continue
        gt_layers = torch.from_numpy(rles_to_layers(gt_rles))
        pred_layers = torch.from_numpy(rles_to_layers(pred_rles))
        iou, _ = overlap_matrices(gt_layers, pred_layers)
        hit = (iou > 0.5).any(dim=1).tolist()
        hits.update(fid for fid, flag in zip(ids, hit) if flag)
    return hits


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"fold 0 already scored: {OUT}")
    print(
        "skip confidence_weighted: frozen cache stores RLEs only, no detection scores",
        flush=True,
    )
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    tune_files = sorted(fn for fn, fold in fold_of.items() if fold != 0)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    if len(tune_files) != 565 or len(val_files) != 142:
        raise SystemExit(f"unexpected split tune {len(tune_files)} val {len(val_files)}")
    if TUNE_JSON.exists():
        tune_rows = json.loads(TUNE_JSON.read_text())["grid"]
        print("tune_cached", json.dumps(tune_rows), flush=True)
    else:
        gt_tune = gt_annotations_to_df(gt_records_for_files(views, by_file, tune_files))
        tune_preds = _frames(tune_files, ("higher_iou_0.7", "soft_nms"), TUNE_DIR)
        tune_rows = []
        for name, pred in tune_preds.items():
            row = _row(name, pred, score_pq(gt_tune, pred))
            tune_rows.append(row)
            print("tune", json.dumps(row), flush=True)
        TUNE_JSON.write_text(json.dumps({"grid": tune_rows, "test_labels_used": False}, indent=2) + "\n")
    best_pq = max(float(r["pq"]) for r in tune_rows)
    band = [r for r in tune_rows if best_pq - float(r["pq"]) <= TIE_PQ]
    chosen_name = "higher_iou_0.7" if any(r["variant"] == "higher_iou_0.7" for r in band) else band[0]["variant"]
    print(f"frozen_variant {chosen_name}", flush=True)

    gt_val = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    val_preds = _frames(val_files, (chosen_name,), FOLD_DIR)
    pred = val_preds[chosen_name]
    scored = _row(chosen_name, pred, score_pq(gt_val, pred))
    pred_csv = FAST_ARTIFACTS / "fusion_last_fold0_pred.csv"
    pred.to_csv(pred_csv, index=False)
    champion = pd.read_csv(CHAMPION_CSV)
    champ_hits = _hit_ids(gt_val, champion)
    new_hits = _hit_ids(gt_val, pred)
    new_tp_set = new_hits != champ_hits
    kept = bool(scored["pq"] > CHAMPION_PQ and new_tp_set)
    payload = {
        "family": "fusion_last_3seed_cache",
        "confidence_weighted": "skipped_cache_has_no_scores",
        "tune_grid": tune_rows,
        "frozen_variant": chosen_name,
        "fold": 0,
        "result": scored,
        "champion_pq": CHAMPION_PQ,
        "delta_pq": scored["pq"] - CHAMPION_PQ,
        "champion_hit_ids": len(champ_hits),
        "new_hit_ids": len(new_hits),
        "hit_ids_only_in_new": len(new_hits - champ_hits),
        "hit_ids_only_in_champion": len(champ_hits - new_hits),
        "new_tp_set": new_tp_set,
        "kept": kept,
        "pred_csv": str(pred_csv),
        "test_labels_used": False,
        "kaggle_submit_called": False,
        "do_not_submit": True,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print("FOLD0", json.dumps(scored), flush=True)
    print(
        f"DONE fusion variant {chosen_name} pq {scored['pq']} kept {kept} "
        f"new_tp_set {new_tp_set} only_new {len(new_hits - champ_hits)} "
        f"only_champion {len(champ_hits - new_hits)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
