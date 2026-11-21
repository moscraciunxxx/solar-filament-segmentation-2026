#!/usr/bin/env python3
"""Ownership decode of the frozen 72-voter caches. CPU only. Does not train or submit.

Champion-equivalent settings (area seed, area ownership, min_frac 0) must
match the stored champion CSVs. The pre-registered candidate is degree seed,
vote-ordered ownership, and a 0.7 remnant drop. Fold 0 is scored once, only
after those checks pass.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import FAST_ARTIFACTS, IMAGE_SIZE, SEED, TEST_IMG_DIR
from filament.rle import encode_rle

PACKAGE = PROJECT_ROOT / "artifacts" / "final_package"
REPRO = PACKAGE / "reproduce"
FROZEN_TEST = Path.home() / ".cache/filament-segmentation-2026/submissions/submission_yolo11s_erode_gt_2048_ens3_tta8x3_v24.csv"
FOLD0_CHAMPION_CSV = REPRO / "fold0_ens3_tta8x3_v24.csv"
FROZEN_SHA = "56ae033451164a1d8c74b1d90ef1655d9810a42bff3d191139c8320e7a2c90c5"
KEEP_PQ = 0.475553
FOLDS14_PQ = 0.475429
SEEDS = (
    ("s20260910", "tta_parent_2048", "tta_parent_2048_test"),
    ("s20260911", "tta_s20260911", "tta_s20260911_test"),
    ("s20260912", "tta_s20260912", "tta_s20260912_test"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rles(stem: str, split: str) -> list[str]:
    rles: list[str] = []
    for _name, train_cache, test_cache in SEEDS:
        cache = FAST_ARTIFACTS / (train_cache if split == "train" else test_cache)
        path = cache / f"{stem}.json"
        if not path.exists():
            raise SystemExit(f"missing view cache {path}")
        one = json.loads(path.read_text())
        for counts in one.values():
            rles.extend(counts)
    return rles


def _cluster(order: list[int], iou: np.ndarray, thr: float = 0.5) -> list[list[int]]:
    used = np.zeros(iou.shape[0], dtype=bool)
    groups = []
    for seed in order:
        if used[seed]:
            continue
        used[seed] = True
        others = np.where((~used) & (iou[seed] >= thr))[0]
        used[others] = True
        groups.append([int(seed), *[int(index) for index in others]])
    return groups


def _merge(group: list[int], crops: list[tuple]) -> tuple:
    x0 = min(crops[k][0] for k in group)
    y0 = min(crops[k][1] for k in group)
    x1 = max(crops[k][2] for k in group)
    y1 = max(crops[k][3] for k in group)
    acc = np.zeros((y1 - y0, x1 - x0), dtype=np.int16)
    for k in group:
        a, b, c, d, mask = crops[k]
        acc[b - y0 : d - y0, a - x0 : c - x0] += mask
    need = (len(group) + 1) // 2
    return x0, y0, x1, y1, acc >= need, len(group)


def decode_rles(rles: list[str], seed_mode: str, own_mode: str, min_frac: float) -> tuple[list[np.ndarray], int]:
    """Return exclusive masks and the number dropped as remnants."""
    if not rles:
        return [], 0
    encoded = [{"size": [IMAGE_SIZE, IMAGE_SIZE], "counts": item} for item in rles]
    iou = np.asarray(mask_utils.iou(encoded, encoded, [0] * len(encoded)))
    crops = []
    areas = []
    for item in encoded:
        mask = mask_utils.decode(item)
        ys = np.flatnonzero(mask.any(1))
        xs = np.flatnonzero(mask.any(0))
        y0, y1 = int(ys[0]), int(ys[-1]) + 1
        x0, x1 = int(xs[0]), int(xs[-1]) + 1
        crop = mask[y0:y1, x0:x1].astype(np.int16)
        crops.append((x0, y0, x1, y1, crop))
        areas.append(int(crop.sum()))
    areas_arr = np.asarray(areas)
    if seed_mode == "area":
        order = sorted(range(len(rles)), key=lambda i: -int(areas_arr[i]))
    elif seed_mode == "degree":
        degree = (iou >= 0.5).sum(1)
        order = sorted(range(len(rles)), key=lambda i: (-int(degree[i]), -int(areas_arr[i])))
    else:
        raise SystemExit(f"unknown seed mode {seed_mode}")
    groups = [group for group in _cluster(order, iou) if len(group) >= 24]
    merged = [_merge(group, crops) for group in groups]
    items = []
    for box in merged:
        area = int(box[4].sum())
        if 96 <= area <= 80000:
            items.append((area, box))
    if own_mode == "area":
        items.sort(key=lambda item: item[0], reverse=True)
    elif own_mode == "votes":
        items.sort(key=lambda item: (item[1][5], item[0]), reverse=True)
    else:
        raise SystemExit(f"unknown ownership mode {own_mode}")
    claimed = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    kept = []
    remnants = 0
    for area, (x0, y0, x1, y1, mask, _votes) in items:
        exclusive = mask & ~claimed[y0:y1, x0:x1]
        size = int(exclusive.sum())
        if size < 32 or size < min_frac * area:
            if size >= 32 and size < min_frac * area:
                remnants += 1
            continue
        claimed[y0:y1, x0:x1] |= exclusive
        full = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        full[y0:y1, x0:x1] = exclusive
        kept.append(full)
    return kept, remnants


def _frames(files: list[str], split: str, seed_mode: str, own_mode: str, min_frac: float) -> tuple[pd.DataFrame, int]:
    frames = []
    remnants = 0
    t0 = time.perf_counter()
    for index, name in enumerate(files, start=1):
        stem = Path(name).stem
        masks, dropped = decode_rles(_rles(stem, split), seed_mode, own_mode, min_frac)
        remnants += dropped
        if masks:
            frames.append(pred_to_df(stem, [encode_rle(mask) for mask in masks]))
        if index == 1 or index % 25 == 0 or index == len(files):
            print(f"{split} {seed_mode} {index}/{len(files)} sec {time.perf_counter() - t0:.0f}", flush=True)
    pred = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    return pred, remnants


def _csv_bytes(pred: pd.DataFrame) -> bytes:
    from io import StringIO

    buf = StringIO()
    pred.to_csv(buf, index=False)
    return buf.getvalue().encode()


def check_champion() -> dict:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    fold0 = sorted(name for name, fold in fold_of.items() if fold == 0)
    pred, _ = _frames(fold0, "train", "area", "area", 0.0)
    got = _csv_bytes(pred)
    expected = FOLD0_CHAMPION_CSV.read_bytes()
    images = sorted(TEST_IMG_DIR.glob("*.jpeg"))
    test_pred, _ = _frames([path.name for path in images], "test", "area", "area", 0.0)
    test_bytes = _csv_bytes(test_pred)
    frozen = FROZEN_TEST.read_bytes()
    report = {
        "fold0_bytes_equal": got == expected,
        "fold0_rows": int(len(pred)),
        "test_sha256": hashlib.sha256(test_bytes).hexdigest(),
        "test_byte_identical": test_bytes == frozen and hashlib.sha256(frozen).hexdigest() == FROZEN_SHA,
    }
    (REPRO / "own_champion_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["fold0_bytes_equal"] or not report["test_byte_identical"]:
        raise SystemExit("champion-equivalent decode does not match the frozen CSVs")
    return report


def check_folds14() -> dict:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    files = sorted(name for name, fold in fold_of.items() if fold in {1, 2, 3, 4})
    pred, remnants = _frames(files, "train", "degree", "votes", 0.7)
    gt = gt_annotations_to_df(gt_records_for_files(views, by_file, files, rasterizer="coco"))
    scored = score_pq(gt, pred)
    report = {
        "pq": scored.pq,
        "tp": scored.tp,
        "fp": scored.fp,
        "fn": scored.fn,
        "mean_tp_iou": scored.mean_tp_iou,
        "remnants_dropped": remnants,
        "target_pq": FOLDS14_PQ,
        "target_tp": 4229,
        "target_fp": 1404,
        "target_fn": 2381,
        "pq_ok": abs(scored.pq - FOLDS14_PQ) <= 0.0001,
        "counts_ok": scored.tp == 4229 and scored.fp == 1404 and scored.fn == 2381,
    }
    (REPRO / "own07_folds14.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["pq_ok"] or not report["counts_ok"]:
        raise SystemExit("folds 1-4 own07 decode does not match the pre-registered number")
    return report


def _hit_ids(gt: pd.DataFrame, pred: pd.DataFrame) -> set[str]:
    from filament.metric import overlap_matrices
    from filament.rle import rles_to_layers
    import torch

    gt_keys = gt["filament_id"].str.split("_", n=1).str[0]
    pred_keys = pred["filament_id"].str.split("_", n=1).str[0]
    hits = set()
    for annotator_image in gt_keys.unique():
        image_id = annotator_image.split("-", maxsplit=1)[1]
        gt_rows = gt.loc[gt_keys == annotator_image]
        pred_rles = pred.loc[pred_keys == image_id, "segmentation_rle"].tolist()
        if not pred_rles or gt_rows.empty:
            continue
        iou, _dice = overlap_matrices(
            torch.from_numpy(rles_to_layers(gt_rows["segmentation_rle"].tolist())),
            torch.from_numpy(rles_to_layers(pred_rles)),
        )
        matched = iou > 0.5
        for index, filament_id in enumerate(gt_rows["filament_id"].tolist()):
            if bool(matched[index].any()):
                hits.add(filament_id)
    return hits


def score_fold0() -> dict:
    gate = REPRO / "own07_fold0.json"
    if gate.exists():
        raise SystemExit(f"refusing a second fold-0 look: {gate}")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    files = sorted(name for name, fold in fold_of.items() if fold == 0)
    pred, remnants = _frames(files, "train", "degree", "votes", 0.7)
    gt = gt_annotations_to_df(gt_records_for_files(views, by_file, files, rasterizer="coco"))
    scored = score_pq(gt, pred)
    champion = pd.read_csv(FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_ensemble3_tta_fold0_pred.csv")
    new_hits = _hit_ids(gt, pred)
    old_hits = _hit_ids(gt, champion)
    report = {
        "pq": scored.pq,
        "tp": scored.tp,
        "fp": scored.fp,
        "fn": scored.fn,
        "mean_tp_iou": scored.mean_tp_iou,
        "n_pred": int(len(pred)),
        "remnants_dropped": remnants,
        "gt_gained": len(new_hits - old_hits),
        "gt_lost": len(old_hits - new_hits),
        "keep_bar": KEEP_PQ,
        "keep": bool(scored.pq >= KEEP_PQ),
        "kaggle_submit_called": False,
    }
    gate.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if report["keep"]:
        images = sorted(TEST_IMG_DIR.glob("*.jpeg"))
        test_pred, _ = _frames([path.name for path in images], "test", "degree", "votes", 0.7)
        out = Path.home() / ".cache/filament-segmentation-2026/submissions/submission_yolo11s_erode_gt_2048_ens3_tta8x3_v24_own07.csv"
        test_pred.to_csv(out, index=False)
        stems = {path.stem for path in images}
        id_stems = set(test_pred["filament_id"].astype(str).str.rsplit("_", n=1).str[0]) if len(test_pred) else set()
        meta = {
            "path": str(out),
            "sha256": _sha256(out),
            "rows": int(len(test_pred)),
            "test_jpegs": len(images),
            "images_with_predictions": len(id_stems),
            "images_without_predictions": sorted(stems - id_stems),
            "non_test_stems": sorted(id_stems - stems),
            "ask_before_submit": True,
            "finals_unchanged": True,
        }
        (REPRO / "own07_test_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta, indent=2), flush=True)
    else:
        _log_miss(report)
    return report


def _sha_file(path: Path) -> str:
    return _sha256(path)


def _log_miss(report: dict) -> None:
    path = PROJECT_ROOT / "artifacts" / "mistake_log.json"
    data = json.loads(path.read_text())
    data["entries"].append(
        {
            "id": "fusion_ownership_own07",
            "believed": "Degree-seeded clusters, vote-ordered ownership, and a 0.7 remnant drop would beat the champion by 0.01 on fold 0.",
            "happened": (
                f"Fold-0 pycocotools PQ {report['pq']}, TP {report['tp']}, FP {report['fp']}, "
                f"FN {report['fn']}, mean TP IoU {report['mean_tp_iou']}, "
                f"GT gained {report['gt_gained']}, lost {report['gt_lost']}, remnants {report['remnants_dropped']}."
            ),
            "lesson": "The folds 1-4 gain did not clear the fold-0 bar of 0.475553.",
            "rule": "Do not retune votes, IoU, or the remnant fraction on fold 0.",
        }
    )
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "qa"
    if command == "check-champion":
        check_champion()
    elif command == "check-folds14":
        check_folds14()
    elif command == "fold0":
        if not (REPRO / "own_champion_check.json").exists() or not (REPRO / "own07_folds14.json").exists():
            raise SystemExit("fold 0 is refused until both QA checks exist")
        score_fold0()
    elif command == "qa":
        check_champion()
        check_folds14()
    else:
        raise SystemExit(f"unknown command {command}")


if __name__ == "__main__":
    main()
