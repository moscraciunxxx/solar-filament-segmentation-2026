#!/usr/bin/env python3
"""Add miss-specialist masks that do not touch the 2048 parent.

The specialist is yolo11n trained on 640 crops of medoid filaments that
m@1280 missed on folds other than 0. Fold-0 images are scored with a fixed
confidence of 0.50 and a 320 px window stride. No sweep. Does not read test
labels. Does not submit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.disjoint_add import disjoint_add
from filament.infer import _collect_yolo_instances, pick_device
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.miss_crop import CROP, SPECIALIST_CONF, STRIDE, window_origins
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, IMAGE_SIZE, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.postprocess import resolve_overlaps
from filament.rle import encode_rle

PARENT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
WEIGHTS = WEIGHTS_DIR / "yolo11n_seg_miss" / "weights" / "best.pt"
OUT = ARTIFACTS / "stream_miss_specialist_fold0.json"
EXPECTED_PQ = 0.4387235694048039
PUBLIC_GAP = 0.069


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


def specialist_masks(model, image: np.ndarray, device: str) -> list[np.ndarray]:
    origins = window_origins(IMAGE_SIZE, CROP, STRIDE)
    found: list[np.ndarray] = []
    for y0 in origins:
        for x0 in origins:
            crop = image[y0 : y0 + CROP, x0 : x0 + CROP]
            results = model.predict(
                source=crop,
                conf=SPECIALIST_CONF,
                iou=0.6,
                imgsz=CROP,
                retina_masks=True,
                verbose=False,
                device=device,
            )
            if not results:
                continue
            for mask, _conf in _collect_yolo_instances(results[0]):
                if mask.shape != (CROP, CROP):
                    mask = cv2.resize(mask, (CROP, CROP), interpolation=cv2.INTER_NEAREST)
                if int(mask.sum()) < 32:
                    continue
                full = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
                full[y0 : y0 + CROP, x0 : x0 + CROP] = mask
                found.append(full)
    found.sort(key=lambda mask: int(mask.sum()), reverse=True)
    return resolve_overlaps(found)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    weights = args.weights
    out = args.out
    if not weights.exists():
        raise SystemExit(f"missing specialist weights {weights}")
    from ultralytics import YOLO

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
    parent_by = _by_stem(parent)
    device = pick_device()
    model = YOLO(str(weights))
    frames = []
    added = 0
    for name in val_files:
        image = cv2.imread(str(TRAIN_IMG_DIR / name), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"missing {name}")
        masks = specialist_masks(model, image, device)
        donor = [encode_rle(mask) for mask in masks if int(mask.sum()) > 0]
        stem = Path(name).stem
        kept, n_added = disjoint_add(parent_by.get(stem, []), donor)
        added += n_added
        if kept:
            frames.append(pred_to_df(stem, kept))
        print(f"{stem} added {n_added}", flush=True)
    pred = pd.concat(frames, ignore_index=True)
    scored = _row(score_pq(gt_df, pred), len(pred))
    forecast = scored["pq"] - PUBLIC_GAP
    payload = {
        "split": "grouped year x count fold 0",
        "seed": SEED,
        "hypothesis": f"add {weights.parent.parent.name} masks at confidence 0.50 only when they share no pixel with the 2048 parent",
        "weights": str(weights),
        "conf": SPECIALIST_CONF,
        "stride": STRIDE,
        "masks_added": added,
        "parent": first,
        "parent_second_call": second,
        "specialist": scored,
        "lift_vs_2048": scored["pq"] - first["pq"],
        "fp_delta": scored["fp"] - first["fp"],
        "fn_delta": scored["fn"] - first["fn"],
        "public_gap_used": PUBLIC_GAP,
        "public_forecast": forecast,
        "promote": bool(forecast > 0.6),
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
