#!/usr/bin/env python3
"""Rescore the champion with OpenCV GT and pycocotools GT.

Predictions stay the cached 3-seed TTA instances. Fold 0 and folds 1-4 are
both reported. Does not submit. Does not read test labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco, rasterize_polygons
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED
from filament.rle import decode_rle

CHAMPION_DIR = FAST_ARTIFACTS / "tile_addon" / "champion"
OUT = ARTIFACTS / "champion_coco_raster_rescore.json"
KNOWN_FOLD0 = 0.4741684688395591


def _pred_df(files: list[str]) -> pd.DataFrame:
    frames = []
    for file_name in files:
        path = CHAMPION_DIR / f"{Path(file_name).stem}.json"
        rles = json.loads(path.read_text())
        frames.append(pred_to_df(Path(file_name).stem, rles))
    if not frames:
        return pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    return pd.concat(frames, ignore_index=True)


def _score(views, by_file, files: list[str], pred: pd.DataFrame, rasterizer: str) -> dict:
    gt = gt_annotations_to_df(gt_records_for_files(views, by_file, files, rasterizer=rasterizer))
    scored = score_pq(gt, pred)
    return {
        "rasterizer": rasterizer,
        "n_images": len(files),
        "pq": scored.pq,
        "tp": scored.tp,
        "fp": scored.fp,
        "fn": scored.fn,
        "n_pred": int(len(pred)),
        "mean_tp_iou": scored.mean_tp_iou,
    }


def _pixel_delta(views, by_file, files: list[str], limit: int = 40) -> dict:
    from filament.coco_io import polygon_to_coco_rle

    checked = 0
    changed = 0
    abs_pix = 0
    coco_only = 0
    cv_only = 0
    for file_name in files:
        for image_id in by_file[file_name]:
            for filament in views[image_id].filaments:
                if len(filament.polygon) < 3:
                    continue
                cv = rasterize_polygons([filament.polygon])
                coco = decode_rle(polygon_to_coco_rle(filament.polygon))
                diff = cv.astype(np.int16) - coco.astype(np.int16)
                n = int(np.abs(diff).sum())
                checked += 1
                abs_pix += n
                if n:
                    changed += 1
                    coco_only += int((diff < 0).sum())
                    cv_only += int((diff > 0).sum())
                if checked >= limit:
                    return {
                        "filaments_checked": checked,
                        "filaments_changed": changed,
                        "mean_abs_pixels": abs_pix / max(1, checked),
                        "cv_only_pixels": cv_only,
                        "coco_only_pixels": coco_only,
                    }
    return {
        "filaments_checked": checked,
        "filaments_changed": changed,
        "mean_abs_pixels": abs_pix / max(1, checked),
        "cv_only_pixels": cv_only,
        "coco_only_pixels": coco_only,
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"already scored: {OUT}")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    fold0 = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    folds14 = sorted(fn for fn, fold in fold_of.items() if fold != 0)
    if len(fold0) != 142 or len(folds14) != 565:
        raise SystemExit(f"unexpected split {len(fold0)} {len(folds14)}")
    missing = [fn for fn in fold0 + folds14 if not (CHAMPION_DIR / f"{Path(fn).stem}.json").exists()]
    if missing:
        raise SystemExit(f"missing champion cache {missing[:3]}")
    pixel = _pixel_delta(views, by_file, fold0)
    print("pixel", json.dumps(pixel), flush=True)
    rows = {}
    for name, files in (("fold0", fold0), ("folds14", folds14)):
        pred = _pred_df(files)
        cv = _score(views, by_file, files, pred, "cv2")
        coco = _score(views, by_file, files, pred, "coco")
        rows[name] = {
            "cv2": cv,
            "coco": coco,
            "delta_pq": coco["pq"] - cv["pq"],
            "delta_tp": coco["tp"] - cv["tp"],
            "delta_fp": coco["fp"] - cv["fp"],
            "delta_fn": coco["fn"] - cv["fn"],
        }
        print(name, json.dumps(rows[name]), flush=True)
    payload = {
        "pixel_sample": pixel,
        "known_fold0_cv2_pq": KNOWN_FOLD0,
        "fold0_cv2_minus_known": rows["fold0"]["cv2"]["pq"] - KNOWN_FOLD0,
        "splits": rows,
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print("DONE champion_coco_raster", json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
