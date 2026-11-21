#!/usr/bin/env python3
"""Build 640 crops of medoid filaments the m@1280 model missed on folds other than 0.

Fold 0 is never written. Folds 1-3 are train. Fold 4 is the trainer holdout.
Does not read test labels. Does not train. Does not submit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco, medoid_view_id, rasterize_polygons
from filament.disjoint_add import mask_iou
from filament.miss_crop import CROP, MISS_IOU, crop_origin, polygon_in_crop
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, FAST_CACHE, SEED, TRAIN_IMG_DIR, YOLO_DIR
from filament.rle import decode_rle

PARENT = FAST_ARTIFACTS / "yolo_yolo11m_seg_1280_foldne0_conf0.35_sz1280_m0_base.csv"
OUT_DIR = FAST_CACHE / "yolo_seg_miss"
META = ARTIFACTS / "miss_crop_build.json"
VAL_FOLD = 4


def _by_stem(frame: pd.DataFrame) -> dict[str, list[str]]:
    stems = frame["filament_id"].str.split("_", n=1).str[0]
    grouped: dict[str, list[str]] = {}
    for stem, rle in zip(stems, frame["segmentation_rle"], strict=True):
        grouped.setdefault(stem, []).append(rle)
    return grouped


def _image_path(file_name: str) -> Path:
    cached = YOLO_DIR / "images" / "train" / file_name
    if cached.exists():
        return cached
    return TRAIN_IMG_DIR / file_name


def main() -> None:
    if OUT_DIR.resolve() == YOLO_DIR.resolve():
        raise SystemExit("refusing to write miss crops over yolo_seg")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    preds = _by_stem(pd.read_csv(PARENT))
    if OUT_DIR.exists():
        raise SystemExit(f"refusing to overwrite {OUT_DIR}")
    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True)

    counts = {"train": 0, "val": 0, "fold0_skipped": 0, "misses": 0}
    for file_name, fold in sorted(fold_of.items()):
        if fold == 0:
            counts["fold0_skipped"] += 1
            continue
        split = "val" if fold == VAL_FOLD else "train"
        image_ids = by_file[file_name]
        view = views[medoid_view_id(views, image_ids)]
        stem = Path(file_name).stem
        parent_masks = [decode_rle(rle) for rle in preds.get(stem, [])]
        image_path = _image_path(file_name)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"missing image {image_path}")
        filaments = []
        for filament in view.filaments:
            mask = rasterize_polygons([filament.polygon])
            best = max((mask_iou(mask, other) for other in parent_masks), default=0.0)
            filaments.append((filament, best))
        for index, (filament, best) in enumerate(filaments, start=1):
            if best >= MISS_IOU:
                continue
            cx = float(filament.polygon[:, 0].mean())
            cy = float(filament.polygon[:, 1].mean())
            x0, y0 = crop_origin(cx, cy)
            lines = []
            for other, _best in filaments:
                line = polygon_in_crop(other.polygon, x0, y0, CROP)
                if line:
                    lines.append(line)
            if not lines:
                continue
            crop = image[y0 : y0 + CROP, x0 : x0 + CROP]
            if crop.shape[0] != CROP or crop.shape[1] != CROP:
                continue
            name = f"{stem}_m{index}.jpg"
            cv2.imwrite(str(OUT_DIR / "images" / split / name), crop)
            (OUT_DIR / "labels" / split / f"{stem}_m{index}.txt").write_text(
                "\n".join(lines) + "\n"
            )
            counts[split] += 1
            counts["misses"] += 1
    if counts["train"] == 0 or counts["val"] == 0:
        raise SystemExit(f"empty split: {counts}")
    data = {
        "path": str(OUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "filament"},
    }
    import yaml

    (OUT_DIR / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    payload = {
        "out_dir": str(OUT_DIR),
        "parent_csv": str(PARENT),
        "val_fold": VAL_FOLD,
        "crop": CROP,
        "miss_iou": MISS_IOU,
        "counts": counts,
        "fold0_in_dataset": False,
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    META.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
