#!/usr/bin/env python3
"""Add two quiet-Sun crops per training JPEG to the miss-crop set.

Positives stay the folds-other-than-0 miss crops. A negative window shares
no pixel with the medoid mask and its center lies on the solar disk.
Fold 0 is not written. Does not train. Does not read test labels.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco, medoid_view_id, rasterize_polygons
from filament.miss_crop import CROP, NEG_PER_IMAGE, quiet_origins
from filament.paths import ARTIFACTS, FAST_CACHE, IMAGE_SIZE, SEED, TRAIN_IMG_DIR, YOLO_DIR

POS_DIR = FAST_CACHE / "yolo_seg_miss"
OUT_DIR = FAST_CACHE / "yolo_seg_miss_neg"
META = ARTIFACTS / "miss_neg_build.json"
VAL_FOLD = 4


def _image_path(file_name: str) -> Path:
    cached = YOLO_DIR / "images" / "train" / file_name
    if cached.exists():
        return cached
    return TRAIN_IMG_DIR / file_name


def _link_tree(split: str) -> int:
    src_images = POS_DIR / "images" / split
    src_labels = POS_DIR / "labels" / split
    dst_images = OUT_DIR / "images" / split
    dst_labels = OUT_DIR / "labels" / split
    dst_images.mkdir(parents=True)
    dst_labels.mkdir(parents=True)
    n = 0
    for image in sorted(src_images.glob("*.jpg")):
        (dst_images / image.name).symlink_to(image.resolve())
        label = src_labels / f"{image.stem}.txt"
        if not label.exists():
            raise SystemExit(f"missing label {label}")
        (dst_labels / label.name).symlink_to(label.resolve())
        n += 1
    return n


def main() -> None:
    if not POS_DIR.exists():
        raise SystemExit(f"missing positive crops {POS_DIR}")
    if OUT_DIR.exists():
        raise SystemExit(f"refusing to overwrite {OUT_DIR}")
    if OUT_DIR.resolve() == POS_DIR.resolve() or OUT_DIR.resolve() == YOLO_DIR.resolve():
        raise SystemExit("refusing to write negatives over an existing YOLO set")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    rng = np.random.default_rng(SEED)
    n_pos = {split: _link_tree(split) for split in ("train", "val")}
    counts = {"train": 0, "val": 0, "fold0_skipped": 0}
    for file_name, fold in sorted(fold_of.items()):
        if fold == 0:
            counts["fold0_skipped"] += 1
            continue
        split = "val" if fold == VAL_FOLD else "train"
        view = views[medoid_view_id(views, by_file[file_name])]
        union = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        for filament in view.filaments:
            union = np.maximum(union, rasterize_polygons([filament.polygon]))
        image = cv2.imread(str(_image_path(file_name)), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"missing image {file_name}")
        stem = Path(file_name).stem
        for index, (x0, y0) in enumerate(quiet_origins(union, rng, NEG_PER_IMAGE), start=1):
            crop = image[y0 : y0 + CROP, x0 : x0 + CROP]
            if crop.shape[0] != CROP or crop.shape[1] != CROP:
                continue
            name = f"{stem}_q{index}"
            cv2.imwrite(str(OUT_DIR / "images" / split / f"{name}.jpg"), crop)
            (OUT_DIR / "labels" / split / f"{name}.txt").write_text("")
            counts[split] += 1
    if counts["train"] == 0 or counts["val"] == 0:
        shutil.rmtree(OUT_DIR)
        raise SystemExit(f"empty negative split: {counts}")
    data = {
        "path": str(OUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "filament"},
    }
    (OUT_DIR / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    payload = {
        "out_dir": str(OUT_DIR),
        "positives": n_pos,
        "negatives": counts,
        "neg_per_image": NEG_PER_IMAGE,
        "val_fold": VAL_FOLD,
        "fold0_in_dataset": False,
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    META.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
