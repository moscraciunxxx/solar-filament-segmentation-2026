#!/usr/bin/env python3
"""Add one extra copy of faint training images to the eroded YOLO set.

The median is computed on folds other than 0. Fold 0 stays the validation
split and is not duplicated. Images and labels are symlinks. Does not
submit and does not read test labels.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco
from filament.contrast_floor import mask_contrast
from filament.faint_oversample import image_is_faint
from filament.medoid_gt import medoid_gt_records
from filament.paths import ARTIFACTS, FAST_CACHE, SEED, TRAIN_IMG_DIR, YOLO_ERODE_GT_DIR
from filament.preprocess import read_gray
from filament.rle import decode_rle

SOURCE = YOLO_ERODE_GT_DIR
DEST = FAST_CACHE / "yolo_seg_erode_gt_faint"
RECEIPT = ARTIFACTS / "faint_oversample_build.json"


def _link(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        if dest.is_symlink() and dest.resolve() == src.resolve():
            return
        raise SystemExit(f"refusing to replace {dest}")
    dest.symlink_to(src)


def _contrasts(views, by_file, name: str, image) -> list[float | None]:
    values: list[float | None] = []
    for rec in medoid_gt_records(views, by_file[name]):
        values.append(mask_contrast(decode_rle(rec["rle"]), image))
    return values


def main() -> None:
    if (DEST / "data.yaml").exists():
        raise SystemExit(f"dataset already built: {DEST}")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    train_names = sorted(name for name, fold in fold_of.items() if fold != 0)
    measured: list[float] = []
    per_image: dict[str, list[float | None]] = {}
    for name in train_names:
        image_path = SOURCE / "images" / "train" / name
        if not image_path.exists():
            image_path = TRAIN_IMG_DIR / name
        image = read_gray(image_path)
        contrasts = _contrasts(views, by_file, name, image)
        per_image[name] = contrasts
        measured.extend(value for value in contrasts if value is not None)
    if len(measured) < 30:
        raise SystemExit(f"too few training filaments: {len(measured)}")
    median = float(np.median(measured))
    faint_names = [name for name in train_names if image_is_faint(per_image[name], median)]
    for split in ("train", "val"):
        image_dir = SOURCE / "images" / split
        label_dir = SOURCE / "labels" / split
        for image in sorted(image_dir.glob("*.jpeg")):
            _link(image, DEST / "images" / split / image.name)
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                raise SystemExit(f"missing label {label}")
            _link(label, DEST / "labels" / split / label.name)
    for name in faint_names:
        image = SOURCE / "images" / "train" / name
        label = SOURCE / "labels" / "train" / f"{Path(name).stem}.txt"
        extra = f"{Path(name).stem}_faint"
        _link(image, DEST / "images" / "train" / f"{extra}.jpeg")
        _link(label, DEST / "labels" / "train" / f"{extra}.txt")
    (DEST / "data.yaml").write_text(
        "path: " + str(DEST) + "\ntrain: images/train\nval: images/val\nnames:\n  0: filament\nnc: 1\n"
    )
    payload = {
        "source": str(SOURCE),
        "dest": str(DEST),
        "median_contrast": median,
        "n_train_images": len(train_names),
        "n_train_filaments": len(measured),
        "n_faint_images": len(faint_names),
        "fold0_duplicated": False,
        "test_labels_used": False,
    }
    RECEIPT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
