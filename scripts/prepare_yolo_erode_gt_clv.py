#!/usr/bin/env python3
"""Flat-field the erode-gt YOLO images. Labels stay the parent polygons.

The only pixel change is center_to_limb_flat, repeated to three channels.
Does not modify yolo_seg_erode_gt or any weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco
from filament.flatfield import center_to_limb_flat
from filament.paths import FAST_CACHE, SEED
from filament.preprocess import read_gray

SOURCE = FAST_CACHE / "yolo_seg_erode_gt"
DEST = FAST_CACHE / "yolo_seg_erode_gt_clv"


def _stems(split: str, root: Path) -> set[str]:
    folder = root / "images" / split
    return {path.stem for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}}


def _write_flat(src: Path, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        return
    gray = read_gray(src)
    flat = center_to_limb_flat(gray)
    pixels = np.clip(np.round(flat * 255.0), 0, 255).astype(np.uint8)
    rgb = np.stack([pixels, pixels, pixels], axis=-1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest), rgb):
        raise RuntimeError(f"failed to write {dest}")


def main() -> None:
    if not (SOURCE / "data.yaml").is_file():
        raise SystemExit(f"missing parent export {SOURCE}")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    fold0 = {Path(name).stem for name, fold in fold_of.items() if fold == 0}
    train_folds = {Path(name).stem for name, fold in fold_of.items() if fold != 0}
    src_train = _stems("train", SOURCE)
    src_val = _stems("val", SOURCE)
    if src_train != train_folds or src_val != fold0:
        raise SystemExit(
            f"erode-gt split is not folds 1-4 / fold 0: "
            f"train {len(src_train)} vs {len(train_folds)}, val {len(src_val)} vs {len(fold0)}"
        )
    if len(src_val) != 142 or len(src_train) != 565:
        raise SystemExit(f"unexpected counts train={len(src_train)} val={len(src_val)}")
    for split in ("train", "val"):
        (DEST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DEST / "labels" / split).mkdir(parents=True, exist_ok=True)
        files = sorted((SOURCE / "images" / split).iterdir())
        for index, src in enumerate(files, start=1):
            if src.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            dest = DEST / "images" / split / f"{src.stem}.png"
            _write_flat(src, dest)
            label_src = SOURCE / "labels" / split / f"{src.stem}.txt"
            label_dest = DEST / "labels" / split / f"{src.stem}.txt"
            if not label_src.is_file():
                raise SystemExit(f"missing label {label_src}")
            if label_dest.is_symlink() or label_dest.exists():
                if label_dest.is_symlink() and label_dest.resolve() == label_src.resolve():
                    pass
                else:
                    raise SystemExit(f"label path is not the parent symlink: {label_dest}")
            else:
                label_dest.symlink_to(label_src)
            if index % 50 == 0 or index == len(files):
                print(f"clv {split} {index}/{len(files)}", flush=True)
    yaml_path = DEST / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {DEST}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: filament",
                "nc: 1",
                "",
            ]
        )
    )
    report = {
        "source": str(SOURCE),
        "dest": str(DEST),
        "n_train": 565,
        "n_val": 142,
        "val_is_fold0": True,
        "train_is_folds_1_to_4": True,
        "labels": "symlinks to the erode-gt polygons",
        "pixels": "center_to_limb_flat repeated to 3 uint8 channels, PNG",
        "parent_images_modified": False,
    }
    (DEST / "clv_export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
