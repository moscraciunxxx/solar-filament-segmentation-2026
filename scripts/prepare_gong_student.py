#!/usr/bin/env python3
"""YOLO set for the no-test-JPEG Phase B student.

Train images are folds 1-3 eroded-medoid labels plus propagated GONG frames
from those folds. Fold 4 originals are the validation set. Fold 0 is absent.
Propagated labels are the warped polygons plus teacher instances that do not
overlap them at IoU 0.5. Images with a small or outer-disk filament are
copied twice so copy-paste sees them more often. Does not read test labels.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco
from filament.paths import FAST_CACHE, SEED, YOLO_ERODE_GT_DIR

OUT = FAST_CACHE / "yolo_seg_gong_student"
WARPED = ROOT / "artifacts" / "gong_propagate_warped.jsonl"
JPG_DIR = ROOT / "data" / "gong_propagate" / "jpg"
LABEL_DIR = ROOT / "data" / "gong_propagate" / "labels"
PSEUDO_DIR = ROOT / "data" / "gong_propagate" / "pseudo"
RADIUS_PX = 1000.0
CENTER = 1023.5


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def _polys(text: str) -> list[np.ndarray]:
    polys = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        coords = list(map(float, parts[1:]))
        polys.append(np.array(coords, dtype=np.float32).reshape(-1, 2))
    return polys


def _mask(poly: np.ndarray) -> np.ndarray:
    pts = np.round(np.column_stack([poly[:, 0] * 2048, poly[:, 1] * 2048])).astype(np.int32)
    import cv2

    mask = np.zeros((512, 512), dtype=np.uint8)
    pts = np.round(pts * (512 / 2048)).astype(np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 1)
    return mask


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def _biased(text: str) -> bool:
    for poly in _polys(text):
        xs = poly[:, 0] * 2048
        ys = poly[:, 1] * 2048
        # shoelace area in pixels
        area = 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
        r = float(np.hypot(xs.mean() - CENTER, CENTER - ys.mean()) / RADIUS_PX)
        if area < 1500 or r > 0.7:
            return True
    return False


def _merge(warped: str, pseudo: str) -> str:
    base = _polys(warped)
    extra = []
    base_masks = [_mask(poly) for poly in base]
    for poly in _polys(pseudo):
        mask = _mask(poly)
        if any(_iou(mask, other) >= 0.5 for other in base_masks):
            continue
        extra.append(poly)
        base_masks.append(mask)
    lines = []
    for poly in base + extra:
        coords = " ".join(f"{v:.6f}" for v in poly.reshape(-1))
        lines.append(f"0 {coords}")
    return "\n".join(lines) + ("\n" if lines else "")


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
    n_train = n_val = n_gong = n_pseudo = n_bias = 0
    src_images = YOLO_ERODE_GT_DIR / "images"
    src_labels = YOLO_ERODE_GT_DIR / "labels"
    for split in ("train", "val"):
        for image in (src_images / split).iterdir():
            if not image.is_file() and not image.is_symlink():
                continue
            fold = fold_of.get(image.name)
            if fold == 0:
                continue
            dest_split = "val" if fold == 4 else "train"
            _link(image.resolve(), OUT / "images" / dest_split / image.name)
            label = src_labels / split / f"{image.stem}.txt"
            text = label.read_text() if label.exists() else ""
            (OUT / "labels" / dest_split / f"{image.stem}.txt").write_text(text)
            if dest_split == "val":
                n_val += 1
                continue
            n_train += 1
            if text and _biased(text):
                for copy_id in (1, 2):
                    name = f"{image.stem}__cp{copy_id}.jpeg"
                    _link(image.resolve(), OUT / "images" / "train" / name)
                    (OUT / "labels" / "train" / f"{image.stem}__cp{copy_id}.txt").write_text(text)
                n_bias += 1
    if WARPED.exists():
        for line in WARPED.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("use_in_student_train") or int(row["source_fold"]) == 4:
                continue
            image = JPG_DIR / row["gong_file"]
            label = LABEL_DIR / f"{Path(row['gong_file']).stem}.txt"
            if not image.exists() or not label.exists():
                continue
            text = label.read_text()
            pseudo = PSEUDO_DIR / f"{Path(row['gong_file']).stem}.txt"
            if pseudo.exists():
                text = _merge(text, pseudo.read_text())
                n_pseudo += 1
            stem = Path(row["gong_file"]).stem
            _link(image, OUT / "images" / "train" / f"{stem}.jpg")
            (OUT / "labels" / "train" / f"{stem}.txt").write_text(text)
            n_gong += 1
            n_train += 1
    yaml_path = OUT / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(OUT),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "filament"},
                "nc": 1,
            },
            sort_keys=False,
        )
    )
    stats = {
        "n_train": n_train,
        "n_val_fold4": n_val,
        "n_gong": n_gong,
        "n_gong_with_pseudo": n_pseudo,
        "n_bias_copies": n_bias,
        "fold0_included": False,
        "test_jpegs_included": False,
        "test_labels_used": False,
        "yaml": str(yaml_path),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
