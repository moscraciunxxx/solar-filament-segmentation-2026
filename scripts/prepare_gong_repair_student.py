#!/usr/bin/env python3
"""Build the leak-free student set. Does not modify champion labels or weights."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco
from filament.paths import FAST_CACHE, IMAGE_SIZE, SEED, YOLO_ERODE_GT_DIR

SUMMARY = ROOT / "artifacts" / "phaseB_repair" / "repair_summary.json"
PROVENANCE = ROOT / "artifacts" / "phaseB_repair" / "provenance.jsonl"
OUT = FAST_CACHE / "yolo_seg_gong_repair"
PLAN = ROOT / "artifacts" / "phaseB_repair" / "student_plan.json"
JPG_DIR = ROOT / "data" / "gong_propagate" / "jpg"
CHAMPION_STEPS = 40 * math.ceil(565 / 4)


def _link(src: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    os.symlink(src, dest)


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    if summary.get("limit") or not summary.get("passed_gate"):
        raise SystemExit(f"refusing dataset build, repair gate is {summary.get('passed_gate')}")
    if summary.get("shared_label_filenames"):
        raise SystemExit("refusing dataset build, shared label filenames")
    if OUT.exists():
        raise SystemExit(f"refusing to replace existing {OUT}")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    real = sorted(name for name, fold in fold_of.items() if fold in {1, 2, 3, 4})
    if len(real) != 565:
        raise SystemExit(f"expected 565 folds 1-4 images, found {len(real)}")
    (OUT / "images" / "train").mkdir(parents=True)
    (OUT / "images" / "val").mkdir(parents=True)
    (OUT / "labels" / "train").mkdir(parents=True)
    (OUT / "labels" / "val").mkdir(parents=True)
    for name in real:
        stem = Path(name).stem
        image = YOLO_ERODE_GT_DIR / "images" / "train" / name
        label = YOLO_ERODE_GT_DIR / "labels" / "train" / f"{stem}.txt"
        if not image.exists() or not label.exists():
            raise SystemExit(f"missing eroded real pair {name}")
        _link(image.resolve(), OUT / "images" / "train" / name)
        _link(label.resolve(), OUT / "labels" / "train" / f"{stem}.txt")
    gong = 0
    folds = {1: 0, 2: 0, 3: 0, 4: 0}
    with PROVENANCE.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") != "kept":
                continue
            fold = int(row["source_fold"])
            if fold not in folds:
                raise SystemExit(f"kept row outside folds 1-4: {row['train_file']}")
            label = Path(row["label_path"])
            image = JPG_DIR / row["gong_file"]
            if not label.exists() or not image.exists():
                raise SystemExit(f"missing repaired pair {label.name}")
            stem = label.stem
            _link(image.resolve(), OUT / "images" / "train" / f"{stem}.jpg")
            _link(label.resolve(), OUT / "labels" / "train" / f"{stem}.txt")
            folds[fold] += 1
            gong += 1
    blank = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    cv2.imwrite(str(OUT / "images" / "val" / "dummy_blank.jpeg"), blank)
    (OUT / "labels" / "val" / "dummy_blank.txt").write_text(
        "0 0.480 0.480 0.520 0.480 0.520 0.520 0.480 0.520\n"
    )
    n_train = len(real) + gong
    epochs = min(range(1, 41), key=lambda epoch: abs(epoch * math.ceil(n_train / 4) - CHAMPION_STEPS))
    steps = epochs * math.ceil(n_train / 4)
    close_mosaic = 8 if epochs > 8 else max(1, round(epochs * 8 / 40))
    yaml = OUT / "data.yaml"
    yaml.write_text(
        "\n".join(
            [
                f"path: {OUT}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: filament",
                "nc: 1",
                "",
            ]
        )
    )
    plan = {
        "dataset": str(OUT),
        "yaml": str(yaml),
        "real_images": len(real),
        "gong_images": gong,
        "gong_by_ancestor_fold": folds,
        "fold0_real_images": 0,
        "fold0_gong_images": 0,
        "teacher_pseudo_used": False,
        "n_train": n_train,
        "champion_steps": CHAMPION_STEPS,
        "epochs": epochs,
        "batch": 4,
        "image_steps": steps,
        "close_mosaic": close_mosaic,
        "close_mosaic_note": "Champion uses 8 of 40 epochs. A shorter run keeps that fraction, at least one epoch.",
        "seed": SEED,
        "weights_name": "yolo11s_seg_gong_repair_s20260910",
        "score_weights": "last.pt",
        "val": "dummy_blank, not a fold",
    }
    PLAN.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2), flush=True)


if __name__ == "__main__":
    main()
