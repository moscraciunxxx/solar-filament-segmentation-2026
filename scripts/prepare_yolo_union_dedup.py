#!/usr/bin/env python3
"""YOLO labels: one instance per filament, deduped across annotators at IoU>0.5.

Train is folds 1-4. Val is fold 0 with the same dedup, matching the parent
recipe's use of fold 0 for early stopping only. Official Panoptic Quality is
scored later against every raw annotator, not these labels.
Does not read test labels. Does not train. Does not submit.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco, medoid_view_id, rasterize_polygons
from filament.paths import FAST_CACHE, IMAGE_SIZE, SEED, TRAIN_IMG_DIR, YOLO_ERODE_GT_DIR
from filament.yolo_export import polygon_to_yolo_line

OUT = FAST_CACHE / "yolo_seg_union_dedup"
COMPARE = 512
IOU_THR = 0.5


def _mask(polygon: np.ndarray) -> np.ndarray:
    return rasterize_polygons([polygon * (COMPARE / float(IMAGE_SIZE))], size=COMPARE)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def _kept_polygons(views, image_ids: list[str]) -> list[np.ndarray]:
    medoid = medoid_view_id(views, image_ids)
    packed = []
    for image_id in image_ids:
        is_medoid = image_id == medoid
        for filament in views[image_id].filaments:
            if len(filament.polygon) < 3:
                continue
            packed.append(
                {
                    "polygon": filament.polygon,
                    "mask": _mask(filament.polygon),
                    "area": float(filament.area),
                    "medoid": is_medoid,
                }
            )
    order = sorted(range(len(packed)), key=lambda i: (packed[i]["medoid"], packed[i]["area"]), reverse=True)
    used = [False] * len(packed)
    kept = []
    for i in order:
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in order:
            if used[j]:
                continue
            if _iou(packed[i]["mask"], packed[j]["mask"]) > IOU_THR:
                group.append(j)
                used[j] = True
        chosen = max(group, key=lambda k: (packed[k]["medoid"], packed[k]["area"]))
        kept.append(packed[chosen]["polygon"])
    return kept


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    train_files = sorted(fn for fn, fold in fold_of.items() if fold != 0)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    if len(train_files) != 565 or len(val_files) != 142:
        raise SystemExit(f"unexpected split train {len(train_files)} val {len(val_files)}")

    if OUT.exists():
        # Labels only. Images are symlinks into the eroded export.
        for split in ("train", "val"):
            label_dir = OUT / "labels" / split
            if label_dir.exists():
                for path in label_dir.glob("*.txt"):
                    path.unlink()
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "train_medoid": 0, "val_medoid": 0}
    for split, files in (("train", train_files), ("val", val_files)):
        for file_name in files:
            image_ids = by_file[file_name]
            medoid = medoid_view_id(views, image_ids)
            n_medoid = sum(1 for f in views[medoid].filaments if len(f.polygon) >= 3)
            kept = _kept_polygons(views, image_ids)
            lines = [polygon_to_yolo_line(poly) for poly in kept]
            lines = [ln for ln in lines if ln]
            stem = Path(file_name).stem
            (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            src = YOLO_ERODE_GT_DIR / "images" / split / file_name
            if not src.exists():
                src = TRAIN_IMG_DIR / file_name
            _link(src, OUT / "images" / split / file_name)
            counts[split] += len(lines)
            counts[f"{split}_medoid"] += n_medoid

    yaml_path = OUT / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(OUT.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "filament"},
                "nc": 1,
            },
            sort_keys=False,
        )
    )
    stats = {
        "mode": "union_dedup_iou_gt_0.5",
        "keep": "medoid polygon if that instance is in the cluster, else the largest",
        "compare_size": COMPARE,
        "seed": SEED,
        "n_train_files": len(train_files),
        "n_val_files": len(val_files),
        "n_train_instances": counts["train"],
        "n_train_medoid": counts["train_medoid"],
        "n_train_added": counts["train"] - counts["train_medoid"],
        "n_val_instances": counts["val"],
        "n_val_medoid": counts["val_medoid"],
        "n_val_added": counts["val"] - counts["val_medoid"],
        "yaml": str(yaml_path),
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    (OUT / "dedup_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
