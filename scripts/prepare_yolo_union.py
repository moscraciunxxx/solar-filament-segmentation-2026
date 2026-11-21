#!/usr/bin/env python3
"""Export yolo_seg_union: one JPEG / sample, train = stacked annotator instances.

Does not edit yolo_export.py or prepare_yolo.py. Does not rmtree live YOLO dirs.
Does not train. Does not submit. Train JSON only. CPU export.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco
from filament.paths import ARTIFACTS, SEED, TRAIN_IMG_DIR, TRAIN_JSON
from filament.yolo_union_export import YOLO_UNION_DIR, write_yolo_union


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=YOLO_UNION_DIR)
    args = parser.parse_args()

    if TRAIN_JSON.name.lower().find("test") >= 0:
        raise SystemExit("Refusing to load a test JSON")

    views, by_file = load_coco(TRAIN_JSON)
    fold_of = grouped_folds(by_file, views, seed=SEED)
    train_files = sorted(fn for fn, fold in fold_of.items() if fold != args.val_fold)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == args.val_fold)

    yaml_path, stats = write_yolo_union(
        views,
        by_file,
        train_files,
        val_files,
        TRAIN_IMG_DIR,
        args.out_dir,
        seed=SEED,
    )

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    val_cmp = stats["val_compare"]
    meta = {
        "val_fold": args.val_fold,
        "mode": stats["mode"],
        "choice": stats["choice"],
        "val_labels": stats["val_labels"],
        "val_note": stats["val_note"],
        "duplicate_note": stats["duplicate_note"],
        "n_train_files": stats["n_train_files"],
        "n_val_files": stats["n_val_files"],
        "n_train_instances": stats["n_train_instances"],
        "n_train_instances_medoid": stats["n_train_instances_medoid"],
        "n_train_instances_delta": stats["n_train_instances_delta"],
        "n_val_instances": stats["n_val_instances"],
        "n_val_labels_identical_to_yolo_seg": val_cmp["n_val_labels_identical_to_yolo_seg"],
        "val_labels_byte_identical_to_yolo_seg": val_cmp["val_labels_byte_identical_to_yolo_seg"],
        "n_train_multi_annotator": stats["n_train_multi_annotator"],
        "n_train_files_union_gt_medoid": stats["n_train_files_union_gt_medoid"],
        "n_cross_rater_iou50_pairs": stats["n_cross_rater_iou50_pairs"],
        "n_train_files_with_cross_rater_duplicate": stats["n_train_files_with_cross_rater_duplicate"],
        "duplicate_iou_threshold": stats["duplicate_iou_threshold"],
        "n_dataset_multi_annotator": stats["n_dataset_multi_annotator"],
        "yaml": str(yaml_path),
        "out_dir": stats["out_dir"],
        "seed": SEED,
        "did_not_touch": stats["did_not_touch"],
        "test_labels_used": False,
        "test_overlap": stats["test_overlap"],
        "role": (
            "climber-plan rank 4 union_mask_s1024. Conditional later look if "
            "all-annotator gains TPs but explodes FPs. NOT fail-path (phys is). "
            "NOT Slot-2. Do not train. Do not submit."
        ),
    }
    (ARTIFACTS / "yolo_union_split.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
