#!/usr/bin/env python3
"""Export match-any H-alpha YOLO labels. Train JSON only. Never submits.

One real rater polygon per filament cluster (IoU>0.5 across annotators).
Does not write yolo_seg_union, yolo_seg, erode, shift, CLAHE, or weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco
from filament.paths import ARTIFACTS, SEED, TRAIN_IMG_DIR, YOLO_PQ_ANY_DIR
from filament.yolo_export import write_yolo_pq_any


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    train_files = sorted(fn for fn, fold in fold_of.items() if fold != 0)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    yaml_path, stats = write_yolo_pq_any(
        views, by_file, train_files, val_files, TRAIN_IMG_DIR, YOLO_PQ_ANY_DIR
    )
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    meta = {
        "val_fold": 0,
        "mode": stats["mode"],
        "family": stats["family"],
        "choice": stats["choice"],
        "val_labels": stats["val_labels"],
        "official_score": stats["official_score"],
        "pixels": stats["pixels"],
        "n_train_files": stats["n_train_files"],
        "n_val_files": stats["n_val_files"],
        "train": stats["train"],
        "val": stats["val"],
        "multi_rater_kept_over_raw_train": stats["multi_rater_kept_over_raw_train"],
        "images_labels_match": stats["images_labels_match"],
        "test_stem_overlap": stats["test_stem_overlap"],
        "test_labels_used": False,
        "val_files_match_yolo_seg": stats["val_files_match_yolo_seg"],
        "yaml": str(yaml_path),
        "out_dir": stats["out_dir"],
        "seed": SEED,
        "extras_or": False,
        "did_not_touch": stats["did_not_touch"],
        "do_not_submit": True,
        "kaggle_submit_called": False,
    }
    (ARTIFACTS / "yolo_seg_pq_any_export.json").write_text(json.dumps(meta, indent=2) + "\n")
    if not stats["val_files_match_yolo_seg"]:
        raise SystemExit("pq_any val files do not match yolo_seg fold-0")
    ratio = stats["multi_rater_kept_over_raw_train"]
    print(json.dumps({k: meta[k] for k in meta if k != "did_not_touch"}, indent=2))
    print(f"collapse_ratio_train_multi {ratio}")


if __name__ == "__main__":
    main()
