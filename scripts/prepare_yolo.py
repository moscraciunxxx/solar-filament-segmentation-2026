#!/usr/bin/env python3
"""Export a leakage-safe YOLO-seg dataset using medoid annotators and grouped folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco
from filament.paths import (
    ARTIFACTS,
    SEED,
    TRAIN_IMG_DIR,
    YOLO_ALL_DIR,
    YOLO_CLAHE_DIR,
    YOLO_DIR,
    YOLO_ERODE_GT_DIR,
    YOLO_FAINT_DIR,
    YOLO_PHYS_DIR,
    YOLO_RESIDUAL_DIR,
    YOLO_RATER_DIR,
    YOLO_SHIFT_DIR,
    YOLO_TILE_DIR,
)
from filament.yolo_export import (
    write_yolo_all_annotators,
    write_yolo_clahe,
    write_yolo_erode_gt,
    write_yolo_faint,
    write_yolo_phys_stack,
    write_yolo_residual_stack,
    write_yolo_random_rater,
    write_yolo_split,
    write_yolo_tiles,
    write_yolo_year_upsample,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--all-annotators", action="store_true")
    parser.add_argument(
        "--random-rater",
        action="store_true",
        help="Train: one random annotator per JPEG (seed 20260910). Val stays medoid. Writes yolo_seg_rater, not yolo_seg.",
    )
    parser.add_argument("--year-upsample", action="store_true")
    parser.add_argument("--clahe", action="store_true", help="CLAHE copy of existing medoid yolo_seg (does not touch yolo_seg)")
    parser.add_argument("--phys-stack", action="store_true", help="gray+CLAHE+residual copy of medoid yolo_seg")
    parser.add_argument(
        "--residual-stack",
        action="store_true",
        help="flatten+raw+Sobel copy of medoid yolo_seg into yolo_seg_residual. Does not touch yolo_seg_phys.",
    )
    parser.add_argument(
        "--tiles",
        action="store_true",
        help="Native-2048 GT-centric tiles + 2 hard-neg/JPEG into yolo_seg_tiles. Does not touch live yolo_seg*.",
    )
    parser.add_argument(
        "--faint",
        action="store_true",
        help="Medoid JPEGs, small/unidentifiable labels only, into yolo_seg_faint. Does not touch live yolo_seg*.",
    )
    parser.add_argument(
        "--erode-gt",
        action="store_true",
        help="Medoid instance masks with Slot1 erode 3x3x1 into yolo_seg_erode_gt. Does not touch live yolo_seg*.",
    )
    parser.add_argument("--overwrite-medoid", action="store_true")
    args = parser.parse_args()
    other_modes = [
        args.all_annotators,
        args.random_rater,
        args.year_upsample,
        args.clahe,
        args.phys_stack,
        args.residual_stack,
        args.tiles,
        args.faint,
    ]
    if args.erode_gt and any(other_modes):
        raise SystemExit("--erode-gt refuses any other mode flag")
    n_modes = sum([*other_modes, args.erode_gt])
    if n_modes > 1:
        raise SystemExit("Choose one export mode")

    if args.residual_stack:
        yaml_path = write_yolo_residual_stack(src_dir=YOLO_DIR, out_dir=YOLO_RESIDUAL_DIR)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        meta = {
            "val_fold": args.val_fold,
            "residual_stack": True,
            "channels": ["median_flatten", "raw", "sobel"],
            "src": str(YOLO_DIR),
            "yaml": str(yaml_path),
            "seed": SEED,
            "not_phys_stack": True,
            "overwrote_yolo_seg_phys": False,
            "note": "Medoid labels + Method 2 BGR. Infer with --residual-stack. Did not rmtree yolo_seg or yolo_seg_phys.",
        }
        (ARTIFACTS / "yolo_residual_split.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
        return

    if args.phys_stack:
        yaml_path = write_yolo_phys_stack(src_dir=YOLO_DIR, out_dir=YOLO_PHYS_DIR)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        meta = {
            "val_fold": args.val_fold,
            "phys_stack": True,
            "channels": ["gray", "clahe", "dark_residual"],
            "src": str(YOLO_DIR),
            "yaml": str(yaml_path),
            "seed": SEED,
            "note": "Medoid labels + physics BGR. Infer with --phys-stack. Did not rmtree yolo_seg.",
        }
        (ARTIFACTS / "yolo_phys_split.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
        return

    if args.clahe:
        yaml_path = write_yolo_clahe(src_dir=YOLO_DIR, out_dir=YOLO_CLAHE_DIR)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        meta = {
            "val_fold": args.val_fold,
            "clahe": True,
            "clip_limit": 2.0,
            "tile": 8,
            "src": str(YOLO_DIR),
            "yaml": str(yaml_path),
            "seed": SEED,
            "note": "Medoid labels + CLAHE pixels. Infer with --clahe. Did not rmtree yolo_seg.",
        }
        (ARTIFACTS / "yolo_clahe_split.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
        return

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    train_files = sorted(fn for fn, fold in fold_of.items() if fold != args.val_fold)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == args.val_fold)
    if args.tiles:
        yaml_path, stats = write_yolo_tiles(
            views, by_file, train_files, val_files, TRAIN_IMG_DIR, YOLO_TILE_DIR, seed=SEED
        )
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        meta = {
            "val_fold": args.val_fold,
            "family": "tile_native_s640",
            "mode": stats["mode"],
            "n_train_tiles": stats["n_train_tiles"],
            "n_val_tiles": stats["n_val_tiles"],
            "n_hard_neg": stats["n_hard_neg"],
            "n_val_hard_neg": stats["n_val_hard_neg"],
            "n_train_gt_tiles": stats["n_train_gt_tiles"],
            "n_val_gt_tiles": stats["n_val_gt_tiles"],
            "n_train_source_jpegs": stats["n_train_source_jpegs"],
            "n_val_source_jpegs": stats["n_val_source_jpegs"],
            "tile": stats["tile"],
            "overlap": stats["overlap"],
            "stride": stats["stride"],
            "later_imgsz": stats["later_imgsz"],
            "labels": stats["labels"],
            "val_recipe": stats["val_recipe"],
            "images_labels_match": stats["images_labels_match"],
            "test_stem_overlap": stats["test_stem_overlap"],
            "test_labels_used": False,
            "val_files_match_yolo_seg": stats["val_files_match_yolo_seg"],
            "yaml": str(yaml_path),
            "out_dir": stats["out_dir"],
            "seed": SEED,
            "merge_gap": 0,
            "cc_split": False,
            "did_not_touch": stats["did_not_touch"],
            "stitch_eval_contract": stats["stitch_eval_contract"],
            "do_not_train": True,
            "do_not_submit": True,
        }
        (ARTIFACTS / "yolo_tile_split.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
        return
    if args.faint:
        yaml_path, stats = write_yolo_faint(
            views, by_file, train_files, val_files, TRAIN_IMG_DIR, YOLO_FAINT_DIR
        )
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        meta = {
            "val_fold": args.val_fold,
            "family": stats["family"],
            "mode": stats["mode"],
            "area_p25": stats["area_p25"],
            "unidentifiable_category_id": stats["unidentifiable_category_id"],
            "keep_rule": stats["keep_rule"],
            "n_train_files": stats["n_train_files"],
            "n_val_files": stats["n_val_files"],
            "train": stats["train"],
            "val": stats["val"],
            "images_labels_match": stats["images_labels_match"],
            "test_stem_overlap": stats["test_stem_overlap"],
            "test_labels_used": False,
            "val_files_match_yolo_seg": stats["val_files_match_yolo_seg"],
            "yaml": str(yaml_path),
            "out_dir": stats["out_dir"],
            "seed": SEED,
            "did_not_touch": stats["did_not_touch"],
            "infer": stats["infer"],
            "official_pq": stats["official_pq"],
            "do_not_submit_specialist_alone": True,
        }
        (ARTIFACTS / "yolo_faint_split.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
        return
    if args.erode_gt:
        yaml_path, stats = write_yolo_erode_gt(
            views, by_file, train_files, val_files, TRAIN_IMG_DIR, YOLO_ERODE_GT_DIR
        )
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        meta = {
            "val_fold": args.val_fold,
            "mode": stats["mode"],
            "family": stats["family"],
            "op": stats["op"],
            "k": stats["k"],
            "iters": stats["iters"],
            "kernel": stats["kernel"],
            "same_as_slot1_erode": stats["same_as_slot1_erode"],
            "n_train": stats["n_train"],
            "n_val": stats["n_val"],
            "train": stats["train"],
            "val": stats["val"],
            "images_labels_match": stats["images_labels_match"],
            "test_stem_overlap": stats["test_stem_overlap"],
            "val_files_match_yolo_seg": stats["val_files_match_yolo_seg"],
            "yaml": str(yaml_path),
            "data_yaml_exists": bool(stats["data_yaml_exists"]),
            "dest": stats["dest"],
            "out_dir": stats["out_dir"],
            "seed": SEED,
            "did_not_touch": stats["did_not_touch"],
            "closed_dest_written": False,
            "do_not_train_now": True,
            "do_not_submit": True,
        }
        (ARTIFACTS / "yolo_seg_erode_gt_export.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta, indent=2))
        return
    if args.random_rater:
        yaml_path, stats = write_yolo_random_rater(
            views, by_file, train_files, val_files, TRAIN_IMG_DIR, YOLO_RATER_DIR, seed=SEED
        )
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        meta = {
            "val_fold": args.val_fold,
            "mode": stats["mode"],
            "choice": stats["choice"],
            "val_labels": stats["val_labels"],
            "val_note": stats["val_note"],
            "n_train_files": stats["n_train_files"],
            "n_val_files": stats["n_val_files"],
            "n_train_multi_annotator": stats["n_train_multi_annotator"],
            "n_train_differ_from_medoid": stats["n_train_differ_from_medoid"],
            "n_train_more_instances": stats["n_train_more_instances"],
            "n_train_fewer_instances": stats["n_train_fewer_instances"],
            "n_train_same_count_different_polygons": stats["n_train_same_count_different_polygons"],
            "n_instances_added": stats["n_instances_added"],
            "n_instances_removed": stats["n_instances_removed"],
            "n_dataset_multi_annotator": stats["n_dataset_multi_annotator"],
            "yaml": str(yaml_path),
            "out_dir": stats["out_dir"],
            "seed": SEED,
            "did_not_touch": [str(YOLO_DIR), str(YOLO_ALL_DIR)],
            "test_labels_used": False,
            "n_train_differ_files": [
                row["file_name"] for row in stats["train_assignments"] if row["differ_from_medoid"]
            ],
        }
        (ARTIFACTS / "yolo_rater_split.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps({k: v for k, v in meta.items() if k != "n_train_differ_files"}, indent=2))
        return
    if args.all_annotators:
        yaml_path = write_yolo_all_annotators(
            views, by_file, train_files, val_files, TRAIN_IMG_DIR, YOLO_ALL_DIR
        )
        meta_name = "yolo_all_split.json"
    elif args.year_upsample:
        yaml_path = write_yolo_year_upsample(
            views, by_file, train_files, val_files, TRAIN_IMG_DIR, YOLO_SHIFT_DIR
        )
        meta_name = "yolo_shift_split.json"
    else:
        if YOLO_DIR.exists() and not args.overwrite_medoid:
            raise SystemExit(
                f"Refusing to rebuild {YOLO_DIR} while a train may be using it. "
                "Pass --overwrite-medoid, --all-annotators, --random-rater, or --year-upsample."
            )
        yaml_path = write_yolo_split(views, by_file, train_files, val_files, TRAIN_IMG_DIR, YOLO_DIR)
        meta_name = "yolo_split.json"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    meta = {
        "val_fold": args.val_fold,
        "n_train_files": len(train_files),
        "n_val_files": len(val_files),
        "all_annotators": args.all_annotators,
        "year_upsample": args.year_upsample,
        "yaml": str(yaml_path),
        "seed": SEED,
    }
    (ARTIFACTS / meta_name).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
