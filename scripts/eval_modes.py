#!/usr/bin/env python3
"""Official-PQ ablation of inference algorithms on the grouped holdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.infer import load_refiner, pick_device, run_yolo_infer, weights_tag
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-val",
        type=int,
        default=24,
        help="Cap holdout images. 0 = full grouped fold (142 JPEGs on fold 0).",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--weights", default=str(WEIGHTS_DIR / "yolo11s_seg" / "weights" / "best.pt"))
    parser.add_argument("--refiner", default="", help="If set, also score crop-UNet refine on the detector")
    parser.add_argument("--device", default="auto", help="cpu | mps | cuda | auto")
    parser.add_argument("--out-dir", default=str(FAST_ARTIFACTS))
    args = parser.parse_args()

    from ultralytics import YOLO

    device = pick_device(args.device)
    model = YOLO(args.weights)
    refiner = load_refiner(args.refiner, device) if args.refiner else None
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == args.fold)
    if args.max_val:
        val_files = val_files[: args.max_val]
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))

    modes = [
        ("baseline_1024", dict(imgsz=1024, augment=False, merge_gap=0, tta=False, refine=False)),
        ("imgsz_1280", dict(imgsz=1280, augment=False, merge_gap=0, tta=False, refine=False)),
        ("ultralytics_tta", dict(imgsz=1024, augment=True, merge_gap=0, tta=False, refine=False)),
        ("flip_tta_1024", dict(imgsz=1024, augment=False, merge_gap=0, tta=True, refine=False)),
        ("flip_tta_1280", dict(imgsz=1280, augment=False, merge_gap=0, tta=True, refine=False)),
        ("merge_gap9_1024", dict(imgsz=1024, augment=False, merge_gap=9, tta=False, refine=False)),
        ("merge_gap9_1280", dict(imgsz=1280, augment=False, merge_gap=9, tta=False, refine=False)),
    ]
    if refiner is not None:
        modes.append(
            ("crop_refiner_1024", dict(imgsz=1024, augment=False, merge_gap=0, tta=False, refine=True))
        )
        modes.append(
            ("crop_refiner_1280", dict(imgsz=1280, augment=False, merge_gap=0, tta=False, refine=True))
        )

    rows = []
    for name, spec in modes:
        frames = []
        for file_name in tqdm(val_files, desc=name):
            rles = run_yolo_infer(
                model,
                TRAIN_IMG_DIR / file_name,
                conf=args.conf,
                imgsz=spec["imgsz"],
                augment=spec["augment"],
                merge_gap=spec["merge_gap"],
                tta=spec["tta"],
                refiner=refiner if spec["refine"] else None,
                device=device,
            )
            frames.append(pred_to_df(Path(file_name).stem, rles))
        pred_df = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
        )
        br = score_pq(gt_df, pred_df)
        row = {
            "mode": name,
            "weights": args.weights,
            "device": device,
            "conf": args.conf,
            "n_val": len(val_files),
            "n_pred": int(len(pred_df)),
            "pq": br.pq,
            "tp": br.tp,
            "fp": br.fp,
            "fn": br.fn,
            "mean_tp_iou": br.mean_tp_iou,
            "metric": "official host PQ (IoU>0.5, not Hungarian)",
            **{k: v for k, v in spec.items() if k != "refine"},
            "refiner": args.refiner if spec["refine"] else None,
        }
        rows.append(row)
        print(json.dumps(row))

    rows.sort(key=lambda r: r["pq"], reverse=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"infer_modes_{weights_tag(args.weights)}_fold{args.fold}.json"
    path.write_text(json.dumps(rows, indent=2))
    print("BEST", json.dumps(rows[0]))
    print("path", path)


if __name__ == "__main__":
    main()
