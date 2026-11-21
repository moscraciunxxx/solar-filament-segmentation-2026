#!/usr/bin/env python3
"""Grouped-holdout PQ for a trained YOLO-seg checkpoint (official matching)."""

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


def _mode_tag(args: argparse.Namespace) -> str:
    if args.refiner:
        return "refiner"
    if args.tta:
        return "tta"
    if args.phys:
        return "phys"
    if args.clahe:
        return "clahe"
    if getattr(args, "tiles", False):
        return "tiles"
    if args.augment:
        return "aug"
    return "base"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(WEIGHTS_DIR / "yolo11s_seg" / "weights" / "best.pt"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--max-val",
        type=int,
        default=48,
        help="Cap holdout images. 0 = full grouped fold (142 JPEGs on fold 0).",
    )
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--tta", action="store_true", help="4-flip TTA with IoU clustering")
    parser.add_argument("--clahe", action="store_true", help="CLAHE the JPEG before YOLO (must match a CLAHE-trained family)")
    parser.add_argument("--phys-stack", dest="phys", action="store_true", help="gray+CLAHE+residual stack (must match yolo_seg_phys)")
    parser.add_argument("--tiles", action="store_true", help="Native-2048 tile stitch (not full-frame YOLO)")
    parser.add_argument("--merge-gap", type=int, default=0)
    parser.add_argument("--refiner", default="", help="Path to TinyUNet refiner.pt")
    parser.add_argument("--device", default="auto", help="cpu | mps | cuda | auto")
    parser.add_argument("--out-dir", default=str(FAST_ARTIFACTS))
    parser.add_argument("--tag", default="", help="Optional extra filename stem")
    parser.add_argument("--save-pred", action="store_true", help="Also write pred CSV next to the PQ JSON")
    args = parser.parse_args()

    if args.tta and args.refiner:
        print("warning: --refiner overrides --tta (not combined)", file=sys.stderr)
    if "tile" in Path(args.weights).as_posix().lower() or "tile" in args.tag.lower():
        args.tiles = True
        if args.imgsz == 1024:
            args.imgsz = 640
    if args.tiles and (args.phys or args.clahe or args.tta or args.refiner):
        raise SystemExit("tile stitch is native crops only")

    from ultralytics import YOLO

    device = pick_device(args.device)
    model = YOLO(args.weights)
    refiner = load_refiner(args.refiner, device) if args.refiner else None

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == args.fold)
    if args.max_val:
        val_files = val_files[: args.max_val]

    frames = []
    for file_name in tqdm(val_files, desc="yolo val"):
        rles = run_yolo_infer(
            model,
            TRAIN_IMG_DIR / file_name,
            conf=args.conf,
            imgsz=args.imgsz,
            augment=args.augment,
            merge_gap=args.merge_gap,
            tta=args.tta,
            refiner=refiner,
            device=device,
            clahe=args.clahe,
            phys=args.phys,
            tiles=args.tiles,
        )
        frames.append(pred_to_df(Path(file_name).stem, rles))
    pred_df = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    )
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    br = score_pq(gt_df, pred_df)
    out = {
        "weights": args.weights,
        "device": device,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "tta": args.tta,
        "clahe": args.clahe,
        "phys_stack": args.phys,
        "tiles": args.tiles,
        "augment": args.augment,
        "merge_gap": args.merge_gap,
        "refiner": args.refiner or None,
        "fold": args.fold,
        "n_val": len(val_files),
        "n_pred": int(len(pred_df)),
        "pq": br.pq,
        "tp": br.tp,
        "fp": br.fp,
        "fn": br.fn,
        "mean_tp_iou": br.mean_tp_iou,
        "metric": "official host PQ (IoU>0.5, not Hungarian)",
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = f"_{args.tag}" if args.tag else ""
    stem = (
        f"yolo_{weights_tag(args.weights)}_fold{args.fold}_conf{args.conf:.2f}"
        f"_sz{args.imgsz}_m{args.merge_gap}_{_mode_tag(args)}{extra}_pq"
    )
    path = out_dir / f"{stem}.json"
    path.write_text(json.dumps(out, indent=2))
    out["path"] = str(path)
    if args.save_pred:
        pred_path = out_dir / f"{stem}.csv"
        pred_df.to_csv(pred_path, index=False)
        out["pred_csv"] = str(pred_path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
