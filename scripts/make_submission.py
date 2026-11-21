#!/usr/bin/env python3
"""Build an upload-ready submission.csv from YOLO (preferred) or classical fallback.

Never reads MAGFiLO public-test annotations. Inference uses only test JPEGs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.classical import ClassicalConfig
from filament.infer import infer_classical_image, load_refiner, pick_device, run_yolo_infer, weights_tag
from filament.metric import pred_to_df
from filament.paths import FAST_ARTIFACTS, FAST_SUBMISSION_DIR, TEST_IMG_DIR, WEIGHTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("auto", "yolo", "classical"), default="auto")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--tta", action="store_true", help="4-flip TTA with IoU clustering")
    parser.add_argument("--clahe", action="store_true", help="CLAHE the JPEG before YOLO (must match a CLAHE-trained family)")
    parser.add_argument("--phys-stack", dest="phys", action="store_true", help="gray+CLAHE+residual stack (must match yolo_seg_phys)")
    parser.add_argument("--tiles", action="store_true", help="Native-2048 tile stitch (must match yolo_seg_tiles)")
    parser.add_argument("--merge-gap", type=int, default=0)
    parser.add_argument("--refiner", default="", help="Path to TinyUNet refiner.pt")
    parser.add_argument("--device", default="auto", help="cpu | mps | cuda | auto")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    if args.tta and args.refiner:
        print("warning: --refiner overrides --tta (not combined)", file=sys.stderr)
    if args.weights and "tile" in Path(args.weights).as_posix().lower():
        args.tiles = True
        if args.imgsz == 1024:
            args.imgsz = 640
    if args.tiles and (args.phys or args.clahe or args.tta or args.refiner):
        raise SystemExit("tile stitch is native crops only")

    default_weights = WEIGHTS_DIR / "yolo11s_seg" / "weights" / "best.pt"
    weights = Path(args.weights) if args.weights else default_weights
    use_yolo = args.mode == "yolo" or (args.mode == "auto" and weights.exists())
    if args.mode == "yolo" and not weights.exists():
        raise FileNotFoundError(weights)

    model = None
    refiner = None
    device = pick_device(args.device)
    if use_yolo:
        from ultralytics import YOLO

        model = YOLO(str(weights))
        if args.refiner:
            refiner = load_refiner(args.refiner, device)

    cfg = ClassicalConfig()
    images = sorted(TEST_IMG_DIR.glob("*.jpeg"))
    if not images:
        raise FileNotFoundError(TEST_IMG_DIR)

    frames = []
    for path in tqdm(images, desc="test infer"):
        if model is not None:
            rles = run_yolo_infer(
                model,
                path,
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
        else:
            rles = infer_classical_image(path, cfg)
        if not rles:
            # Official PQ treats empty images as FN, not as a 1.0 hack.
            # Leave empty: no rows for this stem.
            continue
        frames.append(pred_to_df(path.stem, rles))

    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame(columns=["filament_id", "segmentation_rle"])

    FAST_SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if args.out:
        out = Path(args.out)
    else:
        if refiner is not None:
            mode = "refiner"
        elif args.tta:
            mode = "tta"
        elif args.tiles:
            mode = "tiles"
        else:
            mode = f"m{args.merge_gap}"
        name = (
            f"submission_{weights_tag(weights)}_conf{args.conf:.2f}"
            f"_sz{args.imgsz}_{mode}.csv"
            if model is not None
            else "submission_classical.csv"
        )
        out = FAST_SUBMISSION_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    meta = {
        "mode": "yolo" if model is not None else "classical",
        "device": device,
        "n_images": len(images),
        "n_rows": int(len(df)),
        "n_images_with_preds": int(df["filament_id"].str.rsplit("_", n=1).str[0].nunique()) if len(df) else 0,
        "path": str(out),
        "weights": str(weights) if model is not None else None,
        "conf": args.conf if model is not None else None,
        "imgsz": args.imgsz if model is not None else None,
        "tta": args.tta,
        "clahe": args.clahe,
        "phys_stack": args.phys,
        "tiles": args.tiles,
        "augment": args.augment,
        "merge_gap": args.merge_gap,
        "refiner": args.refiner or None,
        "leakage": "test JPEGs only; no MAGFiLO public-test labels",
    }
    meta_path = FAST_ARTIFACTS / f"{out.stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    meta["meta_path"] = str(meta_path)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
