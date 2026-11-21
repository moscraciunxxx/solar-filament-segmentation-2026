#!/usr/bin/env python3
"""Official fold-0 PQ for 3ch disk+CLAHE+Sobel TinyUNet. Gated. Never submits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.paths import ARTIFACTS, FAST_ARTIFACTS, IMAGE_SIZE, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.preprocess import read_gray
from gate_unet_family import PATCH_GATE, refuse_3ch

DEFAULT_WEIGHTS = WEIGHTS_DIR / "unet_3ch_disk" / "best.pt"
WEIGHTS_MISSING = "weights not trained yet"


def infer_prob_3ch(model, path: Path, imgsz: int, device: str) -> np.ndarray:
    from filament.classical import disk_clahe_sobel_stack

    gray = read_gray(path)
    hwc = disk_clahe_sobel_stack(gray)
    small = cv2.resize(hwc, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.transpose(small.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(tensor)[0, 0].detach().cpu().numpy()
    prob = 1.0 / (1.0 + np.exp(-logit))
    return cv2.resize(prob.astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-refuse", action="store_true")
    parser.add_argument("--gate-json", default=str(PATCH_GATE))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--thrs", default="0.35,0.40,0.45,0.50")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--base", type=int, default=32)
    args = parser.parse_args()

    reason = refuse_3ch(Path(args.gate_json))
    if args.dry_refuse:
        if reason:
            print(reason)
        raise SystemExit(0)
    if reason:
        print(reason, file=sys.stderr)
        raise SystemExit(2)
    weights = Path(args.weights)
    if not weights.exists():
        print(WEIGHTS_MISSING, file=sys.stderr)
        raise SystemExit(2)

    import pandas as pd
    from tqdm import tqdm

    from eval_unet import decode_instances, gate_unet
    from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
    from filament.infer import pick_device
    from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
    from filament.unet import build_fulldisk_unet
    from score_family import year_slices

    device = pick_device(args.device)
    thrs = [float(x) for x in args.thrs.split(",") if x.strip()]
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == args.fold)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    model = build_fulldisk_unet(in_ch=3, base=args.base)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()
    probs = {}
    for file_name in tqdm(val_files, desc="3ch probs"):
        probs[file_name] = infer_prob_3ch(model, TRAIN_IMG_DIR / file_name, args.imgsz, device)
    sweep = []
    pred_cache = {}
    for thr in thrs:
        frames = {fn: pred_to_df(Path(fn).stem, decode_instances(probs[fn], thr)) for fn in val_files}
        pred_cache[thr] = frames
        br = score_pq(gt_df, pd.concat(frames.values(), ignore_index=True))
        sweep.append({"thr": thr, "pq": br.pq, "tp": br.tp, "fp": br.fp, "fn": br.fn, "mean_tp_iou": br.mean_tp_iou, "n_pred": int(br.tp + br.fp)})
        print(sweep[-1], flush=True)
    locked = max(sweep, key=lambda r: (r["pq"], -abs(r["thr"] - 0.45)))
    decision = gate_unet(locked)
    slices = year_slices(val_files, pred_cache[locked["thr"]], views, by_file)
    out = {
        "family": "unet_3ch_disk_clahe_sobel",
        "weights": str(weights.resolve()),
        "locked": locked["thr"],
        "locked_row": locked,
        "sweep": sweep,
        "year_slice": slices,
        "gate": decision,
        "do_not_submit": True,
        "not_yolo_phys": True,
    }
    text = json.dumps(out, indent=2)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"unet_3ch_disk_fold{args.fold}_official.json").write_text(text)
    (FAST_ARTIFACTS / f"unet_3ch_disk_fold{args.fold}_official.json").write_text(text)
    print(json.dumps({"path": str(ARTIFACTS / f"unet_3ch_disk_fold{args.fold}_official.json"), "locked": locked, "gate": decision}, indent=2))


if __name__ == "__main__":
    main()
