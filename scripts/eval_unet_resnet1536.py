#!/usr/bin/env python3
"""Official fold-0 PQ for ResNet34 U-Net @1536. Never overwrites 1024 locks.

Writes artifacts/unet_resnet34_1536_fold0_official.json only.
Never writes unet_resnet34_fold0_official.json. Never submits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR

DEST_NAME = "unet_resnet34_1536"
FORBIDDEN = "unet_resnet34"
DEFAULT_WEIGHTS = WEIGHTS_DIR / DEST_NAME / "best.pt"
CLOSED_OFFICIAL = "unet_resnet34_fold"
WEIGHTS_MISSING = "1536 weights not trained yet"


def _refuse_paths(weights: Path, out_name: str) -> str | None:
    if DEST_NAME not in Path(weights).parts:
        return f"Refusing 1536 eval: weights must live under {DEST_NAME} (got {weights})."
    if Path(weights).parent.name == FORBIDDEN:
        return f"Refusing 1536 eval: would load locked 1024 weights {weights}."
    if CLOSED_OFFICIAL in out_name and DEST_NAME not in out_name:
        return f"Refusing 1536 eval: output {out_name} would overwrite the 1024 official."
    if DEST_NAME not in out_name:
        return f"Refusing 1536 eval: output {out_name} must contain {DEST_NAME}."
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--thrs", default="0.35,0.40,0.45,0.50")
    parser.add_argument("--min-area", type=int, default=642)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    out_name = f"{DEST_NAME}_fold{args.fold}_official.json"
    reason = _refuse_paths(Path(args.weights), out_name)
    if reason:
        print(reason, file=sys.stderr)
        raise SystemExit(2)
    weights = Path(args.weights)
    if not weights.exists():
        print(WEIGHTS_MISSING, file=sys.stderr)
        raise SystemExit(2)

    import pandas as pd
    import torch
    from tqdm import tqdm

    from eval_unet import decode_instances, gate_unet, infer_prob
    from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
    from filament.infer import pick_device
    from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
    from filament.unet_resnet import build_resnet34_unet
    from score_family import year_slices

    device = pick_device(args.device)
    thrs = [float(x) for x in args.thrs.split(",") if x.strip()]
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == args.fold)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))

    model = build_resnet34_unet(in_ch=3, pretrained=False)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()

    probs = {}
    for file_name in tqdm(val_files, desc="resnet34_1536 probs"):
        probs[file_name] = infer_prob(model, TRAIN_IMG_DIR / file_name, args.imgsz, device)

    sweep = []
    pred_cache = {}
    for thr in thrs:
        frames = {}
        for file_name in val_files:
            rles = decode_instances(probs[file_name], thr, min_area=int(args.min_area))
            frames[file_name] = pred_to_df(Path(file_name).stem, rles)
        pred_cache[thr] = frames
        pred_df = pd.concat(frames.values(), ignore_index=True)
        br = score_pq(gt_df, pred_df)
        sweep.append(
            {
                "thr": thr,
                "pq": br.pq,
                "tp": br.tp,
                "fp": br.fp,
                "fn": br.fn,
                "mean_tp_iou": br.mean_tp_iou,
                "n_pred": int(len(pred_df)),
                "min_area": int(args.min_area),
            }
        )
        print(sweep[-1], flush=True)

    locked = max(sweep, key=lambda r: (r["pq"], -abs(r["thr"] - 0.45)))
    slices = year_slices(val_files, pred_cache[locked["thr"]], views, by_file)
    decision = gate_unet(locked)
    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = {
        "family": "unet_resnet34_s1536",
        "weights": str(weights.resolve()),
        "builder": "build_resnet34_unet",
        "not_tinyunet": True,
        "overwrote_unet_resnet34_official": False,
        "fold": args.fold,
        "imgsz": int(args.imgsz),
        "min_area": int(args.min_area),
        "n_val": len(val_files),
        "sweep": sweep,
        "locked": locked["thr"],
        "locked_row": locked,
        "year_slice": slices,
        "gate": decision,
        "do_not_submit": True,
    }
    text = json.dumps(out, indent=2)
    dest = ARTIFACTS / out_name
    if dest.name == "unet_resnet34_fold0_official.json":
        raise SystemExit("Refusing 1536 eval: dest collapsed to 1024 official.")
    dest.write_text(text)
    (FAST_ARTIFACTS / out_name).write_text(text)
    print(json.dumps({"path": str(dest), "locked": locked, "gate": decision}, indent=2))


if __name__ == "__main__":
    main()
