#!/usr/bin/env python3
"""Slot-1 NMS-merge with 3ch U-Net instances. Gated. Never submits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from gate_unet_family import PATCH_GATE, refuse_3ch

DEFAULT_WEIGHTS = WEIGHTS_DIR / "unet_3ch_disk" / "best.pt"
WEIGHTS_MISSING = "weights not trained yet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-refuse", action="store_true")
    parser.add_argument("--gate-json", default=str(PATCH_GATE))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--thrs", default="0.45,0.50,0.55,0.60")
    parser.add_argument("--min-area", type=int, default=642)
    parser.add_argument("--min-areas", default="642,1034")
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
    import torch
    from tqdm import tqdm

    from eval_unet import decode_instances
    from eval_unet_3ch import infer_prob_3ch
    from eval_unet_merge import gate_unet_merge
    from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
    from filament.infer import pick_device
    from filament.merge import merge_rles
    from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
    from filament.unet import build_fulldisk_unet
    from score_faint_merge import load_or_infer_slot1
    from score_family import year_slices

    device = pick_device(args.device)
    thrs = [float(x) for x in args.thrs.split(",") if x.strip()]
    min_areas = sorted({int(x.strip()) for x in str(args.min_areas).split(",") if x.strip()} | {int(args.min_area)})
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == args.fold)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    slot1_frames = load_or_infer_slot1(val_files, device)
    slot1_df = pd.concat(slot1_frames.values(), ignore_index=True)
    slot1_br = score_pq(gt_df, slot1_df)
    slot1_row = {"pq": slot1_br.pq, "tp": slot1_br.tp, "fp": slot1_br.fp, "fn": slot1_br.fn, "mean_tp_iou": slot1_br.mean_tp_iou, "n_pred": int(len(slot1_df))}
    slot1_slices = year_slices(val_files, slot1_frames, views, by_file)
    model = build_fulldisk_unet(in_ch=3, base=args.base)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()
    probs = {}
    for file_name in tqdm(val_files, desc="3ch merge probs"):
        probs[file_name] = infer_prob_3ch(model, TRAIN_IMG_DIR / file_name, args.imgsz, device)
    sweep = []
    pred_cache = {}
    for min_area in min_areas:
        for thr in thrs:
            frames = {}
            n_added = 0
            for file_name in val_files:
                extra = decode_instances(probs[file_name], thr, min_area=min_area)
                slot_rles = slot1_frames[file_name]["segmentation_rle"].tolist()
                merged, added = merge_rles(slot_rles, extra)
                n_added += added
                frames[file_name] = pred_to_df(Path(file_name).stem, merged)
            pred_cache[(thr, min_area)] = frames
            br = score_pq(gt_df, pd.concat(frames.values(), ignore_index=True))
            sweep.append({"thr": thr, "pq": br.pq, "tp": br.tp, "fp": br.fp, "fn": br.fn, "mean_tp_iou": br.mean_tp_iou, "n_pred": int(len(pd.concat(frames.values(), ignore_index=True))), "n_unet_added": n_added, "min_area": min_area})
            print(sweep[-1], flush=True)
    fp_cap = 2 * int(slot1_br.fp)
    pool = [r for r in sweep if int(r["fp"]) < fp_cap] or sweep
    locked = max(pool, key=lambda r: (r["pq"], -abs(r["thr"] - 0.45), r["min_area"]))
    slices = year_slices(val_files, pred_cache[(locked["thr"], locked["min_area"])], views, by_file)
    decision = gate_unet_merge(locked, slot1_row, slices["2017-2020"], slot1_slices["2017-2020"])
    out = {
        "family": "unet_3ch_disk_slot1_merge",
        "weights": str(weights.resolve()),
        "slot1": slot1_row,
        "sweep_merged": sweep,
        "locked": locked["thr"],
        "locked_row": locked,
        "min_areas": min_areas,
        "min_area": locked["min_area"],
        "year_slice": slices,
        "gate": decision,
        "do_not_submit": True,
        "not_yolo_phys": True,
    }
    text = json.dumps(out, indent=2)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    dest = ARTIFACTS / f"unet_3ch_disk_slot1_merge_fold{args.fold}_official.json"
    dest.write_text(text)
    (FAST_ARTIFACTS / dest.name).write_text(text)
    print(json.dumps({"path": str(dest), "locked": locked, "gate": decision}, indent=2))


if __name__ == "__main__":
    main()
