#!/usr/bin/env python3
"""Official fold-0 PQ: frozen Slot 1 NMS-merged with full-disk U-Net instances.

Standalone U-Net PQ (eval_unet.py) is diagnosis. This merge is the Slot-1
replace path: keep m@1280 objects, add U-Net proposals YOLO never boxed.
Never submits. Never reads test labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_unet import decode_instances, infer_prob
from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.infer import pick_device
from filament.merge import MERGE_IOU, merge_rles
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.unet import build_fulldisk_unet
from score_faint_merge import SLOT1_PRED_CSV, _stem_frames, load_or_infer_slot1
from score_family import M_1280, S_MEDOID, year_slices


def gate_unet_merge(merged: dict, slot1: dict, desert: dict, slot1_desert: dict) -> dict:
    pq = float(merged["pq"])
    tp = int(merged["tp"])
    d_pq = pq - float(slot1["pq"])
    d_tp = tp - int(slot1["tp"])
    desert_drop = float(slot1_desert["pq"]) - float(desert["pq"])
    fp_exploded = int(merged["fp"]) >= 2 * int(slot1["fp"])
    if pq >= 0.42 and desert_drop < 0.05 and not fp_exploded:
        decision = "PROMOTE_LOOK"
        next_action = (
            "Slot-1 + U-Net merge cleared 0.42 without a solar-min dump. "
            "After quota reset, at most one public look of the MERGED csv. Year-shift stays Slot 2."
        )
    elif d_tp <= 0:
        decision = "KILL_NO_NEW_TP"
        next_action = "U-Net added no new TPs vs Slot 1. Same crop-refiner failure. Do not upload."
    elif d_pq <= -0.05 or desert_drop >= 0.05:
        decision = "KILL_WORSE_OR_DESERT"
        next_action = "Merged PQ worse or 2017-2020 dump. Kill. Do not upload. Do not replace Slot 2."
    elif 0.38 <= pq < 0.42:
        decision = "RESEARCH_HOLD_SPINE_V2"
        next_action = "TP movement without Slot-1 replace. Spine-split only if fragments show."
    else:
        decision = "RESEARCH_HOLD"
        next_action = "Diagnosis only. Do not upload. Year-shift stays Slot 2."
    return {
        "decision": decision,
        "next_action": next_action,
        "vs_slot1": {"d_pq": d_pq, "d_tp": d_tp, "fp_exploded": fp_exploded},
        "desert_2017_2020_drop": desert_drop,
        "merge_iou": MERGE_IOU,
        "do_not_submit_from_this_script": True,
        "do_not_replace_slot2": True,
        "baselines": {"s_medoid": S_MEDOID, "m_1280": M_1280},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(WEIGHTS_DIR / "unet_semantic" / "best.pt"))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--thrs", default="0.35,0.40,0.45,0.50")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--base", type=int, default=32)
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(weights)
    device = pick_device(args.device)
    thrs = [float(x) for x in args.thrs.split(",") if x.strip()]

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == args.fold)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))

    slot1_frames = load_or_infer_slot1(val_files, device)
    slot1_df = pd.concat(slot1_frames.values(), ignore_index=True)
    slot1_br = score_pq(gt_df, slot1_df)
    slot1_row = {
        "pq": slot1_br.pq,
        "tp": slot1_br.tp,
        "fp": slot1_br.fp,
        "fn": slot1_br.fn,
        "mean_tp_iou": slot1_br.mean_tp_iou,
        "n_pred": int(len(slot1_df)),
    }
    slot1_slices = year_slices(val_files, slot1_frames, views, by_file)

    model = build_fulldisk_unet(base=args.base)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()

    probs = {}
    for file_name in tqdm(val_files, desc="unet probs"):
        probs[file_name] = infer_prob(model, TRAIN_IMG_DIR / file_name, args.imgsz, device)

    sweep = []
    pred_cache = {}
    for thr in thrs:
        frames = {}
        n_added = 0
        for file_name in val_files:
            extra = decode_instances(probs[file_name], thr)
            slot_rles = slot1_frames[file_name]["segmentation_rle"].tolist()
            merged, added = merge_rles(slot_rles, extra)
            n_added += added
            frames[file_name] = pred_to_df(Path(file_name).stem, merged)
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
                "n_unet_added": n_added,
            }
        )
        print(sweep[-1], flush=True)

    locked = max(sweep, key=lambda r: (r["pq"], -abs(r["thr"] - 0.45)))
    slices = year_slices(val_files, pred_cache[locked["thr"]], views, by_file)
    decision = gate_unet_merge(locked, slot1_row, slices["2017-2020"], slot1_slices["2017-2020"])

    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    pred_csv = FAST_ARTIFACTS / f"unet_slot1_merge_fold{args.fold}_thr{locked['thr']:.2f}_pred.csv"
    pd.concat(pred_cache[locked["thr"]].values(), ignore_index=True).to_csv(pred_csv, index=False)
    out = {
        "family": "unet_slot1_merge",
        "weights": str(weights.resolve()),
        "slot1_csv": str(SLOT1_PRED_CSV),
        "metric": "official host PQ after Slot-1 NMS-merge with U-Net CC instances",
        "fold": args.fold,
        "n_val": len(val_files),
        "slot1": slot1_row,
        "sweep_merged": sweep,
        "locked": locked["thr"],
        "locked_row": locked,
        "year_slice": slices,
        "gate": decision,
        "pred_csv": str(pred_csv),
        "do_not_submit": True,
    }
    text = json.dumps(out, indent=2)
    (FAST_ARTIFACTS / f"unet_slot1_merge_fold{args.fold}_official.json").write_text(text)
    repo = ARTIFACTS / f"unet_slot1_merge_fold{args.fold}_official.json"
    repo.write_text(text)
    print(json.dumps({"path": str(repo), "locked": locked, "gate": decision}, indent=2))


if __name__ == "__main__":
    main()
