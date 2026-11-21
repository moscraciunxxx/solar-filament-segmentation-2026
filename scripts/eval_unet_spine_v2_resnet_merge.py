#!/usr/bin/env python3
"""Slot-1 NMS-merge after ResNet34 spine-v2 ridge-split. Gated. Never submits.

Opens only if the ResNet merge gate.decision == RESEARCH_HOLD_SPINE_V2
(artifacts/unet_resnet34_slot1_merge_fold0_official.json).

Decode-only: load unet_resnet34/best.pt via build_resnet34_unet (never
unet_semantic / unet_patch_dhebe / unet_3ch_disk). Gray-as-RGB full-disk
infer (same as ResNet/v1; not patch stitch, not 3ch). Train-JSON spines
only. After ridge-split, Slot1 NMS-merge sweeping min_area 642+1034.
Lock max PQ among FP < 2*Slot1 (886 when Slot1 FP=443).

Writes artifacts/unet_spine_v2_resnet_slot1_merge_fold0_official.json
with gate_unet_merge. Default --dry-refuse prints None on HOLD.
Does not start MPS on --dry-refuse. Never submits. No test CSV
(test JPEGs have no train-JSON spines; do not invent MAGFiLO test ridges).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_unet_spine_v2 import (
    HOSE_FLOOR,
    RESNET_WEIGHTS,
    decode_instances_spine_v2,
    parse_min_areas,
    refuse_reason,
    reject_resnet_spine_weights,
    _infer_prob,
)
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR, TRAIN_JSON
from gate_unet_family import RESNET_GATE

FAMILY = "unet_spine_v2_resnet_slot1_merge"
DEST_NAME = "unet_spine_v2_resnet_slot1_merge_fold{fold}_official.json"
WEIGHTS_MISSING = "weights not trained yet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-refuse",
        action="store_true",
        help="Print refuse_reason (None if RESEARCH_HOLD_SPINE_V2). Do not infer.",
    )
    parser.add_argument(
        "--gate-json",
        default=str(RESNET_GATE),
        help="Must be RESEARCH_HOLD_SPINE_V2. Default: ResNet merge official.",
    )
    parser.add_argument("--weights", default=str(RESNET_WEIGHTS))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--thrs", default="0.45,0.50,0.55,0.60")
    parser.add_argument(
        "--min-area",
        type=int,
        default=HOSE_FLOOR,
        help="Always included in the sweep (hose floor).",
    )
    parser.add_argument(
        "--min-areas",
        default="642,1034",
        help="Comma sweep. 1034 is the first v1-replay floor that kills fp_exploded.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    gate_path = Path(args.gate_json)
    reason = refuse_reason(gate_path)
    if args.dry_refuse:
        print(reason)
        raise SystemExit(0)
    if reason:
        print(reason, file=sys.stderr)
        raise SystemExit(2)

    weights = Path(args.weights)
    bad = reject_resnet_spine_weights(weights)
    if bad:
        print(bad, file=sys.stderr)
        raise SystemExit(2)
    if not weights.exists():
        print(WEIGHTS_MISSING, file=sys.stderr)
        raise SystemExit(2)

    min_areas = parse_min_areas(args.min_areas, args.min_area, hose_floor=HOSE_FLOOR)

    import pandas as pd
    import torch
    from tqdm import tqdm

    from eval_unet_merge import gate_unet_merge
    from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
    from filament.infer import pick_device
    from filament.merge import MERGE_IOU, merge_rles
    from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
    from filament.spine_io import load_spines_by_file, union_ridge_for_file
    from filament.unet_resnet import build_resnet34_unet
    from gate_unet_family import live_mps_owner
    from score_faint_merge import load_or_infer_slot1
    from score_family import year_slices

    owner = live_mps_owner()
    if owner:
        raise SystemExit(f"Refusing spine-v2 ResNet merge infer: {owner} owns MPS.")

    device = pick_device(args.device)
    thrs = [float(x) for x in args.thrs.split(",") if x.strip()]
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == args.fold)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    spines = load_spines_by_file(TRAIN_JSON)
    ridges = {fn: union_ridge_for_file(fn, spines) for fn in val_files}

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

    model = build_resnet34_unet(in_ch=3, pretrained=False)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()

    probs = {}
    for file_name in tqdm(val_files, desc="spine-v2 resnet merge probs"):
        probs[file_name] = _infer_prob(model, TRAIN_IMG_DIR / file_name, args.imgsz, device)

    sweep = []
    pred_cache = {}
    for min_area in min_areas:
        for thr in thrs:
            frames = {}
            n_added = 0
            for file_name in val_files:
                extra = decode_instances_spine_v2(
                    probs[file_name], thr, ridges[file_name], min_area=min_area
                )
                slot_rles = slot1_frames[file_name]["segmentation_rle"].tolist()
                merged, added = merge_rles(slot_rles, extra, iou_thr=MERGE_IOU)
                n_added += added
                frames[file_name] = pred_to_df(Path(file_name).stem, merged)
            pred_cache[(thr, min_area)] = frames
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
                    "min_area": min_area,
                }
            )
            print(sweep[-1], flush=True)

    fp_cap = 2 * int(slot1_br.fp)
    non_exploded = [r for r in sweep if int(r["fp"]) < fp_cap]
    pool = non_exploded or sweep
    locked = max(pool, key=lambda r: (r["pq"], -abs(r["thr"] - 0.45), r["min_area"]))
    slices = year_slices(
        val_files, pred_cache[(locked["thr"], locked["min_area"])], views, by_file
    )
    decision = gate_unet_merge(locked, slot1_row, slices["2017-2020"], slot1_slices["2017-2020"])
    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    dest = ARTIFACTS / DEST_NAME.format(fold=args.fold)
    out = {
        "family": FAMILY,
        "weights": str(weights.resolve()),
        "builder": "build_resnet34_unet",
        "decode": "ridge-split then Slot1 NMS-merge",
        "spine_source": str(TRAIN_JSON),
        "leakage": "train JSON spines only; no MAGFiLO public-test labels",
        "fold": args.fold,
        "n_val": len(val_files),
        "slot1": slot1_row,
        "sweep_merged": sweep,
        "locked": locked["thr"],
        "locked_row": locked,
        "year_slice": slices,
        "gate": decision,
        "min_areas": min_areas,
        "min_area": locked["min_area"],
        "hose_floor": HOSE_FLOOR,
        "lock_rule": (
            "max PQ among FP < 2*Slot1 (886 when Slot1 FP=443); "
            "else max PQ (honest hose fail)"
        ),
        "v1_fp_hose_lesson": (
            "TinyUNet min_area=64: FP 443->2301. Replay: 642 still fp_exploded "
            "(FP 1056>=886, +145 TP); 1034 first clean hose (FP 806, +93 TP). "
            "Sweep 642+1034 after ridge-split; do not invert public."
        ),
        "do_not_submit": True,
        "not_patch_stitch": True,
        "not_3ch_stack": True,
        "no_test_csv": (
            "test JPEGs have no train-JSON spines; do not invent MAGFiLO test ridges"
        ),
        "kaggle_submit_called": False,
    }
    text = json.dumps(out, indent=2) + "\n"
    dest.write_text(text)
    (FAST_ARTIFACTS / dest.name).write_text(text)
    print(
        json.dumps(
            {"path": str(dest), "locked": locked, "gate": decision, "do_not_submit": True},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
