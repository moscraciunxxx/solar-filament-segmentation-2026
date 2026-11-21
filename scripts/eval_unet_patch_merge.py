#!/usr/bin/env python3
"""Official fold-0 PQ: frozen Slot 1 NMS-merged with patch-U-Net instances.

Stitch-max + one full-disk CC, then NMS-merge extras onto Slot 1. Same
refuse_patch() as eval_unet_patch.py. Gate thresholds are copied from
eval_unet_merge.gate_unet_merge — this file does not import that module
(it would load torch/MPS on the refuse path).

Cannot infer until refuse_patch() is None AND unet_patch_dhebe weights
exist. Slot-1 preds come from the frozen CSV only — never YOLO infer.
Never submits. Never reads MAGFiLO public-test labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_unet_patch import DEFAULT_WEIGHTS, WEIGHTS_MISSING, S_MEDOID, decode_instances_stitched, infer_stitched_prob
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR
from gate_unet_family import GATE, refuse_patch, resolve_start_gate

# Copied from eval_unet_merge.gate_unet_merge / filament.merge — do not import those here.
MERGE_IOU = 0.5
PROMOTE_PQ = 0.42
DESERT_DROP_KILL = 0.05
PQ_DROP_KILL = 0.05
SPINE_PQ_LO = 0.38
SLOT1_PRED_CSV = (
    FAST_ARTIFACTS / "yolo_yolo11m_seg_1280_fold0_conf0.35_sz1280_m0_base_m1280_pq.csv"
)


def gate_unet_patch_merge(merged: dict, slot1: dict, desert: dict, slot1_desert: dict) -> dict:
    """Copy of eval_unet_merge.gate_unet_merge numbers. Do not import that module."""
    pq = float(merged["pq"])
    tp = int(merged["tp"])
    d_pq = pq - float(slot1["pq"])
    d_tp = tp - int(slot1["tp"])
    desert_drop = float(slot1_desert["pq"]) - float(desert["pq"])
    fp_exploded = int(merged["fp"]) >= 2 * int(slot1["fp"])
    if pq >= PROMOTE_PQ and desert_drop < DESERT_DROP_KILL and not fp_exploded:
        decision = "PROMOTE_LOOK"
        next_action = (
            "Slot-1 + U-Net merge cleared 0.42 without a solar-min dump. "
            "After quota reset, at most one public look of the MERGED csv. Year-shift stays Slot 2."
        )
    elif d_tp <= 0:
        decision = "KILL_NO_NEW_TP"
        next_action = "U-Net added no new TPs vs Slot 1. Same crop-refiner failure. Do not upload."
    elif d_pq <= -PQ_DROP_KILL or desert_drop >= DESERT_DROP_KILL:
        decision = "KILL_WORSE_OR_DESERT"
        next_action = "Merged PQ worse or 2017-2020 dump. Kill. Do not upload. Do not replace Slot 2."
    elif SPINE_PQ_LO <= pq < PROMOTE_PQ:
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
        "baselines": {"s_medoid": S_MEDOID, "m_1280": {"pq": 0.410, "tp": 925, "fp": 443, "fn": 664}},
    }


def _stem_frames(df, val_files: list[str]) -> dict:
    frames = {}
    for file_name in val_files:
        stem = Path(file_name).stem
        sub = df[df["filament_id"].str.startswith(stem + "_")].copy()
        frames[file_name] = sub.reset_index(drop=True)
    return frames


def load_slot1_csv_only(val_files: list[str], slot1_csv: Path) -> dict:
    """Frozen Slot-1 cache only. Never starts YOLO / MPS to rebuild it."""
    import pandas as pd

    if not slot1_csv.exists():
        raise SystemExit(f"Slot-1 pred csv missing ({slot1_csv}); will not infer Slot1.")
    df = pd.read_csv(slot1_csv)
    frames = _stem_frames(df, val_files)
    if sum(len(f) for f in frames.values()) == 0:
        raise SystemExit(f"Slot-1 cache has no preds: {slot1_csv}")
    return frames


def run_infer_and_score(args: argparse.Namespace) -> None:
    """Stitch-then-CC, NMS-merge with frozen Slot1, score_pq. After refuse + weights only."""
    import pandas as pd
    import torch
    from tqdm import tqdm

    from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
    from filament.infer import pick_device
    from filament.merge import merge_rles
    from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
    from filament.patch_grid import patch_boxes
    from filament.unet import build_fulldisk_unet
    from score_family import year_slices

    weights = Path(args.weights)
    device = pick_device(args.device)
    thrs = [float(x) for x in args.thrs.split(",") if x.strip()]
    boxes = patch_boxes()

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == args.fold)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))

    slot1_frames = load_slot1_csv_only(val_files, Path(args.slot1_csv))
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
    for file_name in tqdm(val_files, desc="patch stitch_max"):
        probs[file_name] = infer_stitched_prob(
            model, TRAIN_IMG_DIR / file_name, device, boxes
        )

    min_areas = sorted(
        {
            int(x.strip())
            for x in str(args.min_areas).split(",")
            if x.strip()
        }
        | {int(args.min_area)}
    )

    sweep = []
    pred_cache = {}
    for min_area in min_areas:
        for thr in thrs:
            frames = {}
            n_added = 0
            for file_name in val_files:
                extra = decode_instances_stitched(probs[file_name], thr, min_area=min_area)
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
    decision = gate_unet_patch_merge(locked, slot1_row, slices["2017-2020"], slot1_slices["2017-2020"])

    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    pred_csv = FAST_ARTIFACTS / f"unet_patch_slot1_merge_fold{args.fold}_thr{locked['thr']:.2f}_pred.csv"
    pd.concat(pred_cache[(locked["thr"], locked["min_area"])].values(), ignore_index=True).to_csv(pred_csv, index=False)
    out = {
        "family": "unet_patch_slot1_merge",
        "weights": str(weights.resolve()),
        "slot1_csv": str(Path(args.slot1_csv)),
        "metric": "official host PQ after stitch_max + one CC, then Slot-1 NMS-merge",
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
        "lock_rule": "max PQ among FP < 2*Slot1; else max PQ (honest hose fail)",
        "v1_fp_hose_lesson": (
            "TinyUNet min_area=64 hosed FP 443->2301. Patch merge sweeps 642 and 1034."
        ),
        "pred_csv": str(pred_csv),
        "do_not_submit": True,
    }
    text = json.dumps(out, indent=2)
    (FAST_ARTIFACTS / f"unet_patch_slot1_merge_fold{args.fold}_official.json").write_text(text)
    repo = ARTIFACTS / f"unet_patch_slot1_merge_fold{args.fold}_official.json"
    repo.write_text(text)
    print(json.dumps({"path": str(repo), "locked": locked, "gate": decision}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-refuse",
        action="store_true",
        help="Print refuse_patch() reason and exit 0. Do not infer.",
    )
    parser.add_argument("--gate-json", default=str(GATE))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--slot1-csv", default=str(SLOT1_PRED_CSV))
    parser.add_argument("--thrs", default="0.45,0.50,0.55,0.60")
    parser.add_argument(
        "--min-area",
        type=int,
        default=642,
        help="Always included in the sweep (large-identifiable FN floor).",
    )
    parser.add_argument(
        "--min-areas",
        default="642,1034",
        help="Comma sweep. 1034 is the first v1-replay floor that kills fp_exploded.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--base", type=int, default=32)
    args = parser.parse_args()

    reason = refuse_patch(resolve_start_gate(Path(args.gate_json)))
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

    run_infer_and_score(args)


if __name__ == "__main__":
    main()
