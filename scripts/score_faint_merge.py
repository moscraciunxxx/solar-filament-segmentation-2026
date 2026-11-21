#!/usr/bin/env python3
"""Official fold-0 PQ for faint specialist NMS-merged with frozen Slot 1.

Slot 1 stays locked at m@1280 conf=0.35. Specialist conf is swept independently.
Gate is on the MERGED file. Standalone specialist PQ is diagnosis only.
Never submits. Never reads MAGFiLO test labels.
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
sys.path.insert(0, str(ROOT / "scripts"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.infer import pick_device, run_yolo_infer, weights_tag
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.merge import MERGE_IOU, merge_rles
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from score_family import M_1280, S_MEDOID, year_slices

SLOT1_WEIGHTS = WEIGHTS_DIR / "yolo11m_seg_1280" / "weights" / "best.pt"
SLOT1_CONF = 0.35
SLOT1_IMGSZ = 1280
SLOT1_PRED_CSV = (
    FAST_ARTIFACTS / "yolo_yolo11m_seg_1280_fold0_conf0.35_sz1280_m0_base_m1280_pq.csv"
)
EMPTY_SLOT1_STEMS = ("20180801232450Bh", "20190602025430Lh", "20200619123010Th")


def _stem_frames(df: pd.DataFrame, val_files: list[str]) -> dict[str, pd.DataFrame]:
    frames = {}
    for file_name in val_files:
        stem = Path(file_name).stem
        sub = df[df["filament_id"].str.startswith(stem + "_")].copy()
        frames[file_name] = sub.reset_index(drop=True)
    return frames


def gate_faint_merge(merged: dict, slot1: dict, desert: dict, slot1_desert: dict) -> dict:
    pq = float(merged["pq"])
    tp = int(merged["tp"])
    d_pq = pq - float(slot1["pq"])
    d_tp = tp - int(slot1["tp"])
    desert_drop = float(slot1_desert["pq"]) - float(desert["pq"])
    fp_exploded = int(merged["fp"]) >= 2 * int(slot1["fp"])
    if pq >= 0.43 and desert_drop < 0.05 and not fp_exploded:
        decision = "PROMOTE_LOOK"
        next_action = (
            "Merged official PQ cleared 0.43 without a solar-min dump. "
            "After quota reset, at most one public look of the MERGED csv. "
            "Year-shift stays Slot 2. Do not upload the specialist alone."
        )
    elif d_tp <= 0:
        decision = "KILL_NO_NEW_TP"
        next_action = (
            "Specialist added no new TPs vs frozen Slot 1. Kill faint. "
            "Do not upload. Do not replace Slot 2. Do not retune merge IoU from this look."
        )
    elif d_pq <= -0.05:
        decision = "KILL_WORSE"
        next_action = (
            "Merged PQ is >=0.05 worse than Slot 1. Kill faint. "
            "Do not upload. Do not replace Slot 2."
        )
    elif desert_drop >= 0.05:
        decision = "KILL_DESERT_DUMP"
        next_action = (
            "2017-2020 fold-0 PQ dropped >=0.05 vs Slot 1. Solar-min FP dump. Kill faint."
        )
    else:
        decision = "RESEARCH_HOLD"
        next_action = (
            "Merged PQ moved but missed the 0.43 gate. Diagnosis only. "
            "Do not upload. Do not replace Slot 2. Do not retune merge from public."
        )
    return {
        "decision": decision,
        "next_action": next_action,
        "vs_slot1": {"d_pq": d_pq, "d_tp": d_tp, "fp_exploded": fp_exploded},
        "desert_2017_2020_drop": desert_drop,
        "merge_iou": MERGE_IOU,
        "do_not_submit_from_this_script": True,
        "do_not_upload_specialist_alone": True,
        "do_not_replace_slot2": True,
    }


def load_or_infer_slot1(val_files: list[str], device: str) -> dict[str, pd.DataFrame]:
    if SLOT1_PRED_CSV.exists():
        df = pd.read_csv(SLOT1_PRED_CSV)
        frames = _stem_frames(df, val_files)
        n_pred = sum(len(f) for f in frames.values())
        if n_pred == 0:
            raise SystemExit(f"Slot-1 cache has no preds: {SLOT1_PRED_CSV}")
        return frames
    if not SLOT1_WEIGHTS.exists():
        raise FileNotFoundError(SLOT1_WEIGHTS)
    from ultralytics import YOLO

    model = YOLO(str(SLOT1_WEIGHTS))
    frames = {}
    for file_name in tqdm(val_files, desc="slot1 m@1280 conf=0.35"):
        rles = run_yolo_infer(
            model,
            TRAIN_IMG_DIR / file_name,
            conf=SLOT1_CONF,
            imgsz=SLOT1_IMGSZ,
            merge_gap=0,
            device=device,
        )
        frames[file_name] = pred_to_df(Path(file_name).stem, rles)
    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    pd.concat(frames.values(), ignore_index=True).to_csv(SLOT1_PRED_CSV, index=False)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="faint specialist best.pt")
    parser.add_argument("--family", default="yolo11s_seg_faint")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--confs", default="0.25,0.35,0.40")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(weights)
    family = args.family or weights_tag(weights)
    device = pick_device(args.device)
    confs = [float(x) for x in args.confs.split(",") if x.strip()]

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == args.fold)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))

    slot1_frames = load_or_infer_slot1(val_files, device)
    slot1_df = pd.concat(slot1_frames.values(), ignore_index=True)
    slot1_br = score_pq(gt_df, slot1_df)
    slot1_row = {
        "conf": SLOT1_CONF,
        "pq": slot1_br.pq,
        "tp": slot1_br.tp,
        "fp": slot1_br.fp,
        "fn": slot1_br.fn,
        "mean_tp_iou": slot1_br.mean_tp_iou,
        "n_pred": int(len(slot1_df)),
    }
    slot1_slices = year_slices(val_files, slot1_frames, views, by_file)

    model = YOLO(str(weights))
    sweep = []
    pred_cache: dict[float, dict[str, pd.DataFrame]] = {}
    added_cache: dict[float, int] = {}
    standalone_sweep = []
    for conf in confs:
        frames = {}
        n_added = 0
        standalone_frames = {}
        for file_name in tqdm(val_files, desc=f"{family} merge conf={conf:.2f}"):
            spec_rles = run_yolo_infer(
                model,
                TRAIN_IMG_DIR / file_name,
                conf=conf,
                imgsz=args.imgsz,
                merge_gap=0,
                device=device,
            )
            standalone_frames[file_name] = pred_to_df(Path(file_name).stem, spec_rles)
            slot_rles = slot1_frames[file_name]["segmentation_rle"].tolist()
            merged, added = merge_rles(slot_rles, spec_rles)
            n_added += added
            frames[file_name] = pred_to_df(Path(file_name).stem, merged)
        pred_cache[conf] = frames
        added_cache[conf] = n_added
        pred_df = pd.concat(frames.values(), ignore_index=True)
        br = score_pq(gt_df, pred_df)
        sweep.append(
            {
                "conf": conf,
                "pq": br.pq,
                "tp": br.tp,
                "fp": br.fp,
                "fn": br.fn,
                "mean_tp_iou": br.mean_tp_iou,
                "n_pred": int(len(pred_df)),
                "n_specialist_added": n_added,
            }
        )
        spec_df = pd.concat(standalone_frames.values(), ignore_index=True)
        spec_br = score_pq(gt_df, spec_df)
        standalone_sweep.append(
            {
                "conf": conf,
                "pq": spec_br.pq,
                "tp": spec_br.tp,
                "fp": spec_br.fp,
                "fn": spec_br.fn,
                "mean_tp_iou": spec_br.mean_tp_iou,
                "n_pred": int(len(spec_df)),
                "note": "diagnosis only; do not upload",
            }
        )
        print({"merged": sweep[-1], "specialist_alone": standalone_sweep[-1]})

    locked = max(sweep, key=lambda r: (r["pq"], -abs(r["conf"] - 0.35)))
    slices = year_slices(val_files, pred_cache[locked["conf"]], views, by_file)
    empty_files = [f for f in val_files if Path(f).stem in EMPTY_SLOT1_STEMS]
    empty_watch = []
    for file_name in empty_files:
        stem = Path(file_name).stem
        n_slot = int(len(slot1_frames[file_name]))
        n_merged = int(len(pred_cache[locked["conf"]][file_name]))
        empty_watch.append(
            {
                "file_name": file_name,
                "stem": stem,
                "slot1_n_pred": n_slot,
                "merged_n_pred": n_merged,
                "specialist_added": n_merged - n_slot,
                "note": (
                    "Th: only faint-TP chance on empty Slot-1. "
                    "Bh/Lh: large identifiable hard misses — extra preds are FP hose, not faint TPs."
                    if stem.endswith("Th")
                    else "Large identifiable Slot-1 hard miss. Faint cannot label it; extras are FP risk."
                ),
            }
        )
    empty_pred_df = (
        pd.concat([pred_cache[locked["conf"]][f] for f in empty_files], ignore_index=True)
        if empty_files
        else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    )
    empty_gt = gt_annotations_to_df(gt_records_for_files(views, by_file, empty_files))
    empty_br = score_pq(empty_gt, empty_pred_df)
    empty_slot1_df = pd.concat([slot1_frames[f] for f in empty_files], ignore_index=True)
    empty_slot1_br = score_pq(empty_gt, empty_slot1_df)
    empty_slot1_report = {
        "stems": list(EMPTY_SLOT1_STEMS),
        "n_images": len(empty_files),
        "per_image": empty_watch,
        "merged": {
            "pq": empty_br.pq,
            "tp": empty_br.tp,
            "fp": empty_br.fp,
            "fn": empty_br.fn,
            "n_pred": int(len(empty_pred_df)),
        },
        "slot1": {
            "pq": empty_slot1_br.pq,
            "tp": empty_slot1_br.tp,
            "fp": empty_slot1_br.fp,
            "fn": empty_slot1_br.fn,
            "n_pred": int(len(empty_slot1_df)),
        },
        "watch": "KILL_DESERT_DUMP if faint floods these three. Bh/Lh extras cannot be faint TPs.",
    }
    decision = gate_faint_merge(
        locked,
        slot1_row,
        slices["2017-2020"],
        slot1_slices["2017-2020"],
    )

    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    pred_csv = FAST_ARTIFACTS / f"{family}_fold{args.fold}_merged_conf{locked['conf']:.2f}_pred.csv"
    pd.concat(pred_cache[locked["conf"]].values(), ignore_index=True).to_csv(pred_csv, index=False)

    out = {
        "family": family,
        "weights": str(weights.resolve()),
        "device": device,
        "imgsz": args.imgsz,
        "metric": "official host PQ after Slot-1 NMS-merge (IoU>0.5, not Hungarian, all annotators)",
        "fold": args.fold,
        "n_val": len(val_files),
        "slot1": {
            "weights": str(SLOT1_WEIGHTS),
            "conf": SLOT1_CONF,
            "imgsz": SLOT1_IMGSZ,
            "pred_csv": str(SLOT1_PRED_CSV),
            "locked_row": slot1_row,
            "year_slice": slot1_slices,
        },
        "merge_iou": MERGE_IOU,
        "sweep_merged": sweep,
        "sweep_specialist_alone": standalone_sweep,
        "locked": locked["conf"],
        "locked_row": locked,
        "year_slice": slices,
        "empty_slot1_three": empty_slot1_report,
        "gate": decision,
        "baselines": {"s_medoid": S_MEDOID, "m_1280": M_1280},
        "pred_csv": str(pred_csv),
        "do_not_submit": True,
        "do_not_upload_specialist_alone": True,
    }
    fast_path = FAST_ARTIFACTS / f"{family}_fold{args.fold}_official.json"
    repo_path = ARTIFACTS / f"{family}_fold{args.fold}_official.json"
    text = json.dumps(out, indent=2)
    fast_path.write_text(text)
    repo_path.write_text(text)
    print(json.dumps({"path": str(repo_path), "locked": locked, "gate": decision}, indent=2))


if __name__ == "__main__":
    main()
