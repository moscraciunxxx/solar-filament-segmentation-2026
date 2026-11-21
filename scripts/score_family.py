#!/usr/bin/env python3
"""Official fold-0 PQ + independent conf + year-slice for one YOLO family.

Always scores the FULL grouped holdout (--max-val 0). Ultralytics mask mAP
is not this number. Never submits. Never reads MAGFiLO test labels.
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

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.infer import pick_device, run_yolo_infer, weights_tag
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR

S_MEDOID = {"pq": 0.357, "tp": 809, "fp": 475, "fn": 780}
M_1280 = {"pq": 0.410, "tp": 925, "fp": 443, "fn": 664}


def jpeg_year(file_name: str) -> int:
    return int(Path(file_name).stem[:4])


def _row(br, n_images: int, n_pred: int) -> dict:
    return {
        "n_images": n_images,
        "n_pred": n_pred,
        "pq": br.pq,
        "tp": br.tp,
        "fp": br.fp,
        "fn": br.fn,
        "mean_tp_iou": br.mean_tp_iou,
    }


def year_slices(val_files: list[str], pred_by_file: dict[str, pd.DataFrame], views, by_file) -> dict:
    slices = {
        "all": val_files,
        "pre-2017": [f for f in val_files if jpeg_year(f) < 2017],
        "2017+": [f for f in val_files if jpeg_year(f) >= 2017],
        "2020+": [f for f in val_files if jpeg_year(f) >= 2020],
        "2017-2020": [f for f in val_files if 2017 <= jpeg_year(f) <= 2020],
    }
    out = {}
    for name, files in slices.items():
        pred_df = (
            pd.concat([pred_by_file[f] for f in files], ignore_index=True)
            if files
            else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
        )
        gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, files))
        br = score_pq(gt_df, pred_df)
        out[name] = _row(br, len(files), int(len(pred_df)))
        out[name]["pred_per_img"] = (len(pred_df) / len(files)) if files else 0.0
    return out


def gate(best: dict) -> dict:
    pq = float(best["pq"])
    tp = int(best["tp"])
    fp = int(best["fp"])
    d_pq = pq - S_MEDOID["pq"]
    d_tp = tp - S_MEDOID["tp"]
    fp_exploded = fp >= 2 * S_MEDOID["fp"]
    if pq >= 0.38 and not fp_exploded:
        decision = "PROMOTE_LOOK"
        next_action = "Build test CSV after quota reset. One public look. Year-shift stays Slot 2. Do not submit today if quota is 0."
    elif d_pq <= -0.08:
        decision = "KILL_START_PHYS"
        next_action = "Start yolo11s_seg_phys when MPS is idle. Infer --phys-stack. Not a Slot-2 replace."
    elif pq <= S_MEDOID["pq"] and d_tp <= 0:
        decision = "FAIL_START_PHYS"
        next_action = "All-annotator missed the gate. Start yolo11s_seg_phys when MPS is idle."
    elif d_tp > 0 and fp_exploded:
        decision = "TP_GAIN_FP_BOMB_RATER"
        next_action = "Inclusive views add objects but dump FPs. Prefer yolo11s_seg_rater (job 4) over union (job 5). XOR — not both the same day. Not phys first."
    elif d_tp > 0 and 0.36 <= pq < 0.38:
        decision = "RESEARCH_HOLD"
        next_action = "TP gain without a 0.38 PQ. Hold phys. Do not upload. Do not copy Slot 1 conf."
    else:
        decision = "FAIL_START_PHYS"
        next_action = "No clean TP story. Overwatch picks the next ranked family. Do not restart phys."
    return {
        "decision": decision,
        "next_action": next_action,
        "vs_s_medoid": {"d_pq": d_pq, "d_tp": d_tp, "fp_exploded": fp_exploded},
        "vs_m1280_d_pq": pq - M_1280["pq"],
        "density_2.3_is_not_a_fail": True,
        "do_not_submit_from_this_script": True,
        "do_not_replace_slot2": True,
    }


def gate_tiles(best: dict) -> dict:
    pq = float(best["pq"])
    tp = int(best["tp"])
    fp = int(best["fp"])
    d_pq = pq - S_MEDOID["pq"]
    d_tp = tp - S_MEDOID["tp"]
    fp_exploded = fp >= 2 * S_MEDOID["fp"]
    if pq >= 0.43 and not fp_exploded:
        decision = "PROMOTE_LOOK"
        next_action = (
            "Stitch PQ cleared 0.43. After quota reset, at most one public look. "
            "Year-shift stays Slot 2. Do not upload a conf twin."
        )
    elif d_pq <= -0.03 or pq < 0.20:
        decision = "KILL_START_FAINT"
        next_action = (
            "Tile stitch lost vs s-medoid. Kill tiles. Next ranked family is "
            "faint_specialist_s1024. Not rater/union. Not m@1280-all."
        )
    elif fp_exploded:
        decision = "KILL_FRAGMENT_FP"
        next_action = "Fragments became FPs under non-Hungarian PQ. Do not retune stitch from this look as a submit."
    elif 0.36 <= pq < 0.43:
        decision = "RESEARCH_HOLD"
        next_action = "Stitch PQ is real but below the 0.43 Slot-1 gate. Diagnosis only. Do not upload."
    else:
        decision = "FAIL_START_FAINT"
        next_action = "No stitch climb. Overwatch starts faint_specialist_s1024 when MPS is idle."
    return {
        "decision": decision,
        "next_action": next_action,
        "vs_s_medoid": {"d_pq": d_pq, "d_tp": d_tp, "fp_exploded": fp_exploded},
        "vs_m1280_d_pq": pq - M_1280["pq"],
        "density_2.3_is_not_a_fail": True,
        "do_not_submit_from_this_script": True,
        "do_not_replace_slot2": True,
        "official_is_stitch_not_tile_map": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--family", default="")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--confs", default="0.25,0.35,0.40")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--phys-stack", dest="phys", action="store_true")
    parser.add_argument(
        "--residual-stack",
        dest="residual",
        action="store_true",
        help="flatten+raw+Sobel (Method 2). Not --phys-stack.",
    )
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument(
        "--tiles",
        action="store_true",
        help="Native-2048 tile stitch (tile_native_s640). Not full-frame YOLO.",
    )
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    if sum([bool(args.phys), bool(args.clahe), bool(getattr(args, "residual", False))]) > 1:
        raise SystemExit("Choose at most one of --phys-stack / --clahe / --residual-stack")
    if args.tiles and (args.phys or args.clahe or getattr(args, "residual", False)):
        raise SystemExit("tile stitch is native crops only")

    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(weights)
    family = args.family or weights_tag(weights)
    if "tile" in family.lower():
        args.tiles = True
        if args.imgsz == 1024:
            args.imgsz = 640
    device = pick_device(args.device)
    confs = [float(x) for x in args.confs.split(",") if x.strip()]

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == args.fold)
    model = YOLO(str(weights))

    sweep = []
    pred_cache: dict[float, dict[str, pd.DataFrame]] = {}
    for conf in confs:
        frames = {}
        for file_name in tqdm(val_files, desc=f"{family} conf={conf:.2f}"):
            rles = run_yolo_infer(
                model,
                TRAIN_IMG_DIR / file_name,
                conf=conf,
                imgsz=args.imgsz,
                merge_gap=0,
                device=device,
                clahe=args.clahe,
                phys=args.phys,
                residual=getattr(args, "residual", False),
                tiles=args.tiles,
            )
            frames[file_name] = pred_to_df(Path(file_name).stem, rles)
        pred_cache[conf] = frames
        pred_df = pd.concat(frames.values(), ignore_index=True)
        gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
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
            }
        )
        print(sweep[-1])

    locked = max(sweep, key=lambda r: (r["pq"], -abs(r["conf"] - 0.35)))
    slices = year_slices(val_files, pred_cache[locked["conf"]], views, by_file)
    decision = gate_tiles(locked) if args.tiles else gate(locked)

    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    pred_csv = FAST_ARTIFACTS / f"{family}_fold{args.fold}_conf{locked['conf']:.2f}_pred.csv"
    pd.concat(pred_cache[locked["conf"]].values(), ignore_index=True).to_csv(pred_csv, index=False)

    out = {
        "family": family,
        "weights": str(weights.resolve()),
        "device": device,
        "imgsz": args.imgsz,
        "phys_stack": args.phys,
        "residual_stack": getattr(args, "residual", False),
        "clahe": args.clahe,
        "tiles": args.tiles,
        "metric": (
            "official host PQ after tile stitch (IoU>0.5, not Hungarian, all annotators)"
            if args.tiles
            else "official host PQ (IoU>0.5, not Hungarian, all annotators)"
        ),
        "fold": args.fold,
        "n_val": len(val_files),
        "independent_of_slot1": True,
        "sweep": sweep,
        "locked": locked["conf"],
        "locked_row": locked,
        "year_slice": slices,
        "gate": decision,
        "baselines": {"s_medoid": S_MEDOID, "m_1280": M_1280},
        "pred_csv": str(pred_csv),
        "do_not_submit": True,
    }
    fast_path = FAST_ARTIFACTS / f"{family}_fold{args.fold}_official.json"
    repo_path = ARTIFACTS / f"{family}_fold{args.fold}_official.json"
    text = json.dumps(out, indent=2)
    fast_path.write_text(text)
    repo_path.write_text(text)
    print(json.dumps({"path": str(repo_path), "locked": locked, "gate": decision}, indent=2))


if __name__ == "__main__":
    main()
