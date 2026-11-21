#!/usr/bin/env python3
"""Official fold-0 instance PQ for full-disk U-Net.

Upsample logits to native 2048, threshold, CC instances, score_pq.
Never submits. Never reads test labels. Semantic dice is not the gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.infer import pick_device
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, IMAGE_SIZE, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.postprocess import connected_instances, resolve_overlaps
from filament.preprocess import read_gray
from filament.rle import encode_rle
from filament.semantic_dataset import gray_as_rgb
from filament.unet import build_fulldisk_unet
from score_family import M_1280, S_MEDOID, year_slices


def infer_prob(model, path: Path, imgsz: int, device: str) -> np.ndarray:
    gray = read_gray(path)
    small = cv2.resize(gray, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(gray_as_rgb(small)).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(tensor)[0, 0].detach().cpu().numpy()
    prob = 1.0 / (1.0 + np.exp(-logit))
    return cv2.resize(prob.astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)


def decode_instances(prob: np.ndarray, thr: float, min_area: int = 64) -> list[str]:
    binary = (prob >= thr).astype(np.uint8)
    masks = connected_instances(binary)
    masks = resolve_overlaps(masks)
    return [encode_rle(m) for m in masks if int(m.sum()) >= int(min_area)]


def gate_unet(best: dict) -> dict:
    pq = float(best["pq"])
    tp = int(best["tp"])
    fp = int(best["fp"])
    d_pq = pq - M_1280["pq"]
    d_tp = tp - M_1280["tp"]
    fp_exploded = fp >= 2 * M_1280["fp"]
    if pq >= 0.42 and not fp_exploded:
        decision = "PROMOTE_LOOK"
        next_action = "U-Net cleared 0.42. After quota reset, at most one public look. Year-shift stays Slot 2."
    elif pq < 0.20 or d_pq <= -0.08:
        decision = "KILL"
        next_action = "Full-disk U-Net collapsed. Do not upload. Do not retune from public."
    elif d_tp <= 0 and pq <= M_1280["pq"]:
        decision = "FAIL_NO_NEW_PROPOSALS"
        next_action = "Did not add TPs vs Slot 1. Same crop-refiner failure mode. Do not upload."
    elif 0.38 <= pq < 0.42:
        decision = "RESEARCH_HOLD_SPINE_V2"
        next_action = "PQ is real but below Slot-1 replace. Spine-split v2 only if fragments/RQ show up."
    else:
        decision = "RESEARCH_HOLD"
        next_action = "Diagnosis only. Do not upload. Year-shift stays Slot 2."
    return {
        "decision": decision,
        "next_action": next_action,
        "vs_m1280": {"d_pq": d_pq, "d_tp": d_tp, "fp_exploded": fp_exploded},
        "vs_s_medoid_d_pq": pq - S_MEDOID["pq"],
        "do_not_submit_from_this_script": True,
        "do_not_replace_slot2": True,
        "official_is_instance_pq_not_dice": True,
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

    model = build_fulldisk_unet(base=args.base)
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    probs = {}
    for file_name in tqdm(val_files, desc="unet probs"):
        probs[file_name] = infer_prob(model, TRAIN_IMG_DIR / file_name, args.imgsz, device)

    sweep = []
    pred_cache: dict[float, dict[str, pd.DataFrame]] = {}
    for thr in thrs:
        frames = {}
        for file_name in val_files:
            rles = decode_instances(probs[file_name], thr)
            frames[file_name] = pred_to_df(Path(file_name).stem, rles)
        pred_cache[thr] = frames
        pred_df = pd.concat(frames.values(), ignore_index=True) if frames else pd.DataFrame(
            columns=["filament_id", "segmentation_rle"]
        )
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
            }
        )
        print(sweep[-1], flush=True)

    locked = max(sweep, key=lambda r: (r["pq"], -abs(r["thr"] - 0.45)))
    slices = year_slices(val_files, pred_cache[locked["thr"]], views, by_file)
    decision = gate_unet(locked)

    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    pred_csv = FAST_ARTIFACTS / f"unet_fulldisk_fold{args.fold}_thr{locked['thr']:.2f}_pred.csv"
    pd.concat(pred_cache[locked["thr"]].values(), ignore_index=True).to_csv(pred_csv, index=False)
    out = {
        "family": "unet_fulldisk_s1024",
        "weights": str(weights.resolve()),
        "input": "gray-as-RGB",
        "device": device,
        "imgsz": args.imgsz,
        "metric": "official host PQ after CC instances (IoU>0.5, not Hungarian, all annotators)",
        "fold": args.fold,
        "n_val": len(val_files),
        "sweep": sweep,
        "locked": locked["thr"],
        "locked_row": locked,
        "year_slice": slices,
        "gate": decision,
        "baselines": {"s_medoid": S_MEDOID, "m_1280": M_1280},
        "pred_csv": str(pred_csv),
        "do_not_submit": True,
    }
    text = json.dumps(out, indent=2)
    (FAST_ARTIFACTS / f"unet_fulldisk_fold{args.fold}_official.json").write_text(text)
    repo = ARTIFACTS / f"unet_fulldisk_fold{args.fold}_official.json"
    repo.write_text(text)
    print(json.dumps({"path": str(repo), "locked": locked, "gate": decision}, indent=2))


if __name__ == "__main__":
    main()
