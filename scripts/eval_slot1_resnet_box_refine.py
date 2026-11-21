#!/usr/bin/env python3
"""Official fold-0 PQ: grow frozen Slot1 instances through frozen ResNet@0.6.

Seed-constrained grow only. No new extras. Does not overwrite unet_resnet34.
Locked pad is 16 px (a priori). Never submits.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_unet import infer_prob
from eval_unet_merge import gate_unet_merge
from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.infer import pick_device
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, IMAGE_SIZE, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.postprocess import resolve_overlaps
from filament.rle import decode_rle, encode_rle
from filament.unet_resnet import build_resnet34_unet
from gate_unet_family import live_mps_owner
from score_faint_merge import load_or_infer_slot1
from score_family import year_slices

RESNET_WEIGHTS = WEIGHTS_DIR / "unet_resnet34" / "best.pt"
LOCKED_THR = 0.6
LOCKED_PAD = 16
OFFICIAL = ARTIFACTS / "slot1_resnet_box_refine_fold0_official.json"
FORBIDDEN = "unet_resnet34"


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


def grow_one(rle: str, prob: np.ndarray, thr: float, pad: int) -> np.ndarray:
    m = decode_rle(rle, IMAGE_SIZE, IMAGE_SIZE)
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return m
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(m.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(m.shape[1], int(xs.max()) + pad + 1)
    extra = np.zeros_like(m)
    extra[y0:y1, x0:x1] = (prob[y0:y1, x0:x1] >= thr).astype(np.uint8)
    union = np.maximum(m, extra)
    nlab, labels = cv2.connectedComponents(union, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, nlab):
        comp = labels == i
        if np.any(comp & m):
            keep[comp] = 1
    return keep if int(keep.sum()) > 0 else m


def grow_image(rles: list[str], prob: np.ndarray, thr: float, pad: int) -> list[str]:
    masks = [grow_one(rle, prob, thr, pad) for rle in rles]
    masks = [m for m in masks if int(m.sum()) > 0]
    masks = resolve_overlaps(masks)
    return [encode_rle(m) for m in masks if int(m.sum()) > 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=1024)
    args = parser.parse_args()

    owner = live_mps_owner(ignore=frozenset({"eval_slot1_resnet_box_refine.py"}))
    if owner:
        print(f"Refusing: {owner} owns MPS.", file=sys.stderr)
        raise SystemExit(2)
    if FORBIDDEN in Path(RESNET_WEIGHTS).parts and not RESNET_WEIGHTS.exists():
        raise SystemExit(f"missing {RESNET_WEIGHTS}")
    if not RESNET_WEIGHTS.exists():
        raise SystemExit(f"missing {RESNET_WEIGHTS}")

    device = pick_device(args.device)
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    slot1 = load_or_infer_slot1(val_files, device=device)

    model = build_resnet34_unet()
    state = torch.load(RESNET_WEIGHTS, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    probs = {}
    for file_name in tqdm(val_files, desc="resnet34 box-refine probs"):
        probs[file_name] = infer_prob(model, TRAIN_IMG_DIR / file_name, args.imgsz, device)

    ident_frames = slot1
    ident_df = pd.concat(ident_frames.values(), ignore_index=True)
    ident_br = score_pq(gt_df, ident_df)
    ident = {"op": "slot1_identity", **_row(ident_br, len(val_files), int(len(ident_df)))}

    diagnosis_pads = [8, 16, 32]
    sweep = []
    pred_cache: dict[int, dict[str, pd.DataFrame]] = {}
    for pad in diagnosis_pads:
        frames = {}
        for file_name in val_files:
            rles = slot1[file_name]["segmentation_rle"].astype(str).tolist()
            grown = grow_image(rles, probs[file_name], LOCKED_THR, pad)
            frames[file_name] = pred_to_df(Path(file_name).stem, grown)
        pred_df = pd.concat(frames.values(), ignore_index=True)
        br = score_pq(gt_df, pred_df)
        row = {"thr": LOCKED_THR, "pad": pad, **_row(br, len(val_files), int(len(pred_df)))}
        sweep.append(row)
        pred_cache[pad] = frames
        print(row, flush=True)

    locked = next(r for r in sweep if r["pad"] == LOCKED_PAD)
    slices = year_slices(val_files, pred_cache[LOCKED_PAD], views, by_file)
    slot1_slices = year_slices(val_files, ident_frames, views, by_file)
    gate = gate_unet_merge(locked, ident, slices["2017-2020"], slot1_slices["2017-2020"])

    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    pred_csv = FAST_ARTIFACTS / "slot1_resnet_box_refine_fold0_pred.csv"
    pd.concat(pred_cache[LOCKED_PAD].values(), ignore_index=True).to_csv(pred_csv, index=False)

    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "family": "slot1_resnet_box_refine",
        "did_not_add_extras": True,
        "did_not_overwrite_unet_resnet34": True,
        "kaggle_submit_called": False,
        "do_not_submit": True,
        "locked_a_priori": {"thr": LOCKED_THR, "pad": LOCKED_PAD},
        "locked_not_argmax_of_sweep": True,
        "sweep_diagnosis_only": sweep,
        "locked_row": locked,
        "slot1_identity": ident,
        "year_slice": slices,
        "gate": gate,
        "pred_csv": str(pred_csv),
        "weights_read": str(RESNET_WEIGHTS),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OFFICIAL.write_text(json.dumps(out, indent=2) + "\n")
    (FAST_ARTIFACTS / OFFICIAL.name).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"path": str(OFFICIAL), "locked": locked, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
