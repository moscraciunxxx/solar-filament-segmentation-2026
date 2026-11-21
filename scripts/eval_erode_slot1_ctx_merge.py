#!/usr/bin/env python3
"""Official fold-0: locked erode-Slot1 + frozen ctx extras (no refit, no dest write).

Decode locked at thr 0.6 / min_area 1034. Rejector read-only from
resnet_extra_filter_ctx. Never overwrites unet_resnet34 or ctx dest. Never submits.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_unet import decode_instances, infer_prob
from eval_unet_merge import gate_unet_merge
from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.infer import pick_device
from filament.merge import extra_rles, merge_rles
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.rle import decode_rle
from filament.unet_resnet import build_resnet34_unet
from gate_unet_family import live_mps_owner
from resnet_extra_filter import (
    REJECTOR_THR,
    features_for_rles_context,
    jpeg_and_residual,
    keep_extra_rles,
    load_rejector,
)
from score_faint_merge import SLOT1_PRED_CSV, _stem_frames
from score_family import year_slices

ERODE_CSV = FAST_ARTIFACTS / "slot1_erode_3x1_fold0_pred.csv"
CTX_DEST = WEIGHTS_DIR / "resnet_extra_filter_ctx"
RESNET = WEIGHTS_DIR / "unet_resnet34" / "best.pt"
LOCKED_THR = 0.6
LOCKED_MIN_AREA = 1034
OFFICIAL = ARTIFACTS / "erode_slot1_ctx_merge_fold0_official.json"


def _row(br, n_pred: int) -> dict:
    return {
        "pq": br.pq,
        "tp": br.tp,
        "fp": br.fp,
        "fn": br.fn,
        "mean_tp_iou": br.mean_tp_iou,
        "n_pred": n_pred,
    }


def main() -> None:
    owner = live_mps_owner(ignore=frozenset({"eval_erode_slot1_ctx_merge.py"}))
    if owner:
        raise SystemExit(f"Refusing: {owner} owns MPS.")
    if not ERODE_CSV.exists():
        raise SystemExit(f"missing {ERODE_CSV}")
    if not RESNET.exists():
        raise SystemExit(f"missing {RESNET}")
    clf, meta = load_rejector(CTX_DEST)
    thr_keep = float(meta.get("threshold", REJECTOR_THR))

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    erode_frames = _stem_frames(pd.read_csv(ERODE_CSV), val_files)
    orig_frames = _stem_frames(pd.read_csv(SLOT1_PRED_CSV), val_files)

    erode_df = pd.concat(erode_frames.values(), ignore_index=True)
    orig_df = pd.concat(orig_frames.values(), ignore_index=True)
    erode_row = _row(score_pq(gt_df, erode_df), int(len(erode_df)))
    orig_row = _row(score_pq(gt_df, orig_df), int(len(orig_df)))
    erode_slices = year_slices(val_files, erode_frames, views, by_file)
    orig_slices = year_slices(val_files, orig_frames, views, by_file)

    device = pick_device("auto")
    model = build_resnet34_unet(in_ch=3, pretrained=False)
    model.load_state_dict(torch.load(RESNET, map_location=device))
    model.to(device)
    model.eval()

    n_extras = 0
    n_kept = 0
    n_added = 0
    frames = {}
    for file_name in tqdm(val_files, desc="erode-slot1 + frozen ctx extras"):
        prob = infer_prob(model, TRAIN_IMG_DIR / file_name, 1024, device)
        cands = decode_instances(prob, LOCKED_THR, min_area=LOCKED_MIN_AREA)
        slot_rles = erode_frames[file_name]["segmentation_rle"].astype(str).tolist()
        extras = extra_rles(slot_rles, cands)
        n_extras += len(extras)
        image, residual = jpeg_and_residual(file_name)
        slot_masks = [decode_rle(rle) for rle in slot_rles]
        feats = features_for_rles_context(extras, image, residual, slot_masks)
        kept = keep_extra_rles(extras, feats, clf, thr_keep)
        n_kept += len(kept)
        merged, added = merge_rles(slot_rles, kept)
        n_added += added
        frames[file_name] = pred_to_df(Path(file_name).stem, merged)

    pred_df = pd.concat(frames.values(), ignore_index=True)
    br = score_pq(gt_df, pred_df)
    locked = {
        "thr": LOCKED_THR,
        "min_area": LOCKED_MIN_AREA,
        "rejector_thr": thr_keep,
        "n_extras_before_filter": n_extras,
        "n_extras_kept": n_kept,
        "n_unet_added": n_added,
        **_row(br, int(len(pred_df))),
    }
    slices = year_slices(val_files, frames, views, by_file)
    gate_vs_erode = gate_unet_merge(locked, erode_row, slices["2017-2020"], erode_slices["2017-2020"])
    gate_vs_orig = gate_unet_merge(locked, orig_row, slices["2017-2020"], orig_slices["2017-2020"])
    pred_csv = FAST_ARTIFACTS / "erode_slot1_ctx_merge_fold0_pred.csv"
    pred_df.to_csv(pred_csv, index=False)
    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "family": "erode_slot1_ctx_merge",
        "did_not_refit_rejector": True,
        "did_not_overwrite_ctx_or_resnet": True,
        "kaggle_submit_called": False,
        "do_not_submit": True,
        "locked_a_priori": {"thr": LOCKED_THR, "min_area": LOCKED_MIN_AREA, "rejector_thr": thr_keep},
        "locked_row": locked,
        "erode_slot1": erode_row,
        "orig_slot1": orig_row,
        "year_slice": slices,
        "gate_vs_erode_slot1": gate_vs_erode,
        "gate_vs_orig_slot1": gate_vs_orig,
        "pred_csv": str(pred_csv),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OFFICIAL.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"path": str(OFFICIAL), "locked": locked, "gate_vs_erode": gate_vs_erode}, indent=2))


if __name__ == "__main__":
    main()
