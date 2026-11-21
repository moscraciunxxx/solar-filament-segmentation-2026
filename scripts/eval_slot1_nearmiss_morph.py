#!/usr/bin/env python3
"""CPU official fold-0 PQ: morph grow/shrink frozen Slot1 masks.

Does not add instances. Does not train. Does not submit.
Locked official row is dilate 3x3 x1 (a priori thin-filament width), not the
max of a PQ sweep. Other kernels are diagnosis only.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, IMAGE_SIZE, SEED
from filament.postprocess import resolve_overlaps
from filament.rle import decode_rle, encode_rle
from score_faint_merge import load_or_infer_slot1
from score_family import year_slices
from eval_unet_merge import gate_unet_merge

PYTHON_NOTE = "~/.venvs/filament/bin/python"
OFFICIAL = ARTIFACTS / "slot1_nearmiss_morph_fold0_official.json"
LOCKED = ("dilate", 3, 1)


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


def morph_rles(rles: list[str], op: str, k: int, iters: int) -> list[str]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = []
    masks = []
    for rle in rles:
        m = decode_rle(rle, IMAGE_SIZE, IMAGE_SIZE)
        if op == "dilate":
            m = cv2.dilate(m, kernel, iterations=iters)
        elif op == "close":
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=iters)
        elif op == "erode":
            m = cv2.erode(m, kernel, iterations=iters)
        else:
            raise ValueError(op)
        if int(m.sum()) == 0:
            continue
        masks.append(m)
    masks = resolve_overlaps(masks)
    for m in masks:
        if int(m.sum()) > 0:
            out.append(encode_rle(m))
    return out


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    slot1 = load_or_infer_slot1(val_files, device="cpu")

    recipes = [
        ("identity", 1, 0),
        ("dilate", 3, 1),
        ("dilate", 5, 1),
        ("close", 3, 1),
        ("close", 5, 1),
        ("erode", 3, 1),
    ]
    sweep = []
    pred_cache: dict[str, dict[str, pd.DataFrame]] = {}
    for op, k, iters in recipes:
        frames = {}
        for file_name in val_files:
            rles = slot1[file_name]["segmentation_rle"].astype(str).tolist()
            if op == "identity":
                new_rles = rles
            else:
                new_rles = morph_rles(rles, op, k, iters)
            frames[file_name] = pred_to_df(Path(file_name).stem, new_rles)
        pred_df = pd.concat(frames.values(), ignore_index=True)
        br = score_pq(gt_df, pred_df)
        key = f"{op}_{k}x{iters}"
        row = {"op": op, "k": k, "iters": iters, **_row(br, len(val_files), int(len(pred_df)))}
        sweep.append(row)
        pred_cache[key] = frames
        print(row, flush=True)

    locked_key = f"{LOCKED[0]}_{LOCKED[1]}x{LOCKED[2]}"
    locked = next(r for r in sweep if r["op"] == LOCKED[0] and r["k"] == LOCKED[1] and r["iters"] == LOCKED[2])
    ident = next(r for r in sweep if r["op"] == "identity")
    slices = year_slices(val_files, pred_cache[locked_key], views, by_file)
    slot1_slices = year_slices(val_files, slot1, views, by_file)
    gate = gate_unet_merge(locked, ident, slices["2017-2020"], slot1_slices["2017-2020"])
    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "family": "slot1_nearmiss_morph",
        "cpu_only": True,
        "did_not_train": True,
        "kaggle_submit_called": False,
        "do_not_submit": True,
        "locked_a_priori": {"op": LOCKED[0], "k": LOCKED[1], "iters": LOCKED[2]},
        "locked_not_argmax_of_sweep": True,
        "sweep_diagnosis_only": sweep,
        "locked_row": locked,
        "slot1_identity": ident,
        "year_slice": slices,
        "gate": gate,
        "python": PYTHON_NOTE,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OFFICIAL.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"path": str(OFFICIAL), "locked": locked, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
