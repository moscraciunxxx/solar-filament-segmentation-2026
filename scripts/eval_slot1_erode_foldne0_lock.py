#!/usr/bin/env python3
"""Lock Slot1 erode 3x3 x1 on fold!=0, then confirm official fold-0 PQ.

Hypothesis is pre-registered: erode 3x3 x1 (not a fold-0 argmax).
CPU only. Does not submit. Year-shift stays Slot 2.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_slot1_nearmiss_morph import morph_rles
from eval_unet_merge import gate_unet_merge
from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED
from score_faint_merge import SLOT1_PRED_CSV, _stem_frames, load_or_infer_slot1
from score_family import year_slices

FOLDNE0_CSV = FAST_ARTIFACTS / "yolo_yolo11m_seg_1280_foldne0_conf0.35_sz1280_m0_base.csv"
OFFICIAL = ARTIFACTS / "slot1_erode_foldne0_lock_fold0_official.json"
HYP = ("erode", 3, 1)


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


def score_frames(val_files, frames, views, by_file) -> dict:
    pred_df = pd.concat((frames[f] for f in val_files), ignore_index=True)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    br = score_pq(gt_df, pred_df)
    return _row(br, len(val_files), int(len(pred_df)))


def apply_erode(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {}
    for file_name, df in frames.items():
        rles = df["segmentation_rle"].astype(str).tolist()
        new_rles = morph_rles(rles, HYP[0], HYP[1], HYP[2])
        out[file_name] = pred_to_df(Path(file_name).stem, new_rles)
    return out


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    fold0 = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    foldne0 = sorted(fn for fn, fold in fold_of.items() if fold != 0)
    if not FOLDNE0_CSV.exists():
        raise SystemExit(f"missing {FOLDNE0_CSV}")
    ne0 = _stem_frames(pd.read_csv(FOLDNE0_CSV), foldne0)
    ne0_erode = apply_erode(ne0)
    fit_ident = score_frames(foldne0, ne0, views, by_file)
    fit_erode = score_frames(foldne0, ne0_erode, views, by_file)
    lock_ok = fit_erode["pq"] > fit_ident["pq"] and fit_erode["tp"] >= fit_ident["tp"]
    print({"foldne0_identity": fit_ident, "foldne0_erode": fit_erode, "lock_ok": lock_ok}, flush=True)

    slot1_fold0 = load_or_infer_slot1(fold0, device="cpu")
    fold0_ident = score_frames(fold0, slot1_fold0, views, by_file)
    fold0_erode_frames = apply_erode(slot1_fold0)
    fold0_erode = score_frames(fold0, fold0_erode_frames, views, by_file)
    slices = year_slices(fold0, fold0_erode_frames, views, by_file)
    slot1_slices = year_slices(fold0, slot1_fold0, views, by_file)
    if lock_ok:
        gate = gate_unet_merge(fold0_erode, fold0_ident, slices["2017-2020"], slot1_slices["2017-2020"])
    else:
        gate = {
            "decision": "KILL_FOLDNE0_LOCK_FAILED",
            "next_action": "Erode lost on fold!=0. Do not promote the fold-0 diagnosis peek. Do not upload.",
            "do_not_submit_from_this_script": True,
            "do_not_replace_slot2": True,
        }

    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "family": "slot1_erode_3x1",
        "cpu_only": True,
        "did_not_train": True,
        "kaggle_submit_called": False,
        "do_not_submit": True,
        "hypothesis_pre_registered": {"op": HYP[0], "k": HYP[1], "iters": HYP[2]},
        "fold0_peek_already_happened": True,
        "fold0_peek_note": "Diagnosis sweep printed erode 0.4365 on fold-0 before this lock. Confirmation is still fold!=0-locked.",
        "foldne0": {"n_images": len(foldne0), "identity": fit_ident, "erode": fit_erode, "lock_ok": lock_ok},
        "locked_row": fold0_erode if lock_ok else fold0_ident,
        "fold0_erode": fold0_erode,
        "slot1_identity": fold0_ident,
        "year_slice": slices,
        "gate": gate,
        "pred_csv": None,
    }
    if lock_ok:
        pred_csv = FAST_ARTIFACTS / "slot1_erode_3x1_fold0_pred.csv"
        pd.concat(fold0_erode_frames.values(), ignore_index=True).to_csv(pred_csv, index=False)
        out["pred_csv"] = str(pred_csv)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OFFICIAL.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"path": str(OFFICIAL), "lock_ok": lock_ok, "fold0_erode": fold0_erode, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
