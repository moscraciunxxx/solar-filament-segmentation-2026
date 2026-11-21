#!/usr/bin/env python3
"""Official fold-0 PQ for the published merckle-046 classical recipe.

Locks DARK_PERCENTILE=90. Does not sweep from public 0.46. Does not submit.
Does not read MAGFiLO test labels. Our older classical 0.005 used a different
CLAHE + sunspot + top-40 pipeline on 24 images — not this recipe.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.classical import detect_merckle
from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.infer import masks_to_rles
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR
from filament.preprocess import read_gray
from score_family import M_1280, S_MEDOID, year_slices


def _predict_one(file_name: str) -> tuple[str, list[str]]:
    image = read_gray(TRAIN_IMG_DIR / file_name)
    return file_name, masks_to_rles(detect_merckle(image, dark_percentile=90.0))


def gate_merckle(row: dict, desert: dict) -> dict:
    pq = float(row["pq"])
    desert_pq = float(desert["pq"])
    desert_drop = pq - desert_pq
    d_slot1 = pq - M_1280["pq"]
    if pq >= 0.40 and desert_drop < 0.08:
        decision = "PROMOTE_LOOK"
        next_action = (
            "Merckle-faithful cleared ~Slot-1 locally. After quota reset, at most "
            "one public look of this STANDALONE csv. Year-shift stays Slot 2 until "
            "that look. Do not attach to Slot 1 from this number."
        )
    elif pq < 0.15:
        decision = "KILL"
        next_action = (
            "Local official PQ collapsed. Public 0.46 is not a certificate. "
            "Do not upload. Do not retune percentile from the public score."
        )
    elif pq >= 0.30:
        decision = "RESEARCH_HOLD"
        next_action = (
            "Real morphology signal below Slot-1 replace. Diagnosis only. "
            "Do not upload. Do not replace Slot 2. Do not hybrid-attach from this look."
        )
    else:
        decision = "RESEARCH_HOLD"
        next_action = (
            "Below 0.30 official. Public 0.46 does not authorize a look. "
            "Do not upload. Year-shift stays Slot 2."
        )
    return {
        "decision": decision,
        "next_action": next_action,
        "vs_m1280_d_pq": d_slot1,
        "vs_s_medoid_d_pq": pq - S_MEDOID["pq"],
        "desert_2017_2020_drop_vs_overall": desert_drop,
        "public_046_is_not_local_certificate": True,
        "do_not_submit_from_this_script": True,
        "do_not_replace_slot2": True,
        "do_not_retune_percentile_from_public": True,
        "baselines": {"s_medoid": S_MEDOID, "m_1280": M_1280},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == args.fold)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))

    frames: dict[str, pd.DataFrame] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for file_name, rles in tqdm(
            pool.map(_predict_one, val_files),
            total=len(val_files),
            desc="merckle-046 fold-0",
        ):
            frames[file_name] = pred_to_df(Path(file_name).stem, rles)

    pred_df = pd.concat(frames.values(), ignore_index=True)
    br = score_pq(gt_df, pred_df)
    locked = {
        "dark_percentile": 90.0,
        "pq": br.pq,
        "tp": br.tp,
        "fp": br.fp,
        "fn": br.fn,
        "mean_tp_iou": br.mean_tp_iou,
        "n_pred": int(len(pred_df)),
        "n_val": len(val_files),
        "pred_per_img": len(pred_df) / max(len(val_files), 1),
    }
    slices = year_slices(val_files, frames, views, by_file)
    decision = gate_merckle(locked, slices["2017-2020"])

    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    pred_csv = FAST_ARTIFACTS / f"merckle046_fold{args.fold}_pred.csv"
    pred_df.to_csv(pred_csv, index=False)
    out = {
        "family": "merckle046_classical",
        "source_kernel": "refs/kernels/merckle-046/simple-dark-filament-segmentation-0-46.ipynb",
        "metric": "official host PQ on grouped fold-0 (every annotator-image)",
        "fold": args.fold,
        "n_val": len(val_files),
        "locked_recipe": {
            "clahe": False,
            "blur_sigma": 40.0,
            "border": "BORDER_REFLECT",
            "dark_percentile": 90.0,
            "threshold": "max(1, p90(dark_residual[disk]))",
            "morph": "dilate 3x3 x1",
            "min_area": 80,
            "max_area": 150000,
            "max_fill": 0.99,
            "top_k": 120,
            "no_circularity_sunspot_filter": True,
            "no_merge_nearby": True,
        },
        "not_our_old_classical": {
            "old_fold0_pq": 0.005,
            "old_n_val": 24,
            "old_used_clahe_sunspot_top40": True,
        },
        "locked_row": locked,
        "year_slice": slices,
        "gate": decision,
        "pred_csv": str(pred_csv),
        "do_not_submit": True,
    }
    text = json.dumps(out, indent=2)
    (FAST_ARTIFACTS / f"merckle046_fold{args.fold}_official.json").write_text(text)
    repo = ARTIFACTS / f"merckle046_fold{args.fold}_official.json"
    repo.write_text(text)
    print(json.dumps({"path": str(repo), "locked": locked, "gate": decision}, indent=2))


if __name__ == "__main__":
    main()
