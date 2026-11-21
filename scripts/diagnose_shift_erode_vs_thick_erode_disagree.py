#!/usr/bin/env python3
"""CPU: shift+erode HOLD vs thick s@1408 erode3x1 HOLD pred-pred disagreement.

Hedge diagnosis, not a merge. extras-OR stays vetoed even if unique_either >= 220.
Wave37: unique_either is both directions; OR-relevant increment is each side's
*_only count (239 unique_either vs Slot1 was only 120 extras-OR onto Slot1).
No train, no submit, no MPS, no leftover start, no merged CSV.
Matching: official IoU>0.5, not Hungarian. Pred-pred is per JPEG stem.
k_to_0.50=220. Parent decision NO_LOOK. pair stays 56173362 / 56148889.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_erode_large_fn_overlap import _by_stem, _layers
from filament.coco_io import grouped_folds, load_coco, rasterize_polygons
from filament.metric import overlap_matrices
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED
from filament.rle import encode_rle, rles_to_layers

SHIFT_CSV = FAST_ARTIFACTS / "yolo11l_seg_shift_erode3x1_fold0_pred.csv"
THICK_CSV = FAST_ARTIFACTS / "yolo11s_seg_1408_erode3x1_fold0_pred.csv"
OUT = ARTIFACTS / "shift_erode_vs_thick_erode_disagree.json"
IOU_THR = 0.5
K_TO_050 = 220
SHIFT_OFFICIAL = {
    "file": "artifacts/yolo11l_seg_shift_erode3x1_fold0_official.json",
    "tp": 969,
    "fp": 370,
    "fn": 620,
    "pq": 0.4429,
}
THICK_OFFICIAL = {
    "file": "artifacts/yolo11s_seg_1408_erode3x1_fold0_official.json",
    "tp": 950,
    "fp": 387,
    "fn": 639,
    "pq": 0.4388,
}


def main() -> None:
    if not SHIFT_CSV.exists():
        raise SystemExit(f"missing {SHIFT_CSV}")
    if not THICK_CSV.exists():
        raise SystemExit(f"missing {THICK_CSV}")

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)

    shift_df = pd.read_csv(SHIFT_CSV)
    thick_df = pd.read_csv(THICK_CSV)
    n_shift_pred = int(len(shift_df))
    n_thick_pred = int(len(thick_df))
    shift = _by_stem(SHIFT_CSV)
    thick = _by_stem(THICK_CSV)

    n_matched_pairs = 0
    n_shift_only = 0
    n_thick_only = 0
    n_shift_matched_inst = 0
    n_thick_matched_inst = 0

    n_gt_tp_both = 0
    n_gt_tp_shift_only = 0
    n_gt_tp_thick_only = 0
    n_gt_fn_both = 0
    images_shift_hits_thick_misses: set[str] = set()
    images_thick_hits_shift_misses: set[str] = set()

    for file_name in val_files:
        stem = Path(file_name).stem
        a_layers = _layers(shift.get(stem, []))
        b_layers = _layers(thick.get(stem, []))
        n_a, n_b = int(a_layers.shape[0]), int(b_layers.shape[0])

        if n_a and n_b:
            iou, _ = overlap_matrices(a_layers, b_layers)
            hit = iou > IOU_THR
            n_matched_pairs += int(hit.sum().item())
            a_hit = hit.any(dim=1)
            b_hit = hit.any(dim=0)
            n_shift_matched_inst += int(a_hit.sum().item())
            n_thick_matched_inst += int(b_hit.sum().item())
            n_shift_only += int((~a_hit).sum().item())
            n_thick_only += int((~b_hit).sum().item())
        else:
            n_shift_only += n_a
            n_thick_only += n_b

        for image_id in by_file[file_name]:
            view = views[image_id]
            if not view.filaments:
                continue
            fils = list(view.filaments)
            gt_layers = torch.from_numpy(
                rles_to_layers([encode_rle(rasterize_polygons([fil.polygon])) for fil in fils])
            )
            if n_a:
                ai, _ = overlap_matrices(gt_layers, a_layers)
                a_max = ai.max(dim=1).values
            else:
                a_max = torch.zeros(len(fils))
            if n_b:
                bi, _ = overlap_matrices(gt_layers, b_layers)
                b_max = bi.max(dim=1).values
            else:
                b_max = torch.zeros(len(fils))
            for i in range(len(fils)):
                a_ok = float(a_max[i]) > IOU_THR
                b_ok = float(b_max[i]) > IOU_THR
                if a_ok and b_ok:
                    n_gt_tp_both += 1
                elif a_ok:
                    n_gt_tp_shift_only += 1
                    images_shift_hits_thick_misses.add(stem)
                elif b_ok:
                    n_gt_tp_thick_only += 1
                    images_thick_hits_shift_misses.add(stem)
                else:
                    n_gt_fn_both += 1

    n_gt_tp_unique_either = n_gt_tp_shift_only + n_gt_tp_thick_only
    n_images_shift_hits_thick_misses = len(images_shift_hits_thick_misses)
    n_images_thick_hits_shift_misses = len(images_thick_hits_shift_misses)
    agree_frac_shift = (n_shift_matched_inst / n_shift_pred) if n_shift_pred else None
    agree_frac_thick = (n_thick_matched_inst / n_thick_pred) if n_thick_pred else None
    mostly_agree = bool(
        agree_frac_shift is not None
        and agree_frac_thick is not None
        and agree_frac_shift >= 0.90
        and agree_frac_thick >= 0.90
    )

    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": "cpu_only_shift_erode_vs_thick_erode_disagree",
        "skill": "hidden-exam-discipline",
        "diagnosis_not_a_builder": True,
        "cpu_only": True,
        "did_not_train": True,
        "did_not_start_mps": True,
        "did_not_submit": True,
        "kaggle_submit_called": False,
        "do_not_submit": True,
        "do_not_or": True,
        "k_to_0.50": K_TO_050,
        "unique_either": n_gt_tp_unique_either,
        "shift_only": n_gt_tp_shift_only,
        "thick_only": n_gt_tp_thick_only,
        "unique_either_is_not_or_license": True,
        "n_gt_tp_unique_either": n_gt_tp_unique_either,
        "n_gt_tp_shift_only": n_gt_tp_shift_only,
        "n_gt_tp_thick_only": n_gt_tp_thick_only,
        "n_gt_tp_unique_either_lt_k": n_gt_tp_unique_either < K_TO_050,
        "or_relevant_increment_is_side_only_not_unique_either": True,
        "leftover_to_start": False,
        "next_id": None,
        "recommend_start": False,
        "parent_decision": "NO_LOOK",
        "remaining_looks": 2,
        "pair_stays": [56173362, 56148889],
        "matching": "IoU>0.5, not Hungarian; pred-pred per JPEG stem; GT TPs per annotator-image",
        "shift_csv": str(SHIFT_CSV),
        "thick_csv": str(THICK_CSV),
        "officials": {"shift_erode": SHIFT_OFFICIAL, "thick_s1408_erode": THICK_OFFICIAL},
        "n_shift_pred": n_shift_pred,
        "n_thick_pred": n_thick_pred,
        "n_matched": n_matched_pairs,
        "n_matched_IoUgt0.5": n_matched_pairs,
        "n_shift_matched_inst": n_shift_matched_inst,
        "n_thick_matched_inst": n_thick_matched_inst,
        "n_shift_pred_only": n_shift_only,
        "n_thick_pred_only": n_thick_only,
        "agree_frac_shift": agree_frac_shift,
        "agree_frac_thick": agree_frac_thick,
        "mostly_agree": mostly_agree,
        "gt_tp": {
            "both": n_gt_tp_both,
            "shift_only": n_gt_tp_shift_only,
            "thick_only": n_gt_tp_thick_only,
            "unique_either": n_gt_tp_unique_either,
        },
        "gt_tp_side": {
            "n_gt_tp_both": n_gt_tp_both,
            "n_gt_tp_shift_only": n_gt_tp_shift_only,
            "n_gt_tp_thick_only": n_gt_tp_thick_only,
            "n_gt_tp_unique_either": n_gt_tp_unique_either,
            "n_gt_fn_both": n_gt_fn_both,
            "n_images_shift_hits_thick_misses": n_images_shift_hits_thick_misses,
            "n_images_thick_hits_shift_misses": n_images_thick_hits_shift_misses,
        },
        "why": (
            "Pred-pred IoU>0.5 pairing on fold-0 HOLD CSVs. Disagreement is "
            "diagnosis, not an OR license (same rule as 116 / 130 unique). "
            "Wave37: unique_either is both directions; extras-OR onto one side "
            "is that side's *_only (239 unique_either vs Slot1 was only 120 "
            "thick_only). extras-OR stays vetoed even if unique_either >= 220. "
            "k_to_0.50=220. shift+erode HOLD vs thick s@1408 erode HOLD; parent "
            "NO_LOOK. pair stays 56173362 / 56148889. leftovers stay empty. "
            "next_id null."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                "leftover_to_start": False,
                "next_id": None,
                "recommend_start": False,
                "unique_either": n_gt_tp_unique_either,
                "shift_only": n_gt_tp_shift_only,
                "thick_only": n_gt_tp_thick_only,
                "unique_either_is_not_or_license": True,
                "n_matched": n_matched_pairs,
                "n_gt_tp_unique_either": n_gt_tp_unique_either,
                "n_gt_tp_both": n_gt_tp_both,
                "n_gt_tp_shift_only": n_gt_tp_shift_only,
                "n_gt_tp_thick_only": n_gt_tp_thick_only,
                "do_not_or": True,
                "do_not_submit": True,
                "parent_decision": "NO_LOOK",
                "kaggle_submit_called": False,
            },
            indent=2,
        )
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
