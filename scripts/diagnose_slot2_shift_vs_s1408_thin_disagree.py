#!/usr/bin/env python3
"""CPU: locked Slot2 year-shift vs thin s@1408 HOLD pred-pred disagreement.

Hedge diagnosis for the locked private pair vs the best dest-safe HOLD.
NOT a license to replace Slot2 or extras-OR. No train, no submit, no MPS,
no leftover start, no merged CSV.

Slot2 upload is 56148889 = submission_yolo11s_shift_conf035.csv
(YOLO11s-seg year-shift imgsz=1024 conf=0.35). Prefer that pred CSV, not
yolo11l_seg_shift (later dest; never the uploaded Slot2).

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

# Uploaded Slot2 identity (yolo11s year-shift). L-shift exists but is not 56148889.
SLOT2_CSV = FAST_ARTIFACTS / "yolo_yolo11s_seg_shift_fold0_conf0.35_sz1024_m0_base_shift_pq.csv"
L_SHIFT_CSV = FAST_ARTIFACTS / "yolo11l_seg_shift_fold0_conf0.35_pred.csv"
S1408_CSV = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_1408_fold0_conf0.35_pred.csv"
OUT = ARTIFACTS / "slot2_shift_vs_s1408_thin_disagree.json"
IOU_THR = 0.5
K_TO_050 = 220
SLOT2_OFFICIAL = {
    "file": "artifacts/yolo11s_shift_fold0_pq.json",
    "tp": 852,
    "fp": 629,
    "fn": 737,
    "pq": 0.35227813650808426,
    "n_pred": 918,
    "kaggle_ref": 56148889,
    "kaggle_file": "submission_yolo11s_shift_conf035.csv",
    "public": 0.24,
    "weights": "~/.cache/filament-segmentation-2026/weights/yolo11s_seg_shift/weights/best.pt",
}
S1408_OFFICIAL = {
    "file": "artifacts/yolo11s_seg_erode_gt_1408_fold0_official.json",
    "tp": 959,
    "fp": 410,
    "fn": 630,
    "pq": 0.4384,
}


def main() -> None:
    if not SLOT2_CSV.exists():
        raise SystemExit(f"missing {SLOT2_CSV}")
    if not S1408_CSV.exists():
        raise SystemExit(f"missing {S1408_CSV}")

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)

    slot2_df = pd.read_csv(SLOT2_CSV)
    s1408_df = pd.read_csv(S1408_CSV)
    n_slot2_pred = int(len(slot2_df))
    n_s1408_pred = int(len(s1408_df))
    slot2 = _by_stem(SLOT2_CSV)
    s1408 = _by_stem(S1408_CSV)

    n_matched_pairs = 0
    n_slot2_pred_only = 0
    n_s1408_pred_only = 0
    n_slot2_matched_inst = 0
    n_s1408_matched_inst = 0

    n_gt_tp_both = 0
    n_gt_tp_shift_only = 0
    n_gt_tp_s1408_only = 0
    n_gt_fn_both = 0
    images_shift_hits_s1408_misses: set[str] = set()
    images_s1408_hits_shift_misses: set[str] = set()

    for file_name in val_files:
        stem = Path(file_name).stem
        a_layers = _layers(slot2.get(stem, []))
        b_layers = _layers(s1408.get(stem, []))
        n_a, n_b = int(a_layers.shape[0]), int(b_layers.shape[0])

        if n_a and n_b:
            iou, _ = overlap_matrices(a_layers, b_layers)
            hit = iou > IOU_THR
            n_matched_pairs += int(hit.sum().item())
            a_hit = hit.any(dim=1)
            b_hit = hit.any(dim=0)
            n_slot2_matched_inst += int(a_hit.sum().item())
            n_s1408_matched_inst += int(b_hit.sum().item())
            n_slot2_pred_only += int((~a_hit).sum().item())
            n_s1408_pred_only += int((~b_hit).sum().item())
        else:
            n_slot2_pred_only += n_a
            n_s1408_pred_only += n_b

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
                    images_shift_hits_s1408_misses.add(stem)
                elif b_ok:
                    n_gt_tp_s1408_only += 1
                    images_s1408_hits_shift_misses.add(stem)
                else:
                    n_gt_fn_both += 1

    n_gt_tp_unique_either = n_gt_tp_shift_only + n_gt_tp_s1408_only
    n_images_shift_hits_s1408_misses = len(images_shift_hits_s1408_misses)
    n_images_s1408_hits_shift_misses = len(images_s1408_hits_shift_misses)
    agree_frac_slot2 = (n_slot2_matched_inst / n_slot2_pred) if n_slot2_pred else None
    agree_frac_s1408 = (n_s1408_matched_inst / n_s1408_pred) if n_s1408_pred else None
    mostly_agree = bool(
        agree_frac_slot2 is not None
        and agree_frac_s1408 is not None
        and agree_frac_slot2 >= 0.90
        and agree_frac_s1408 >= 0.90
    )

    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": "cpu_only_slot2_shift_vs_s1408_thin_disagree",
        "skill": "hidden-exam-discipline",
        "diagnosis_not_a_builder": True,
        "cpu_only": True,
        "did_not_train": True,
        "did_not_start_mps": True,
        "did_not_submit": True,
        "kaggle_submit_called": False,
        "do_not_submit": True,
        "do_not_or": True,
        "do_not_replace_slot2": True,
        "k_to_0.50": K_TO_050,
        "unique_either": n_gt_tp_unique_either,
        "shift_only": n_gt_tp_shift_only,
        "s1408_only": n_gt_tp_s1408_only,
        "unique_either_is_not_or_license": True,
        "n_gt_tp_unique_either": n_gt_tp_unique_either,
        "n_gt_tp_shift_only": n_gt_tp_shift_only,
        "n_gt_tp_s1408_only": n_gt_tp_s1408_only,
        "n_gt_tp_unique_either_lt_k": n_gt_tp_unique_either < K_TO_050,
        "or_relevant_increment_is_side_only_not_unique_either": True,
        "leftover_to_start": False,
        "next_id": None,
        "recommend_start": False,
        "parent_decision": "NO_LOOK",
        "remaining_looks": 2,
        "pair_stays": [56173362, 56148889],
        "matching": "IoU>0.5, not Hungarian; pred-pred per JPEG stem; GT TPs per annotator-image",
        "slot2_csv_used": str(SLOT2_CSV),
        "slot2_identity": {
            "ref": 56148889,
            "family": "year-shift",
            "dest": "yolo11s_seg_shift",
            "kaggle_file": "submission_yolo11s_shift_conf035.csv",
            "why_not_l_shift": (
                "kaggle_quota_* and submission_yolo11s_shift_conf035_meta.json "
                "pin Slot2 56148889 to yolo11s_seg_shift @1024 conf=0.35. "
                "yolo11l_seg_shift is a later dest, not the uploaded Slot2."
            ),
            "l_shift_pred_exists": L_SHIFT_CSV.exists(),
            "l_shift_pred": str(L_SHIFT_CSV),
        },
        "s1408_csv": str(S1408_CSV),
        "officials": {"slot2_year_shift": SLOT2_OFFICIAL, "thin_s1408": S1408_OFFICIAL},
        "n_slot2_pred": n_slot2_pred,
        "n_s1408_pred": n_s1408_pred,
        "n_matched": n_matched_pairs,
        "n_matched_IoUgt0.5": n_matched_pairs,
        "n_slot2_matched_inst": n_slot2_matched_inst,
        "n_s1408_matched_inst": n_s1408_matched_inst,
        "n_slot2_pred_only": n_slot2_pred_only,
        "n_s1408_pred_only": n_s1408_pred_only,
        "agree_frac_slot2": agree_frac_slot2,
        "agree_frac_s1408": agree_frac_s1408,
        "mostly_agree": mostly_agree,
        "gt_tp": {
            "both": n_gt_tp_both,
            "shift_only": n_gt_tp_shift_only,
            "s1408_only": n_gt_tp_s1408_only,
            "unique_either": n_gt_tp_unique_either,
        },
        "gt_tp_side": {
            "n_gt_tp_both": n_gt_tp_both,
            "n_gt_tp_shift_only": n_gt_tp_shift_only,
            "n_gt_tp_s1408_only": n_gt_tp_s1408_only,
            "n_gt_tp_unique_either": n_gt_tp_unique_either,
            "n_gt_fn_both": n_gt_fn_both,
            "n_images_shift_hits_s1408_misses": n_images_shift_hits_s1408_misses,
            "n_images_s1408_hits_shift_misses": n_images_s1408_hits_shift_misses,
        },
        "why": (
            "Pred-pred IoU>0.5 pairing on fold-0. Locked-pair hedge (Slot2 "
            "yolo11s year-shift 56148889) vs best dest-safe HOLD (thin s@1408 "
            "0.4384). Disagreement is diagnosis, not an OR license and not a "
            "Slot2 replace (same rule as 116 / 130 unique). extras-OR stays "
            "vetoed even if unique_either >= 220. k_to_0.50=220. parent "
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
                "s1408_only": n_gt_tp_s1408_only,
                "slot2_csv_used": str(SLOT2_CSV),
                "unique_either_is_not_or_license": True,
                "do_not_or": True,
                "do_not_replace_slot2": True,
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
