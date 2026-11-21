#!/usr/bin/env python3
"""CPU: Slot1-erode vs s-erode@1408 pred-pred disagreement on fold-0.

Hedge diagnosis, not a merge. extras-OR stays vetoed. No train, no submit.
Matching: official IoU>0.5, not Hungarian. Pred-pred is per JPEG stem.
116 / 130 unique GT TPs is not an OR license; k_to_0.50=220.
Parent already vetoed yolo11s_seg_erode_gt_1408 (official 0.4384, forecast ~0.332).
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

SLOT1_CSV = FAST_ARTIFACTS / "slot1_erode_3x1_fold0_pred.csv"
S1408_CSV = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_1408_fold0_conf0.35_pred.csv"
OUT = ARTIFACTS / "s1408_vs_slot1_disagree.json"
IOU_THR = 0.5
K_TO_050 = 220
SLOT1_OFFICIAL = {
    "file": "artifacts/slot1_erode_foldne0_lock_fold0_official.json",
    "tp": 949,
    "fp": 385,
    "fn": 640,
    "pq": 0.4365,
}
S1408_OFFICIAL = {
    "file": "artifacts/yolo11s_seg_erode_gt_1408_fold0_official.json",
    "tp": 959,
    "fp": 410,
    "fn": 630,
    "pq": 0.4384,
}


def main() -> None:
    if not SLOT1_CSV.exists():
        raise SystemExit(f"missing {SLOT1_CSV}")
    if not S1408_CSV.exists():
        raise SystemExit(f"missing {S1408_CSV}")

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)

    slot1_df = pd.read_csv(SLOT1_CSV)
    s1408_df = pd.read_csv(S1408_CSV)
    n_slot1_pred = int(len(slot1_df))
    n_s1408_pred = int(len(s1408_df))
    slot1 = _by_stem(SLOT1_CSV)
    s1408 = _by_stem(S1408_CSV)

    n_matched_pairs = 0
    n_slot1_only = 0
    n_s1408_only = 0
    n_slot1_matched_inst = 0
    n_s1408_matched_inst = 0

    n_gt_tp_both = 0
    n_gt_tp_slot1_only = 0
    n_gt_tp_s1408_only = 0
    n_gt_fn_both = 0
    images_slot1_hits_s1408_misses: set[str] = set()
    images_s1408_hits_slot1_misses: set[str] = set()

    for file_name in val_files:
        stem = Path(file_name).stem
        s1_layers = _layers(slot1.get(stem, []))
        s_layers = _layers(s1408.get(stem, []))
        n_s1, n_s = int(s1_layers.shape[0]), int(s_layers.shape[0])

        if n_s1 and n_s:
            iou, _ = overlap_matrices(s1_layers, s_layers)
            hit = iou > IOU_THR
            n_matched_pairs += int(hit.sum().item())
            s1_hit = hit.any(dim=1)
            s_hit = hit.any(dim=0)
            n_slot1_matched_inst += int(s1_hit.sum().item())
            n_s1408_matched_inst += int(s_hit.sum().item())
            n_slot1_only += int((~s1_hit).sum().item())
            n_s1408_only += int((~s_hit).sum().item())
        else:
            n_slot1_only += n_s1
            n_s1408_only += n_s

        for image_id in by_file[file_name]:
            view = views[image_id]
            if not view.filaments:
                continue
            fils = list(view.filaments)
            gt_layers = torch.from_numpy(
                rles_to_layers([encode_rle(rasterize_polygons([fil.polygon])) for fil in fils])
            )
            if n_s1:
                s1i, _ = overlap_matrices(gt_layers, s1_layers)
                s1_max = s1i.max(dim=1).values
            else:
                s1_max = torch.zeros(len(fils))
            if n_s:
                si, _ = overlap_matrices(gt_layers, s_layers)
                s_max = si.max(dim=1).values
            else:
                s_max = torch.zeros(len(fils))
            for i in range(len(fils)):
                s1_ok = float(s1_max[i]) > IOU_THR
                s_ok = float(s_max[i]) > IOU_THR
                if s1_ok and s_ok:
                    n_gt_tp_both += 1
                elif s1_ok:
                    n_gt_tp_slot1_only += 1
                    images_slot1_hits_s1408_misses.add(stem)
                elif s_ok:
                    n_gt_tp_s1408_only += 1
                    images_s1408_hits_slot1_misses.add(stem)
                else:
                    n_gt_fn_both += 1

    n_images_slot1_hits_s1408_misses = len(images_slot1_hits_s1408_misses)
    n_images_s1408_hits_slot1_misses = len(images_s1408_hits_slot1_misses)
    agree_frac_slot1 = (n_slot1_matched_inst / n_slot1_pred) if n_slot1_pred else None
    agree_frac_s1408 = (n_s1408_matched_inst / n_s1408_pred) if n_s1408_pred else None
    mostly_agree = bool(
        agree_frac_slot1 is not None
        and agree_frac_s1408 is not None
        and agree_frac_slot1 >= 0.90
        and agree_frac_s1408 >= 0.90
    )

    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": "cpu_only_s1408_vs_slot1_disagree",
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
        "n_gt_tp_s1408_only_lt_k": n_gt_tp_s1408_only < K_TO_050,
        "leftover_to_start": False,
        "next_id": None,
        "recommend_start": False,
        "parent_decision": "NO_LOOK",
        "remaining_looks": 2,
        "pair_stays": [56173362, 56148889],
        "matching": "IoU>0.5, not Hungarian; pred-pred per JPEG stem; GT TPs per annotator-image",
        "slot1_csv": str(SLOT1_CSV),
        "s1408_csv": str(S1408_CSV),
        "officials": {"slot1": SLOT1_OFFICIAL, "s1408": S1408_OFFICIAL},
        "n_slot1_pred": n_slot1_pred,
        "n_s1408_pred": n_s1408_pred,
        "n_matched": n_matched_pairs,
        "n_matched_IoUgt0.5": n_matched_pairs,
        "n_slot1_matched_inst": n_slot1_matched_inst,
        "n_s1408_matched_inst": n_s1408_matched_inst,
        "n_slot1_only": n_slot1_only,
        "n_s1408_only": n_s1408_only,
        "agree_frac_slot1": agree_frac_slot1,
        "agree_frac_s1408": agree_frac_s1408,
        "mostly_agree": mostly_agree,
        "gt_tp": {
            "both": n_gt_tp_both,
            "slot1_only": n_gt_tp_slot1_only,
            "s1408_only": n_gt_tp_s1408_only,
        },
        "gt_tp_side": {
            "n_gt_tp_both": n_gt_tp_both,
            "n_gt_tp_slot1_only": n_gt_tp_slot1_only,
            "n_gt_tp_s1408_only": n_gt_tp_s1408_only,
            "n_gt_fn_both": n_gt_fn_both,
            "n_images_slot1_hits_s1408_misses": n_images_slot1_hits_s1408_misses,
            "n_images_s1408_hits_slot1_misses": n_images_s1408_hits_slot1_misses,
        },
        "why": (
            "Pred-pred IoU>0.5 pairing on fold-0. Disagreement is diagnosis, not an "
            "OR license (same rule as 116 / 130 unique). extras-OR stays vetoed. "
            f"k_to_0.50={K_TO_050}. Parent already vetoed s1408 official 0.4384 "
            "(> bar 0.4365) for hidden-exam / forecast public ~0.332. leftovers "
            "stay empty. next_id null."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                "leftover_to_start": False,
                "next_id": None,
                "recommend_start": False,
                "n_matched": n_matched_pairs,
                "n_slot1_only": n_slot1_only,
                "n_s1408_only": n_s1408_only,
                "n_gt_tp_both": n_gt_tp_both,
                "n_gt_tp_slot1_only": n_gt_tp_slot1_only,
                "n_gt_tp_s1408_only": n_gt_tp_s1408_only,
                "do_not_or": True,
                "parent_decision": "NO_LOOK",
                "kaggle_submit_called": False,
            },
            indent=2,
        )
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
