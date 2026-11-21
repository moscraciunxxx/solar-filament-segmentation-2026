#!/usr/bin/env python3
"""CPU: Slot1-erode vs thick s@1408 erode3x1 pred-pred disagreement on fold-0.

Hedge diagnosis, not a merge. extras-OR stays vetoed even if unique_either >= 220.
No train, no submit, no MPS, no leftover start.
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

SLOT1_CSV = FAST_ARTIFACTS / "slot1_erode_3x1_fold0_pred.csv"
THICK_CSV = FAST_ARTIFACTS / "yolo11s_seg_1408_erode3x1_fold0_pred.csv"
OUT = ARTIFACTS / "s1408_thick_erode_vs_slot1_disagree.json"
IOU_THR = 0.5
K_TO_050 = 220
SLOT1_OFFICIAL = {
    "file": "artifacts/slot1_erode_foldne0_lock_fold0_official.json",
    "tp": 949,
    "fp": 385,
    "fn": 640,
    "pq": 0.4365,
}
THICK_OFFICIAL = {
    "file": "artifacts/yolo11s_seg_1408_erode3x1_fold0_official.json",
    "tp": 950,
    "fp": 387,
    "fn": 639,
    "pq": 0.4388,
}


def main() -> None:
    if not SLOT1_CSV.exists():
        raise SystemExit(f"missing {SLOT1_CSV}")
    if not THICK_CSV.exists():
        raise SystemExit(f"missing {THICK_CSV}")

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)

    slot1_df = pd.read_csv(SLOT1_CSV)
    thick_df = pd.read_csv(THICK_CSV)
    n_slot1_pred = int(len(slot1_df))
    n_thick_pred = int(len(thick_df))
    slot1 = _by_stem(SLOT1_CSV)
    thick = _by_stem(THICK_CSV)

    n_matched_pairs = 0
    n_slot1_only = 0
    n_thick_only = 0
    n_slot1_matched_inst = 0
    n_thick_matched_inst = 0

    n_gt_tp_both = 0
    n_gt_tp_slot1_only = 0
    n_gt_tp_thick_only = 0
    n_gt_fn_both = 0
    images_slot1_hits_thick_misses: set[str] = set()
    images_thick_hits_slot1_misses: set[str] = set()

    for file_name in val_files:
        stem = Path(file_name).stem
        s1_layers = _layers(slot1.get(stem, []))
        t_layers = _layers(thick.get(stem, []))
        n_s1, n_t = int(s1_layers.shape[0]), int(t_layers.shape[0])

        if n_s1 and n_t:
            iou, _ = overlap_matrices(s1_layers, t_layers)
            hit = iou > IOU_THR
            n_matched_pairs += int(hit.sum().item())
            s1_hit = hit.any(dim=1)
            t_hit = hit.any(dim=0)
            n_slot1_matched_inst += int(s1_hit.sum().item())
            n_thick_matched_inst += int(t_hit.sum().item())
            n_slot1_only += int((~s1_hit).sum().item())
            n_thick_only += int((~t_hit).sum().item())
        else:
            n_slot1_only += n_s1
            n_thick_only += n_t

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
            if n_t:
                ti, _ = overlap_matrices(gt_layers, t_layers)
                t_max = ti.max(dim=1).values
            else:
                t_max = torch.zeros(len(fils))
            for i in range(len(fils)):
                s1_ok = float(s1_max[i]) > IOU_THR
                t_ok = float(t_max[i]) > IOU_THR
                if s1_ok and t_ok:
                    n_gt_tp_both += 1
                elif s1_ok:
                    n_gt_tp_slot1_only += 1
                    images_slot1_hits_thick_misses.add(stem)
                elif t_ok:
                    n_gt_tp_thick_only += 1
                    images_thick_hits_slot1_misses.add(stem)
                else:
                    n_gt_fn_both += 1

    n_gt_tp_unique_either = n_gt_tp_slot1_only + n_gt_tp_thick_only
    n_images_slot1_hits_thick_misses = len(images_slot1_hits_thick_misses)
    n_images_thick_hits_slot1_misses = len(images_thick_hits_slot1_misses)
    agree_frac_slot1 = (n_slot1_matched_inst / n_slot1_pred) if n_slot1_pred else None
    agree_frac_thick = (n_thick_matched_inst / n_thick_pred) if n_thick_pred else None
    mostly_agree = bool(
        agree_frac_slot1 is not None
        and agree_frac_thick is not None
        and agree_frac_slot1 >= 0.90
        and agree_frac_thick >= 0.90
    )

    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": "cpu_only_s1408_thick_erode_vs_slot1_disagree",
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
        "n_gt_tp_unique_either": n_gt_tp_unique_either,
        "n_gt_tp_unique_either_lt_k": n_gt_tp_unique_either < K_TO_050,
        "leftover_to_start": False,
        "next_id": None,
        "recommend_start": False,
        "parent_decision": "NO_LOOK",
        "remaining_looks": 2,
        "pair_stays": [56173362, 56148889],
        "matching": "IoU>0.5, not Hungarian; pred-pred per JPEG stem; GT TPs per annotator-image",
        "slot1_csv": str(SLOT1_CSV),
        "thick_csv": str(THICK_CSV),
        "officials": {"slot1": SLOT1_OFFICIAL, "thick_s1408_erode": THICK_OFFICIAL},
        "n_slot1_pred": n_slot1_pred,
        "n_thick_pred": n_thick_pred,
        "n_matched": n_matched_pairs,
        "n_matched_IoUgt0.5": n_matched_pairs,
        "n_slot1_matched_inst": n_slot1_matched_inst,
        "n_thick_matched_inst": n_thick_matched_inst,
        "n_slot1_only": n_slot1_only,
        "n_thick_only": n_thick_only,
        "agree_frac_slot1": agree_frac_slot1,
        "agree_frac_thick": agree_frac_thick,
        "mostly_agree": mostly_agree,
        "gt_tp": {
            "both": n_gt_tp_both,
            "slot1_only": n_gt_tp_slot1_only,
            "thick_only": n_gt_tp_thick_only,
            "unique_either": n_gt_tp_unique_either,
        },
        "gt_tp_side": {
            "n_gt_tp_both": n_gt_tp_both,
            "n_gt_tp_slot1_only": n_gt_tp_slot1_only,
            "n_gt_tp_thick_only": n_gt_tp_thick_only,
            "n_gt_tp_unique_either": n_gt_tp_unique_either,
            "n_gt_fn_both": n_gt_fn_both,
            "n_images_slot1_hits_thick_misses": n_images_slot1_hits_thick_misses,
            "n_images_thick_hits_slot1_misses": n_images_thick_hits_slot1_misses,
        },
        "why": (
            "Pred-pred IoU>0.5 pairing on fold-0. Disagreement is diagnosis, not an "
            "OR license (same rule as 116 / 130 unique). extras-OR stays vetoed even "
            f"if unique_either >= {K_TO_050}. k_to_0.50={K_TO_050}. Parent NO_LOOK; "
            "pair stays 56173362 / 56148889. leftovers stay empty. next_id null."
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
                "n_thick_only": n_thick_only,
                "n_gt_tp_unique_either": n_gt_tp_unique_either,
                "n_gt_tp_both": n_gt_tp_both,
                "n_gt_tp_slot1_only": n_gt_tp_slot1_only,
                "n_gt_tp_thick_only": n_gt_tp_thick_only,
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
