#!/usr/bin/env python3
"""CPU: Slot1-erode vs shift+erode pred-pred disagreement on fold-0.

Hedge diagnosis, not a merge. extras-OR stays vetoed. No train, no submit.
Matching: official IoU>0.5, not Hungarian. Pred-pred is per JPEG stem.
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
SHIFT_CSV = FAST_ARTIFACTS / "yolo11l_seg_shift_erode3x1_fold0_pred.csv"
OUT = ARTIFACTS / "shift_erode_vs_slot1_disagree.json"
IOU_THR = 0.5
SLOT1_OFFICIAL = {
    "file": "artifacts/slot1_erode_foldne0_lock_fold0_official.json",
    "tp": 949,
    "fp": 385,
    "fn": 640,
    "pq": 0.4365,
}
SHIFT_OFFICIAL = {
    "file": "artifacts/yolo11l_seg_shift_erode3x1_fold0_official.json",
    "tp": 969,
    "fp": 370,
    "fn": 620,
    "pq": 0.4429,
}


def main() -> None:
    if not SLOT1_CSV.exists():
        raise SystemExit(f"missing {SLOT1_CSV}")
    if not SHIFT_CSV.exists():
        raise SystemExit(f"missing {SHIFT_CSV}")

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)

    slot1_df = pd.read_csv(SLOT1_CSV)
    shift_df = pd.read_csv(SHIFT_CSV)
    n_slot1_pred = int(len(slot1_df))
    n_shift_pred = int(len(shift_df))
    slot1 = _by_stem(SLOT1_CSV)
    shift = _by_stem(SHIFT_CSV)

    n_matched_pairs = 0
    n_slot1_only = 0
    n_shift_only = 0
    n_slot1_matched_inst = 0
    n_shift_matched_inst = 0

    n_gt_tp_both = 0
    n_gt_tp_slot1_only = 0
    n_gt_tp_shift_only = 0
    n_gt_fn_both = 0
    images_slot1_hits_shift_misses: set[str] = set()
    images_shift_hits_slot1_misses: set[str] = set()

    for file_name in val_files:
        stem = Path(file_name).stem
        s_layers = _layers(slot1.get(stem, []))
        t_layers = _layers(shift.get(stem, []))
        n_s, n_t = int(s_layers.shape[0]), int(t_layers.shape[0])

        if n_s and n_t:
            iou, _ = overlap_matrices(s_layers, t_layers)
            hit = iou > IOU_THR
            n_matched_pairs += int(hit.sum().item())
            s_hit = hit.any(dim=1)
            t_hit = hit.any(dim=0)
            n_slot1_matched_inst += int(s_hit.sum().item())
            n_shift_matched_inst += int(t_hit.sum().item())
            n_slot1_only += int((~s_hit).sum().item())
            n_shift_only += int((~t_hit).sum().item())
        else:
            n_slot1_only += n_s
            n_shift_only += n_t

        for image_id in by_file[file_name]:
            view = views[image_id]
            if not view.filaments:
                continue
            fils = list(view.filaments)
            gt_layers = torch.from_numpy(
                rles_to_layers([encode_rle(rasterize_polygons([fil.polygon])) for fil in fils])
            )
            if n_s:
                si, _ = overlap_matrices(gt_layers, s_layers)
                s_max = si.max(dim=1).values
            else:
                s_max = torch.zeros(len(fils))
            if n_t:
                ti, _ = overlap_matrices(gt_layers, t_layers)
                t_max = ti.max(dim=1).values
            else:
                t_max = torch.zeros(len(fils))
            for i in range(len(fils)):
                s_ok = float(s_max[i]) > IOU_THR
                t_ok = float(t_max[i]) > IOU_THR
                if s_ok and t_ok:
                    n_gt_tp_both += 1
                elif s_ok:
                    n_gt_tp_slot1_only += 1
                    images_slot1_hits_shift_misses.add(stem)
                elif t_ok:
                    n_gt_tp_shift_only += 1
                    images_shift_hits_slot1_misses.add(stem)
                else:
                    n_gt_fn_both += 1

    n_images_slot1_hits_shift_misses = len(images_slot1_hits_shift_misses)
    n_images_shift_hits_slot1_misses = len(images_shift_hits_slot1_misses)
    agree_frac_slot1 = (n_slot1_matched_inst / n_slot1_pred) if n_slot1_pred else None
    agree_frac_shift = (n_shift_matched_inst / n_shift_pred) if n_shift_pred else None
    mostly_agree = bool(
        agree_frac_slot1 is not None
        and agree_frac_shift is not None
        and agree_frac_slot1 >= 0.90
        and agree_frac_shift >= 0.90
    )

    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": "cpu_only_shift_erode_vs_slot1_disagree",
        "skill": "hidden-exam-discipline",
        "diagnosis_not_a_builder": True,
        "cpu_only": True,
        "did_not_train": True,
        "did_not_start_mps": True,
        "did_not_submit": True,
        "kaggle_submit_called": False,
        "do_not_submit": True,
        "do_not_or": True,
        "leftover_to_start": False,
        "next_id": None,
        "recommend_start": False,
        "parent_decision": "NO_LOOK",
        "remaining_looks": 2,
        "pair_stays": [56173362, 56148889],
        "matching": "IoU>0.5, not Hungarian; pred-pred per JPEG stem; GT TPs per annotator-image",
        "slot1_csv": str(SLOT1_CSV),
        "shift_csv": str(SHIFT_CSV),
        "officials": {"slot1": SLOT1_OFFICIAL, "shift_erode": SHIFT_OFFICIAL},
        "n_slot1_pred": n_slot1_pred,
        "n_shift_pred": n_shift_pred,
        "n_matched_IoUgt0.5": n_matched_pairs,
        "n_slot1_matched_inst": n_slot1_matched_inst,
        "n_shift_matched_inst": n_shift_matched_inst,
        "n_slot1_only": n_slot1_only,
        "n_shift_only": n_shift_only,
        "agree_frac_slot1": agree_frac_slot1,
        "agree_frac_shift": agree_frac_shift,
        "mostly_agree": mostly_agree,
        "gt_tp_side": {
            "n_gt_tp_both": n_gt_tp_both,
            "n_gt_tp_slot1_only": n_gt_tp_slot1_only,
            "n_gt_tp_shift_only": n_gt_tp_shift_only,
            "n_gt_fn_both": n_gt_fn_both,
            "n_images_slot1_hits_shift_misses": n_images_slot1_hits_shift_misses,
            "n_images_shift_hits_slot1_misses": n_images_shift_hits_slot1_misses,
        },
        "why": (
            "Pred-pred IoU>0.5 pairing on fold-0. If mostly_agree, shift+erode is a "
            "Slot1-morph clone (strengthens veto). If they disagree a lot they are "
            "different-error files — still do not OR them and still do not upload "
            "shift+erode (year-shift public 0.24). leftovers stay empty. next_id null."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                "n_slot1_pred": n_slot1_pred,
                "n_shift_pred": n_shift_pred,
                "n_matched_IoUgt0.5": n_matched_pairs,
                "n_slot1_only": n_slot1_only,
                "n_shift_only": n_shift_only,
                "leftover_to_start": False,
                "next_id": None,
                "recommend_start": False,
                "mostly_agree": mostly_agree,
                "gt_tp_side": out["gt_tp_side"],
            },
            indent=2,
        )
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
