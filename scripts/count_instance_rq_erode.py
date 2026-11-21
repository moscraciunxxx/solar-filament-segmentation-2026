#!/usr/bin/env python3
"""CPU instance-ID / fragment-merge counts on Slot1 erode + D. No train. No submit.

Method 5 sidecar unlocks only if fragment_fp / merge_fn show a real instance-ID leftover.
Prior audit used old m@1280 Slot1, not public-best erode.
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

from count_instance_rq import _count
from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.metric import gt_annotations_to_df
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED

OUT = ARTIFACTS / "instance_rq_erode_after_d.json"
FAMILIES = {
    "slot1_erode": FAST_ARTIFACTS / "slot1_erode_3x1_fold0_pred.csv",
    "D_yolo11l_seg_erode_gt": FAST_ARTIFACTS / "yolo11l_seg_erode_gt_fold0_conf0.35_pred.csv",
    "B_yolo11l_erode3x1": FAST_ARTIFACTS / "yolo11l_seg_1280_erode3x1_fold0_pred.csv",
}


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": "cpu_only_instance_rq_erode",
        "skill": "hidden-exam-discipline",
        "did_not_train": True,
        "did_not_submit": True,
        "recommend_start": False,
        "next_id": None,
        "fold": 0,
        "prior_gate": "artifacts/method5_sidecar_gate_audit.json",
        "prior_slot1_was": "yolo11m_seg_1280 (not erode)",
        "families": {},
    }
    for name, path in FAMILIES.items():
        if not path.exists():
            out["families"][name] = {"missing": str(path)}
            continue
        pred = pd.read_csv(path)
        row = {"path": str(path), "n_pred_rows": int(len(pred)), **_count(gt_df, pred)}
        out["families"][name] = row
        print(name, json.dumps({k: row[k] for k in (
            "fragment_fp_unofficial",
            "merge_fn_unofficial",
            "fn_true_iou_le_0.3",
            "fn_near_0.3_to_0.5",
        )}), flush=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
