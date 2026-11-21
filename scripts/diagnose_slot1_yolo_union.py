#!/usr/bin/env python3
"""CPU diagnosis: Slot1-erode ∪ existing YOLO pred CSVs.

Not a leftover-to-start. Not a look. extras-OR on public-best is the same
veto class as erode+ctx 0.44033 even if union PQ beats 0.4365.
Never submits. Never trains. Never starts MPS.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.merge import extra_rles, merge_rles
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED
from score_family import year_slices
from score_faint_merge import _stem_frames

PARENT_BAR = 0.4365
SLOT1_CSV = FAST_ARTIFACTS / "slot1_erode_3x1_fold0_pred.csv"
OUT = ARTIFACTS / "slot1_yolo_union_diagnosis.json"

CANDIDATES = [
    ("D_yolo11l_seg_erode_gt", FAST_ARTIFACTS / "yolo11l_seg_erode_gt_fold0_conf0.35_pred.csv"),
    ("C_yolo11m_seg_erode_gt", FAST_ARTIFACTS / "yolo11m_seg_erode_gt_fold0_conf0.35_pred.csv"),
    ("A_yolo11l_seg_1280", FAST_ARTIFACTS / "yolo11l_seg_1280_fold0_conf0.35_pred.csv"),
    ("B_yolo11l_seg_1280_erode3x1", FAST_ARTIFACTS / "yolo11l_seg_1280_erode3x1_fold0_pred.csv"),
]


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
    if not SLOT1_CSV.exists():
        raise SystemExit(f"missing {SLOT1_CSV}")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    slot_frames = _stem_frames(pd.read_csv(SLOT1_CSV), val_files)
    slot_df = pd.concat(slot_frames.values(), ignore_index=True)
    slot_row = _row(score_pq(gt_df, slot_df), int(len(slot_df)))

    unions = []
    for name, path in CANDIDATES:
        if not path.exists():
            unions.append({"id": name, "csv": str(path), "missing": True})
            continue
        other_frames = _stem_frames(pd.read_csv(path), val_files)
        n_extras = 0
        n_added = 0
        frames = {}
        for file_name in tqdm(val_files, desc=f"union {name}"):
            slot_rles = slot_frames[file_name]["segmentation_rle"].astype(str).tolist()
            cand_rles = other_frames[file_name]["segmentation_rle"].astype(str).tolist()
            extras = extra_rles(slot_rles, cand_rles)
            n_extras += len(extras)
            merged, added = merge_rles(slot_rles, extras)
            n_added += added
            frames[file_name] = pred_to_df(Path(file_name).stem, merged)
        pred_df = pd.concat(frames.values(), ignore_index=True)
        br = score_pq(gt_df, pred_df)
        locked = {
            "n_extras": n_extras,
            "n_added": n_added,
            **_row(br, int(len(pred_df))),
        }
        unions.append(
            {
                "id": name,
                "csv": str(path),
                "missing": False,
                "other_n_pred": int(sum(len(other_frames[fn]) for fn in val_files)),
                "union_on_slot1_erode": locked,
                "d_pq_vs_slot1": locked["pq"] - slot_row["pq"],
                "d_tp_vs_slot1": locked["tp"] - slot_row["tp"],
                "d_fp_vs_slot1": locked["fp"] - slot_row["fp"],
                "beats_parent_bar": locked["pq"] > PARENT_BAR,
                "extras_or_on_public_best": True,
                "parent_veto_even_if_beats_bar": True,
                "do_not_submit": True,
            }
        )

    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": "cpu_only_slot1_yolo_union_diagnosis",
        "skill": "hidden-exam-discipline",
        "diagnosis_not_a_builder": True,
        "did_not_train": True,
        "did_not_infer": True,
        "did_not_start_mps": True,
        "did_not_submit": True,
        "kaggle_submit_called": False,
        "do_not_submit": True,
        "recommend_start": False,
        "next_id": None,
        "parent_bar": PARENT_BAR,
        "slot1_erode": slot_row,
        "unions": unions,
        "why": (
            "Union of another YOLO CSV onto Slot1 erode is extras-OR on public-best. "
            "Same veto as erode+ctx 0.44033. Diagnosis only. remaining_looks stay 2."
        ),
        "year_slice_slot1": year_slices(val_files, slot_frames, views, by_file),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"wrote": str(OUT), "slot1": slot_row, "unions": [
        {k: u.get(k) for k in ("id", "d_pq_vs_slot1", "d_tp_vs_slot1", "beats_parent_bar", "missing")}
        for u in unions
    ]}, indent=2))


if __name__ == "__main__":
    main()
