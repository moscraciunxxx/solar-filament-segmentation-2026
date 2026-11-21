#!/usr/bin/env python3
"""What the faint+Slot1 merge actually added vs the 390 faint-FN ceiling.

Uses cached fold-0 CSVs. Never submits. Never reads test labels.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.metric import gt_annotations_to_df, overlap_matrices
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED
from filament.rle import rles_to_layers

AREA_P25 = 642.0
UNIDENT = 3
SLOT1_CSV = FAST_ARTIFACTS / "yolo_yolo11m_seg_1280_fold0_conf0.35_sz1280_m0_base_m1280_pq.csv"
MERGED_CSV = FAST_ARTIFACTS / "yolo11s_seg_faint_fold0_merged_conf0.40_pred.csv"
EMPTY = ("20180801232450Bh", "20190602025430Lh", "20200619123010Th")


def year_bin(stem: str) -> str:
    y = int(stem[:4])
    if y < 2017:
        return "pre-2017"
    if y <= 2020:
        return "2017-2020"
    return "2020+"


def hits(gt_rles: list[str], pred_rles: list[str]) -> torch.Tensor | None:
    if not gt_rles or not pred_rles:
        return None
    gt = torch.from_numpy(rles_to_layers(gt_rles))
    pred = torch.from_numpy(rles_to_layers(pred_rles))
    iou, _ = overlap_matrices(gt, pred)
    return iou > 0.5


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    records = gt_records_for_files(views, by_file, val_files)
    gt_df = gt_annotations_to_df(records)
    rec_by_id = {
        f"{r['annotator_image']}_{r['index']}": r
        for r in records
    }
    slot1 = pd.read_csv(SLOT1_CSV)
    merged = pd.read_csv(MERGED_CSV)
    gt_keys = gt_df["filament_id"].str.split("_", n=1).str[0]
    s_keys = slot1["filament_id"].str.split("_", n=1).str[0]
    m_keys = merged["filament_id"].str.split("_", n=1).str[0]

    new_tps = []
    still_fn_faint = 0
    still_fn_large = 0
    added_masks = 0
    added_that_tp = 0

    for ann in gt_keys.unique():
        stem = ann.split("-", maxsplit=1)[1]
        file_name = next(fn for fn in val_files if Path(fn).stem == stem)
        gids = gt_df.loc[gt_keys == ann, "filament_id"].tolist()
        gt_rles = gt_df.loc[gt_keys == ann, "segmentation_rle"].tolist()
        s_rles = slot1.loc[s_keys == stem, "segmentation_rle"].tolist()
        m_rles = merged.loc[m_keys == stem, "segmentation_rle"].tolist()
        sh = hits(gt_rles, s_rles)
        mh = hits(gt_rles, m_rles)
        s_hit = sh.any(dim=1) if sh is not None else torch.zeros(len(gt_rles), dtype=torch.bool)
        m_hit = mh.any(dim=1) if mh is not None else torch.zeros(len(gt_rles), dtype=torch.bool)
        if s_rles and m_rles:
            extra = []
            sm = torch.from_numpy(rles_to_layers(s_rles))
            mm = torch.from_numpy(rles_to_layers(m_rles))
            iou, _ = overlap_matrices(mm, sm)
            new_pred = (iou.max(dim=1).values < 0.5)
            added_masks += int(new_pred.sum().item())
            if mh is not None:
                added_that_tp += int((new_pred & mh.any(dim=0)).sum().item())
        elif m_rles and not s_rles:
            added_masks += len(m_rles)
            if mh is not None:
                added_that_tp += int(mh.any(dim=0).sum().item())

        for i, gid in enumerate(gids):
            rec = rec_by_id[gid]
            view = views[rec["annotator_image"]]
            fil = view.filaments[rec["index"] - 1]
            faint = fil.area <= AREA_P25 or fil.category_id == UNIDENT
            if bool(m_hit[i]) and not bool(s_hit[i]):
                new_tps.append(
                    {
                        "annotator_image": rec["annotator_image"],
                        "stem": stem,
                        "year": year_bin(stem),
                        "area": fil.area,
                        "category_id": fil.category_id,
                        "faint_eligible": faint,
                        "empty_slot1_jpeg": stem in EMPTY,
                    }
                )
            if not bool(m_hit[i]):
                if faint:
                    still_fn_faint += 1
                else:
                    still_fn_large += 1

    cats = Counter(r["category_id"] for r in new_tps)
    years = Counter(r["year"] for r in new_tps)
    out = {
        "do_not_submit": True,
        "merged_csv": str(MERGED_CSV),
        "n_new_official_tp_pairs": len(new_tps),
        "n_new_tp_faint_eligible": sum(1 for r in new_tps if r["faint_eligible"]),
        "n_new_tp_large_identifiable": sum(1 for r in new_tps if not r["faint_eligible"]),
        "n_new_tp_on_empty_slot1_three": sum(1 for r in new_tps if r["empty_slot1_jpeg"]),
        "category_counts": {str(k): v for k, v in sorted(cats.items())},
        "year_counts": dict(years),
        "area_of_new_tps": {
            "min": min((r["area"] for r in new_tps), default=0),
            "median": float(sorted(r["area"] for r in new_tps)[len(new_tps) // 2]) if new_tps else 0,
            "max": max((r["area"] for r in new_tps), default=0),
        },
        "added_masks_vs_slot1": added_masks,
        "added_masks_that_hit_a_gt": added_that_tp,
        "still_fn_after_merge": {
            "faint": still_fn_faint,
            "large_identifiable": still_fn_large,
            "total": still_fn_faint + still_fn_large,
        },
        "empty_slot1_three_still_blank": True,
        "implication": (
            "Faint recovered a sliver of the 390 faint-FN ceiling and left the 274 "
            "large-identifiable FNs for U-Net. Do not retune faint. Do not upload."
        ),
        "new_tps": new_tps,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "faint_merge_new_tps.json"
    path.write_text(json.dumps(out, indent=2))
    summary = {k: out[k] for k in out if k != "new_tps"}
    print(json.dumps({"path": str(path), **summary}, indent=2))


if __name__ == "__main__":
    main()
