#!/usr/bin/env python3
"""CPU diagnosis: existing pred CSVs vs agreed large Slot1 true-miss.

Read-only. Does not train, merge HOLD extras, or submit.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.coco_io import grouped_folds, load_coco, rasterize_polygons, semantic_iou
from filament.metric import overlap_matrices
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED
from filament.rle import encode_rle, rles_to_layers

SLOT1_CSV = FAST_ARTIFACTS / "yolo_yolo11m_seg_1280_fold0_conf0.35_sz1280_m0_base_m1280_pq.csv"
AREA_P25 = 642.0
UNIDENT = 3
IOU_THR = 0.5
NEAR_LO = 0.3
COMPARE = 512

FAMILIES = {
    "faint_merge": "yolo11s_seg_faint_fold0_merged_conf0.40_pred.csv",
    "phys": "yolo11s_seg_phys_fold0_conf0.40_pred.csv",
    "residual": "yolo11s_seg_residual_fold0_conf0.35_pred.csv",
    "tiles": "yolo11s_seg_tiles_fold0_conf0.40_pred.csv",
    "all_annotator": "yolo11s_seg_all_fold0_conf0.40_pred.csv",
    "year_shift_035": "yolo_yolo11s_seg_shift_fold0_conf0.35_sz1024_m0_base_shift_pq.csv",
    "patch_merge": "unet_patch_slot1_merge_fold0_thr0.60_pred.csv",
    "v1_unet_merge_hose": "unet_slot1_merge_fold0_thr0.50_pred.csv",
}


def other_rater_hit(fil, other_views) -> bool:
    scale = COMPARE / 2048.0
    mine = rasterize_polygons([fil.polygon * scale], size=COMPARE)
    for view in other_views:
        for other in view.filaments:
            other_m = rasterize_polygons([other.polygon * scale], size=COMPARE)
            if semantic_iou(mine, other_m) > IOU_THR:
                return True
    return False


def pred_by_stem(csv_path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(csv_path)
    out: dict[str, list[str]] = defaultdict(list)
    for fid, rle in zip(df["filament_id"], df["segmentation_rle"]):
        out[str(fid).rsplit("_", 1)[0]].append(str(rle))
    return out


def covers(gt_rle: str, pred_rles: list[str]) -> bool:
    if not pred_rles:
        return False
    gt_l = torch.from_numpy(rles_to_layers([gt_rle]))
    pr_l = torch.from_numpy(rles_to_layers(pred_rles))
    return overlap_matrices(gt_l, pr_l)[0].max().item() > IOU_THR


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    slot = pred_by_stem(SLOT1_CSV)

    agreed = []
    for file_name in val_files:
        stem = Path(file_name).stem
        n_raters = len(by_file[file_name])
        pred_rles = slot.get(stem, [])
        pred_layers = (
            torch.from_numpy(rles_to_layers(pred_rles))
            if pred_rles
            else torch.zeros((0, 2048, 2048), dtype=torch.float32)
        )
        for image_id in by_file[file_name]:
            view = views[image_id]
            if not view.filaments:
                continue
            fils = list(view.filaments)
            gt_rles = [encode_rle(rasterize_polygons([fil.polygon])) for fil in fils]
            gt_layers = torch.from_numpy(rles_to_layers(gt_rles))
            row_max = (
                [0.0] * len(fils)
                if pred_layers.shape[0] == 0
                else overlap_matrices(gt_layers, pred_layers)[0].max(dim=1).values.tolist()
            )
            others = [views[oid] for oid in by_file[file_name] if oid != image_id]
            for fil, mx in zip(fils, row_max):
                faint = float(fil.area) <= AREA_P25 or int(fil.category_id) == UNIDENT
                if mx > NEAR_LO or faint:
                    continue
                if n_raters < 2 or not other_rater_hit(fil, others):
                    continue
                agreed.append((file_name, encode_rle(rasterize_polygons([fil.polygon]))))

    families = {}
    for name, fname in FAMILIES.items():
        path = FAST_ARTIFACTS / fname
        if not path.exists():
            families[name] = {"exists": False}
            continue
        by = pred_by_stem(path)
        n = sum(covers(gt, by.get(Path(fn).stem, [])) for fn, gt in agreed)
        families[name] = {
            "exists": True,
            "path": str(path),
            "n_cover_iou50": n,
            "of": len(agreed),
            "already_closed_or_hold": True,
        }

    out = {
        "do_not_submit": True,
        "n_agreed_large_true_miss": len(agreed),
        "resnet_ch3_union_keep": 21,
        "never_proposed_by_resnet_or_ch3_cc": 20,
        "families": families,
        "note": "Cover counts are diagnosis. Closed/HOLD families are not restarted. Union of HOLD extras stays forbidden.",
        "recommend_start": False,
        "auto_start": False,
    }
    dest = ARTIFACTS / "existing_csv_cover_agreed_true_miss.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
