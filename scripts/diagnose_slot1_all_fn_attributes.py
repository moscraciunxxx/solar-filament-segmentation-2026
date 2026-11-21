#!/usr/bin/env python3
"""Attribute all Slot-1 fold-0 FNs from train JSON + cached pred CSV. CPU only.

No new infer. No test labels. No submit.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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


def jpeg_year(file_name: str) -> int:
    return int(Path(file_name).stem[:4])


def year_partition(year: int) -> str:
    if year < 2017:
        return "pre-2017"
    if year <= 2020:
        return "2017-2020"
    return "2020+"


def other_rater_hit(fil, other_views) -> bool:
    scale = COMPARE / 2048.0
    mine = rasterize_polygons([fil.polygon * scale], size=COMPARE)
    for view in other_views:
        for other in view.filaments:
            other_m = rasterize_polygons([other.polygon * scale], size=COMPARE)
            if semantic_iou(mine, other_m) > IOU_THR:
                return True
    return False


def main() -> None:
    pred_df = pd.read_csv(SLOT1_CSV)
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    pred_by_stem: dict[str, list[str]] = defaultdict(list)
    for fid, rle in zip(pred_df["filament_id"], pred_df["segmentation_rle"]):
        pred_by_stem[str(fid).rsplit("_", 1)[0]].append(str(rle))

    rows = []
    for file_name in val_files:
        stem = Path(file_name).stem
        n_raters = len(by_file[file_name])
        pred_rles = pred_by_stem.get(stem, [])
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
            if pred_layers.shape[0] == 0:
                row_max = [0.0] * len(fils)
            else:
                iou, _ = overlap_matrices(gt_layers, pred_layers)
                row_max = iou.max(dim=1).values.tolist()
            others = [views[oid] for oid in by_file[file_name] if oid != image_id]
            for fil, mx in zip(fils, row_max):
                if mx > IOU_THR:
                    continue
                faint = float(fil.area) <= AREA_P25 or int(fil.category_id) == UNIDENT
                agreed = other_rater_hit(fil, others) if n_raters >= 2 else None
                rows.append(
                    {
                        "year": year_partition(jpeg_year(file_name)),
                        "area": float(fil.area),
                        "category_id": int(fil.category_id),
                        "faint_or_unident": faint,
                        "max_iou_slot1": float(mx),
                        "true_miss": float(mx) <= NEAR_LO,
                        "n_raters": n_raters,
                        "other_rater_iou50": agreed,
                    }
                )

    n = len(rows)
    true = [r for r in rows if r["true_miss"]]
    near = [r for r in rows if not r["true_miss"]]

    def bucket(subset):
        cat = Counter(r["category_id"] for r in subset)
        year = Counter(r["year"] for r in subset)
        raters = Counter(r["n_raters"] for r in subset)
        agreed = Counter(
            "sole_rater" if r["other_rater_iou50"] is None else ("agreed" if r["other_rater_iou50"] else "minority")
            for r in subset
        )
        areas = [r["area"] for r in subset]
        return {
            "n": len(subset),
            "faint_or_unident": sum(1 for r in subset if r["faint_or_unident"]),
            "large_identifiable": sum(1 for r in subset if not r["faint_or_unident"]),
            "category": {str(k): v for k, v in sorted(cat.items())},
            "year": dict(year),
            "n_raters": {str(k): v for k, v in sorted(raters.items())},
            "cross_rater": dict(agreed),
            "area_median": float(np.median(areas)) if areas else None,
        }

    out = {
        "do_not_submit": True,
        "n_fn": n,
        "all": bucket(rows),
        "true_miss_iou_le_0.3": bucket(true),
        "near_miss_0.3_to_0.5": bucket(near),
        "named_leftover_hint": {
            "if_mostly_faint": "yolo11s_seg_faint already HOLD 0.412",
            "if_mostly_minority": "does not overturn Method 1 SKIP-if-flat; recovering minority may not transfer",
            "if_mostly_agreed_large": "real detector hole; no open named leftover after inventory",
        },
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    dest = ARTIFACTS / "slot1_fn_attributes.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
