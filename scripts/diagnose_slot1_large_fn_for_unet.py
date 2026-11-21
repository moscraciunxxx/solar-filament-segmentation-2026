#!/usr/bin/env python3
"""Large-identifiable Slot-1 fold-0 FNs that faint cannot label (U-Net v1 pool).

CPU only. Train JSON + cached Slot-1 pred CSV. No test labels. No train. No MPS YOLO.
Matching: IoU > 0.5, not Hungarian. Official PQ scores every annotator-image
on the 142 fold-0 JPEGs (not medoid-only).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco, rasterize_polygons
from filament.metric import overlap_matrices
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED
from filament.rle import encode_rle, rles_to_layers

SLOT1_CSV = FAST_ARTIFACTS / "yolo_yolo11m_seg_1280_fold0_conf0.35_sz1280_m0_base_m1280_pq.csv"
AREA_P25 = 642.0
UNIDENT = 3
IOU_THR = 0.5
NEAR_LO = 0.3
YEAR_KEYS = ("pre-2017", "2017-2020", "2020+")


def jpeg_year(file_name: str) -> int:
    return int(Path(file_name).stem[:4])


def year_partition(year: int) -> str:
    if year < 2017:
        return "pre-2017"
    if year <= 2020:
        return "2017-2020"
    return "2020+"


def is_large_identifiable(fil) -> bool:
    return float(fil.area) > AREA_P25 and int(fil.category_id) != UNIDENT


def bbox_elongation(bbox: tuple[float, float, float, float]) -> float:
    _, _, w, h = bbox
    w = float(w)
    h = float(h)
    short = min(w, h)
    long = max(w, h)
    if short <= 0:
        return float("nan")
    return long / short


def quartiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    qs = np.quantile(arr, [0.25, 0.5, 0.75])
    return {
        "q1": float(qs[0]),
        "q2": float(qs[1]),
        "q3": float(qs[2]),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def main() -> None:
    if not SLOT1_CSV.exists():
        raise SystemExit(f"missing Slot-1 pred csv: {SLOT1_CSV}")
    pred_df = pd.read_csv(SLOT1_CSV)
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)

    pred_by_stem: dict[str, list[str]] = defaultdict(list)
    for fid, rle in zip(pred_df["filament_id"], pred_df["segmentation_rle"]):
        stem = str(fid).rsplit("_", 1)[0]
        pred_by_stem[stem].append(str(rle))

    areas: list[float] = []
    elongations: list[float] = []
    max_ious: list[float] = []
    year_counts = {k: 0 for k in YEAR_KEYS}
    n_fn = 0
    n_large_fn = 0
    n_gt = 0
    n_tp_pairs = 0
    n_fp = 0

    for file_name in val_files:
        stem = Path(file_name).stem
        year = jpeg_year(file_name)
        bucket = year_partition(year)
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
            n_gt += len(gt_rles)
            gt_layers = torch.from_numpy(rles_to_layers(gt_rles))
            if pred_layers.shape[0] == 0:
                row_max = torch.zeros(len(fils), dtype=torch.float32)
                n_fp += 0
            else:
                iou, _ = overlap_matrices(gt_layers, pred_layers)
                hit = iou > IOU_THR
                n_tp_pairs += int(hit.sum().item())
                n_fp += int((hit.sum(dim=0) == 0).sum().item())
                row_max = iou.max(dim=1).values
            for fil, mx in zip(fils, row_max.tolist()):
                if mx > IOU_THR:
                    continue
                n_fn += 1
                if not is_large_identifiable(fil):
                    continue
                n_large_fn += 1
                areas.append(float(fil.area))
                elongations.append(bbox_elongation(fil.bbox))
                max_ious.append(float(mx))
                year_counts[bucket] += 1

    elong_arr = np.asarray(elongations, dtype=np.float64)
    elong_ok = elong_arr[np.isfinite(elong_arr)]
    near = sum(1 for v in max_ious if NEAR_LO < v <= IOU_THR)
    true_miss = sum(1 for v in max_ious if v <= NEAR_LO)
    if near + true_miss != n_large_fn:
        raise SystemExit(
            f"IoU bins do not partition large FNs: near={near} true_miss={true_miss} n={n_large_fn}"
        )

    implication = (
        "U-Net v1 gray full-disk is the family that can add these; faint merge cannot"
    )
    out = {
        "count": n_large_fn,
        "area_quartiles": quartiles(areas),
        "mean_bbox_elongation": float(elong_ok.mean()) if len(elong_ok) else None,
        "n_bbox_elongation_undefined": int(np.isnan(elong_arr).sum()) if len(elong_arr) else 0,
        "year_buckets": year_counts,
        "year_bin_note": (
            "pre-2017=year<2017; 2017-2020=2017..2020 inclusive; 2020+=year>2020 (partition)."
        ),
        "iou_vs_slot1": {
            "near_miss_iou_in_0.3_exclusive_to_0.5_inclusive": near,
            "true_miss_iou_le_0.3": true_miss,
            "mean_max_iou": float(np.mean(max_ious)) if max_ious else 0.0,
            "median_max_iou": float(np.median(max_ious)) if max_ious else 0.0,
        },
        "implication": implication,
        "large_identifiable_rule": "area>train_medoid_p25=642.0 AND category_id != 3 Unidentifiable",
        "area_p25": AREA_P25,
        "unidentifiable_category_id": UNIDENT,
        "matching": "IoU>0.5, not Hungarian; every annotator-image on fold-0 JPEGs",
        "fn_total_measured": n_fn,
        "n_val_jpegs": len(val_files),
        "n_gt": n_gt,
        "n_pred": int(len(pred_df)),
        "tp_official_pairs": n_tp_pairs,
        "fp_official": n_fp,
        "slot1_csv": str(SLOT1_CSV),
        "slot1_official": {"pq": 0.410, "tp": 925, "fp": 443, "fn": 664, "mean_tp_iou": 0.655},
        "why_faint_cannot": (
            "Faint specialist keep-rule is area<=642 OR category_id==3. These 274 FNs "
            "are the complement: large and identifiable, so they are never in the faint label set."
        ),
        "why_unet_v1": (
            "U-Net v1 gray-as-RGB full-disk scores every pixel and can propose large bodies "
            "YOLO never boxed. Crop-refiner and faint merge cannot invent those proposals."
        ),
        "do_not_submit": True,
        "test_labels_used": False,
        "trained": False,
        "used_mps_yolo": False,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "slot1_large_fn_for_unet.json"
    path.write_text(json.dumps(out, indent=2) + "\n")

    aq = out["area_quartiles"]
    lines = [
        f"Slot1 fold0 large-identifiable FNs: count={n_large_fn} (expect 274 of FN {n_fn})",
        f"Area quartiles: Q1={aq['q1']:.1f} Q2={aq['q2']:.1f} Q3={aq['q3']:.1f} mean={aq['mean']:.1f}",
        f"Mean bbox elongation max(w,h)/min(w,h)={out['mean_bbox_elongation']:.3f}",
        f"Year buckets: pre-2017={year_counts['pre-2017']} 2017-2020={year_counts['2017-2020']} 2020+={year_counts['2020+']}",
        f"Vs Slot-1 max IoU: near-miss (0.3,0.5]={near}  true-miss IoU<=0.3={true_miss}",
        f"Mean/median max IoU among these FNs: {out['iou_vs_slot1']['mean_max_iou']:.3f} / {out['iou_vs_slot1']['median_max_iou']:.3f}",
        f"Official check: TP_pairs={n_tp_pairs} FP={n_fp} FN={n_fn} (expect 925/443/664)",
        "Faint keep-rule cannot label area>642 AND cat!=3, so faint merge cannot convert this pool.",
        "U-Net v1 gray full-disk is the family that can propose these large identifiable FNs.",
        "Near-misses need better overlap; true misses need new proposals YOLO never fired.",
        "do_not_submit=true; no test labels; no train; no MPS YOLO.",
        f"Wrote {path}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
