#!/usr/bin/env python3
"""How many official Slot-1 fold-0 FNs are faint (area<=p25 or Unidentifiable).

CPU only. Train JSON + cached Slot-1 pred CSV. No test labels. No train.
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
BASE_TP = 925
BASE_FP = 443
BASE_FN = 664
BASE_MEAN_IOU = 0.655
NEW_MEAN_IOU = 0.62
TARGET_PQ = 0.43


def jpeg_year(file_name: str) -> int:
    return int(Path(file_name).stem[:4])


def year_partition(year: int) -> str:
    """Disjoint bins so FN-faint counts sum to the pool."""
    if year < 2017:
        return "pre-2017"
    if year <= 2020:
        return "2017-2020"
    return "2020+"


def climb_pq(n_new_tp: int, extra_fp: int) -> float:
    sum_iou = BASE_TP * BASE_MEAN_IOU + n_new_tp * NEW_MEAN_IOU
    denom = (BASE_TP + n_new_tp) + 0.5 * (BASE_FP + extra_fp) + 0.5 * (BASE_FN - n_new_tp)
    return float(sum_iou / denom) if denom else 0.0


def tps_needed(extra_fp_ratio: float) -> int | None:
    for n in range(0, BASE_FN + 1):
        extra_fp = int(round(n * extra_fp_ratio))
        if climb_pq(n, extra_fp) >= TARGET_PQ:
            return n
    return None


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

    n_gt = 0
    n_fn = 0
    n_tp_gt = 0
    n_tp_pairs = 0
    n_fp = 0
    faint = {"fn": 0, "tp": 0, "gt": 0}
    large = {"fn": 0, "tp": 0, "gt": 0}
    year_keys = ("pre-2017", "2017-2020", "2020+")
    year_fn = {k: {"fn": 0, "fn_faint": 0, "fn_large_identifiable": 0, "gt": 0} for k in year_keys}
    year_fn_faint_ge2020 = 0
    n_ann = 0
    n_ann_with_gt = 0
    tp_ious: list[float] = []

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
            n_ann += 1
            view = views[image_id]
            if not view.filaments:
                continue
            n_ann_with_gt += 1
            gt_rles = []
            tags = []
            for fil in view.filaments:
                mask = rasterize_polygons([fil.polygon])
                gt_rles.append(encode_rle(mask))
                is_faint = float(fil.area) <= AREA_P25 or int(fil.category_id) == UNIDENT
                tags.append(is_faint)
            n_gt += len(gt_rles)
            gt_layers = torch.from_numpy(rles_to_layers(gt_rles))
            if pred_layers.shape[0] == 0:
                row_hit = torch.zeros(len(gt_rles), dtype=torch.bool)
                n_fp += 0
            else:
                iou, _ = overlap_matrices(gt_layers, pred_layers)
                hit = iou > IOU_THR
                n_tp_pairs += int(hit.sum().item())
                tp_ious.extend(iou[hit].tolist())
                n_fp += int((hit.sum(dim=0) == 0).sum().item())
                row_hit = hit.any(dim=1)
            for is_faint, hit_gt in zip(tags, row_hit.tolist()):
                key = "faint" if is_faint else "large"
                {"faint": faint, "large": large}[key]["gt"] += 1
                year_fn[bucket]["gt"] += 1
                if hit_gt:
                    n_tp_gt += 1
                    {"faint": faint, "large": large}[key]["tp"] += 1
                else:
                    n_fn += 1
                    {"faint": faint, "large": large}[key]["fn"] += 1
                    year_fn[bucket]["fn"] += 1
                    year_fn[bucket]["fn_faint" if is_faint else "fn_large_identifiable"] += 1
                    if is_faint and year >= 2020:
                        year_fn_faint_ge2020 += 1

    need0 = tps_needed(0.0)
    need1 = tps_needed(1.0)
    need2 = tps_needed(2.0)
    pool = int(faint["fn"])
    can0 = need0 is not None and pool >= need0
    can1 = need1 is not None and pool >= need1
    can2 = need2 is not None and pool >= need2
    can_hit = bool(can0)
    if can_hit:
        why = (
            f"Faint FN pool is {pool}. Naive climb needs {need0} new TPs at extra FP=0 "
            f"(and {need1} at extra FP=N). Both are inside the pool. extra FP=2N cannot "
            f"reach PQ 0.43 for any N (denominator grows faster than 0.62·N). "
            f"Large-identifiable FNs ({large['fn']}) are outside this specialist label set."
        )
        answer = "yes"
    else:
        why = (
            f"Faint FN pool is {pool}, but even extra FP=0 needs {need0} new TPs to hit 0.43. "
            "The pool is the merge ceiling and it is too small."
        )
        answer = "no"

    sum_iou_approx = BASE_TP * BASE_MEAN_IOU
    mean_pair_iou = float(np.mean(tp_ious)) if tp_ious else 0.0
    out = {
        "n_val_jpegs": len(val_files),
        "n_annotator_images": n_ann,
        "n_annotator_images_with_gt": n_ann_with_gt,
        "n_gt": n_gt,
        "n_pred": int(len(pred_df)),
        "fn_total": n_fn,
        "fn_faint": int(faint["fn"]),
        "fn_large_identifiable": int(large["fn"]),
        "tp_faint": int(faint["tp"]),
        "tp_large": int(large["tp"]),
        "tp_gt_unique": n_tp_gt,
        "tp_official_pairs": n_tp_pairs,
        "fp_official": n_fp,
        "mean_tp_iou_measured": mean_pair_iou,
        "faint_gt": int(faint["gt"]),
        "large_identifiable_gt": int(large["gt"]),
        "per_year_fn_faint": {k: year_fn[k]["fn_faint"] for k in year_keys},
        "per_year": year_fn,
        "per_year_fn_faint_2020plus_inclusive": year_fn_faint_ge2020,
        "year_bin_note": "pre-2017=year<2017; 2017-2020=2017..2020 inclusive; 2020+=year>2020 (partition). 2020plus_inclusive uses year>=2020.",
        "faint_rule": "area<=train_medoid_p25=642.0 OR category_id==3 Unidentifiable",
        "area_p25": AREA_P25,
        "unidentifiable_category_id": UNIDENT,
        "matching": "IoU>0.5, not Hungarian; every annotator-image on fold-0 JPEGs",
        "slot1_csv": str(SLOT1_CSV),
        "slot1_official": {
            "pq": 0.410,
            "tp": BASE_TP,
            "fp": BASE_FP,
            "fn": BASE_FN,
            "mean_tp_iou": BASE_MEAN_IOU,
        },
        "naive_climb_math": {
            "slot1_sum_iou_approx": sum_iou_approx,
            "formula": "PQ=(925*0.655 + 0.62*N) / ((925+N) + 0.5*(443+extra_FP) + 0.5*(664-N))",
            "target_pq": TARGET_PQ,
            "mean_new_tp_iou": NEW_MEAN_IOU,
            "new_tps_needed_extra_fp_0": need0,
            "new_tps_needed_extra_fp_eq_N": need1,
            "new_tps_needed_extra_fp_eq_2N": need2,
            "pq_if_convert_all_faint_fn_extra_fp_0": climb_pq(pool, 0),
            "pq_if_convert_all_faint_fn_extra_fp_N": climb_pq(pool, pool),
            "pq_if_convert_all_faint_fn_extra_fp_2N": climb_pq(pool, 2 * pool),
            "faint_fn_pool": pool,
            "can_hit_043_extra_fp_0": can0,
            "can_hit_043_extra_fp_eq_N": can1,
            "can_hit_043_extra_fp_eq_2N": can2,
        },
        "decision": {
            "can_faint_merge_theoretically_hit_043": can_hit,
            "answer": answer,
            "why": why,
        },
        "do_not_submit": True,
        "test_labels_used": False,
        "trained": False,
        "used_mps_yolo": False,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "slot1_faint_fn_ceiling.json"
    path.write_text(json.dumps(out, indent=2) + "\n")

    lines = [
        f"Slot1 fold0: {len(val_files)} JPEGs, {n_ann} annotator-images, {n_gt} GT, {len(pred_df)} preds",
        f"Official match IoU>0.5: TP_pairs={n_tp_pairs} TP_gt={n_tp_gt} FP={n_fp} FN={n_fn} (expect 925/443/664)",
        f"Faint rule: area<={AREA_P25} or cat={UNIDENT}. GT faint={faint['gt']} large={large['gt']}",
        f"FN total={n_fn}  FN faint={faint['fn']}  FN large-identifiable={large['fn']}",
        f"TP faint={faint['tp']}  TP large={large['tp']}  mean pair IoU={mean_pair_iou:.4f}",
        f"Year FN faint: pre-2017={year_fn['pre-2017']['fn_faint']}  2017-2020={year_fn['2017-2020']['fn_faint']}  2020+={year_fn['2020+']['fn_faint']}",
        f"Naive sum_IoU≈{sum_iou_approx:.3f}; target PQ {TARGET_PQ} at new-TP IoU {NEW_MEAN_IOU}",
        f"New TPs needed: extraFP=0 → {need0}; extraFP=N → {need1}; extraFP=2N → {need2}",
        f"Faint FN pool={pool}. Hit 0.43? FP0={can0} FP=N={can1} FP=2N={can2}",
        f"If convert all faint FN: PQ(FP0)={climb_pq(pool, 0):.4f} PQ(FP=N)={climb_pq(pool, pool):.4f} PQ(FP=2N)={climb_pq(pool, 2 * pool):.4f}",
        f"Decision: {answer.upper()} — faint merge can theoretically hit 0.43 from this FN pool" if can_hit else f"Decision: {answer.upper()} — pool cannot theoretically hit 0.43",
        why,
        "Large-identifiable FNs are out of the specialist keep-rule and do not count toward this ceiling.",
        "do_not_submit=true; no test labels; no train; no MPS YOLO.",
        f"Wrote {path}",
    ]
    print("\n".join(lines[:15]))


if __name__ == "__main__":
    main()
