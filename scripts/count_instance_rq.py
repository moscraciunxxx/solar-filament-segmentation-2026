#!/usr/bin/env python3
"""CPU instance-ID / fragment-merge counts from existing pred CSVs. No train. No submit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.metric import gt_annotations_to_df, overlap_matrices
from filament.paths import ARTIFACTS, FAST_CACHE, SEED
from filament.rle import rles_to_layers

SLOT1 = FAST_CACHE / "artifacts" / "yolo_yolo11m_seg_1280_fold0_conf0.35_sz1280_m0_base_m1280_pq.csv"
FAINT = FAST_CACHE / "artifacts" / "yolo11s_seg_faint_fold0_merged_conf0.40_pred.csv"
V1 = FAST_CACHE / "artifacts" / "unet_slot1_merge_fold0_thr0.50_pred.csv"


def _count(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    gt_keys = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_keys = pred_df["filament_id"].str.split("_", n=1).str[0]
    frag_fp = 0
    merge_fn = 0
    gt_multi_pred_03 = 0
    pred_multi_gt_03 = 0
    fn_near = 0
    fn_true = 0
    fp_near = 0
    n_gt = 0
    n_pred_seen = 0
    for annotator_image in gt_keys.unique():
        image_id = annotator_image.split("-", maxsplit=1)[1]
        gt_rles = gt_df.loc[gt_keys == annotator_image, "segmentation_rle"].tolist()
        pred_rles = pred_df.loc[pred_keys == image_id, "segmentation_rle"].tolist()
        n_gt += len(gt_rles)
        n_pred_seen += len(pred_rles)
        if not gt_rles or not pred_rles:
            fn_true += len(gt_rles)
            continue
        iou, _ = overlap_matrices(
            torch.from_numpy(rles_to_layers(gt_rles)),
            torch.from_numpy(rles_to_layers(pred_rles)),
        )
        near = (iou > 0.3) & (iou <= 0.5)
        hit = iou > 0.5
        pred_hits = hit.sum(dim=0)
        gt_hits = hit.sum(dim=1)
        gt_near_n = (iou > 0.3).sum(dim=1)
        pred_near_n = (iou > 0.3).sum(dim=0)
        gt_multi_pred_03 += int((gt_near_n >= 2).sum().item())
        pred_multi_gt_03 += int((pred_near_n >= 2).sum().item())
        # Fragment FP: pred never hits 0.5 but shares a GT with another pred at IoU>0.3
        for j in range(iou.shape[1]):
            if pred_hits[j] > 0:
                continue
            partners = (iou[:, j] > 0.3).nonzero(as_tuple=False).view(-1)
            if any(int(gt_near_n[i]) >= 2 for i in partners.tolist()):
                frag_fp += 1
            mx = float(iou[:, j].max().item()) if iou.shape[0] else 0.0
            if 0.3 < mx <= 0.5:
                fp_near += 1
        # Merge FN: GT never hits 0.5 but shares a pred with another GT at IoU>0.3
        for i in range(iou.shape[0]):
            if gt_hits[i] > 0:
                continue
            mx = float(iou[i].max().item()) if iou.shape[1] else 0.0
            if 0.3 < mx <= 0.5:
                fn_near += 1
            else:
                fn_true += 1
            partners = (iou[i] > 0.3).nonzero(as_tuple=False).view(-1)
            if any(int(pred_near_n[j]) >= 2 for j in partners.tolist()):
                merge_fn += 1
    return {
        "n_gt_pairs": n_gt,
        "n_pred_rows_summed_over_annotator_images": n_pred_seen,
        "gt_with_2plus_preds_iou_gt_0.3": gt_multi_pred_03,
        "pred_with_2plus_gt_iou_gt_0.3": pred_multi_gt_03,
        "fragment_fp_unofficial": frag_fp,
        "merge_fn_unofficial": merge_fn,
        "fn_near_0.3_to_0.5": fn_near,
        "fn_true_iou_le_0.3": fn_true,
        "fp_near_0.3_to_0.5": fp_near,
    }


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    families = {"slot1": SLOT1, "faint_merge": FAINT, "v1_unet_merge": V1}
    out = {"written": "cpu_instance_rq", "do_not_submit": True, "fold": 0}
    out["families"] = {}
    for name, path in families.items():
        if not path.exists():
            out["families"][name] = {"missing": str(path)}
            continue
        pred = pd.read_csv(path)
        out["families"][name] = {"path": str(path), "n_pred_rows": int(len(pred)), **_count(gt_df, pred)}
        print(name, json.dumps(out["families"][name]), flush=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    dest = ARTIFACTS / "instance_rq_from_existing_csvs.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
