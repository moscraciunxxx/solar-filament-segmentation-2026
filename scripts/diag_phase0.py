#!/usr/bin/env python3
"""Phase 0 diagnosis of the 3-seed champion on fold 0. No training. No test labels."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco, medoid_view_id, rasterize_polygons
from filament.metric import gt_annotations_to_df, overlap_matrices, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, IMAGE_SIZE, SEED, TEST_IMG_DIR, TRAIN_IMG_DIR
from filament.rle import encode_rle, rles_to_layers

CHAMPION = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_ensemble3_tta_fold0_pred.csv"
OUT = ARTIFACTS / "phase0_diagnosis.json"
SQ_HYP = 0.85


def _pq(sq: float, tp: int, fp: int, fn: int) -> float:
    denom = tp + 0.5 * fp + 0.5 * fn
    return (sq * tp / denom) if denom else 0.0


def _elongation(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return 1.0
    cov = np.cov(np.vstack([xs, ys]).astype(np.float64))
    eig = np.linalg.eigvalsh(cov)
    return float(math.sqrt(max(eig[-1], 0.0) / max(eig[0], 1e-6)))


def _contrast(gray: np.ndarray, mask: np.ndarray) -> float:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    ring = (cv2.dilate(mask, kernel) > 0) & (mask == 0)
    if int(ring.sum()) < 20 or int(mask.sum()) < 1:
        return 0.0
    return float(gray[ring].mean() - gray[mask > 0].mean())


def _disk(gray: np.ndarray) -> tuple[float, float]:
    small = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    h, w = small.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    rho = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = int(min(cy, cx))
    r_int = np.clip(rho.astype(np.int32), 0, r_max - 1)
    sums = np.bincount(r_int.ravel(), weights=small.ravel().astype(np.float64), minlength=r_max)
    counts = np.maximum(np.bincount(r_int.ravel(), minlength=r_max).astype(np.float64), 1.0)
    profile = sums / counts
    outer = int(r_max * 0.55)
    drop = np.diff(profile)
    limb = outer + int(np.argmin(drop[outer:])) if outer < len(drop) else r_max - 1
    limb = max(int(limb), 1)
    scale = gray.shape[0] / 512.0
    bright = small > max(float(np.percentile(small, 55)), 12)
    ys, xs = np.nonzero(bright & (rho <= limb))
    if len(xs) == 0:
        offset = 0.0
    else:
        offset = math.hypot(ys.mean() - cy, xs.mean() - cx) * scale
    inside = rho <= limb
    vals = small[inside].astype(np.float32)
    return limb * scale, offset, vals


def _site_year(name: str) -> tuple[str, int]:
    stem = Path(name).stem
    digits = stem[:14]
    site = stem[14:]
    year = int(digits[:4]) if digits[:4].isdigit() else 0
    return site, year


def _image_table(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        limb, offset, vals = _disk(gray)
        site, year = _site_year(path.name)
        rows.append(
            {
                "file": path.name,
                "site": site,
                "year": year,
                "limb_px": limb,
                "center_offset_px": offset,
                "disk_mean": float(vals.mean()) if vals.size else 0.0,
                "disk_std": float(vals.std()) if vals.size else 0.0,
                "disk_p10": float(np.percentile(vals, 10)) if vals.size else 0.0,
                "disk_p50": float(np.percentile(vals, 50)) if vals.size else 0.0,
                "disk_p90": float(np.percentile(vals, 90)) if vals.size else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _cohen(a: np.ndarray, b: np.ndarray) -> float:
    pooled = math.sqrt(0.5 * (a.var() + b.var()))
    if pooled < 1e-6:
        return 0.0
    return float((b.mean() - a.mean()) / pooled)


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    pred = pd.read_csv(CHAMPION)
    pred_by = {
        stem: group["segmentation_rle"].tolist()
        for stem, group in pred.groupby(pred["filament_id"].str.rsplit("_", n=1).str[0])
    }

    gt_rows = []
    fn_rows = []
    split_gt = 0
    merge_pred = 0
    n_gt = 0
    n_pred_scored = 0
    ann_pair_a_covers_two_b = 0
    ann_pair_filaments = 0
    count_ratios = []
    medoid_ids = []
    other_ids = []

    for file_name in val_files:
        gray = cv2.imread(str(TRAIN_IMG_DIR / file_name), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise SystemExit(f"missing {file_name}")
        limb, _offset, _vals = _disk(gray)
        cy = (gray.shape[0] - 1) / 2.0
        cx = (gray.shape[1] - 1) / 2.0
        stem = Path(file_name).stem
        pred_rles = pred_by.get(stem, [])
        pred_layers = torch.from_numpy(rles_to_layers(pred_rles)) if pred_rles else None
        image_ids = by_file[file_name]
        medoid = medoid_view_id(views, image_ids)
        counts = {iid: len(views[iid].filaments) for iid in image_ids}
        if len(image_ids) >= 2:
            count_ratios.append(max(counts.values()) / max(min(counts.values()), 1))
        masks_by = {}
        for image_id in image_ids:
            view = views[image_id]
            masks = []
            rles = []
            feats = []
            for idx, filament in enumerate(view.filaments, start=1):
                mask = rasterize_polygons([filament.polygon])
                masks.append(mask)
                rles.append(encode_rle(mask))
                ys, xs = np.nonzero(mask)
                if len(xs):
                    rho = math.hypot(ys.mean() - cy, xs.mean() - cx) / max(limb, 1.0)
                else:
                    rho = 0.0
                feats.append(
                    {
                        "filament_id": f"{image_id}_{idx}",
                        "rle": rles[-1],
                        "area": int(mask.sum()),
                        "elongation": _elongation(mask),
                        "rho": rho,
                        "contrast": _contrast(gray, mask),
                        "annotator": view.annotator,
                        "is_medoid": image_id == medoid,
                    }
                )
            masks_by[image_id] = masks
            if not rles:
                continue
            gt_layers = torch.from_numpy(np.stack(masks).astype(np.float32))
            if pred_layers is None or pred_layers.shape[0] == 0:
                iou = torch.zeros((len(rles), 0))
            else:
                iou, _ = overlap_matrices(gt_layers, pred_layers)
            n_gt += iou.shape[0]
            n_pred_scored += iou.shape[1]
            max_iou = iou.max(dim=1).values if iou.shape[1] else torch.zeros(iou.shape[0])
            gt_hits = (iou > 0.2).sum(dim=1) if iou.shape[1] else torch.zeros(iou.shape[0])
            pred_hits = (iou > 0.2).sum(dim=0) if iou.shape[1] else torch.zeros(0)
            split_gt += int((gt_hits >= 2).sum().item())
            merge_pred += int((pred_hits >= 2).sum().item())
            for feat, mi, nh in zip(feats, max_iou.tolist(), gt_hits.tolist()):
                feat["max_iou"] = float(mi)
                feat["n_pred_iou_gt_0.2"] = int(nh)
                feat["file"] = file_name
                gt_rows.append(feat)
                if mi <= 0.5:
                    kind = "near_miss" if mi > 0.3 else "no_detection"
                    fn_rows.append({**feat, "kind": kind})
            if image_id == medoid:
                medoid_ids.append(image_id)
            else:
                other_ids.append(image_id)
        if len(image_ids) >= 2:
            for a in image_ids:
                for b in image_ids:
                    if a == b or not masks_by[a] or not masks_by[b]:
                        continue
                    ia = torch.from_numpy(np.stack(masks_by[a]).astype(np.float32))
                    ib = torch.from_numpy(np.stack(masks_by[b]).astype(np.float32))
                    iou_ab, _ = overlap_matrices(ia, ib)
                    covers = int(((iou_ab > 0.2).sum(dim=1) >= 2).sum().item())
                    ann_pair_a_covers_two_b += covers
                    ann_pair_filaments += iou_ab.shape[0]

    gt = pd.DataFrame(gt_rows)
    fn = pd.DataFrame(fn_rows)
    tp = int((gt["max_iou"] > 0.5).sum())
    fn_n = int(len(fn))
    gt_df = gt.rename(columns={"rle": "segmentation_rle", "is_medoid": "medoid"})
    official = score_pq(gt_df[["filament_id", "segmentation_rle"]], pred)
    sq = official.mean_tp_iou
    rq = official.tp / (official.tp + 0.5 * official.fp + 0.5 * official.fn)
    pq_sq = _pq(SQ_HYP, official.tp, official.fp, official.fn)
    pq_recall = _pq(sq, official.tp + official.fn, official.fp, 0)

    def _bucket_report(frame: pd.DataFrame, col: str, edges: list[float], labels: list[str]) -> list[dict]:
        out = []
        cuts = pd.cut(frame[col], bins=edges, labels=labels, include_lowest=True)
        all_cuts = pd.cut(gt[col], bins=edges, labels=labels, include_lowest=True)
        for label in labels:
            sub = frame[cuts == label]
            base = gt[all_cuts == label]
            tp_n = int(((all_cuts == label) & (gt["max_iou"] > 0.5)).sum())
            out.append(
                {
                    "bucket": str(label),
                    "fn": int(len(sub)),
                    "near_miss": int((sub["kind"] == "near_miss").sum()) if len(sub) else 0,
                    "no_detection": int((sub["kind"] == "no_detection").sum()) if len(sub) else 0,
                    "gt": int(len(base)),
                    "tp": tp_n,
                    "fn_rate": (len(sub) / len(base)) if len(base) else 0.0,
                }
            )
        return out

    area_edges = [0, 400, 1500, 5000, 10**9]
    area_labels = ["<400", "400-1500", "1500-5000", ">=5000"]
    elong_edges = [0, 2, 4, 8, 10**3]
    elong_labels = ["<2", "2-4", "4-8", ">=8"]
    rho_edges = [0, 0.5, 0.8, 0.95, 10]
    rho_labels = ["core<0.5", "mid", "outer", "limb>=0.95"]
    contrast_edges = [-10**3, 5, 15, 30, 10**3]
    contrast_labels = ["<5", "5-15", "15-30", ">=30"]

    per_ann = []
    for annotator, group in gt_df.groupby("annotator"):
        scored = score_pq(group[["filament_id", "segmentation_rle"]], pred)
        per_ann.append(
            {
                "annotator": annotator,
                "pq": scored.pq,
                "tp": scored.tp,
                "fp": scored.fp,
                "fn": scored.fn,
                "mean_tp_iou": scored.mean_tp_iou,
                "n_gt": int(len(group)),
            }
        )
    medoid_df = gt_df[gt_df["medoid"]]
    other_df = gt_df[~gt_df["medoid"]]
    medoid_score = score_pq(medoid_df[["filament_id", "segmentation_rle"]], pred)
    other_score = score_pq(other_df[["filament_id", "segmentation_rle"]], pred) if len(other_df) else None

    train_paths = sorted(TRAIN_IMG_DIR.glob("*.jpeg"))
    test_paths = sorted(TEST_IMG_DIR.glob("*.jpeg"))
    train_stats = _image_table(train_paths)
    test_stats = _image_table(test_paths)

    def _mix(frame: pd.DataFrame, col: str) -> dict:
        counts = frame[col].value_counts(normalize=True).sort_index()
        return {str(k): float(v) for k, v in counts.items()}

    shift = {}
    for col in ("limb_px", "center_offset_px", "disk_mean", "disk_std", "disk_p10", "disk_p50", "disk_p90"):
        shift[col] = {
            "train_mean": float(train_stats[col].mean()),
            "test_mean": float(test_stats[col].mean()),
            "cohen_d_test_minus_train": _cohen(train_stats[col].to_numpy(), test_stats[col].to_numpy()),
        }

    payload = {
        "fold0_images": len(val_files),
        "local_gt": n_gt,
        "local_fn": fn_n,
        "local_tp": tp,
        "official": {
            "pq": official.pq,
            "sq": sq,
            "rq": rq,
            "tp": official.tp,
            "fp": official.fp,
            "fn": official.fn,
            "check_pq": sq * rq,
        },
        "counterfactuals": {
            "sq_forced_0.85_pq": pq_sq,
            "sq_forced_0.85_delta": pq_sq - official.pq,
            "recall_100_same_sq_same_fp_pq": pq_recall,
            "recall_100_delta": pq_recall - official.pq,
            "assumption": "100% recall turns every FN into a TP at the current mean IoU and does not add false positives.",
        },
        "fn_buckets": {
            "area_px": _bucket_report(fn, "area", area_edges, area_labels),
            "elongation": _bucket_report(fn, "elongation", elong_edges, elong_labels),
            "rho_over_limb": _bucket_report(fn, "rho", rho_edges, rho_labels),
            "contrast_dn": _bucket_report(fn, "contrast", contrast_edges, contrast_labels),
        },
        "split_merge": {
            "gt_with_2plus_preds_iou_gt_0.2": split_gt,
            "preds_with_2plus_gt_iou_gt_0.2": merge_pred,
            "n_gt": n_gt,
            "annotator_pair_filaments": ann_pair_filaments,
            "annotator_a_covers_2plus_of_b": ann_pair_a_covers_two_b,
            "mean_max_over_min_filament_count": float(np.mean(count_ratios)) if count_ratios else None,
            "images_with_2plus_annotators": len(count_ratios),
        },
        "per_annotator": sorted(per_ann, key=lambda r: r["pq"]),
        "medoid_vs_other": {
            "medoid": {
                "pq": medoid_score.pq,
                "tp": medoid_score.tp,
                "fp": medoid_score.fp,
                "fn": medoid_score.fn,
                "n_gt": int(len(medoid_df)),
                "mean_tp_iou": medoid_score.mean_tp_iou,
            },
            "other": None
            if other_score is None
            else {
                "pq": other_score.pq,
                "tp": other_score.tp,
                "fp": other_score.fp,
                "fn": other_score.fn,
                "n_gt": int(len(other_df)),
                "mean_tp_iou": other_score.mean_tp_iou,
            },
        },
        "shift": {
            "n_train": int(len(train_stats)),
            "n_test": int(len(test_stats)),
            "features": shift,
            "train_site": _mix(train_stats, "site"),
            "test_site": _mix(test_stats, "site"),
            "train_year": _mix(train_stats, "year"),
            "test_year": _mix(test_stats, "year"),
        },
        "test_labels_used": False,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("official", "counterfactuals", "split_merge", "medoid_vs_other")}, indent=2))
    print("DONE phase0", OUT)


if __name__ == "__main__":
    main()
