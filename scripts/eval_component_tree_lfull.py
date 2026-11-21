#!/usr/bin/env python3
"""Component-tree decode of yolo11l_seg_full_2048 on locked fold 0.

The probability map is the sigmoid of the max pre-threshold mask logit
across instances. The area prior is fit on medoid areas from folds other
than 0. Fold 0 is scored once. No merge with the parent file. No submit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco, medoid_view_id
from filament.component_tree import (
    BASE_THRESHOLD,
    LEVELS,
    MIN_AREA,
    baseline_components,
    fit_log_area_prior,
    tree_components,
)
from filament.condinst import should_promote
from filament.metric import gt_annotations_to_df, overlap_matrices, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, IMAGE_SIZE, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.rle import encode_rle, rles_to_layers

WEIGHTS = WEIGHTS_DIR / "yolo11l_seg_full_2048" / "weights" / "best.pt"
PARENT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
OUT = ARTIFACTS / "component_tree_lfull_fold0.json"
PARENT_PQ = 0.4387235694048039
LOGIT_FILL = -20.0


def _polygon_area(polygon: np.ndarray) -> float:
    contour = polygon.astype(np.float32).reshape(-1, 1, 2)
    return float(cv2.contourArea(contour))


def _train_areas(views, by_file, fold_of) -> list[float]:
    areas = []
    for name, fold in fold_of.items():
        if fold == 0:
            continue
        view = views[medoid_view_id(views, by_file[name])]
        for filament in view.filaments:
            area = _polygon_area(filament.polygon)
            if area >= 1.0:
                areas.append(area)
    return areas


def _max_prob_map(proto: torch.Tensor, coeffs: torch.Tensor, boxes: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    """Sigmoid of the max cropped mask logit. proto is (C, H, W)."""
    from ultralytics.utils import ops

    height, width = shape
    if coeffs.numel() == 0:
        return np.zeros(shape, dtype=np.float32)
    channels = proto.shape[0]
    logits = (coeffs @ proto.float().reshape(channels, -1)).reshape(-1, proto.shape[1], proto.shape[2])
    step = max(1, 32_000_000 // (height * width))
    acc = None
    ys = torch.arange(height, device=logits.device)
    xs = torch.arange(width, device=logits.device)
    for start in range(0, logits.shape[0], step):
        stop = min(start + step, logits.shape[0])
        up = ops.scale_masks(logits[start:stop][None], shape)[0]
        box = boxes[start:stop]
        x1, y1, x2, y2 = box.T
        inside = (
            (xs[None, None, :] >= x1[:, None, None])
            & (xs[None, None, :] < x2[:, None, None])
            & (ys[None, :, None] >= y1[:, None, None])
            & (ys[None, :, None] < y2[:, None, None])
        )
        up = torch.where(inside, up, torch.full_like(up, LOGIT_FILL))
        chunk = up.amax(dim=0)
        acc = chunk if acc is None else torch.maximum(acc, chunk)
    return torch.sigmoid(acc).detach().float().cpu().numpy()


def _frames(model, files, median: float, sigma: float):
    """Return (baseline frames, tree frames) of prediction dataframes."""
    import ultralytics.utils.ops as ops

    base_rows = []
    tree_rows = []
    original = ops.process_mask_native
    holder: dict = {}

    def _capture(proto, masks_in, bboxes, shape):
        holder["map"] = _max_prob_map(proto, masks_in, bboxes, shape)
        return original(proto, masks_in, bboxes, shape)

    ops.process_mask_native = _capture
    try:
        for name in files:
            holder.clear()
            model.predict(
                source=str(TRAIN_IMG_DIR / name),
                conf=0.01,
                iou=0.7,
                imgsz=IMAGE_SIZE,
                retina_masks=True,
                max_det=100,
                verbose=False,
                device="mps",
            )
            prob = holder.get("map")
            if prob is None:
                prob = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            stem = Path(name).stem
            base_rles = [encode_rle(mask) for mask in baseline_components(prob)]
            tree_rles = [encode_rle(mask) for mask in tree_components(prob, median, sigma)]
            base_rows.append(pred_to_df(stem, base_rles))
            tree_rows.append(pred_to_df(stem, tree_rles))
            mid = float(((prob > 0.05) & (prob < 0.95)).mean())
            print(
                f"{name} base={len(base_rles)} tree={len(tree_rles)} "
                f"map_max={float(prob.max()):.3f} mid={mid:.4f}",
                flush=True,
            )
    finally:
        ops.process_mask_native = original
    return base_rows, tree_rows


def _hit_ids(gt_df: pd.DataFrame, frame: pd.DataFrame) -> set[str]:
    gt_keys = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_keys = frame["filament_id"].str.split("_", n=1).str[0]
    found: set[str] = set()
    for annotator_image in gt_keys.unique():
        image_id = annotator_image.split("-", maxsplit=1)[1]
        gt_rows = gt_df.loc[gt_keys == annotator_image]
        pred_rles = frame.loc[pred_keys == image_id, "segmentation_rle"].tolist()
        if not pred_rles:
            continue
        gt_layers = torch.from_numpy(rles_to_layers(gt_rows["segmentation_rle"].tolist()))
        pred_layers = torch.from_numpy(rles_to_layers(pred_rles))
        iou, _ = overlap_matrices(gt_layers, pred_layers)
        hit = (iou > 0.5).any(dim=1)
        for filament_id, matched in zip(gt_rows["filament_id"].tolist(), hit.tolist()):
            if matched:
                found.add(filament_id)
    return found


def _pack(scored, n_pred: int) -> dict:
    return {
        "pq": scored.pq,
        "tp": scored.tp,
        "fp": scored.fp,
        "fn": scored.fn,
        "mean_tp_iou": scored.mean_tp_iou,
        "n_pred": n_pred,
        "n_annotator_images": scored.n_annotator_images,
    }


def main() -> None:
    if not WEIGHTS.is_file():
        raise SystemExit(f"missing weights: {WEIGHTS}")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(name for name, fold in fold_of.items() if fold == 0)
    if len(val_files) != 142:
        raise SystemExit(f"fold 0 should be 142 images, got {len(val_files)}")
    areas = _train_areas(views, by_file, fold_of)
    median, sigma = fit_log_area_prior(areas)
    print(f"area prior from {len(areas)} train-fold medoids median={median:.1f} sigma={sigma:.3f}", flush=True)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    parent = pd.read_csv(PARENT)
    parent_score = score_pq(gt_df, parent)
    if abs(parent_score.pq - PARENT_PQ) > 1e-6:
        raise SystemExit(f"parent rescore drifted: {parent_score.pq}")

    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))
    base_rows, tree_rows = _frames(model, val_files, median, sigma)
    base = pd.concat(base_rows, ignore_index=True) if base_rows else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    tree = pd.concat(tree_rows, ignore_index=True) if tree_rows else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    base_path = FAST_ARTIFACTS / "component_tree_cc_fold0_pred.csv"
    tree_path = FAST_ARTIFACTS / "component_tree_tree_fold0_pred.csv"
    base.to_csv(base_path, index=False)
    tree.to_csv(tree_path, index=False)
    base_score = score_pq(gt_df, base)
    tree_score = score_pq(gt_df, tree)
    same = False
    if tree_score.pq >= 0.47 and tree_score.tp == parent_score.tp:
        same = _hit_ids(gt_df, tree) == _hit_ids(gt_df, parent)
    payload = {
        "split": "grouped year x count fold 0",
        "seed": SEED,
        "weights": str(WEIGHTS),
        "fold0_held_out_of_training": True,
        "map": "sigmoid of max pre-threshold instance logit, conf 0.01, max_det 100",
        "levels": [float(x) for x in LEVELS],
        "baseline_threshold": BASE_THRESHOLD,
        "min_area": MIN_AREA,
        "area_prior_median": median,
        "area_prior_sigma": sigma,
        "area_prior_fit_on": "medoid polygon areas, folds other than 0",
        "parent": _pack(parent_score, len(parent)),
        "threshold_cc": _pack(base_score, len(base)),
        "tree": _pack(tree_score, len(tree)),
        "public_gap_used": 0.069,
        "public_forecast": tree_score.pq - 0.069,
        "promote": should_promote(tree_score.pq, tree_score.tp, same_hit_set=same),
        "union_with_parent": False,
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
