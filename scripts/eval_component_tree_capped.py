#!/usr/bin/env python3
"""Two constrained component-tree retries. No merge. No submit.

The area, mean-probability, and elongation cuts are chosen on a fixed
sample of folds 1-4, then frozen. Fold 0 is scored once.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_component_tree_lfull import _max_prob_map, _train_areas
from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.component_tree import build_tree_graph, cap_count, fit_log_area_prior, masks_from_graph
from filament.condinst import should_promote
from filament.metric import gt_annotations_to_df, overlap_matrices, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, IMAGE_SIZE, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.rle import decode_rle, encode_rle, rles_to_layers

LFULL = WEIGHTS_DIR / "yolo11l_seg_full_2048" / "weights" / "best.pt"
PARENT_WEIGHTS = WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048" / "weights" / "best.pt"
PARENT_CSV = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
OUT = ARTIFACTS / "component_tree_capped_fold0.json"
PARENT_PQ = 0.4387235694048039
BOX_DILATE = 32
AREAS = (250, 400, 700)
MEANS = (0.45, 0.55, 0.70)
ELONGS = (2.0, 3.0, 4.5)


def _capture_map(model, path: Path) -> np.ndarray:
    import ultralytics.utils.ops as ops

    holder: dict = {}
    original = ops.process_mask_native

    def _capture(proto, masks_in, bboxes, shape):
        holder["map"] = _max_prob_map(proto, masks_in, bboxes, shape)
        return original(proto, masks_in, bboxes, shape)

    ops.process_mask_native = _capture
    try:
        model.predict(
            source=str(path),
            conf=0.01,
            iou=0.7,
            imgsz=IMAGE_SIZE,
            retina_masks=True,
            max_det=100,
            verbose=False,
            device="mps",
        )
    finally:
        ops.process_mask_native = original
    prob = holder.get("map")
    if prob is None:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    return prob


def _parent_count(model, path: Path) -> int:
    result = model.predict(
        source=str(path),
        conf=0.35,
        iou=0.7,
        imgsz=IMAGE_SIZE,
        retina_masks=True,
        max_det=100,
        verbose=False,
        device="mps",
    )[0]
    if result.masks is None:
        return 0
    return int(result.masks.data.shape[0])


def _tune_files(fold_of: dict[str, int]) -> list[str]:
    rng = np.random.default_rng(SEED)
    chosen = []
    for fold in range(1, 5):
        files = sorted(name for name, value in fold_of.items() if value == fold)
        pick = rng.choice(np.array(files), size=4, replace=False)
        chosen.extend(str(name) for name in pick)
    return sorted(chosen)


def _decode(graph: dict, area: int, mean: float, elong: float, cap: int | None):
    masks, scores = masks_from_graph(graph, min_area=area, min_mean=mean, min_elongation=elong)
    if cap is not None:
        masks = cap_count(masks, scores, cap)
    return [encode_rle(mask) for mask in masks]


def _boxes(rles: list[str]) -> list[tuple[int, int, int, int, int]]:
    boxes = []
    for rle in rles:
        mask = decode_rle(rle)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        x1 = max(0, int(xs.min()) - BOX_DILATE)
        y1 = max(0, int(ys.min()) - BOX_DILATE)
        x2 = min(IMAGE_SIZE, int(xs.max()) + 1 + BOX_DILATE)
        y2 = min(IMAGE_SIZE, int(ys.max()) + 1 + BOX_DILATE)
        boxes.append((x1, y1, x2, y2, int(mask.sum())))
    boxes.sort(key=lambda box: box[4], reverse=True)
    return boxes


def _in_parent(prob: np.ndarray, boxes, median: float, sigma: float, area: int, mean: float, elong: float) -> list[str]:
    claimed = np.zeros(prob.shape, dtype=bool)
    rles = []
    for x1, y1, x2, y2, _area in boxes:
        if x2 <= x1 or y2 <= y1:
            continue
        sub = prob[y1:y2, x1:x2].copy()
        sub[claimed[y1:y2, x1:x2]] = 0
        claimed[y1:y2, x1:x2] = True
        if float(sub.max()) <= 0:
            continue
        graph = build_tree_graph(sub, median, sigma, min_area_floor=64)
        masks, _scores = masks_from_graph(graph, min_area=area, min_mean=mean, min_elongation=elong)
        for mask in masks:
            full = np.zeros(prob.shape, dtype=np.uint8)
            full[y1:y2, x1:x2] = mask
            rles.append(encode_rle(full))
    return rles


def _pack(scored, n_pred: int) -> dict:
    return {
        "pq": scored.pq,
        "tp": scored.tp,
        "fp": scored.fp,
        "fn": scored.fn,
        "mean_tp_iou": scored.mean_tp_iou,
        "n_pred": n_pred,
    }


def _same_hits(gt_df: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame) -> bool:
    def hits(frame: pd.DataFrame) -> set[str]:
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
            matched = (iou > 0.5).any(dim=1)
            for filament_id, flag in zip(gt_rows["filament_id"].tolist(), matched.tolist()):
                if flag:
                    found.add(filament_id)
        return found

    return hits(left) == hits(right)


def main() -> None:
    if not LFULL.is_file() or not PARENT_WEIGHTS.is_file():
        raise SystemExit("missing weights")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(name for name, fold in fold_of.items() if fold == 0)
    if len(val_files) != 142:
        raise SystemExit(f"fold 0 should be 142 images, got {len(val_files)}")
    tune_files = _tune_files(fold_of)
    if any(fold_of[name] == 0 for name in tune_files):
        raise SystemExit("tune sample touched fold 0")
    median, sigma = fit_log_area_prior(_train_areas(views, by_file, fold_of))
    print(f"prior median={median:.1f} sigma={sigma:.3f} tune={len(tune_files)}", flush=True)

    from ultralytics import YOLO

    parent_model = YOLO(str(PARENT_WEIGHTS))
    tune_counts = {}
    for name in tune_files:
        tune_counts[name] = _parent_count(parent_model, TRAIN_IMG_DIR / name)
        print(f"tune-count {name} n={tune_counts[name]}", flush=True)
    del parent_model

    model = YOLO(str(LFULL))
    tune_gt = gt_annotations_to_df(gt_records_for_files(views, by_file, tune_files))
    buckets = {(area, mean, elong): [] for area in AREAS for mean in MEANS for elong in ELONGS}
    for name in tune_files:
        graph = build_tree_graph(_capture_map(model, TRAIN_IMG_DIR / name), median, sigma, min_area_floor=64)
        cap = math.floor(tune_counts[name] * 1.3)
        stem = Path(name).stem
        for area, mean, elong in buckets:
            buckets[(area, mean, elong)].append(pred_to_df(stem, _decode(graph, area, mean, elong, cap)))
        print(f"tune-map {name} nodes={len(graph['areas'])} cap={cap}", flush=True)
        del graph

    best = None
    grid_rows = []
    for (area, mean, elong), frames in buckets.items():
        pred = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
        scored = score_pq(tune_gt, pred)
        row = {"area": area, "mean": mean, "elongation": elong, "pq": scored.pq, "tp": scored.tp, "fp": scored.fp, "fn": scored.fn}
        grid_rows.append(row)
        print(f"tune {row}", flush=True)
        rank = (scored.pq, -abs(area - 400) / 400, -abs(mean - 0.55), -abs(elong - 3.0))
        if best is None or rank > best[0]:
            best = (rank, row)
    chosen = best[1]
    print(f"FROZEN {chosen}", flush=True)
    del buckets

    parent = pd.read_csv(PARENT_CSV)
    by_stem: dict[str, list[str]] = {}
    stems = parent["filament_id"].str.rsplit("_", n=1).str[0]
    for stem, rle in zip(stems.tolist(), parent["segmentation_rle"].tolist()):
        by_stem.setdefault(stem, []).append(rle)

    area = int(chosen["area"])
    mean = float(chosen["mean"])
    elong = float(chosen["elongation"])
    cap_frames = []
    box_frames = []
    for name in val_files:
        stem = Path(name).stem
        rles = by_stem.get(stem, [])
        prob = _capture_map(model, TRAIN_IMG_DIR / name)
        graph = build_tree_graph(prob, median, sigma, min_area_floor=64)
        cap = math.floor(len(rles) * 1.3)
        cap_frames.append(pred_to_df(stem, _decode(graph, area, mean, elong, cap)))
        box_frames.append(pred_to_df(stem, _in_parent(prob, _boxes(rles), median, sigma, area, mean, elong)))
        print(f"{name} cap_n={len(cap_frames[-1])} box_n={len(box_frames[-1])} parent_n={len(rles)}", flush=True)

    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    parent_score = score_pq(gt_df, parent)
    if abs(parent_score.pq - PARENT_PQ) > 1e-6:
        raise SystemExit(f"parent rescore drifted: {parent_score.pq}")
    cap_pred = pd.concat(cap_frames, ignore_index=True)
    box_pred = pd.concat(box_frames, ignore_index=True)
    cap_path = FAST_ARTIFACTS / "component_tree_capped_fold0_pred.csv"
    box_path = FAST_ARTIFACTS / "component_tree_in_parent_fold0_pred.csv"
    cap_pred.to_csv(cap_path, index=False)
    box_pred.to_csv(box_path, index=False)
    cap_score = score_pq(gt_df, cap_pred)
    box_score = score_pq(gt_df, box_pred)

    def promote(scored, frame: pd.DataFrame) -> bool:
        same = False
        if scored.pq >= 0.47 and scored.tp == parent_score.tp:
            same = _same_hits(gt_df, frame, parent)
        return should_promote(scored.pq, scored.tp, same_hit_set=same)

    payload = {
        "split": "grouped year x count fold 0",
        "seed": SEED,
        "tune_files": tune_files,
        "tune_folds": sorted({int(fold_of[name]) for name in tune_files}),
        "frozen": chosen,
        "grid": grid_rows,
        "box_dilate_px": BOX_DILATE,
        "count_cap": "floor(parent_instance_count * 1.3)",
        "fold0_used_to_pick_cuts": False,
        "parent": _pack(parent_score, len(parent)),
        "tree_uncapped_previous": {"pq": 0.1644091695214393, "tp": 1140, "fp": 6318, "fn": 449},
        "tree_caps": _pack(cap_score, len(cap_pred)),
        "tree_in_parent": _pack(box_score, len(box_pred)),
        "promote_caps": promote(cap_score, cap_pred),
        "promote_in_parent": promote(box_score, box_pred),
        "union_with_parent": False,
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("frozen", "parent", "tree_caps", "tree_in_parent", "promote_caps", "promote_in_parent")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
