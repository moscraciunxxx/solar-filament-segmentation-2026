#!/usr/bin/env python3
"""Score the CLV ConvNeXt map on fold 0. Threshold+CC, then the component tree.

Keep rules are chosen on folds 1-4 and frozen before the fold-0 tree score.
Does not submit.
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

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.component_tree import MIN_AREA, build_tree_graph, fit_log_area_prior, masks_from_graph
from filament.condinst import should_promote
from filament.flatfield import center_to_limb_flat
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, IMAGE_SIZE, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.preprocess import read_gray
from filament.rle import encode_rle
from filament.soft_clv_dataset import IMAGENET_MEAN, IMAGENET_STD, soft_annotator_mean
from filament.unet_convnext import ConvNeXtUNet

FAMILY = "unet_convnext_clv_1536"
IMGSZ = 1536
PARENT_PQ = 0.4387235694048039
OLD_UNET_PQ = 0.354
CC_KILL = 0.35
MIN_AREA_GRID = (64, 250, 642, 1000)
PREDECLARED_MIN_AREA = 642
AREAS = (250, 400, 700)
MEANS = (0.45, 0.55, 0.70)
ELONGS = (2.0, 3.0, 4.5)


def _tensor(flat: np.ndarray, device: str) -> torch.Tensor:
    resized = cv2.resize(flat, (IMGSZ, IMGSZ), interpolation=cv2.INTER_AREA)
    rgb = np.stack([resized, resized, resized], axis=-1)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    image = torch.from_numpy(np.transpose(rgb, (2, 0, 1)).astype(np.float32))
    return image.unsqueeze(0).to(device)


@torch.no_grad()
def probability_map(model, gray: np.ndarray, device: str) -> np.ndarray:
    flat = center_to_limb_flat(gray)
    logits = model(_tensor(flat, device))
    prob = torch.sigmoid(logits)[0, 0]
    up = torch.nn.functional.interpolate(prob[None, None], size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    return up[0, 0].float().cpu().numpy()


def cc_masks(prob: np.ndarray, threshold: float = 0.5, min_area: int = MIN_AREA) -> list[np.ndarray]:
    count, labels = cv2.connectedComponents((prob >= threshold).astype(np.uint8), connectivity=8)
    masks = []
    for label in range(1, int(count)):
        mask = labels == label
        if int(mask.sum()) >= min_area:
            masks.append(mask.astype(np.uint8))
    return masks


def rles_by_min_area(prob: np.ndarray, areas: tuple[int, ...] = MIN_AREA_GRID, threshold: float = 0.5) -> dict[int, list[str]]:
    """One connected-component pass. A component is encoded once and kept by every area it clears."""
    count, labels = cv2.connectedComponents((prob >= threshold).astype(np.uint8), connectivity=8)
    sizes = np.bincount(labels.ravel())
    buckets = {area: [] for area in areas}
    for label in range(1, int(count)):
        size = int(sizes[label])
        qualifying = [area for area in areas if size >= area]
        if not qualifying:
            continue
        rle = encode_rle((labels == label).astype(np.uint8))
        for area in qualifying:
            buckets[area].append(rle)
    return buckets


def choose_min_area(rows: list[dict], prefer: int = PREDECLARED_MIN_AREA) -> dict:
    """Highest tune PQ. Ties break toward the predeclared area, then the smaller area."""
    best = max(row["pq"] for row in rows)
    tied = [row for row in rows if abs(row["pq"] - best) <= 1e-12]
    tied.sort(key=lambda row: (abs(int(row["area"]) - prefer), int(row["area"])))
    return dict(tied[0])


def _frames(model, files, device: str, decode) -> pd.DataFrame:
    rows = []
    for name in files:
        prob = probability_map(model, read_gray(TRAIN_IMG_DIR / name), device)
        rles = [encode_rle(mask) for mask in decode(prob)]
        rows.append(pred_to_df(Path(name).stem, rles))
        print(f"{name} n={len(rles)} max={float(prob.max()):.3f}", flush=True)
    if not rows:
        return pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    return pd.concat(rows, ignore_index=True)


def score_threshold_cc(model, files, device: str) -> dict:
    views, by_file = load_coco()
    gt = gt_annotations_to_df(gt_records_for_files(views, by_file, files))
    pred = _frames(model, files, device, cc_masks)
    scored = score_pq(gt, pred)
    return {"pq": scored.pq, "tp": scored.tp, "fp": scored.fp, "fn": scored.fn, "n_pred": len(pred)}


def load_model(path: Path, device: str) -> ConvNeXtUNet:
    model = ConvNeXtUNet(pretrained=False)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def main() -> None:
    weights = WEIGHTS_DIR / FAMILY / "best.pt"
    if not weights.is_file():
        raise SystemExit(f"missing {weights}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(name for name, fold in fold_of.items() if fold == 0)
    tune_files = []
    rng = np.random.default_rng(SEED)
    for fold in range(1, 5):
        pool = sorted(name for name, value in fold_of.items() if value == fold)
        tune_files.extend(str(name) for name in rng.choice(np.array(pool), size=4, replace=False))
    model = load_model(weights, device)
    # Area prior from soft-target mass on the tune files, not fold 0.
    areas = []
    for name in tune_files:
        soft = soft_annotator_mean(views, by_file[name])
        count, labels = cv2.connectedComponents((soft >= 0.5).astype(np.uint8), connectivity=8)
        for label in range(1, int(count)):
            area = int((labels == label).sum())
            if area >= 1:
                areas.append(float(area))
    median, sigma = fit_log_area_prior(areas)
    tune_maps = [(name, probability_map(model, read_gray(TRAIN_IMG_DIR / name), device)) for name in tune_files]
    tune_gt = gt_annotations_to_df(gt_records_for_files(views, by_file, tune_files))
    best = None
    grid = []
    for area in AREAS:
        for mean in MEANS:
            for elong in ELONGS:
                frames = []
                for name, prob in tune_maps:
                    graph = build_tree_graph(prob, median, sigma, min_area_floor=64)
                    masks, _scores = masks_from_graph(graph, min_area=area, min_mean=mean, min_elongation=elong)
                    frames.append(pred_to_df(Path(name).stem, [encode_rle(mask) for mask in masks]))
                pred = pd.concat(frames, ignore_index=True)
                scored = score_pq(tune_gt, pred)
                row = {"area": area, "mean": mean, "elongation": elong, "pq": scored.pq}
                grid.append(row)
                rank = (scored.pq, -abs(area - 400) / 400, -abs(mean - 0.55), -abs(elong - 3.0))
                if best is None or rank > best[0]:
                    best = (rank, row)
                print(f"tune {row}", flush=True)
    chosen = best[1]
    print(f"FROZEN {chosen}", flush=True)
    gt = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    cc_pred = _frames(model, val_files, device, cc_masks)
    cc = score_pq(gt, cc_pred)

    def tree_decode(prob: np.ndarray):
        graph = build_tree_graph(prob, median, sigma, min_area_floor=64)
        masks, _scores = masks_from_graph(
            graph,
            min_area=int(chosen["area"]),
            min_mean=float(chosen["mean"]),
            min_elongation=float(chosen["elongation"]),
        )
        return masks

    tree_pred = _frames(model, val_files, device, tree_decode)
    tree = score_pq(gt, tree_pred)
    parent = pd.read_csv(FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv")
    parent_score = score_pq(gt, parent)
    if abs(parent_score.pq - PARENT_PQ) > 1e-6:
        raise SystemExit(f"parent rescore drifted: {parent_score.pq}")
    payload = {
        "weights": str(weights),
        "imgsz": IMGSZ,
        "frozen": chosen,
        "grid": grid,
        "fold0_used_to_pick_cuts": False,
        "parent_pq": parent_score.pq,
        "old_unet_resnet34_1536_pq": OLD_UNET_PQ,
        "threshold_cc": {"pq": cc.pq, "tp": cc.tp, "fp": cc.fp, "fn": cc.fn, "n_pred": len(cc_pred)},
        "tree": {"pq": tree.pq, "tp": tree.tp, "fp": tree.fp, "fn": tree.fn, "n_pred": len(tree_pred)},
        "promote": should_promote(tree.pq, tree.tp, same_hit_set=True),
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    out = ARTIFACTS / "soft_clv_fold0.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


def epoch3_only() -> None:
    """Score the saved epoch-3 checkpoint. Does not train and does not submit."""
    weights = WEIGHTS_DIR / FAMILY / "best.pt"
    curve = json.loads((weights.parent / "train_curve.json").read_text())
    if len(curve["history"]) != 3 or curve["history"][-1]["epoch"] != 3:
        raise SystemExit("epoch-3 check requires the three-epoch checkpoint, not a later one")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(name for name, fold in fold_of.items() if fold == 0)
    model = load_model(weights, device)
    report = score_threshold_cc(model, val_files, device)
    report["kill_if_under"] = CC_KILL
    report["continue"] = report["pq"] >= CC_KILL
    report["epochs_trained"] = len(curve["history"])
    report["best_train_loss"] = curve["best_train_loss"]
    report["weights"] = str(weights)
    out = ARTIFACTS / "soft_clv_epoch3_cc.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"epoch3_cc": report}), flush=True)


def _require_epoch3_curve(weights: Path) -> dict:
    curve = json.loads((weights.parent / "train_curve.json").read_text())
    if len(curve["history"]) != 3 or curve["history"][-1]["epoch"] != 3:
        raise SystemExit("this rescore requires the three-epoch checkpoint, not a later one")
    return curve


def _score_rows(gt: pd.DataFrame, frames: dict[int, list[pd.DataFrame]], areas: tuple[int, ...]) -> list[dict]:
    rows = []
    for area in areas:
        pred = pd.concat(frames[area], ignore_index=True) if frames[area] else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
        scored = score_pq(gt, pred)
        rows.append({"area": area, "pq": scored.pq, "tp": scored.tp, "fp": scored.fp, "fn": scored.fn, "n_pred": len(pred)})
    return rows


def minarea_rescore() -> None:
    """Rescore the epoch-3 map. Min area 642 is predeclared. The grid is frozen on folds 1-4."""
    weights = WEIGHTS_DIR / FAMILY / "best.pt"
    if not weights.is_file():
        raise SystemExit(f"missing {weights}")
    curve = _require_epoch3_curve(weights)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    tune_files = sorted(name for name, fold in fold_of.items() if fold != 0)
    val_files = sorted(name for name, fold in fold_of.items() if fold == 0)
    if len(val_files) != 142 or any(fold_of[name] == 0 for name in tune_files):
        raise SystemExit("fold split is wrong")
    model = load_model(weights, device)
    tune_gt = gt_annotations_to_df(gt_records_for_files(views, by_file, tune_files))
    tune_frames = {area: [] for area in MIN_AREA_GRID}
    for index, name in enumerate(tune_files, start=1):
        prob = probability_map(model, read_gray(TRAIN_IMG_DIR / name), device)
        buckets = rles_by_min_area(prob)
        stem = Path(name).stem
        for area in MIN_AREA_GRID:
            tune_frames[area].append(pred_to_df(stem, buckets[area]))
        if index % 25 == 0 or index == len(tune_files):
            print(f"tune {index}/{len(tune_files)}", flush=True)
    tune_rows = _score_rows(tune_gt, tune_frames, MIN_AREA_GRID)
    for row in tune_rows:
        print(f"tune {row}", flush=True)
    frozen = choose_min_area(tune_rows)
    print(f"FROZEN {frozen}", flush=True)
    frozen_note = {
        "frozen_min_area": int(frozen["area"]),
        "tune_rows": tune_rows,
        "fold0_used_to_pick_min_area": False,
        "n_tune": len(tune_files),
    }
    (ARTIFACTS / "soft_clv_epoch3_minarea_frozen.json").write_text(json.dumps(frozen_note, indent=2) + "\n")
    fold0_areas = tuple(dict.fromkeys((PREDECLARED_MIN_AREA, int(frozen["area"]))))
    val_gt = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    val_frames = {area: [] for area in fold0_areas}
    for name in val_files:
        prob = probability_map(model, read_gray(TRAIN_IMG_DIR / name), device)
        buckets = rles_by_min_area(prob, areas=fold0_areas)
        stem = Path(name).stem
        for area in fold0_areas:
            val_frames[area].append(pred_to_df(stem, buckets[area]))
        print(f"fold0 {name} " + " ".join(f"a{area}={len(buckets[area])}" for area in fold0_areas), flush=True)
    fold0_rows = {row["area"]: row for row in _score_rows(val_gt, val_frames, fold0_areas)}
    predeclared = fold0_rows[PREDECLARED_MIN_AREA]
    frozen_fold0 = fold0_rows[int(frozen["area"])]
    parent = pd.read_csv(FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv")
    parent_score = score_pq(val_gt, parent)
    if abs(parent_score.pq - PARENT_PQ) > 1e-6:
        raise SystemExit(f"parent rescore drifted: {parent_score.pq}")
    resume = predeclared["pq"] >= CC_KILL or frozen_fold0["pq"] >= CC_KILL
    payload = {
        "weights": str(weights),
        "epochs_trained": len(curve["history"]),
        "best_train_loss": curve["best_train_loss"],
        "threshold": 0.5,
        "predeclared_min_area": PREDECLARED_MIN_AREA,
        "grid": list(MIN_AREA_GRID),
        "tune_rows": tune_rows,
        "frozen_min_area": int(frozen["area"]),
        "fold0_used_to_pick_min_area": False,
        "fold0_min_area_642": predeclared,
        "fold0_frozen_min_area": frozen_fold0,
        "parent_pq": parent_score.pq,
        "parent_tp": parent_score.tp,
        "parent_fp": parent_score.fp,
        "parent_fn": parent_score.fn,
        "old_unet_resnet34_1536_pq": OLD_UNET_PQ,
        "resume_epochs_4_to_8": resume,
        "keep_kill": not resume,
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    out = ARTIFACTS / "soft_clv_epoch3_minarea.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    if "--epoch3" in sys.argv:
        epoch3_only()
    elif "--minarea" in sys.argv:
        minarea_rescore()
    else:
        main()
