#!/usr/bin/env python3
"""Phase A: champion ∪ Mask R-CNN, then a gradient booster on folds 1-4.

Out-of-fold Mask R-CNN instances must already exist for folds 1-4.
Fold 0 is scored only with --fold0, after thresholds are frozen.
Ground truth uses pycocotools polygon rasterization.
Does not submit. Does not read test labels.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco, polygon_to_coco_rle
from filament.contrast_floor import mask_contrast
from filament.limb import DISK_RADIUS
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, IMAGE_SIZE, SEED, TRAIN_IMG_DIR
from filament.rle import decode_rle, encode_rle

CHAMPION_DIR = FAST_ARTIFACTS / "tile_addon" / "champion"
OOF_ROOT = FAST_ARTIFACTS / "maskrcnn_oof"
TTA_DIRS = (
    FAST_ARTIFACTS / "tta_parent_2048",
    FAST_ARTIFACTS / "tta_s20260911",
    FAST_ARTIFACTS / "tta_s20260912",
)
FEATURE_DIR = FAST_ARTIFACTS / "phase_a_features"
TUNE_PATH = ARTIFACTS / "phase_a_gbm_tune_folds14.json"
FOLD0_PATH = ARTIFACTS / "phase_a_gbm_fold0.json"
COMPARE = 512
N_VOTERS = 72
IOU_MERGE = 0.5
THRESHOLDS = (0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40)
BIN_EDGES = (0.5, 0.75, 0.9)
COCO_CHAMPION_FOLD0 = 0.4655531057360983
FEATURES = (
    "vote_fraction",
    "mrcnn_score",
    "agreement_iou",
    "tta_iou_std",
    "area",
    "r_over_R",
    "elongation",
    "contrast",
    "mean_prob",
)


def _small(mask: np.ndarray) -> np.ndarray:
    if mask.shape[0] == COMPARE:
        return mask
    return cv2.resize(mask, (COMPARE, COMPARE), interpolation=cv2.INTER_NEAREST)


def _box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _iou(a: np.ndarray, b: np.ndarray, ba, bb) -> float:
    if ba is None or bb is None:
        return 0.0
    if ba[2] < bb[0] or bb[2] < ba[0] or ba[3] < bb[1] or bb[3] < ba[1]:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def _elongation(mask: np.ndarray) -> float:
    moments = cv2.moments((mask > 0).astype(np.uint8), binaryImage=True)
    area = float(moments["m00"])
    if area <= 0:
        return 1.0
    mu20 = moments["mu20"] / area
    mu02 = moments["mu02"] / area
    mu11 = moments["mu11"] / area
    trace = mu20 + mu02
    det = mu20 * mu02 - mu11 * mu11
    root = max(trace * trace - 4.0 * det, 0.0) ** 0.5
    major = 0.5 * (trace + root)
    minor = 0.5 * (trace - root)
    if minor <= 1e-9:
        return 1.0
    return float((major / minor) ** 0.5)


def _radial(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    cy = (IMAGE_SIZE - 1) / 2.0
    cx = cy
    dist = float(np.hypot(xs.mean() - cx, ys.mean() - cy))
    return dist / DISK_RADIUS


def _bin(r_over_r: float) -> int:
    for index, edge in enumerate(BIN_EDGES):
        if r_over_r < edge:
            return index
    return len(BIN_EDGES)


def _champion_votes(stem: str, champ_small: list[np.ndarray], champ_boxes) -> tuple[list[float], list[float]]:
    votes = [0] * len(champ_small)
    iou_lists: list[list[float]] = [[] for _ in champ_small]
    for directory in TTA_DIRS:
        path = directory / f"{stem}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for rles in payload.values():
            for rle in rles:
                view = _small(decode_rle(rle))
                box = _box(view)
                best_i = -1
                best = 0.0
                for index, champ in enumerate(champ_small):
                    score = _iou(view, champ, box, champ_boxes[index])
                    if score > best:
                        best = score
                        best_i = index
                if best_i >= 0 and best >= IOU_MERGE:
                    votes[best_i] += 1
                    iou_lists[best_i].append(best)
    fractions = [count / N_VOTERS for count in votes]
    stds = [float(np.std(vals)) if len(vals) > 1 else 0.0 for vals in iou_lists]
    return fractions, stds


def _annotator_target(mask: np.ndarray, gt_by_ann: list[list[np.ndarray]]) -> float:
    if not gt_by_ann:
        return 0.0
    box = _box(mask)
    matched = []
    for group in gt_by_ann:
        best = 0.0
        for gt in group:
            best = max(best, _iou(mask, gt, box, _box(gt)))
        if best > IOU_MERGE:
            matched.append(best)
    if not matched:
        return 0.0
    return (len(matched) / len(gt_by_ann)) * float(np.mean(matched))


def _candidates_for_image(stem: str, file_name: str, views, by_file, gray: np.ndarray) -> list[dict]:
    champ_path = CHAMPION_DIR / f"{stem}.json"
    fold = None
    champ_rles = json.loads(champ_path.read_text()) if champ_path.exists() else []
    oof_rows = []
    for fold_id in (0, 1, 2, 3, 4):
        path = OOF_ROOT / f"fold{fold_id}" / f"{stem}.json"
        if path.exists():
            oof_rows = json.loads(path.read_text())
            fold = fold_id
            break
    champ_masks = [decode_rle(rle) for rle in champ_rles]
    mrcnn = []
    for row in oof_rows:
        mrcnn.append(
            {
                "mask": decode_rle(row["rle"]),
                "score": float(row["score"]),
                "mean_prob": float(row["mean_prob"]),
            }
        )
    small_c = [_small(m) for m in champ_masks]
    boxes_c = [_box(m) for m in small_c]
    vote_fraction, tta_std = _champion_votes(stem, small_c, boxes_c) if champ_masks else ([], [])
    items = []
    for index, mask in enumerate(champ_masks):
        items.append(
            {
                "mask": mask,
                "small": small_c[index],
                "box": boxes_c[index],
                "kind": "champion",
                "vote_fraction": vote_fraction[index],
                "tta_iou_std": tta_std[index],
                "score": 0.0,
                "mean_prob": 0.0,
            }
        )
    for row in mrcnn:
        small = _small(row["mask"])
        items.append(
            {
                "mask": row["mask"],
                "small": small,
                "box": _box(small),
                "kind": "mrcnn",
                "vote_fraction": 0.0,
                "tta_iou_std": 0.0,
                "score": row["score"],
                "mean_prob": row["mean_prob"],
            }
        )
    items.sort(key=lambda item: int(item["mask"].sum()), reverse=True)
    used = [False] * len(items)
    clusters = []
    for i, item in enumerate(items):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            if _iou(item["small"], items[j]["small"], item["box"], items[j]["box"]) >= IOU_MERGE:
                group.append(j)
                used[j] = True
        clusters.append(group)
    gt_by_ann = []
    for image_id in by_file[file_name]:
        group = []
        for filament in views[image_id].filaments:
            if len(filament.polygon) < 3:
                continue
            group.append(_small(decode_rle(polygon_to_coco_rle(filament.polygon))))
        if group:
            gt_by_ann.append(group)
    rows = []
    for group in clusters:
        members = [items[i] for i in group]
        champs = [m for m in members if m["kind"] == "champion"]
        rcnn = [m for m in members if m["kind"] == "mrcnn"]
        if champs and rcnn:
            source = "both"
        elif champs:
            source = "champion"
        else:
            source = "mrcnn"
        geometry = champs[0]["mask"] if champs else rcnn[0]["mask"]
        agreement = 0.0
        if champs and rcnn:
            for left in champs:
                for right in rcnn:
                    agreement = max(
                        agreement,
                        _iou(left["small"], right["small"], left["box"], right["box"]),
                    )
        contrast = mask_contrast(geometry, gray)
        rows.append(
            {
                "stem": stem,
                "fold": fold,
                "source": source,
                "rle": encode_rle(geometry),
                "vote_fraction": max((m["vote_fraction"] for m in champs), default=0.0),
                "mrcnn_score": max((m["score"] for m in rcnn), default=0.0),
                "agreement_iou": agreement,
                "tta_iou_std": max((m["tta_iou_std"] for m in champs), default=0.0),
                "area": int(geometry.sum()),
                "r_over_R": _radial(geometry),
                "elongation": _elongation(geometry),
                "contrast": 0.0 if contrast is None else contrast,
                "mean_prob": max((m["mean_prob"] for m in rcnn), default=0.0),
                "target": _annotator_target(_small(geometry), gt_by_ann),
            }
        )
    return rows


def _files_for_folds(fold_of: dict[str, int], folds: set[int]) -> list[str]:
    return sorted(name for name, fold in fold_of.items() if fold in folds)


def build(folds: set[int], dest: Path) -> None:
    if dest.exists():
        print(f"features exist {dest}", flush=True)
        return
    missing = [fold for fold in sorted(folds) if not (OOF_ROOT / f"fold{fold}").exists()]
    if missing:
        raise SystemExit(f"missing OOF folds {missing}")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    files = _files_for_folds(fold_of, folds)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".jsonl.tmp")
    with tmp.open("w") as handle:
        for index, file_name in enumerate(files, start=1):
            stem = Path(file_name).stem
            gray = cv2.imread(str(TRAIN_IMG_DIR / file_name), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise FileNotFoundError(file_name)
            for row in _candidates_for_image(stem, file_name, views, by_file, gray):
                handle.write(json.dumps(row) + "\n")
            if index % 25 == 0:
                print(f"features {index}/{len(files)}", flush=True)
    tmp.replace(dest)
    print(f"DONE features {dest}", flush=True)


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _predict(model, rows: list[dict]) -> np.ndarray:
    frame = pd.DataFrame([{name: row[name] for name in FEATURES} for row in rows])
    return model.predict(frame)


def _kept(rows: list[dict], pred: np.ndarray, thresholds: list[float]) -> list[dict]:
    kept = []
    for row, score in zip(rows, pred):
        if score > thresholds[_bin(row["r_over_R"])]:
            kept.append(row)
    return kept


def _score_rows(rows: list[dict], views, by_file, files: list[str]) -> dict:
    by_stem = defaultdict(list)
    sources = defaultdict(list)
    for row in rows:
        by_stem[row["stem"]].append(row["rle"])
        sources[row["stem"]].append(row["source"])
    frames = []
    tagged = []
    for file_name in files:
        stem = Path(file_name).stem
        rles = by_stem.get(stem, [])
        frames.append(pred_to_df(stem, rles))
        for source in sources.get(stem, []):
            tagged.append((stem, source))
    pred = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    gt = gt_annotations_to_df(gt_records_for_files(views, by_file, files, rasterizer="coco"))
    scored = score_pq(gt, pred)
    return {
        "pq": scored.pq,
        "tp": scored.tp,
        "fp": scored.fp,
        "fn": scored.fn,
        "n_pred": int(len(pred)),
        "mean_tp_iou": scored.mean_tp_iou,
        "by_source": _source_counts(gt, pred, tagged),
    }


def _source_counts(gt: pd.DataFrame, pred: pd.DataFrame, tagged: list[tuple[str, str]]) -> dict:
    from filament.metric import overlap_matrices
    from filament.rle import rles_to_layers
    import torch

    gt_keys = gt["filament_id"].str.split("_", n=1).str[0]
    pred_keys = pred["filament_id"].str.split("_", n=1).str[0]
    source_of = {}
    cursor = defaultdict(int)
    for stem, source in tagged:
        cursor[stem] += 1
        source_of[f"{stem}_{cursor[stem]}"] = source
    counts = {
        name: {"tp": 0, "fp": 0}
        for name in ("champion", "mrcnn", "both")
    }
    for annotator_image in gt_keys.unique():
        image_id = annotator_image.split("-", maxsplit=1)[1]
        gt_rles = gt.loc[gt_keys == annotator_image, "segmentation_rle"].tolist()
        pred_ids = pred.loc[pred_keys == image_id, "filament_id"].tolist()
        pred_rles = pred.loc[pred_keys == image_id, "segmentation_rle"].tolist()
        if not pred_rles:
            continue
        if not gt_rles:
            for filament_id in pred_ids:
                counts[source_of[filament_id]]["fp"] += 1
            continue
        gt_layers = torch.from_numpy(rles_to_layers(gt_rles))
        pred_layers = torch.from_numpy(rles_to_layers(pred_rles))
        iou, _ = overlap_matrices(gt_layers, pred_layers)
        hit = iou > IOU_MERGE
        for index, filament_id in enumerate(pred_ids):
            source = source_of[filament_id]
            tp = int(hit[:, index].sum().item())
            counts[source]["tp"] += tp
            if tp == 0:
                counts[source]["fp"] += 1
    return counts


def tune() -> None:
    path = FEATURE_DIR / "folds14.jsonl"
    rows = _load_rows(path)
    from sklearn.ensemble import GradientBoostingRegressor

    frame = pd.DataFrame([{name: row[name] for name in FEATURES} for row in rows])
    target = np.array([row["target"] for row in rows], dtype=np.float64)
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=SEED,
    )
    model.fit(frame, target)
    pred = model.predict(frame)
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    files = _files_for_folds(fold_of, {1, 2, 3, 4})
    thresholds = [0.34, 0.34, 0.34, 0.34]
    best = _score_rows(_kept(rows, pred, thresholds), views, by_file, files)
    print("start", json.dumps({"thresholds": thresholds, "pq": best["pq"]}), flush=True)
    for _ in range(2):
        for bin_id in range(4):
            chosen = thresholds[bin_id]
            chosen_pq = best["pq"]
            for value in THRESHOLDS:
                trial = list(thresholds)
                trial[bin_id] = value
                scored = _score_rows(_kept(rows, pred, trial), views, by_file, files)
                print(
                    "trial",
                    json.dumps({"bin": bin_id, "thr": value, "pq": scored["pq"]}),
                    flush=True,
                )
                if scored["pq"] > chosen_pq:
                    chosen = value
                    chosen_pq = scored["pq"]
                    best = scored
            thresholds[bin_id] = chosen
    import pickle

    model_path = FEATURE_DIR / "gbm.pkl"
    model_path.write_bytes(pickle.dumps(model))
    payload = {
        "thresholds": thresholds,
        "tune": best,
        "coco_champion_fold0": COCO_CHAMPION_FOLD0,
        "keep_bar": COCO_CHAMPION_FOLD0 + 0.01,
        "literal_bar": 0.4741684688395591 + 0.01,
        "features": list(FEATURES),
        "target": "match_fraction * mean IoU of annotators that overlap the candidate above 0.5",
        "oof": "fold 4 from the folds 1-3 model; folds 1-3 from their own held-out models",
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    TUNE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print("DONE phase_a_tune", json.dumps(payload), flush=True)


def fold0() -> None:
    if FOLD0_PATH.exists():
        raise SystemExit(f"fold 0 already scored: {FOLD0_PATH}")
    if not TUNE_PATH.exists():
        raise SystemExit(f"missing {TUNE_PATH}")
    saved = json.loads(TUNE_PATH.read_text())
    path = FEATURE_DIR / "fold0.jsonl"
    if not path.exists():
        build({0}, path)
    rows = _load_rows(path)
    import pickle

    model = pickle.loads((FEATURE_DIR / "gbm.pkl").read_bytes())
    pred = _predict(model, rows)
    kept = _kept(rows, pred, saved["thresholds"])
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    files = _files_for_folds(fold_of, {0})
    scored = _score_rows(kept, views, by_file, files)
    payload = {
        "fold": 0,
        "thresholds": saved["thresholds"],
        **scored,
        "coco_champion_pq": COCO_CHAMPION_FOLD0,
        "delta_vs_coco_champion": scored["pq"] - COCO_CHAMPION_FOLD0,
        "beats_coco_champion_by_0.01": bool(scored["pq"] > COCO_CHAMPION_FOLD0 + 0.01),
        "beats_literal_0.4842": bool(scored["pq"] > 0.4841684688395591),
        "test_labels_used": False,
        "kaggle_submit_called": False,
        "do_not_submit": True,
    }
    FOLD0_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print("DONE phase_a_fold0", json.dumps(payload), flush=True)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--fold0":
        fold0()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--build-only":
        build({1, 2, 3, 4}, FEATURE_DIR / "folds14.jsonl")
        return
    build({1, 2, 3, 4}, FEATURE_DIR / "folds14.jsonl")
    tune()


if __name__ == "__main__":
    main()
