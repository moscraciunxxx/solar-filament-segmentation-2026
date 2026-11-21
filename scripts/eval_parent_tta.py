#!/usr/bin/env python3
"""Parent yolo11s erode-gt 2048 TTA.

Same weights only. Geometric views (flips and 90-degree rotations) and
inference sizes 1792, 2048, 2304. Fusion IoU and vote count are chosen on
grouped folds 1-4, then fold 0 is scored once. Never submits. Never reads
test labels.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.infer import _collect_yolo_instances, yolo_masks_to_instances
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, IMAGE_SIZE, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.rle import decode_rle, encode_rle

WEIGHTS = WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048" / "weights" / "best.pt"
CACHE = FAST_ARTIFACTS / "tta_parent_2048"
CONF = 0.35
YOLO_IOU = 0.60
SCALES = (1792, 2048, 2304)
PARENT = {
    "pq": 0.4387235694048039,
    "tp": 949,
    "fp": 410,
    "fn": 640,
    "n_pred": 843,
    "mean_tp_iou": 0.6814315503716343,
}
PARENT_PQ = PARENT["pq"]
# Predeclared grid. Folds 1-4 are the parent training set, so those scores
# only rank fusion settings. They are not a generalization estimate.
# If several settings are within 0.005 PQ of the best, keep the most
# conservative: higher vote count, then higher IoU threshold. Every cell
# uses the same 8 views x 3 scales, so view count is not a grid axis.
IOU_GRID = (0.3, 0.5)
VOTE_GRID = (2, 4, 8)
TIE_PQ = 0.005


def _views():
    """Eight disk-symmetric views and their mask inverses."""
    r90 = cv2.ROTATE_90_CLOCKWISE
    r270 = cv2.ROTATE_90_COUNTERCLOCKWISE
    r180 = cv2.ROTATE_180

    def rot(img, code):
        return cv2.rotate(img, code)

    def flip(img, axis):
        return cv2.flip(img, axis)

    return [
        ("id", lambda im: im, lambda m: m),
        ("r90", lambda im: rot(im, r90), lambda m: rot(m, r270)),
        ("r180", lambda im: rot(im, r180), lambda m: rot(m, r180)),
        ("r270", lambda im: rot(im, r270), lambda m: rot(m, r90)),
        ("h", lambda im: flip(im, 1), lambda m: flip(m, 1)),
        ("v", lambda im: flip(im, 0), lambda m: flip(m, 0)),
        ("h_r90", lambda im: rot(flip(im, 1), r90), lambda m: flip(rot(m, r270), 1)),
        ("v_r90", lambda im: rot(flip(im, 0), r90), lambda m: flip(rot(m, r270), 0)),
    ]


VIEWS = _views()


def _cache_path(file_name: str) -> Path:
    return CACHE / f"{Path(file_name).stem}.json"


def _predict_views(model, file_name: str, image_dir: Path | None = None, device: str = "mps") -> dict[str, list[str]]:
    image = cv2.imread(str((image_dir or TRAIN_IMG_DIR) / file_name), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(file_name)
    if image.shape[0] != IMAGE_SIZE or image.shape[1] != IMAGE_SIZE:
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    out: dict[str, list[str]] = {}
    for scale in SCALES:
        for name, forward, inverse in VIEWS:
            view = forward(image)
            results = model.predict(
                source=view,
                conf=CONF,
                iou=YOLO_IOU,
                imgsz=scale,
                retina_masks=True,
                augment=False,
                verbose=False,
                device=device,
            )
            instances = _collect_yolo_instances(results[0] if results else None)
            rles = []
            for mask, _score in instances:
                if mask.shape != view.shape[:2]:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (view.shape[1], view.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                restored = inverse((mask > 0).astype(np.uint8))
                if restored.shape != (IMAGE_SIZE, IMAGE_SIZE):
                    restored = cv2.resize(
                        restored, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST
                    )
                if int(restored.sum()) >= 96:
                    rles.append(encode_rle(restored))
            out[f"{scale}_{name}"] = rles
    return out


def _load_or_infer(model, file_name: str) -> dict[str, list[str]]:
    path = _cache_path(file_name)
    if path.exists():
        return json.loads(path.read_text())
    payload = _predict_views(model, file_name)
    path.write_text(json.dumps(payload))
    return payload


def _decode_all(payload: dict[str, list[str]]) -> list[np.ndarray]:
    masks = []
    for rles in payload.values():
        for rle in rles:
            masks.append(decode_rle(rle))
    return masks


def _box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def fuse_masks(masks: list[np.ndarray], iou_thr: float, min_votes: int) -> list[np.ndarray]:
    if not masks:
        return []
    boxes = [_box(m) for m in masks]
    areas = [int(m.sum()) for m in masks]
    order = sorted(range(len(masks)), key=lambda i: areas[i], reverse=True)
    used = [False] * len(masks)
    out: list[np.ndarray] = []
    for i in order:
        if used[i] or boxes[i] is None:
            continue
        group = [i]
        used[i] = True
        ax0, ay0, ax1, ay1 = boxes[i]
        for j in order:
            if used[j] or boxes[j] is None:
                continue
            bx0, by0, bx1, by1 = boxes[j]
            if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
                continue
            inter = np.logical_and(masks[i], masks[j]).sum()
            union = np.logical_or(masks[i], masks[j]).sum()
            if union and (inter / union) >= iou_thr:
                group.append(j)
                used[j] = True
        if len(group) < min_votes:
            continue
        vote = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        for k in group:
            vote += masks[k]
        merged = (vote >= max(1, (len(group) + 1) // 2)).astype(np.uint8)
        if int(merged.sum()) >= 96:
            out.append(merged)
    return yolo_masks_to_instances(out, merge_gap=0)


def select_frozen(rows: list[dict]) -> dict:
    """Pick the fold-0 setting from the in-sample tune grid."""
    best_pq = max(float(r["pq"]) for r in rows)
    band = [r for r in rows if best_pq - float(r["pq"]) <= TIE_PQ]
    chosen = max(band, key=lambda r: (int(r["min_votes"]), float(r["iou_thr"])))
    out = dict(chosen)
    out["selection"] = (
        "highest min_votes, then highest iou_thr, among folds 1-4 PQ within "
        "0.005 of the best. View count is fixed at 8 geometries x 3 scales."
    )
    out["best_tune_pq"] = best_pq
    out["n_in_band"] = len(band)
    out["tune_is_in_sample"] = True
    return out


def _frames(files: list[str], payloads: dict[str, dict], iou_thr: float, min_votes: int) -> pd.DataFrame:
    frames = []
    for file_name in files:
        masks = fuse_masks(_decode_all(payloads[file_name]), iou_thr, min_votes)
        frames.append(pred_to_df(Path(file_name).stem, [encode_rle(m) for m in masks]))
    if not frames:
        return pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    if not WEIGHTS.exists():
        raise SystemExit(f"missing {WEIGHTS}")
    CACHE.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    tune_files = sorted(fn for fn, fold in fold_of.items() if fold != 0)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    print(f"tune_images {len(tune_files)} fold0_images {len(val_files)} views {len(VIEWS)} scales {SCALES}", flush=True)

    t0 = time.perf_counter()
    _load_or_infer(model, tune_files[0])
    per = time.perf_counter() - t0
    remaining = per * (len(tune_files) - 1 + len(val_files))
    print(
        f"one_image_s {per:.1f} eta_infer_h {remaining / 3600:.2f} "
        f"views_per_image {len(VIEWS) * len(SCALES)}",
        flush=True,
    )

    tune_payloads: dict[str, dict] = {tune_files[0]: json.loads(_cache_path(tune_files[0]).read_text())}
    for i, file_name in enumerate(tune_files[1:], start=2):
        tune_payloads[file_name] = _load_or_infer(model, file_name)
        if i % 10 == 0:
            print(f"tune_cached {i}/{len(tune_files)}", flush=True)

    gt_tune = gt_annotations_to_df(gt_records_for_files(views, by_file, tune_files))
    rows = []
    for iou_thr in IOU_GRID:
        for votes in VOTE_GRID:
            pred = _frames(tune_files, tune_payloads, iou_thr, votes)
            scored = score_pq(gt_tune, pred)
            row = {
                "iou_thr": iou_thr,
                "min_votes": votes,
                "pq": scored.pq,
                "tp": scored.tp,
                "fp": scored.fp,
                "fn": scored.fn,
                "mean_tp_iou": scored.mean_tp_iou,
                "n_pred": int(len(pred)),
            }
            rows.append(row)
            print("tune", json.dumps(row), flush=True)
    _write_tune(rows)
    _score_fold0(model, views, by_file, val_files, json.loads((ARTIFACTS / "parent_tta_tune_folds14.json").read_text())["frozen"])


def _write_tune(rows: list[dict]) -> dict:
    best = select_frozen(rows)
    tune_path = ARTIFACTS / "parent_tta_tune_folds14.json"
    tune_path.write_text(
        json.dumps(
            {
                "grid": rows,
                "frozen": best,
                "max_pq_row": max(rows, key=lambda r: r["pq"]),
                "tune_split": "grouped folds 1-4, in-sample for the parent",
                "test_labels_used": False,
            },
            indent=2,
        )
        + "\n"
    )
    print("frozen", json.dumps(best), flush=True)
    return best


def _score_fold0(model, views, by_file, val_files: list[str], best: dict) -> None:
    val_payloads = {fn: _load_or_infer(model, fn) for fn in val_files}
    pred = _frames(val_files, val_payloads, best["iou_thr"], best["min_votes"])
    pred_csv = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_tta_fold0_pred.csv"
    pred.to_csv(pred_csv, index=False)
    scored = score_pq(gt_annotations_to_df(gt_records_for_files(views, by_file, val_files)), pred)
    payload = {
        "family": "yolo11s_seg_erode_gt_2048_tta",
        "weights": str(WEIGHTS),
        "conf": CONF,
        "scales": list(SCALES),
        "n_geom_views": len(VIEWS),
        "frozen_on": "grouped folds 1-4, in-sample rank only",
        "frozen": best,
        "fold": 0,
        "pq": scored.pq,
        "tp": scored.tp,
        "fp": scored.fp,
        "fn": scored.fn,
        "mean_tp_iou": scored.mean_tp_iou,
        "n_pred": int(len(pred)),
        "parent": PARENT,
        "parent_pq": PARENT_PQ,
        "delta_pq": scored.pq - PARENT_PQ,
        "delta_tp": scored.tp - PARENT["tp"],
        "pred_csv": str(pred_csv),
        "test_labels_used": False,
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "promote_look": bool(scored.pq >= PARENT_PQ + 0.015 and scored.tp > PARENT["tp"]) or bool(scored.pq >= 0.46),
        "beats_parent_fold0": bool(scored.pq > PARENT_PQ),
    }
    text = json.dumps(payload, indent=2) + "\n"
    (ARTIFACTS / "yolo11s_seg_erode_gt_2048_tta_fold0_official.json").write_text(text)
    print(text, flush=True)


def fold0_only() -> None:
    """Score fold 0 from an existing tune grid. Does not rerun the grid."""
    tune_path = ARTIFACTS / "parent_tta_tune_folds14.json"
    if not tune_path.exists():
        raise SystemExit(f"missing {tune_path}")
    saved = json.loads(tune_path.read_text())
    best = _write_tune(saved["grid"])
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    _score_fold0(model, views, by_file, val_files, best)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fold0-only":
        fold0_only()
    else:
        main()
