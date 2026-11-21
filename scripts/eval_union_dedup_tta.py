#!/usr/bin/env python3
"""8x3 TTA for the union-dedup yolo11s. Min votes ranked on folds 1-4.

Geometry and scales match the frozen single-model TTA. IoU stays 0.5.
Vote counts 2, 4, and 8 are ranked on folds 1-4. Fold 0 is scored once.
Does not submit. Does not read test labels. Does not touch the parent cache.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.metric import gt_annotations_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, WEIGHTS_DIR

spec = importlib.util.spec_from_file_location("eval_parent_tta", ROOT / "scripts" / "eval_parent_tta.py")
tta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tta)

FAMILY = "yolo11s_seg_union_dedup_2048"
tta.WEIGHTS = WEIGHTS_DIR / FAMILY / "weights" / "best.pt"
tta.CACHE = FAST_ARTIFACTS / "tta_union_dedup_2048"
tta.IOU_GRID = (0.5,)
tta.VOTE_GRID = (2, 4, 8)
OUT = ARTIFACTS / f"{FAMILY}_tta_fold0.json"
TUNE = ARTIFACTS / f"{FAMILY}_tta_tune_folds14.json"
SINGLE_TTA_PQ = 0.4651884607358067


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"fold 0 already scored: {OUT}")
    if not tta.WEIGHTS.exists():
        raise SystemExit(f"missing {tta.WEIGHTS}")
    tta.CACHE.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    model = YOLO(str(tta.WEIGHTS))
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    tune_files = sorted(fn for fn, fold in fold_of.items() if fold != 0)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    if len(tune_files) != 565 or len(val_files) != 142:
        raise SystemExit(f"unexpected split {len(tune_files)} {len(val_files)}")

    t0 = time.perf_counter()
    first = tta._load_or_infer(model, tune_files[0])
    per = time.perf_counter() - t0
    print(
        f"one_image_s {per:.1f} eta_infer_h {per * (len(tune_files) + len(val_files) - 1) / 3600:.2f}",
        flush=True,
    )
    payloads = {tune_files[0]: first}
    for i, file_name in enumerate(tune_files[1:], start=2):
        payloads[file_name] = tta._load_or_infer(model, file_name)
        if i % 25 == 0:
            print(f"tune_cached {i}/{len(tune_files)}", flush=True)
    gt_tune = gt_annotations_to_df(gt_records_for_files(views, by_file, tune_files))
    rows = []
    for votes in tta.VOTE_GRID:
        pred = tta._frames(tune_files, payloads, 0.5, votes)
        scored = score_pq(gt_tune, pred)
        row = {
            "iou_thr": 0.5,
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
    frozen = tta.select_frozen(rows)
    TUNE.write_text(json.dumps({"grid": rows, "frozen": frozen, "test_labels_used": False}, indent=2) + "\n")
    print("frozen", json.dumps(frozen), flush=True)

    val_payloads = {fn: tta._load_or_infer(model, fn) for fn in val_files}
    pred = tta._frames(val_files, val_payloads, frozen["iou_thr"], frozen["min_votes"])
    pred_csv = FAST_ARTIFACTS / f"{FAMILY}_tta_fold0_pred.csv"
    pred.to_csv(pred_csv, index=False)
    scored = score_pq(gt_annotations_to_df(gt_records_for_files(views, by_file, val_files)), pred)
    payload = {
        "family": f"{FAMILY}_tta",
        "weights": str(tta.WEIGHTS),
        "scales": list(tta.SCALES),
        "n_geom_views": len(tta.VIEWS),
        "frozen": frozen,
        "fold": 0,
        "pq": scored.pq,
        "tp": scored.tp,
        "fp": scored.fp,
        "fn": scored.fn,
        "n_pred": int(len(pred)),
        "mean_tp_iou": scored.mean_tp_iou,
        "single_model_tta_pq": SINGLE_TTA_PQ,
        "delta_vs_single_tta": scored.pq - SINGLE_TTA_PQ,
        "beats_single_tta_by_0.01": bool(scored.pq > SINGLE_TTA_PQ + 0.01),
        "pred_csv": str(pred_csv),
        "test_labels_used": False,
        "kaggle_submit_called": False,
        "do_not_submit": True,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print("DONE union_dedup_tta", json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
