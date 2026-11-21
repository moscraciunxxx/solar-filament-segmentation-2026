#!/usr/bin/env python3
"""One pre-registered confidence on the frozen 2048 weights.

Hypothesis: the 640 fold-0 misses are low-score detections of this same
model, not objects it never proposes. Confidence 0.15 is fixed. No sweep.
Does not read test labels. Does not submit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.dark_delta import MIN_DELTA, dark_enough
from filament.disjoint_add import disjoint_add
from filament.infer import pick_device, run_yolo_infer
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.rle import decode_rle

WEIGHTS = WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048" / "weights" / "best.pt"
PARENT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
OUT = ARTIFACTS / "stream_conf015_fold0.json"
LOCKED_CONF = 0.15
LOCKED_IMGSZ = 2048
EXPECTED_PQ = 0.4387235694048039


def _row(br, n_pred: int) -> dict:
    return {
        "pq": br.pq,
        "tp": br.tp,
        "fp": br.fp,
        "fn": br.fn,
        "mean_tp_iou": br.mean_tp_iou,
        "n_pred": n_pred,
        "n_annotator_images": br.n_annotator_images,
    }


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(name for name, fold in fold_of.items() if fold == 0)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    parent = pd.read_csv(PARENT)
    parent_row = _row(score_pq(gt_df, parent), len(parent))
    if abs(parent_row["pq"] - EXPECTED_PQ) > 1e-6:
        raise SystemExit(f"parent rescore drifted: {parent_row}")

    from ultralytics import YOLO

    device = pick_device("mps")
    model = YOLO(str(WEIGHTS))
    frames = []
    for name in tqdm(val_files, desc="conf015"):
        path = TRAIN_IMG_DIR / name
        rles = run_yolo_infer(
            model, path, conf=LOCKED_CONF, imgsz=LOCKED_IMGSZ, merge_gap=0, device=device
        )
        if rles:
            frames.append(pred_to_df(Path(name).stem, rles))
    pred = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["filament_id", "segmentation_rle"]
    )
    low = _row(score_pq(gt_df, pred), len(pred))
    lift = low["pq"] - parent_row["pq"]

    def _by_stem(frame: pd.DataFrame) -> dict[str, list[str]]:
        stems = frame["filament_id"].str.split("_", n=1).str[0]
        grouped: dict[str, list[str]] = {}
        for stem, rle in zip(stems, frame["segmentation_rle"], strict=True):
            grouped.setdefault(stem, []).append(rle)
        return grouped

    parent_g = _by_stem(parent)
    low_g = _by_stem(pred)
    gated_rows: dict[str, list[str]] = {}
    n_disjoint = 0
    n_dark = 0
    for name in val_files:
        stem = Path(name).stem
        parent_rles = parent_g.get(stem, [])
        merged, n_add = disjoint_add(parent_rles, low_g.get(stem, []))
        n_disjoint += n_add
        image = cv2.imread(str(TRAIN_IMG_DIR / name), cv2.IMREAD_GRAYSCALE)
        survivors = []
        for rle in merged[len(parent_rles):]:
            if dark_enough(decode_rle(rle), image, MIN_DELTA):
                survivors.append(rle)
        n_dark += len(survivors)
        gated_rows[stem] = list(parent_rles) + survivors
    gated_frames = [
        pred_to_df(stem, rles) for stem, rles in gated_rows.items() if rles
    ]
    gated_df = pd.concat(gated_frames, ignore_index=True)
    gated = _row(score_pq(gt_df, gated_df), len(gated_df))
    gated_lift = gated["pq"] - parent_row["pq"]
    payload = {
        "split": "grouped year x count fold 0",
        "seed": SEED,
        "weights": str(WEIGHTS),
        "conf": LOCKED_CONF,
        "imgsz": LOCKED_IMGSZ,
        "chosen_after_score": False,
        "parent_conf_0.35": parent_row,
        "conf_0.15": low,
        "lift_vs_0.35": lift,
        "disjoint_dark_adds": n_disjoint,
        "disjoint_dark_kept": n_dark,
        "disjoint_dark": gated,
        "disjoint_dark_lift": gated_lift,
        "disjoint_dark_promote": bool(gated_lift >= 0.01 and gated["fp"] < 2 * parent_row["fp"]),
        "promote": bool(lift >= 0.01 and low["fp"] < 2 * parent_row["fp"]),
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
