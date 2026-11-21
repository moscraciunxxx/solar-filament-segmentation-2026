#!/usr/bin/env python3
"""One fold-0 vote of the Mask R-CNN with the champion. No submit.

The rule is predeclared, not picked on fold 0: IoU 0.5, and a detection is
kept only when both models have a mask in the cluster. The kept mask is the
champion mask. Fold 0 is scored once.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, IMAGE_SIZE, SEED
from filament.rle import decode_rle, encode_rle

CHAMPION = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_ensemble3_tta_fold0_pred.csv"
OURS = FAST_ARTIFACTS / "maskrcnn_convnext_tiny_1536_raw_medoid_fold0_pred.csv"
OUT = ARTIFACTS / "maskrcnn_convnext_tiny_1536_raw_medoid_vote_champion_fold0.json"
CHAMPION_PQ = 0.4741684688395591
IOU = 0.5


def _groups(frame: pd.DataFrame) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = defaultdict(list)
    for filament_id, rle in zip(frame["filament_id"], frame["segmentation_rle"]):
        stem = str(filament_id).rsplit("_", 1)[0]
        out[stem].append(decode_rle(rle))
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"fold 0 already scored: {OUT}")
    champion = _groups(pd.read_csv(CHAMPION))
    ours = _groups(pd.read_csv(OURS))
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    files = sorted(name for name, fold in fold_of.items() if fold == 0)
    frames = []
    for file_name in files:
        stem = Path(file_name).stem
        champ = champion.get(stem, [])
        extra = ours.get(stem, [])
        used = [False] * len(extra)
        kept = []
        for mask in champ:
            hit = None
            for j, other in enumerate(extra):
                if used[j]:
                    continue
                if _iou(mask, other) >= IOU:
                    hit = j
                    break
            if hit is None:
                continue
            used[hit] = True
            kept.append(mask)
        frames.append(pred_to_df(stem, [encode_rle(m) for m in kept]))
    pred = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    scored = score_pq(gt_annotations_to_df(gt_records_for_files(views, by_file, files)), pred)
    pred_csv = FAST_ARTIFACTS / "maskrcnn_convnext_tiny_1536_raw_medoid_vote_champion_fold0_pred.csv"
    pred.to_csv(pred_csv, index=False)
    payload = {
        "family": "maskrcnn_convnext_vote_champion",
        "rule": "IoU 0.5 agreement. Keep the champion mask only when the Mask R-CNN also overlaps it. Predeclared. Not tuned on fold 0.",
        "fold": 0,
        "pq": scored.pq,
        "tp": scored.tp,
        "fp": scored.fp,
        "fn": scored.fn,
        "n_pred": int(len(pred)),
        "mean_tp_iou": scored.mean_tp_iou,
        "champion_pq": CHAMPION_PQ,
        "delta_vs_champion": scored.pq - CHAMPION_PQ,
        "image_size_checked": IMAGE_SIZE,
        "pred_csv": str(pred_csv),
        "test_labels_used": False,
        "kaggle_submit_called": False,
        "do_not_submit": True,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print("DONE maskrcnn_vote", json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
