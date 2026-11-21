#!/usr/bin/env python3
"""Pick YOLO confidence by official PQ on the grouped holdout, not by public LB."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.infer import infer_yolo_image
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR


def main() -> None:
    weights = WEIGHTS_DIR / "yolo11s_seg" / "weights" / "best.pt"
    from ultralytics import YOLO

    model = YOLO(str(weights))
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)[:40]
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))

    rows = []
    for conf in (0.12, 0.18, 0.25, 0.35, 0.45):
        frames = []
        for file_name in tqdm(val_files, desc=f"conf={conf}"):
            rles = infer_yolo_image(model, TRAIN_IMG_DIR / file_name, conf=conf)
            frames.append(pred_to_df(Path(file_name).stem, rles))
        pred_df = pd.concat(frames, ignore_index=True)
        br = score_pq(gt_df, pred_df)
        rows.append(
            {
                "conf": conf,
                "pq": br.pq,
                "tp": br.tp,
                "fp": br.fp,
                "fn": br.fn,
                "mean_tp_iou": br.mean_tp_iou,
                "n_pred": int(len(pred_df)),
            }
        )
        print(rows[-1])

    rows.sort(key=lambda r: r["pq"], reverse=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "yolo_conf_sweep.json").write_text(json.dumps(rows, indent=2))
    print("BEST", rows[0])


if __name__ == "__main__":
    main()
