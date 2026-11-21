#!/usr/bin/env python3
"""One-shot Mask R-CNN rescore. Does not train. Does not submit. Does not sweep."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_maskrcnn_r50_1024_pilot as pilot
from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED

THRESH = 0.70


def main() -> None:
    ckpt = pilot.DEST / "last.pt"
    if not ckpt.exists():
        raise SystemExit(f"missing {ckpt}")
    pilot.SCORE_THRESH = THRESH
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    device = torch.device("mps")
    model = pilot._build_model().to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    frames = []
    for file_name in val_files:
        rles = pilot._predict_rles(model, device, file_name)
        frames.append(pred_to_df(Path(file_name).stem, rles))
        print(f"score {file_name} n={len(rles)}", flush=True)
    pred_df = pd.concat(frames, ignore_index=True)
    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    pred_csv = FAST_ARTIFACTS / "maskrcnn_r50_1024_fold0_conf0.70_pred.csv"
    pred_df.to_csv(pred_csv, index=False)
    scored = score_pq(gt_annotations_to_df(gt_records_for_files(views, by_file, val_files)), pred_df)
    payload = {
        "family": "maskrcnn_r50_1024",
        "why": "Single check that the 1384 false positives at 0.35 are low-score detections. Not a threshold sweep.",
        "score_thresh": THRESH,
        "pq": scored.pq,
        "tp": scored.tp,
        "fp": scored.fp,
        "fn": scored.fn,
        "mean_tp_iou": scored.mean_tp_iou,
        "n_pred": int(len(pred_df)),
        "pred_csv": str(pred_csv),
        "compared_to_conf_0.35": {"pq": 0.30961380994226523, "tp": 982, "fp": 1384, "fn": 609},
        "slot1_pq": 0.4365241382777018,
        "test_labels_used": False,
        "do_not_submit": True,
        "kaggle_submit_called": False,
    }
    text = json.dumps(payload, indent=2) + "\n"
    (ARTIFACTS / "maskrcnn_r50_1024_fold0_conf0.70_official.json").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
