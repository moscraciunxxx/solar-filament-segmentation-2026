#!/usr/bin/env python3
"""Score the ConvNeXt Mask R-CNN once on fold 0. No submit. No test labels."""

from __future__ import annotations

import importlib.util
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
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, IMAGE_SIZE, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.rle import encode_rle

spec = importlib.util.spec_from_file_location(
    "train_maskrcnn_convnext", ROOT / "scripts" / "train_maskrcnn_convnext.py"
)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)

FAMILY = trainer.FAMILY
WEIGHTS = WEIGHTS_DIR / FAMILY / "best.pt"
OUT = ARTIFACTS / f"{FAMILY}_fold0.json"
CHAMPION_PQ = 0.4741684688395591
SCORE_THRESH = 0.5


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"fold 0 already scored: {OUT}")
    if not WEIGHTS.is_file():
        raise SystemExit(f"missing {WEIGHTS}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = trainer.build_model()
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(name for name, fold in fold_of.items() if fold == 0)
    if len(val_files) != 142:
        raise SystemExit(f"fold 0 has {len(val_files)} images")
    frames = []
    for index, file_name in enumerate(val_files, start=1):
        image = cv2.imread(str(TRAIN_IMG_DIR / file_name), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(file_name)
        if image.shape[0] != IMAGE_SIZE or image.shape[1] != IMAGE_SIZE:
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(image[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            pred = model([tensor.to(device)])[0]
        rles = []
        scores = pred["scores"].detach().cpu()
        masks = pred["masks"].detach().cpu()
        for score, mask in zip(scores, masks):
            if float(score) < SCORE_THRESH:
                continue
            binary = (mask[0].numpy() > 0.5).astype(np.uint8)
            if binary.shape != (IMAGE_SIZE, IMAGE_SIZE):
                binary = cv2.resize(binary, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
            if int(binary.sum()) < 96:
                continue
            rles.append(encode_rle(binary))
        frames.append(pred_to_df(Path(file_name).stem, rles))
        if index % 20 == 0:
            print(f"fold0 {index}/{len(val_files)}", flush=True)
    pred_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    scored = score_pq(gt_annotations_to_df(gt_records_for_files(views, by_file, val_files)), pred_df)
    pred_csv = FAST_ARTIFACTS / f"{FAMILY}_fold0_pred.csv"
    pred_df.to_csv(pred_csv, index=False)
    payload = {
        "family": FAMILY,
        "weights": str(WEIGHTS),
        "checkpoint_epoch": state.get("epoch"),
        "labels": "raw medoid",
        "imgsz": trainer.IMGSZ,
        "score_thresh": SCORE_THRESH,
        "fold": 0,
        "pq": scored.pq,
        "tp": scored.tp,
        "fp": scored.fp,
        "fn": scored.fn,
        "n_pred": int(len(pred_df)),
        "mean_tp_iou": scored.mean_tp_iou,
        "champion_pq": CHAMPION_PQ,
        "delta_vs_champion": scored.pq - CHAMPION_PQ,
        "beats_champion_by_0.01": bool(scored.pq > CHAMPION_PQ + 0.01),
        "pred_csv": str(pred_csv),
        "test_labels_used": False,
        "kaggle_submit_called": False,
        "do_not_submit": True,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print("DONE maskrcnn_fold0", json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
