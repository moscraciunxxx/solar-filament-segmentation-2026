#!/usr/bin/env python3
"""Score CondInst alone on locked fold 0. No merge. No test labels. No submit."""

from __future__ import annotations

import argparse
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
from filament.condinst import CondInst, predict_maps, should_promote
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, IMAGE_SIZE, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR

# IMGSZ lives next to the train script constant. Importing the train script
# would re-parse nothing, but the size is fixed here so eval cannot drift.
IMGSZ = 1024
CONF = 0.35
FAMILY = "condinst_r18_1024"
PARENT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_image(path: Path) -> torch.Tensor:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    small = cv2.resize(gray, (IMGSZ, IMGSZ), interpolation=cv2.INTER_AREA)
    rgb = np.stack([small, small, small], axis=-1).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return torch.from_numpy(rgb.transpose(2, 0, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default=FAMILY)
    args = parser.parse_args()
    family = args.family
    weights = WEIGHTS_DIR / family / "best.pt"
    if not weights.is_file():
        raise SystemExit(f"missing weights: {weights}")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(name for name, fold in fold_of.items() if fold == 0)
    if len(val_files) != 142:
        raise SystemExit(f"fold 0 should be 142 images, got {len(val_files)}")
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    parent = pd.read_csv(PARENT)
    parent_score = score_pq(gt_df, parent)
    if abs(parent_score.pq - 0.4387235694048039) > 1e-6:
        raise SystemExit(f"parent rescore drifted: {parent_score.pq}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CondInst(backbone="resnet18", pretrained=False)
    state = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    frames = []
    for name in val_files:
        image = _load_image(TRAIN_IMG_DIR / name).to(device)
        masks = predict_maps(
            model,
            image,
            conf=CONF,
            imgsz=IMGSZ,
            out_size=IMAGE_SIZE,
        )
        rles = []
        from filament.rle import encode_rle

        for mask in masks:
            arr = mask.numpy()
            if int(arr.sum()) <= 0:
                continue
            rles.append(encode_rle(arr))
        frames.append(pred_to_df(Path(name).stem, rles))
        print(f"{name} n={len(rles)}", flush=True)
    pred = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    pred_out = FAST_ARTIFACTS / f"{family}_fold0_conf035_pred.csv"
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(pred_out, index=False)
    scored = score_pq(gt_df, pred)
    same_hit_set = False
    if scored.pq >= 0.47 and scored.tp == parent_score.tp:
        from filament.metric import overlap_matrices
        from filament.rle import rles_to_layers

        def _hits(frame: pd.DataFrame) -> set[str]:
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
                hit = (iou > 0.5).any(dim=1)
                for filament_id, matched in zip(gt_rows["filament_id"].tolist(), hit.tolist()):
                    if matched:
                        found.add(filament_id)
            return found

        same_hit_set = _hits(pred) == _hits(parent)
    promote = should_promote(scored.pq, scored.tp, same_hit_set=same_hit_set)
    payload = {
        "split": "grouped year x count fold 0",
        "seed": SEED,
        "hypothesis": (
            "CondInst body-positive class and centerness, ResNet-18 stride 8, medoid labels, imgsz 1024, conf 0.35, scored alone"
            if family.endswith("bodypos")
            else "CondInst dynamic-filter head, ResNet-18 stride 8, medoid labels, imgsz 1024, conf 0.35, scored alone"
        ),
        "weights": str(weights),
        "conf": CONF,
        "imgsz": IMGSZ,
        "n_val": len(val_files),
        "parent": {
            "pq": parent_score.pq,
            "tp": parent_score.tp,
            "fp": parent_score.fp,
            "fn": parent_score.fn,
            "mean_tp_iou": parent_score.mean_tp_iou,
        },
        "condinst": {
            "pq": scored.pq,
            "tp": scored.tp,
            "fp": scored.fp,
            "fn": scored.fn,
            "mean_tp_iou": scored.mean_tp_iou,
            "n_pred": int(len(pred)),
            "n_annotator_images": scored.n_annotator_images,
        },
        "formula_pq": scored.pq,
        "pred_csv": str(pred_out),
        "public_gap_used": 0.069,
        "public_forecast": scored.pq - 0.069,
        "promote": promote,
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    out = ARTIFACTS / f"{family}_fold0.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
