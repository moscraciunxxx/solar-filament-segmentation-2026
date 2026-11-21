#!/usr/bin/env python3
"""Score the faint-oversample fine-tune of eroded YOLO11s at 2048.

Confidence stays at 0.35. The whole fold is scored. Does not read test
labels. Does not submit. Refuses to run while training still owns the GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.infer import pick_device, run_yolo_infer
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from gate_unet_family import live_mps_owner

PARENT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
WEIGHTS = WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048_faint_e30" / "weights" / "best.pt"
OUT = ARTIFACTS / "stream_faint_e30_fold0.json"
PRED_OUT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_faint_e30_fold0_conf0.35_pred.csv"
EXPECTED_PQ = 0.4387235694048039
PUBLIC_GAP = 0.069
PUBLIC_BAR = 0.6
CONF = 0.35
IMGSZ = 2048


def should_promote(pq: float, gap: float = PUBLIC_GAP, bar: float = PUBLIC_BAR) -> bool:
    return (pq - gap) > bar


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
    owner = live_mps_owner()
    if owner:
        raise SystemExit(f"refusing to score while {owner} owns MPS")
    if not WEIGHTS.is_file():
        raise SystemExit(f"missing weights: {WEIGHTS}")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(name for name, fold in fold_of.items() if fold == 0)
    if len(val_files) != 142:
        raise SystemExit(f"fold 0 should be 142 images, got {len(val_files)}")
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    parent = pd.read_csv(PARENT)
    first = _row(score_pq(gt_df, parent), len(parent))
    second = _row(score_pq(gt_df, parent), len(parent))
    if first != second:
        raise SystemExit(f"parent scored twice and differed: {first} vs {second}")
    if abs(first["pq"] - EXPECTED_PQ) > 1e-6:
        raise SystemExit(f"parent rescore drifted: {first}")

    from ultralytics import YOLO

    device = pick_device("auto")
    model = YOLO(str(WEIGHTS))
    frames = []
    for name in val_files:
        rles = run_yolo_infer(
            model,
            TRAIN_IMG_DIR / name,
            conf=CONF,
            imgsz=IMGSZ,
            device=device,
        )
        frames.append(pred_to_df(Path(name).stem, rles))
    pred = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["filament_id", "segmentation_rle"]
    )
    PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(PRED_OUT, index=False)
    scored = _row(score_pq(gt_df, pred), len(pred))
    formula = scored["mean_tp_iou"] * scored["tp"] / (
        scored["tp"] + 0.5 * scored["fp"] + 0.5 * scored["fn"]
    )
    if scored["tp"] and abs(formula - scored["pq"]) > 1e-9:
        raise SystemExit(f"formula check failed: {formula} vs {scored['pq']}")
    forecast = scored["pq"] - PUBLIC_GAP
    payload = {
        "split": "grouped year x count fold 0",
        "seed": SEED,
        "hypothesis": "continue the faint fine-tune for 15 more epochs from its best checkpoint, same data and learning rate",
        "weights": str(WEIGHTS),
        "conf": CONF,
        "imgsz": IMGSZ,
        "n_val": len(val_files),
        "parent": first,
        "parent_second_call": second,
        "faint_e30": scored,
        "formula_pq": formula,
        "pred_csv": str(PRED_OUT),
        "lift_vs_2048": scored["pq"] - first["pq"],
        "fp_delta": scored["fp"] - first["fp"],
        "fn_delta": scored["fn"] - first["fn"],
        "public_gap_used": PUBLIC_GAP,
        "public_forecast": forecast,
        "promote": should_promote(scored["pq"]),
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
