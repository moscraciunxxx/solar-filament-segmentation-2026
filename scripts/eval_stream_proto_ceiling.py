#!/usr/bin/env python3
"""Score full medoid outlines after a prototype-grid round trip.

This is a ceiling for what a stride-4 grid and a stride-2 grid can
represent on locked fold 0. The masks are fold-0 medoid polygons, so
nothing here is a submission. Does not read Kaggle test labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco, medoid_view_id, rasterize_polygons
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED
from filament.proto_grid import roundtrip_mask
from filament.rle import encode_rle

PARENT = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_fold0_conf0.35_pred.csv"
OUT = ARTIFACTS / "stream_proto_ceiling_fold0.json"
EXPECTED_PQ = 0.4387235694048039
PUBLIC_GAP = 0.069
RATIOS = (4, 2)
HYPOTHESIS = (
    "ceiling only: fold-0 medoids round-tripped through prototype grids 4 and 2. "
    "Not a model and not a submission."
)


def may_upload(pq: float) -> bool:
    """Label round-trips are not submissions, whatever the score is."""
    del pq
    return False


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


def _ceilings(views, by_file, val_files) -> dict[int, tuple[pd.DataFrame, int, int]]:
    built: dict[int, tuple[list, int, int]] = {ratio: ([], 0, 0) for ratio in RATIOS}
    for name in val_files:
        stem = Path(name).stem
        view = views[medoid_view_id(views, by_file[name])]
        per_ratio: dict[int, list[str]] = {ratio: [] for ratio in RATIOS}
        for filament in view.filaments:
            mask = rasterize_polygons([filament.polygon])
            if int(mask.sum()) == 0:
                for ratio in RATIOS:
                    frames, kept, dropped = built[ratio]
                    built[ratio] = (frames, kept, dropped + 1)
                continue
            for ratio in RATIOS:
                restored = roundtrip_mask(mask, ratio)
                frames, kept, dropped = built[ratio]
                if int(restored.sum()) == 0:
                    built[ratio] = (frames, kept, dropped + 1)
                    continue
                per_ratio[ratio].append(encode_rle(restored))
                built[ratio] = (frames, kept + 1, dropped)
        for ratio in RATIOS:
            if per_ratio[ratio]:
                frames, kept, dropped = built[ratio]
                frames.append(pred_to_df(stem, per_ratio[ratio]))
                built[ratio] = (frames, kept, dropped)
    packed: dict[int, tuple[pd.DataFrame, int, int]] = {}
    empty = pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    for ratio, (frames, kept, dropped) in built.items():
        pred = pd.concat(frames, ignore_index=True) if frames else empty
        packed[ratio] = (pred, kept, dropped)
    return packed


def main() -> None:
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
    ceilings = _ceilings(views, by_file, val_files)
    scored: dict[str, dict] = {}
    for ratio in RATIOS:
        pred, kept, dropped = ceilings[ratio]
        row = _row(score_pq(gt_df, pred), len(pred))
        formula = row["mean_tp_iou"] * row["tp"] / (
            row["tp"] + 0.5 * row["fp"] + 0.5 * row["fn"]
        )
        if row["tp"] and abs(formula - row["pq"]) > 1e-9:
            raise SystemExit(f"formula check failed at ratio {ratio}: {formula} vs {row['pq']}")
        scored[str(ratio)] = {
            **row,
            "formula_pq": formula,
            "instances_kept": kept,
            "instances_dropped": dropped,
            "public_forecast": row["pq"] - PUBLIC_GAP,
        }
    payload = {
        "split": "grouped year x count fold 0",
        "seed": SEED,
        "hypothesis": HYPOTHESIS,
        "oracle": True,
        "ratios": list(RATIOS),
        "proto_note": (
            "YOLO11 Proto upsamples P3/8 by 2, so the prototype is stride 4. "
            "mask_ratio does not change that spatial size."
        ),
        "parent": first,
        "parent_second_call": second,
        "ceilings": scored,
        "public_gap_used": PUBLIC_GAP,
        "promote": may_upload(max(item["pq"] for item in scored.values())),
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
